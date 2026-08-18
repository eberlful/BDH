"""Tests for Sudoku Chain-of-Thought (CoT) dataset, tokenization, and training."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import tiktoken
import torch

from src.core.config import dump_config
from src.data.sudoku_cot import (
    SudokuCoTDataModule,
    SudokuCoTDataset,
    build_sudoku_cot_full_text,
    build_sudoku_cot_prompt,
    format_sudoku_grid,
    generate_cot_steps,
    parse_sudoku_grid,
)
from src.model.bdh import ConfiguredBDH, GPTModel
from src.runtime import build_components
from src.cli import run_generate


class SudokuCoTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_puzzle = (
            5, 3, 0, 0, 7, 0, 0, 0, 0,
            6, 0, 0, 1, 9, 5, 0, 0, 0,
            0, 9, 8, 0, 0, 0, 0, 6, 0,
            8, 0, 0, 0, 6, 0, 0, 0, 3,
            4, 0, 0, 8, 0, 3, 0, 0, 1,
            7, 0, 0, 0, 2, 0, 0, 0, 6,
            0, 6, 0, 0, 0, 0, 2, 8, 0,
            0, 0, 0, 4, 1, 9, 0, 0, 5,
            0, 0, 0, 0, 8, 0, 0, 7, 9,
        )
        self.sample_solution = (
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        )

    def test_grid_formatting_and_parsing(self) -> None:
        grid_str = format_sudoku_grid(self.sample_puzzle)
        lines = grid_str.strip().split("\n")
        self.assertEqual(len(lines), 9)
        self.assertEqual(lines[0], "5 3 0 0 7 0 0 0 0")
        parsed = parse_sudoku_grid(grid_str)
        self.assertEqual(tuple(parsed), self.sample_puzzle)

    def test_cot_steps_format(self) -> None:
        steps = generate_cot_steps(self.sample_puzzle, self.sample_solution)
        step_lines = steps.split("\n")
        self.assertTrue(step_lines[0].startswith("- R 1 C 3 = 4"))
        self.assertTrue(step_lines[1].startswith("- R 1 C 4 = 6"))
        self.assertEqual(len(step_lines), 81 - 30)  # 51 blank cells

    def test_gpt2_tokenization_cleanliness(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        step_str = "- R 1 C 3 = 4\n"
        tokens = tokenizer.encode(step_str)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        self.assertEqual(decoded_tokens, ["-", " R", " 1", " C", " 3", " =", " 4", "\n"])

    def test_dataset_prompt_loss_masking(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        samples = [{"puzzle": self.sample_puzzle, "solution": self.sample_solution}]
        dataset = SudokuCoTDataset(samples, tokenizer, context_length=1024)
        item = dataset[0]

        input_ids = item["input_ids"]
        target_ids = item["target_ids"]
        self.assertEqual(input_ids.shape, torch.Size([1024]))
        self.assertEqual(target_ids.shape, torch.Size([1024]))

        prompt_text = build_sudoku_cot_prompt(self.sample_puzzle)
        prompt_ids = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)

        # Ensure all prompt transitions in target_ids are masked to -100
        for i in range(prompt_len - 1):
            self.assertEqual(target_ids[i].item(), -100)

        # The first token after the prompt should NOT be -100
        self.assertNotEqual(target_ids[prompt_len - 1].item(), -100)

    def test_byte_tokenized_padding_stays_within_vocabulary(self) -> None:
        data_module = SudokuCoTDataModule(
            num_samples=2,
            validation_fraction=0.5,
            clues=30,
            batch_size=1,
            context_length=512,
            reasoning_mode="none",
            tokenizer="byte",
        )
        data_module.setup()
        
        item = data_module.train_dataset[0]
        self.assertEqual(data_module.eos_token_id, 256)
        self.assertLess(int(item["input_ids"].max()), data_module.vocab_size)
        self.assertEqual(int(item["input_ids"].max()), data_module.eos_token_id)

    def test_model_training_step_with_cot_batch(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        samples = [{"puzzle": self.sample_puzzle, "solution": self.sample_solution}]
        dataset = SudokuCoTDataset(samples, tokenizer, context_length=256)
        item = dataset[0]
        batch = {
            "input_ids": item["input_ids"].unsqueeze(0),
            "target_ids": item["target_ids"].unsqueeze(0),
        }

        model = ConfiguredBDH(vocab_size=50257, context_length=256, n_layer=2, n_embd=64, n_head=2)
        out = model.training_step(batch, 0)
        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))

        transformer = GPTModel(vocab_size=50257, context_length=256, d_model=64, n_heads=2, n_layers=2)
        out_tf = transformer.training_step(batch, 0)
        self.assertIn("loss", out_tf)
        self.assertFalse(torch.isnan(out_tf["loss"]))

    def test_small_sudoku_cot_fit_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = {
                "seed": 42,
                "device": "cpu",
                "runs_dir": str(Path(temp_dir) / "runs"),
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 128,
                        "d_model": 32,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "sudoku_cot",
                    "params": {
                        "num_samples": 4,
                        "validation_fraction": 0.5,
                        "clues": 60,
                        "batch_size": 2,
                        "context_length": 128,
                        "tokenizer": "gpt2",
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "log_every_n_steps": 1,
                    "validate_every_n_epochs": 1,
                },
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.fit()

            dump_config(config, run_dir / "config.yaml")
            prompt = "".join(str(v) for v in self.sample_puzzle)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(run_generate(run_dir, prompt, max_tokens=10), 0)
            self.assertIn("Sudoku:", output.getvalue())
            self.assertIn("Thinking:\n", output.getvalue())

    def test_small_sudoku_direct_fit_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = {
                "seed": 42,
                "device": "cpu",
                "runs_dir": str(Path(temp_dir) / "runs"),
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 128,
                        "d_model": 32,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "sudoku_cot",
                    "params": {
                        "num_samples": 4,
                        "validation_fraction": 0.5,
                        "clues": 60,
                        "batch_size": 2,
                        "context_length": 128,
                        "reasoning_mode": "none",
                        "tokenizer": "gpt2",
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "log_every_n_steps": 1,
                    "validate_every_n_epochs": 1,
                },
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.fit()

            dump_config(config, run_dir / "config.yaml")
            prompt = "".join(str(v) for v in self.sample_puzzle)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(run_generate(run_dir, prompt, max_tokens=10), 0)
            self.assertIn("Sudoku:", output.getvalue())
            self.assertIn("Solution:\n", output.getvalue())
            self.assertNotIn("Thinking:\n", output.getvalue())

    def test_small_sudoku_context_only_fit_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = {
                "seed": 42,
                "device": "cpu",
                "runs_dir": str(Path(temp_dir) / "runs"),
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 512,
                        "d_model": 32,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "sudoku_cot",
                    "params": {
                        "num_samples": 4,
                        "validation_fraction": 0.5,
                        "clues": 70,
                        "batch_size": 2,
                        "context_length": 512,
                        "reasoning_mode": "context_only",
                        "tokenizer": "gpt2",
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "log_every_n_steps": 1,
                    "validate_every_n_epochs": 1,
                },
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.fit()

            dump_config(config, run_dir / "config.yaml")
            prompt = "".join(str(v) for v in self.sample_puzzle)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(run_generate(run_dir, prompt, max_tokens=10), 0)
            self.assertIn("Sudoku:", output.getvalue())
            self.assertIn("Thinking:\n", output.getvalue())

    def test_reasoning_modes_validation(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        samples = [{"puzzle": self.sample_puzzle, "solution": self.sample_solution}]

        with self.assertRaises(ValueError):
            build_sudoku_cot_prompt(self.sample_puzzle, reasoning_mode="invalid")

        with self.assertRaises(ValueError):
            build_sudoku_cot_full_text(self.sample_puzzle, self.sample_solution, reasoning_mode="invalid")

        with self.assertRaises(ValueError):
            SudokuCoTDataset(samples, tokenizer, reasoning_mode="invalid")

        with self.assertRaises(ValueError):
            SudokuCoTDataModule(num_samples=4, reasoning_mode="invalid")

    def test_reasoning_mode_none(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        prompt_text, full_text = build_sudoku_cot_full_text(
            self.sample_puzzle, self.sample_solution, reasoning_mode="none"
        )
        self.assertIn("Solution:\n", prompt_text)
        self.assertNotIn("Thinking:\n", prompt_text)
        self.assertNotIn("- R 1 C", full_text)

        samples = [{"puzzle": self.sample_puzzle, "solution": self.sample_solution}]
        dataset = SudokuCoTDataset(samples, tokenizer, context_length=256, reasoning_mode="none")
        item = dataset[0]
        input_ids = item["input_ids"]
        target_ids = item["target_ids"]

        prompt_ids = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)
        # All prompt transitions in target_ids should be -100
        for i in range(prompt_len - 1):
            self.assertEqual(target_ids[i].item(), -100)
        # First token of solution should be supervised
        self.assertNotEqual(target_ids[prompt_len - 1].item(), -100)

    def test_reasoning_mode_context_only(self) -> None:
        tokenizer = tiktoken.get_encoding("gpt2")
        samples = [{"puzzle": self.sample_puzzle, "solution": self.sample_solution}]
        dataset = SudokuCoTDataset(samples, tokenizer, context_length=1024, reasoning_mode="context_only")
        item = dataset[0]
        target_ids = item["target_ids"]

        prompt_text = build_sudoku_cot_prompt(self.sample_puzzle, reasoning_mode="context_only")
        cot_steps = generate_cot_steps(self.sample_puzzle, self.sample_solution)
        context_prefix = f"{prompt_text}{cot_steps}\n\nSolution:\n"
        context_ids = tokenizer.encode(context_prefix)
        context_len = len(context_ids)

        # Everything up to Solution:\n should be masked with -100
        for i in range(context_len - 1):
            self.assertEqual(target_ids[i].item(), -100)
        # First token of the solution must not be -100
        self.assertNotEqual(target_ids[context_len - 1].item(), -100)


if __name__ == "__main__":
    unittest.main()

