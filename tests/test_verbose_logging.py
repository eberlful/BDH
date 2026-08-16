"""Unit tests for verbose logging of training data and predicted output."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console
import torch
import yaml

from src.cli import build_parser, run_train
from src.core.config import load_config, normalize_config, validate_config
from src.data.data import SUDOKU_SEPARATOR_TOKEN, SudokuDataModule, _load_tokenizer
from src.data.sudoku_cot import SudokuCoTDataModule
from src.logging.loggers import TerminalLogger, _decode_tokens
from src.runtime import build_components


class TestVerboseLogging(unittest.TestCase):
    def test_decode_tokens_with_tokenizer(self) -> None:
        tokenizer = _load_tokenizer("byte")
        data_module = MagicMock()
        data_module.tokenizer = tokenizer

        text = "hello world"
        tokens = tokenizer.encode(text)
        decoded = _decode_tokens(tokens, data_module)
        self.assertEqual(decoded, text)

        tensor_tokens = torch.tensor(tokens, dtype=torch.long)
        decoded_tensor = _decode_tokens(tensor_tokens, data_module)
        self.assertEqual(decoded_tensor, text)

    def test_decode_tokens_with_masked_targets(self) -> None:
        tokenizer = _load_tokenizer("gpt2")
        data_module = MagicMock()
        data_module.tokenizer = tokenizer

        prefix = "Sudoku:\n1 2 3"
        suffix = "\nSolution:\n4 5 6"
        prefix_ids = tokenizer.encode(prefix)
        suffix_ids = tokenizer.encode(suffix)

        # Simulating masked target where prompt has -100
        targets = [-100] * len(prefix_ids) + suffix_ids
        decoded = _decode_tokens(targets, data_module)
        self.assertTrue(decoded.startswith("[MASK]"))
        self.assertIn("Solution:", decoded)

    def test_decode_tokens_without_tokenizer(self) -> None:
        raw_tokens = [1, 2, 3, 4, SUDOKU_SEPARATOR_TOKEN, 10, 5]
        decoded = _decode_tokens(raw_tokens, data_module=None)
        self.assertEqual(decoded, "1 2 3 4 91 10 5")

    def test_terminal_logger_verbose_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            logger = TerminalLogger(run_dir=run_dir, verbose=True, max_display_length=100)

            # Mock trainer
            trainer = MagicMock()
            trainer.state.global_step = 0
            trainer.state.epoch = 0
            trainer.log_every_n_steps = 1
            tokenizer = _load_tokenizer("byte")
            trainer.data_module.tokenizer = tokenizer

            input_tokens = torch.tensor([[ord(c) for c in "input prompt"]], dtype=torch.long)
            target_tokens = torch.tensor([[ord(c) for c in "target text"]], dtype=torch.long)
            # Logits: shape [1, seq_len, vocab_size]
            logits = torch.zeros((1, 12, 256), dtype=torch.float32)
            # Set argmax to spell "prediction!!"
            pred_chars = "prediction!!"
            for i, c in enumerate(pred_chars):
                logits[0, i, ord(c)] = 10.0

            batch = {"input_ids": input_tokens, "target_ids": target_tokens}
            output = {"loss": torch.tensor(1.23), "logits": logits}

            capture = io.StringIO()
            logger.console.file = capture

            logger.on_train_batch_end(trainer, batch, batch_idx=0, output=output)

            printed = capture.getvalue()
            self.assertIn("Input Data:", printed)
            self.assertIn("input prompt", printed)
            self.assertIn("Target Data:", printed)
            self.assertIn("target text", printed)
            self.assertIn("Predicted Model Output:", printed)
            self.assertIn("prediction!!", printed)

    def test_terminal_logger_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            logger = TerminalLogger(run_dir=run_dir, verbose=True, max_display_length=20)
            logger.console = Console(width=200)

            trainer = MagicMock()
            trainer.state.global_step = 0
            trainer.state.epoch = 0
            trainer.log_every_n_steps = 1
            trainer.data_module.tokenizer = _load_tokenizer("byte")

            long_text = "a" * 100
            input_tokens = torch.tensor([[ord(c) for c in long_text]], dtype=torch.long)
            batch = {"input_ids": input_tokens}
            output = {"loss": torch.tensor(1.0)}

            capture = io.StringIO()
            logger.console.file = capture

            logger.on_train_batch_end(trainer, batch, batch_idx=0, output=output)

            printed = capture.getvalue()
            self.assertIn("[truncated, total 100 chars]", printed)

    def test_verbose_training_end_to_end_tiny_shakespeare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "input.txt"
            data_path.write_text("abcdefghijklmnopqrstuvwxyz " * 200, encoding="utf-8")
            run_dir = root / "run"
            config = {
                "seed": 42,
                "device": "cpu",
                "verbose": True,
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 4,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "tiny_shakespeare",
                    "params": {
                        "input_file_path": str(data_path),
                        "tokenizer": "byte",
                        "context_length": 8,
                        "batch_size": 4,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 2,
                    "log_every_n_steps": 1,
                },
                "callbacks": [],
                "loggers": ["terminal"],
            }
            normalize_config(config)
            validate_config(config)
            trainer = build_components(config, run_dir)
            terminal_logger = next(l for l in trainer.loggers if isinstance(l, TerminalLogger))
            self.assertTrue(terminal_logger.verbose)

            capture = io.StringIO()
            terminal_logger.console.file = capture
            trainer.fit()

            printed = capture.getvalue()
            self.assertIn("Input Data:", printed)
            self.assertIn("Target Data:", printed)
            self.assertIn("Predicted Model Output:", printed)

    def test_cli_verbose_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "input.txt"
            data_path.write_text("abcdefghijklmnopqrstuvwxyz " * 200, encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "runs_dir": str(root / "runs"),
                        "device": "cpu",
                        "model": {
                            "name": "bdh",
                            "params": {
                                "vocab_size": "auto",
                                "context_length": 8,
                                "n_embd": 16,
                                "n_head": 4,
                                "n_layer": 1,
                                "mlp_internal_dim_multiplier": 16,
                            },
                        },
                        "data": {
                            "name": "tiny_shakespeare",
                            "params": {
                                "input_file_path": str(data_path),
                                "tokenizer": "byte",
                                "context_length": 8,
                                "batch_size": 4,
                            },
                        },
                        "trainer": {"max_epochs": 1, "max_steps": 1, "log_every_n_steps": 1},
                        "loggers": ["terminal"],
                    }
                ),
                encoding="utf-8",
            )
            # Test parser
            args = build_parser().parse_args(["train", str(config_path), "-v"])
            self.assertTrue(args.verbose)

            args_long = build_parser().parse_args(["train", str(config_path), "--verbose"])
            self.assertTrue(args_long.verbose)

            # Test run_train with verbose=True
            self.assertEqual(run_train(config_path, [], verbose=True), 0)

    def test_verbose_training_bdh_cq(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            config = {
                "seed": 42,
                "device": "cpu",
                "verbose": True,
                "model": {
                    "name": "bdh_cq",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 128,
                        "n_embd": 16,
                        "n_head": 4,
                        "n_layer": 1,
                        "mlp_internal_dim_multiplier": 16,
                        "latent_reasoning_steps": 2,
                    },
                },
                "data": {
                    "name": "sudoku",
                    "params": {
                        "num_samples": 4,
                        "validation_fraction": 0.5,
                        "clues": 70,
                        "context_length": 128,
                        "batch_size": 2,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 2,
                    "log_every_n_steps": 1,
                },
                "callbacks": [],
                "loggers": ["terminal"],
            }
            normalize_config(config)
            validate_config(config)
            trainer = build_components(config, run_dir)
            terminal_logger = next(l for l in trainer.loggers if isinstance(l, TerminalLogger))
            self.assertTrue(terminal_logger.verbose)

            capture = io.StringIO()
            terminal_logger.console.file = capture
            trainer.fit()

            printed = capture.getvalue()
            self.assertIn("Input Data:", printed)
            self.assertIn("Target Data:", printed)
            self.assertIn("Predicted Model Output:", printed)


if __name__ == "__main__":
    unittest.main()
