from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import yaml

from src.core.config import load_config, validate_config
from src.cli import build_parser, run_generate, run_resume, run_train, run_validate
from src.data.data import (
    SUDOKU_END_TOKEN,
    SUDOKU_POSITION_OFFSET,
    SUDOKU_SEPARATOR_TOKEN,
    SudokuDataModule,
    TinyShakespeareDataModule,
    is_valid_sudoku_board,
)
from src.model.bdh import BDHTransformer
from src.runtime import build_components


class TrainingEnvironmentTests(unittest.TestCase):
    def test_config_overrides_support_yaml_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "bdh_transformer", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(
                path,
                ["trainer.max_epochs=3", "model.params.d_model=16", "data.params.shuffle=false"],
            )
            self.assertEqual(config["trainer"]["max_epochs"], 3)
            self.assertEqual(config["model"]["params"]["d_model"], 16)
            self.assertFalse(config["data"]["params"]["shuffle"])

    def test_transformer_forward_and_loss(self) -> None:
        model = BDHTransformer(vocab_size=32, context_length=8, d_model=16, n_heads=4, n_layers=1)
        batch = {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "target_ids": torch.randint(0, 32, (2, 8)),
        }
        output = model.training_step(batch, 0)
        self.assertEqual(model(batch["input_ids"]).shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(output["loss"]))
        with self.assertRaisesRegex(ValueError, "exceeds context_length"):
            model.generate(torch.randint(0, 32, (1, 9)), max_new_tokens=1)

    def test_byte_tokenized_data_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_text("abcdefghijklmnopqrstuvwxyz " * 100, encoding="utf-8")
            module = TinyShakespeareDataModule(
                input_file_path=path,
                tokenizer="byte",
                context_length=8,
                batch_size=4,
            )
            module.prepare_data()
            module.setup("fit")
            batch = next(iter(module.train_dataloader()))
            self.assertEqual(batch["input_ids"].shape, (4, 8))
            self.assertEqual(batch["target_ids"].shape, (4, 8))

    def test_sudoku_examples_are_valid_and_trace_every_blank(self) -> None:
        module = SudokuDataModule(
            num_samples=8,
            validation_fraction=0.25,
            clues=60,
            context_length=128,
            batch_size=2,
            seed=19,
        )
        module.prepare_data()
        module.setup("fit")
        assert module.train_dataset is not None
        assert module.val_dataset is not None
        train_puzzles = {example.puzzle for example in module.train_dataset.examples}
        val_puzzles = {example.puzzle for example in module.val_dataset.examples}
        self.assertTrue(train_puzzles.isdisjoint(val_puzzles))
        for example in [*module.train_dataset.examples, *module.val_dataset.examples]:
            self.assertTrue(is_valid_sudoku_board(example.solution))
            for puzzle_value, solution_value in zip(example.puzzle, example.solution):
                if puzzle_value:
                    self.assertEqual(puzzle_value, solution_value)
            tokens = example.token_ids
            self.assertEqual(tokens[81], SUDOKU_SEPARATOR_TOKEN)
            self.assertEqual(tokens[-1], SUDOKU_END_TOKEN)
            trace = tokens[82:-1]
            positions = [trace[index] - SUDOKU_POSITION_OFFSET for index in range(0, len(trace), 2)]
            values = [trace[index] for index in range(1, len(trace), 2)]
            expected_positions = [index for index, value in enumerate(example.puzzle) if value == 0]
            self.assertEqual(positions, expected_positions)
            self.assertEqual(values, [example.solution[index] for index in expected_positions])

    def test_sudoku_data_is_seeded_and_batches_are_in_range(self) -> None:
        first = SudokuDataModule(num_samples=6, validation_fraction=0.5, clues=70, context_length=128, seed=5)
        second = SudokuDataModule(num_samples=6, validation_fraction=0.5, clues=70, context_length=128, seed=5)
        first.setup("fit")
        second.setup("fit")
        assert first.train_dataset is not None
        assert second.train_dataset is not None
        self.assertEqual(first.train_dataset.examples, second.train_dataset.examples)
        batch = next(iter(first.train_dataloader()))
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["input_ids"].shape[1], 81 + 2 * 11 + 1)
        self.assertLess(int(batch["input_ids"].max()), first.vocab_size)

    def test_sudoku_context_length_and_training_smoke(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds context_length"):
            SudokuDataModule(num_samples=4, clues=30, context_length=100)
        config = {
            "seed": 3,
            "device": "cpu",
            "model": {
                "name": "bdh_transformer",
                "params": {
                    "vocab_size": "auto",
                    "context_length": 128,
                    "d_model": 16,
                    "n_heads": 4,
                    "n_layers": 1,
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
            "trainer": {"name": "torch", "max_epochs": 1, "max_steps": 1, "log_every_n_steps": 1},
            "callbacks": [],
            "loggers": [],
        }
        validate_config(config)
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_components(config, Path(directory) / "run")
            trainer.fit()
            self.assertEqual(trainer.state.global_step, 1)

    def test_short_training_creates_checkpoints_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "input.txt"
            data_path.write_text("abcdefghijklmnopqrstuvwxyz " * 200, encoding="utf-8")
            run_dir = root / "run"
            config = {
                "seed": 7,
                "device": "cpu",
                "model": {
                    "name": "bdh_transformer",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 4,
                        "n_layers": 1,
                        "learning_rate": 0.001,
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
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [
                    {"name": "terminal", "params": {}},
                    {"name": "text_file", "params": {}},
                ],
            }
            validate_config(config)
            trainer = build_components(config, run_dir)
            trainer.fit()
            self.assertTrue((run_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((run_dir / "training.log").exists())
            terminal_logger = next(logger for logger in trainer.loggers if logger.__class__.__name__ == "TerminalLogger")
            self.assertIsNotNone(terminal_logger.progress)
            self.assertIsNotNone(terminal_logger.epoch_task_id)
            task = terminal_logger.progress.tasks[terminal_logger.epoch_task_id]
            self.assertEqual(task.completed, 1)
            self.assertEqual(task.total, 1)
            self.assertIn("train/loss=", task.fields["metrics"])
            self.assertIn("remaining", terminal_logger.progress.columns[3].text_format)

            resumed = build_components(config, run_dir)
            resumed.fit(run_dir / "checkpoints" / "last.pt")
            self.assertEqual(resumed.state.epoch, 1)
            resumed_terminal_logger = next(
                logger for logger in resumed.loggers if logger.__class__.__name__ == "TerminalLogger"
            )
            resumed_task = resumed_terminal_logger.progress.tasks[resumed_terminal_logger.epoch_task_id]
            self.assertEqual(resumed_task.completed, 1)

    def test_cli_train_validate_and_resume(self) -> None:
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
                            "name": "bdh_transformer",
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
                        "trainer": {"max_epochs": 1, "max_steps": 1, "log_every_n_steps": 1},
                        "loggers": [{"name": "text_file", "params": {}}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_validate(config_path, []), 0)
            self.assertEqual(run_train(config_path, []), 0)
            run_dirs = list((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            self.assertEqual(run_resume(run_dirs[0], ["trainer.max_epochs=2"]), 0)

    def test_generate_from_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "input.txt"
            data_path.write_text("abcdefghijklmnopqrstuvwxyz " * 200, encoding="utf-8")
            run_dir = root / "run"
            config = {
                "device": "cpu",
                "model": {
                    "name": "bdh_transformer",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 16,
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
                        "context_length": 16,
                        "batch_size": 4,
                    },
                },
                "trainer": {"name": "torch", "max_epochs": 1, "max_steps": 1},
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.fit()
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            parsed = build_parser().parse_args(["generate", str(run_dir), "hello", "--max-tokens", "3"])
            self.assertEqual(parsed.max_tokens, 3)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(run_generate(run_dir, "hello", 3), 0)
            generated_text = output.getvalue().strip()
            self.assertTrue(generated_text.startswith("hello"))
            self.assertNotEqual(generated_text, "hello")

    def test_generate_rejects_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "config.yaml").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 1"):
                run_generate(run_dir, "hello", 0)

            with self.assertRaisesRegex(FileNotFoundError, "No best checkpoint"):
                run_generate(run_dir, "hello", 1)


if __name__ == "__main__":
    unittest.main()
