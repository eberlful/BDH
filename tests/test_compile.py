from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.cli import run_train, run_validate, run_resume
from src.core.config import DEFAULT_CONFIG, load_config, validate_config
from src.model.gpt import GPTModel
from src.runtime import build_components
from src.training.trainer import TorchTrainer


class CompileTests(unittest.TestCase):
    def test_default_config_includes_compile_false(self) -> None:
        self.assertIn("compile", DEFAULT_CONFIG)
        self.assertFalse(DEFAULT_CONFIG["compile"])
        self.assertIn("compile", DEFAULT_CONFIG["trainer"])
        self.assertFalse(DEFAULT_CONFIG["trainer"]["compile"])

    def test_compile_config_loading_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "compile": True,
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            validate_config(config)
            self.assertTrue(config["compile"])

    def test_compile_dict_and_str_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "compile": {"mode": "default"},
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            validate_config(config)
            self.assertEqual(config["compile"], {"mode": "default"})

    def test_invalid_compile_type_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "compile": [1, 2, 3],
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "compile must be a boolean"):
                load_config(path)

    def test_cli_overrides_for_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "compile": False,
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path, ["compile=true"])
            self.assertTrue(config["compile"])

            config_trainer = load_config(path, ["trainer.compile=true"])
            self.assertTrue(config_trainer["trainer"]["compile"])

    def test_trainer_setup_compiles_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = {
                "compile": True,
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": 256,
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "tiny_shakespeare",
                    "params": {
                        "tokenizer": "byte",
                        "context_length": 8,
                        "batch_size": 2,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                },
                "callbacks": [],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            self.assertTrue(trainer.compile)
            trainer.setup()
            # Verify that self.model is wrapped in torch.compile's OptimizedModule
            self.assertTrue(hasattr(trainer.model, "_orig_mod"))
            self.assertIsInstance(trainer.model._orig_mod, GPTModel)

    def test_compiled_model_training_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = {
                "compile": True,
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": 256,
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "tiny_shakespeare",
                    "params": {
                        "tokenizer": "byte",
                        "context_length": 8,
                        "batch_size": 2,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 2,
                },
                "callbacks": [],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.fit()
            self.assertEqual(trainer.state.global_step, 2)

    def test_checkpoint_saving_and_restoring_with_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            config = {
                "compile": True,
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": 256,
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 2,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "tiny_shakespeare",
                    "params": {
                        "tokenizer": "byte",
                        "context_length": 8,
                        "batch_size": 2,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                },
                "callbacks": [],
                "loggers": [],
            }
            trainer = build_components(config, run_dir)
            trainer.setup()
            checkpoint_state = trainer.checkpoint_state()

            # State dict must not contain _orig_mod. prefix
            for key in checkpoint_state["model"].keys():
                self.assertFalse(
                    key.startswith("_orig_mod."),
                    f"Checkpoint model key {key} contains _orig_mod. prefix",
                )

            # Save checkpoint to disk
            ckpt_path = run_dir / "test.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_state, ckpt_path)

            # Restore into uncompiled trainer
            config_uncompiled = dict(config)
            config_uncompiled["compile"] = False
            trainer_uncompiled = build_components(config_uncompiled, run_dir)
            trainer_uncompiled.setup()
            trainer_uncompiled.restore_checkpoint(ckpt_path)

            # Restore into another compiled trainer
            trainer_compiled2 = build_components(config, run_dir)
            trainer_compiled2.setup()
            trainer_compiled2.restore_checkpoint(ckpt_path)

    def test_cli_train_with_compile_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            runs_dir = Path(directory) / "runs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "compile": True,
                        "runs_dir": str(runs_dir),
                        "model": {
                            "name": "gpt_model",
                            "params": {
                                "vocab_size": 256,
                                "context_length": 8,
                                "d_model": 16,
                                "n_heads": 2,
                                "n_layers": 1,
                            },
                        },
                        "data": {
                            "name": "tiny_shakespeare",
                            "params": {
                                "tokenizer": "byte",
                                "context_length": 8,
                                "batch_size": 2,
                            },
                        },
                        "trainer": {
                            "name": "torch",
                            "max_epochs": 1,
                            "max_steps": 2,
                        },
                        "callbacks": [
                            {"name": "checkpoint", "params": {"save_best": True}},
                        ],
                        "loggers": [{"name": "text_file"}],
                    }
                ),
                encoding="utf-8",
            )
            exit_code = run_train(config_path, overrides=[])
            self.assertEqual(exit_code, 0)

            # Verify checkpoint can be resumed
            run_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            active_run_dir = run_dirs[0]

            resume_code = run_resume(active_run_dir, overrides=[])
            self.assertEqual(resume_code, 0)
