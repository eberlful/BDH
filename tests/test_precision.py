from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.cli import run_train, run_validate
from src.core.config import load_config, validate_config
from src.model.bdh import ConfiguredBDH
from src.model.bdh_cq import ConfiguredBDHCQ
from src.model.gpt import GPTModel
from src.runtime import build_components
from src.training.trainer import TorchTrainer


class PrecisionTests(unittest.TestCase):
    def test_default_config_includes_precision_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["dtype"], "float16")
            self.assertFalse(config["mixed_precision"])
            self.assertEqual(config["trainer"]["dtype"], "float16")
            self.assertFalse(config["trainer"]["mixed_precision"])

    def test_precision_overrides_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "dtype": "bfloat16",
                        "mixed_precision": True,
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config["dtype"], "bfloat16")
            self.assertTrue(config["mixed_precision"])

            # Test CLI overrides
            overridden = load_config(
                path,
                ["dtype=float32", "mixed_precision=false"],
            )
            self.assertEqual(overridden["dtype"], "float32")
            self.assertFalse(overridden["mixed_precision"])

    def test_invalid_dtype_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "dtype": "int8",
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid dtype"):
                load_config(path)

    def test_all_config_files_in_configs_dir_are_valid(self) -> None:
        configs_dir = Path("configs")
        yaml_files = list(configs_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 0)
        for cfg_path in yaml_files:
            cfg = load_config(cfg_path)
            validate_config(cfg)
            self.assertEqual(cfg["dtype"], "float16", f"Failed for {cfg_path}")
            self.assertFalse(cfg["mixed_precision"], f"Failed for {cfg_path}")

    def test_trainer_dtype_resolution(self) -> None:
        self.assertEqual(TorchTrainer._resolve_dtype("float16"), torch.float16)
        self.assertEqual(TorchTrainer._resolve_dtype("fp16"), torch.float16)
        self.assertEqual(TorchTrainer._resolve_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(TorchTrainer._resolve_dtype("bf16"), torch.bfloat16)
        self.assertEqual(TorchTrainer._resolve_dtype("float32"), torch.float32)
        self.assertEqual(TorchTrainer._resolve_dtype("fp32"), torch.float32)
        self.assertEqual(TorchTrainer._resolve_dtype("float"), torch.float32)
        self.assertEqual(TorchTrainer._resolve_dtype(torch.bfloat16), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
            TorchTrainer._resolve_dtype("complex64")

    def test_pure_precision_training_steps_all_models(self) -> None:
        models = [
            (
                "bdh",
                ConfiguredBDH(
                    vocab_size=32,
                    context_length=8,
                    n_layer=2,
                    n_embd=16,
                    n_head=2,
                    mlp_internal_dim_multiplier=4,
                ),
            ),
            (
                "bdh_cq",
                ConfiguredBDHCQ(
                    vocab_size=32,
                    context_length=8,
                    n_layer=2,
                    n_embd=16,
                    n_head=2,
                    mlp_internal_dim_multiplier=4,
                ),
            ),
            (
                "gpt",
                GPTModel(
                    vocab_size=32,
                    context_length=8,
                    d_model=16,
                    n_heads=2,
                    n_layers=2,
                ),
            ),
        ]
        batch = {
            "input_ids": torch.randint(0, 32, (2, 8)),
            "target_ids": torch.randint(0, 32, (2, 8)),
        }
        for dt in [torch.float16, torch.bfloat16, torch.float32]:
            for name, model in models:
                model.to(dtype=dt)
                optimizer = model.configure_optimizers()
                optimizer.zero_grad()
                output = model.training_step(batch, 0)
                loss = output["loss"]
                self.assertEqual(loss.dtype, dt, f"{name} with {dt}")
                loss.backward()
                optimizer.step()

    def test_mixed_precision_training_and_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {
                "seed": 42,
                "device": "cpu",
                "dtype": "bfloat16",
                "mixed_precision": True,
                "model": {
                    "name": "gpt_model",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 8,
                        "d_model": 16,
                        "n_heads": 2,
                        "n_layers": 2,
                    },
                },
                "data": {
                    "name": "tiny_shakespeare",
                    "params": {
                        "input_file_path": Path(directory) / "input.txt",
                        "tokenizer": "byte",
                        "context_length": 8,
                        "batch_size": 2,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 2,
                    "log_every_n_steps": 1,
                    "gradient_clip_norm": 1.0,
                },
                "callbacks": [],
                "loggers": [],
            }
            (Path(directory) / "input.txt").write_text("abcdefghijklmnopqrstuvwxyz" * 20, encoding="utf-8")
            trainer = build_components(config, run_dir)
            self.assertEqual(trainer.dtype, torch.bfloat16)
            self.assertTrue(trainer.mixed_precision)
            trainer.fit()

            # Test checkpoint saving & restoring
            state = trainer.checkpoint_state()
            ckpt_path = run_dir / "ckpt.pt"
            torch.save(state, ckpt_path)

            trainer2 = build_components(config, run_dir)
            trainer2.setup()
            trainer2.restore_checkpoint(ckpt_path)
            self.assertEqual(trainer2.state.global_step, 2)

    def test_bdh_and_bdh_cq_trainer_fit_pure_and_mixed(self) -> None:
        for model_name in ["bdh", "bdh_cq"]:
            for mixed in [False, True]:
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    config = {
                        "seed": 42,
                        "device": "cpu",
                        "dtype": "bfloat16" if mixed else "float16",
                        "mixed_precision": mixed,
                        "model": {
                            "name": model_name,
                            "params": {
                                "vocab_size": "auto",
                                "context_length": 8,
                                "n_layer": 2,
                                "n_embd": 16,
                                "n_head": 2,
                                "mlp_internal_dim_multiplier": 4,
                            },
                        },
                        "data": {
                            "name": "tiny_shakespeare",
                            "params": {
                                "input_file_path": Path(directory) / "input.txt",
                                "tokenizer": "byte",
                                "context_length": 8,
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
                        "loggers": [],
                    }
                    (Path(directory) / "input.txt").write_text("abcdefghijklmnopqrstuvwxyz" * 20, encoding="utf-8")
                    trainer = build_components(config, run_dir)
                    trainer.fit()
                    self.assertEqual(trainer.state.global_step, 2)

    def test_cli_train_with_precision_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "config.yaml"
            input_path = Path(directory) / "input.txt"
            input_path.write_text("abcdefghijklmnopqrstuvwxyz" * 20, encoding="utf-8")
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 42,
                        "device": "cpu",
                        "runs_dir": str(Path(directory) / "runs"),
                        "model": {
                            "name": "gpt_model",
                            "params": {
                                "vocab_size": "auto",
                                "context_length": 8,
                                "d_model": 16,
                                "n_heads": 2,
                                "n_layers": 2,
                            },
                        },
                        "data": {
                            "name": "tiny_shakespeare",
                            "params": {
                                "input_file_path": str(input_path),
                                "tokenizer": "byte",
                                "context_length": 8,
                                "batch_size": 2,
                            },
                        },
                        "trainer": {
                            "name": "torch",
                            "max_epochs": 1,
                            "max_steps": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_validate(cfg_path, ["dtype=bfloat16", "mixed_precision=true"]), 0)
            self.assertEqual(run_train(cfg_path, ["dtype=bfloat16", "mixed_precision=true"]), 0)


if __name__ == "__main__":
    unittest.main()
