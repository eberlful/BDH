from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.core.config import load_config, validate_config
from src.cli import run_resume, run_train, run_validate
from src.data.data import TinyShakespeareDataModule
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
                "loggers": [{"name": "text_file", "params": {}}],
            }
            validate_config(config)
            trainer = build_components(config, run_dir)
            trainer.fit()
            self.assertTrue((run_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((run_dir / "training.log").exists())

            resumed = build_components(config, run_dir)
            resumed.fit(run_dir / "checkpoints" / "last.pt")
            self.assertEqual(resumed.state.epoch, 1)

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


if __name__ == "__main__":
    unittest.main()
