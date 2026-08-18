from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from src.callbacks.checkpoint import CheckpointCallback
from src.core.base import BaseTrainer


class DummyTrainer(BaseTrainer):
    def __init__(self, step_value: int = 1, run_dir: Path | None = None) -> None:
        super().__init__(run_dir=run_dir)
        self.step_value = step_value

    def fit(self, checkpoint_path: Path | None = None) -> None:
        pass

    def checkpoint_state(self) -> dict[str, Any]:
        return {"step": self.step_value}

    def restore_checkpoint(self, checkpoint_path: Path) -> None:
        pass


class CheckpointCallbackTests(unittest.TestCase):
    def test_auto_mode_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            
            # Minimization metrics
            for metric in ["val/loss", "loss", "train/error", "val/err", "val/perplexity"]:
                cb = CheckpointCallback(run_dir=run_dir, monitor=metric, mode="auto")
                self.assertEqual(cb.resolved_mode, "min", f"Expected min for {metric}")
                self.assertEqual(cb.best_metric, float("inf"))

            # Maximization metrics
            for metric in [
                "val/accuracy",
                "val/acc",
                "val/sudoku_board_accuracy",
                "val/validity_rate",
                "val/score",
                "reward",
            ]:
                cb = CheckpointCallback(run_dir=run_dir, monitor=metric, mode="auto")
                self.assertEqual(cb.resolved_mode, "max", f"Expected max for {metric}")
                self.assertEqual(cb.best_metric, float("-inf"))

            # Default fallback for unknown metrics in auto mode
            cb_unknown = CheckpointCallback(run_dir=run_dir, monitor="val/custom", mode="auto")
            self.assertEqual(cb_unknown.resolved_mode, "min")

    def test_explicit_min_and_max_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb_min = CheckpointCallback(run_dir=run_dir, monitor="val/accuracy", mode="min")
            self.assertEqual(cb_min.resolved_mode, "min")
            self.assertEqual(cb_min.best_metric, float("inf"))

            cb_max = CheckpointCallback(run_dir=run_dir, monitor="val/loss", mode="max")
            self.assertEqual(cb_max.resolved_mode, "max")
            self.assertEqual(cb_max.best_metric, float("-inf"))

    def test_invalid_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with self.assertRaisesRegex(ValueError, "Invalid mode 'invalid'"):
                CheckpointCallback(run_dir=run_dir, monitor="val/loss", mode="invalid")

    def test_maximization_checkpoint_saving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb = CheckpointCallback(
                run_dir=run_dir,
                monitor="val/sudoku_board_accuracy",
                mode="max",
                save_best=True,
            )
            trainer = DummyTrainer(step_value=1, run_dir=run_dir)
            best_path = run_dir / "checkpoints" / "best.pt"

            # Epoch 1: accuracy 0.5 -> saves best.pt (0.5 > -inf)
            trainer.step_value = 1
            cb.on_epoch_end(trainer, 0, {"val/sudoku_board_accuracy": 0.5})
            self.assertEqual(cb.best_metric, 0.5)
            self.assertTrue(best_path.exists())
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)
            self.assertEqual(loaded["callback_state"]["best_metric"], 0.5)

            # Epoch 2: accuracy 0.3 -> does NOT overwrite best.pt (0.3 < 0.5)
            trainer.step_value = 2
            cb.on_epoch_end(trainer, 1, {"val/sudoku_board_accuracy": 0.3})
            self.assertEqual(cb.best_metric, 0.5)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)

            # Epoch 3: accuracy 0.8 -> overwrites best.pt (0.8 > 0.5)
            trainer.step_value = 3
            cb.on_epoch_end(trainer, 2, {"val/sudoku_board_accuracy": 0.8})
            self.assertEqual(cb.best_metric, 0.8)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 3)
            self.assertEqual(loaded["callback_state"]["best_metric"], 0.8)

    def test_minimization_checkpoint_saving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb = CheckpointCallback(
                run_dir=run_dir,
                monitor="val/loss",
                mode="min",
                save_best=True,
            )
            trainer = DummyTrainer(step_value=1, run_dir=run_dir)
            best_path = run_dir / "checkpoints" / "best.pt"

            # Epoch 1: loss 2.0 -> saves best.pt (2.0 < inf)
            trainer.step_value = 1
            cb.on_epoch_end(trainer, 0, {"val/loss": 2.0})
            self.assertEqual(cb.best_metric, 2.0)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)

            # Epoch 2: loss 2.5 -> does NOT overwrite best.pt (2.5 > 2.0)
            trainer.step_value = 2
            cb.on_epoch_end(trainer, 1, {"val/loss": 2.5})
            self.assertEqual(cb.best_metric, 2.0)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)

            # Epoch 3: loss 1.5 -> overwrites best.pt (1.5 < 2.0)
            trainer.step_value = 3
            cb.on_epoch_end(trainer, 2, {"val/loss": 1.5})
            self.assertEqual(cb.best_metric, 1.5)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 3)

    def test_callback_state_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb = CheckpointCallback(
                run_dir=run_dir,
                monitor="val/sudoku_board_accuracy",
                mode="auto",
            )
            self.assertEqual(cb.resolved_mode, "max")
            self.assertEqual(cb.best_metric, float("-inf"))

            saved_state = {
                "best_metric": 0.85,
                "mode": "auto",
                "resolved_mode": "max",
                "monitor": "val/sudoku_board_accuracy",
            }
            cb.restore_state(saved_state)
            self.assertEqual(cb.best_metric, 0.85)
            self.assertEqual(cb.resolved_mode, "max")
