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
            last_path = run_dir / "checkpoints" / "last.pt"

            # Epoch 1: accuracy 0.5 -> saves best.pt (0.5 > -inf) and last.pt, no named epoch files
            trainer.step_value = 1
            cb.on_epoch_end(trainer, 0, {"val/sudoku_board_accuracy": 0.5})
            self.assertEqual(cb.best_metric, 0.5)
            self.assertTrue(best_path.exists())
            self.assertTrue(last_path.exists())
            self.assertFalse((run_dir / "checkpoints" / "epoch-0001.pt").exists())
            named_best_ep1 = run_dir / "checkpoints" / "best_epoch-0001_val_sudoku_board_accuracy-0.5000.pt"
            self.assertFalse(named_best_ep1.exists())
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)
            self.assertEqual(loaded["callback_state"]["best_metric"], 0.5)

            # Epoch 2: accuracy 0.3 -> does NOT overwrite best.pt (0.3 < 0.5), overwrites last.pt
            trainer.step_value = 2
            cb.on_epoch_end(trainer, 1, {"val/sudoku_board_accuracy": 0.3})
            self.assertEqual(cb.best_metric, 0.5)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)
            loaded_last = torch.load(last_path, weights_only=False)
            self.assertEqual(loaded_last["step"], 2)

            # Epoch 3: accuracy 0.8 -> overwrites best.pt (0.8 > 0.5) and last.pt
            trainer.step_value = 3
            cb.on_epoch_end(trainer, 2, {"val/sudoku_board_accuracy": 0.8})
            self.assertEqual(cb.best_metric, 0.8)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 3)
            self.assertEqual(loaded["callback_state"]["best_metric"], 0.8)
            loaded_last = torch.load(last_path, weights_only=False)
            self.assertEqual(loaded_last["step"], 3)

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
            last_path = run_dir / "checkpoints" / "last.pt"

            # Epoch 1: loss 2.0 -> saves best.pt (2.0 < inf) and last.pt
            trainer.step_value = 1
            cb.on_epoch_end(trainer, 0, {"val/loss": 2.0})
            self.assertEqual(cb.best_metric, 2.0)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)
            loaded_last = torch.load(last_path, weights_only=False)
            self.assertEqual(loaded_last["step"], 1)

            # Epoch 2: loss 2.5 -> does NOT overwrite best.pt (2.5 > 2.0), overwrites last.pt
            trainer.step_value = 2
            cb.on_epoch_end(trainer, 1, {"val/loss": 2.5})
            self.assertEqual(cb.best_metric, 2.0)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 1)
            loaded_last = torch.load(last_path, weights_only=False)
            self.assertEqual(loaded_last["step"], 2)

            # Epoch 3: loss 1.5 -> overwrites best.pt (1.5 < 2.0) and last.pt
            trainer.step_value = 3
            cb.on_epoch_end(trainer, 2, {"val/loss": 1.5})
            self.assertEqual(cb.best_metric, 1.5)
            loaded = torch.load(best_path, weights_only=False)
            self.assertEqual(loaded["step"], 3)
            loaded_last = torch.load(last_path, weights_only=False)
            self.assertEqual(loaded_last["step"], 3)

    def test_explicit_save_epoch_creates_named_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb = CheckpointCallback(
                run_dir=run_dir,
                monitor="val/loss",
                mode="min",
                save_best=True,
                save_epoch=True,
            )
            trainer = DummyTrainer(step_value=1, run_dir=run_dir)
            cb.on_epoch_end(trainer, 0, {"val/loss": 2.0})

            self.assertTrue((run_dir / "checkpoints" / "epoch-0001.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best_epoch-0001_val_loss-2.0000.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "last.pt").exists())

    def test_checkpoint_logging_to_loggers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cb = CheckpointCallback(
                run_dir=run_dir,
                monitor="val/loss",
                mode="min",
                save_best=True,
            )
            trainer = DummyTrainer(step_value=1, run_dir=run_dir)
            
            # Create a mock logger
            messages: list[str] = []
            events: list[tuple[str, dict[str, Any]]] = []

            class MockLogger:
                def _print(self, message: str) -> None:
                    messages.append(message)

                def _write(self, event: str, payload: dict[str, Any]) -> None:
                    events.append((event, payload))

            mock_logger = MockLogger()
            setattr(trainer, "loggers", [mock_logger])

            # Epoch 1: Initial best checkpoint
            cb.on_epoch_end(trainer, 0, {"val/loss": 2.5})
            self.assertEqual(len(messages), 1)
            self.assertIn("New best checkpoint", messages[0])
            self.assertIn("best.pt", messages[0])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][0], "checkpoint")
            self.assertEqual(events[0][1]["epoch"], 1)
            self.assertEqual(events[0][1]["best_metric"], 2.5)
            self.assertIsNone(events[0][1]["previous_best"])

            # Epoch 2: Worse loss, no new best log
            cb.on_epoch_end(trainer, 1, {"val/loss": 3.0})
            self.assertEqual(len(messages), 1)
            self.assertEqual(len(events), 1)

            # Epoch 3: Improved loss -> logged with improvement
            cb.on_epoch_end(trainer, 2, {"val/loss": 1.25})
            self.assertEqual(len(messages), 2)
            self.assertIn("best.pt", messages[1])
            self.assertIn("improved from", messages[1])
            self.assertIn("2.5000", messages[1])
            self.assertEqual(len(events), 2)
            self.assertEqual(events[1][1]["epoch"], 3)
            self.assertEqual(events[1][1]["previous_best"], 2.5)
            self.assertEqual(events[1][1]["best_metric"], 1.25)

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

    def test_restore_checkpoint_logging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            ckpt_path = run_dir / "checkpoints" / "best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            class MockModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.layer = torch.nn.Linear(2, 2)

                def configure_optimizers(self) -> torch.optim.Optimizer:
                    return torch.optim.SGD(self.parameters(), lr=0.01)

                def setup(self, data_module: Any, trainer: Any) -> None:
                    pass

            model = MockModel()
            payload = {
                "model": model.state_dict(),
                "optimizer": model.configure_optimizers().state_dict(),
                "state": {"epoch": 2, "global_step": 150, "best_metric": 1.2345},
                "callback_state": {"monitor": "val/loss", "best_metric": 1.2345},
            }
            torch.save(payload, ckpt_path)

            messages: list[str] = []
            events: list[tuple[str, dict[str, Any]]] = []

            class MockLogger:
                def _print(self, message: str) -> None:
                    messages.append(message)

                def _write(self, event: str, payload: dict[str, Any]) -> None:
                    events.append((event, payload))

            mock_logger = MockLogger()

            class MockData:
                def prepare_data(self) -> None:
                    pass

                def setup(self, stage: str) -> None:
                    pass

                def train_dataloader(self) -> list[Any]:
                    return []

                def val_dataloader(self) -> list[Any]:
                    return []

            from src.training.trainer import TorchTrainer

            trainer = TorchTrainer(
                model=MockModel(),
                data_module=MockData(),
                loggers=[mock_logger],
                run_dir=run_dir,
            )
            trainer.setup()
            trainer.restore_checkpoint(ckpt_path)

            self.assertEqual(len(messages), 1)
            self.assertIn("Loaded checkpoint", messages[0])
            self.assertIn(str(ckpt_path), messages[0])
            self.assertIn("epoch: [bold]2[/bold]", messages[0])
            self.assertIn("step: [bold]150[/bold]", messages[0])
            self.assertIn("val/loss: [bold]1.2345[/bold]", messages[0])

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][0], "checkpoint")
            self.assertEqual(events[0][1]["event"], "checkpoint_restore")
            self.assertEqual(events[0][1]["checkpoint"], str(ckpt_path))
            self.assertEqual(events[0][1]["epoch"], 2)
            self.assertEqual(events[0][1]["global_step"], 150)
            self.assertEqual(events[0][1]["best_metric"], 1.2345)
            self.assertEqual(events[0][1]["monitor"], "val/loss")
            self.assertGreater(events[0][1]["size_bytes"], 0)

