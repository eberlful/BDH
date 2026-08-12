"""Checkpoint callback with epoch, best, and resumable-last checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..core.base import BaseCallback, BaseTrainer
from ..core.registry import CALLBACK_REGISTRY


@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path | None = None,
        save_best: bool = True,
        save_epoch: bool = True,
        monitor: str = "val/loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(run_dir, **kwargs)
        if self.run_dir is None:
            raise ValueError("CheckpointCallback requires a run directory.")
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_epoch = save_epoch
        self.monitor = monitor
        self.best_metric = float("inf")

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        metric = metrics.get(self.monitor)
        if self.save_best and metric is not None and metric < self.best_metric:
            self.best_metric = metric
        payload = trainer.checkpoint_state()
        payload["callback_state"] = {"best_metric": self.best_metric}
        if self.save_epoch:
            self._save(payload, self.checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")
        self._save(payload, self.checkpoint_dir / "last.pt")
        if self.save_best and metric is not None and metric <= self.best_metric:
            self._save(payload, self.checkpoint_dir / "best.pt")

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if "best_metric" in state:
            self.best_metric = float(state["best_metric"])

    def _save(self, payload: Mapping[str, Any], destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
