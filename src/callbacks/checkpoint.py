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
        mode: str = "auto",
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
        self.mode = mode.lower()
        if self.mode not in {"auto", "min", "max"}:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of 'auto', 'min', 'max'.")
        self.resolved_mode = self._resolve_mode(self.monitor, self.mode)
        self.best_metric = float("-inf") if self.resolved_mode == "max" else float("inf")

    @staticmethod
    def _resolve_mode(monitor: str, mode: str) -> str:
        mode_normalized = mode.lower()
        if mode_normalized in {"min", "max"}:
            return mode_normalized
        if mode_normalized == "auto":
            lower_monitor = monitor.lower()
            min_keywords = ("loss", "err", "error", "perplexity")
            max_keywords = ("acc", "accuracy", "rate", "score", "reward", "validity")
            if any(kw in lower_monitor for kw in min_keywords):
                return "min"
            if any(kw in lower_monitor for kw in max_keywords):
                return "max"
            return "min"
        raise ValueError(f"Invalid mode '{mode}'. Expected one of 'auto', 'min', 'max'.")

    def _is_better(self, metric: float) -> bool:
        if self.resolved_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        metric = metrics.get(self.monitor)
        is_best = False
        if self.save_best and metric is not None and self._is_better(metric):
            self.best_metric = metric
            is_best = True
        payload = trainer.checkpoint_state()
        payload["callback_state"] = {
            "best_metric": self.best_metric,
            "mode": self.mode,
            "resolved_mode": self.resolved_mode,
            "monitor": self.monitor,
        }
        if self.save_epoch:
            self._save(payload, self.checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")
        self._save(payload, self.checkpoint_dir / "last.pt")
        if is_best:
            self._save(payload, self.checkpoint_dir / "best.pt")

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if "best_metric" in state:
            self.best_metric = float(state["best_metric"])
        if "mode" in state:
            self.mode = str(state["mode"])
        if "resolved_mode" in state:
            self.resolved_mode = str(state["resolved_mode"])
        elif "mode" in state or "monitor" in state:
            self.resolved_mode = self._resolve_mode(self.monitor, self.mode)

    def _save(self, payload: Mapping[str, Any], destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)

