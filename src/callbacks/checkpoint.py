"""Checkpoint callback with epoch, best, and resumable-last checkpoints."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from ..core.base import BaseCallback, BaseTrainer
from ..core.registry import CALLBACK_REGISTRY


def _format_metric_val(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4f}"


def _sanitize_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())
    return sanitized.strip("_")


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
        prev_best = self.best_metric
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
        if is_best and metric is not None:
            clean_monitor = _sanitize_metric_name(self.monitor)
            val_str = _format_metric_val(metric)
            best_filename = f"best_epoch-{epoch + 1:04d}_{clean_monitor}-{val_str}.pt"
            self._save(payload, self.checkpoint_dir / best_filename)
            self._save(payload, self.checkpoint_dir / "best.pt")
            self._log_best_checkpoint(trainer, epoch, prev_best, metric, best_filename)

    def _log_best_checkpoint(
        self,
        trainer: BaseTrainer,
        epoch: int,
        prev_best: float,
        new_best: float,
        checkpoint_filename: str,
    ) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        has_prev = prev_best not in (float("inf"), float("-inf"))
        prev_str = _format_metric_val(prev_best) if has_prev else None
        new_str = _format_metric_val(new_best)

        improvement_suffix = f" (improved from {prev_str})" if prev_str is not None else ""
        rich_improvement = f" (improved from [dim]{prev_str}[/dim])" if prev_str is not None else ""

        rich_msg = (
            f"[dim cyan][{now_str}][/dim cyan] │ ⭐ [bold green]New best checkpoint:[/bold green] "
            f"[cyan]{checkpoint_filename}[/cyan] │ {self.monitor}: [bold]{new_str}[/bold]{rich_improvement}"
        )
        plain_msg = (
            f"[{now_str}] │ ⭐ New best checkpoint: {checkpoint_filename} │ "
            f"{self.monitor}: {new_str}{improvement_suffix}"
        )

        for logger in getattr(trainer, "loggers", []):
            if hasattr(logger, "_print"):
                logger._print(rich_msg)
            elif hasattr(logger, "log_message"):
                logger.log_message(plain_msg)
            if hasattr(logger, "_write"):
                logger._write(
                    "checkpoint",
                    {
                        "epoch": epoch + 1,
                        "event": "best_checkpoint",
                        "monitor": self.monitor,
                        "previous_best": prev_best if has_prev else None,
                        "best_metric": new_best,
                        "filename": checkpoint_filename,
                    },
                )

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


