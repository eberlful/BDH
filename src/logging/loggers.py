"""Terminal, text-file, and TensorBoard loggers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

from ..core.base import BaseLogger, BaseTrainer
from ..core.registry import LOGGER_REGISTRY


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


@LOGGER_REGISTRY.register("terminal")
class TerminalLogger(BaseLogger):
    def __init__(self, run_dir: Path | None = None, **kwargs: Any) -> None:
        super().__init__(run_dir, **kwargs)
        self.console = Console()
        self.progress: Progress | None = None
        self.epoch_task_id: int | None = None

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        self.console.print(f"[bold green]Run directory:[/bold green] {self.run_dir}")

    def on_train_start(self, trainer: BaseTrainer) -> None:
        total_epochs = max(0, int(getattr(trainer, "max_epochs", 0)))
        completed_epochs = min(total_epochs, max(0, int(getattr(trainer.state, "epoch", 0))))
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("remaining: {task.fields[remaining]}"),
            TextColumn("{task.fields[metrics]}"),
            console=self.console,
        )
        self.progress.start()
        self.epoch_task_id = self.progress.add_task(
            "Epochs",
            total=total_epochs,
            completed=completed_epochs,
            remaining=total_epochs - completed_epochs,
            metrics="",
        )

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        if self.progress is None or self.epoch_task_id is None:
            return
        completed_epochs = min(self.progress.tasks[self.epoch_task_id].total, trainer.state.epoch)
        remaining = max(0, int(self.progress.tasks[self.epoch_task_id].total - completed_epochs))
        metric_text = " ".join(
            f"{key}={value:.4f}" for key, value in _scalar_metrics(metrics).items()
        )
        self.progress.update(
            self.epoch_task_id,
            completed=completed_epochs,
            remaining=remaining,
            metrics=metric_text,
        )

    def on_train_end(self, trainer: BaseTrainer) -> None:
        if self.progress is not None:
            self.progress.stop()

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan")
        table.add_column(style="white")
        table.add_row("step", str(step))
        for key, value in _scalar_metrics(metrics).items():
            table.add_row(key, f"{value:.6f}")
        self.console.print(table)


@LOGGER_REGISTRY.register("text_file")
class TextFileLogger(BaseLogger):
    def __init__(self, run_dir: Path | None = None, filename: str = "training.log", **kwargs: Any) -> None:
        super().__init__(run_dir, **kwargs)
        if self.run_dir is None:
            raise ValueError("TextFileLogger requires a run directory.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.handle = self.path.open("a", encoding="utf-8")

    def _write(self, event: str, payload: Mapping[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.handle.write(f"{timestamp} {event} {json.dumps(payload, sort_keys=True, default=str)}\n")
        self.handle.flush()

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        self._write("hyperparameters", parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._write("metrics", {"step": step, **_scalar_metrics(metrics)})

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


@LOGGER_REGISTRY.register("tensorboard")
class TensorBoardLogger(BaseLogger):
    def __init__(self, run_dir: Path | None = None, subdirectory: str = "tensorboard", **kwargs: Any) -> None:
        super().__init__(run_dir, **kwargs)
        if self.run_dir is None:
            raise ValueError("TensorBoardLogger requires a run directory.")
        self.writer = SummaryWriter(log_dir=str(self.run_dir / subdirectory))

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        flattened = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flattened[prefix] = str(value)

        visit("", parameters)
        self.writer.add_hparams(flattened, {})

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in _scalar_metrics(metrics).items():
            self.writer.add_scalar(key, value, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
