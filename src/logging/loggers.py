"""Terminal, text-file, and TensorBoard loggers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console
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

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        self.console.print(f"[bold green]Run directory:[/bold green] {self.run_dir}")

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

