"""Terminal, text-file, and TensorBoard loggers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
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


def _decode_tokens(tokens: list[int] | torch.Tensor, data_module: Any = None) -> str:
    """Decode a list or 1D Tensor of token IDs into a human-readable string."""
    if isinstance(tokens, torch.Tensor):
        token_list = tokens.detach().cpu().tolist()
    else:
        token_list = list(tokens)

    if not token_list:
        return ""

    tokenizer = getattr(data_module, "tokenizer", None)
    if tokenizer is not None and callable(getattr(tokenizer, "decode", None)):
        try:
            # Handle negative / masked tokens (e.g. -100 in target sequences)
            has_masked = any(isinstance(t, int) and t < 0 for t in token_list)
            if has_masked:
                chunks: list[str] = []
                current_valid: list[int] = []
                for t in token_list:
                    if t < 0:
                        if current_valid:
                            chunks.append(tokenizer.decode(current_valid))
                            current_valid = []
                        if not chunks or chunks[-1] != "[MASK]":
                            chunks.append("[MASK]")
                    else:
                        current_valid.append(t)
                if current_valid:
                    chunks.append(tokenizer.decode(current_valid))
                return "".join(chunks)
            return tokenizer.decode(token_list)
        except Exception:
            pass

    return " ".join(str(t) for t in token_list)


@LOGGER_REGISTRY.register("terminal")
class TerminalLogger(BaseLogger):
    def __init__(
        self,
        run_dir: Path | None = None,
        verbose: bool = False,
        max_display_length: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(run_dir, **kwargs)
        self.console = Console()
        self.progress: Progress | None = None
        self.epoch_task_id: int | None = None
        self.verbose = verbose
        self.max_display_length = max_display_length

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        self.console.print(f"[bold green]Run directory:[/bold green] {self.run_dir}")
        if self.verbose:
            self.console.print(
                "[bold yellow]Verbose logging enabled:[/bold yellow] "
                "displaying training data and model predictions"
            )

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

    def on_train_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        if not self.verbose:
            return
        step = trainer.state.global_step + 1
        if step == 1 or step % trainer.log_every_n_steps == 0:
            self._log_verbose_batch(trainer, batch, batch_idx, output, step=step)

    def _log_verbose_batch(
        self,
        trainer: BaseTrainer,
        batch: Any,
        batch_idx: int,
        output: Mapping[str, Any],
        step: int,
    ) -> None:
        input_text = "N/A"
        target_text = "N/A"
        pred_text = "N/A"

        data_module = getattr(trainer, "data_module", None)

        if isinstance(batch, Mapping):
            if "input_ids" in batch:
                input_ids = batch["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    sample_input = input_ids[0] if input_ids.ndim >= 2 else input_ids
                    input_text = _decode_tokens(sample_input, data_module)
            if "target_ids" in batch:
                target_ids = batch["target_ids"]
                if isinstance(target_ids, torch.Tensor):
                    sample_target = target_ids[0] if target_ids.ndim >= 2 else target_ids
                    target_text = _decode_tokens(sample_target, data_module)
            elif "labels" in batch:
                target_ids = batch["labels"]
                if isinstance(target_ids, torch.Tensor):
                    sample_target = target_ids[0] if target_ids.ndim >= 2 else target_ids
                    target_text = _decode_tokens(sample_target, data_module)

        logits = output.get("logits") if isinstance(output, Mapping) else None
        if logits is None and hasattr(trainer, "model") and isinstance(batch, Mapping) and "input_ids" in batch:
            try:
                with torch.no_grad():
                    logits = trainer.model(batch["input_ids"])
            except Exception:
                logits = None

        if logits is not None and isinstance(logits, torch.Tensor):
            if logits.ndim >= 3:
                preds = logits[0].argmax(dim=-1)
            elif logits.ndim == 2:
                preds = logits.argmax(dim=-1)
            else:
                preds = logits
            pred_text = _decode_tokens(preds, data_module)

        if len(input_text) > self.max_display_length:
            input_text = input_text[: self.max_display_length] + f" ... [truncated, total {len(input_text)} chars]"
        if len(target_text) > self.max_display_length:
            target_text = target_text[: self.max_display_length] + f" ... [truncated, total {len(target_text)} chars]"
        if len(pred_text) > self.max_display_length:
            pred_text = pred_text[: self.max_display_length] + f" ... [truncated, total {len(pred_text)} chars]"

        content = (
            f"[bold cyan]Input Data:[/bold cyan]\n{escape(input_text)}\n\n"
            f"[bold yellow]Target Data:[/bold yellow]\n{escape(target_text)}\n\n"
            f"[bold green]Predicted Model Output:[/bold green]\n{escape(pred_text)}"
        )
        epoch_num = getattr(trainer.state, "epoch", 0) + 1
        panel = Panel(
            content,
            title=f"[bold magenta]Step {step} | Epoch {epoch_num} Training Sample[/bold magenta]",
            border_style="blue",
        )
        self.console.print(panel)

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
