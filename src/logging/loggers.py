"""Terminal, text-file, and TensorBoard loggers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
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


def _format_time(seconds: float) -> str:
    if seconds < 0:
        return "--:--"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_metric_val(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4f}"


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
        filename: str | None = "terminal.log",
        **kwargs: Any,
    ) -> None:
        super().__init__(run_dir, **kwargs)
        self.console = Console()
        self.verbose = verbose
        self.max_display_length = max_display_length
        self.filename = filename
        self.file_handle: Any = None
        self.file_console: Console | None = None

        # Lifecycle and timing state
        self.total_epochs: int = 0
        self.completed_epochs: int = 0
        self.total_steps: int | None = None
        self.train_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.last_step_time: float | None = None
        self.last_step: int = 0

        if self.run_dir is not None and self.filename:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.file_handle = (self.run_dir / self.filename).open("a", encoding="utf-8")
            self.file_console = Console(
                file=self.file_handle,
                force_terminal=False,
                width=120,
                no_color=True,
                highlight=False,
            )

    def _print(self, *renderables: Any, **kwargs: Any) -> None:
        self.console.print(*renderables, **kwargs)
        if self.file_console is not None:
            self.file_console.print(*renderables, **kwargs)
            if self.file_handle is not None:
                self.file_handle.flush()

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        self._print(f"[bold cyan]Run directory:[/bold cyan] {self.run_dir}")
        if self.verbose:
            self._print(
                "[bold yellow]Verbose logging enabled:[/bold yellow] "
                "displaying training data and model predictions"
            )

    def on_train_start(self, trainer: BaseTrainer) -> None:
        self.total_epochs = max(0, int(getattr(trainer, "max_epochs", 0)))
        self.completed_epochs = min(self.total_epochs, max(0, int(getattr(trainer.state, "epoch", 0))))
        self.total_steps = getattr(trainer, "max_steps", None)
        if self.total_steps is None and hasattr(trainer, "train_loader") and trainer.train_loader is not None:
            try:
                self.total_steps = len(trainer.train_loader) * self.total_epochs
            except (TypeError, AttributeError):
                self.total_steps = None

        self.train_start_time = time.time()
        self.last_step_time = time.time()
        self.last_step = getattr(trainer.state, "global_step", 0)

        self._print(Rule(title="[bold green]Starting Training[/bold green]", characters="━", style="green"))

    def on_epoch_start(self, trainer: BaseTrainer, epoch: int) -> None:
        self.epoch_start_time = time.time()
        total_ep = self.total_epochs or (epoch + 1)
        self._print(Rule(title=f"[bold cyan]Epoch {epoch + 1}/{total_ep}[/bold cyan]", characters="━", style="cyan"))

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        self.completed_epochs = epoch + 1
        elapsed = time.time() - (self.epoch_start_time or time.time())
        now_str = datetime.now().strftime("%H:%M:%S")
        total_ep = self.total_epochs or (epoch + 1)

        scalar_dict = _scalar_metrics(metrics)
        metric_parts = [
            f"[dim]{key}:[/dim] [bold]{_format_metric_val(value)}[/bold]"
            for key, value in scalar_dict.items()
        ]
        metric_text = " │ ".join(metric_parts) if metric_parts else ""

        self._print(
            Rule(
                title=f"[bold green]✦ Epoch {epoch + 1} Summary[/bold green]",
                characters="─",
                style="green",
            )
        )
        summary_line = (
            f"[dim cyan][{now_str}][/dim cyan] [bold green]Epoch {epoch + 1} completed[/bold green] "
            f"in {_format_time(elapsed)}"
        )
        if metric_text:
            summary_line += f" │ {metric_text}"
        self._print(summary_line)

    def on_train_end(self, trainer: BaseTrainer) -> None:
        total_elapsed = time.time() - (self.train_start_time or time.time())
        now_str = datetime.now().strftime("%H:%M:%S")
        self._print(Rule(title="[bold green]✔ Training Complete[/bold green]", characters="━", style="green"))
        self._print(
            f"[dim cyan][{now_str}][/dim cyan] Finished {self.completed_epochs} epoch(s) in {_format_time(total_elapsed)}."
        )

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
        self._print(panel)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        now = time.time()
        now_str = datetime.now().strftime("%H:%M:%S")

        # Step string
        if self.total_steps:
            step_str = f"Step {step:>5d}/{self.total_steps}"
        else:
            step_str = f"Step {step:>5d}"

        # Speed calculation
        speed_str = ""
        eta_str = ""
        if self.last_step_time is not None and step > self.last_step:
            dt = now - self.last_step_time
            d_step = step - self.last_step
            if dt > 0:
                step_rate = d_step / dt
                speed_str = f"{step_rate:.1f} step/s"
                if self.total_steps and step < self.total_steps:
                    remaining_steps = self.total_steps - step
                    eta_sec = remaining_steps / step_rate
                    eta_str = f"ETA: {_format_time(eta_sec)}"
            self.last_step_time = now
            self.last_step = step

        # Scalar metrics formatting
        scalar_dict = _scalar_metrics(metrics)
        metric_parts: list[str] = []
        for key, value in scalar_dict.items():
            formatted_val = _format_metric_val(value)
            metric_parts.append(f"[dim]{key}:[/dim] [bold]{formatted_val}[/bold]")

        parts = [f"[dim cyan][{now_str}][/dim cyan]", f"[bold white]{step_str}[/bold white]"]
        if metric_parts:
            parts.append(" │ ".join(metric_parts))
        if speed_str:
            parts.append(f"[dim]{speed_str}[/dim]")
        if eta_str:
            parts.append(f"[dim green]{eta_str}[/dim green]")

        line = " │ ".join(parts)
        self._print(line)

    def log_message(self, message: str) -> None:
        self._print(message)

    def flush(self) -> None:
        if self.file_handle is not None and not getattr(self.file_handle, "closed", True):
            self.file_handle.flush()

    def close(self) -> None:
        if self.file_handle is not None and not getattr(self.file_handle, "closed", True):
            self.file_handle.close()

    def __del__(self) -> None:
        self.close()


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

    def log_message(self, message: str) -> None:
        self._write("log", {"message": message})

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
