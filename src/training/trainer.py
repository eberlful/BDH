"""A small, extensible PyTorch training loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..core.base import BaseCallback, BaseDataModule, BaseLogger, BaseModel, BaseTrainer, BaseValidator
from ..core.registry import TRAINER_REGISTRY


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _format_metric_val(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4f}"


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move_to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(child, device) for child in value)
    return value


def _as_metrics(output: Any) -> dict[str, float]:
    values = output if isinstance(output, Mapping) else {"loss": output}
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result


@TRAINER_REGISTRY.register("torch")
class TorchTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        data_module: BaseDataModule,
        callbacks: list[BaseCallback] | None = None,
        loggers: list[BaseLogger] | None = None,
        validators: list[BaseValidator] | None = None,
        config: Mapping[str, Any] | None = None,
        run_dir: Path | None = None,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "float16",
        mixed_precision: bool = False,
        compile: bool | Mapping[str, Any] | str = False,
        max_epochs: int = 1,
        max_steps: int | None = None,
        log_every_n_steps: int = 10,
        validate_every_n_epochs: int = 1,
        gradient_clip_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
        **_: Any,
    ) -> None:
        super().__init__(run_dir=run_dir)
        self.model = model
        self.data_module = data_module
        self.callbacks = callbacks or []
        self.loggers = loggers or []
        self.validators = validators or []

        self.config = dict(config or {})
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.mixed_precision = bool(mixed_precision)
        self.compile = compile
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=bool(
                self.mixed_precision
                and self.dtype == torch.float16
                and self.device.type in ("cuda", "mps", "cpu")
            ),
        )
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.validate_every_n_epochs = max(1, validate_every_n_epochs)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.state = TrainingState()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.train_loader: Any = None
        self.val_loader: Any = None
        self._is_setup = False

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype_str = str(dtype).lower().strip()
        if dtype_str in ("float16", "fp16", "torch.float16"):
            return torch.float16
        if dtype_str in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16
        if dtype_str in ("float32", "fp32", "float", "torch.float32"):
            return torch.float32
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported dtypes are 'bfloat16', 'float16', 'float32'."
        )

    def setup(self) -> None:
        if self._is_setup:
            return
        self.data_module.prepare_data()
        self.data_module.setup("fit")
        self.model.setup(self.data_module, self)
        if self.mixed_precision:
            self.model.to(self.device)
        else:
            self.model.to(device=self.device, dtype=self.dtype)
        if self.compile:
            if isinstance(self.compile, dict):
                self.model = torch.compile(self.model, **self.compile)
            elif isinstance(self.compile, str):
                self.model = torch.compile(self.model, mode=self.compile)
            else:
                self.model = torch.compile(self.model)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        configured = self.model.configure_optimizers()
        if isinstance(configured, torch.optim.Optimizer):
            self.optimizer = configured
        elif isinstance(configured, Mapping):
            self.optimizer = configured.get("optimizer")
            self.scheduler = configured.get("scheduler")
        elif isinstance(configured, (tuple, list)) and configured:
            self.optimizer = configured[0]
            self.scheduler = configured[1] if len(configured) > 1 else None
        else:
            raise TypeError("configure_optimizers() must return an optimizer or optimizer mapping.")
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("configure_optimizers() did not provide a valid optimizer.")
        if not self.mixed_precision and self.dtype == torch.float16:
            for group in self.optimizer.param_groups:
                if "eps" in group and group["eps"] < 1e-5:
                    group["eps"] = 1e-4
        self._is_setup = True

    def fit(self, checkpoint_path: Path | None = None) -> None:
        self.setup()
        if checkpoint_path is not None:
            self.restore_checkpoint(checkpoint_path)
        self.on_train_start()
        try:
            while self.state.epoch < self.max_epochs:
                if self.max_steps is not None and self.state.global_step >= self.max_steps:
                    break
                self._fit_epoch(self.state.epoch)
        finally:
            self.on_train_end()

    def _fit_epoch(self, epoch: int) -> None:
        assert self.optimizer is not None
        self.model.train()
        self.on_epoch_start(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for batch_idx, raw_batch in enumerate(self.train_loader):
            if self.max_steps is not None and self.state.global_step >= self.max_steps:
                break
            batch = _move_to_device(raw_batch, self.device)
            self.on_train_batch_start(batch, batch_idx)
            if self.mixed_precision:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=True):
                    output = self.model.training_step(batch, batch_idx)
            else:
                output = self.model.training_step(batch, batch_idx)
            metrics = _as_metrics(output)
            if "loss" not in metrics:
                raise ValueError("training_step() must return a loss metric.")
            loss = output["loss"] if isinstance(output, Mapping) else output
            if not isinstance(loss, torch.Tensor):
                raise TypeError("The training loss must be a torch.Tensor.")
            scaled_loss = loss / self.gradient_accumulation_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or batch_idx + 1 == len(self.train_loader)
            )
            if should_step:
                if self.scaler.is_enabled():
                    if self.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            losses.append(metrics["loss"])
            output_dict = dict(output) if isinstance(output, Mapping) else {"loss": output}
            output_dict.setdefault("loss", metrics["loss"])
            self.on_train_batch_end(batch, batch_idx, output_dict)
            self.state.global_step += 1
            if self.state.global_step % self.log_every_n_steps == 0:
                self.log_metrics({f"train/{key}": value for key, value in metrics.items()})

        epoch_metrics = {"train/loss": float(np.mean(losses)) if losses else float("nan")}
        if (epoch + 1) % self.validate_every_n_epochs == 0 and self.val_loader is not None:
            epoch_metrics.update(self._validate())
            if "val/loss" in epoch_metrics:
                self.state.best_metric = min(self.state.best_metric, epoch_metrics["val/loss"])
        if self.scheduler is not None:
            self.scheduler.step()
        # Checkpoints written by on_epoch_end resume at the next epoch.
        self.state.epoch = epoch + 1
        self.log_metrics(epoch_metrics)
        self.on_epoch_end(epoch, epoch_metrics)

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        self.on_validation_start()
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(self.val_loader):
                batch = _move_to_device(raw_batch, self.device)
                self._call_hook("on_validation_batch_start", batch, batch_idx)
                if self.mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=True):
                        output = self.model.validation_step(batch, batch_idx)
                else:
                    output = self.model.validation_step(batch, batch_idx)
                metrics = _as_metrics(output)
                if "loss" in metrics:
                    losses.append(metrics["loss"])
                self._call_hook("on_validation_batch_end", batch, batch_idx, metrics)
                for validator in self.validators:
                    validator.on_validation_batch(self, batch, batch_idx, metrics)
        result = {"val/loss": float(np.mean(losses)) if losses else float("nan")}
        for validator in self.validators:
            epoch_metrics = validator.on_validation_epoch_end(self)
            for k, v in epoch_metrics.items():
                key = k if k.startswith("val/") else f"val/{k}"
                result[key] = float(v)
        self.on_validation_end(result)
        return result

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, self.state.global_step)
        for logger in self.loggers:
            logger.flush()

    def log_message(self, message: str) -> None:
        for logger in self.loggers:
            logger.log_message(message)
        for logger in self.loggers:
            logger.flush()

    def _call_hook(self, name: str, *args: Any) -> None:
        for logger in self.loggers:
            method = getattr(logger, name, None)
            if method is not None:
                method(self, *args)
        for callback in self.callbacks:
            getattr(callback, name)(self, *args)

    def on_train_start(self) -> None:
        self.data_module.on_train_start(self)
        self.model.on_train_start(self)
        for component in [*self.loggers, *self.callbacks]:
            component.on_train_start(self)
        for logger in self.loggers:
            logger.log_hyperparameters(self.config)

    def on_train_end(self) -> None:
        self.model.on_train_end(self)
        self.data_module.on_train_end(self)
        for component in [*self.loggers, *self.callbacks]:
            component.on_train_end(self)
        for logger in self.loggers:
            logger.close()
        self.data_module.teardown("fit")
        self.model.teardown(self)

    def on_epoch_start(self, epoch: int) -> None:
        self.model.on_epoch_start(self, epoch)
        for component in [*self.loggers, *self.callbacks]:
            component.on_epoch_start(self, epoch)

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        self.model.on_epoch_end(self, epoch, metrics)
        for component in [*self.loggers, *self.callbacks]:
            component.on_epoch_end(self, epoch, metrics)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self.model.on_train_batch_start(self, batch, batch_idx)
        for component in [*self.loggers, *self.callbacks]:
            component.on_train_batch_start(self, batch, batch_idx)

    def on_train_batch_end(self, batch: Any, batch_idx: int, output: Mapping[str, Any]) -> None:
        self.model.on_train_batch_end(self, batch, batch_idx, output)
        for component in [*self.loggers, *self.callbacks]:
            component.on_train_batch_end(self, batch, batch_idx, output)

    def on_validation_start(self) -> None:
        self.model.on_validation_start(self)
        for component in [*self.callbacks, *self.loggers, *self.validators]:
            component.on_validation_start(self)

    def on_validation_end(self, metrics: Mapping[str, float]) -> None:
        self.model.on_validation_end(self, metrics)
        for component in [*self.callbacks, *self.loggers, *self.validators]:
            component.on_validation_end(self, metrics)


    def checkpoint_state(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("Cannot create a checkpoint before trainer setup.")
        raw_model = getattr(self.model, "_orig_mod", self.model)
        return {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
            },
            "random": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def restore_checkpoint(self, checkpoint_path: Path) -> None:
        if self.optimizer is None:
            raise RuntimeError("Cannot restore a checkpoint before trainer setup.")
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in state_dict.items()
            }
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler.is_enabled() and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint.get("state", {})
        self.state = TrainingState(
            epoch=int(state.get("epoch", 0)),
            global_step=int(state.get("global_step", 0)),
            best_metric=float(state.get("best_metric", float("inf"))),
        )
        random_state = checkpoint.get("random", {})
        if random_state.get("python") is not None:
            random.setstate(random_state["python"])
        if random_state.get("numpy") is not None:
            np.random.set_state(random_state["numpy"])
        if random_state.get("torch") is not None:
            torch.set_rng_state(random_state["torch"].detach().cpu())
        if random_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in random_state["cuda"]])
        callback_state = checkpoint.get("callback_state", {})
        for callback in self.callbacks:
            restore_state = getattr(callback, "restore_state", None)
            if restore_state is not None:
                restore_state(callback_state)
        self._log_checkpoint_restored(checkpoint_path, checkpoint)

    def _log_checkpoint_restored(self, checkpoint_path: Path, checkpoint: Mapping[str, Any]) -> None:
        state = checkpoint.get("state", {})
        callback_state = checkpoint.get("callback_state", {})
        epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_metric = callback_state.get("best_metric", state.get("best_metric"))
        monitor = callback_state.get("monitor")

        size_bytes: int | None = None
        size_str: str | None = None
        try:
            if checkpoint_path.exists():
                size_bytes = checkpoint_path.stat().st_size
                size_str = _format_size(size_bytes)
        except OSError:
            pass

        now_str = datetime.now().strftime("%H:%M:%S")

        metric_part_rich = ""
        metric_part_plain = ""
        if best_metric is not None:
            try:
                val = float(best_metric)
                if val not in (float("inf"), float("-inf")):
                    formatted_val = _format_metric_val(val)
                    if monitor:
                        metric_part_rich = f" │ {monitor}: [bold]{formatted_val}[/bold]"
                        metric_part_plain = f" │ {monitor}: {formatted_val}"
                    else:
                        metric_part_rich = f" │ best_metric: [bold]{formatted_val}[/bold]"
                        metric_part_plain = f" │ best_metric: {formatted_val}"
            except (ValueError, TypeError):
                pass

        size_part_rich = f" │ size: [dim]{size_str}[/dim]" if size_str else ""
        size_part_plain = f" │ size: {size_str}" if size_str else ""

        rich_msg = (
            f"[dim cyan][{now_str}][/dim cyan] │ 📦 [bold cyan]Loaded checkpoint:[/bold cyan] "
            f"[cyan]{checkpoint_path}[/cyan] │ epoch: [bold]{epoch}[/bold] │ "
            f"step: [bold]{global_step}[/bold]{metric_part_rich}{size_part_rich}"
        )
        plain_msg = (
            f"[{now_str}] │ 📦 Loaded checkpoint: {checkpoint_path} │ epoch: {epoch} │ "
            f"step: {global_step}{metric_part_plain}{size_part_plain}"
        )

        for logger in getattr(self, "loggers", []):
            if hasattr(logger, "_print"):
                logger._print(rich_msg)
            elif hasattr(logger, "log_message"):
                logger.log_message(plain_msg)
            if hasattr(logger, "_write"):
                payload: dict[str, Any] = {
                    "event": "checkpoint_restore",
                    "checkpoint": str(checkpoint_path),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if best_metric is not None:
                    try:
                        val = float(best_metric)
                        if val not in (float("inf"), float("-inf")):
                            payload["best_metric"] = val
                    except (ValueError, TypeError):
                        pass
                if monitor:
                    payload["monitor"] = monitor
                if size_bytes is not None:
                    payload["size_bytes"] = size_bytes
                logger._write("checkpoint", payload)
            if hasattr(logger, "flush"):
                logger.flush()
