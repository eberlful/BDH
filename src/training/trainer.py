"""A small, extensible PyTorch training loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..core.base import BaseCallback, BaseDataModule, BaseLogger, BaseModel, BaseTrainer
from ..core.registry import TRAINER_REGISTRY


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
        config: Mapping[str, Any] | None = None,
        run_dir: Path | None = None,
        device: str | torch.device = "auto",
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
        self.config = dict(config or {})
        self.device = self._resolve_device(device)
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
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def setup(self) -> None:
        if self._is_setup:
            return
        self.data_module.prepare_data()
        self.data_module.setup("fit")
        self.model.setup(self.data_module, self)
        self.model.to(self.device)
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
            output = self.model.training_step(batch, batch_idx)
            metrics = _as_metrics(output)
            if "loss" not in metrics:
                raise ValueError("training_step() must return a loss metric.")
            loss = output["loss"] if isinstance(output, Mapping) else output
            if not isinstance(loss, torch.Tensor):
                raise TypeError("The training loss must be a torch.Tensor.")
            (loss / self.gradient_accumulation_steps).backward()
            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or batch_idx + 1 == len(self.train_loader)
            )
            if should_step:
                if self.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            losses.append(metrics["loss"])
            self.on_train_batch_end(batch, batch_idx, {key: value for key, value in metrics.items()})
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
                output = self.model.validation_step(batch, batch_idx)
                metrics = _as_metrics(output)
                if "loss" in metrics:
                    losses.append(metrics["loss"])
                self._call_hook("on_validation_batch_end", batch, batch_idx, metrics)
        result = {"val/loss": float(np.mean(losses)) if losses else float("nan")}
        self.on_validation_end(result)
        return result

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, self.state.global_step)
        for logger in self.loggers:
            logger.flush()

    def _call_hook(self, name: str, *args: Any) -> None:
        for callback in self.callbacks:
            getattr(callback, name)(self, *args)
        for logger in self.loggers:
            method = getattr(logger, name, None)
            if method is not None:
                method(self, *args)

    def on_train_start(self) -> None:
        self.data_module.on_train_start(self)
        self.model.on_train_start(self)
        for component in [*self.callbacks, *self.loggers]:
            component.on_train_start(self)
        for logger in self.loggers:
            logger.log_hyperparameters(self.config)

    def on_train_end(self) -> None:
        self.model.on_train_end(self)
        self.data_module.on_train_end(self)
        for component in [*self.callbacks, *self.loggers]:
            component.on_train_end(self)
        for logger in self.loggers:
            logger.close()
        self.data_module.teardown("fit")
        self.model.teardown(self)

    def on_epoch_start(self, epoch: int) -> None:
        self.model.on_epoch_start(self, epoch)
        for component in [*self.callbacks, *self.loggers]:
            component.on_epoch_start(self, epoch)

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        self.model.on_epoch_end(self, epoch, metrics)
        for component in [*self.callbacks, *self.loggers]:
            component.on_epoch_end(self, epoch, metrics)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self.model.on_train_batch_start(self, batch, batch_idx)
        for component in self.callbacks:
            component.on_train_batch_start(self, batch, batch_idx)

    def on_train_batch_end(self, batch: Any, batch_idx: int, output: Mapping[str, Any]) -> None:
        self.model.on_train_batch_end(self, batch, batch_idx, output)
        for component in self.callbacks:
            component.on_train_batch_end(self, batch, batch_idx, output)

    def on_validation_start(self) -> None:
        self.model.on_validation_start(self)
        for component in [*self.callbacks, *self.loggers]:
            component.on_validation_start(self)

    def on_validation_end(self, metrics: Mapping[str, float]) -> None:
        self.model.on_validation_end(self, metrics)
        for component in [*self.callbacks, *self.loggers]:
            component.on_validation_end(self, metrics)

    def checkpoint_state(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("Cannot create a checkpoint before trainer setup.")
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
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
            torch.set_rng_state(random_state["torch"])
        if random_state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_state["cuda"])
        callback_state = checkpoint.get("callback_state", {})
        for callback in self.callbacks:
            restore_state = getattr(callback, "restore_state", None)
            if restore_state is not None:
                restore_state(callback_state)
