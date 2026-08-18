"""Base extension points for models, data, callbacks, loggers, and trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module, ABC):
    """A model with explicit hooks used by :class:`BaseTrainer`."""

    def setup(self, data_module: BaseDataModule | None = None, trainer: BaseTrainer | None = None) -> None:
        """Prepare model state after the data module has been initialized."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute model outputs."""

    def configure_optimizers(self) -> Any:
        """Return an optimizer, or a mapping containing optimizer/scheduler."""
        raise NotImplementedError("Models must implement configure_optimizers().")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | list[int] | set[int] | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from a causal language model."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) < 1:
            raise ValueError("input_ids must contain at least one token.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        context_length = getattr(self, "context_length", None)
        if context_length is not None and input_ids.size(1) > context_length:
            raise ValueError(
                f"Prompt length {input_ids.size(1)} exceeds context_length={context_length}."
            )

        for _ in range(max_new_tokens):
            idx_cond = input_ids
            if context_length is not None:
                idx_cond = idx_cond[:, -context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k < 1:
                    raise ValueError("top_k must be positive when provided.")
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None:
                if isinstance(eos_token_id, int):
                    if (next_token == eos_token_id).all():
                        break
                elif isinstance(eos_token_id, (list, tuple, set)):
                    if all(t.item() in eos_token_id for t in next_token.view(-1)):
                        break
        return input_ids

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any] | torch.Tensor:
        """Return a loss tensor or a mapping containing a ``loss`` tensor."""
        raise NotImplementedError("Models must implement training_step().")

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any] | torch.Tensor:
        """Return validation metrics in the same shape as training_step."""
        raise NotImplementedError("Models must implement validation_step().")

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: BaseTrainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        pass

    def on_train_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass

    def on_validation_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_validation_end(self, trainer: BaseTrainer, metrics: Mapping[str, float]) -> None:
        pass

    def on_validation_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_validation_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass

    def teardown(self, trainer: BaseTrainer) -> None:
        pass


class BaseDataModule(ABC):
    """Own data preparation, splitting, and DataLoader creation."""

    def prepare_data(self) -> None:
        """Download or otherwise prepare shared data exactly once."""

    def setup(self, stage: str | None = None) -> None:
        """Create datasets and loaders for the requested stage."""

    @abstractmethod
    def train_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """Return the training loader."""

    @abstractmethod
    def val_dataloader(self) -> torch.utils.data.DataLoader[Any] | None:
        """Return the validation loader, if configured."""

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass

    def teardown(self, stage: str | None = None) -> None:
        pass


class BaseCallback(ABC):
    """Optional behavior attached to the trainer lifecycle."""

    def __init__(self, run_dir: Path | None = None, **_: Any) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: BaseTrainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        pass

    def on_train_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass

    def on_validation_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_validation_end(self, trainer: BaseTrainer, metrics: Mapping[str, float]) -> None:
        pass

    def on_validation_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_validation_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass


class BaseLogger(ABC):
    """Metric and lifecycle logging interface."""

    def __init__(self, run_dir: Path | None = None, **_: Any) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def on_train_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: BaseTrainer, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Mapping[str, float]) -> None:
        pass

    def on_train_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass

    def on_validation_start(self, trainer: BaseTrainer) -> None:
        pass

    def on_validation_end(self, trainer: BaseTrainer, metrics: Mapping[str, float]) -> None:
        pass

    def on_validation_batch_start(self, trainer: BaseTrainer, batch: Any, batch_idx: int) -> None:
        pass

    def on_validation_batch_end(
        self, trainer: BaseTrainer, batch: Any, batch_idx: int, output: Mapping[str, Any]
    ) -> None:
        pass

    @abstractmethod
    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Record the resolved training configuration."""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record scalar metrics at a global step."""

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()


class BaseTrainer(ABC):
    """Trainer lifecycle contract and public state/checkpoint hooks."""

    def __init__(self, run_dir: Path | None = None, **_: Any) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def setup(self) -> None:
        pass

    @abstractmethod
    def fit(self, checkpoint_path: Path | None = None) -> None:
        """Run training, optionally restoring a checkpoint first."""

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(self, batch: Any, batch_idx: int, output: Mapping[str, Any]) -> None:
        pass

    def on_validation_start(self) -> None:
        pass

    def on_validation_end(self, metrics: Mapping[str, float]) -> None:
        pass

    def checkpoint_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def restore_checkpoint(self, checkpoint_path: Path) -> None:
        raise NotImplementedError


class BaseValidator(ABC):
    """Task-specific validator lifecycle and evaluation contract."""

    def __init__(self, run_dir: Path | None = None, **_: Any) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None

    def on_validation_start(self, trainer: BaseTrainer) -> None:
        """Called when validation begins."""

    def on_validation_batch(
        self,
        trainer: BaseTrainer,
        batch: Any,
        batch_idx: int,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        """Called for each validation batch during standard validation loops."""

    def on_validation_epoch_end(self, trainer: BaseTrainer) -> Mapping[str, float]:
        """Compute and return metrics at the end of validation batches."""
        return {}

    def validate(
        self,
        model: BaseModel,
        data_module: BaseDataModule,
        trainer: BaseTrainer | None = None,
    ) -> Mapping[str, float]:
        """Perform validation and return computed metrics."""
        return {}

