"""Core configuration, registries, and extension APIs."""

from .base import BaseCallback, BaseDataModule, BaseLogger, BaseModel, BaseTrainer
from .registry import (
    CALLBACK_REGISTRY,
    DATA_REGISTRY,
    LOGGER_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    load_builtin_components,
)

__all__ = [
    "BaseCallback",
    "BaseDataModule",
    "BaseLogger",
    "BaseModel",
    "BaseTrainer",
    "CALLBACK_REGISTRY",
    "DATA_REGISTRY",
    "LOGGER_REGISTRY",
    "MODEL_REGISTRY",
    "TRAINER_REGISTRY",
    "load_builtin_components",
]

