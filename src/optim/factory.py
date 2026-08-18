"""Optimizer factory for configuring AdamW, Adafactor, and other optimizers."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim import Optimizer

from .adafactor import Adafactor


def build_optimizer(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    optimizer: str | dict[str, Any] = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    optimizer_params: dict[str, Any] | None = None,
    dtype_has_fp16: bool = False,
) -> Optimizer:
    """Instantiate and configure an optimizer based on name or config dict."""
    opt_name = "adamw"
    custom_params: dict[str, Any] = {}

    if isinstance(optimizer, str):
        opt_name = optimizer.strip().lower()
    elif isinstance(optimizer, dict):
        opt_name = str(optimizer.get("name", "adamw")).strip().lower()
        if isinstance(optimizer.get("params"), dict):
            custom_params.update(optimizer["params"])
    else:
        raise TypeError(f"optimizer must be a str or dict, got {type(optimizer).__name__}")

    if optimizer_params is not None:
        if not isinstance(optimizer_params, dict):
            raise TypeError(f"optimizer_params must be a dict, got {type(optimizer_params).__name__}")
        custom_params.update(optimizer_params)

    if opt_name in {"adamw", "adam_w"}:
        default_eps = 1e-4 if dtype_has_fp16 else 1e-8
        kwargs: dict[str, Any] = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "eps": default_eps,
        }
        kwargs.update(custom_params)
        return torch.optim.AdamW(params, **kwargs)

    elif opt_name in {"adafactor"}:
        kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "scale_parameter": False,
            "relative_step": False,
        }
        kwargs.update(custom_params)
        return Adafactor(params, **kwargs)

    elif opt_name in {"adam"}:
        default_eps = 1e-4 if dtype_has_fp16 else 1e-8
        kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "eps": default_eps,
        }
        kwargs.update(custom_params)
        return torch.optim.Adam(params, **kwargs)

    elif opt_name in {"sgd"}:
        kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        kwargs.update(custom_params)
        return torch.optim.SGD(params, **kwargs)

    else:
        raise ValueError(
            f"Unsupported optimizer {opt_name!r}. Expected one of 'adamw', 'adafactor', 'adam', 'sgd'."
        )
