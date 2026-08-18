"""Adafactor optimizer implementation in pure PyTorch."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.

    Reference:
        Shazeer & Stern (2018): https://arxiv.org/abs/1804.04235
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float | None = 1e-3,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: float | None = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = False,
        relative_step: bool = False,
        warmup_init: bool = False,
    ) -> None:
        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps[0] < 0.0 or eps[1] < 0.0:
            raise ValueError(f"Invalid eps values: {eps}")
        if clip_threshold < 0.0:
            raise ValueError(f"Invalid clip_threshold value: {clip_threshold}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / math.sqrt(max(1, tensor.numel()))

    def _get_lr(self, param_group: dict[str, Any], state: dict[str, Any]) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(state["step"]))
        if rel_step_sz is None:
            rel_step_sz = 1.0

        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], state["RMS"])
        return float(param_scale * rel_step_sz)

    def _get_decay_rate(self, state: dict[str, Any], decay_rate: float) -> float:
        return float(1.0 - (state["step"] ** decay_rate))

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = len(grad_shape) >= 2
                use_first_moment = group["beta1"] is not None and group["beta1"] > 0.0

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1], dtype=grad.dtype, device=grad.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)

                state["step"] += 1
                state["RMS"] = float(self._rms(p))
                lr = self._get_lr(group, state)

                beta2t = self._get_decay_rate(state, group["decay_rate"])
                eps1, eps2 = group["eps"]

                update = (grad**2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    r_factor = (
                        exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp(min=eps1)
                    ).rsqrt()
                    c_factor = exp_avg_sq_col.rsqrt()
                    update = grad * (r_factor.unsqueeze(-1) * c_factor.unsqueeze(-2))
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = grad * exp_avg_sq.rsqrt()

                rms_update = float(self._rms(update))
                if group["clip_threshold"] > 0.0:
                    clip_mult = max(1.0, rms_update / group["clip_threshold"])
                    update.div_(clip_mult)

                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1.0 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * lr)

                p.add_(-update)

        return loss
