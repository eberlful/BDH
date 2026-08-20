from __future__ import annotations

import dataclasses
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.base import BaseModel
from ..core.registry import MODEL_REGISTRY
from ..optim import build_optimizer


@dataclasses.dataclass
class BDHCQConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    latent_reasoning_steps: int = 1
    loss_schedule: str = "ramp"
    enable_pondernet: bool = False
    ponder_lambda_p: float = 0.2
    ponder_beta: float = 0.01
    ponder_halt_threshold: float = 0.95


def compute_loss_schedule_weights(steps: int, schedule: str) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if schedule == "ramp":
        total = sum(range(1, steps + 1))
        return [r / total for r in range(1, steps + 1)]
    elif schedule == "uniform":
        return [1.0 / steps for _ in range(steps)]
    elif schedule == "final_only":
        weights = [0.0] * steps
        weights[-1] = 1.0
        return weights
    else:
        raise ValueError(
            f"Invalid loss_schedule '{schedule}'. Expected one of 'ramp', 'uniform', 'final_only'."
        )


def compute_masked_cross_entropy_per_sample(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token cross-entropy per sample without diluting masked positions."""
    if logits.ndim != target_ids.ndim + 1:
        raise ValueError(
            "logits must have exactly one more dimension than target_ids "
            f"(got {logits.shape} and {target_ids.shape})"
        )
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            "logits and target_ids must agree on batch/sequence dimensions "
            f"(got {logits.shape} and {target_ids.shape})"
        )

    valid = target_ids.ne(ignore_index)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(target_ids)
    valid_count = valid.sum(dim=-1).clamp_min(1)
    return token_losses.sum(dim=-1) / valid_count


def compute_geometric_prior(
    steps: int,
    lambda_p: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes a geometric prior distribution over reasoning steps [1..R]."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not (0.0 < lambda_p <= 1.0):
        raise ValueError("lambda_p must be in (0, 1].")
    if steps == 1:
        return torch.tensor([1.0], device=device, dtype=dtype if dtype is not None else torch.float32)

    r_idx = torch.arange(steps - 1, device=device, dtype=torch.float32)
    p_unscaled = ((1.0 - lambda_p) ** r_idx) * lambda_p
    p_last = torch.clamp(1.0 - p_unscaled.sum(), min=1e-8)
    prior = torch.cat([p_unscaled, p_last.unsqueeze(0)])
    prior = torch.clamp(prior, min=1e-8)
    prior = prior / prior.sum()
    if dtype is not None:
        prior = prior.to(dtype=dtype)
    return prior


def compute_halting_probabilities(
    lambdas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given halting probabilities lambda_{i, r} in (0, 1) with shape [B, R],
    computes:
        p: step execution probability distribution [B, R] where sum_r p_{i, r} = 1
        cum_p: cumulative probability [B, R]
    """
    B, R = lambdas.shape
    if R == 1:
        p = torch.ones_like(lambdas)
        return p, p

    p_list: list[torch.Tensor] = []
    unhalted = torch.ones((B, 1), device=lambdas.device, dtype=lambdas.dtype)
    for r in range(R - 1):
        lam = lambdas[:, r : r + 1]
        step_p = unhalted * lam
        p_list.append(step_p)
        unhalted = unhalted * (1.0 - lam)
    p_list.append(unhalted)
    p = torch.cat(p_list, dim=-1)
    p = torch.clamp(p, min=1e-8)
    p = p / p.sum(dim=-1, keepdim=True)
    cum_p = torch.cumsum(p, dim=-1)
    return p, cum_p


def compute_ponder_kl_loss(
    p: torch.Tensor,
    lambda_p: float,
) -> torch.Tensor:
    """Computes KL(p || prior) averaged across the batch."""
    _, R = p.shape
    prior = compute_geometric_prior(
        R, lambda_p, device=p.device, dtype=p.dtype
    ).unsqueeze(0)
    kl = (
        p
        * (
            torch.log(p.clamp(min=1e-8))
            - torch.log(prior.clamp(min=1e-8))
        )
    ).sum(dim=-1)
    return kl.mean()


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    def quantize(t: torch.Tensor, q: int = 2) -> torch.Tensor:
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    def __init__(self, config: BDHCQConfig) -> None:
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        assert K is Q
        _, _, T, _ = Q.size()

        freqs = self.freqs.float()
        r_phases = (
            torch.arange(
                0,
                T,
                device=freqs.device,
                dtype=torch.float32,
            ).view(1, 1, -1, 1)
        ) * freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDHCQ(nn.Module):
    """BDH-CQ baseline architecture with associative memory & latent reasoning support."""

    def __init__(self, config: BDHCQConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        if config.enable_pondernet:
            self.halt_head = nn.Linear(D, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_halt_lambda(self, h_seq: torch.Tensor) -> torch.Tensor:
        h_pool = h_seq.squeeze(1).mean(dim=1)
        if hasattr(self, "halt_head"):
            return torch.sigmoid(self.halt_head(h_pool))
        return torch.full(
            (h_seq.size(0), 1), 0.5, device=h_seq.device, dtype=h_seq.dtype
        )

    def encode_contextual_memory(self, demo_idx: torch.Tensor) -> list[torch.Tensor]:
        """Sequentially ingest demonstration tokens and return per-layer fast-weights rho_{K, l}."""
        B, T_demo = demo_idx.size()
        nh = self.config.n_head
        x_demo = self.embed(demo_idx).unsqueeze(1)
        x_demo = self.ln(x_demo)

        memories: list[torch.Tensor] = []
        for level in range(self.config.n_layer):
            x_demo_latent = x_demo @ self.encoder
            x_demo_sparse = F.relu(x_demo_latent)
            rho_l = x_demo_sparse.transpose(2, 3) @ x_demo.expand(-1, nh, -1, -1)
            memories.append(rho_l)

            yKV = self.attn(Q=x_demo_sparse, K=x_demo_sparse, V=x_demo)
            yKV = self.ln(yKV)
            y_demo_latent = yKV @ self.encoder_v
            y_demo_sparse = F.relu(y_demo_latent)
            xy_demo_sparse = self.drop(x_demo_sparse * y_demo_sparse)
            yMLP = (
                xy_demo_sparse.transpose(1, 2).reshape(B, 1, T_demo, -1) @ self.decoder
            )
            y_demo = self.ln(yMLP)
            x_demo = self.ln(x_demo + y_demo)

        return memories

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        demo_len: int = 0,
        contextual_memory: list[torch.Tensor] | None = None,
        latent_reasoning_steps: int | None = None,
        return_contextual_memory: bool = False,
        return_intermediate_logits: bool = False,
        return_ponder_info: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            list[torch.Tensor],
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        C = self.config

        R = (
            latent_reasoning_steps
            if latent_reasoning_steps is not None
            else C.latent_reasoning_steps
        )
        if R < 1:
            raise ValueError("latent_reasoning_steps must be at least 1.")

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        accumulated_memory: list[torch.Tensor] = []
        intermediate_logits: list[torch.Tensor] = []
        step_lambdas: list[torch.Tensor] = []

        collect_ponder = return_ponder_info or C.enable_pondernet

        if contextual_memory is not None:
            # Query-only sequence evaluated against precomputed contextual memory
            h = x
            for r in range(R):
                for level in range(C.n_layer):
                    rho_l = contextual_memory[level]
                    x_latent = h @ self.encoder
                    x_sparse = F.relu(x_latent)  # B, nh, T, N

                    a_self = self.attn(Q=x_sparse, K=x_sparse, V=h)
                    a_mem = x_sparse @ rho_l  # B, nh, T, D
                    yKV = self.ln(a_self + a_mem)

                    y_latent = yKV @ self.encoder_v
                    y_sparse = F.relu(y_latent)
                    xy_sparse = self.drop(x_sparse * y_sparse)

                    yMLP = (
                        xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
                    )
                    y = self.ln(yMLP)
                    h = self.ln(h + y)

                if collect_ponder:
                    step_lambdas.append(self._compute_halt_lambda(h))

                if return_intermediate_logits:
                    step_logits = h.view(B, T, D) @ self.lm_head
                    intermediate_logits.append(step_logits)

            x = h

        elif demo_len > 0:
            # Unified sequence with demonstrations followed by query tokens
            T_demo = demo_len
            T_query = T - T_demo

            x_demo = x[:, :, :T_demo, :]
            x_query = x[:, :, T_demo:, :]

            for level in range(C.n_layer):
                x_demo_latent = x_demo @ self.encoder
                x_demo_sparse = F.relu(x_demo_latent)  # B, nh, T_demo, N

                # Accumulate fast-weights rho_{K, l}
                rho_l = x_demo_sparse.transpose(2, 3) @ x_demo.expand(-1, nh, -1, -1)
                accumulated_memory.append(rho_l)

                # Demo attention
                yKV_demo = self.attn(Q=x_demo_sparse, K=x_demo_sparse, V=x_demo)
                yKV_demo = self.ln(yKV_demo)
                y_demo_latent = yKV_demo @ self.encoder_v
                y_demo_sparse = F.relu(y_demo_latent)
                xy_demo_sparse = self.drop(x_demo_sparse * y_demo_sparse)
                yMLP_demo = (
                    xy_demo_sparse.transpose(1, 2).reshape(B, 1, T_demo, N * nh)
                    @ self.decoder
                )
                y_demo = self.ln(yMLP_demo)
                x_demo = self.ln(x_demo + y_demo)

            # Query hybrid recurrent reasoning passes
            if T_query > 0:
                h = x_query
                for r in range(R):
                    for level in range(C.n_layer):
                        rho_l = accumulated_memory[level]
                        x_query_latent = h @ self.encoder
                        x_query_sparse = F.relu(x_query_latent)  # B, nh, T_query, N

                        a_self_query = self.attn(
                            Q=x_query_sparse, K=x_query_sparse, V=h
                        )
                        a_mem_query = x_query_sparse @ rho_l  # B, nh, T_query, D
                        yKV_query = self.ln(a_self_query + a_mem_query)

                        y_query_latent = yKV_query @ self.encoder_v
                        y_query_sparse = F.relu(y_query_latent)
                        xy_query_sparse = self.drop(x_query_sparse * y_query_sparse)
                        yMLP_query = (
                            xy_query_sparse.transpose(1, 2).reshape(
                                B, 1, T_query, N * nh
                            )
                            @ self.decoder
                        )
                        y_query = self.ln(yMLP_query)
                        h = self.ln(h + y_query)

                    if collect_ponder:
                        step_lambdas.append(self._compute_halt_lambda(h))

                    if return_intermediate_logits:
                        step_x = torch.cat([x_demo, h], dim=2)
                        step_logits = step_x.view(B, T, D) @ self.lm_head
                        intermediate_logits.append(step_logits)

                x_query = h
                x = torch.cat([x_demo, x_query], dim=2)
            else:
                x = x_demo
                if collect_ponder:
                    step_lambdas = [self._compute_halt_lambda(x_demo)] * R
                if return_intermediate_logits:
                    demo_logits = x.view(B, T, D) @ self.lm_head
                    intermediate_logits = [demo_logits] * R

        else:
            # Baseline execution without demonstrations
            h = x
            for r in range(R):
                for level in range(C.n_layer):
                    x_latent = h @ self.encoder
                    x_sparse = F.relu(x_latent)  # B, nh, T, N

                    yKV = self.attn(
                        Q=x_sparse,
                        K=x_sparse,
                        V=h,
                    )
                    yKV = self.ln(yKV)

                    y_latent = yKV @ self.encoder_v
                    y_sparse = F.relu(y_latent)
                    xy_sparse = self.drop(x_sparse * y_sparse)  # B, nh, T, N

                    yMLP = (
                        xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
                    )  # B, 1, T, D
                    y = self.ln(yMLP)
                    h = self.ln(h + y)

                if collect_ponder:
                    step_lambdas.append(self._compute_halt_lambda(h))

                if return_intermediate_logits:
                    step_logits = h.view(B, T, D) @ self.lm_head
                    intermediate_logits.append(step_logits)

            x = h

        if step_lambdas:
            lambdas = torch.cat(step_lambdas, dim=-1)
            ponder_probs, _ = compute_halting_probabilities(lambdas)
        else:
            lambdas = torch.zeros((B, R), device=idx.device, dtype=torch.float32)
            ponder_probs = (
                torch.ones((B, R), device=idx.device, dtype=torch.float32) / R
            )

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_ponder_info:
            if return_contextual_memory and return_intermediate_logits:
                return (
                    logits,
                    loss,
                    accumulated_memory,
                    intermediate_logits,
                    ponder_probs,
                    lambdas,
                )
            if return_contextual_memory:
                return logits, loss, accumulated_memory, ponder_probs, lambdas
            if return_intermediate_logits:
                return logits, loss, intermediate_logits, ponder_probs, lambdas
            return logits, loss, ponder_probs, lambdas

        if return_contextual_memory and return_intermediate_logits:
            return logits, loss, accumulated_memory, intermediate_logits
        if return_contextual_memory:
            return logits, loss, accumulated_memory
        if return_intermediate_logits:
            return logits, loss, intermediate_logits
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        demo_len: int = 0,
        contextual_memory: list[torch.Tensor] | None = None,
        latent_reasoning_steps: int | None = None,
        eos_token_id: int | list[int] | set[int] | None = None,
    ) -> torch.Tensor:
        prefix_demo = None
        if contextual_memory is None and demo_len > 0:
            prefix_demo = idx[:, :demo_len]
            contextual_memory = self.encode_contextual_memory(prefix_demo)
            idx = idx[:, demo_len:]

        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(
                idx_cond,
                contextual_memory=contextual_memory,
                latent_reasoning_steps=latent_reasoning_steps,
            )
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None:
                if isinstance(eos_token_id, int):
                    if (idx_next == eos_token_id).all():
                        break
                elif isinstance(eos_token_id, (list, tuple, set)):
                    if all(t.item() in eos_token_id for t in idx_next.view(-1)):
                        break

        if prefix_demo is not None:
            idx = torch.cat((prefix_demo, idx), dim=1)
        return idx


@MODEL_REGISTRY.register("bdh_cq")
class ConfiguredBDHCQ(BaseModel):
    """Training adapter for the Dragon Hatchling Contextual Query (BDH-CQ) architecture."""

    def __init__(
        self,
        vocab_size: int | str,
        context_length: int = 256,
        n_layer: int = 6,
        n_embd: int = 256,
        n_head: int = 4,
        dropout: float = 0.1,
        mlp_internal_dim_multiplier: int = 128,
        latent_reasoning_steps: int = 1,
        loss_schedule: str = "ramp",
        enable_pondernet: bool = False,
        ponder_lambda_p: float = 0.2,
        ponder_beta: float = 0.01,
        ponder_halt_threshold: float = 0.95,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        optimizer: str | dict[str, Any] = "adamw",
        optimizer_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if vocab_size == "auto":
            raise ValueError(
                "vocab_size='auto' must be resolved from the data module before model creation."
            )
        vocab_size_int = int(vocab_size)
        if vocab_size_int < 2:
            raise ValueError("vocab_size must be at least 2.")
        if (
            context_length < 1
            or n_layer < 1
            or n_embd < 1
            or n_head < 1
            or mlp_internal_dim_multiplier < 1
        ):
            raise ValueError(
                "context_length, n_layer, n_embd, n_head, and mlp_internal_dim_multiplier must be positive."
            )
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if latent_reasoning_steps < 1:
            raise ValueError("latent_reasoning_steps must be at least 1.")
        if loss_schedule not in ("ramp", "uniform", "final_only"):
            raise ValueError(
                f"Invalid loss_schedule '{loss_schedule}'. Expected one of 'ramp', 'uniform', 'final_only'."
            )
        if ponder_lambda_p <= 0.0 or ponder_lambda_p > 1.0:
            raise ValueError("ponder_lambda_p must be in (0, 1].")
        if ponder_beta < 0.0:
            raise ValueError("ponder_beta must be non-negative.")
        if ponder_halt_threshold <= 0.0 or ponder_halt_threshold > 1.0:
            raise ValueError("ponder_halt_threshold must be in (0, 1].")

        self.vocab_size = vocab_size_int
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params
        self.loss_schedule = loss_schedule
        self.config = BDHCQConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            dropout=dropout,
            n_head=n_head,
            mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
            vocab_size=self.vocab_size,
            latent_reasoning_steps=latent_reasoning_steps,
            loss_schedule=loss_schedule,
            enable_pondernet=enable_pondernet,
            ponder_lambda_p=ponder_lambda_p,
            ponder_beta=ponder_beta,
            ponder_halt_threshold=ponder_halt_threshold,
        )
        self.network = BDHCQ(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        demo_len: int = 0,
        contextual_memory: list[torch.Tensor] | None = None,
        latent_reasoning_steps: int | None = None,
        return_intermediate_logits: bool = False,
        return_ponder_info: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds context_length={self.context_length}."
            )
        if return_ponder_info:
            logits, _, intermediate_logits, ponder_probs, lambdas = self.network(
                input_ids,
                demo_len=demo_len,
                contextual_memory=contextual_memory,
                latent_reasoning_steps=latent_reasoning_steps,
                return_intermediate_logits=True,
                return_ponder_info=True,
            )
            return logits, intermediate_logits, ponder_probs, lambdas

        if return_intermediate_logits:
            logits, _, intermediate_logits = self.network(
                input_ids,
                demo_len=demo_len,
                contextual_memory=contextual_memory,
                latent_reasoning_steps=latent_reasoning_steps,
                return_intermediate_logits=True,
            )
            return logits, intermediate_logits

        logits, _ = self.network(
            input_ids,
            demo_len=demo_len,
            contextual_memory=contextual_memory,
            latent_reasoning_steps=latent_reasoning_steps,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        demo_len: int = 0,
        contextual_memory: list[torch.Tensor] | None = None,
        latent_reasoning_steps: int | None = None,
        eos_token_id: int | list[int] | set[int] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) < 1:
            raise ValueError("input_ids must contain at least one token.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Prompt length {input_ids.size(1)} exceeds context_length={self.context_length}."
            )
        return self.network.generate(
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            demo_len=demo_len,
            contextual_memory=contextual_memory,
            latent_reasoning_steps=latent_reasoning_steps,
            eos_token_id=eos_token_id,
        )

    def _compute_loss_and_logits(
        self,
        batch: Mapping[str, torch.Tensor | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        demo_len = int(batch.get("demo_len", 0))
        latent_reasoning_steps = (
            int(batch["latent_reasoning_steps"])
            if "latent_reasoning_steps" in batch
            else self.config.latent_reasoning_steps
        )
        loss_schedule = (
            str(batch["loss_schedule"])
            if "loss_schedule" in batch
            else self.loss_schedule
        )
        target_ids = batch["target_ids"]

        if self.config.enable_pondernet:
            logits, _, intermediate_logits, ponder_probs, _ = self.network(
                batch["input_ids"],
                demo_len=demo_len,
                latent_reasoning_steps=latent_reasoning_steps,
                return_intermediate_logits=True,
                return_ponder_info=True,
            )
            B, T = batch["input_ids"].shape
            R = len(intermediate_logits)

            step_losses_per_sample = []
            for step_logits in intermediate_logits:
                ce = compute_masked_cross_entropy_per_sample(step_logits, target_ids)
                step_losses_per_sample.append(ce)

            stacked_step_losses = torch.stack(step_losses_per_sample, dim=1)  # [B, R]
            task_loss_per_sample = (ponder_probs * stacked_step_losses).sum(dim=1)  # [B]
            task_loss = task_loss_per_sample.mean()

            kl_loss = compute_ponder_kl_loss(ponder_probs, self.config.ponder_lambda_p)
            total_loss = task_loss + self.config.ponder_beta * kl_loss

            step_indices = torch.arange(
                1, R + 1, device=ponder_probs.device, dtype=ponder_probs.dtype
            ).unsqueeze(0)
            expected_steps = (ponder_probs * step_indices).sum(dim=1).mean()

            stacked_logits = torch.stack(intermediate_logits, dim=1)  # [B, R, T, V]
            weighted_logits = (ponder_probs.view(B, R, 1, 1) * stacked_logits).sum(dim=1)

            extra_metrics = {
                "ponder/task_loss": float(task_loss.item()),
                "ponder/kl_loss": float(kl_loss.item()),
                "ponder/expected_steps": float(expected_steps.item()),
            }
            return total_loss, weighted_logits, extra_metrics

        weights = compute_loss_schedule_weights(latent_reasoning_steps, loss_schedule)
        if len(weights) == 1:
            logits = self(
                batch["input_ids"],
                demo_len=demo_len,
                latent_reasoning_steps=latent_reasoning_steps,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
            )
            return loss, logits, {}

        logits, intermediate_logits = self(
            batch["input_ids"],
            demo_len=demo_len,
            latent_reasoning_steps=latent_reasoning_steps,
            return_intermediate_logits=True,
        )
        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for w, step_logits in zip(weights, intermediate_logits):
            if w > 0:
                step_loss = F.cross_entropy(
                    step_logits.reshape(-1, step_logits.size(-1)),
                    target_ids.reshape(-1),
                )
                loss = loss + w * step_loss
        return loss, logits, {}

    def _compute_loss(
        self,
        batch: Mapping[str, torch.Tensor | int | str],
    ) -> torch.Tensor:
        loss, _, _ = self._compute_loss_and_logits(batch)
        return loss

    def training_step(
        self, batch: Mapping[str, torch.Tensor | int | str], batch_idx: int
    ) -> Mapping[str, torch.Tensor | float]:
        loss, logits, extra_metrics = self._compute_loss_and_logits(batch)
        out: dict[str, torch.Tensor | float] = {"loss": loss, "logits": logits}
        out.update(extra_metrics)
        return out

    def validation_step(
        self, batch: Mapping[str, torch.Tensor | int | str], batch_idx: int
    ) -> Mapping[str, torch.Tensor | float]:
        loss, logits, extra_metrics = self._compute_loss_and_logits(batch)
        out: dict[str, torch.Tensor | float] = {"loss": loss, "logits": logits}
        out.update(extra_metrics)
        return out

    def configure_optimizers(self) -> torch.optim.Optimizer:
        has_fp16 = any(p.dtype == torch.float16 for p in self.parameters())
        return build_optimizer(
            self.parameters(),
            optimizer=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_params=self.optimizer_params,
            dtype_has_fp16=has_fp16,
        )
