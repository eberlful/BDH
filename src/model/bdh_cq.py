from __future__ import annotations

import dataclasses
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.base import BaseModel
from ..core.registry import MODEL_REGISTRY


@dataclasses.dataclass
class BDHCQConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    latent_reasoning_steps: int = 1


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
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
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

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor]]
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

                    if return_intermediate_logits:
                        step_x = torch.cat([x_demo, h], dim=2)
                        step_logits = step_x.view(B, T, D) @ self.lm_head
                        intermediate_logits.append(step_logits)

                x_query = h
                x = torch.cat([x_demo, x_query], dim=2)
            else:
                x = x_demo
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

                if return_intermediate_logits:
                    step_logits = h.view(B, T, D) @ self.lm_head
                    intermediate_logits.append(step_logits)

            x = h

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
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

        self.vocab_size = vocab_size_int
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.config = BDHCQConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            dropout=dropout,
            n_head=n_head,
            mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
            vocab_size=self.vocab_size,
            latent_reasoning_steps=latent_reasoning_steps,
        )
        self.network = BDHCQ(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        demo_len: int = 0,
        contextual_memory: list[torch.Tensor] | None = None,
        latent_reasoning_steps: int | None = None,
        return_intermediate_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds context_length={self.context_length}."
            )
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
        )

    def training_step(
        self, batch: Mapping[str, torch.Tensor | int], batch_idx: int
    ) -> Mapping[str, torch.Tensor]:
        demo_len = int(batch.get("demo_len", 0))
        latent_reasoning_steps = (
            int(batch["latent_reasoning_steps"])
            if "latent_reasoning_steps" in batch
            else None
        )
        logits = self(
            batch["input_ids"],
            demo_len=demo_len,
            latent_reasoning_steps=latent_reasoning_steps,
        )
        return {
            "loss": F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch["target_ids"].reshape(-1)
            )
        }

    def validation_step(
        self, batch: Mapping[str, torch.Tensor | int], batch_idx: int
    ) -> Mapping[str, torch.Tensor]:
        demo_len = int(batch.get("demo_len", 0))
        latent_reasoning_steps = (
            int(batch["latent_reasoning_steps"])
            if "latent_reasoning_steps" in batch
            else None
        )
        logits = self(
            batch["input_ids"],
            demo_len=demo_len,
            latent_reasoning_steps=latent_reasoning_steps,
        )
        return {
            "loss": F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch["target_ids"].reshape(-1)
            )
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
