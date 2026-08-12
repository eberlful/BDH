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
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
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

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@MODEL_REGISTRY.register("bdh")
class ConfiguredBDH(BaseModel):
    """Training adapter for the repository's original BDH architecture."""

    def __init__(
        self,
        vocab_size: int | str,
        context_length: int = 256,
        n_layer: int = 6,
        n_embd: int = 256,
        n_head: int = 4,
        dropout: float = 0.1,
        mlp_internal_dim_multiplier: int = 128,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size == "auto":
            raise ValueError("vocab_size='auto' must be resolved from the data module before model creation.")
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.config = BDHConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            dropout=dropout,
            n_head=n_head,
            mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
            vocab_size=int(vocab_size),
        )
        self.network = BDH(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds context_length={self.context_length}."
            )
        logits, _ = self.network(input_ids)
        return logits

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        logits = self(batch["input_ids"])
        return {"loss": F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["target_ids"].reshape(-1))}

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        logits = self(batch["input_ids"])
        return {"loss": F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["target_ids"].reshape(-1))}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


"""Configurable causal Transformer language model."""


@MODEL_REGISTRY.register("bdh_transformer")
class BDHTransformer(BaseModel):
    def __init__(
        self,
        vocab_size: int | str,
        context_length: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size == "auto":
            raise ValueError("vocab_size='auto' must be resolved from the data module before model creation.")
        if int(vocab_size) < 2:
            raise ValueError("vocab_size must be at least 2.")
        if context_length < 1 or d_model < 1 or n_heads < 1 or n_layers < 1:
            raise ValueError("context_length, d_model, n_heads, and n_layers must be positive.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.vocab_size = int(vocab_size)
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        _, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context_length={self.context_length}."
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.lm_head(self.final_norm(hidden))

    @staticmethod
    def _loss(logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        logits = self(batch["input_ids"])
        loss = self._loss(logits, batch["target_ids"])
        return {"loss": loss}

    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        logits = self(batch["input_ids"])
        return {"loss": self._loss(logits, batch["target_ids"])}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
