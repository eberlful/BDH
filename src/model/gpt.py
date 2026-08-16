from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.base import BaseModel
from ..core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gpt_model")
@MODEL_REGISTRY.register("bdh_transformer")
class GPTModel(BaseModel):
    """Configurable causal Transformer (GPT) language model."""

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
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        logits = self(batch["input_ids"])
        return {"loss": self._loss(logits, batch["target_ids"]), "logits": logits}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


BDHTransformer = GPTModel
