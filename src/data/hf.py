"""Hugging Face streaming and in-memory text data modules."""

from __future__ import annotations

import re
from typing import Any, Iterable
import warnings

import datasets
import torch
from torch.utils.data import DataLoader

from ..core.base import BaseDataModule
from ..core.registry import DATA_REGISTRY
from .data import TokenBlockDataset, _load_tokenizer

_ARTICLE_HEADING_PATTERN = re.compile(r"^=\s+[^=]+?\s+=$")


def _resolve_dataset_name(name: str) -> str:
    """Normalize well-known hub dataset names if needed."""
    if "/" not in name:
        if name == "wikitext":
            return "Salesforce/wikitext"
    return name


@DATA_REGISTRY.register("hf_text")
class HuggingFaceTextDataModule(BaseDataModule):
    """Zero-disk in-memory or cached text streaming DataModule using Hugging Face datasets."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None = None,
        tokenizer: str = "gpt2",
        context_length: int = 256,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        train_split: str = "train",
        val_split: str | None = "validation",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        in_memory: bool = True,
        text_column: str = "text",
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        add_eos_token: bool = True,
        load_dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if context_length < 1 or batch_size < 1:
            raise ValueError("context_length and batch_size must be positive.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_name = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.train_split = train_split
        self.val_split = val_split
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.in_memory = in_memory
        self.text_column = text_column
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token
        self.load_dataset_kwargs = dict(load_dataset_kwargs or {})

        try:
            self.tokenizer = _load_tokenizer(tokenizer)
        except Exception as exc:
            raise ValueError(
                f"Could not load tokenizer {tokenizer!r}. Standard tiktoken encodings may "
                "need to be downloaded once before an offline run."
            ) from exc
        self.vocab_size = self.tokenizer.n_vocab
        self.train_dataset: TokenBlockDataset | None = None
        self.val_dataset: TokenBlockDataset | None = None

        # End of text token ID
        self.eos_token_id: int = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def _load_hf_split(self, split: str, streaming: bool) -> Any:
        hf_name = _resolve_dataset_name(self.dataset_name)
        kwargs = dict(self.load_dataset_kwargs)
        if self.dataset_config is not None:
            return datasets.load_dataset(
                hf_name,
                self.dataset_config,
                split=split,
                streaming=streaming,
                **kwargs,
            )
        return datasets.load_dataset(
            hf_name,
            split=split,
            streaming=streaming,
            **kwargs,
        )

    def _process_stream(
        self,
        stream_or_dataset: Iterable[dict[str, Any]],
        max_samples: int | None,
    ) -> list[int]:
        """Convert a stream or dataset of examples into a list of token IDs."""
        token_ids: list[int] = []
        count = 0
        for example in stream_or_dataset:
            if max_samples is not None and count >= max_samples:
                break
            text = example.get(self.text_column, "")
            if not isinstance(text, str) or not text:
                count += 1
                continue
            encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            if encoded:
                token_ids.extend(encoded)
                if self.add_eos_token:
                    token_ids.append(self.eos_token_id)
            count += 1
        return token_ids

    def prepare_data(self) -> None:
        """Download or cache data if needed."""
        if not self.in_memory:
            try:
                self._load_hf_split(self.train_split, streaming=False)
                if self.val_split is not None:
                    self._load_hf_split(self.val_split, streaming=False)
            except Exception:
                pass

    def setup(self, stage: str | None = None) -> None:
        if self.in_memory:
            datasets.disable_caching()

        streaming = self.in_memory

        # Try to resolve validation split
        train_raw = None
        val_raw = None
        has_val_split = False

        if self.val_split is not None:
            try:
                val_raw = self._load_hf_split(self.val_split, streaming=streaming)
                # Check if non-empty / valid by peeking if needed or checking iterator
                has_val_split = True
            except Exception:
                has_val_split = False
                val_raw = None

        train_raw = self._load_hf_split(self.train_split, streaming=streaming)

        if has_val_split and val_raw is not None:
            train_tokens = self._process_stream(train_raw, self.max_train_samples)
            val_tokens = self._process_stream(val_raw, self.max_val_samples)
        else:
            all_tokens = self._process_stream(train_raw, self.max_train_samples)
            if len(all_tokens) <= self.context_length:
                raise ValueError("The corpus is too small for the configured context_length and split.")
            split_index = int(len(all_tokens) * (1.0 - self.validation_fraction))
            split_index = max(self.context_length + 1, min(split_index, len(all_tokens) - self.context_length - 1))
            train_tokens = all_tokens[:split_index]
            val_tokens = all_tokens[split_index:]

        self.train_dataset = TokenBlockDataset(train_tokens, self.context_length)
        self.val_dataset = TokenBlockDataset(val_tokens, self.context_length)

        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            raise ValueError("The corpus is too small for the configured context_length and split.")

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the training DataLoader.")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting the validation DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


@DATA_REGISTRY.register("wikitext")
class WikiTextDataModule(HuggingFaceTextDataModule):
    """WikiText language modeling dataset with zero-disk in-memory streaming support."""

    def __init__(
        self,
        dataset_config: str = "wikitext-2-raw-v1",
        tokenizer: str = "gpt2",
        context_length: int = 256,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        train_split: str = "train",
        val_split: str | None = "validation",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        in_memory: bool = True,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_name="wikitext",
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            context_length=context_length,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            train_split=train_split,
            val_split=val_split,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            in_memory=in_memory,
            text_column="text",
            num_workers=num_workers,
            seed=seed,
            shuffle=shuffle,
            add_eos_token=False,  # Custom boundary handling per article
            **kwargs,
        )

    def _process_stream(
        self,
        stream_or_dataset: Iterable[dict[str, Any]],
        max_samples: int | None,
    ) -> list[int]:
        """Process WikiText stream: filter whitespace lines, delimit articles with <|endoftext|>."""
        token_ids: list[int] = []
        count = 0
        for example in stream_or_dataset:
            if max_samples is not None and count >= max_samples:
                break
            raw_text = example.get(self.text_column, "")
            count += 1
            if not isinstance(raw_text, str):
                continue
            stripped = raw_text.strip()
            if not stripped:
                continue

            # Check if this line starts a new article
            if _ARTICLE_HEADING_PATTERN.match(stripped):
                if token_ids and token_ids[-1] != self.eos_token_id:
                    token_ids.append(self.eos_token_id)

            encoded = self.tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
            token_ids.extend(encoded)

        if token_ids and token_ids[-1] != self.eos_token_id:
            token_ids.append(self.eos_token_id)

        return token_ids
