"""Data modules for language-model training."""

from __future__ import annotations

from pathlib import Path

import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.base import BaseDataModule
from ..core.registry import DATA_REGISTRY


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_ids: list[int], context_length: int) -> None:
        if len(token_ids) <= context_length:
            raise ValueError(
                f"The dataset needs more than context_length={context_length} tokens; "
                f"only {len(token_ids)} were found."
            )
        values = torch.tensor(token_ids, dtype=torch.long)
        self.inputs = values[:-1]
        self.targets = values[1:]
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, (len(self.inputs) - self.context_length) // self.context_length + 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.context_length
        end = start + self.context_length
        return {"input_ids": self.inputs[start:end], "target_ids": self.targets[start:end]}


@DATA_REGISTRY.register("tiny_shakespeare")
class TinyShakespeareDataModule(BaseDataModule):
    """Download, tokenize, split, and batch the Tiny Shakespeare corpus."""

    def __init__(
        self,
        input_file_path: str = "data/tinyshakespeare.txt",
        data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        tokenizer: str = "gpt2",
        context_length: int = 256,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        download_timeout: float = 30.0,
    ) -> None:
        if context_length < 1 or batch_size < 1:
            raise ValueError("context_length and batch_size must be positive.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        self.input_file_path = Path(input_file_path)
        self.data_url = data_url
        self.tokenizer_name = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.download_timeout = download_timeout
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

    def prepare_data(self) -> None:
        if self.input_file_path.exists():
            return
        self.input_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(self.data_url, timeout=self.download_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not download training data from {self.data_url!r}.") from exc
        self.input_file_path.write_text(response.text, encoding="utf-8")

    def setup(self, stage: str | None = None) -> None:
        if not self.input_file_path.exists():
            raise FileNotFoundError(
                f"Training data was not found at {self.input_file_path}. Call prepare_data() first."
            )
        text = self.input_file_path.read_text(encoding="utf-8")
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        split_index = int(len(token_ids) * (1.0 - self.validation_fraction))
        split_index = max(self.context_length + 1, min(split_index, len(token_ids) - self.context_length - 1))
        train_ids = token_ids[:split_index]
        val_ids = token_ids[split_index:]
        self.train_dataset = TokenBlockDataset(train_ids, self.context_length)
        self.val_dataset = TokenBlockDataset(val_ids, self.context_length)
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


def _load_tokenizer(name: str):
    if name == "byte":
        # A deterministic offline tokenizer useful for smoke tests and tiny
        # local corpora. It is still represented by a tiktoken.Encoding.
        return tiktoken.Encoding(
            name="bdh_byte",
            pat_str=r"(?s:.)",
            mergeable_ranks={bytes([value]): value for value in range(256)},
            special_tokens={"<|endoftext|>": 256},
        )
    return tiktoken.get_encoding(name)
