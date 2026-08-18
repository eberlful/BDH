"""Sudoku Chain-of-Thought (CoT) dataset and data module for BDH with GPT-2 tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.base import BaseDataModule
from ..core.registry import DATA_REGISTRY
from .data import (
    SUDOKU_CELL_COUNT,
    SUDOKU_SIZE,
    _generate_solved_board,
    _load_tokenizer,
    is_valid_sudoku_board,
)


def format_sudoku_grid(board: tuple[int, ...] | list[int]) -> str:
    """Format an 81-digit board into a 9x9 space-separated grid string."""
    if len(board) != SUDOKU_CELL_COUNT:
        raise ValueError(f"Board must have {SUDOKU_CELL_COUNT} cells, got {len(board)}.")
    rows: list[str] = []
    for r in range(SUDOKU_SIZE):
        start = r * SUDOKU_SIZE
        row_digits = [str(board[start + c]) for c in range(SUDOKU_SIZE)]
        rows.append(" ".join(row_digits))
    return "\n".join(rows)


def parse_sudoku_grid(text: str) -> list[int]:
    """Parse a Sudoku grid string (81 digits or 9x9 space/newline separated) to list of 81 ints."""
    digits = [int(c) for c in text if c.isdigit()]
    if len(digits) != SUDOKU_CELL_COUNT:
        raise ValueError(f"Expected {SUDOKU_CELL_COUNT} digits, found {len(digits)}.")
    return digits


def generate_cot_steps(puzzle: tuple[int, ...] | list[int], solution: tuple[int, ...] | list[int]) -> str:
    """Generate clean space-separated CoT thinking steps for all blank cells in the puzzle.
    
    Each line uses the format: '- R {row} C {col} = {digit}'
    Spaced tokens ensure clean, single-token decomposition in GPT-2 BPE.
    """
    steps: list[str] = []
    for r in range(1, SUDOKU_SIZE + 1):
        for c in range(1, SUDOKU_SIZE + 1):
            idx = (r - 1) * SUDOKU_SIZE + (c - 1)
            if puzzle[idx] == 0:
                steps.append(f"- R {r} C {c} = {solution[idx]}")
    return "\n".join(steps)


ALLOWED_REASONING_MODES = ("full", "none", "context_only")


def build_sudoku_cot_prompt(
    puzzle: tuple[int, ...] | list[int],
    reasoning_mode: str = "full",
) -> str:
    """Build the prompt prefix for Sudoku based on reasoning mode."""
    if reasoning_mode not in ALLOWED_REASONING_MODES:
        raise ValueError(
            f"reasoning_mode must be one of {ALLOWED_REASONING_MODES}, got {reasoning_mode!r}."
        )
    grid_str = format_sudoku_grid(puzzle)
    if reasoning_mode == "none":
        return f"Sudoku:\n{grid_str}\n\nSolution:\n"
    return f"Sudoku:\n{grid_str}\n\nThinking:\n"


def build_sudoku_cot_full_text(
    puzzle: tuple[int, ...] | list[int],
    solution: tuple[int, ...] | list[int],
    reasoning_mode: str = "full",
    eos_token: str = "<|endoftext|>",
) -> tuple[str, str]:
    """Build prompt and full target text for a Sudoku sample under a specific reasoning mode."""
    if reasoning_mode not in ALLOWED_REASONING_MODES:
        raise ValueError(
            f"reasoning_mode must be one of {ALLOWED_REASONING_MODES}, got {reasoning_mode!r}."
        )
    prompt_text = build_sudoku_cot_prompt(puzzle, reasoning_mode=reasoning_mode)
    solution_grid = format_sudoku_grid(solution)
    if reasoning_mode == "none":
        full_text = f"{prompt_text}{solution_grid}\n{eos_token}"
    else:
        cot_steps = generate_cot_steps(puzzle, solution)
        full_text = f"{prompt_text}{cot_steps}\n\nSolution:\n{solution_grid}\n{eos_token}"
    return prompt_text, full_text


@dataclass(frozen=True)
class SudokuCoTSample:
    puzzle: tuple[int, ...]
    solution: tuple[int, ...]
    prompt_text: str
    full_text: str
    input_ids: torch.Tensor
    target_ids: torch.Tensor


class SudokuCoTDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset for Sudoku Chain-of-Thought with configurable reasoning modes and prompt loss masking."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        context_length: int = 1024,
        reasoning_mode: str = "full",
        pad_token_id: int = 50256,  # <|endoftext|> in gpt2
    ) -> None:
        if not samples:
            raise ValueError("SudokuCoTDataset must contain at least one sample.")
        if reasoning_mode not in ALLOWED_REASONING_MODES:
            raise ValueError(
                f"reasoning_mode must be one of {ALLOWED_REASONING_MODES}, got {reasoning_mode!r}."
            )
        self.samples = samples
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.reasoning_mode = reasoning_mode
        self.pad_token_id = pad_token_id

        self.input_ids_list: list[torch.Tensor] = []
        self.target_ids_list: list[torch.Tensor] = []
        self._encode_all()

    def _encode_text(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            except TypeError:
                return self.tokenizer.encode(text)
        raise TypeError("Tokenizer must have an encode method.")

    def _encode_all(self) -> None:
        for item in self.samples:
            puzzle = item["puzzle"]
            solution = item["solution"]
            prompt_text, full_text = build_sudoku_cot_full_text(
                puzzle, solution, reasoning_mode=self.reasoning_mode
            )

            prompt_ids = self._encode_text(prompt_text)
            full_ids = self._encode_text(full_text)

            # Causal LM: input_ids = full_ids[:-1], target_ids = full_ids[1:]
            seq_input = full_ids[:-1]
            seq_target = full_ids[1:]

            # Truncate if exceeds context_length
            if len(seq_input) > self.context_length:
                seq_input = seq_input[: self.context_length]
                seq_target = seq_target[: self.context_length]

            curr_len = len(seq_input)
            input_tensor = torch.full((self.context_length,), self.pad_token_id, dtype=torch.long)
            target_tensor = torch.full((self.context_length,), -100, dtype=torch.long)

            input_tensor[:curr_len] = torch.tensor(seq_input, dtype=torch.long)
            target_tensor[:curr_len] = torch.tensor(seq_target, dtype=torch.long)

            # Determine loss masking boundary based on reasoning_mode:
            # - 'full': mask prompt prefix transitions (prompt_len - 1)
            # - 'none': mask prompt prefix transitions (prompt_len - 1)
            # - 'context_only': mask prompt and CoT steps up to '\n\nSolution:\n'
            if self.reasoning_mode == "context_only":
                cot_steps = generate_cot_steps(puzzle, solution)
                context_prefix = f"{prompt_text}{cot_steps}\n\nSolution:\n"
                context_ids = self._encode_text(context_prefix)
                mask_len = min(max(0, len(context_ids) - 1), curr_len)
            else:
                mask_len = min(max(0, len(prompt_ids) - 1), curr_len)

            target_tensor[:mask_len] = -100

            self.input_ids_list.append(input_tensor)
            self.target_ids_list.append(target_tensor)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids_list[index],
            "target_ids": self.target_ids_list[index],
            "labels": self.target_ids_list[index],
        }


def _generate_sudoku_cot_raw_samples(
    count: int,
    clues: int,
    seed: int,
    forbidden_puzzles: set[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    forbidden = forbidden_puzzles or set()
    samples: list[dict[str, Any]] = []
    while len(samples) < count:
        solution = _generate_solved_board(rng)
        blank_positions = rng.sample(range(SUDOKU_CELL_COUNT), SUDOKU_CELL_COUNT - clues)
        puzzle_values = list(solution)
        for position in blank_positions:
            puzzle_values[position] = 0
        puzzle = tuple(puzzle_values)
        if puzzle in forbidden or any(s["puzzle"] == puzzle for s in samples):
            continue
        samples.append({"puzzle": puzzle, "solution": solution})
    return samples


@DATA_REGISTRY.register("sudoku_cot")
class SudokuCoTDataModule(BaseDataModule):
    """Generate Sudoku puzzles with configurable reasoning modes tokenized with GPT-2."""

    def __init__(
        self,
        num_samples: int = 10_000,
        validation_fraction: float = 0.1,
        clues: int = 30,
        batch_size: int = 16,
        context_length: int = 1024,
        reasoning_mode: str = "full",
        tokenizer: str = "gpt2",
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if num_samples < 2 or batch_size < 1 or context_length < 1:
            raise ValueError("num_samples must be at least 2; batch_size and context_length must be positive.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0 <= clues <= SUDOKU_CELL_COUNT:
            raise ValueError(f"clues must be between 0 and {SUDOKU_CELL_COUNT}.")
        if reasoning_mode not in ALLOWED_REASONING_MODES:
            raise ValueError(
                f"reasoning_mode must be one of {ALLOWED_REASONING_MODES}, got {reasoning_mode!r}."
            )

        self.num_samples = num_samples
        self.validation_fraction = validation_fraction
        self.clues = clues
        self.batch_size = batch_size
        self.context_length = context_length
        self.reasoning_mode = reasoning_mode
        self.tokenizer_name = tokenizer
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle

        try:
            self.tokenizer = _load_tokenizer(tokenizer)
        except Exception as exc:
            raise ValueError(f"Could not load tokenizer {tokenizer!r}.") from exc

        self.vocab_size = getattr(self.tokenizer, "n_vocab", 50257)
        self.eos_token_id: int = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.train_dataset: SudokuCoTDataset | None = None
        self.val_dataset: SudokuCoTDataset | None = None

    def prepare_data(self) -> None:
        """Sudoku examples are generated dynamically."""

    def setup(self, stage: str | None = None) -> None:
        train_count = int(self.num_samples * (1.0 - self.validation_fraction))
        val_count = self.num_samples - train_count
        train_samples = _generate_sudoku_cot_raw_samples(train_count, self.clues, self.seed)
        forbidden = {s["puzzle"] for s in train_samples}
        val_samples = _generate_sudoku_cot_raw_samples(val_count, self.clues, self.seed + 1_000_003, forbidden)

        self.train_dataset = SudokuCoTDataset(
            train_samples,
            self.tokenizer,
            context_length=self.context_length,
            reasoning_mode=self.reasoning_mode,
            pad_token_id=self.eos_token_id,
        )
        self.val_dataset = SudokuCoTDataset(
            val_samples,
            self.tokenizer,
            context_length=self.context_length,
            reasoning_mode=self.reasoning_mode,
            pad_token_id=self.eos_token_id,
        )

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

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting the validation DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
