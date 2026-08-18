"""Sudoku task validator with generative evaluation and constraint checking."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import tiktoken
import torch

from ..core.base import BaseDataModule, BaseModel, BaseTrainer, BaseValidator
from ..core.registry import VALIDATOR_REGISTRY
from ..data.data import (
    SUDOKU_CELL_COUNT,
    SUDOKU_END_TOKEN,
    SUDOKU_POSITION_OFFSET,
    SUDOKU_SEPARATOR_TOKEN,
    is_valid_sudoku_board,
)
from ..data.sudoku_cot import build_sudoku_cot_prompt


def extract_solution_grid_from_text(text: str) -> tuple[list[int], bool]:
    """Extract an 81-digit Sudoku solution grid from raw or Chain-of-Thought text."""
    # Check for "Solution:" marker (ignoring earlier CoT numbers)
    solution_match = re.search(r"solution\s*:", text, flags=re.IGNORECASE)
    if solution_match:
        solution_text = text[solution_match.end():]
        digits = [int(c) for c in solution_text if c.isdigit()]
        if len(digits) >= SUDOKU_CELL_COUNT:
            return digits[:SUDOKU_CELL_COUNT], True
        if len(digits) > 0:
            return digits[:SUDOKU_CELL_COUNT], False

    # If no solution marker found or no digits after marker, check whole text
    digits = [int(c) for c in text if c.isdigit()]
    if len(digits) >= SUDOKU_CELL_COUNT:
        return digits[:SUDOKU_CELL_COUNT], True
    return digits[:SUDOKU_CELL_COUNT], False


def extract_solution_grid_from_tokens(
    tokens: list[int], puzzle: tuple[int, ...] | list[int]
) -> tuple[list[int], bool]:
    """Extract an 81-digit board from token-level position/value sequences."""
    board = list(puzzle)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == SUDOKU_END_TOKEN:
            break
        if SUDOKU_POSITION_OFFSET <= t < SUDOKU_POSITION_OFFSET + SUDOKU_CELL_COUNT:
            pos = t - SUDOKU_POSITION_OFFSET
            if i + 1 < len(tokens):
                val = tokens[i + 1]
                if 1 <= val <= 9:
                    board[pos] = val
                    i += 2
                    continue
        elif 1 <= t <= 9 and len(tokens) >= SUDOKU_CELL_COUNT:
            digits = [tok for tok in tokens if 1 <= tok <= 9]
            if len(digits) >= SUDOKU_CELL_COUNT:
                board = digits[:SUDOKU_CELL_COUNT]
                break
        i += 1

    parsed = len(board) == SUDOKU_CELL_COUNT and all(1 <= v <= 9 for v in board)
    return board, parsed


@VALIDATOR_REGISTRY.register("sudoku")
class SudokuValidator(BaseValidator):
    """Task-specific validator for Sudoku reasoning and solution accuracy."""

    def __init__(
        self,
        num_eval_samples: int | str = 64,
        max_new_tokens: int | None = None,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = 1,
        run_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(run_dir=run_dir, **kwargs)
        self.num_eval_samples = num_eval_samples
        self.max_new_tokens = max_new_tokens
        self.greedy = greedy
        self.temperature = temperature
        self.top_k = top_k

    def _resolve_sample_count(self, total: int) -> int:
        if isinstance(self.num_eval_samples, str) and self.num_eval_samples.lower() in ("all", "-1"):
            return total
        try:
            count = int(self.num_eval_samples)
            if count < 0:
                return total
            return min(count, total)
        except (ValueError, TypeError):
            return min(64, total)

    def on_validation_epoch_end(self, trainer: BaseTrainer) -> Mapping[str, float]:
        model = getattr(trainer, "model", None)
        data_module = getattr(trainer, "data_module", None)
        if model is None or data_module is None:
            return {}
        return self.validate(model, data_module, trainer=trainer)

    def validate(
        self,
        model: BaseModel,
        data_module: BaseDataModule,
        trainer: BaseTrainer | None = None,
    ) -> Mapping[str, float]:
        val_dataset = getattr(data_module, "val_dataset", None)
        if val_dataset is None:
            return {
                "val/sudoku_board_accuracy": 0.0,
                "val/sudoku_validity_rate": 0.0,
                "val/sudoku_cell_accuracy": 0.0,
                "val/sudoku_parse_rate": 0.0,
            }

        # Identify sample source
        is_cot = False
        if hasattr(val_dataset, "samples") and val_dataset.samples:
            raw_samples = val_dataset.samples
            is_cot = True
        elif hasattr(val_dataset, "examples") and val_dataset.examples:
            raw_samples = val_dataset.examples
            is_cot = False
        else:
            raw_samples = [val_dataset[i] for i in range(len(val_dataset))]
            is_cot = False

        total_samples = len(raw_samples)
        if total_samples == 0:
            return {
                "val/sudoku_board_accuracy": 0.0,
                "val/sudoku_validity_rate": 0.0,
                "val/sudoku_cell_accuracy": 0.0,
                "val/sudoku_parse_rate": 0.0,
            }

        count = self._resolve_sample_count(total_samples)
        selected_samples = raw_samples[:count]

        # Resolve device
        device = getattr(trainer, "device", None)
        if device is None:
            params = list(model.parameters())
            device = params[0].device if params else torch.device("cpu")
        elif isinstance(device, str):
            if device == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device(device)

        was_training = model.training
        model.eval()

        board_accuracies: list[float] = []
        validity_rates: list[float] = []
        cell_accuracies: list[float] = []
        parse_rates: list[float] = []

        temp = 1.0 if self.greedy else self.temperature
        top_k = 1 if self.greedy else self.top_k

        try:
            with torch.no_grad():
                for sample in selected_samples:
                    if is_cot:
                        puzzle = sample["puzzle"]
                        solution = list(sample["solution"])
                        reasoning_mode = getattr(data_module, "reasoning_mode", "full")
                        prompt_text = build_sudoku_cot_prompt(puzzle, reasoning_mode=reasoning_mode)
                        tokenizer = getattr(data_module, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = tiktoken.get_encoding("gpt2")
                        prompt_ids = tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
                        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

                        tokens_to_gen = self.max_new_tokens
                        if tokens_to_gen is None:
                            context_len = getattr(model, "context_length", 1024)
                            tokens_to_gen = max(1, min(512, context_len - prompt_tensor.size(1)))

                        eos_id = getattr(data_module, "eos_token_id", None)
                        output_ids = model.generate(
                            input_ids=prompt_tensor,
                            max_new_tokens=tokens_to_gen,
                            temperature=temp,
                            top_k=top_k,
                            eos_token_id=eos_id,
                        )
                        gen_tokens = output_ids[0, prompt_tensor.size(1):].tolist()
                        gen_text = tokenizer.decode(gen_tokens)
                        board, parsed = extract_solution_grid_from_text(gen_text)
                    else:
                        if hasattr(sample, "puzzle"):
                            puzzle = sample.puzzle
                            solution = list(sample.solution)
                        elif isinstance(sample, dict) and "puzzle" in sample:
                            puzzle = sample["puzzle"]
                            solution = list(sample["solution"])
                        else:
                            continue

                        prompt_ids = list(puzzle) + [SUDOKU_SEPARATOR_TOKEN]
                        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

                        tokens_to_gen = self.max_new_tokens
                        if tokens_to_gen is None:
                            context_len = getattr(model, "context_length", 256)
                            tokens_to_gen = max(1, min(200, context_len - prompt_tensor.size(1)))

                        eos_id = SUDOKU_END_TOKEN
                        output_ids = model.generate(
                            input_ids=prompt_tensor,
                            max_new_tokens=tokens_to_gen,
                            temperature=temp,
                            top_k=top_k,
                            eos_token_id=eos_id,
                        )
                        gen_tokens = output_ids[0, prompt_tensor.size(1):].tolist()
                        board, parsed = extract_solution_grid_from_tokens(gen_tokens, puzzle)

                    # Compute metrics for this sample
                    parse_rates.append(1.0 if parsed else 0.0)
                    board_accuracies.append(1.0 if (parsed and board == solution) else 0.0)
                    validity_rates.append(1.0 if (parsed and is_valid_sudoku_board(board)) else 0.0)
                    if board:
                        matches = sum(1 for p, s in zip(board, solution[:len(board)]) if p == s)
                        cell_accuracies.append(matches / float(SUDOKU_CELL_COUNT))
                    else:
                        cell_accuracies.append(0.0)
        finally:
            model.train(was_training)

        n = len(board_accuracies) or 1
        return {
            "val/sudoku_board_accuracy": float(sum(board_accuracies) / n),
            "val/sudoku_validity_rate": float(sum(validity_rates) / n),
            "val/sudoku_cell_accuracy": float(sum(cell_accuracies) / n),
            "val/sudoku_parse_rate": float(sum(parse_rates) / n),
        }
