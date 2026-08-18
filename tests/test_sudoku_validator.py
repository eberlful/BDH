from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import tiktoken
import torch
from torch import nn

from src.core.base import BaseModel
from src.core.registry import VALIDATOR_REGISTRY, load_builtin_components
from src.data.data import (
    SUDOKU_CELL_COUNT,
    SUDOKU_END_TOKEN,
    SUDOKU_POSITION_OFFSET,
    SUDOKU_SEPARATOR_TOKEN,
    SudokuDataModule,
)
from src.data.sudoku_cot import SudokuCoTDataModule, build_sudoku_cot_prompt, format_sudoku_grid
from src.validation.sudoku import SudokuValidator, extract_solution_grid_from_text



class MockGenerationModel(BaseModel):
    def __init__(self, mode: str, tokenizer: Any = None, context_length: int = 1024) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.mode = mode
        self.tokenizer = tokenizer or tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        self.custom_solution: list[int] | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return torch.zeros(1, 1, 10)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | list[int] | set[int] | None = None,
    ) -> torch.Tensor:
        # Generate based on model mode
        if self.mode == "perfect":
            # Return valid ground truth representation
            if self.custom_solution is not None:
                grid_str = format_sudoku_grid(self.custom_solution)
            else:
                grid_str = format_sudoku_grid([1] * 81)
            text = f"Thinking steps here\n\nSolution:\n{grid_str}\n<|endoftext|>"
            gen_tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        elif self.mode == "invalid_rules":
            # 81 digits of all 1s (violates uniqueness)
            grid_str = format_sudoku_grid([1] * 81)
            text = f"Solution:\n{grid_str}\n<|endoftext|>"
            gen_tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        elif self.mode == "gibberish":
            text = "I don't know how to solve this puzzle sorry!"
            gen_tokens = self.tokenizer.encode(text)
        elif self.mode == "token_perfect":
            # Return token format: position-value pairs for blanks followed by END
            gen_tokens = []
            if self.custom_solution is not None:
                for idx, val in enumerate(self.custom_solution):
                    gen_tokens.extend([SUDOKU_POSITION_OFFSET + idx, val])
            gen_tokens.append(SUDOKU_END_TOKEN)
        elif self.mode == "token_gibberish":
            gen_tokens = [999, 999]
        else:
            raise ValueError(f"Unknown mock mode {self.mode}")

        out = torch.cat(
            [input_ids, torch.tensor([gen_tokens], dtype=torch.long, device=input_ids.device)],
            dim=1,
        )
        return out


class SudokuValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        load_builtin_components()

    def test_registered_in_validator_registry(self) -> None:
        self.assertIn("sudoku", VALIDATOR_REGISTRY.names())
        validator = VALIDATOR_REGISTRY.instantiate("sudoku")
        self.assertIsInstance(validator, SudokuValidator)

    def test_extract_solution_grid_from_text(self) -> None:
        # Perfect 9x9 grid after Solution: marker
        valid_sol = [
            5, 3, 4, 6, 7, 8, 9, 1, 2,
            6, 7, 2, 1, 9, 5, 3, 4, 8,
            1, 9, 8, 3, 4, 2, 5, 6, 7,
            8, 5, 9, 7, 6, 1, 4, 2, 3,
            4, 2, 6, 8, 5, 3, 7, 9, 1,
            7, 1, 3, 9, 2, 4, 8, 5, 6,
            9, 6, 1, 5, 3, 7, 2, 8, 4,
            2, 8, 7, 4, 1, 9, 6, 3, 5,
            3, 4, 5, 2, 8, 6, 1, 7, 9,
        ]
        text_with_cot = f"- R 1 C 1 = 5\n- R 1 C 2 = 3\n\nSolution:\n{format_sudoku_grid(valid_sol)}\n<|endoftext|>"
        board, parsed = extract_solution_grid_from_text(text_with_cot)
        self.assertTrue(parsed)
        self.assertEqual(board, valid_sol)

        # Direct format without Solution: marker
        raw_text = format_sudoku_grid(valid_sol)
        board, parsed = extract_solution_grid_from_text(raw_text)
        self.assertTrue(parsed)
        self.assertEqual(board, valid_sol)

        # Gibberish text
        gibberish = "Invalid output without digits"
        board, parsed = extract_solution_grid_from_text(gibberish)
        self.assertFalse(parsed)
        self.assertEqual(len(board), 0)

    def test_eval_sudoku_cot_perfect_model(self) -> None:
        dm = SudokuCoTDataModule(num_samples=10, validation_fraction=0.5, clues=30, batch_size=2, seed=42)
        dm.prepare_data()
        dm.setup("fit")
        assert dm.val_dataset is not None

        # Pick ground truth of first val sample for mock model
        first_solution = list(dm.val_dataset.samples[0]["solution"])
        model = MockGenerationModel(mode="perfect", tokenizer=dm.tokenizer)
        model.custom_solution = first_solution

        validator = SudokuValidator(num_eval_samples=1)
        metrics = validator.validate(model, dm)

        self.assertEqual(metrics["val/sudoku_board_accuracy"], 1.0)
        self.assertEqual(metrics["val/sudoku_validity_rate"], 1.0)
        self.assertEqual(metrics["val/sudoku_cell_accuracy"], 1.0)
        self.assertEqual(metrics["val/sudoku_parse_rate"], 1.0)

    def test_eval_sudoku_cot_rule_violating_model(self) -> None:
        dm = SudokuCoTDataModule(num_samples=10, validation_fraction=0.5, clues=30, batch_size=2, seed=42)
        dm.prepare_data()
        dm.setup("fit")

        model = MockGenerationModel(mode="invalid_rules", tokenizer=dm.tokenizer)
        validator = SudokuValidator(num_eval_samples=2)
        metrics = validator.validate(model, dm)

        self.assertEqual(metrics["val/sudoku_board_accuracy"], 0.0)
        self.assertEqual(metrics["val/sudoku_validity_rate"], 0.0)
        self.assertEqual(metrics["val/sudoku_parse_rate"], 1.0)
        self.assertGreaterEqual(metrics["val/sudoku_cell_accuracy"], 0.0)
        self.assertLess(metrics["val/sudoku_cell_accuracy"], 1.0)

    def test_eval_sudoku_cot_unparseable_model(self) -> None:
        dm = SudokuCoTDataModule(num_samples=10, validation_fraction=0.5, clues=30, batch_size=2, seed=42)
        dm.prepare_data()
        dm.setup("fit")

        model = MockGenerationModel(mode="gibberish", tokenizer=dm.tokenizer)
        validator = SudokuValidator(num_eval_samples=2)
        metrics = validator.validate(model, dm)

        self.assertEqual(metrics["val/sudoku_board_accuracy"], 0.0)
        self.assertEqual(metrics["val/sudoku_validity_rate"], 0.0)
        self.assertEqual(metrics["val/sudoku_cell_accuracy"], 0.0)
        self.assertEqual(metrics["val/sudoku_parse_rate"], 0.0)

    def test_eval_token_level_sudoku_data_module(self) -> None:
        dm = SudokuDataModule(num_samples=10, validation_fraction=0.5, clues=30, batch_size=2, seed=42)
        dm.prepare_data()
        dm.setup("fit")
        assert dm.val_dataset is not None

        first_solution = list(dm.val_dataset.examples[0].solution)
        model = MockGenerationModel(mode="token_perfect")
        model.custom_solution = first_solution

        validator = SudokuValidator(num_eval_samples=1)
        metrics = validator.validate(model, dm)

        self.assertEqual(metrics["val/sudoku_board_accuracy"], 1.0)
        self.assertEqual(metrics["val/sudoku_validity_rate"], 1.0)
        self.assertEqual(metrics["val/sudoku_cell_accuracy"], 1.0)
        self.assertEqual(metrics["val/sudoku_parse_rate"], 1.0)

    def test_eval_all_samples_configuration(self) -> None:
        dm = SudokuCoTDataModule(num_samples=6, validation_fraction=0.5, clues=30, batch_size=2, seed=42)
        dm.prepare_data()
        dm.setup("fit")

        model = MockGenerationModel(mode="gibberish", tokenizer=dm.tokenizer)
        validator = SudokuValidator(num_eval_samples="all")
        metrics = validator.validate(model, dm)
        self.assertIn("val/sudoku_board_accuracy", metrics)
        self.assertEqual(metrics["val/sudoku_parse_rate"], 0.0)
