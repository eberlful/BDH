"""Tests for random Sudoku puzzle generator script."""

from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import patch

from scripts.generate_random_start_sudoku import (
    DIFFICULTY_CLUES,
    format_pretty_grid,
    generate_sudoku,
    get_prompt_tokens,
    get_serialized_solution_tokens,
    main,
)
from src.data.data import (
    SUDOKU_END_TOKEN,
    SUDOKU_SEPARATOR_TOKEN,
    encode_sudoku_prompt,
    is_valid_sudoku_board,
)


class GenerateRandomStartSudokuTests(unittest.TestCase):
    def test_difficulty_clue_counts(self) -> None:
        for difficulty, expected_clues in DIFFICULTY_CLUES.items():
            puzzle, solution = generate_sudoku(difficulty=difficulty, seed=42)
            self.assertEqual(len(puzzle), 81)
            self.assertEqual(len(solution), 81)
            clue_count = sum(1 for c in puzzle if c != "0")
            self.assertEqual(clue_count, expected_clues)
            self.assertTrue(is_valid_sudoku_board([int(c) for c in solution]))
            # Verify it encodes without errors for causal inference
            encoded = encode_sudoku_prompt(puzzle)
            self.assertEqual(len(encoded), 82)
            self.assertEqual(encoded[-1], SUDOKU_SEPARATOR_TOKEN)

    def test_explicit_clues_override(self) -> None:
        puzzle, solution = generate_sudoku(clues=25, seed=123)
        self.assertEqual(sum(1 for c in puzzle if c != "0"), 25)
        self.assertTrue(is_valid_sudoku_board([int(c) for c in solution]))

    def test_invalid_difficulty_and_clues(self) -> None:
        with self.assertRaises(ValueError):
            generate_sudoku(difficulty="extreme")
        with self.assertRaises(ValueError):
            generate_sudoku(clues=90)
        with self.assertRaises(ValueError):
            generate_sudoku(clues=-1)

    def test_format_pretty_grid(self) -> None:
        puzzle, _ = generate_sudoku(difficulty="low", seed=42)
        grid = format_pretty_grid(puzzle)
        lines = grid.strip().split("\n")
        self.assertEqual(len(lines), 13)
        self.assertEqual(lines[0], "+-------+-------+-------+")

    def test_special_tokens_helpers(self) -> None:
        puzzle, solution = generate_sudoku(difficulty="medium", seed=42)
        tokens = get_prompt_tokens(puzzle)
        self.assertEqual(len(tokens), 82)
        self.assertEqual(tokens[-1], SUDOKU_SEPARATOR_TOKEN)

        full_tokens = get_serialized_solution_tokens(puzzle, solution)
        self.assertEqual(full_tokens[81], SUDOKU_SEPARATOR_TOKEN)
        self.assertEqual(full_tokens[-1], SUDOKU_END_TOKEN)

    def test_main_cli_execution_digits(self) -> None:
        stdout_capture = StringIO()
        with patch("sys.stdout", stdout_capture):
            exit_code = main(["--difficulty", "high", "--seed", "99"])
        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue().strip()
        self.assertEqual(len(output), 81)
        self.assertTrue(output.isdigit())

    def test_main_cli_execution_tokens(self) -> None:
        stdout_capture = StringIO()
        with patch("sys.stdout", stdout_capture):
            exit_code = main(["--difficulty", "medium", "--seed", "42", "--tokens"])
        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue().strip()
        tokens = [int(t) for t in output.split()]
        self.assertEqual(len(tokens), 82)
        self.assertEqual(tokens[-1], SUDOKU_SEPARATOR_TOKEN)

    def test_main_cli_execution_cot(self) -> None:
        stdout_capture = StringIO()
        with patch("sys.stdout", stdout_capture):
            exit_code = main(["--difficulty", "low", "--seed", "42", "--cot"])
        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue().strip()
        self.assertTrue(output.startswith("Sudoku:\n"))
        self.assertTrue(output.endswith("\nThinking:"))


if __name__ == "__main__":
    unittest.main()
