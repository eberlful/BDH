"""Unit tests for randomized Sudoku clues and difficulty ranges."""

from __future__ import annotations

import unittest
from src.data.data import (
    SudokuDataModule,
    _generate_sudoku_examples,
    _sample_clue_count,
    _validate_clues_spec,
)
from src.data.sudoku_cot import (
    SudokuCoTDataModule,
    _generate_sudoku_cot_raw_samples,
)


class TestSudokuRandomClues(unittest.TestCase):
    def test_sample_clue_count(self) -> None:
        import random
        rng = random.Random(42)

        # Single int
        self.assertEqual(_sample_clue_count(30, rng), 30)

        # Range [min, max]
        for _ in range(50):
            val = _sample_clue_count([20, 45], rng)
            self.assertTrue(20 <= val <= 45)

        # Discrete choices [22, 30, 40]
        for _ in range(50):
            val = _sample_clue_count([22, 30, 40], rng)
            self.assertIn(val, [22, 30, 40])

    def test_validate_clues_spec(self) -> None:
        _validate_clues_spec(30)
        _validate_clues_spec([20, 45])
        _validate_clues_spec((20, 45))
        _validate_clues_spec([22, 30, 40])

        with self.assertRaises(ValueError):
            _validate_clues_spec(-1)
        with self.assertRaises(ValueError):
            _validate_clues_spec(82)
        with self.assertRaises(ValueError):
            _validate_clues_spec([50, 20])  # min > max
        with self.assertRaises(ValueError):
            _validate_clues_spec([])
        with self.assertRaises(ValueError):
            _validate_clues_spec("invalid")  # type: ignore

    def test_sudoku_cot_random_clues_generation(self) -> None:
        samples = _generate_sudoku_cot_raw_samples(
            count=20,
            clues=[20, 45],
            seed=123,
        )
        self.assertEqual(len(samples), 20)
        clue_counts = [sum(1 for cell in s["puzzle"] if cell != 0) for s in samples]
        for count in clue_counts:
            self.assertTrue(20 <= count <= 45)
        # Verify diversity in clue counts
        self.assertGreater(len(set(clue_counts)), 1)

    def test_sudoku_cot_datamodule_with_clues_range_and_val_clues(self) -> None:
        dm = SudokuCoTDataModule(
            num_samples=20,
            validation_fraction=0.5,
            clues=[20, 40],
            val_clues=30,
            reasoning_mode="none",
            tokenizer="byte",
            context_length=512,
            batch_size=4,
            seed=42,
        )
        dm.setup()
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNotNone(dm.val_dataset)
        self.assertEqual(len(dm.train_dataset), 10)
        self.assertEqual(len(dm.val_dataset), 10)

        # Val samples should all have exactly 30 clues
        for s in dm.val_dataset.samples:
            clues = sum(1 for c in s["puzzle"] if c != 0)
            self.assertEqual(clues, 30)

        # Train samples should have varied clues in [20, 40]
        train_clues = [sum(1 for c in s["puzzle"] if c != 0) for s in dm.train_dataset.samples]
        for c in train_clues:
            self.assertTrue(20 <= c <= 40)
        self.assertGreater(len(set(train_clues)), 1)

    def test_sudoku_base_datamodule_with_clues_range(self) -> None:
        dm = SudokuDataModule(
            num_samples=20,
            validation_fraction=0.5,
            min_clues=25,
            max_clues=35,
            context_length=256,
            batch_size=4,
            seed=42,
        )
        dm.setup()
        self.assertIsNotNone(dm.train_dataset)
        self.assertEqual(len(dm.train_dataset), 10)


if __name__ == "__main__":
    unittest.main()
