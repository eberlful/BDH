#!/usr/bin/env python3
"""Generate a random Sudoku start board for model inference.

Supports standard digit strings, discrete token sequences with special tokens
(SEPARATOR_TOKEN=91, END_TOKEN=92), and Chain-of-Thought (CoT) text prompts.

Usage examples:
    # 1. Generate an 81-digit Sudoku string (medium difficulty)
    uv run python scripts/generate_random_start_sudoku.py

    # 2. Generate token IDs with special separator token (91) for causal inference
    uv run python scripts/generate_random_start_sudoku.py --tokens

    # 3. Generate Chain-of-Thought (CoT) prompt for text-based inference
    uv run python scripts/generate_random_start_sudoku.py --cot

    # 4. Generate with high difficulty and visual grid display
    uv run python scripts/generate_random_start_sudoku.py --difficulty high --grid --solution

    # 5. Use directly in inference with bdh CLI
    uv run python -m src.cli generate runs/<run_dir> $(uv run python scripts/generate_random_start_sudoku.py --difficulty medium) --max-tokens 256
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data import (
    SUDOKU_CELL_COUNT,
    SUDOKU_END_TOKEN,
    SUDOKU_POSITION_OFFSET,
    SUDOKU_SEPARATOR_TOKEN,
    SUDOKU_SIZE,
    _generate_solved_board,
    _serialize_sudoku,
    encode_sudoku_prompt,
    is_valid_sudoku_board,
)
from src.data.sudoku_cot import (
    build_sudoku_cot_full_text,
    build_sudoku_cot_prompt,
    format_sudoku_grid,
)

DIFFICULTY_CLUES = {
    "low": 40,
    "medium": 30,
    "high": 22,
}


def generate_sudoku(
    difficulty: str = "medium",
    clues: int | None = None,
    seed: int | None = None,
) -> tuple[str, str]:
    """Generate a random Sudoku puzzle and its solution.

    Args:
        difficulty: Difficulty level preset ('low', 'medium', 'high').
        clues: Optional explicit number of clues (1 to 81). Overrides difficulty.
        seed: Optional random seed for reproducible generation.

    Returns:
        A tuple of (puzzle_string, solution_string), each containing 81 digit characters.
    """
    if clues is None:
        if difficulty not in DIFFICULTY_CLUES:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}. Choose from {list(DIFFICULTY_CLUES.keys())}."
            )
        target_clues = DIFFICULTY_CLUES[difficulty]
    else:
        if not 0 <= clues <= SUDOKU_CELL_COUNT:
            raise ValueError(f"Clues must be between 0 and {SUDOKU_CELL_COUNT}, got {clues}.")
        target_clues = clues

    rng = random.Random(seed)
    solution = _generate_solved_board(rng)
    if not is_valid_sudoku_board(solution):
        raise RuntimeError("Generated solution is not a valid Sudoku board.")

    blank_count = SUDOKU_CELL_COUNT - target_clues
    blank_positions = rng.sample(range(SUDOKU_CELL_COUNT), blank_count)

    puzzle_list = list(solution)
    for pos in blank_positions:
        puzzle_list[pos] = 0

    puzzle_str = "".join(str(d) for d in puzzle_list)
    solution_str = "".join(str(d) for d in solution)
    return puzzle_str, solution_str


def get_prompt_tokens(puzzle_str: str) -> list[int]:
    """Encode an 81-character puzzle string to token IDs with SUDOKU_SEPARATOR_TOKEN (91)."""
    return encode_sudoku_prompt(puzzle_str)


def get_serialized_solution_tokens(puzzle_str: str, solution_str: str) -> list[int]:
    """Get the full serialized token sequence with position tokens and SUDOKU_END_TOKEN (92)."""
    puzzle_tuple = tuple(int(c) for c in puzzle_str)
    solution_tuple = tuple(int(c) for c in solution_str)
    return list(_serialize_sudoku(puzzle_tuple, solution_tuple))


def format_pretty_grid(board_str: str) -> str:
    """Format an 81-character Sudoku string into an ASCII 9x9 box grid."""
    if len(board_str) != SUDOKU_CELL_COUNT:
        raise ValueError(f"Expected {SUDOKU_CELL_COUNT} digits, got {len(board_str)}.")

    lines: list[str] = []
    divider = "+-------+-------+-------+"
    for row in range(SUDOKU_SIZE):
        if row % 3 == 0:
            lines.append(divider)
        row_cells = []
        for col in range(SUDOKU_SIZE):
            idx = row * SUDOKU_SIZE + col
            val = board_str[idx]
            display_char = "." if val == "0" else val
            row_cells.append(display_char)
        line = (
            f"| {' '.join(row_cells[0:3])} "
            f"| {' '.join(row_cells[3:6])} "
            f"| {' '.join(row_cells[6:9])} |"
        )
        lines.append(line)
    lines.append(divider)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a random Sudoku start string or token sequence with special tokens for model inference."
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        choices=list(DIFFICULTY_CLUES.keys()),
        default="medium",
        help="Sudoku difficulty level preset (low: 40 clues, medium: 30 clues, high: 22 clues; default: medium).",
    )
    parser.add_argument(
        "-c",
        "--clues",
        type=int,
        default=None,
        help="Explicit number of given clues (0-81). Overrides --difficulty.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of Sudoku puzzles to generate (default: 1).",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        "--special-tokens",
        action="store_true",
        dest="tokens",
        help=(
            f"Output discrete integer token IDs with special separator token "
            f"({SUDOKU_SEPARATOR_TOKEN}) and end token ({SUDOKU_END_TOKEN})."
        ),
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Output formatted Chain-of-Thought (CoT) text prompt (Sudoku:\n...\n\nThinking:\n).",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["full", "none", "context_only"],
        default="full",
        help="Reasoning mode for --cot output (default: full).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Print visual 9x9 ASCII grid representation.",
    )
    parser.add_argument(
        "--solution",
        action="store_true",
        help="Also output the corresponding solved Sudoku string/tokens/grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_seed = args.seed
    for i in range(args.count):
        seed = None if base_seed is None else base_seed + i
        puzzle_str, solution_str = generate_sudoku(
            difficulty=args.difficulty,
            clues=args.clues,
            seed=seed,
        )
        puzzle_digits = [int(c) for c in puzzle_str]
        solution_digits = [int(c) for c in solution_str]
        prompt_tokens = get_prompt_tokens(puzzle_str)
        serialized_tokens = get_serialized_solution_tokens(puzzle_str, solution_str)

        if args.cot:
            if args.solution:
                _, full_cot_text = build_sudoku_cot_full_text(
                    puzzle_digits,
                    solution_digits,
                    reasoning_mode=args.reasoning_mode,
                )
                print(full_cot_text)
            else:
                cot_prompt = build_sudoku_cot_prompt(
                    puzzle_digits,
                    reasoning_mode=args.reasoning_mode,
                )
                print(cot_prompt, end="")
        elif args.tokens:
            prompt_tokens_str = " ".join(str(t) for t in prompt_tokens)
            if args.solution:
                full_tokens_str = " ".join(str(t) for t in serialized_tokens)
                print(f"Prompt tokens (with separator {SUDOKU_SEPARATOR_TOKEN}):\n{prompt_tokens_str}")
                print(f"Full tokens (with separator {SUDOKU_SEPARATOR_TOKEN} and end token {SUDOKU_END_TOKEN}):\n{full_tokens_str}")
            else:
                print(prompt_tokens_str)
        elif args.grid:
            clue_count = sum(1 for c in puzzle_str if c != "0")
            print(f"--- Sudoku Puzzle ({clue_count} clues, difficulty={args.difficulty}) ---")
            print(f"Digits: {puzzle_str}")
            print(f"Tokens (with separator {SUDOKU_SEPARATOR_TOKEN}): {' '.join(str(t) for t in prompt_tokens)}")
            print(format_pretty_grid(puzzle_str))
            if args.solution:
                print("\n--- Solution ---")
                print(f"Digits: {solution_str}")
                print(f"Full tokens (ending with {SUDOKU_END_TOKEN}): {' '.join(str(t) for t in serialized_tokens)}")
                print(format_pretty_grid(solution_str))
            if i < args.count - 1:
                print()
        else:
            if args.solution:
                print(f"{puzzle_str} {solution_str}")
            else:
                print(puzzle_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
