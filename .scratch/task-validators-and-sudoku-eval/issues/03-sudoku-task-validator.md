# 03 — Sudoku Task Validator with Generative Evaluation and Constraint Checking

**What to build:** Implement `SudokuValidator` registered in `VALIDATOR_REGISTRY` to perform generative and teacher-forced evaluation of Sudoku solutions, computing board accuracy, constraint validity, cell accuracy, and parse rate across direct and Chain-of-Thought reasoning modes.

**Blocked by:** 01 — Core Validator Extension Point, Registry, and Configuration Schema.

**Status:** ready-for-agent

- [x] `SudokuValidator` is implemented in `src/validation/sudoku.py` and registered as `"sudoku"` in `VALIDATOR_REGISTRY`.
- [x] Supports configurable `num_eval_samples: int = 64` and deterministic greedy decoding (`temperature=1.0, top_k=1`).
- [x] Computes and returns scalar metrics: `val/sudoku_board_accuracy` (exact match), `val/sudoku_validity_rate` (Sudoku rules: unique 1-9 per row/col/box), `val/sudoku_cell_accuracy`, and `val/sudoku_parse_rate`.
- [x] Extracts solution grids reliably from both direct prompt completions (`reasoning_mode: none`) and Chain-of-Thought traces (`reasoning_mode: full`).
- [x] Supports both token-level data formats and GPT-2 tokenized strings.
- [x] Unit tests verify evaluation against synthetic models producing perfect, partially correct, constraint-violating, and unparseable outputs.

