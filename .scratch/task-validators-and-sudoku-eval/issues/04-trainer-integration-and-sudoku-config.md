# 04 — Trainer Integration and End-to-End Sudoku Benchmark Configuration

**What to build:** Integrate task validators into the `TorchTrainer` validation cycle, merge validator metrics into the `val/` metric namespace for all active loggers, and update `configs/bdh_sudoku.yaml` to demonstrate end-to-end Sudoku validation and checkpoint saving on board accuracy.

**Blocked by:** 02 — Checkpoint Callback Optimization Direction Support, 03 — Sudoku Task Validator with Generative Evaluation and Constraint Checking.

**Status:** ready-for-agent

- [x] `TorchTrainer` accepts an optional list of `BaseValidator` instances.
- [x] During `TorchTrainer._validate()`, validator hooks and evaluation run, and returned metrics are merged into `epoch_metrics` with `val/` prefix.
- [x] Loggers (`TerminalLogger`, `text_file`, `tensorboard`) display and log validator metrics alongside `val/loss`.
- [x] `configs/bdh_sudoku.yaml` is updated to configure `validator: { name: sudoku, params: { num_eval_samples: 32 } }` and monitor `val/sudoku_board_accuracy` for best checkpoint saving.
- [x] Integration tests verify an end-to-end multi-step training run with `SudokuValidator` and `CheckpointCallback`.

