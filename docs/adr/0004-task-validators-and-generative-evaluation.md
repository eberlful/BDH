# 0004. Task Validators and Generative Evaluation

## Context

The training framework previously relied exclusively on cross-entropy validation loss (`val/loss`) to monitor training progress and select the best checkpoint. For complex reasoning and structured generation tasks like Sudoku, low cross-entropy loss does not guarantee syntactic validity or correct constraint satisfaction (e.g. valid Sudoku rows/columns/boxes without duplicate digits).

Furthermore, different tasks require domain-specific validation semantics (e.g., Sudoku board validity and exact grid match vs. code execution or language model perplexity), and evaluation may require autoregressively decoding completions from prompt inputs rather than just teacher-forced next-token scoring.

## Decision

We introduce a first-class **Task Validator** subsystem:

1. **`BaseValidator` Contract and Registry**:
   - Add `BaseValidator` in `src/core/base.py` providing lifecycle hooks (`on_validation_start`, `on_validation_batch`, `on_validation_epoch_end`, and an extensible `validate` entry point).
   - Register validators in a new `VALIDATOR_REGISTRY` in `src/core/registry.py`.

2. **Dual Validation Mechanism**:
   - Compute teacher-forced batch metrics efficiently across validation batches.
   - Run **Generative Evaluation** by autoregressively sampling completions from prompt inputs on a configurable subset of validation examples (`num_eval_samples`, default 64) with deterministic greedy decoding (`temperature=1.0, top_k=1`).

3. **Domain-Specific `SudokuValidator`**:
   - Implement `SudokuValidator` in `src/validation/` to compute four key metrics:
     - `val/sudoku_board_accuracy`: Fraction of generated boards matching the ground-truth solution.
     - `val/sudoku_validity_rate`: Fraction of generated boards that satisfy all Sudoku row, column, and $3\times3$ box uniqueness constraints.
     - `val/sudoku_cell_accuracy`: Fraction of individual cells correctly predicted.
     - `val/sudoku_parse_rate`: Fraction of model outputs that successfully parse into an 81-digit grid.
   - Support both direct (`reasoning_mode: none`) and Chain-of-Thought (`reasoning_mode: full`) sequence representations, extracting the solution grid reliably.

4. **YAML Configuration and Checkpointing**:
   - Support `validator:` (single specification) and `validators:` (list) in YAML training configs.
   - Merge all validator metrics into the `val/` namespace.
   - Enhance `CheckpointCallback` with `mode: "auto" | "min" | "max"` to automatically detect maximization metrics like `val/sudoku_board_accuracy` or minimization metrics like `val/loss`.

## Consequences

- Different tasks can define arbitrary domain evaluation metrics and plug into any training run declaratively via YAML configuration.
- Checkpointing can save best models based on actual task performance rather than just cross-entropy loss.
- Generative evaluation on a sample subset bounds validation latency while providing true end-to-end task metrics.
- Existing loss-only configurations continue to work without modification.
