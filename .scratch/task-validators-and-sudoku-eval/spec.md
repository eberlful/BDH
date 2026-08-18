Status: ready-for-agent

# Spec: Task-Specific Validators and Generative Sudoku Evaluation

## Problem Statement

When training language models and recurrent architectures on reasoning benchmarks (such as Sudoku), cross-entropy loss (`val/loss`) alone does not indicate whether a model is generating structurally valid or semantically correct solutions. A model can achieve decreasing cross-entropy loss while continuing to generate invalid Sudoku grids (e.g. repeating numbers in rows, columns, or $3\times3$ boxes, or hallucinating invalid syntax).

Furthermore, different tasks in the training framework require task-specific evaluation semantics (e.g. board accuracy and rule validity for Sudoku vs. exact match or perplexity for other tasks). The training framework lacks a pluggable, first-class validation mechanism to compute domain-specific metrics and perform generative evaluation during training.

## Solution

Introduce a modular **Task Validator** subsystem to the BDH training framework that allows registering and configuring domain-specific validators via YAML. Implement a comprehensive **`SudokuValidator`** that performs both teacher-forced evaluation and generative evaluation to evaluate rule satisfaction, exact solution match, cell accuracy, and parse success.

Key capabilities include:
- A `BaseValidator` lifecycle interface and `VALIDATOR_REGISTRY` in the core architecture.
- Declarative configuration in YAML via `validator:` (single component) or `validators:` (list of components).
- Seamless merging of validator metrics into the validation dictionary (`val/` prefix) so loggers (`TerminalLogger`, `text_file`, `tensorboard`) display them.
- Enhanced `CheckpointCallback` supporting `mode: "auto" | "min" | "max"` to monitor task metrics (such as `val/sudoku_board_accuracy`) for `save_best` checkpointing.
- Robust solution grid extraction across both direct (`reasoning_mode: none`) and Chain-of-Thought (`reasoning_mode: full`) reasoning modes and data module formats.

## User Stories

1. As an ML researcher, I want to configure `validator: { name: sudoku }` in my training YAML, so that my Sudoku training runs automatically evaluate rule validity and board accuracy at validation time.
2. As an ML researcher, I want the `SudokuValidator` to compute `val/sudoku_board_accuracy`, so that I know the percentage of validation puzzles where the model generates the exact ground-truth solution grid.
3. As an ML researcher, I want the `SudokuValidator` to compute `val/sudoku_validity_rate`, so that I can track whether the model understands Sudoku rules (no duplicate digits in rows, columns, or $3\times3$ boxes) even when it predicts an alternative valid solution.
4. As an ML researcher, I want the `SudokuValidator` to compute `val/sudoku_cell_accuracy`, so that I can observe incremental learning progress on individual cell predictions before full boards are solved.
5. As an ML researcher, I want the `SudokuValidator` to compute `val/sudoku_parse_rate`, so that I can distinguish syntax or formatting failures from logical constraint violations.
6. As a researcher working with Chain-of-Thought reasoning (`reasoning_mode: full`), I want the Sudoku validator to reliably parse the final solution grid from after the `Solution:\n` marker, ignoring intermediate reasoning steps.
7. As a researcher working with direct inference (`reasoning_mode: none`), I want the Sudoku validator to parse direct prompt-to-solution generations without requiring a thinking trace.
8. As an ML engineer, I want generative evaluation to use deterministic greedy decoding by default (`temperature=1.0, top_k=1`), so that validation metrics reflect true model capability without sampling noise.
9. As an ML engineer on memory/compute-constrained devices, I want `num_eval_samples` to be configurable (e.g. `num_eval_samples: 32`), so that generative evaluation does not excessively slow down validation epochs.
10. As an ML engineer, I want the option to set `num_eval_samples: -1` or `all` to run generative evaluation over the entire validation dataset when full benchmark evaluation is desired.
11. As a practitioner configuring checkpoints, I want to set `callbacks: [ { name: checkpoint, params: { monitor: "val/sudoku_board_accuracy", save_best: true } } ]` and have the system automatically understand that higher is better (`mode: auto`), so that the best checkpoint is saved based on board accuracy.
12. As a practitioner, I want to specify multiple validators via `validators: [ ... ]` in YAML when evaluating composite or multi-aspect benchmarks.
13. As a developer writing new domain tasks, I want to register a custom validator with `@VALIDATOR_REGISTRY.register("my_task")` implementing `BaseValidator`, so that I can add new task benchmarks without modifying the core training loop.
14. As a user with existing YAML configuration files that do not specify a `validator`, I want training to proceed with standard loss-only validation without breaking backward compatibility or throwing errors.
15. As an ML engineer passing CLI overrides, I want to pass `--set validator.params.num_eval_samples=64` or `--set callbacks.0.params.monitor=val/sudoku_board_accuracy`, so that I can adjust validation behavior on the fly.

## Implementation Decisions

- **Base Validator Interface**:
  - `BaseValidator` is added to `src/core/base.py`.
  - Exposes lifecycle methods: `on_validation_start(trainer)`, `on_validation_batch(trainer, batch, batch_idx, output)`, `on_validation_epoch_end(trainer) -> Mapping[str, float]`, and `validate(model, data_module, trainer) -> dict[str, float]`.
  - Subclasses can either implement the batch/epoch hooks or override `validate` directly for complete control over validation data loops.
- **Component Registry**:
  - `VALIDATOR_REGISTRY` is created in `src/core/registry.py`.
  - `load_builtin_components()` is updated to import `src.validation`.
- **Sudoku Validator**:
  - Implemented in `src/validation/sudoku.py` and exported in `src/validation/__init__.py`.
  - Registered as `sudoku` in `VALIDATOR_REGISTRY`.
  - Supports configuration parameters: `num_eval_samples` (default: 64), `max_new_tokens` (optional int), `greedy` (default: true).
  - Emits metrics with standard prefixes: `val/sudoku_board_accuracy`, `val/sudoku_validity_rate`, `val/sudoku_cell_accuracy`, and `val/sudoku_parse_rate`.
  - Contains robust parsing logic supporting 81-digit raw token arrays, 9x9 formatted ASCII grids, and GPT-2 tokenized strings.
- **Configuration & Runtime Assembly**:
  - `src/core/config.py` validates `validator` (optional string or mapping) and `validators` (optional list of strings or mappings).
  - `src/runtime.py` instantiates configured validators via `VALIDATOR_REGISTRY` and passes them to `TRAINER_REGISTRY.instantiate`.
- **Trainer Execution Loop**:
  - `TorchTrainer` accepts an optional `validators: list[BaseValidator]`.
  - In `TorchTrainer._validate()`, iterates through registered validators, invokes evaluation, and merges returned metrics into the epoch validation dictionary.
- **Checkpoint Callback Mode**:
  - `CheckpointCallback` adds a `mode: str = "auto"` parameter supporting `"auto"`, `"min"`, and `"max"`.
  - In `"auto"` mode, automatically infers `"min"` if `monitor` contains `loss`, `err`, `error`, `perplexity`, and `"max"` if `monitor` contains `acc`, `accuracy`, `rate`, `score`, `reward`, `validity`.
- **Glossary & ADR Alignment**:
  - Adheres to terms in `CONTEXT.md` (**Task Validator**, **Generative Evaluation**, **Teacher-Forced Validation**).
  - Complies with `docs/adr/0004-task-validators-and-generative-evaluation.md`.

## Testing Decisions

- **Testing Seams**:
  1. *Validator Unit Seam* (`src/validation/sudoku.py`): Test `SudokuValidator` directly with known dummy models / mock completions (perfect solutions, partially correct boards, invalid Sudoku rule violations, and unparseable gibberish) to assert exact metric values.
  2. *Checkpoint Callback Seam* (`src/callbacks/checkpoint.py`): Test `CheckpointCallback` with `mode="auto"`, `mode="max"`, and `mode="min"` to verify proper `best_metric` updating and checkpoint saving.
  3. *Configuration & Registry Seam* (`src/core/registry.py`, `src/core/config.py`): Test registration, YAML schema validation, and component instantiation.
  4. *Trainer Integration Seam* (`src/training/trainer.py`): Test multi-step training execution with an active `SudokuValidator` verifying that metrics are merged and logged during validation.
- **Prior Art**: Follow the patterns in `tests/test_sudoku_cot.py`, `tests/test_training_environment.py`, and `tests/test_verbose_logging.py`.

## Out of Scope

- Non-autoregressive specialized constraint solvers (e.g. SAT/ILP solvers during training forward passes).
- Custom GUI / visual board renderers in terminal loggers (covered by textual metric reporting).
- Online reinforcement learning / RLHF policy gradient rewards based on validator outputs.

## Further Notes

- Setting `num_eval_samples: 32` or `64` ensures generative validation remains fast on Mac/Apple Silicon without impacting per-epoch training throughput.
