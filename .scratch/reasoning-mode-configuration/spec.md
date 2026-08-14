Status: ready-for-agent

# Spec: Configurable Reasoning Mode for Training Datasets

## Problem Statement

Researchers and developers training language models on reasoning tasks (such as Sudoku) need to compare direct input-to-solution generation with intermediate step-by-step reasoning. Currently, reasoning data modules like Sudoku Chain-of-Thought (CoT) only support fixed sequence formatting with full supervision across both reasoning steps and final solutions. There is no configurable mechanism to train models directly on solutions (omitting intermediate CoT traces) or to condition models on reasoning traces without backpropagating loss onto the reasoning tokens.

## Solution

Introduce a standardized `reasoning_mode` configuration parameter across reasoning data modules with three explicit operating modes:
- `"full"`: Includes the CoT trace in the sequence and trains on both the intermediate reasoning steps and the final solution.
- `"none"`: Omits the CoT trace completely, creating direct prompt-to-solution sequences with loss computed solely on the solution tokens.
- `"context_only"`: Formats the sequence with the CoT trace in context, but masks both the prompt and reasoning steps from loss computation (`target_ids = -100`), supervising only the final solution tokens.

Additionally, align inference generation (`generate` command) to detect the configured `reasoning_mode` from the run's saved configuration and prompt the model with the matching prefix (`Thinking:\n` vs `Solution:\n`).

## User Stories

1. As an ML researcher, I want to configure `reasoning_mode: "none"` in my training YAML, so that I can train a baseline model to predict the final solution directly without generating intermediate reasoning steps.
2. As an ML researcher, I want to configure `reasoning_mode: "full"` in my training YAML, so that my model learns to generate step-by-step reasoning traces followed by the final solution.
3. As an ML researcher, I want to configure `reasoning_mode: "context_only"` in my training YAML, so that I can evaluate whether conditioning on pre-generated reasoning traces improves final solution prediction without penalizing reasoning trace generation.
4. As an ML engineer, I want the system to default to `reasoning_mode: "full"` when the parameter is omitted, so that existing configuration files remain completely backward-compatible.
5. As an ML engineer, I want invalid values for `reasoning_mode` (such as `"invalid"` or numerical inputs) to raise clear, descriptive validation errors at dataset initialization time, so that misconfigurations are caught immediately before training begins.
6. As a practitioner using the CLI `generate` command, I want the tool to automatically inspect the model's training configuration and use the correct prompt prefix (`Thinking:\n` for `"full"` / `"context_only"` or `Solution:\n` for `"none"`), so that I don't have to manually format prompt strings for different checkpoint types.
7. As a practitioner, I want to pass command-line overrides such as `--set data.params.reasoning_mode=none` to `main.py train`, so that I can easily run ablation studies across different reasoning modes without creating duplicate YAML files.
8. As a developer writing tests, I want deterministic validation that each reasoning mode produces the expected input tokens, target tokens, and loss masks, so that data pipeline regressions are caught early.

## Implementation Decisions

- **Reasoning Mode Parameter**: The configuration parameter is named `reasoning_mode` and resides within `data.params` of the reasoning data modules.
- **Allowed Values**: The parameter accepts three string values: `"full"`, `"none"`, and `"context_only"`. Any other value will raise a `ValueError`.
- **Default Strategy**: If omitted, `reasoning_mode` defaults to `"full"`.
- **Prompt and Sequence Formatting**:
  - In `"full"` mode: Sequence format is `Sudoku:\n<grid>\n\nThinking:\n<cot_steps>\n\nSolution:\n<solution_grid>\n<|endoftext|>`. Loss is masked on `Sudoku:\n<grid>\n\nThinking:\n` prefix transitions and computed across `<cot_steps>` and `<solution_grid>`.
  - In `"none"` mode: Sequence format is `Sudoku:\n<grid>\n\nSolution:\n<solution_grid>\n<|endoftext|>`. Loss is masked on the `Sudoku:\n<grid>\n\nSolution:\n` prefix and computed across `<solution_grid>`.
  - In `"context_only"` mode: Sequence format is identical to `"full"`, but the loss mask covers the prompt AND the entire thinking section up to `\n\nSolution:\n`, computing loss strictly on `<solution_grid>`.
- **Inference CLI Integration**: The CLI generation function inspects the loaded run configuration for `reasoning_mode`. If `"none"`, it automatically wraps raw grid inputs with the direct `Solution:\n` prompt; otherwise, it formats with `Thinking:\n`.
- **Glossary & ADR Alignment**: Uses domain terms defined in `CONTEXT.md` (**Reasoning Mode**, **CoT Trace**, **Prompt**, **Solution**) and complies with `docs/adr/0001-reasoning-mode-configuration.md`.

## Testing Decisions

- **Seams**:
  - Primary testing seam: Data module and Dataset layer (`SudokuCoTDataset` and `SudokuCoTDataModule`). Verify token sequences, target IDs, prompt masking lengths, and dictionary outputs across all three modes.
  - CLI integration seam: Generation helper prompt builder and configuration validation.
  - End-to-end training seam: Small multi-step training execution verifying loss calculation and checkpoint generation under each mode.
- **Test Criteria**:
  - Target masks must contain `-100` strictly for un-supervised tokens and valid token IDs for supervised tokens.
  - Context length bounds must be respected across all modes.
  - Data module initialization with invalid reasoning modes must fail with a descriptive `ValueError`.
- **Prior Art**: Follow the patterns in `tests/test_sudoku_cot.py` and `tests/test_training_environment.py`.

## Out of Scope

- Architectural latent reasoning recurrence inside the neural network model layers (e.g. BDH-CQ recurrent loops).
- Dynamic runtime switching of reasoning mode during active training steps within a single epoch.
- Tokenizers other than the configured dataset tokenizers (e.g. GPT-2 BPE for `sudoku_cot`).

## Further Notes

- In `"none"` mode, sequence lengths are significantly shorter (~100 tokens vs ~600 tokens for `"full"` mode), which reduces memory usage and training time.
