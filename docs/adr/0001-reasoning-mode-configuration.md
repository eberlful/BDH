# 0001. Reasoning Mode Configuration for Training Datasets

## Context

Reasoning benchmarks like Sudoku can be trained either directly (mapping problem input to final solution), with explicit step-by-step Chain-of-Thought (CoT) traces, or conditioned on CoT reasoning traces without calculating loss on the reasoning tokens. Previously, `sudoku_cot` only supported fixed full CoT supervision.

## Decision

We introduce a canonical `reasoning_mode` parameter in data module configurations (`data.params.reasoning_mode`) with three supported strategies:
- `"full"` (default): Formats the sequence with the CoT trace and supervises both the intermediate CoT steps and the final solution.
- `"none"`: Omit the CoT trace entirely, training directly from the prompt (`Sudoku:\n<grid>\n\nSolution:\n`) to the target solution grid.
- `"context_only"`: Formats the sequence with the CoT trace in context, but masks the prompt and CoT trace tokens from training loss (`target_ids = -100`), supervising only the final solution tokens.

CLI generation (`main.py generate`) detects `reasoning_mode` from the run's saved configuration to format the appropriate prompt prefix automatically.

## Consequences

- Data modules can toggle between direct and CoT reasoning supervision without code duplication.
- Backwards compatibility is preserved by defaulting `reasoning_mode` to `"full"`.
- Loss computation and evaluation cleanly isolate reasoning trace performance versus final answer prediction.
