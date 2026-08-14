# BDH Domain Context

Language model training and evaluation framework for Dragon Hatchling (BDH) architectures and reasoning benchmarks.

## Language

**Reasoning Mode**:
The strategy used by a data module to format token sequences and apply loss masking across intermediate reasoning steps and final solutions.
_Avoid_: Thought toggle, CoT flag

**CoT Trace**:
The explicit sequence of intermediate step-by-step reasoning tokens that bridges a problem prompt and its final solution.
_Avoid_: Thinking path, scratchpad tokens

**Prompt**:
The problem representation presented to the model at inference and masked out during loss calculation in training.
_Avoid_: Input grid, problem prefix

**Solution**:
The final ground-truth target output produced by the model.
_Avoid_: Completion, answer grid
