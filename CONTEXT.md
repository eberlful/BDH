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

**Contextual Memory**:
The accumulated associative recurrent state (fast-weights) constructed by sequentially ingesting task demonstrations into the model without an explicit key-value cache.
_Avoid_: KV cache, demonstration buffer

**Latent Workspace**:
The continuous, high-dimensional intermediate representation iteratively refined over recurrent reasoning passes to solve a query without discrete token verbalization.
_Avoid_: Scratchpad, hidden thoughts

**Latent Reasoning Steps**:
The number of recurrent depth iterations executed over the latent workspace for a query.
_Avoid_: Thinking depth, recurrent unroll count

**Deep Supervision**:
The training regime of computing loss over intermediate latent reasoning steps in addition to the final step to stabilize gradient flow and trajectory convergence.
_Avoid_: Layer-wise loss, intermediate penalty

**Task Validator**:
A domain-specific evaluation component executed during the validation phase to compute task-level metrics (e.g. accuracy, syntactic validity, constraint satisfaction) beyond raw token cross-entropy loss.
_Avoid_: Test runner, score checker, metric callback

**Generative Evaluation**:
The process of autoregressively decoding completions from task prompts during validation to measure end-to-end task success and structural correctness.
_Avoid_: Rollout test, generation scoring, free-form testing

**Teacher-Forced Validation**:
Evaluating next-token predictions and cross-entropy loss over ground-truth target token sequences without autoregressive generation.
_Avoid_: Standard eval, non-generative test


