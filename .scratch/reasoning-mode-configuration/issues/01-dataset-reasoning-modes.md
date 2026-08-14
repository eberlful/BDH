# 01 — Dataset Reasoning Modes and Loss Masking

**What to build:** Support `"full"`, `"none"`, and `"context_only"` reasoning modes in reasoning datasets and data modules, generating the corresponding token sequences and loss target masks, with thorough unit test coverage.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] Reasoning data module and dataset accept `reasoning_mode` parameter with validation for `"full"`, `"none"`, and `"context_only"`, defaulting to `"full"`.
- [x] `"full"` mode constructs prompt with thinking prefix, includes CoT trace and solution, and masks only the prompt prefix transitions.
- [x] `"none"` mode constructs prompt with direct solution prefix, omits CoT trace, and masks prompt prefix transitions, calculating loss only on solution tokens.
- [x] `"context_only"` mode constructs full prompt with CoT trace in context, but masks all tokens prior to solution values with `-100`.
- [x] Comprehensive unit tests verify input IDs, target IDs, mask boundaries, and error validation across all three modes.
