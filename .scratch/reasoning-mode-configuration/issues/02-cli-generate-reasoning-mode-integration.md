# 02 — CLI Generation Reasoning Mode Integration

**What to build:** Enable the inference generation CLI (`main.py generate`) to automatically detect the `reasoning_mode` from the run's saved configuration and construct the matching prompt prefix.

**Blocked by:** 01 — Dataset Reasoning Modes and Loss Masking

**Status:** resolved

- [x] Generation handler inspects `reasoning_mode` in the loaded run configuration.
- [x] Raw input grids under `"none"` mode are formatted with the direct `Solution:\n` prompt prefix.
- [x] Raw input grids under `"full"` and `"context_only"` modes are formatted with the `Thinking:\n` prompt prefix.
- [x] Automated tests verify CLI generation behavior and prompt prefix construction for different run configurations.
