# 01 — Core Validator Extension Point, Registry, and Configuration Schema

**What to build:** Provide the fundamental architecture for task validators across the training framework, including the `BaseValidator` interface, the `VALIDATOR_REGISTRY` component registry, YAML configuration schema validation for `validator` and `validators` fields, and runtime component assembly.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `BaseValidator` is defined in `src/core/base.py` with hooks for validation lifecycle (`on_validation_start`, `on_validation_batch`, `on_validation_epoch_end`, `validate`).
- [x] `VALIDATOR_REGISTRY` is defined in `src/core/registry.py` and included in `load_builtin_components()`.
- [x] YAML configuration validation in `src/core/config.py` allows optional `validator` (mapping or string) and `validators` (list) without schema errors.
- [x] `src/runtime.py` instantiates configured validators from `VALIDATOR_REGISTRY` and passes them to trainer instantiation.
- [x] Existing configuration files without validator definitions continue to load and validate without errors.

