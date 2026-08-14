# 01 — Baseline BDH-CQ Architecture and Model Registration

**What to build:** A registered model architecture `"bdh_cq"` that exposes standard configuration parameters, instantiates the BDH-CQ module with proper weights and layer norms, and executes basic forward passes producing token logits.

**Blocked by:** None — can start immediately.

**Status:** closed

- [x] Model is registered in `MODEL_REGISTRY` under the canonical key `"bdh_cq"`.
- [x] Model can be instantiated via `MODEL_REGISTRY.get("bdh_cq")` with custom parameters (e.g. `n_layer`, `n_embd`, `n_head`, `vocab_size`).
- [x] Invalid configurations (e.g. non-positive dimensions, invalid heads) raise descriptive `ValueError`s.
- [x] Module exports cleanly through the model package `__init__.py`.
- [x] Parameter count and tensor shapes match the theoretical $O(nd)$ architecture specification.
