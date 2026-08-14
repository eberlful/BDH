# 02 — In-Context Recurrent Contextual Memory

**What to build:** Ingestion of in-context demonstration sequences into fixed-size, per-layer associative fast-weight matrices ($\boldsymbol{\rho}_{K, l}$) without an expanding key-value cache.

**Blocked by:** 01 — Baseline BDH-CQ Architecture and Model Registration

**Status:** closed

- [x] Demonstration tokens are sequentially ingested up to the query boundary delimiter.
- [x] Each layer $l \in [1, L]$ accumulates fast-weight associative matrix $\boldsymbol{\rho}_{K, l} = \sum_{\tau \le T_{demo}} v^*_{\tau, l-1} x_{\tau, l}^T \in \mathbb{R}^{B \times H \times (N/H) \times D}$.
- [x] Contextual memory state $S_K$ is frozen upon entering query processing.
- [x] Memory footprint remains constant and bounded regardless of demonstration length.
- [x] Unit tests verify that demonstration state accumulation correctly alters downstream attention readout.
