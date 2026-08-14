# 03 — Recurrent Latent Workspace Reasoning

**What to build:** Continuous iterative reasoning loop over $R$ recurrent passes in the latent workspace ($H_0 \to H_r \to H_R$) without discrete token verbalization, supporting scalable test-time compute.

**Blocked by:** 02 — In-Context Recurrent Contextual Memory

**Status:** ready-for-agent

- [ ] Query representation $H_0 = v^\star$ is iteratively updated for $R$ recurrent reasoning passes ($r = 0 \dots R-1$).
- [ ] Attention computes hybrid reasoning $a^* = a_{self}^* + a_{mem}^*$ combining spatial query self-attention and associative retrieval from $\boldsymbol{\rho}_{K, l}$.
- [ ] Layer updates apply sparse synaptic gating ($x \odot y$), low-rank decoding, and residual connections.
- [ ] Reasoning compute ($R$) can be configured dynamically at inference time without modifying model weights.
- [ ] Autoregressive generation executes the $R$-step latent reasoning loop prior to predicting next tokens.
