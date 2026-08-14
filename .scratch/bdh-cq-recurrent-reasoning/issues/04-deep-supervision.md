# 04 — Multi-Step Deep Supervision and Loss Schedules

**What to build:** Multi-step auxiliary loss supervision across intermediate latent reasoning iterations ($r \in [1, R]$) with configurable loss schedules (`"ramp"`, `"uniform"`, `"final_only"`).

**Blocked by:** 03 — Recurrent Latent Workspace Reasoning

**Status:** ready-for-agent

- [ ] Training adapter computes intermediate logits $\hat{y}_r = H_r @ W_{readout}$ for each reasoning step $r \in [1, R]$.
- [ ] Supports `"ramp"` weighting ($w_r = \frac{r}{\sum_{j=1}^R j}$), `"uniform"` weighting ($w_r = \frac{1}{R}$), and `"final_only"` weighting ($w_R = 1.0$).
- [ ] Training loss is computed as the weighted sum $\mathcal{L} = \sum_{r=1}^R w_r \mathcal{L}_r$.
- [ ] Backpropagation propagates valid non-zero gradients through all intermediate reasoning passes and weights.
- [ ] End-to-end training step and validation step produce deterministic loss outputs.
