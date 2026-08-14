# 0002. BDH-CQ In-Context Recurrent Memory and Latent Reasoning

## Context

The BDH-CQ architecture introduces two complementary capabilities over baseline BDH:
1. **In-context learning through recurrent memory**: Tasks specified via demonstration pairs $D = \{(x_t, y_t)\}_{t=1}^K$ are ingested sequentially into associative fast-weight synaptic states ($S_K$) without an explicit KV cache.
2. **Recurrent latent reasoning**: Given query input $x^\star$, the model iteratively refines a continuous, high-dimensional latent workspace $H_r$ for $R$ recurrent passes without serializing intermediate thoughts into discrete tokens, then decodes the final prediction $\hat{y} = G_\theta(H_R)$.

We need a clean integration of these capabilities into the codebase that preserves baseline `BDH` compatibility while allowing flexible evaluation on reasoning benchmarks.

## Decision

We introduce a dedicated model implementation `BDHCQ` and training wrapper `ConfiguredBDHCQ` in `src/model/bdh_cq.py` registered as `"bdh_cq"` in `MODEL_REGISTRY`:

1. **Per-Layer Associative Contextual Memory**:
   - During the demonstration phase (tokens prior to query delimiter), each layer $l \in [1, L]$ accumulates associative fast-weights $\boldsymbol{\rho}_{K, l} = \sum_{\tau \le T_{demo}} v^*_{\tau, l-1} x_{\tau, l}^T$.
   - This state is frozen at the end of the demonstration sequence.

2. **Weight-Tied Recurrent Latent Workspace**:
   - The query latent state $H_0 = v^\star$ is iteratively updated over $R$ recurrent passes ($r = 0, \ldots, R-1$).
   - In each pass $r$ and layer $l$, linear attention performs hybrid reasoning:
     $a^* = a_{self}^* + a_{mem}^*$, where $a_{self}^*$ is spatial self-attention over query positions and $a_{mem}^* = x_r \boldsymbol{\rho}_{K,l}$ is associative retrieval from demonstration memory.
   - The layer update follows BDH sparse synaptic gating ($x \odot y$) and low-rank projection with residual connection.

3. **Multi-Step Deep Supervision**:
   - Training supports intermediate loss calculation across reasoning steps $r \in [1, R]$ with configurable `loss_schedule` (`"ramp"`, `"uniform"`, `"final_only"`).
   - In `"ramp"` mode, step weights scale as $w_r = \frac{r}{\sum_{j=1}^R j}$.

4. **Interface**:
   - Sequences are processed as unified token sequences with configurable delimiter/segment markers separating demonstrations, queries, and target solutions.

## Consequences

- Baseline `BDH` remains unchanged, preventing regression in standard language modeling tasks.
- Latent reasoning compute can be scaled at inference time simply by increasing $R$ without retraining or growing model parameter count.
- Memory consumption during inference remains bounded and independent of demonstration length since demonstrations are collapsed into fixed-size fast-weight matrices $\boldsymbol{\rho}_{K,l}$.
