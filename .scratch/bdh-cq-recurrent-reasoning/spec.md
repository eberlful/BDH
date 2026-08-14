Status: ready-for-agent

# Spec: BDH-CQ In-Context Recurrent Memory and Latent Reasoning

## Problem Statement

Standard language models and Dragon Hatchling (BDH) sequence architectures process tokens autoregressively, requiring intermediate reasoning to be serialized as discrete tokens (Chain-of-Thought). This couples computation to token budget, increases inference latency, and forces intermediate hypotheses through a discrete vocabulary bottleneck. Furthermore, standard models require explicit key-value cache expansion to retain context across demonstration examples.

Researchers and engineers working on reasoning benchmarks need an architecture that:
1. Ingests task demonstrations sequentially into compact, fixed-size associative memory (fast-weights) without maintaining an explicit growing KV cache.
2. Solves queries through iterative continuous reasoning in a structured latent workspace without emitting intermediate discrete tokens.
3. Supports scalable test-time reasoning compute ($R$ passes) with parameter efficiency and stable gradient optimization.

## Solution

Introduce `BDHCQ` (and its training adapter `ConfiguredBDHCQ` registered as `"bdh_cq"` in the model registry), implementing the BDH-CQ architecture from the research paper:

1. **In-Context Recurrent Contextual Memory ($S_t$)**:
   - The model ingests task demonstrations sequentially, accumulating associative fast-weight synaptic states $\boldsymbol{\rho}_{K, l} = \sum_{\tau \le T_{demo}} v^*_{\tau, l-1} x_{\tau, l}^T \in \mathbb{R}^{B \times H \times (N/H) \times D}$ across all layers $l \in [1, L]$.
   - This state is frozen upon reaching the query boundary, providing an associative context representation with zero KV-cache memory overhead.

2. **Weight-Tied Recurrent Latent Workspace ($H_r$)**:
   - The query representation $H_0 = v^\star$ is iteratively updated across $R$ recurrent reasoning passes ($r = 0, \ldots, R-1$).
   - Within each pass $r$ and layer $l$, linear attention executes hybrid reasoning:
     $a^* = a_{self}^* + a_{mem}^*$, where $a_{self}^*$ is spatial self-attention across query positions and $a_{mem}^* = x_r @ \boldsymbol{\rho}_{K, l}$ is associative retrieval from demonstration memory.
   - The layer update applies BDH sparse synaptic gating ($x \odot y$), low-rank decoding, and residual LayerNorm integration.

3. **Configurable Deep Supervision**:
   - Computes weighted auxiliary loss over intermediate reasoning iterations $r \in [1, R]$ using a configurable schedule (`"ramp"`, `"uniform"`, or `"final_only"`).
   - In `"ramp"` mode, step weights scale as $w_r = \frac{r}{\sum_{j=1}^R j}$, stabilizing latent trajectory convergence while focusing loss on the final step.

## User Stories

1. As an ML researcher, I want to instantiate `"bdh_cq"` from the model registry, so that I can train and evaluate models with recurrent latent reasoning and in-context fast-weight memory.
2. As an ML researcher, I want to configure the number of latent reasoning steps (`reasoning_steps_R`) via configuration files or CLI overrides, so that I can study how scaling test-time latent compute impacts benchmark accuracy.
3. As an ML researcher, I want to configure `loss_schedule` (`"ramp"`, `"uniform"`, `"final_only"`), so that I can experiment with different multi-step deep supervision regimes.
4. As an ML engineer, I want the model to ingest demonstration tokens into fixed-size per-layer associative matrices ($\boldsymbol{\rho}_{K, l}$), so that GPU memory remains constant and bounded regardless of the number of demonstration examples.
5. As an ML engineer, I want the model to support standard sequence input dictionaries with segment/delimiter boundaries, so that existing dataset modules and training pipelines can feed demonstrations and queries seamlessly.
6. As a developer writing tests, I want deterministic unit tests verifying that recurrent latent passes propagate non-zero gradients to all model parameters and intermediate reasoning projections.
7. As an ML engineer, I want generation (`generate`) to execute the $R$ latent reasoning passes before decoding logits, so that inference accurately reflects the trained continuous reasoning trajectory.
8. As a maintainer, I want the baseline `BDH` implementation to remain completely untouched, so that existing benchmarks and configurations continue running with zero regression.

## Implementation Decisions

- **Registered Model Identifier**: The model is registered in `MODEL_REGISTRY` under the canonical key `"bdh_cq"`.
- **Model Classes**:
  - `BDHCQConfig`: Dataclass containing architectural parameters (`n_layer`, `n_embd`, `n_head`, `dropout`, `mlp_internal_dim_multiplier`, `vocab_size`, `reasoning_steps_R`, `loss_schedule`, `query_delimiter_id`).
  - `BDHCQ`: Core PyTorch `nn.Module` executing sequential demonstration memory ingestion, latent workspace recurrence, and readout projection.
  - `ConfiguredBDHCQ`: `BaseModel` adapter providing `forward`, `training_step`, `validation_step`, and `configure_optimizers` compatible with the training runtime.
- **Contextual Memory Ingestion**:
  - Ingests tokens up to the query delimiter token or boundary index.
  - Layer-wise state $\boldsymbol{\rho}_{K, l}$ has shape $[B, H, (N/H), D]$ and is detached/frozen as Contextual Memory for subsequent query reasoning.
- **Latent Reasoning Updates**:
  - $H_0 = \text{LayerNorm}(\text{Embedding}(x^\star))$.
  - For $r = 0 \dots R-1$:
    - For $l = 1 \dots L$:
      - $x = \text{ReLU}(H_r @ D_x)$
      - $a_{self}^* = \text{RoPE\_LinearAttention}(Q=x, K=x, V=H_r)$
      - $a_{mem}^* = x @ \boldsymbol{\rho}_{K, l}$
      - $a^* = a_{self}^* + a_{mem}^*$
      - $y = \text{ReLU}(\text{LayerNorm}(a^*) @ D_y) \odot x$
      - $y_{MLP} = y.transpose(1, 2).reshape(B, 1, T, N) @ E$
      - $H_{r+1} = \text{LayerNorm}(H_r + \text{LayerNorm}(y_{MLP}))$
- **Deep Supervision Loss**:
  - At each step $r \in [1, R]$, intermediate logits $\hat{y}_r = H_r @ W_{readout}$ are computed.
  - Total loss is the weighted sum $\mathcal{L} = \sum_{r=1}^R w_r \mathcal{L}_r( \hat{y}_r, \text{targets} )$.
  - Schedule weights:
    - `"ramp"`: $w_r = \frac{r}{\sum_{j=1}^R j}$
    - `"uniform"`: $w_r = \frac{1}{R}$
    - `"final_only"`: $w_r = 1.0$ for $r=R$, $0.0$ otherwise.
- **Glossary & ADR Alignment**:
  - Strictly follows terms in `CONTEXT.md` (**Contextual Memory**, **Latent Workspace**, **Latent Reasoning Steps**, **Deep Supervision**).
  - Aligns with `docs/adr/0002-bdh-cq-recurrent-memory-latent-reasoning.md`.

## Testing Decisions

- **Seam**:
  - Primary testing seam: The registered model adapter (`ConfiguredBDHCQ` / `BDHCQ`) via direct module testing and `MODEL_REGISTRY.get("bdh_cq")`.
- **Test Criteria**:
  - Model initialization with valid configs succeeds; invalid configs (e.g. non-positive `reasoning_steps_R` or invalid `loss_schedule`) raise descriptive `ValueError`s.
  - Forward pass produces logits with correct shape $[B, T, V]$ for both direct queries and demonstration-augmented inputs.
  - Training step calculates correct weighted loss across deep supervision steps and verifies backprop produces valid gradients for all parameters ($D_x, D_y, E, W_{readout}, W_{embed}$).
  - Changing `reasoning_steps_R` at inference dynamically scales the recurrent depth passes without requiring architecture reconfiguration.
  - In-context demonstration ingestion correctly updates the per-layer fast-weight tensor $\boldsymbol{\rho}_{K, l}$.
  - Autoregressive generation executes the latent workspace reasoning loop before emitting next tokens.
- **Prior Art**:
  - Follow the test structure in `tests/test_sudoku_cot.py` and `tests/test_training_environment.py`.

## Out of Scope

- ARC dataset parsing and visual augmentation pipelines (these belong in separate data modules).
- Hardware-specific CUDA kernels for associative memory (standard PyTorch tensor arithmetic is used).
- Modifying baseline `BDH` or `BDHTransformer`.

## Further Notes

- Because all $R$ reasoning steps share the same weights of the $L$ layers, scaling $R$ from 1 to 16 increases reasoning compute during inference with 0 additional parameters and minimal memory overhead.
