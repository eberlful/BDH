========================================================================================================
                                     BDH-CQ ARCHITECTURE OVERVIEW
========================================================================================================

  [Demonstrations D1 ... Dk]
              │
              ▼  (Sequential Forward Pass)
    ┌─────────────────────────────────────────────────────────┐
    │ BDH Layer Stack (Layer 1 ... L)                         │
    │  Accumulates Associative Fast-Weights:                  │
    │  ρ_{K, l} = Σ (v*_τ · x_τ^T)  ∈ R^{B x H x (N/H) x D}   │
    └─────────────────────────────────────────────────────────┘
              │
              │ Freeze Contextual Memory S_K = {ρ_{K, 1}, ..., ρ_{K, L}}
              ▼
  [Query x*] ──► H_0 = Initial Query Representation
                   │
                   ▼  Recurrent Latent Reasoning Loop (r = 0 ... R-1)
                 ┌────────────────────────────────────────────────────────┐
                 │ For each reasoning pass r:                             │
                 │   For each layer l:                                    │
                 │     1. x_r = ReLU(H_r @ D_x)                           │
                 │     2. a*_self = RoPE_LinearAttention(Q=x_r, K=x_r, V) │
                 │     3. a*_mem  = x_r @ ρ_{K, l}                        │
                 │     4. a* = a*_self + a*_mem                           │
                 │     5. y_r = ReLU(LN(a*) @ D_y) ⊙ x_r                  │
                 │     6. H_{r+1} = LN(H_r + LN(y_r @ E))                 │
                 └────────────────────────────────────────────────────────┘
                   │
                   ▼  (Deep Supervision / Readout)
                 Logits_r = H_r @ W_readout
                 Loss = Σ w_r · CrossEntropy(Logits_r, Targets)   (Schedule: "ramp")
========================================================================================================
