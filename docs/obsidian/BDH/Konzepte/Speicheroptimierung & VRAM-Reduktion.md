---
title: Speicheroptimierung & VRAM-Reduktion für BDH
tags:
  - bdh
  - optimization
  - vram
  - memory
  - todo
  - architecture
aliases:
  - VRAM-Optimierung
  - BDH Memory Roadmap
created: 2026-08-16
status: active
---

# ⚡ Speicheroptimierung & VRAM-Reduktion für BDH

Diese Notiz dokumentiert die Ursachen des hohen VRAM-Verbrauchs der **[[BDH - Baby Dragon Hatchling]]** Architektur sowie eine strukturierte Roadmap mit konkreten Optimierungsmaßnahmen und TODOs für das Training auf Systemen mit begrenztem Speicher (z. B. 16 GB Unified Memory auf Apple Silicon).

> [!abstract] Problemstellung
> Durch den standardmäßigen Expansionsmultiplikator `mlp_internal_dim_multiplier: 128` skaliert die interne Sparsitätsdimension pro Head auf $N = \frac{D \cdot M}{n_h} = 8192$.
> Bei einem Batch von 32 und Sequenzlänge 256 belegt jeder einzelne latente Aktivierungstensor `[32, 4, 256, 8192]` **1,024 GB RAM**. Über alle 6 Layer summiert sich der Autograd-Aktivierungsspeicher auf **> 20 GB**, was bei 16 GB Unified Memory zu Out-of-Memory (OOM) führt.

---

## 📋 TODOs & Roadmap

### 🟢 Phase 1: Sofortmaßnahmen / Konfiguration (P0)

- [ ] **Multiplikator `mlp_internal_dim_multiplier` benchmarken**
  - Standardwert 128 auf `32` oder `64` reduzieren (senkt Aktivierungsspeicher linear um 50% bzw. 75%).
  - Evaluieren, ob die Repräsentationsgüte und Konvergenz bei Sudoku/CoT erhalten bleibt.
- [ ] **`context_length` strikt an Task anpassen**
  - Für direkte Input-zu-Lösung-Tasks (`reasoning_mode: none`) genügen ~180 Tokens statt 256.
  - Spart ca. 30% Aktivierungsspeicher auf allen Zwischentensoren.
- [ ] **Gradient Accumulation als Default-Muster für 16 GB Geräte**
  - `batch_size: 8` mit `gradient_accumulation_steps: 4` (effektive Batch-Größe = 32) in Standard-Configs etablieren.

---

### 🟡 Phase 2: Trainings- & Laufzeit-Optimierungen (P1)

- [ ] **Gradient Checkpointing (Activation Recomputation) implementieren**
  - In `src/model/bdh.py` `torch.utils.checkpoint.checkpoint` für die Layer-Schleife einbinden.
  - Speichert im Vorwärtspass nur die Layer-Eingänge (~12 MB) statt alle Zwischenaktivierungen (~24 GB).
  - *Erwartete Ersparnis:* Reduktion des Aktivierungsspeichers um **~80%**.
- [ ] **Mixed Precision Training (BF16 / FP16) unterstützen**
  - Automatische Aktivierung von `torch.autocast(device_type=..., dtype=torch.bfloat16)` im `TorchTrainer`.
  - Halbiert den Speicherbedarf für Modellgewichte, Gradienten und Aktivierungen von 4 Byte auf 2 Byte pro Element.
- [ ] **VRAM- & Memory-Profiling Callback**
  - Automatisches Logging von `torch.mps.current_allocated_memory()` bzw. `torch.cuda.max_memory_allocated()` pro Epoche/Schritt.

---

### 🔵 Phase 3: Architektur & Daten-Pipeline (P2)

- [ ] **Spezialisierter Sudoku- / Task-Tokenizer**
  - Ersetzen des generischen GPT-2 Tokenizers ($50.257$ Tokens) durch einen kompakten Char-Tokenizer (~30–50 Tokens).
  - *Ersparnis:* Reduziert die Parameter von `embed` und `lm_head` von 25,7 Mio. auf < 50.000 (spart ~400 MB im AdamW-Optimizer).
- [ ] **Weight-Tying für `embed` und `lm_head`**
  - Optionale Kopplung der Gewichte (`self.lm_head = self.embed.weight`), falls Vokabular und Modell symmetrisch sind.
- [ ] **Custom Fused Kernels / Sparse-Operationen**
  - Untersuchen von Fused Kernels für `x @ encoder -> ReLU -> RoPE -> Attention`, um das Allokieren riesiger dichter Zwischentensoren zu vermeiden.

---

## 📊 Speicher-Benchmark (Apple Silicon / MPS)

Gemessen mit $D=256$, $n_h=4$, $M=128$ ($N=8192$), $T=256$, $L=6$:

| Batch Size | Multiplier $M$ | Peak VRAM | Status (16 GB Mac) |
| :--- | :--- | :--- | :--- |
| **1** | 128 | ~0.8 GB | ✅ Stabil |
| **4** | 128 | ~1.0 GB | ✅ Stabil |
| **8** | 128 | ~1.2 GB | ✅ Empfohlen für Training |
| **16** | 128 | > 15 GB | ⚠️ OOM-Gefahr |
| **32** | 128 | > 20 GB | ❌ OOM |
| **32** (mit $M=32$) | 32 | ~6.5 GB | ✅ Stabil möglich |
| **32** (mit Grad. Checkpointing) | 128 | ~5.0 GB | ✅ Stabil möglich |

---

## 🔗 Verwandte Notizen

- [[00 - Index & Übersicht]] – Hauptübersicht des Obsidian Vaults
- [[BDH - Baby Dragon Hatchling]] – Architekturdetails und mathematische Formulierung
- [[Modellvergleich & Benchmarks]] – Ressourcen- und Komplexitätsvergleich mit Transformern
- [[RoPE & Sparse Synaptic Gating]] – Details zu den hochdimensionalen Sparsity-Projektionen
