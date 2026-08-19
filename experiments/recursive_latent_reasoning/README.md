# Experiment: Rekurrente Denkzyklen im latenten Raum (BDH-CQ Sudoku)

Dieses Experiment evaluiert die Fähigkeit der **BDH-CQ (Contextual Query)** Architektur, komplexe relationale Constraint-Probleme (Sudoku 9x9) mittels **rekurrenter Denkzyklen im latenten Raum** (*Latent Reasoning Steps*) ohne explizite Verbalisierung von Chain-of-Thought (CoT) Tokens zu lösen.

---

## 1. Theoretischer Hintergrund & Hypothesen

### Problemstellung
Standard-Transformer und autoregressive Modelle operieren bei direkter Vorhersage (`reasoning_mode: none`) mit einer festen Schichttiefe $L$. Bei Sudoku erfordert die Lösung eine iterative Informationsausbreitung (Zeilen-, Spalten- und 3x3-Block-Constraint-Propagation). 

Herkömmlich wird dieses Problem durch **Chain-of-Thought (CoT)** gelöst, was jedoch:
1. Den Inferenz- und Trainingskontext massiv aufbläht ($\approx 500$–$1500$ Tokens pro Board).
2. Hohe Latenzen durch sequentielle Token-Generierung verursacht.

### Der BDH-CQ Ansatz
BDH-CQ führt stattdessen $R$ rekursente Denkzyklen ($r = 0, \dots, R-1$) im kontinuierlichen Vektorraum (*Latent Workspace* $H_r$) aus:

$$H_0 = \text{Embedding}(\text{Prompt})$$
$$H_{r+1} = \text{LayerNorm}\Big(H_r + \text{LayerNorm}(xy_r \cdot W_{dec})\Big)$$

Hierbei durchläuft der Zustand $H_r$ alle $L$ Layer des Modells unter Beibehaltung der Gewichte (Weight-Tying über die Rekurrenztiefe).

### Kernhypothesen
* **H1 (Tiefenskalierung)**: Mit steigender Anzahl latenter Schritte ($R=1 \to 2 \to 4 \to 8$) steigen `val/sudoku_cell_accuracy` und `val/sudoku_board_accuracy` signifikant, ohne dass die Modellparameter zunehmen.
* **H2 (Deep Supervision)**: Ein linear gewichteter Loss-Schedule (`loss_schedule: ramp`) über alle Zwischenschritte $H_r$ stabilisiert den Gradientenfluss und verhindert Gradient Vanishing/Explosion während des BPTT (*Backpropagation Through Time*).
* **H3 (Token-Effizienz)**: Ein Modell mit $R=4$ und `reasoning_mode: none` erreicht vergleichbare Genauigkeit wie ein $R=1$ Modell mit vollständigem CoT-Trace (`reasoning_mode: full`), benötigt aber nur einen Bruchteil der Sequenzlänge.

---

## 2. Hardware- & Memory-Profile (16GB Apple Silicon)

Da das Modell bei $R$ rekurrenten Schritten über $L$ Schichten abgerollt wird, vervielfacht sich der Speicherbedarf für Aktivierungen im Backward-Pass:

| Parameter | GPT-2 Tokenizer (`bdh_cq_sudoku.yaml`) | Byte Tokenizer (`bdh_cq_sudoku_byte.yaml`) | Rationale |
| :--- | :--- | :--- | :--- |
| **`vocab_size`** | `50257` | `257` | Enorme Parameter- und Speicherersparnis im LM-Head bei Byte-Level. |
| **`context_length`** | `256` | `512` | Byte-Grid benötigt ca. $344$ Bytes (Zeichen) für Prompt $\to$ Solution. |
| **`batch_size`** | `8` | `4` | Verhindert MPS-OOM während des unrollierten BPTT über $R=4$. |
| **`gradient_accumulation_steps`** | `4` (Eff. Batch: 32) | `8` (Eff. Batch: 32) | Ergibt eine stabile effektive Batch-Size von $32$. |
| **`mlp_internal_dim_multiplier`** | `32` | `32` | Reduziert den internen Sparse-Vektorraum von $128 \times D$ auf $32 \times D$. |

---

## 3. Experiment-Phasen & Durchführungsplan

### Phase 1: Baseline ($R=1$, Feedforward)
Training des Basismodells ohne Rekurrenz als Referenzpunkt.

```bash
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=1 \
  --set model.params.loss_schedule=final_only
```

---

### Phase 2: Latente Tiefenskalierung ($R \in \{2, 4, 8\}$)
Untersuchung des Einflusses der Denkzyklen bei aktivem Deep Supervision (`loss_schedule: ramp`).

```bash
# 2 Denkzyklen
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=2 \
  --set model.params.loss_schedule=ramp

# 4 Denkzyklen (Standard-Konfiguration)
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.loss_schedule=ramp

# 8 Denkzyklen (Extremum)
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=8 \
  --set model.params.loss_schedule=ramp \
  --set data.params.batch_size=4 \
  --set trainer.gradient_accumulation_steps=8
```

---

### Phase 3: Ablation von Deep Supervision
Vergleich verschiedener Loss-Schedules bei fixer Tiefe $R=4$:

```bash
# A: Linear ansteigende Gewichtung (Ramp: w_r = r / sum(1..R))
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.loss_schedule=ramp

# B: Gleichmäßige Gewichtung (Uniform: w_r = 1 / R)
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.loss_schedule=uniform

# C: Nur finaler Schritt (Final-Only: w_R = 1.0, alle anderen 0.0)
uv run python -m src.cli train configs/bdh_cq_sudoku.yaml \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.loss_schedule=final_only
```

---

### Phase 4: Latent Reasoning vs. Explicit Chain-of-Thought (CoT)
Vergleich von implizitem Schließen im latenten Raum gegen explizite Token-Generierung:

```bash
# Expliziter CoT-Lauf (BDH Basismodell mit CoT-Tokens)
uv run python -m src.cli train configs/bdh_sudoku.yaml \
  --set data.params.reasoning_mode=full \
  --set data.params.context_length=512 \
  --set model.params.context_length=512
```

---

## 4. Evaluierungs-Metriken

Die Validierung erfolgt automatisch durch den [SudokuValidator](file:///Users/markuseberl/Markus/Projekte/BDH/src/validation/sudoku.py):

| Metrik | Beschreibung | Ziel / Benchmark |
| :--- | :--- | :--- |
| **`val/sudoku_cell_accuracy`** | Anteil korrekt vorhergesagter Zellen (0.0 bis 1.0). | $> 0.85$ bei $R=4$ |
| **`val/sudoku_board_accuracy`** | Anteil vollständig fehlerfrei gelöster 81-Zellen-Boards. | Maximierung |
| **`val/sudoku_validity_rate`** | Anteil generierter Boards ohne Regelverletzungen (Zeilen, Spalten, 3x3 Blöcke). | $> 0.90$ |
| **`val/sudoku_parse_rate`** | Syntax-Gültigkeit der generierten Tokens als 81-Zahlen-Gitter. | $1.00$ |

---

## 5. Ergebnis-Matrix (Template)

| Experiment | $R$ (Steps) | Loss Schedule | Reasoning Mode | Params | Peak MPS Mem | Cell Acc (%) | Board Acc (%) | Validity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 1 | final_only | none | ~3.2M | ~1.1 GB | | | |
| Recurrent-2 | 2 | ramp | none | ~3.2M | ~1.4 GB | | | |
| Recurrent-4 | 4 | ramp | none | ~3.2M | ~2.1 GB | | | |
| Recurrent-8 | 8 | ramp | none | ~3.2M | ~3.6 GB | | | |
| Ablation-Uniform | 4 | uniform | none | ~3.2M | ~2.1 GB | | | |
| Ablation-Final | 4 | final_only | none | ~3.2M | ~2.1 GB | | | |
| CoT-Reference | 1 | n/a | full | ~3.2M | ~1.6 GB | | | |
