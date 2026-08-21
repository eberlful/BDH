---
title: PonderNet-Hyperparameter-Screening für BDH-CQ
date: 2026-08-19
tags:
  - experiment
  - bdh-cq
  - pondernet
  - sudoku
status: in-progress
---

# PonderNet-Hyperparameter-Screening für BDH-CQ

## Ziel

Verschiedene PonderNet- und BDH-CQ-Konfigurationen mit kleinem Zeitbudget vergleichen, bevor ein vollständiger 9×9-Sudoku-Lauf gestartet wird.

Die Experimente sollen beantworten:

1. Lernt das Modell auf einer kleinen Aufgabe grundsätzlich stabil?
2. Verbessert PonderNet die Qualität gegenüber fester Reasoning-Schrittzahl?
3. Welche Kombination aus `learning_rate`, `R_max`, `ponder_lambda_p` und `ponder_beta` ist für einen vollständigen Lauf am vielversprechendsten?
4. Lernt PonderNet unterschiedliche Denkzeiten für unterschiedlich schwierige Beispiele?

> [!warning] Interpretation der Trainingszeit
> In der aktuellen Implementierung werden beim PonderNet-Training alle `R_max` Schritte vollständig ausgerollt. Der adaptive Frühabbruch wirkt primär bei der Inferenz. `ponder_halt_threshold` ist deshalb kein Trainingszeit-Hyperparameter.

## Agentenprotokoll

Der ausführende Agent arbeitet die Phasen in Reihenfolge ab.

- [ ] Vor jedem Lauf prüfen, ob die Konfiguration und der Code unverändert zum Vergleichsstand sind.
- [ ] Jeden Lauf mit eindeutiger Bezeichnung und festem Seed starten.
- [ ] Nach jedem Lauf Run-Verzeichnis, Dauer, Loss und Sudoku-Metriken eintragen.
- [ ] Einen Schritt erst abhaken, wenn der Lauf erfolgreich beendet wurde oder ein reproduzierbarer Fehler dokumentiert ist.
- [ ] Fehlgeschlagene Läufe nicht als erfolgreiche Experimente markieren.
- [ ] Nach jeder Screening-Phase nur die laut Auswahlregel besten Konfigurationen weiterführen.
- [ ] Keine vollständigen 7-Stunden-Läufe starten, bevor die Vorphasen abgeschlossen sind.

Verwendeter Startbefehl:

```bash
uv run python main.py train <config> --set <key>=<value>
```

Die Override-Syntax kann mehrfach verwendet werden. Alle Ergebnisse liegen standardmäßig unter `runs/`.

## Automatischer Nachtlauf

Die diagnostischen Overfit-Experimente können nacheinander mit folgendem Script gestartet werden:

```bash
chmod +x scripts/run_pondernet_experiments.sh
nohup ./scripts/run_pondernet_experiments.sh > runs/pondernet-screening.out 2>&1 &
echo $!
```

Das Script verwendet `configs/bdh_cq_sudoku_byte.yaml`, prüft zuerst die Sudoku-Dataset-/Validator-Tests und die Konfiguration und startet danach zwei deterministische Kleindatenläufe mit acht Beispielen, `batch_size=1` und 400 Steps: eine feste R=2-Baseline sowie PonderNet mit R=4. Die Daten werden mit `seed=42`, vier Trainings- und vier Validierungsbeispielen erzeugt. Nach jedem Training werden das erste Trainingsbeispiel autoregressiv dekodiert und zusätzlich Teacher-Forcing-Argmax gegen dieselbe Ground Truth verglichen. Puzzle, Ground Truth, erste Abweichungsposition und Tokenvorschauen werden im Run-Batch gespeichert. Der Rechner wird auf macOS während des Laufs mit `caffeinate` am Einschlafen gehindert, sofern dieser Befehl verfügbar ist.

Die Ergebnisse landen in einem automatisch erzeugten Verzeichnis unter `runs/pondernet-screening-<timestamp>/`. Dort befinden sich `status.tsv`, die Einzel-Logs und die von der Trainingsumgebung erzeugten Run-Verzeichnisse. Fehlgeschlagene Einzel-Experimente werden protokolliert; die nachfolgenden Experimente laufen trotzdem weiter. Das Script endet mit einem Fehlercode, sobald mindestens ein Lauf fehlgeschlagen ist.

Die langen Phase-3- und Phase-5-Läufe werden bewusst nicht automatisch gestartet. Ihre Hyperparameter müssen erst anhand der Screening-Ergebnisse ausgewählt und anschließend manuell gestartet werden.

## Vergleichsmetriken

Primäre Auswahlmetrik:

- `val/sudoku_cell_accuracy`

Sekundäre Metriken:

- `val/sudoku_board_accuracy`
- `val/sudoku_validity_rate`
- `val/sudoku_parse_rate`
- `ponder/task_loss`
- `ponder/kl_loss`
- `ponder/expected_steps`
- Trainingszeit pro 100 Steps
- maximaler bzw. durchschnittlicher GPU-Speicherverbrauch, falls verfügbar

Für alle Phasen denselben festen Evaluationssatz verwenden. Die Evaluationsgröße soll klein bleiben, zum Beispiel 64 oder 128 Rätsel. Die Bewertung soll nicht durch wechselnde Validierungsdaten verzerrt werden.

## Phase 0 – Code- und Laufzeitsanitycheck

Ziel: Sicherstellen, dass Baseline und PonderNet-Lauf funktionieren und die Laufzeit pro Step bekannt ist.

- [ ] Tests für PonderNet ausführen:

  ```bash
  uv run python -m unittest tests/test_pondernet.py -v
  ```

- [x] Einen festen-Schritt-Lauf mit `R=2` für 50 Steps starten.
- [x] Einen PonderNet-Lauf mit `R_max=2` für 50 Steps starten.
- [x] Laufzeit, Speicherverbrauch und Endloss dokumentieren.
- [x] Prüfen, dass `ponder/expected_steps` und `ponder/kl_loss` geloggt werden.
- [ ] Bei NaNs, CUDA-Fehlern oder fehlenden Metriken stoppen und Fehler dokumentieren.

Startbefehle:

```bash
# Feste-Reasoning-Baseline mit R=2
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=false \
  --set model.params.latent_reasoning_steps=2 \
  --set trainer.max_steps=50

# PonderNet mit R_max=2
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.latent_reasoning_steps=2 \
  --set trainer.max_steps=50
```

Dokumentation:

| Lauf | Konfiguration | Steps | Dauer | Endloss | Erwartete Steps | Ergebnis |
|---|---|---:|---:|---:|---:|---|
| Baseline | `20260819-192253-12c3b5` | 50 | 07:35 | 9.4497 | – | PASS |
| PonderNet | `20260819-193030-e35fd7` | 50 | 09:54 | 3.3399 | nicht eingetragen | PASS |

## Phase 1 – Kleine 4×4-Sudoku-Aufgabe

Ziel: PonderNet auf einer wesentlich billigeren, aber weiterhin strukturierten Aufgabe testen.

Falls 4×4-Sudoku noch nicht als DataModule unterstützt wird, soll der Agent eine kleine synthetische Variante ergänzen oder ersatzweise mit 9×9-Sudoku und stark reduziertem Datensatz fortfahren. Die Modell- und Evaluationslogik soll dabei unverändert bleiben.

Empfohlene Einstellungen:

```yaml
model.params:
  n_layer: 3
  n_embd: 128
  latent_reasoning_steps: 2
  enable_pondernet: true

data.params:
  num_samples: 1000

trainer:
  max_steps: 300
  max_epochs: 3
```

Startbefehle. Die vorhandene Konfiguration nutzt 9×9-Sudoku. Sobald ein 4×4-DataModule vorhanden ist, muss nur der entsprechende Daten- bzw. Config-Name ersetzt werden.

```bash
# Feste-Reasoning-Baseline mit R=2
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=false \
  --set model.params.latent_reasoning_steps=2 \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set data.params.num_samples=1000 \
  --set trainer.max_steps=300 \
  --set trainer.max_epochs=3

# PonderNet mit R_max=2
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.latent_reasoning_steps=2 \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set data.params.num_samples=1000 \
  --set trainer.max_steps=300 \
  --set trainer.max_epochs=3

# PonderNet mit R_max=4
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set data.params.num_samples=1000 \
  --set trainer.max_steps=300 \
  --set trainer.max_epochs=3
```

- [x] Feste `R=2`-Baseline trainieren.
- [x] PonderNet mit `R_max=2` trainieren.
- [x] PonderNet mit `R_max=4` trainieren (abgebrochen: MPS-Out-of-Memory nach Epoch 1).
- [x] Cell-Accuracy, Board-Accuracy und `expected_steps` vergleichen.
- [ ] Prüfen, ob schwierigere Rätsel im Mittel mehr Schritte verwenden.
- [x] Die besten zwei Kandidaten für Phase 2 auswählen; wegen der fehlenden Lernqualität nur als technisches Screening, nicht als Qualitätsentscheidung.

Auswahlregel: Eine Konfiguration kommt weiter, wenn sie nicht deutlich schlechter als die Baseline ist und entweder bessere Sudoku-Metriken oder ein plausibles, nicht kollabiertes Halteverhalten zeigt.

Ergebnisse:

| ID | Modell | R/R_max | LR | λ_prior | β_KL | Steps | Cell-Acc. | Board-Acc. | Expected Steps | Dauer | Weiter? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | feste Baseline | 2 | 3e-4 | – | – | 300 | 0.1111 | 0.0000 | – | 08:02 | technische Referenz |
| 2 | PonderNet | 2 | 3e-4 | 0.2 | 0.01 | 300 | 0.1107 | 0.0000 | 1.8945 | 11:35 | weiter mit Vorsicht |
| 3 | PonderNet | 4 | 3e-4 | 0.2 | 0.01 | 300 | 0.0035* | 0.0000 | 2.6648* | 46:10* | OOM |

`*` letzter gemessener Wert vor dem MPS-Out-of-Memory-Fehler; der Lauf erreichte nur etwa 113 von 300 Steps.

## Phase 2 – Hyperparameter-Screening mit kleiner 9×9-Aufgabe

Ziel: Mit begrenztem Budget die wichtigsten Hyperparameter aussortieren.

Für jede Konfiguration 300 Steps trainieren. Die Trainingsdaten auf 1.000 Beispiele begrenzen und die Modellgröße klein halten.

| ID | LR | R_max | `ponder_lambda_p` | `ponder_beta` |
|---|---:|---:|---:|---:|
| A | 1e-4 | 2 | 0.15 | 0.001 |
| B | 3e-4 | 2 | 0.20 | 0.010 |
| C | 1e-3 | 2 | 0.40 | 0.010 |
| D | 1e-4 | 4 | 0.15 | 0.001 |
| E | 3e-4 | 4 | 0.20 | 0.010 |
| F | 1e-3 | 4 | 0.40 | 0.050 |

Startbefehle für die sechs Screening-Konfigurationen:

```bash
# A: LR=1e-4, R_max=2, lambda_p=0.15, beta=0.001
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.0001 --set model.params.latent_reasoning_steps=2 --set model.params.ponder_lambda_p=0.15 --set model.params.ponder_beta=0.001 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300

# B: LR=3e-4, R_max=2, lambda_p=0.20, beta=0.010
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.0003 --set model.params.latent_reasoning_steps=2 --set model.params.ponder_lambda_p=0.20 --set model.params.ponder_beta=0.010 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300

# C: LR=1e-3, R_max=2, lambda_p=0.40, beta=0.010
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.001 --set model.params.latent_reasoning_steps=2 --set model.params.ponder_lambda_p=0.40 --set model.params.ponder_beta=0.010 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300

# D: LR=1e-4, R_max=4, lambda_p=0.15, beta=0.001
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.0001 --set model.params.latent_reasoning_steps=4 --set model.params.ponder_lambda_p=0.15 --set model.params.ponder_beta=0.001 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300

# E: LR=3e-4, R_max=4, lambda_p=0.20, beta=0.010
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.0003 --set model.params.latent_reasoning_steps=4 --set model.params.ponder_lambda_p=0.20 --set model.params.ponder_beta=0.010 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300

# F: LR=1e-3, R_max=4, lambda_p=0.40, beta=0.050
uv run python main.py train configs/bdh_cq_sudoku.yaml --set model.params.enable_pondernet=true --set model.params.learning_rate=0.001 --set model.params.latent_reasoning_steps=4 --set model.params.ponder_lambda_p=0.40 --set model.params.ponder_beta=0.050 --set model.params.n_layer=3 --set model.params.n_embd=128 --set data.params.num_samples=1000 --set trainer.max_steps=300
```

- [x] Konfiguration A trainieren.
- [x] Konfiguration B trainieren.
- [x] Konfiguration C trainieren.
- [x] Konfiguration D trainieren (abgebrochen: MPS-Out-of-Memory nach Epoch 1).
- [x] Konfiguration E trainieren (abgebrochen: MPS-Out-of-Memory nach Epoch 1).
- [x] Konfiguration F trainieren (abgebrochen: MPS-Out-of-Memory nach Epoch 1).
- [x] Ergebnisse nach `val/sudoku_cell_accuracy` und `val/sudoku_board_accuracy` sortieren.
- [ ] Die besten zwei Konfigurationen für Phase 3 auswählen; zunächst ist eine technische Speicher-Reparatur erforderlich.

Ergebnisse:

| ID | Run-Verzeichnis | Cell-Acc. | Board-Acc. | Validity | Expected Steps | Dauer | Entscheidung |
|---|---|---:|---:|---:|---:|---:|---|
| A | `20260819-204618-4ae2d9` | 0.1119 | 0.0000 | 0.0000 | 1.7758 | 09:35 | PASS; bester Cell-Wert, aber praktisch Zufallsniveau |
| B | `20260819-205552-70b9eb` | 0.1107 | 0.0000 | 0.0000 | 1.8947 | 11:33 | PASS; stabiler als C, dennoch kein Board-Erfolg |
| C | `20260819-210727-3a0b77` | 0.1092** | 0.0000 | 0.0000 | 1.3935 | 12:06 | PASS; finaler Cell-Wert fiel auf 0.0548 |
| D | `20260819-211936-8ad4e4` | 0.0035* | 0.0000 | 0.0000 | 2.1254* | 48:40* | FAIL: MPS-Out-of-Memory |
| E | `20260819-220817-461802` | 0.0035* | 0.0000 | 0.0000 | 2.6649* | 49:15* | FAIL: MPS-Out-of-Memory |
| F | `20260819-225731-30e5ee` | 0.1111* | 0.0000 | 0.0000 | 2.2686* | 49:57* | FAIL: MPS-Out-of-Memory |

`*` letzter gemessener Wert vor dem OOM-Fehler. `**` bester Zwischenwert aus Epoch 2; der finale Wert von C betrug 0.0548.

## Auswertung des Nachtlaufs vom 19./20.08.2026

Run-Batch: `runs/pondernet-screening-20260819-212251/`

### Was funktioniert hat

- Die Konfiguration wurde erfolgreich validiert.
- Alle `R=2`-Experimente liefen technisch vollständig durch.
- PonderNet protokolliert `task_loss`, `kl_loss` und `expected_steps` korrekt.
- PonderNet mit `R_max=2` erreicht eine erwartete Schrittzahl zwischen etwa 1.39 und 1.89 und kollabiert in diesen Läufen nicht vollständig auf einen einzigen Schritt.
- Die sechs Kandidaten zeigen, dass die Lernrate einen großen Einfluss auf den Loss hat; `1e-3` senkt den Loss schneller als `1e-4`.

### Was nicht funktioniert hat

- Die R=4-Läufe scheitern auf dem M4 reproduzierbar am MPS-Speicherlimit. Der Fehler lautet sinngemäß: 18.30 GiB MPS-Speicher belegt, zusätzlicher Versuch 1.53 GiB.
- Die Läufe verwendeten den GPT-2-Tokenizer mit einer sehr großen Vokabulargröße. Beim PonderNet-Training werden die Logits aller Reasoning-Schritte für den Loss gespeichert. Zusammen mit Batch-Größe 8, Sequenzlänge 256 und `R_max=4` führt das zu hohem Speicherbedarf.
- Die Sudoku-Metriken zeigen noch kein echtes Lernen: `board_accuracy=0`, `validity_rate=0` und Cell-Accuracy ungefähr 0.1111. Das ist nahe dem Niveau einer konstanten bzw. zufälligen Ziffernvorhersage.
- Die Kandidaten A–C sind deshalb noch keine belastbare Qualitätsrangliste. A ist nur nach Cell-Accuracy minimal besser; der Unterschied ist bei 300 Steps nicht aussagekräftig.

## Konsequenz für Phase 3

Phase 3 und der vollständige 9×9-Lauf werden noch nicht gestartet. Vorher muss der Speicherbedarf für PonderNet reduziert und auf einer kleinen Konfiguration nachgewiesen werden, dass überhaupt Sudoku-Qualität entsteht.

Der wahrscheinlich wichtigste Hebel ist der Byte-Tokenizer aus [bdh_cq_sudoku_byte.yaml](../../configs/bdh_cq_sudoku_byte.yaml). Er reduziert das Vokabular von GPT-2 auf ungefähr 257 Tokens, erhöht aber die Sequenzlänge. Daher muss gleichzeitig die Batch-Größe klein gewählt werden.

## Nächster Handlungsbedarf

- [x] PonderNet-Speicherverhalten mit Byte-Tokenizer und `R_max=4` testen.
- [x] Für den nächsten Test `batch_size=1` oder `2`, `num_samples=256` und `max_steps=100` verwenden.
- [ ] Prüfen, ob die Logits aller Reasoning-Schritte unnötig materialisiert werden; langfristig sollte der Loss schrittweise berechnet oder die Ausgabe für die Loss-Berechnung speichersparender gestaltet werden.
- [ ] Erst nach einem erfolgreichen R=4-Test mit Byte-Tokenizer die Hyperparameter A–F erneut starten.
- [ ] Einen Lernfortschrittstest mit mindestens 1.000–3.000 Steps durchführen; 300 Steps reichen hier nur für technische Sanitychecks.
- [ ] Wenn `board_accuracy` und `validity_rate` weiterhin 0 bleiben, zuerst Daten-/Target-Ausrichtung und Decoding prüfen, bevor weitere Hyperparameterläufe gestartet werden.
- [ ] Danach R=2 und R=4 mit identischer Byte-Tokenizer-Basis vergleichen.

Empfohlener sicherer R=4-Test:

```bash
uv run python main.py train configs/bdh_cq_sudoku_byte.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set model.params.learning_rate=0.0003 \
  --set model.params.ponder_lambda_p=0.2 \
  --set model.params.ponder_beta=0.01 \
  --set data.params.num_samples=256 \
  --set data.params.batch_size=1 \
  --set trainer.max_steps=100 \
  --set trainer.max_epochs=1 \
  --set runs_dir=runs/pondernet-byte-r4-memory-test
```

## Auswertung des Byte-Tokenizer-Laufs vom 20.08.2026

Run-Batch: `runs/pondernet-byte-screening-20260820-072139/`

### Ergebnisse

| ID | Konfiguration | Run | Steps | Dauer | Train-Loss | Val-Loss | Cell-Acc. | Board-Acc. | Validity | Parse | Expected Steps | Ergebnis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | R=4, LR 3e-4, λ 0.2, β 0.01 | `20260820-052142-15f5be` | 100 | 02:49 | 1.3554 | 1.1607 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 1.8143 | PASS |
| 1 | feste Baseline, R=2 | `20260820-052433-c48969` | 100 | 01:49 | 4.2378 | 3.6181 | 0.1125 | 0.0000 | 0.0000 | 1.0000 | – | PASS |
| 2 | PonderNet, R=2, LR 3e-4 | `20260820-052624-50def5` | 100 | 01:45 | 1.3402 | 1.1503 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 1.6455 | PASS |
| A | PonderNet, R=2, LR 1e-4 | `20260820-052811-c0b27d` | 100 | 01:42 | 1.4458 | 1.2815 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 1.6325 | PASS |
| B | PonderNet, R=2, LR 3e-4 | `20260820-052955-b154bc` | 100 | 01:58 | 1.3402 | 1.1503 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 1.6455 | PASS |
| C | PonderNet, R=2, LR 1e-3 | `20260820-053155-78f901` | 100 | 01:49 | 1.1628 | 0.8095 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6239 | PASS, aber schlechte Generierung |
| D | PonderNet, R=4, LR 1e-4 | `20260820-053346-c1148c` | 100 | 02:46 | 1.4601 | 1.2917 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.7366 | PASS, aber schlechte Generierung |
| E | PonderNet, R=4, LR 3e-4 | `20260820-053634-303581` | 100 | 02:47 | 1.3554 | 1.1607 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 1.8144 | PASS |
| F | PonderNet, R=4, LR 1e-3 | `20260820-053922-b3b73f` | 100 | 02:47 | 1.1775 | 0.8195 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 2.1792 | PASS |

### Befund

- Der R=4-Byte-Lauf ist speicherstabil. Der frühere GPT-2-MPS-OOM ist damit auf dieser Konfiguration nicht mehr reproduziert.
- Die R=4-Trainingsgeschwindigkeit liegt bei ungefähr 3,7–3,8 Steps pro Sekunde mit `batch_size=1`.
- Die zehn Läufe liefen insgesamt in ungefähr 21 Minuten durch. Weitere kurze Screening-Läufe sind damit auf dem Mac Mini praktikabel.
- Die feste Byte-Baseline erreicht mit `val/cell_accuracy=0.1125` minimal mehr als die PonderNet-Läufe, aber keine vollständigen Boards.
- Kandidat C hat zwar den niedrigsten Loss, aber `parse_rate=0` und `cell_accuracy=0`. Der niedrigere Loss ist deshalb kein ausreichendes Auswahlkriterium.
- Kandidat E ist derzeit der sinnvollste Standardkandidat für einen längeren R=4-Lauf: stabile Generierung, erwartete Schrittzahl 1.81 und mittlere Lernrate `3e-4`.
- Die 100 Steps entsprechen bei 256 Beispielen und Batch-Größe 1 noch keinem belastbaren Training. Die Ergebnisse sagen vor allem, dass die Implementierung läuft und der Speicher reicht.

## Aktualisierte Entscheidung für die nächsten Läufe

Phase 3 wird jetzt nicht als Hyperparameter-Ranking, sondern als Lernfortschrittstest gestartet.

- [x] R=4-Speicherstabilität mit Byte-Tokenizer nachweisen.
- [x] Kandidat E mit 1.000 Steps trainieren.
- [x] Kandidat B mit 1.000 Steps trainieren.
- [x] Feste R=2-Byte-Baseline mit 1.000 Steps trainieren.
- [x] Cell-Accuracy, Parse-Rate und Validity-Rate über mehrere Validierungen vergleichen.
- [ ] Nur wenn die Metriken steigen, auf 3.000–5.000 Steps erweitern.
- [ ] Erst danach einen vollständigen Lauf mit 10.000 Beispielen planen.

Empfohlene Reihenfolge:

1. R=4 PonderNet: `learning_rate=3e-4`, `ponder_lambda_p=0.2`, `ponder_beta=0.01`
2. R=2 PonderNet mit denselben Hyperparametern
3. feste R=2 Byte-Baseline

Für den nächsten Lauf sollten `batch_size=1` und der Byte-Tokenizer beibehalten werden. Die Ergebnisse des 100-Step-Laufs sind nicht ausreichend, um bereits eine PonderNet-Qualitätsverbesserung zu behaupten.

## Auswertung des 1.000-Step-Progress-Laufs vom 20.08.2026

Run-Batch: `runs/pondernet-byte-progress-20260820-082040/`

| Lauf | Run | Dauer | Train-Loss | Val-Loss | beste Cell-Acc. | beste Board-Acc. | Validity | Parse | Expected Steps am Ende | Ergebnis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PonderNet R=4, Kandidat E | `20260820-062042-129946` | 16:06 | 0.4434 | 0.4394 | 0.1125 | 0.0000 | 0.0000 | 1.0000 | 2.9638 | technisch PASS |
| PonderNet R=2, Kandidat B | `20260820-063650-2da080` | 10:13 | 0.4385 | 0.4343 | 0.1121 | 0.0000 | 0.0000 | 1.0000 | 1.8010 | technisch PASS |
| feste R=2-Baseline | `20260820-064706-92c372` | 10:24 | 1.3160 | 1.3099 | 0.1116 | 0.0000 | 0.0000 | 1.0000 | – | technisch PASS |

### Verlauf und Interpretation

- Alle drei Modelle reduzieren ihren Trainings- und Validierungs-Loss über fünf Epochen deutlich.
- Der niedrigere Loss der PonderNet-Modelle führt nicht zu einer besseren Sudoku-Lösung. Der Task-Validator bleibt bei `board_accuracy=0` und `validity_rate=0`.
- `parse_rate=1.0` zeigt, dass die Ausgabe formal extrahierbar ist. Das Modell erzeugt also eine syntaktisch lesbare Lösung, aber nicht das korrekte Sudoku-Board.
- R=4 verschiebt `expected_steps` von ungefähr 1.8 am Anfang auf ungefähr 3.0 am Ende. Das Modell nutzt für die Aufgabe fast die maximale Rechentiefe, ohne dadurch bessere Boards zu lösen.
- R=2 und R=4 unterscheiden sich bei der besten Cell-Accuracy nur um 0.0004. Das ist kein belastbarer Qualitätsvorteil für R=4.
- Die feste Baseline ist beim Loss deutlich schlechter, aber die task-spezifischen Metriken sind genauso schlecht. Der PonderNet-Loss ist daher nicht direkt als Sudoku-Erfolg interpretierbar.
- Der Byte-Tokenizer hat das Speicherproblem gelöst: R=4 lief vollständig und ohne MPS-Out-of-Memory.

### Entscheidung

Die PonderNet-Implementierung ist jetzt technisch lauffähig und speicherstabil, aber ein Sudoku-Erfolg ist nicht nachgewiesen. Weitere reine Hyperparameter-Suchen werden vorerst nicht gestartet. Der nächste Engpass ist wahrscheinlich die Lern-/Evaluationsstrecke: Target-Sequenz, Loss-Maske, Decoding oder die Schwierigkeit der Trainingsdaten.

## Aktualisierter Handlungsbedarf nach dem 1.000-Step-Lauf

- [x] R=4-Speicherstabilität mit Byte-Tokenizer nachweisen.
- [x] R=4 gegen R=2 und feste Baseline über 1.000 Steps vergleichen.
- [ ] Einen festen Validierungsfall mit Puzzle, Ground-Truth-Lösung und Modelloutput ausgeben.
- [ ] Prüfen, ob die extrahierte 81-Ziffern-Lösung tatsächlich die vom Modell erzeugte Lösung und nicht nur ein Fallback/Trivialwert ist.
- [ ] Input-/Target-Tokenisierung für Byte-Sudoku auf einem einzelnen Beispiel manuell vergleichen.
- [ ] Loss-Maske prüfen: Nur Lösungstokens dürfen zum Trainings-Loss beitragen; Prompt-Tokens dürfen nicht die Metrik dominieren.
- [ ] Prüfen, ob `val/loss` und `val/sudoku_cell_accuracy` dieselbe Ausgabe bzw. denselben Tokenbereich bewerten.
- [ ] Einen sehr kleinen Overfit-Test mit 1–8 festen Sudoku-Beispielen durchführen.
- [ ] Erst wenn dieser Overfit-Test eine steigende Board-Accuracy zeigt, wieder längere oder größere Trainingsläufe starten.

Empfohlener nächster Test: kein weiterer Hyperparameterlauf, sondern ein Overfit-/Decoding-Diagnoselauf mit einem festen Puzzle. Ein korrekt implementierter Lernpfad sollte auf wenigen festen Beispielen deutlich über das Cell-Accuracy-Niveau von etwa 1/9 steigen und schließlich mindestens einzelne vollständige Boards lösen.

## Auswertung des Overfit-Diagnoselaufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-092800/`

| Lauf | Run | Unit-Tests | konfigurierte Steps | tatsächlich ausgeführte Steps | Dauer | finaler Train-Loss | finaler Val-Loss | Cell-Acc. | Board-Acc. | Validity | Parse | Ergebnis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| feste R=2-Baseline | `20260820-072809-dacbe3` | PASS | 2.000 | 80 | 05:13 | 1.0472 | 1.0297 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | technisch PASS, Overfit-Test unvollständig |
| PonderNet R=4 | `20260820-073323-296a07` | PASS | 2.000 | 80 | 07:30 | 1.0472 | 1.0297 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | technisch PASS, Overfit-Test unvollständig |

### Kritischer Befund zum Testaufbau

- Die Unit-Tests für Dataset und Validator liefen erfolgreich durch.
- Der Runner war technisch erfolgreich, aber `max_steps=2000` wurde nicht erreicht.
- Wegen `num_samples=8` und `validation_fraction=0.5` gab es nur vier Trainingsbeispiele. Mit `max_epochs=20` endete jeder Lauf nach `4 × 20 = 80` Steps.
- Der Validator bewertete die vier Validierungsbeispiele, nicht die vier Trainingsbeispiele. Damit ist dieser Lauf kein echter Overfit-Nachweis.
- Beide Modelle zeigen nach 80 Steps dasselbe Verhalten: sinkender Loss, aber keine korrekten Boards.
- `parse_rate=1.0` bedeutet erneut, dass die Ausgabe formal auswertbar ist; `board_accuracy=0` und `validity_rate=0` zeigen, dass die Lösung inhaltlich falsch ist.

### Konsequenz

Aus diesem Lauf darf noch nicht geschlossen werden, dass das Modell nicht overfitten kann. Der Test muss mit einer korrigierten Epochengrenze und einer Bewertung der Trainingsbeispiele wiederholt werden.

Für vier Trainingsbeispiele gilt:

```text
400 Steps  = 100 Epochen
2.000 Steps = 500 Epochen
```

Ein 400-Step-Lauf ist der sinnvollere nächste Zwischentest. Erst wenn der Trainings-Loss weiter fällt und die Trainingsbeispiele trotzdem nicht lösbar werden, muss die Daten-/Decoding-Strecke genauer untersucht werden.

## Aktualisierter Handlungsbedarf nach dem Overfit-Lauf

- [x] Dataset- und Validator-Unit-Tests ausführen.
- [x] Baseline und PonderNet auf derselben deterministischen Kleindatenkonfiguration ausführen.
- [x] Fehler im Testbudget dokumentieren: 2.000 konfigurierte, aber nur 80 ausgeführte Steps.
- [x] Runner auf `max_epochs=100` für einen 400-Step-Zwischentest anpassen.
- [x] Nach jedem Training das erste deterministische Trainingsbeispiel mit dem besten Checkpoint dekodieren.
- [ ] Trainingsbeispiele direkt evaluieren; die aktuelle Validierung verwendet nur `val_dataset`.
- [x] Für mindestens ein Trainingsbeispiel Puzzle, Ground Truth und generierte Lösung ausgeben.
- [ ] Bei erfolgreichem Trainings-Overfit die vier Validierungsbeispiele separat auf Generalisierung prüfen.
- [ ] Erst danach entscheiden, ob ein 2.000-Step-Overfit-Lauf oder ein größerer Trainingslauf sinnvoll ist.

## Auswertung des korrigierten 400-Step-Overfit-Laufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-094635/`

| Lauf | Run | tatsächliche Steps | Dauer | finaler Train-Loss | finaler Val-Loss | beste Cell-Acc. | Board-Acc. | Validity | Parse | Decoding des Trainingsbeispiels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| feste R=2-Baseline | `20260820-074645-c33112` | 400 | 04:05 | 1.3994 | 1.3953 | 0.1142 | 0.0000 | 0.0000 | 1.0000 | formal parsebar, aber nicht die Ground Truth |
| PonderNet R=4 | `20260820-075057-c2bfb7` | 400 | 05:34 | 0.4772 | 0.4756 | 0.1142 | 0.0000 | 0.0000 | 1.0000 | bricht nach wenigen Tokens ab, nicht die Ground Truth |

Unit-Tests und Konfigurationsvalidierung waren erfolgreich.

### Direkter Trainingsbeispiel-Vergleich

Gespeichertes Trainingsbeispiel: [training-example.txt](../../runs/sudoku-overfit-20260820-094635/training-example.txt)

Ground Truth:

```text
498321657365487219271569384154632978629718435783954126836145792517296843942873561
```

Die Decoding-Logs liegen hier:

- [Baseline-Decoding](../../runs/sudoku-overfit-20260820-094635/logs/00-fixed-r2-training-example.log)
- [PonderNet-Decoding](../../runs/sudoku-overfit-20260820-094635/logs/01-ponder-r4-training-example.log)

Keines der beiden Modelle erzeugt die Ground Truth. Die Baseline erzeugt eine lange, byte-sequenzartige Ausgabe mit falschen Ziffern und Sonderzeichen. PonderNet erzeugt nur wenige Tokens und beendet die Ausgabe frühzeitig.

### Interpretation

- Der Epochengrenzenfehler ist behoben: Es wurden tatsächlich 400 statt 80 Steps ausgeführt.
- Der Loss sinkt bei beiden Modellen, bei PonderNet deutlich stärker.
- Der niedrigere PonderNet-Loss führt trotzdem nicht zu besserem autoregressivem Decoding.
- Die Validierungsmetriken bleiben auf praktisch trivialem Niveau. Das Problem ist damit nicht nur fehlende Trainingsdauer.
- Der direkte Trainingsbeispiel-Test zeigt eine Diskrepanz zwischen Teacher-Forcing-Training und autoregressiver Generierung.
- `status=PASS` im Decoding-Log bedeutet bisher nur, dass der CLI-Aufruf ohne Prozessfehler beendet wurde; es bedeutet nicht, dass die Ground Truth erzeugt wurde.

## Neuer technischer Schwerpunkt

Vor weiteren Trainingsläufen muss die Ausgabe-Strecke isoliert werden:

1. Einen einzelnen Trainingsbatch laden.
2. Unter Teacher Forcing die Argmax-Ausgabe an den Lösungstoken-Positionen berechnen.
3. Diese Ausgabe gegen `target_ids` vergleichen.
4. Danach dieselbe Eingabe autoregressiv mit `model.generate()` ausführen.
5. Beide Tokenfolgen und die erste Abweichungsposition ausgeben.

Wenn Teacher Forcing korrekt ist, aber Generation scheitert, liegt der Fehler wahrscheinlich in Prompt-/EOS-Behandlung oder im Generationspfad. Wenn bereits Teacher Forcing falsche Lösungstokens erzeugt, liegt das Problem in Training, Target-Maskierung oder Modellkapazität.

### Aktualisierter Handlungsbedarf

- [x] Tatsächliche 400 Steps ausführen.
- [x] Ground Truth und autoregressiven Output des Trainingsbeispiels speichern.
- [x] Nachweisen, dass beide Modelle das Trainingsbeispiel noch nicht lösen.
- [x] Teacher-Forced-Argmax auf dem ersten Trainingsbeispiel implementieren.
- [x] Erste Abweichungsposition zwischen Target und Teacher-Forced-Prediction ausgeben.
- [x] EOS-Token, Prompt-Länge und Lösungslänge für Byte-Tokenizer prüfen.
- [ ] Prüfen, ob das Modell beim Training auf Lösungstokens oder überwiegend auf Prompt-/Padding-Tokens optimiert wird.
- [ ] Erst nach diesem Vergleich entscheiden, ob mehr Trainingsschritte oder eine Codekorrektur notwendig sind.

## Auswertung des Teacher-Forcing-Laufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-100809/`

| Lauf | Unit-Tests | Checkpoint | Prompt-Tokens | Completion-Tokens | erste Teacher-Forcing-Abweichung | erste autoregressive Abweichung | Teacher-Forcing exakt | Autoregressiv exakt |
|---|---|---|---:|---:|---:|---:|---|---|
| feste R=2-Baseline | PASS | `20260820-080820-f8231a`, Epoch 100 / Step 400 | 181 | 163 | 0 | 0 | Nein | Nein |
| PonderNet R=4 | PASS | `20260820-081155-473faf`, Epoch 70 / Step 280 | 181 | 163 | 0 | 0 | Nein | Nein |

### Lokalisierung des Fehlers

- Beide Verfahren sagen bereits das erste Lösungstoken falsch voraus. Teacher Forcing und autoregressive Generierung zeigen also dasselbe Grundproblem.
- Der Fehler liegt damit nicht ausschließlich im autoregressiven Feedback oder im EOS-Abbruch.
- Der Byte-Tokenizer und die Prompt-/Completion-Längen sind konsistent: 181 Prompt-Tokens, 163 Completion-Tokens, EOS `256`.
- Die Teacher-Forced-Ausgaben bestehen überwiegend aus wiederholten Ziffern (`2` bei der Baseline, `5`/`6` bei PonderNet) und passen nicht zur Ground Truth.
- Der direkte Decoding-Check ist damit aussagekräftig: Beide Modelle haben das Trainingsbeispiel nach 400 Steps nicht memorisiert.

### Konkreter Codebefund im PonderNet-Loss

In [bdh_cq.py](../../src/model/bdh_cq.py) wird beim PonderNet-Training pro Schritt zunächst ein Token-Loss mit `reduction="none"` berechnet und anschließend über die gesamte Sequenz gemittelt. Dabei liefern maskierte `target_ids == -100`-Positionen den Wert 0 und werden trotzdem in den Mittelwert aufgenommen. Prompt- und Padding-Positionen verdünnen dadurch den PonderNet-Loss.

Die feste Baseline verwendet dagegen den Standard-Cross-Entropy-Mittelwert, der `-100` korrekt ignoriert. Dadurch sind die Loss-Werte der Baseline und von PonderNet nicht direkt vergleichbar.

Nächster Code-Schritt:

```python
valid = target_ids.ne(-100)
ce = F.cross_entropy(
    step_logits.reshape(-1, step_logits.size(-1)),
    target_ids.reshape(-1),
    reduction="none",
    ignore_index=-100,
).view(B, T)
step_loss = ce.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
```

Die Korrektur ist in [bdh_cq.py](../../src/model/bdh_cq.py) umgesetzt. Die neue Hilfsfunktion `compute_masked_cross_entropy_per_sample` mittelt nur über gültige Lösungstokens; vollständig maskierte Beispiele liefern einen Loss von 0. Die Regressionstests in [test_pondernet.py](../../tests/test_pondernet.py) laufen erfolgreich.

### Aktualisierter Handlungsbedarf

- [x] Teacher-Forced-Argmax und autoregressive Generierung auf demselben Beispiel vergleichen.
- [x] Erste Abweichung bei beiden Pfaden bestimmen.
- [x] Nachweisen, dass das Problem bereits beim ersten Lösungstoken besteht.
- [x] PonderNet-Loss so ändern, dass `-100`-Positionen nicht in den Mittelwert eingehen.
- [x] Einen Test für maskierten PonderNet-Loss ergänzen.
- [ ] Baseline- und PonderNet-Loss danach über dieselben aktiven Lösungstokens vergleichen.
- [ ] Den korrigierten 400-Step-Overfit-Lauf erneut ausführen.
- [ ] Erst bei deutlich besserem Teacher-Forced-Output weitere Trainingsläufe starten.

## Phase 3 – Successive Halving

Ziel: Gute Konfigurationen mit wachsendem Budget bestätigen.

- [ ] Die besten vier Konfigurationen aus Phase 2 mit 1.000 Steps erneut trainieren.
- [ ] Die besten zwei Konfigurationen mit 5.000 Steps trainieren.
- [ ] Die beste Konfiguration für den vollständigen Lauf bestimmen.
- [ ] Prüfen, ob die Rangfolge aus Phase 2 stabil bleibt.
- [ ] Prüfen, ob `expected_steps` mit wachsender Trainingsdauer sinnvoll sinkt oder sich stabilisiert.

Startbefehle. `<CONFIG_ID>` und die vier Hyperparameter werden durch die ausgewählte Konfiguration aus Phase 2 ersetzt:

```bash
# Top-4 mit 1.000 Steps
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.learning_rate=<LR> \
  --set model.params.latent_reasoning_steps=<R_MAX> \
  --set model.params.ponder_lambda_p=<LAMBDA_P> \
  --set model.params.ponder_beta=<BETA> \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set data.params.num_samples=1000 \
  --set trainer.max_steps=1000

# Top-2 mit 5.000 Steps
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.learning_rate=<LR> \
  --set model.params.latent_reasoning_steps=<R_MAX> \
  --set model.params.ponder_lambda_p=<LAMBDA_P> \
  --set model.params.ponder_beta=<BETA> \
  --set model.params.n_layer=3 \
  --set model.params.n_embd=128 \
  --set data.params.num_samples=1000 \
  --set trainer.max_steps=5000
```

## Phase 4 – Hardware- und Präzisionsvergleich

Ziel: Die Kosten des finalen Trainings reduzieren.

- [ ] Einen kurzen Lauf mit `dtype: float32`, `mixed_precision: false` messen.
- [ ] Einen kurzen Lauf mit `dtype: float16`, `mixed_precision: true` messen.
- [ ] Prüfen, ob Loss und Gradienten numerisch stabil bleiben.
- [ ] Die größte Batch-Größe bestimmen, die in den GPU-Speicher passt.
- [ ] Effektive Batch-Größe über Gradient Accumulation konstant halten.

Empfohlener Versuch für eine NVIDIA-T4/g4dn.xlarge:

```yaml
dtype: float16
mixed_precision: true
```

Startbefehle für einen 100-Step-Vergleich:

```bash
# FP32
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set dtype=float32 \
  --set mixed_precision=false \
  --set trainer.max_steps=100

# FP16 Mixed Precision
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set dtype=float16 \
  --set mixed_precision=true \
  --set trainer.max_steps=100
```

## Phase 5 – Finaler 9×9-Sudoku-Lauf

- [ ] Nur die beste(n) Konfiguration(en) aus Phase 3 verwenden.
- [ ] Vor dem Start die Laufzeit anhand eines 100-Step-Benchmarks schätzen.
- [ ] Vollständigen Lauf mit 10.000 Beispielen und 20 Epochen starten.
- [ ] Bestes Checkpoint anhand von `val/sudoku_cell_accuracy` sichern.
- [ ] Nach Abschluss zusätzlich Board-Accuracy, Validity-Rate und Inferenzschritte dokumentieren.
- [ ] Ergebnisse mit der festen-Reasoning-Baseline vergleichen.

Startbefehl. Die Platzhalter werden durch die beste Konfiguration aus Phase 3 ersetzt:

```bash
uv run python main.py train configs/bdh_cq_sudoku.yaml \
  --set model.params.enable_pondernet=true \
  --set model.params.learning_rate=<LR> \
  --set model.params.latent_reasoning_steps=<R_MAX> \
  --set model.params.ponder_lambda_p=<LAMBDA_P> \
  --set model.params.ponder_beta=<BETA> \
  --set data.params.num_samples=10000 \
  --set trainer.max_epochs=20 \
  --set dtype=float16 \
  --set mixed_precision=true
```

## Alternative Aufgaben, falls 9×9-Sudoku weiterhin zu teuer ist

In dieser Reihenfolge testen:

- [ ] 4×4-Sudoku
- [ ] 6×6-Sudoku
- [ ] kleine Pfadsuche/Maze-Aufgabe
- [ ] Sequenzkopieren mit Distraktoren
- [ ] Klammer-Matching oder kleine Sortieraufgabe

Die Aufgabe ist für PonderNet besonders geeignet, wenn Beispiele unterschiedlich viele iterative Schritte benötigen und die Lösung automatisch exakt überprüft werden kann.

## Abschlussbericht

Am Ende dieses Dokuments ergänzen:

```text
Beste Konfiguration:
Run:
Aufgabe:
Finale Cell-Accuracy:
Finale Board-Accuracy:
Finale Validity-Rate:
Durchschnittliche Expected Steps:
Trainingsdauer:
Vergleich zur festen Baseline:
Nächster sinnvoller Schritt:
```

- [ ] Alle tatsächlich durchgeführten Läufe sind in den Tabellen eingetragen.
- [ ] Alle nicht durchgeführten Läufe sind als offen erkennbar.
- [ ] Die beste Konfiguration ist begründet ausgewählt.
- [ ] Der finale Bericht ist ausgefüllt.

## Auswertung des korrigierten Overfit-Laufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-103138/`

### Technischer Status

- [x] Dataset-, Validator- und PonderNet-Regressionstests erfolgreich (`12 + 8 + 10` Tests).
- [x] Konfiguration erfolgreich validiert.
- [x] Baseline und PonderNet jeweils mit 400 Steps abgeschlossen.
- [x] Decoding- und Teacher-Forcing-Diagnosen ausgeführt.

### Ergebnisse

| Modell | finaler Train-Loss | finaler Val-Loss | Cell-Accuracy | Board-Accuracy | Validity | Parse-Rate | Expected Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed R=2 | 1.3993 | 1.3953 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | – |
| PonderNet R=4 | 1.4042 | 1.4005 | 0.1111 | 0.0000 | 0.0000 | 1.0000 | 2.784 |

Die Loss-Korrektur hat den erwarteten Effekt: Der PonderNet-Loss liegt nun in derselben Größenordnung wie der Fixed-R=2-Loss. Die frühere starke Differenz war daher tatsächlich durch die Verdünnung über maskierte Prompt-Tokens verursacht.

### Inhaltliche Auswertung

- Beide Modelle lösen kein einziges Validierungs-Sudoku vollständig.
- Die Cell-Accuracy bleibt bei `0.1111`, also ungefähr auf dem Niveau einer konstanten Ziffernvorhersage.
- Teacher-Forcing und autoregressives Decoding scheitern weiterhin beim ersten Lösungstoken (`first_mismatch=0`).
- Die Baseline erzeugt überwiegend `6`, PonderNet überwiegend `5`; das Modell memorisiert das Trainingsbeispiel auch nach 400 Steps nicht.
- PonderNet nutzt im Mittel `2.784` von maximal 4 Reasoning-Schritten. Es gibt damit ein plausibles Halteverhalten, aber noch keinen Qualitätsgewinn.
- Der Runner lädt für die Diagnose `best.pt`, das nach `val/sudoku_cell_accuracy` ausgewählt wird. Da diese Metrik konstant bzw. sehr flach bleibt, wurde bei der Baseline Step 40 und bei PonderNet Step 80 geladen, nicht der finale Step 400. Das muss bei einem echten Overfit-Test getrennt betrachtet werden.

### Entscheidung und nächster Handlungsbedarf

- [x] Maskierten PonderNet-Loss korrigiert und getestet.
- [x] Vergleichbarkeit der Loss-Werte zwischen Baseline und PonderNet wiederhergestellt.
- [x] Runner so erweitern, dass zusätzlich der finale Checkpoint (`last.pt`) gegen `best.pt` evaluiert wird.
- [x] Einen echten Ein-Beispiel-Overfit-Test mit deutlich höherem Budget durchführen.
- [x] Nachweisen, dass die Fixed-Baseline und PonderNet jeweils ein Trainingsbeispiel memorisieren können.
- [ ] Mit 4 Trainingsbeispielen prüfen, ob die Memorisation über ein einzelnes Beispiel hinaus skaliert.

Der nächste sinnvolle Schritt ist deshalb kein weiterer PonderNet-Sweep, sondern ein kontrollierter Ein-Beispiel-Test mit Auswertung von `best.pt` **und** dem finalen Checkpoint. Erst wenn die Baseline dabei die Lösung lernt, lohnt sich die weitere Untersuchung des PonderNet-Halteverhaltens.

## Auswertung des Ein-Beispiel-Overfit-Laufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-110350/`

### Ergebnisse

| Modell | Checkpoint | Step | Train-Loss | Val-Cell-Accuracy | Teacher-Forced exakt | Autoregressiv exakt |
|---|---|---:|---:|---:|---:|---:|
| Fixed R=2 | `best.pt` | 800 | – | 0.1605 | Ja | Ja |
| Fixed R=2 | `last.pt` | 2000 | 0.0021 | 0.0864 | Ja | Ja |
| PonderNet R=4 | `best.pt` | 100 | – | Nein | Nein |
| PonderNet R=4 | `last.pt` | 2000 | 0.0041 | 0.0741 | Ja | Ja |

Zusätzliche PonderNet-Metriken im finalen Checkpoint:

- Task-Loss: `0.00189`
- Gesamt-Loss: `0.00410`
- Expected Steps: `3.469` von maximal 4
- KL-Loss: `0.2203`

### Befund

- Der zentrale Overfit-Test ist bestanden: Sowohl Baseline als auch PonderNet können ein einzelnes Sudoku exakt memorisieren.
- Teacher-Forcing und autoregressives Decoding stimmen im erfolgreichen Fall vollständig mit der Ground Truth überein.
- Die korrigierte Loss-Maskierung funktioniert damit praktisch; ein grundlegender Target-/Logit-Alignment-Fehler liegt nicht mehr vor.
- `best.pt` ist für diesen Overfit-Test kein geeigneter Qualitätsindikator, weil es nach der Validierungsmetrik auf einem anderen Sudoku ausgewählt wird. Bei PonderNet wurde dadurch Step 100 geladen, obwohl `last.pt` bei Step 2000 das Trainingsbeispiel exakt löst.
- Die niedrige Validierungs-Cell-Accuracy ist in diesem Test erwartbar: Es wird auf einem separaten Sudoku validiert, während nur ein Trainingsbeispiel memorisiert wird.

### Aktualisierter Handlungsbedarf

- [x] Fixed-Baseline kann ein Trainingsbeispiel memorisieren.
- [x] PonderNet kann ein Trainingsbeispiel memorisieren.
- [x] Finalen Checkpoint zusätzlich zu `best.pt` evaluieren.
- [ ] Mit 4–8 Trainingsbeispielen prüfen, wie schnell Baseline und PonderNet memorisieren.
- [ ] Für den nächsten Vergleich `train/loss` und Teacher-Forced-Accuracy auf Trainingsbeispielen als Overfit-Metrik erfassen.
- [ ] Erst danach wieder kleine PonderNet-Hyperparameter-Sweeps starten.

Die Implementierung ist damit grundsätzlich funktionsfähig. Der nächste Engpass ist nicht mehr die Loss-Maskierung, sondern Generalisierung und die Auswahl einer passenden Validierungsmetrik für PonderNet.

## Auswertung des ersten Generalisierungstests vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-135258/`

Konfiguration: 64 Beispiele, 48 Training / 16 Validierung, 3.000 Steps, Fixed R=2 gegen PonderNet R=4 mit `ponder_beta=0.001` und `ponder_lambda_p=0.40`.

| Modell | Train-Loss bei Step 3000 | Bester Val-Loss | Beste Val-Cell-Accuracy | Board-Accuracy | Validity | Parse-Rate | Expected Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed R=2 | 0.3497 | 0.5442 | 0.5532 | 0.0000 | 0.0000 | 1.0000 | – |
| PonderNet R=4 | 0.1008 | 0.4119 | 0.5872 | 0.0000 | 0.0000 | 1.0000 | 1.067 |

### Interpretation

- PonderNet erzielt erstmals einen messbaren Vorteil auf der Validierungsmenge: höhere Cell-Accuracy (`0.5872` vs. `0.5532`) und niedrigerer Val-Loss (`0.4119` vs. `0.5442`).
- PonderNet benötigt dabei im Mittel nur `1.067` von 4 Reasoning-Schritten. Der Vorteil entsteht in diesem Lauf daher nicht durch maximale Rechentiefe, sondern durch frühes, gelerntes Halten.
- Die Board-Accuracy und Validity-Rate bleiben bei beiden Modellen bei 0. Die Cell-Accuracy ist somit ein frühes Lernsignal, aber noch kein vollständiger Sudoku-Erfolg.
- Die Diagnose der ersten vier Trainingsbeispiele ergibt für die Baseline `0/4` und für PonderNet `2/4` im finalen Checkpoint. Das ist nur ein Teilindikator, da insgesamt 48 Trainingsbeispiele verwendet wurden.

### Entscheidung und nächster Schritt

Der PonderNet-Kandidat `beta=0.001`, `lambda_p=0.40` wird für den nächsten größeren Lauf beibehalten. Als nächstes sollte der Datensatz auf 256 Beispiele vergrößert und die Validierungsmenge ebenfalls auf mindestens 32 Beispiele erhöht werden. Dabei sollten Cell-Accuracy, Val-Loss, Board-Accuracy, Validity-Rate und Expected Steps gemeinsam verfolgt werden.

Der Runner ist für den nächsten Skalierungstest auf `num_samples=8` konfiguriert. Das ergibt vier Trainings- und vier Validierungsbeispiele bei `max_steps=2000`; die vier Trainingsbeispiele werden über `best.pt` und `last.pt` diagnostiziert.

## Auswertung des Vier-Beispiele-Laufs vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-112516/`

### Ergebnisse

| Modell | Checkpoint | Step | Train-Loss | Val-Loss | Val-Cell-Accuracy | Erstes diagnostiziertes Trainingsbeispiel |
|---|---|---:|---:|---:|---:|---|
| Fixed R=2 | `best.pt` | 1900 | – | – | 0.2500 | exakt gelöst |
| Fixed R=2 | `last.pt` | 2000 | 0.0141 | 1.9025 | 0.2377 | exakt gelöst |
| PonderNet R=4 | `best.pt` | – | – | 0.1142 | Fehler beim ersten Token |
| PonderNet R=4 | `last.pt` | 2000 | 0.0726 | 2.5628 | 0.0833 | erster Fehler bei Token 108 |

PonderNet-Metriken im finalen Checkpoint:

- Task-Loss: `0.0706`
- Gesamt-Loss: `0.0726`
- Expected Steps: `3.985` von 4
- KL-Loss: `0.6206`

### Befund

- Der Lauf ist technisch vollständig erfolgreich; alle Tests und alle vier Trainings-/Diagnosephasen liefen durch.
- Die Fixed-Baseline kann mindestens das erste der vier Trainingsbeispiele exakt memorisieren.
- PonderNet ist deutlich besser als im vorherigen 400-Step-Lauf: Der Fehler liegt beim ersten Beispiel erst bei Token 108 statt bei Token 0. Vollständige Lösung wird nach 2.000 Steps aber noch nicht erreicht.
- Die PonderNet-Halting-Policy ist praktisch kollabiert: `3.985/4` bedeutet, dass fast immer alle Reasoning-Schritte ausgeführt werden. Adaptives frühes Stoppen findet in diesem Lauf nicht statt.
- Die Validierung bleibt auf separaten Beispielen schwach. Das ist für diesen Overfit-Test nicht das primäre Kriterium.
- Die aktuelle Diagnose prüft weiterhin nur `dataset.samples[0]`. Aus diesem Lauf kann daher noch nicht geschlossen werden, dass die Baseline alle vier Trainingsbeispiele memorisiert.

### Nächster Handlungsbedarf

- [x] Diagnose-Script auf alle Trainingsbeispiele erweitern und pro Beispiel Teacher-Forced-/autoregressiven Trefferstatus ausgeben.
- [ ] Baseline und PonderNet über die vier Trainingsbeispiele vergleichen, bevor weitere Hyperparameter getestet werden.
- [ ] Bei PonderNet prüfen, ob `ponder_lambda_p=0.20` und `ponder_beta=0.010` die maximale Rechentiefe erzwingen.
- [ ] Falls alle vier Beispiele memorisiert werden: mit acht oder mehr Trainingsbeispielen die Generalisierung untersuchen.

## Auswertung der Mehrfach-Sample-Diagnose vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-121240/`

### Trefferquoten über alle vier Trainingsbeispiele

| Modell | Checkpoint | Teacher-Forced exakt | Autoregressiv exakt | Erste Abweichungen |
|---|---|---:|---:|---|
| Fixed R=2 | `best.pt` | 4/4 (100 %) | 4/4 (100 %) | keine |
| Fixed R=2 | `last.pt` | 4/4 (100 %) | 4/4 (100 %) | keine |
| PonderNet R=4 | `best.pt` | 0/4 (0 %) | 0/4 (0 %) | Token 22, 14, 102, 2 |
| PonderNet R=4 | `last.pt` | 0/4 (0 %) | 0/4 (0 %) | Token 104, 2, 128, 8 |

### Interpretation

- Die Fixed-Baseline memorisiert alle vier Trainingsbeispiele vollständig. Target-Tokenisierung, Loss-Maske und Decoding sind damit für diesen Test bestätigt.
- PonderNet scheitert trotz 2.000 Steps bei allen vier Beispielen. Das Problem ist jetzt spezifisch für die PonderNet-Optimierung.
- PonderNet erreicht im finalen Lauf einen Task-Loss von `0.0963` und einen Gesamt-Loss von `0.1002`; die Baseline liegt bei `0.0142`.
- `expected_steps=3.857` zeigt, dass PonderNet fast immer die maximale Tiefe verwendet. Das Modell erhält zwar Rechenkapazität, lernt aber die Lösungstrajektorie nicht so effektiv wie die fixe Baseline.
- Der Validator zeigt für die Baseline `0.2593` Cell-Accuracy und für PonderNet `0.1327`; diese Werte sind für die Trainings-Memorisation nachrangig, bestätigen aber den Nachteil von PonderNet in diesem Lauf.

### Entscheidung

Weitere allgemeine Hyperparameter-Sweeps sind noch nicht sinnvoll. Der Runner ist jetzt auf diesen fokussierten Vergleich der PonderNet-Regularisierung angepasst:

- Fixed-R=2-Baseline als Referenz.
- PonderNet R=4 mit `ponder_beta=0.0`, `lambda_p=0.20`.
- PonderNet R=4 mit `ponder_beta=0.001`, `lambda_p=0.20`.
- PonderNet R=4 mit `ponder_beta=0.001`, `lambda_p=0.40`.
- Pro Variante 4/4-Trainings-Treffer für `best.pt` und `last.pt` erfassen.

## Auswertung des PonderNet-Regularisierungs-Sweeps vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-124205/`

| Variante | Checkpoint | Train-Loss | Val-Cell-Accuracy | Expected Steps | Teacher-Forced | Autoregressiv |
|---|---|---:|---:|---:|---:|---:|
| Fixed R=2 | `last.pt` | 0.0142 | 0.2685 | – | 4/4 | 4/4 |
| PonderNet `beta=0.0`, `lambda=0.20` | `best.pt` | – | 0.2500 | – | 2/4 | 2/4 |
| PonderNet `beta=0.0`, `lambda=0.20` | `last.pt` | 0.0178 | 0.1944 | 1.743 | 4/4 | 4/4 |
| PonderNet `beta=0.001`, `lambda=0.20` | `best.pt` | – | 0.3056 | – | 2/4 | 2/4 |
| PonderNet `beta=0.001`, `lambda=0.20` | `last.pt` | 0.0179 | 0.2500 | 2.181 | 4/4 | 4/4 |
| PonderNet `beta=0.001`, `lambda=0.40` | `best.pt` | – | 0.2901 | – | 4/4 | 4/4 |
| PonderNet `beta=0.001`, `lambda=0.40` | `last.pt` | 0.0185 | 0.2901 | 1.925 | 4/4 | 4/4 |

### Interpretation

- Die PonderNet-Varianten können bei 2.000 Steps alle vier Trainingsbeispiele memorisieren. Der frühere `0/4`-Befund war damit ein Optimierungs-/Hyperparameterproblem und kein grundsätzlicher PonderNet-Fehler.
- `ponder_beta=0.0` reduziert die erwartete Rechentiefe am stärksten auf `1.743`, erreicht aber trotzdem `4/4` im finalen Checkpoint.
- `ponder_beta=0.001`, `lambda_p=0.20` erzielt die beste gemessene Validierungs-Cell-Accuracy (`0.3056` beim besten Checkpoint), aber nur `2/4` Trainings-Treffer in diesem Checkpoint.
- `ponder_beta=0.001`, `lambda_p=0.40` ist der robusteste Overfit-Kandidat: `best.pt` und `last.pt` lösen beide `4/4`; zugleich bleibt die erwartete Tiefe mit `1.925` deutlich unter 4.
- Die Fixed-Baseline bleibt beim Trainings-Loss etwas besser, aber PonderNet erreicht mit geeigneter Regularisierung denselben vollständigen Overfit.

### Entscheidung

Für weitere Tests wird `ponder_beta=0.001`, `ponder_lambda_p=0.40`, `R=4` als PonderNet-Kandidat verwendet. Der nächste Test ist auf 256 Beispiele mit 32 Validierungsbeispielen, 5.000 Steps und Validierung alle zwei Epochen eingestellt. Damit werden Generalisierung und die tatsächliche adaptive Rechentiefe verglichen.

## Auswertung des 256-Beispiele-Generalisierungstests vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-145317/`

Konfiguration: 256 Beispiele, 224 Training / 32 Validierung, 5.000 Steps, Fixed R=2 gegen PonderNet R=4 mit `ponder_beta=0.001` und `ponder_lambda_p=0.40`.

| Modell | Train-Loss bei Step 5000 | Bester Val-Loss | Beste Val-Cell-Accuracy | Board-Accuracy | Validity | Parse-Rate | Expected Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed R=2 | 0.1028 | 0.1810 | 0.6998 | 0.0000 | 0.0000 | 1.0000 | – |
| PonderNet R=4 | 0.1019 | 0.1835 | 0.6863 | 0.0000 | 0.0000 | 1.0000 | 1.020 |

### Interpretation

- Bei 256 Beispielen verschwindet der kleine PonderNet-Vorteil des 64-Beispiele-Laufs: Die Fixed-Baseline liegt bei der Cell-Accuracy leicht vorne (`0.6998` vs. `0.6863`).
- Die Val-Loss-Werte sind nahezu gleich (`0.1810` vs. `0.1835`). Der Unterschied ist daher klein und kein belastbarer Nachweis für eine klare Qualitätsüberlegenheit der Baseline.
- Beide Modelle erreichen weiterhin keine vollständigen oder gültigen Sudoku-Boards. Cell-Accuracy ist hier ein Lernsignal, aber noch kein End-to-End-Erfolg.
- PonderNet hält sehr früh (`1.020/4` Schritte). Der Kandidat spart also Rechenaufwand, zeigt in diesem Lauf aber keinen Qualitätsgewinn.
- Die Diagnose der ersten vier Trainingsbeispiele ist bei dieser größeren Datenmenge nur ein Stichprobentest und kein vollständiger Trainings-Overfit-Nachweis.

### Entscheidung

Der aktuelle PonderNet-Kandidat ist technisch stabil und rechnet adaptiv, aber ein Qualitätsvorteil gegenüber der Fixed-Baseline ist bei 256 Beispielen nicht nachgewiesen. Der nächste Runner-Schritt ist deshalb ein Multi-Seed-Vergleich mit den Seeds `42`, `43` und `44`. Weitere breite Hyperparameter-Sweeps sind vorerst nicht gerechtfertigt.

## Auswertung des Multi-Seed-Tests vom 20.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-173454/`

Konfiguration: 256 Beispiele, 32 Validierungsbeispiele, 5.000 Steps pro Lauf. Verglichen wurden Fixed R=2 und PonderNet R=4 mit `ponder_beta=0.001` und `ponder_lambda_p=0.40`.

| Seed | Fixed R=2: beste Val-Cell-Accuracy | PonderNet: beste Val-Cell-Accuracy | PonderNet Expected Steps |
|---:|---:|---:|---:|
| 42 | 0.6806 | **0.7253** | 1.014 |
| 43 | 0.5270 | **0.7296** | 1.013 |
| 44 | 0.6759 | 0.6150 | 1.016 |
| **Mittelwert** | **0.6278** | **0.6899** | **1.014** |

Die geschätzte Seed-Streuung der Cell-Accuracy beträgt ungefähr `0.087` für die Baseline und `0.065` für PonderNet. PonderNet ist in diesem kleinen Multi-Seed-Test damit im Mittel besser und etwas stabiler.

### Einschränkungen

- Bei allen sechs Läufen bleiben Board-Accuracy und Validity-Rate bei `0`. Die Verbesserung betrifft bisher nur die Cell-Accuracy.
- Die Teacher-Forcing-Diagnose der ersten vier Trainingsbeispiele ergibt bei allen sechs Läufen `0/4`; daraus folgt, dass noch keine vollständige Memorisation dieser Stichprobe vorliegt.
- Die Ergebnisse sind ein gutes Signal, aber wegen nur drei Seeds noch kein endgültiger Nachweis einer allgemeinen Überlegenheit.

### Entscheidung

Der PonderNet-Kandidat wird für weitere Untersuchungen beibehalten. Der Runner ist jetzt auf einen fokussierten 10.000-Step-Lauf mit Seed 42 eingestellt, um zu prüfen, ob die höhere Cell-Accuracy in vollständige Sudoku-Boards übergeht. Weitere Regularisierungs-Sweeps sind zunächst nicht nötig.

## Auswertung des 10.000-Step-Laufs vom 21.08.2026

Run-Batch: `runs/sudoku-overfit-20260820-215315/`

| Modell | Train-Loss bei Step 10000 | Bester Val-Loss | Beste Val-Cell-Accuracy | Board-Accuracy | Validity | Expected Steps | Trainingsdiagnose (4 Samples) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed R=2 | 0.0128 | 0.1944 | 0.7087 | 0.0000 | 0.0000 | – | 3/4 |
| PonderNet R=4 | 0.0124 | 0.1955 | **0.7203** | 0.0000 | 0.0000 | 1.010 | 3/4 |

### Interpretation

- Die längere Trainingsdauer verbessert die Memorisation deutlich: Beide Modelle erreichen `3/4` exakt gelöste diagnostizierte Trainingsbeispiele statt `0/4` im vorherigen 5.000-Step-Lauf.
- PonderNet liegt bei der besten Val-Cell-Accuracy leicht vor der Baseline (`0.7203` vs. `0.7087`). Der Abstand ist klein, aber konsistent mit dem vorherigen Multi-Seed-Signal.
- Die Val-Loss-Werte sind praktisch gleich (`0.1955` vs. `0.1944`).
- Beide Modelle erzeugen weiterhin kein vollständig korrektes oder gültiges Validierungs-Board. Mehr Steps allein haben das End-to-End-Problem bisher nicht gelöst.
- PonderNet hält weiterhin sehr früh (`1.010/4` Schritte) und erreicht den Cell-Accuracy-Vorteil ohne zusätzliche Rechentiefe.

### Entscheidung

Ein weiterer reiner Lauf mit mehr Steps ist vorerst nicht der beste nächste Schritt. Die Lernsignale sind gut, aber die Board-Accuracy bleibt 0. Als nächstes sollten die Validierungs-Outputs auf Near-Misses analysiert werden: Anzahl falscher Zellen pro Puzzle, Positionen der Fehler und ob die Ausgabe formal eine vollständige Sudoku-Lösung enthält. Danach lässt sich entscheiden, ob Decoder, Validator oder Modellkapazität der eigentliche Engpass ist.

Das Script [diagnose_sudoku_near_misses.py](../../scripts/diagnose_sudoku_near_misses.py) ist dafür ergänzt. Der Runner wertet damit für `best.pt` und `last.pt` jeweils 16 Validierungsbeispiele aus und protokolliert pro Puzzle Cell-Accuracy, Anzahl falscher Zellen, Parse-Status, Validity und exakten Board-Treffer.

## Auswertung der Near-Miss-Diagnose vom 21.08.2026

Run-Batch: `runs/sudoku-overfit-20260821-072338/`

Der Runner lief vollständig durch: Unit-Tests, beide Trainingsläufe und alle Teacher-Forcing-/Near-Miss-Diagnosen meldeten `PASS`. Die Near-Miss-Diagnose wurde auf 16 Validierungsbeispiele pro Checkpoint angewendet.

| Modell | Checkpoint | Samples | Cell-Accuracy | Falsche Zellen/Puzzle | Parse-Rate | Validity-Rate | Board-Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Fixed R=2 | `best.pt` | 16 | 0.7153 | 23.06 | 1.0000 | 0.0000 | 0.0000 |
| Fixed R=2 | `last.pt` | 16 | 0.6960 | 24.62 | 1.0000 | 0.0000 | 0.0000 |
| PonderNet R=4 | `best.pt` | 16 | **0.7261** | **22.19** | 1.0000 | 0.0000 | 0.0000 |
| PonderNet R=4 | `last.pt` | 16 | **0.7230** | **22.44** | 1.0000 | 0.0000 | 0.0000 |

Die Trainingsmetriken bestätigen das bisherige Bild: Die Fixed-Baseline erreicht beim besten Checkpoint eine Val-Cell-Accuracy von `0.7087`, PonderNet `0.7014`; der Near-Miss-Check liefert für PonderNet trotzdem den leicht besseren Mittelwert. PonderNet hält weiterhin bei etwa `1.013` erwarteten Schritten und nutzt damit praktisch nur einen Reasoning-Schritt.

### Befund

- Alle 64 geprüften Ausgaben (2 Modelle × 2 Checkpoints × 16 Beispiele) sind parsebar. Ein Fehler in Ausgabeformat, Parser oder fehlender Grid-Länge ist daher als Hauptursache unwahrscheinlich.
- Kein einziges ausgegebenes Board ist gültig oder exakt korrekt. Die Modelle produzieren vollständige Gitter, aber mit vielen Duplikaten in Zeilen, Spalten oder Blöcken.
- Die Fehler sind breit verteilt: PonderNet `best.pt` liegt im Mittel bei rund 22 falschen Zellen pro Puzzle, nicht bei nur einem einzelnen Off-by-one-Fehler. Das spricht gegen ein reines Checkpoint- oder Stoppkriteriumsproblem.
- PonderNet ist in diesem Lauf bei Cell-Accuracy und Near-Miss-Abstand leicht besser als die Baseline, aber der Vorteil reicht nicht bis zur Sudoku-Gültigkeit. Die adaptive Rechentiefe ist außerdem faktisch kollabiert auf `≈1/4` Schritte.

### Entscheidung und nächster Handlungsbedarf

Ein weiterer Lauf mit nur mehr Steps ist nicht priorisiert. Als nächstes soll der Einfluss des Decoders getrennt werden:

- [x] Near-Miss-Auswertung für 16 Validierungsbeispiele ergänzt und ausgeführt.
- [ ] Für dieselben Checkpoints eine constrained-/validity-aware Decoding-Variante testen, die bei jeder Position nur noch zulässige Sudoku-Ziffern zulässt.
- [ ] Unconstrained gegen constrained Decoding mit identischen Logits vergleichen: Cell-Accuracy, Parse-Rate, Validity-Rate und Board-Accuracy.
- [ ] Teacher-Forcing und autoregressives Decoding auf denselben Validierungsbeispielen gegenüberstellen, um Exposure Bias von fehlender Sudoku-Repräsentation zu trennen.
- [ ] Erst wenn constrained Decoding gültige Boards erzeugt, PonderNet-Tiefe (`R=4` versus höhere Tiefe) erneut untersuchen.

Die Near-Miss-Daten zeigen damit: Das Projekt ist nicht an der Ausgabeparsing-Schicht blockiert. Der nächste informative Test ist ein kontrollierter Decoder-Vergleich, nicht ein weiterer breiter Hyperparameter-Sweep.
