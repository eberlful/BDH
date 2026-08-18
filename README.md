# BDH

Configurable PyTorch language-model training environment with YAML-defined
models, data modules, callbacks, loggers, and trainers.

## Quick start

```bash
uv run python main.py validate configs/tiny_shakespeare.yaml
uv run python main.py train configs/tiny_shakespeare.yaml
```

The example downloads Tiny Shakespeare if it is not already present. New runs
are created under `runs/` and contain the resolved YAML configuration, logs,
TensorBoard events, and checkpoints.

Override configuration values without editing YAML:

```bash
uv run python main.py train configs/tiny_shakespeare.yaml \
  --set trainer.max_epochs=20 \
  --set model.params.d_model=384 \
  --set compile=true
```

### Running in the background (nohup)

Run training detached in the background using `nohup`:

```bash
nohup uv run python main.py train configs/tiny_shakespeare.yaml > train.log 2>&1 &
```

- **Monitor progress:** `tail -f train.log`
- **Check process:** `ps aux | grep "python main.py"`
- **Stop training:** `kill <PID>`


Resume a run from its latest complete checkpoint:

```bash
uv run python main.py resume runs/20260812-120000-a1b2c3 \
  --set trainer.max_epochs=40
```

Generate text from the best checkpoint of a completed text run:

```bash
uv run python main.py generate runs/20260812-120000-a1b2c3 "To be or not to be" \
  --max-tokens 80
```

Generation requires `checkpoints/best.pt` and a text data module with a
tokenizer. The token limit counts newly generated tokens; the prompt is
included in the printed output.

For Sudoku runs, pass an 81-character grid using `0` for blank cells. The
command prints the raw serialized Sudoku token sequence:

```bash
uv run python main.py generate runs/20260812-120000-a1b2c3 \
  530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
  --max-tokens 103
```

Use `2 * blank_count + 1` for `--max-tokens` to request all position/value
pairs and the end token.

For Sudoku Chain-of-Thought (CoT) runs with GPT-2 tokenization:

```bash
uv run python main.py train configs/sudoku_cot.yaml
```

Configure the reasoning path via `data.params.reasoning_mode`:
- `full` (default): Trains on both step-by-step thinking traces and the final solution.
- `none`: Trains directly on prompt-to-solution mappings without intermediate reasoning steps.
- `context_only`: Includes reasoning steps in prompt context but masks loss, training only on solution tokens.

```bash
uv run python main.py train configs/sudoku_cot.yaml \
  --set data.params.reasoning_mode=none \
  --set data.params.context_length=256
```

To generate predictions from a checkpoint (the CLI automatically matches the trained reasoning mode):

```bash
uv run python main.py generate runs/20260812-120000-a1b2c3 \
  530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
  --max-tokens 600
```

### Generating and testing random Sudoku puzzles

Use `scripts/generate_random_start_sudoku.py` to create random puzzles with configurable difficulty (`low`, `medium`, `high`) or clue counts:

```bash
# Generate an 81-character puzzle string (medium difficulty / 30 clues)
uv run python scripts/generate_random_start_sudoku.py --difficulty medium

# Print visual 9x9 ASCII grids for both the puzzle and its ground-truth solution
uv run python scripts/generate_random_start_sudoku.py --difficulty high --grid --solution

# Generate a Chain-of-Thought (CoT) text prompt
uv run python scripts/generate_random_start_sudoku.py --cot
```

Test a trained model checkpoint on a freshly generated Sudoku in a single command:

```bash
# Standard discrete token Sudoku run:
uv run python main.py generate runs/20260812-120000-a1b2c3 \
  $(uv run python scripts/generate_random_start_sudoku.py --difficulty medium) \
  --max-tokens 103

# Sudoku Chain-of-Thought (CoT) run:
uv run python main.py generate runs/20260812-120000-a1b2c3 \
  $(uv run python scripts/generate_random_start_sudoku.py --difficulty medium) \
  --max-tokens 600
```

For BDH-CQ (In-Context Recurrent Memory & Latent Workspace Reasoning) runs:

```bash
uv run python main.py train configs/bdh_cq.yaml
```

Configure recurrent latent reasoning steps and deep supervision loss schedules:
- `model.params.latent_reasoning_steps`: Number of recurrent passes ($R$) in the latent workspace without discrete token verbalization (supports dynamic test-time compute scaling).
- `model.params.loss_schedule`: Multi-step deep supervision weighting across intermediate reasoning steps (`ramp`, `uniform`, or `final_only`).

```bash
uv run python main.py train configs/bdh_cq.yaml \
  --set model.params.latent_reasoning_steps=4 \
  --set model.params.loss_schedule=ramp
```

For WikiText language modeling runs (zero-disk streaming from Hugging Face):

```bash
uv run python main.py train configs/wikitext.yaml
```

Override dataset configurations, context lengths, or reduce dataset size with sample limits:
- `data.params.max_train_samples`: Maximum training samples to stream into memory (e.g. 3,670 for ~10% of WikiText-2, or 180,000 for ~10% of WikiText-103).
- `data.params.max_val_samples`: Maximum validation samples to stream into memory.

```bash
uv run python main.py train configs/wikitext.yaml \
  --set data.params.dataset_config=wikitext-103-raw-v1 \
  --set data.params.max_train_samples=180000 \
  --set data.params.context_length=512
```

For TinyStories synthetic narrative reasoning runs:

```bash
uv run python main.py train configs/tiny_stories.yaml
```

Configure sample limits and memory streaming:
- `data.params.max_train_samples`: Maximum training stories to stream into memory (default: 100,000 to keep RAM <200MB; e.g. 50,000 for ~2.5% of TinyStories).
- `data.params.max_val_samples`: Maximum validation stories to stream into memory.
- `data.params.in_memory`: Set to `true` (default) for zero-disk streaming without writing Arrow files to disk.

```bash
uv run python main.py train configs/tiny_stories.yaml \
  --set data.params.max_train_samples=50000 \
  --set trainer.max_epochs=5
```

### Optimizer configuration (AdamW, Adafactor)

Models support configurable optimizers via `model.params.optimizer` (default: `adamw`) and `model.params.optimizer_params`:

- `adamw` (default): Standard decoupled weight-decay AdamW.
- `adafactor`: Sublinear memory optimizer that factorizes 2D second-moment states into rank-1 row/column statistics ($O(d_1 + d_2)$ instead of $O(d_1 \cdot d_2)$), reducing optimizer state memory by 50% to 75% natively in PyTorch on Apple Silicon (MPS), CPU, and CUDA.

Configure in YAML:

```yaml
model:
  name: bdh_cq
  params:
    optimizer: adafactor # default is 'adamw'
    learning_rate: 0.0003
    weight_decay: 0.01
    optimizer_params:
      clip_threshold: 1.0
      beta1: null # null/0.0 disables 1st momentum for maximum memory savings
```

Or override via CLI flags:

```bash
uv run python main.py train configs/bdh_cq.yaml \
  --set model.params.optimizer=adafactor \
  --set model.params.learning_rate=0.001
```

The original BDH architecture is registered as `bdh`; the contextual-query recurrent reasoning architecture is registered as `bdh_cq`; and the configurable causal GPT reference model is registered as `gpt_model` (with `bdh_transformer` as alias). Custom components can register themselves from Python modules listed in the YAML `plugins` list.

Run the tests with:

```bash
uv run python -m unittest discover -v
```

