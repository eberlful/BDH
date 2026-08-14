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
  --set model.params.d_model=384
```

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

The original BDH architecture is registered as `bdh`; the configurable causal
Transformer reference model is registered as `bdh_transformer`. Custom
components can register themselves from Python modules listed in the YAML
`plugins` list.

Run the tests with:

```bash
uv run python -m unittest discover -v
```
