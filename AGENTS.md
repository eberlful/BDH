
## Local executions

This device has 16GB of unified memory. Before you start a training, think about the necessary memory and whether it fits. It's likely that you need to reduce the model size or the batch size.

### Code execution

To execute code use uv.

```bash
uv run python your_script.py
```

## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses canonical triage label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repository layout. See `docs/agents/domain.md`.
