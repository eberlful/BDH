# 02 — Checkpoint Callback Optimization Direction Support

**What to build:** Extend `CheckpointCallback` to support optimizing metrics where higher values represent better performance (e.g. accuracy, validity rate, score) in addition to minimization metrics (e.g. loss, perplexity), with automatic mode detection.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `CheckpointCallback` accepts a `mode: str = "auto"` parameter supporting `"auto"`, `"min"`, and `"max"`.
- [x] In `"auto"` mode, metric names containing `loss`, `err`, `error`, `perplexity` default to minimization (`"min"`), while metric names containing `acc`, `accuracy`, `rate`, `score`, `reward`, `validity` default to maximization (`"max"`).
- [x] When monitoring a maximization metric with `save_best: true`, `best.pt` is updated only when the metric is strictly greater than (or equal to) previous values.
- [x] Callback state serialization and restoration properly preserve the metric and direction.
- [x] Unit tests verify `"auto"`, `"min"`, and `"max"` behaviors across different monitored metrics.

