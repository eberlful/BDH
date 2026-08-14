# 03 — Training Configurations, ADR Documentation, and End-to-End CLI Verification

**What to build:** Ready-to-run configuration presets for WikiText and TinyStories, formal ADR 0003 documenting the zero-disk streaming architecture and memory bounds, updated README instructions, and end-to-end multi-step training CLI verification.

**Blocked by:** 01 — In-Memory Hugging Face Base DataModule and WikiText Integration, 02 — In-Memory TinyStories Integration and Memory Safeguards

**Status:** ready-for-agent

- [ ] Create `configs/wikitext.yaml` pre-configured for `bdh_transformer` + `wikitext` (`wikitext-2-raw-v1`, `in_memory: true`, `context_length: 256`, `batch_size: 32`).
- [ ] Create `configs/tiny_stories.yaml` pre-configured for `bdh_transformer` + `tiny_stories` (`max_train_samples: 100000`, `in_memory: true`, `context_length: 256`, `batch_size: 32`).
- [ ] Record ADR `docs/adr/0003-in-memory-huggingface-datasets.md` detailing the zero-disk streaming design and 16GB memory constraints.
- [ ] Update `README.md` with CLI training examples and configuration flags for WikiText and TinyStories.
- [ ] End-to-end integration tests verify multi-step training executions using `main.py train` across both datasets.
