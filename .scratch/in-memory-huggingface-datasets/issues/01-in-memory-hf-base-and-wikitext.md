# 01 — In-Memory Hugging Face Base DataModule and WikiText Integration

**What to build:** An in-memory streaming text data module base class and a dedicated WikiText data module that streams text directly into RAM tensors with zero disk cache files, formats contiguous token blocks with `<|endoftext|>` article boundaries, resolves official validation splits, and provides unit tests verifying zero disk footprint and accurate token batching.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `HuggingFaceTextDataModule` base class supports `in_memory=True` with `streaming=True` and disabled disk caching (`datasets.disable_caching()`), compiling text streams directly into an in-memory `TokenBlockDataset`.
- [ ] `HuggingFaceTextDataModule` supports `in_memory=False` for standard disk-cached Arrow storage.
- [ ] `WikiTextDataModule` registered under `"wikitext"` defaults to `dataset_config="wikitext-2-raw-v1"`, accepts `wikitext-103-raw-v1`, filters whitespace-only lines, and inserts `<|endoftext|>` on document/article boundaries.
- [ ] Resolves official `split="validation"` from Hugging Face by default, falling back to `validation_fraction` if only a single split is present.
- [ ] Unit tests verify dataset token block generation, split separation, batching, and zero disk writes when `in_memory=True`.
