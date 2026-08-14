# 0003. Zero-Disk In-Memory Hugging Face Dataset Streaming and Memory Safeguards

## Context

Standard Hugging Face `datasets` workflows download and cache Arrow tables to disk (by default under `~/.cache/huggingface/datasets`). For large corpora like TinyStories or systems running in containerized/ephemeral environments or local development machines with constrained storage or unified memory (such as 16GB Apple Silicon devices), disk caching introduces significant disk overhead, redundant I/O, and disk fragmentation.

Furthermore, streaming large corpora directly without bounds can exhaust system RAM if unbounded collections are accumulated during tokenization.

## Decision

We introduce an in-memory streaming data architecture supporting Hugging Face datasets without caching to disk:

1. **Zero-Disk Streaming Core (`HuggingFaceTextDataModule`)**:
   - Registered under `"hf_text"`.
   - When `in_memory: true` (the default), invokes `datasets.disable_caching()` and requests `streaming=True` from `datasets.load_dataset()`.
   - Streams raw text records directly over the network, tokenizes them on-the-fly, and appends token IDs into contiguous memory buffers.
   - When `in_memory: false`, falls back to standard cached Arrow dataset loading.

2. **Specialized Dataset Modules**:
   - `"wikitext"` (`WikiTextDataModule`): Configured for Salesforce/wikitext variants (`wikitext-2-raw-v1`, `wikitext-103-raw-v1`). Filters empty and whitespace lines and uses regex matching on top-level headings (`= <title> =`) to insert `<|endoftext|>` article boundaries.
   - `"tiny_stories"` (`TinyStoriesDataModule`): Configured for `roneneldan/TinyStories`. Inserts `<|endoftext|>` boundaries after each narrative and enforces default memory safety limits.

3. **Memory Limits and Safeguards**:
   - Configurable `max_train_samples` (default: 100,000 for `tiny_stories`) and `max_val_samples` (default: 10,000 for `tiny_stories`).
   - Caps total token memory consumption to well under 200MB in RAM for 100k stories (~20M tokens), easily fitting within the 16GB host memory budget while providing sufficient training diversity.

4. **Integration with Trainer Pipeline**:
   - Token streams are packaged into `TokenBlockDataset` instances providing unified non-overlapping blocks of `(input_ids, target_ids)` matching `context_length`.
   - Compatible with `bdh_transformer`, `bdh`, and `bdh_cq` models via registry instantiation and CLI YAML configs (`configs/wikitext.yaml`, `configs/tiny_stories.yaml`).

## Consequences

- Zero local disk space consumed in `~/.cache/huggingface/` when `in_memory: true`.
- Fast dataset initialization time since only the required number of samples are streamed and tokenized into RAM.
- Predictable and bounded memory usage on 16GB memory devices.
- Seamless configuration via YAML and command-line overrides (`--set data.params.max_train_samples=...`).
