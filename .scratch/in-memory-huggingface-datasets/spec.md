Status: ready-for-agent

# Spec: In-Memory Hugging Face Datasets (WikiText & TinyStories)

## Problem Statement

Researchers and developers training language models need to train and benchmark on standard text corpora such as WikiText and TinyStories using Hugging Face's `datasets` ecosystem. However, on memory-constrained devices (such as machines with 16GB of unified memory and limited or restricted local disk space), default Hugging Face data loading writes hundreds of megabytes or gigabytes of Arrow and raw cache files to the local disk (`~/.cache/huggingface/datasets`). Furthermore, there are no built-in data modules in this repository for Hugging Face streaming text datasets, forcing developers to rely on local file-based datasets (like TinyShakespeare).

## Solution

Introduce a unified, zero-disk Hugging Face causal text data module framework with dedicated `wikitext` and `tiny_stories` data modules:
1. **Zero-Disk In-Memory Ingestion (`in_memory=True`)**: By default, datasets stream over the network using `load_dataset(..., streaming=True)` with disk caching explicitly disabled (`datasets.disable_caching()`). Raw text examples are tokenized in memory using `tiktoken` (standardizing on `gpt2`) and compiled directly into contiguous in-memory `TokenBlockDataset` tensors in RAM, writing 0 bytes to the local filesystem.
2. **Configurable Local Cache Fallback (`in_memory=False`)**: For execution environments where disk caching is desired across training runs, users can set `in_memory=False` to use standard disk-backed Arrow tables.
3. **Dedicated WikiText Support (`WikiTextDataModule` / `@DATA_REGISTRY.register("wikitext")`)**:
   - Defaults to `wikitext-2-raw-v1` for fast iterations, while allowing configuration to `wikitext-103-raw-v1`.
   - Filters blank/whitespace lines, preserves paragraph breaks, and delimits document/article boundaries with `<|endoftext|>`.
4. **Dedicated TinyStories Support (`TinyStoriesDataModule` / `@DATA_REGISTRY.register("tiny_stories")`)**:
   - Ingests `roneneldan/TinyStories`.
   - Supports memory safeguards with configurable `max_train_samples` (default `100_000` to respect 16GB unified memory) and `max_val_samples` (default `10_000`), or `None` for full load.
   - Delimits individual stories with `<|endoftext|>`.
5. **Standardized Preset Configs**: Provide `configs/wikitext.yaml` and `configs/tiny_stories.yaml` ready for CLI training.

## User Stories

1. As an ML researcher, I want to configure `data.name: "wikitext"` in my YAML config, so that I can train language models on standard WikiText benchmarks without downloading and saving dataset files to my local hard drive.
2. As an ML researcher, I want `wikitext` to default to `dataset_config: "wikitext-2-raw-v1"`, so that development and smoke runs initialize quickly without excessive memory or bandwidth usage.
3. As an ML researcher, I want to configure `dataset_config: "wikitext-103-raw-v1"` via YAML or CLI overrides (`--set data.params.dataset_config=wikitext-103-raw-v1`), so that I can scale up training to the full 103-million token corpus.
4. As an ML researcher, I want to configure `data.name: "tiny_stories"` in my YAML config, so that I can train models on synthetic narrative reasoning datasets.
5. As an ML engineer on a memory-constrained machine (16GB RAM), I want `tiny_stories` to support a `max_train_samples` parameter (defaulting to a safe `100_000` samples), so that dataset loading does not exhaust available system memory.
6. As an ML engineer on a high-memory cluster or remote instance, I want to set `max_train_samples: null` and `in_memory: false`, so that I can utilize local disk caching and load the full 2.1-million story corpus.
7. As a developer, I want all Hugging Face text data modules to default to `in_memory: true` with zero disk footprint, so that automated tests and local runs remain clean and reproducible without residual cache files.
8. As a developer, I want consecutive stories and articles to be separated with `<|endoftext|>`, so that the causal language model learns proper document boundary handling.
9. As a practitioner, I want both data modules to automatically resolve official validation splits from Hugging Face (`split="validation"`), falling back to `validation_fraction` splitting if a custom single-split dataset is provided.
10. As a developer writing automated tests, I want fast, mockable in-memory dataset tests that verify token block generation, split assignment, batching, and zero disk writes.

## Implementation Decisions

- **Base Class Architecture**:
  - Implement `HuggingFaceTextDataModule(BaseDataModule)` registered under `hf_text`.
  - Subclasses: `WikiTextDataModule(HuggingFaceTextDataModule)` registered under `wikitext` and `TinyStoriesDataModule(HuggingFaceTextDataModule)` registered under `tiny_stories`.
- **Zero-Disk / In-Memory Ingestion Mechanism**:
  - Parameter: `in_memory: bool = True`.
  - When `in_memory=True`:
    - Call `datasets.disable_caching()`.
    - Call `datasets.load_dataset(dataset_name, dataset_config, split=..., streaming=True)`.
    - Iterate stream, extract the text field (`"text"`), tokenize tokens in chunks using `tiktoken`, and pack into an in-memory `TokenBlockDataset` (1D `torch.Tensor` of token IDs).
  - When `in_memory=False`:
    - Call `datasets.load_dataset(dataset_name, dataset_config, split=..., streaming=False)`.
- **Memory Safeguards**:
  - Parameters: `max_train_samples: int | None` and `max_val_samples: int | None`.
  - During streaming ingestion, break after reaching `max_train_samples` / `max_val_samples`.
- **Tokenization & Delimiters**:
  - Parameter: `tokenizer: str = "gpt2"` (supports `"gpt2"`, `"byte"`, and standard `tiktoken` encodings).
  - Story / Document separator: `<|endoftext|>` token ID appended between distinct documents/stories.
  - WikiText whitespace filtering: Filter out empty lines (`not line.strip()`), joining non-empty lines with `\n` and delimiting main headings (` = ... = `) with `<|endoftext|>`.
- **Split Resolution**:
  - If official `"validation"` split exists, load it directly as the validation stream.
  - If only `"train"` split exists or custom split is provided, partition into train/val via `validation_fraction` (default `0.1`).
- **Configuration Presets**:
  - `configs/wikitext.yaml`: default configuration for `bdh_transformer` + `wikitext` (`wikitext-2-raw-v1`, `in_memory: true`, `context_length: 256`, `batch_size: 32`).
  - `configs/tiny_stories.yaml`: default configuration for `bdh_transformer` + `tiny_stories` (`max_train_samples: 100000`, `in_memory: true`, `context_length: 256`, `batch_size: 32`).
- **Glossary & ADR**:
  - Record ADR `0003-in-memory-huggingface-datasets.md` detailing the zero-disk streaming rationale and 16GB memory constraints.

## Testing Decisions

- **Testing Seams**:
  - **Data Module Seam (Primary)**: Test `WikiTextDataModule` and `TinyStoriesDataModule` directly and via `DATA_REGISTRY.build(...)`. Verify stream ingestion, `max_train_samples` limits, `TokenBlockDataset` shape and batch generation, `<|endoftext|>` insertion, and validation split resolution.
  - **Disk Footprint Seam**: Assert that running `setup()` in `in_memory=True` mode does not create files in local cache or dataset directories.
  - **End-to-End Runtime Seam**: Run a multi-step training execution with `build_components` on a small in-memory stream using the byte/gpt2 tokenizer to verify model convergence and dataloader compatibility.
- **Prior Art**:
  - Follow the testing architecture in `tests/test_training_environment.py` (`test_byte_tokenized_data_module`, `TinyShakespeareDataModule` test cases).

## Out of Scope

- Non-text multimodal or sequence-to-sequence datasets (e.g. image-text, audio).
- Distributed multi-node sharded streaming with `torch.distributed`.
- Custom Hugging Face tokenizers outside `tiktoken` encodings.

## Further Notes

- WikiText-2 has ~2 million tokens (~8MB RAM in int32), making it ideal for rapid unit and integration testing.
- TinyStories with 100,000 samples occupies ~50MB RAM, allowing fast training experiments on the 16GB Mac environment without swap or memory pressure.
