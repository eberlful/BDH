# 02 — In-Memory TinyStories Integration and Memory Safeguards

**What to build:** A dedicated TinyStories data module that streams stories from `roneneldan/TinyStories` purely into RAM, enforces sample limits (`max_train_samples`, `max_val_samples`) to stay comfortably within the 16GB memory ceiling, delimits individual stories with `<|endoftext|>`, and includes unit tests verifying memory caps and streaming tokenization.

**Blocked by:** 01 — In-Memory Hugging Face Base DataModule and WikiText Integration

**Status:** ready-for-agent

- [ ] `TinyStoriesDataModule` registered under `"tiny_stories"` targets `dataset_name="roneneldan/TinyStories"`.
- [ ] Enforces configurable `max_train_samples` (default `100_000`) and `max_val_samples` (default `10_000`) to guarantee memory safety on 16GB unified memory, while supporting `None` for full load.
- [ ] Delimits distinct stories with `<|endoftext|>` tokens before packing into contiguous context blocks.
- [ ] Supports both `in_memory=True` (streaming RAM ingestion) and `in_memory=False` (disk caching).
- [ ] Unit tests verify sample limit truncation, boundary token delimiter insertion, and dataloader batch outputs.
