#!/usr/bin/env python3
"""Compare teacher-forced and autoregressive predictions on train samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.core.config import load_config
from src.data.sudoku_cot import build_sudoku_cot_full_text, build_sudoku_cot_prompt
from src.runtime import build_components, seed_everything


def first_mismatch(predicted: list[int], expected: list[int]) -> int | None:
    for index, (pred, target) in enumerate(zip(predicted, expected)):
        if pred != target:
            return index
    if len(predicted) != len(expected):
        return min(len(predicted), len(expected))
    return None


def token_preview(tokenizer, tokens: list[int], limit: int = 240) -> str:
    clipped = tokens[:limit]
    try:
        return repr(tokenizer.decode(clipped))
    except Exception:
        return repr(clipped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename or path relative to run_dir/checkpoints (default: best.pt).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of training samples to inspect; 0 means all (default: 0).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yaml"
    checkpoint_arg = Path(args.checkpoint)
    checkpoint_path = (
        checkpoint_arg
        if checkpoint_arg.is_absolute()
        else run_dir / "checkpoints" / checkpoint_arg
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    trainer = build_components(config, run_dir)
    trainer.setup()
    trainer.restore_checkpoint(checkpoint_path)

    model = trainer.model
    data_module = trainer.data_module
    dataset = data_module.train_dataset
    if dataset is None or not getattr(dataset, "samples", None):
        raise RuntimeError("Expected a SudokuCoT training dataset with raw samples.")

    reasoning_mode = getattr(data_module, "reasoning_mode", "none")
    tokenizer = data_module.tokenizer
    device = trainer.device
    samples = dataset.samples
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"tokenizer={getattr(data_module, 'tokenizer_name', 'unknown')}")
    print(f"reasoning_mode={reasoning_mode}")
    print(f"samples_evaluated={len(samples)}")
    print(f"eos_token_id={data_module.eos_token_id}")

    model.eval()
    teacher_exact_count = 0
    autoregressive_exact_count = 0
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            puzzle = tuple(sample["puzzle"])
            solution = tuple(sample["solution"])
            prompt_text = build_sudoku_cot_prompt(puzzle, reasoning_mode=reasoning_mode)
            _, full_text = build_sudoku_cot_full_text(
                puzzle, solution, reasoning_mode=reasoning_mode
            )
            prompt_ids = tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
            full_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
            completion_target = full_ids[len(prompt_ids) :]
            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)

            logits = model(
                input_ids,
                latent_reasoning_steps=int(
                    config["model"]["params"].get("latent_reasoning_steps", 1)
                ),
            )
            teacher_predictions = logits.argmax(dim=-1)[0].tolist()
            generated = model.generate(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
                max_new_tokens=max(1, len(completion_target) + 8),
                temperature=1.0,
                top_k=1,
                eos_token_id=data_module.eos_token_id,
                latent_reasoning_steps=int(
                    config["model"]["params"].get("latent_reasoning_steps", 1)
                ),
            )[0, len(prompt_ids) :].tolist()

            prompt_boundary = max(0, len(prompt_ids) - 1)
            teacher_completion = teacher_predictions[
                prompt_boundary : prompt_boundary + len(completion_target)
            ]
            teacher_mismatch = first_mismatch(teacher_completion, completion_target)
            generated_mismatch = first_mismatch(generated, completion_target)
            teacher_exact = teacher_mismatch is None
            autoregressive_exact = generated_mismatch is None
            teacher_exact_count += int(teacher_exact)
            autoregressive_exact_count += int(autoregressive_exact)

            print(f"sample_index={sample_index}")
            print(f"prompt_tokens={len(prompt_ids)}")
            print(f"completion_tokens={len(completion_target)}")
            print(
                "active_completion_tokens="
                f"{sum(1 for token in completion_target if token != -100)}"
            )
            print(f"teacher_forced_first_mismatch={teacher_mismatch}")
            print(f"autoregressive_first_mismatch={generated_mismatch}")
            print(f"teacher_forced_exact={teacher_exact}")
            print(f"autoregressive_exact={autoregressive_exact}")
            print(f"target_preview={token_preview(tokenizer, completion_target)}")
            print(f"teacher_preview={token_preview(tokenizer, teacher_completion)}")
            print(f"generated_preview={token_preview(tokenizer, generated)}")
            print("puzzle=" + "".join(map(str, puzzle)))
            print("solution=" + "".join(map(str, solution)))

    print(f"teacher_forced_exact_count={teacher_exact_count}")
    print(f"autoregressive_exact_count={autoregressive_exact_count}")
    print(f"teacher_forced_exact_rate={teacher_exact_count / len(samples):.4f}")
    print(f"autoregressive_exact_rate={autoregressive_exact_count / len(samples):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
