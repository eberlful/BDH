#!/usr/bin/env python3
"""Report per-puzzle Sudoku validation errors for one checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.core.config import load_config
from src.data.sudoku_cot import build_sudoku_cot_prompt
from src.data.data import is_valid_sudoku_board
from src.runtime import build_components, seed_everything
from src.validation.sudoku import extract_solution_grid_from_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--max-samples", type=int, default=16)
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
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive.")

    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    trainer = build_components(config, run_dir)
    trainer.setup()
    trainer.restore_checkpoint(checkpoint_path)

    data_module = trainer.data_module
    dataset = data_module.val_dataset
    if dataset is None or not getattr(dataset, "samples", None):
        raise RuntimeError("Expected a SudokuCoT validation dataset with raw samples.")

    samples = dataset.samples[: args.max_samples]
    tokenizer = data_module.tokenizer
    reasoning_mode = getattr(data_module, "reasoning_mode", "none")
    model = trainer.model
    model.eval()
    device = trainer.device
    latent_steps = int(config["model"]["params"].get("latent_reasoning_steps", 1))
    max_new_tokens = int(config.get("validator", {}).get("params", {}).get("max_new_tokens", 180))

    print(f"run_dir={run_dir}")
    print(f"checkpoint={checkpoint_path}")
    print("split=validation")
    print(f"samples_evaluated={len(samples)}")
    print(f"device={device}")

    total_cells = 0
    total_correct = 0
    total_wrong = 0
    parsed_count = 0
    valid_count = 0
    exact_count = 0
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            puzzle = tuple(sample["puzzle"])
            solution = list(sample["solution"])
            prompt_text = build_sudoku_cot_prompt(puzzle, reasoning_mode=reasoning_mode)
            prompt_ids = tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            output_ids = model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,
                eos_token_id=data_module.eos_token_id,
                latent_reasoning_steps=latent_steps,
            )
            generated_tokens = output_ids[0, prompt_tensor.size(1) :].tolist()
            generated_text = tokenizer.decode(generated_tokens)
            board, parsed = extract_solution_grid_from_text(generated_text)
            correct_cells = sum(
                pred == target for pred, target in zip(board, solution)
            )
            wrong_cells = len(solution) - correct_cells
            valid = parsed and is_valid_sudoku_board(board)
            exact = parsed and board == solution

            parsed_count += int(parsed)
            valid_count += int(valid)
            exact_count += int(exact)
            total_correct += correct_cells
            total_wrong += wrong_cells
            total_cells += len(solution)

            print(f"sample_index={sample_index}")
            print(f"cell_accuracy={correct_cells / len(solution):.4f}")
            print(f"wrong_cells={wrong_cells}")
            print(f"parsed={parsed}")
            print(f"valid={valid}")
            print(f"exact={exact}")
            print(f"generated_preview={generated_text[:180]!r}")

    count = len(samples)
    print(f"mean_cell_accuracy={total_correct / total_cells:.4f}")
    print(f"mean_wrong_cells={total_wrong / count:.2f}")
    print(f"parse_rate={parsed_count / count:.4f}")
    print(f"validity_rate={valid_count / count:.4f}")
    print(f"board_accuracy={exact_count / count:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
