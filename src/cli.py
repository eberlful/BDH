"""Command-line interface for training, validation, and resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .core.config import dump_config, load_config, load_plugin_modules, validate_config
from .data.data import encode_sudoku_prompt
from .runtime import build_components, component_signature, create_run_dir, seed_everything, write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdh", description="Configurable PyTorch LLM training environment.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start a new training run")
    train.add_argument("config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    train.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging of training data and model predictions")

    validate = commands.add_parser("validate", help="validate configuration and registered components")
    validate.add_argument("config", type=Path)
    validate.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")

    resume = commands.add_parser("resume", help="resume the latest checkpoint in a run directory")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    resume.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging of training data and model predictions")

    generate = commands.add_parser("generate", help="generate text from the best checkpoint in a run directory")
    generate.add_argument("run_dir", type=Path)
    generate.add_argument("prompt")
    generate.add_argument("--max-tokens", type=int, required=True, dest="max_tokens")
    return parser


def _validate_and_load(path: Path, overrides: list[str]) -> dict[str, Any]:
    config = load_config(path, overrides)
    validate_config(config)
    return config


def run_train(config_path: Path, overrides: list[str], verbose: bool = False) -> int:
    config = _validate_and_load(config_path, overrides)
    if verbose:
        config["verbose"] = True
    seed_everything(int(config.get("seed", 42)))
    runs_dir = Path(config.get("runs_dir", "runs"))
    run_dir = create_run_dir(runs_dir)
    write_run_metadata(run_dir, config, f"train {config_path}")
    dump_config(config, run_dir / "resolved_config.yaml")
    trainer = build_components(config, run_dir)
    print(f"Starting run in {run_dir}")
    trainer.fit()
    return 0


def run_validate(config_path: Path, overrides: list[str]) -> int:
    config = _validate_and_load(config_path, overrides)
    print(f"Configuration valid: {config_path}")
    return 0


def run_resume(run_dir: Path, overrides: list[str], verbose: bool = False) -> int:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory does not contain {config_path.name}: {run_dir}")
    config = _validate_and_load(config_path, overrides)
    if verbose:
        config["verbose"] = True
    original = load_config(config_path)
    if component_signature(config) != component_signature(original):
        raise ValueError("Resume overrides cannot change model or data component names/parameters.")
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No resumable checkpoint found at {checkpoint_path}.")
    dump_config(config, run_dir / "resolved_config.yaml")
    trainer = build_components(config, run_dir)
    print(f"Resuming run in {run_dir}")
    trainer.fit(checkpoint_path)
    return 0


def run_generate(run_dir: Path, prompt: str, max_tokens: int) -> int:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory does not contain {config_path.name}: {run_dir}")
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best checkpoint found at {checkpoint_path}.")

    config = _validate_and_load(config_path, [])
    trainer = build_components(config, run_dir)
    trainer.setup()
    eos_token_id: int | None = None
    if config["data"]["name"] == "sudoku":
        token_ids = encode_sudoku_prompt(prompt)
    elif config["data"]["name"] == "sudoku_cot":
        tokenizer = getattr(trainer.data_module, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("SudokuCoT requires a valid tokenizer on data module.")
        if not prompt.startswith("Sudoku:"):
            try:
                digits = [int(c) for c in prompt if c.isdigit()]
                if len(digits) == 81:
                    from .data.sudoku_cot import build_sudoku_cot_prompt
                    reasoning_mode = config.get("data", {}).get("params", {}).get("reasoning_mode", "full")
                    prompt = build_sudoku_cot_prompt(digits, reasoning_mode=reasoning_mode)
            except Exception:
                pass
        token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        if not token_ids:
            raise ValueError("Prompt must contain at least one token.")
        eos_token_id = getattr(trainer.data_module, "eos_token_id", None)
        if eos_token_id is None:
            try:
                eos_encoded = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
                if eos_encoded:
                    eos_token_id = eos_encoded[0]
            except Exception:
                pass
    else:
        tokenizer = getattr(trainer.data_module, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)) or not callable(
            getattr(tokenizer, "decode", None)
        ):
            raise ValueError("Generate mode requires a text data module exposing tokenizer.encode/decode.")
        token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        if not token_ids:
            raise ValueError("Prompt must contain at least one token.")
        eos_token_id = getattr(trainer.data_module, "eos_token_id", None)
        if eos_token_id is None:
            try:
                eos_encoded = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
                if eos_encoded:
                    eos_token_id = eos_encoded[0]
            except Exception:
                pass

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=trainer.device)
    try:
        trainer.restore_checkpoint(checkpoint_path)
        trainer.model.eval()
        with torch.inference_mode():
            generated = trainer.model.generate(
                input_ids, max_new_tokens=max_tokens, eos_token_id=eos_token_id
            )
        if config["data"]["name"] == "sudoku":
            print(" ".join(str(token) for token in generated[0].tolist()))
        else:
            print(tokenizer.decode(generated[0].tolist()))
    finally:
        for logger in getattr(trainer, "loggers", []):
            if hasattr(logger, "close"):
                logger.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return run_train(args.config, args.overrides, verbose=args.verbose)
        if args.command == "validate":
            return run_validate(args.config, args.overrides)
        if args.command == "resume":
            return run_resume(args.run_dir, args.overrides, verbose=args.verbose)
        if args.command == "generate":
            return run_generate(args.run_dir, args.prompt, args.max_tokens)
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"bdh: error: {exc}", file=sys.stderr)
        return 2
