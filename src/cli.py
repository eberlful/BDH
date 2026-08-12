"""Command-line interface for training, validation, and resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core.config import dump_config, load_config, load_plugin_modules, validate_config
from .runtime import build_components, component_signature, create_run_dir, seed_everything, write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdh", description="Configurable PyTorch LLM training environment.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start a new training run")
    train.add_argument("config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")

    validate = commands.add_parser("validate", help="validate configuration and registered components")
    validate.add_argument("config", type=Path)
    validate.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")

    resume = commands.add_parser("resume", help="resume the latest checkpoint in a run directory")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    return parser


def _validate_and_load(path: Path, overrides: list[str]) -> dict[str, Any]:
    config = load_config(path, overrides)
    validate_config(config)
    return config


def run_train(config_path: Path, overrides: list[str]) -> int:
    config = _validate_and_load(config_path, overrides)
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


def run_resume(run_dir: Path, overrides: list[str]) -> int:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory does not contain {config_path.name}: {run_dir}")
    config = _validate_and_load(config_path, overrides)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return run_train(args.config, args.overrides)
        if args.command == "validate":
            return run_validate(args.config, args.overrides)
        if args.command == "resume":
            return run_resume(args.run_dir, args.overrides)
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"bdh: error: {exc}", file=sys.stderr)
        return 2
