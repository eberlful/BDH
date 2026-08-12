"""Run-directory creation, reproducibility, and component assembly."""

from __future__ import annotations

import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core.config import dump_config
from .core.registry import (
    CALLBACK_REGISTRY,
    DATA_REGISTRY,
    LOGGER_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    load_builtin_components,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(runs_dir: str | Path) -> Path:
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    while True:
        candidate = root / f"{timestamp}-{secrets.token_hex(3)}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def write_run_metadata(run_dir: Path, config: dict[str, Any], command: str) -> None:
    dump_config(config, run_dir / "config.yaml")
    metadata = {
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_components(config: dict[str, Any], run_dir: Path):
    """Instantiate data, model, callbacks, and loggers in dependency order."""
    load_builtin_components()
    data = DATA_REGISTRY.instantiate(config["data"], seed=int(config.get("seed", 42)))
    model_params = dict(config["model"].get("params", {}))
    if model_params.get("vocab_size") == "auto":
        if not hasattr(data, "vocab_size"):
            raise ValueError("model.params.vocab_size=auto requires the data module to expose vocab_size.")
        model_params["vocab_size"] = int(data.vocab_size)
    model = MODEL_REGISTRY.instantiate({"name": config["model"]["name"], "params": model_params})
    callbacks = [CALLBACK_REGISTRY.instantiate(item, run_dir=run_dir) for item in config["callbacks"]]
    loggers = [LOGGER_REGISTRY.instantiate(item, run_dir=run_dir) for item in config["loggers"]]
    trainer_settings = dict(config["trainer"])
    trainer_name = trainer_settings.pop("name", "torch")
    trainer = TRAINER_REGISTRY.instantiate(
        {"name": trainer_name, "params": trainer_settings},
        model=model,
        data_module=data,
        callbacks=callbacks,
        loggers=loggers,
        config=config,
        run_dir=run_dir,
        device=config.get("device", "auto"),
    )
    return trainer


def component_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "data": config["data"],
    }

