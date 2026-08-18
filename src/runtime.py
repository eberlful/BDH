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
    VALIDATOR_REGISTRY,
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
    is_verbose = bool(config.get("verbose") or config.get("trainer", {}).get("verbose"))
    loggers = []
    for item in config["loggers"]:
        item_copy = dict(item)
        if is_verbose and item_copy.get("name") == "terminal":
            params = dict(item_copy.get("params", {}))
            params.setdefault("verbose", True)
            item_copy["params"] = params
        loggers.append(LOGGER_REGISTRY.instantiate(item_copy, run_dir=run_dir))
    validator_specs: list[dict[str, Any]] = []
    if "validator" in config and config["validator"]:
        validator_specs.append(config["validator"])
    if "validators" in config and config["validators"]:
        validator_specs.extend(config["validators"])
    validators = [
        VALIDATOR_REGISTRY.instantiate(item, run_dir=run_dir)
        for item in validator_specs
    ]
    trainer_settings = dict(config["trainer"])
    trainer_name = trainer_settings.pop("name", "torch")
    dtype = trainer_settings.pop("dtype", config.get("dtype", "float16"))
    mixed_precision = trainer_settings.pop("mixed_precision", config.get("mixed_precision", False))
    trainer = TRAINER_REGISTRY.instantiate(
        {"name": trainer_name, "params": trainer_settings},
        model=model,
        data_module=data,
        callbacks=callbacks,
        loggers=loggers,
        validators=validators,
        config=config,
        run_dir=run_dir,
        device=config.get("device", "auto"),
        dtype=dtype,
        mixed_precision=mixed_precision,
    )
    return trainer



def component_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "data": config["data"],
    }

