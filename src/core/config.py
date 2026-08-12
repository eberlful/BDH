"""YAML loading, defaults, overrides, and component validation."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import yaml

from .registry import (
    CALLBACK_REGISTRY,
    DATA_REGISTRY,
    LOGGER_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    load_builtin_components,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "runs_dir": "runs",
    "plugins": [],
    "trainer": {
        "name": "torch",
        "max_epochs": 1,
        "max_steps": None,
        "log_every_n_steps": 10,
        "validate_every_n_epochs": 1,
        "gradient_clip_norm": None,
        "gradient_accumulation_steps": 1,
    },
    "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
    "loggers": ["terminal", "text_file", "tensorboard"],
}

_ROOT_KEYS = {
    "seed",
    "device",
    "runs_dir",
    "plugins",
    "trainer",
    "callbacks",
    "loggers",
    "model",
    "data",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration {config_path} must contain a YAML mapping at its root.")
    unknown = set(raw) - _ROOT_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(sorted(unknown))}.")
    config = deep_merge(DEFAULT_CONFIG, raw)
    normalize_config(config)
    if overrides:
        apply_overrides(config, overrides)
    normalize_config(config)
    return config


def load_plugin_modules(config: dict[str, Any]) -> None:
    plugins = config.get("plugins", [])
    if not isinstance(plugins, list) or not all(isinstance(item, str) for item in plugins):
        raise TypeError("plugins must be a list of Python module import paths.")
    for module_name in plugins:
        importlib.import_module(module_name)


def normalize_config(config: dict[str, Any]) -> None:
    for section in ("model", "data"):
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping.")
        if not isinstance(value.get("name"), str) or not value["name"]:
            raise ValueError(f"Configuration section {section!r} requires a component name.")
        params = value.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"{section}.params must be a mapping.")

    trainer = config.setdefault("trainer", {})
    if not isinstance(trainer, dict):
        raise TypeError("trainer must be a mapping.")
    trainer.setdefault("name", "torch")

    for section in ("callbacks", "loggers"):
        values = config.get(section, [])
        if not isinstance(values, list):
            raise TypeError(f"{section} must be a list.")
        normalized: list[dict[str, Any]] = []
        for value in values:
            if isinstance(value, str):
                normalized.append({"name": value, "params": {}})
            elif isinstance(value, dict):
                if not isinstance(value.get("name"), str) or not value["name"]:
                    raise ValueError(f"Every {section} entry requires a component name.")
                if not isinstance(value.get("params", {}), dict):
                    raise TypeError(f"Parameters for a {section} entry must be a mapping.")
                normalized.append({"name": value["name"], "params": dict(value.get("params", {}))})
            else:
                raise TypeError(f"Every {section} entry must be a string or mapping.")
        config[section] = normalized


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must use path=value syntax.")
        path, raw_value = override.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError("Override paths cannot be empty.")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value in override {override!r}.") from exc
        _set_path(config, path.split("."), value)


def _set_path(root: Any, parts: list[str], value: Any) -> None:
    current = root
    for index, part in enumerate(parts[:-1]):
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")
            current = current[int(part)]
        elif isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")
            current = current[part]
        else:
            raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")

    leaf = parts[-1]
    allow_new = "params" in parts[:-1]
    if isinstance(current, list):
        if not leaf.isdigit() or int(leaf) >= len(current):
            raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")
        current[int(leaf)] = value
    elif isinstance(current, dict):
        if leaf not in current and not allow_new:
            raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")
        current[leaf] = value
    else:
        raise ValueError(f"Invalid configuration path: {'.'.join(parts)}")


def validate_config(config: dict[str, Any]) -> None:
    load_builtin_components()
    load_plugin_modules(config)
    MODEL_REGISTRY.get(config["model"]["name"])
    DATA_REGISTRY.get(config["data"]["name"])
    TRAINER_REGISTRY.get(config["trainer"].get("name", "torch"))
    for item in config["callbacks"]:
        CALLBACK_REGISTRY.get(item["name"])
    for item in config["loggers"]:
        LOGGER_REGISTRY.get(item["name"])


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
