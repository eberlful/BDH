from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.core.base import BaseDataModule, BaseModel, BaseTrainer, BaseValidator
from src.core.config import load_config, validate_config
from src.core.registry import VALIDATOR_REGISTRY, load_builtin_components
from src.runtime import build_components


class DummyValidator(BaseValidator):
    def __init__(self, multiplier: int = 1, run_dir: Path | None = None, **kwargs: Any) -> None:
        super().__init__(run_dir=run_dir, **kwargs)
        self.multiplier = multiplier
        self.started = False
        self.batch_count = 0

    def on_validation_start(self, trainer: BaseTrainer) -> None:
        self.started = True

    def on_validation_batch(
        self,
        trainer: BaseTrainer,
        batch: Any,
        batch_idx: int,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        self.batch_count += 1

    def on_validation_epoch_end(self, trainer: BaseTrainer) -> Mapping[str, float]:
        return {"val/dummy_metric": float(self.batch_count * self.multiplier)}

    def validate(
        self,
        model: BaseModel,
        data_module: BaseDataModule,
        trainer: BaseTrainer | None = None,
    ) -> Mapping[str, float]:
        return {"val/dummy_direct": 42.0 * self.multiplier}


class BaseValidatorAndRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        load_builtin_components()
        if "test_dummy_validator" not in VALIDATOR_REGISTRY.names():
            VALIDATOR_REGISTRY.register("test_dummy_validator", DummyValidator)

    def test_base_validator_default_hooks(self) -> None:
        validator = BaseValidator()
        self.assertIsNone(validator.run_dir)
        # Should execute default no-op methods without raising exceptions
        validator.on_validation_start(None)  # type: ignore[arg-type]
        validator.on_validation_batch(None, {}, 0, {})  # type: ignore[arg-type]
        self.assertEqual(validator.on_validation_epoch_end(None), {})  # type: ignore[arg-type]
        self.assertEqual(validator.validate(None, None, None), {})  # type: ignore[arg-type]

    def test_custom_validator_lifecycle(self) -> None:
        validator = DummyValidator(multiplier=2)
        validator.on_validation_start(None)  # type: ignore[arg-type]
        self.assertTrue(validator.started)
        validator.on_validation_batch(None, {}, 0, None)  # type: ignore[arg-type]
        validator.on_validation_batch(None, {}, 1, None)  # type: ignore[arg-type]
        metrics = validator.on_validation_epoch_end(None)  # type: ignore[arg-type]
        self.assertEqual(metrics, {"val/dummy_metric": 4.0})
        direct_metrics = validator.validate(None, None, None)  # type: ignore[arg-type]
        self.assertEqual(direct_metrics, {"val/dummy_direct": 84.0})

    def test_validator_registry_instantiate_string(self) -> None:
        instance = VALIDATOR_REGISTRY.instantiate("test_dummy_validator")
        self.assertIsInstance(instance, DummyValidator)
        self.assertEqual(instance.multiplier, 1)

    def test_validator_registry_instantiate_dict(self) -> None:
        instance = VALIDATOR_REGISTRY.instantiate(
            {"name": "test_dummy_validator", "params": {"multiplier": 5}}
        )
        self.assertIsInstance(instance, DummyValidator)
        self.assertEqual(instance.multiplier, 5)

    def test_validator_registry_unknown_component_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown validator 'non_existent'"):
            VALIDATOR_REGISTRY.get("non_existent")

    def test_yaml_config_with_single_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validator": {
                            "name": "test_dummy_validator",
                            "params": {"multiplier": 3},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                config["validator"],
                {"name": "test_dummy_validator", "params": {"multiplier": 3}},
            )
            validate_config(config)

    def test_yaml_config_with_validator_as_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validator": "test_dummy_validator",
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                config["validator"],
                {"name": "test_dummy_validator", "params": {}},
            )
            validate_config(config)

    def test_yaml_config_with_validators_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validators": [
                            "test_dummy_validator",
                            {"name": "test_dummy_validator", "params": {"multiplier": 4}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(
                config["validators"],
                [
                    {"name": "test_dummy_validator", "params": {}},
                    {"name": "test_dummy_validator", "params": {"multiplier": 4}},
                ],
            )
            validate_config(config)

    def test_yaml_config_backward_compatibility_without_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertNotIn("validator", config)
            validate_config(config)

    def test_yaml_config_override_validator_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validator": {
                            "name": "test_dummy_validator",
                            "params": {"multiplier": 2},
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(path, overrides=["validator.params.multiplier=10"])
            self.assertEqual(config["validator"]["params"]["multiplier"], 10)

    def test_yaml_config_invalid_validator_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validator": 12345,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "validator must be a string or mapping"):
                load_config(path)

    def test_yaml_config_invalid_validators_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32}},
                        "data": {"name": "tiny_shakespeare", "params": {"tokenizer": "byte"}},
                        "validators": "not-a-list",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "validators must be a list"):
                load_config(path)

    def test_runtime_build_components_instantiates_validators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            config_path = Path(directory) / "config.yaml"
            input_file = Path(directory) / "input.txt"
            input_file.write_text("abcdefghijklmnopqrstuvwxyz " * 10, encoding="utf-8")
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name": "gpt_model", "params": {"vocab_size": 32, "context_length": 8, "d_model": 16, "n_heads": 4, "n_layers": 1}},
                        "data": {"name": "tiny_shakespeare", "params": {"input_file_path": str(input_file), "tokenizer": "byte", "context_length": 8, "batch_size": 2}},
                        "validator": {"name": "test_dummy_validator", "params": {"multiplier": 7}},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            trainer = build_components(config, run_dir)
            self.assertIsNotNone(trainer)
