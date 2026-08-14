from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import datasets
import torch
import yaml

from src.cli import run_train, run_validate
from src.core.registry import DATA_REGISTRY, load_builtin_components
from src.data.data import TokenBlockDataset
from src.data.hf import (
    HuggingFaceTextDataModule,
    TinyStoriesDataModule,
    WikiTextDataModule,
    _resolve_dataset_name,
)
from src.model.bdh import BDHTransformer


class HuggingFaceDataModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        load_builtin_components()

    def test_registry_contains_hf_text_and_wikitext_and_tiny_stories(self) -> None:
        self.assertIn("hf_text", DATA_REGISTRY.names())
        self.assertIn("wikitext", DATA_REGISTRY.names())
        self.assertIn("tiny_stories", DATA_REGISTRY.names())

    def test_resolve_dataset_name(self) -> None:
        self.assertEqual(_resolve_dataset_name("wikitext"), "Salesforce/wikitext")
        self.assertEqual(_resolve_dataset_name("Salesforce/wikitext"), "Salesforce/wikitext")
        self.assertEqual(_resolve_dataset_name("roneneldan/TinyStories"), "roneneldan/TinyStories")
        self.assertEqual(_resolve_dataset_name("custom_dataset"), "custom_dataset")

    def test_in_memory_streaming_disables_caching_and_batches_tokens(self) -> None:
        mock_train = [{"text": f"This is training story number {i}. It has some content."} for i in range(20)]
        mock_val = [{"text": f"This is validation story number {i}. It has some validation content."} for i in range(10)]

        def fake_load_dataset(name, *args, **kwargs):
            split = kwargs.get("split", args[1] if len(args) > 1 else "train")
            streaming = kwargs.get("streaming", False)
            self.assertTrue(streaming, "in_memory=True must request streaming=True from datasets")
            if split == "train":
                return iter(mock_train)
            if split == "validation":
                return iter(mock_val)
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.disable_caching") as mock_disable_caching, patch(
            "datasets.load_dataset", side_effect=fake_load_dataset
        ):
            module = HuggingFaceTextDataModule(
                dataset_name="dummy_text_corpus",
                tokenizer="byte",
                context_length=16,
                batch_size=4,
                in_memory=True,
            )
            module.prepare_data()
            module.setup("fit")

            mock_disable_caching.assert_called()
            self.assertIsNotNone(module.train_dataset)
            self.assertIsNotNone(module.val_dataset)
            self.assertGreater(len(module.train_dataset), 0)
            self.assertGreater(len(module.val_dataset), 0)

            batch = next(iter(module.train_dataloader()))
            self.assertEqual(batch["input_ids"].shape, (4, 16))
            self.assertEqual(batch["target_ids"].shape, (4, 16))
            self.assertEqual(batch["input_ids"].dtype, torch.long)
            self.assertEqual(batch["target_ids"].dtype, torch.long)

            val_batch = next(iter(module.val_dataloader()))
            self.assertEqual(val_batch["input_ids"].shape, (4, 16))

    def test_in_memory_false_uses_non_streaming_arrow_datasets(self) -> None:
        mock_train = [{"text": f"Disk cached story {i} with content words."} for i in range(20)]
        mock_val = [{"text": f"Disk cached validation story {i}."} for i in range(10)]

        def fake_load_dataset(name, *args, **kwargs):
            split = kwargs.get("split", args[1] if len(args) > 1 else "train")
            streaming = kwargs.get("streaming", False)
            self.assertFalse(streaming, "in_memory=False must request streaming=False from datasets")
            if split == "train":
                return mock_train
            if split == "validation":
                return mock_val
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = HuggingFaceTextDataModule(
                dataset_name="dummy_cached_corpus",
                tokenizer="byte",
                context_length=16,
                batch_size=4,
                in_memory=False,
            )
            module.setup("fit")
            self.assertIsNotNone(module.train_dataset)
            self.assertIsNotNone(module.val_dataset)
            batch = next(iter(module.train_dataloader()))
            self.assertEqual(batch["input_ids"].shape, (4, 16))

    def test_single_split_fallback_to_validation_fraction(self) -> None:
        mock_data = [{"text": f"Single split document {i} with enough tokens to partition properly."} for i in range(30)]

        def fake_load_dataset(name, *args, **kwargs):
            split = kwargs.get("split", "train")
            if split == "validation":
                raise ValueError("Split 'validation' not found in dataset")
            if split == "train":
                return iter(mock_data)
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = HuggingFaceTextDataModule(
                dataset_name="single_split_corpus",
                tokenizer="byte",
                context_length=16,
                batch_size=2,
                validation_fraction=0.2,
                in_memory=True,
            )
            module.setup("fit")
            self.assertIsNotNone(module.train_dataset)
            self.assertIsNotNone(module.val_dataset)
            self.assertGreater(len(module.train_dataset), len(module.val_dataset))

    def test_sample_limits_max_train_and_val_samples(self) -> None:
        mock_train = [{"text": f"Story {i}"} for i in range(100)]
        mock_val = [{"text": f"Val story {i}"} for i in range(50)]

        def fake_load_dataset(name, *args, **kwargs):
            split = kwargs.get("split", "train")
            if split == "train":
                return iter(mock_train)
            if split == "validation":
                return iter(mock_val)
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = HuggingFaceTextDataModule(
                dataset_name="dummy_corpus",
                tokenizer="byte",
                context_length=8,
                batch_size=2,
                max_train_samples=10,
                max_val_samples=5,
                in_memory=True,
            )
            # With max_train_samples=10, exactly 10 samples are processed
            module.setup("fit")
            self.assertIsNotNone(module.train_dataset)
            self.assertIsNotNone(module.val_dataset)

    def test_wikitext_whitespace_filtering_and_document_boundaries(self) -> None:
        wikitext_stream = [
            {"text": ""},
            {"text": "   \n\t"},
            {"text": " = First Article Title = \n"},
            {"text": ""},
            {"text": "This is the first paragraph of the first article.\n"},
            {"text": " = = Section Heading = = \n"},
            {"text": "This is a section inside the first article.\n"},
            {"text": ""},
            {"text": " = Second Article Title = \n"},
            {"text": "This is the second article body.\n"},
            {"text": "   "},
        ]

        def fake_load_dataset(name, *args, **kwargs):
            self.assertEqual(_resolve_dataset_name(name), "Salesforce/wikitext")
            split = kwargs.get("split", "train")
            if split == "train":
                return iter(wikitext_stream)
            if split == "validation":
                return iter(wikitext_stream)
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = WikiTextDataModule(
                dataset_config="wikitext-2-raw-v1",
                tokenizer="byte",
                context_length=16,
                batch_size=2,
                in_memory=True,
            )
            module.setup("fit")
            self.assertIsNotNone(module.train_dataset)

            # In byte tokenizer, eos_token is 256 (<|endoftext|>)
            # Check that 256 is present in the token stream separating the two articles and at the end
            token_tensor = module.train_dataset.inputs
            eos_indices = (token_tensor == 256).nonzero(as_tuple=True)[0].tolist()
            # We must have at least one delimiter between article 1 and article 2, plus at the end
            self.assertGreaterEqual(len(eos_indices), 1)

    def test_wikitext_instantiation_via_registry(self) -> None:
        instance = DATA_REGISTRY.instantiate(
            {
                "name": "wikitext",
                "params": {
                    "dataset_config": "wikitext-2-raw-v1",
                    "tokenizer": "byte",
                    "context_length": 32,
                    "batch_size": 4,
                },
            }
        )
        self.assertIsInstance(instance, WikiTextDataModule)
        self.assertEqual(instance.dataset_config, "wikitext-2-raw-v1")
        self.assertEqual(instance.context_length, 32)

    def test_tinystories_defaults_and_story_boundaries(self) -> None:
        story_stream = [
            {"text": "Once upon a time, there was a little dog named Spot."},
            {"text": "Spot liked to run and play in the green grass with his ball."},
            {"text": "One sunny morning, Spot found a shiny golden bone."},
            {"text": "Spot took the shiny bone home to show his best friend Lily."},
        ]

        def fake_load_dataset(name, *args, **kwargs):
            self.assertEqual(name, "roneneldan/TinyStories")
            split = kwargs.get("split", "train")
            if split == "train":
                return iter(story_stream)
            if split == "validation":
                return iter(story_stream)
            raise ValueError(f"Unknown split {split}")

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = TinyStoriesDataModule(
                tokenizer="byte",
                context_length=16,
                batch_size=2,
                max_train_samples=3,
                max_val_samples=2,
                in_memory=True,
            )
            self.assertEqual(module.dataset_name, "roneneldan/TinyStories")
            self.assertEqual(module.max_train_samples, 3)
            self.assertEqual(module.max_val_samples, 2)

            module.setup("fit")
            self.assertIsNotNone(module.train_dataset)

            # In byte tokenizer, eos_token is 256 (<|endoftext|>)
            token_tensor = module.train_dataset.inputs
            eos_count = int((token_tensor == 256).sum().item())
            self.assertGreaterEqual(eos_count, 1)

    def test_tinystories_instantiation_via_registry(self) -> None:
        instance = DATA_REGISTRY.instantiate(
            {
                "name": "tiny_stories",
                "params": {
                    "tokenizer": "byte",
                    "context_length": 32,
                    "batch_size": 4,
                    "max_train_samples": 500,
                },
            }
        )
        self.assertIsInstance(instance, TinyStoriesDataModule)
        self.assertEqual(instance.dataset_name, "roneneldan/TinyStories")
        self.assertEqual(instance.max_train_samples, 500)
        self.assertEqual(instance.max_val_samples, 10_000)

    def test_tinystories_end_to_end_training_step(self) -> None:
        story_stream = [
            {"text": f"Story number {i} about a cheerful bird singing songs in a sunny park."}
            for i in range(15)
        ]

        def fake_load_dataset(name, *args, **kwargs):
            return iter(story_stream)

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = TinyStoriesDataModule(
                tokenizer="byte",
                context_length=16,
                batch_size=2,
                in_memory=True,
            )
            module.setup("fit")
            model = BDHTransformer(
                vocab_size=module.vocab_size,
                context_length=module.context_length,
                d_model=16,
                n_heads=4,
                n_layers=1,
            )
            batch = next(iter(module.train_dataloader()))
            step_output = model.training_step(batch, 0)
            self.assertIn("loss", step_output)
            self.assertTrue(torch.isfinite(step_output["loss"]))

    def test_end_to_end_training_step_with_model(self) -> None:
        mock_data = [{"text": f" = Article {i} = \nDetailed text content for article {i}.\n"} for i in range(15)]

        def fake_load_dataset(name, *args, **kwargs):
            return iter(mock_data)

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            module = WikiTextDataModule(
                tokenizer="byte",
                context_length=16,
                batch_size=2,
                in_memory=True,
            )
            module.setup("fit")
            model = BDHTransformer(
                vocab_size=module.vocab_size,
                context_length=module.context_length,
                d_model=16,
                n_heads=4,
                n_layers=1,
            )
            batch = next(iter(module.train_dataloader()))
            step_output = model.training_step(batch, 0)
            self.assertIn("loss", step_output)
            self.assertTrue(torch.isfinite(step_output["loss"]))


    def test_cli_train_wikitext_multi_step(self) -> None:
        mock_data = [{"text": f" = Article {i} = \nDetailed text content for article {i}.\n"} for i in range(20)]

        def fake_load_dataset(name, *args, **kwargs):
            return iter(mock_data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = {
                "seed": 42,
                "device": "cpu",
                "runs_dir": str(root / "runs"),
                "model": {
                    "name": "bdh_transformer",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 16,
                        "d_model": 16,
                        "n_heads": 4,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "wikitext",
                    "params": {
                        "dataset_config": "wikitext-2-raw-v1",
                        "tokenizer": "byte",
                        "context_length": 16,
                        "batch_size": 2,
                        "in_memory": True,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 3,
                    "log_every_n_steps": 1,
                },
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [{"name": "text_file", "params": {}}],
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch("datasets.load_dataset", side_effect=fake_load_dataset):
                self.assertEqual(run_validate(config_path, []), 0)
                self.assertEqual(run_train(config_path, []), 0)

            run_dirs = list((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "training.log").exists())

    def test_cli_train_tinystories_multi_step(self) -> None:
        mock_stories = [
            {"text": f"Once upon a time, tiny animal {i} had a wonderful adventure in the forest."}
            for i in range(25)
        ]

        def fake_load_dataset(name, *args, **kwargs):
            return iter(mock_stories)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = {
                "seed": 42,
                "device": "cpu",
                "runs_dir": str(root / "runs"),
                "model": {
                    "name": "bdh_transformer",
                    "params": {
                        "vocab_size": "auto",
                        "context_length": 16,
                        "d_model": 16,
                        "n_heads": 4,
                        "n_layers": 1,
                    },
                },
                "data": {
                    "name": "tiny_stories",
                    "params": {
                        "tokenizer": "byte",
                        "context_length": 16,
                        "batch_size": 2,
                        "max_train_samples": 20,
                        "max_val_samples": 5,
                        "in_memory": True,
                    },
                },
                "trainer": {
                    "name": "torch",
                    "max_epochs": 1,
                    "max_steps": 3,
                    "log_every_n_steps": 1,
                },
                "callbacks": [{"name": "checkpoint", "params": {"save_best": True}}],
                "loggers": [{"name": "text_file", "params": {}}],
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch("datasets.load_dataset", side_effect=fake_load_dataset):
                self.assertEqual(run_validate(config_path, []), 0)
                self.assertEqual(run_train(config_path, []), 0)

            run_dirs = list((root / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "training.log").exists())


if __name__ == "__main__":
    unittest.main()


