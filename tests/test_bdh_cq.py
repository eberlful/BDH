import unittest
import torch
from torch import nn

from src.core.base import BaseModel
from src.core.registry import MODEL_REGISTRY, load_builtin_components
from src.model.bdh_cq import BDHCQ, BDHCQConfig, ConfiguredBDHCQ


class TestBDHCQArchitecture(unittest.TestCase):
    def setUp(self):
        load_builtin_components()

    def test_model_registration(self):
        registered = MODEL_REGISTRY.get("bdh_cq")
        self.assertIs(registered, ConfiguredBDHCQ)
        self.assertIn("bdh_cq", MODEL_REGISTRY.names())

    def test_instantiation_via_registry(self):
        model = MODEL_REGISTRY.instantiate(
            {"name": "bdh_cq", "params": {"vocab_size": 32, "context_length": 64, "n_layer": 2, "n_embd": 64, "n_head": 2}}
        )
        self.assertIsInstance(model, ConfiguredBDHCQ)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.config.vocab_size, 32)
        self.assertEqual(model.config.n_layer, 2)
        self.assertEqual(model.config.n_embd, 64)
        self.assertEqual(model.config.n_head, 2)

    def test_invalid_configurations(self):
        # Auto vocab
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size="auto")

        # Non-positive vocab_size
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=1)

        # Non-positive context_length
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, context_length=0)

        # Non-positive n_layer
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, n_layer=-1)

        # Non-positive n_embd
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, n_embd=0)

        # Non-positive n_head
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, n_head=0)

        # Incompatible n_embd and n_head (not divisible)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, n_embd=65, n_head=4)

        # Invalid dropout
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, dropout=-0.1)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, dropout=1.5)

        # Invalid latent_reasoning_steps
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, latent_reasoning_steps=0)

    def test_parameter_count_and_shapes(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh  # 16 * 64 // 2 = 512

        self.assertEqual(model.embed.weight.shape, (32, 64))
        self.assertEqual(model.encoder.shape, (nh, D, N))  # (2, 64, 512)
        self.assertEqual(model.encoder_v.shape, (nh, D, N))  # (2, 64, 512)
        self.assertEqual(model.decoder.shape, (nh * N, D))  # (1024, 64)
        self.assertEqual(model.lm_head.shape, (D, 32))  # (64, 32)
        self.assertFalse(model.ln.elementwise_affine)

    def test_forward_pass_and_loss(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        targets = torch.randint(0, 32, (batch_size, seq_len))

        # Forward without targets
        logits, loss = model(input_ids)
        self.assertEqual(logits.shape, (batch_size, seq_len, 32))
        self.assertIsNone(loss)

        # Forward with targets
        logits, loss = model(input_ids, targets=targets)
        self.assertEqual(logits.shape, (batch_size, seq_len, 32))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_configured_bdh_cq_lifecycle(self):
        model = ConfiguredBDHCQ(
            vocab_size=32,
            context_length=16,
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
        )

        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        # Test forward
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 8, 32))

        # Test context_length violation
        too_long = torch.randint(0, 32, (2, 20))
        with self.assertRaises(ValueError):
            model(too_long)

        # Test training_step
        batch = {"input_ids": input_ids, "target_ids": target_ids}
        train_out = model.training_step(batch, batch_idx=0)
        self.assertIn("loss", train_out)
        self.assertTrue(torch.isfinite(train_out["loss"]))

        # Test validation_step
        val_out = model.validation_step(batch, batch_idx=0)
        self.assertIn("loss", val_out)
        self.assertTrue(torch.isfinite(val_out["loss"]))

        # Test configure_optimizers
        opt = model.configure_optimizers()
        self.assertIsInstance(opt, torch.optim.Optimizer)

        # Test generation
        prompt = torch.randint(0, 32, (1, 4))
        generated = model.generate(prompt, max_new_tokens=4, temperature=1.0)
        self.assertEqual(generated.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
