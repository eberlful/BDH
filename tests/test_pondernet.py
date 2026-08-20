import unittest
import torch
from torch.nn import functional as F

from src.core.registry import MODEL_REGISTRY, load_builtin_components
from src.model.bdh_cq import (
    BDHCQ,
    BDHCQConfig,
    ConfiguredBDHCQ,
    compute_geometric_prior,
    compute_halting_probabilities,
    compute_ponder_kl_loss,
    compute_masked_cross_entropy_per_sample,
)


class TestPonderNetComponents(unittest.TestCase):
    def test_masked_cross_entropy_ignores_prompt_positions_in_mean(self):
        # Only one of four positions is a target. The masked positions must
        # not dilute the loss for the valid solution token.
        logits = torch.zeros(1, 4, 3)
        target_ids = torch.tensor([[0, -100, -100, -100]])

        loss = compute_masked_cross_entropy_per_sample(logits, target_ids)

        expected = torch.log(torch.tensor(3.0))
        self.assertTrue(torch.allclose(loss, expected.unsqueeze(0)))

    def test_masked_cross_entropy_is_zero_without_valid_targets(self):
        logits = torch.zeros(1, 2, 3)
        target_ids = torch.full((1, 2), -100)

        loss = compute_masked_cross_entropy_per_sample(logits, target_ids)

        self.assertEqual(loss.item(), 0.0)

    def test_geometric_prior(self):
        # Edge case: R = 1
        prior_1 = compute_geometric_prior(1, 0.2)
        self.assertEqual(prior_1.shape, (1,))
        self.assertAlmostEqual(prior_1.item(), 1.0, places=5)

        # R = 4, lambda_p = 0.3
        R = 4
        lambda_p = 0.3
        prior = compute_geometric_prior(R, lambda_p)
        self.assertEqual(prior.shape, (R,))
        self.assertAlmostEqual(float(prior.sum().item()), 1.0, places=5)
        self.assertTrue((prior >= 0.0).all())

        # Invalid arguments
        with self.assertRaises(ValueError):
            compute_geometric_prior(0, 0.2)
        with self.assertRaises(ValueError):
            compute_geometric_prior(4, 0.0)
        with self.assertRaises(ValueError):
            compute_geometric_prior(4, 1.5)

    def test_halting_probabilities_computation(self):
        B, R = 3, 4
        torch.manual_seed(42)
        lambdas = torch.sigmoid(torch.randn(B, R))

        probs, cum_probs = compute_halting_probabilities(lambdas)
        self.assertEqual(probs.shape, (B, R))
        self.assertEqual(cum_probs.shape, (B, R))

        # Check that probabilities sum to 1 per sample
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Check manual calculation for step 0 and 1
        expected_p0 = lambdas[:, 0]
        self.assertTrue(torch.allclose(probs[:, 0], expected_p0, atol=1e-5))

        expected_p1 = (1.0 - lambdas[:, 0]) * lambdas[:, 1]
        self.assertTrue(torch.allclose(probs[:, 1], expected_p1, atol=1e-5))

    def test_ponder_kl_loss(self):
        B, R = 4, 5
        torch.manual_seed(42)
        raw_lambdas = torch.randn(B, R, requires_grad=True)
        lambdas = torch.sigmoid(raw_lambdas)
        probs, _ = compute_halting_probabilities(lambdas)

        kl = compute_ponder_kl_loss(probs, lambda_p=0.2)
        self.assertTrue(torch.isfinite(kl))
        self.assertGreaterEqual(float(kl.item()), 0.0)

        # Test gradient flow
        kl.backward()
        self.assertIsNotNone(raw_lambdas.grad)
        self.assertTrue(torch.isfinite(raw_lambdas.grad).all())


class TestBDHCQPonderNet(unittest.TestCase):
    def setUp(self):
        load_builtin_components()

    def test_bdhcq_ponder_disabled_by_default(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            latent_reasoning_steps=3,
        )
        model = BDHCQ(config)
        self.assertFalse(config.enable_pondernet)
        self.assertFalse(hasattr(model, "halt_head"))

    def test_bdhcq_ponder_enabled_architecture(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            latent_reasoning_steps=4,
            enable_pondernet=True,
            ponder_lambda_p=0.25,
            ponder_beta=0.02,
        )
        model = BDHCQ(config)
        self.assertTrue(hasattr(model, "halt_head"))

        input_ids = torch.randint(0, 32, (2, 8))
        logits, loss, inter_logits, ponder_probs, lambdas = model(
            input_ids,
            return_intermediate_logits=True,
            return_ponder_info=True,
        )

        self.assertEqual(logits.shape, (2, 8, 32))
        self.assertEqual(len(inter_logits), 4)
        self.assertEqual(ponder_probs.shape, (2, 4))
        self.assertEqual(lambdas.shape, (2, 4))
        self.assertTrue(
            torch.allclose(ponder_probs.sum(dim=-1), torch.ones(2), atol=1e-5)
        )

    def test_configured_bdhcq_ponder_training_step_and_gradients(self):
        model = ConfiguredBDHCQ(
            vocab_size=32,
            context_length=16,
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            latent_reasoning_steps=3,
            enable_pondernet=True,
            ponder_lambda_p=0.2,
            ponder_beta=0.05,
        )

        batch_size, seq_len = 3, 8
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        target_ids = torch.randint(0, 32, (batch_size, seq_len))
        batch = {"input_ids": input_ids, "target_ids": target_ids}

        # Training step
        out = model.training_step(batch, batch_idx=0)
        self.assertIn("loss", out)
        self.assertIn("logits", out)
        self.assertIn("ponder/task_loss", out)
        self.assertIn("ponder/kl_loss", out)
        self.assertIn("ponder/expected_steps", out)

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(out["ponder/expected_steps"], 0.0)

        # Backward pass
        out["loss"].backward()
        self.assertIsNotNone(model.network.halt_head.weight.grad)
        self.assertTrue(torch.isfinite(model.network.halt_head.weight.grad).all())
        self.assertIsNotNone(model.network.encoder.grad)
        self.assertTrue(torch.isfinite(model.network.encoder.grad).all())

    def test_configured_bdhcq_registry_instantiation(self):
        model = MODEL_REGISTRY.instantiate(
            {
                "name": "bdh_cq",
                "params": {
                    "vocab_size": 32,
                    "context_length": 64,
                    "n_layer": 2,
                    "n_embd": 64,
                    "n_head": 2,
                    "latent_reasoning_steps": 4,
                    "enable_pondernet": True,
                    "ponder_lambda_p": 0.3,
                    "ponder_beta": 0.01,
                },
            }
        )
        self.assertIsInstance(model, ConfiguredBDHCQ)
        self.assertTrue(model.config.enable_pondernet)
        self.assertEqual(model.config.ponder_lambda_p, 0.3)
        self.assertEqual(model.config.ponder_beta, 0.01)

    def test_invalid_ponder_parameters(self):
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, ponder_lambda_p=0.0)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, ponder_lambda_p=1.2)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, ponder_beta=-0.1)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, ponder_halt_threshold=0.0)
        with self.assertRaises(ValueError):
            ConfiguredBDHCQ(vocab_size=32, ponder_halt_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
