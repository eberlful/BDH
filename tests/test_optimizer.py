"""Tests for Adafactor optimizer, optimizer factory, and model optimizer configuration."""

import unittest
import torch
from torch import nn

from src.model.bdh import ConfiguredBDH
from src.model.bdh_cq import ConfiguredBDHCQ
from src.model.gpt import GPTModel
from src.optim.adafactor import Adafactor
from src.optim.factory import build_optimizer


class TestAdafactor(unittest.TestCase):
    def test_adafactor_init_and_defaults(self) -> None:
        param = nn.Parameter(torch.randn(10, 20))
        opt = Adafactor([param], lr=1e-3, weight_decay=0.01)
        self.assertEqual(opt.defaults["lr"], 1e-3)
        self.assertEqual(opt.defaults["weight_decay"], 0.01)
        self.assertFalse(opt.defaults["scale_parameter"])
        self.assertFalse(opt.defaults["relative_step"])

    def test_adafactor_step_1d_and_2d(self) -> None:
        torch.manual_seed(42)
        linear = nn.Linear(16, 32, bias=True)
        opt = Adafactor(linear.parameters(), lr=1e-2, weight_decay=0.01)

        x = torch.randn(4, 16)
        loss = linear(x).sum()
        loss.backward()

        opt.step()

        # Weight (2D) should have factored second moments
        weight_state = opt.state[linear.weight]
        self.assertIn("exp_avg_sq_row", weight_state)
        self.assertIn("exp_avg_sq_col", weight_state)
        self.assertNotIn("exp_avg_sq", weight_state)
        self.assertEqual(weight_state["exp_avg_sq_row"].shape, (32,))
        self.assertEqual(weight_state["exp_avg_sq_col"].shape, (16,))

        # Bias (1D) should have standard non-factored second moments
        bias_state = opt.state[linear.bias]
        self.assertIn("exp_avg_sq", bias_state)
        self.assertNotIn("exp_avg_sq_row", bias_state)
        self.assertEqual(bias_state["exp_avg_sq"].shape, (32,))

    def test_adafactor_optimization_convergence(self) -> None:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = Adafactor(model.parameters(), lr=0.05, weight_decay=0.0)

        x = torch.randn(32, 8)
        y = torch.randn(32, 1)

        initial_loss = nn.functional.mse_loss(model(x), y).item()
        for _ in range(50):
            opt.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        self.assertLess(final_loss, initial_loss * 0.5)

    def test_adafactor_state_dict_serialization(self) -> None:
        torch.manual_seed(42)
        linear = nn.Linear(8, 8)
        opt1 = Adafactor(linear.parameters(), lr=1e-3, beta1=0.9)

        x = torch.randn(2, 8)
        linear(x).sum().backward()
        opt1.step()

        state_dict = opt1.state_dict()

        opt2 = Adafactor(linear.parameters(), lr=1e-3, beta1=0.9)
        opt2.load_state_dict(state_dict)

        p = list(linear.parameters())[0]
        self.assertEqual(opt1.state[p]["step"], opt2.state[p]["step"])
        self.assertTrue(torch.equal(opt1.state[p]["exp_avg"], opt2.state[p]["exp_avg"]))


class TestOptimizerFactory(unittest.TestCase):
    def test_default_adamw(self) -> None:
        linear = nn.Linear(8, 8)
        opt = build_optimizer(linear.parameters())
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(opt.defaults["lr"], 3e-4)
        self.assertEqual(opt.defaults["weight_decay"], 0.1)

    def test_adafactor_string_name(self) -> None:
        linear = nn.Linear(8, 8)
        opt = build_optimizer(
            linear.parameters(),
            optimizer="adafactor",
            learning_rate=1e-3,
            weight_decay=0.01,
            optimizer_params={"clip_threshold": 0.5},
        )
        self.assertIsInstance(opt, Adafactor)
        self.assertEqual(opt.defaults["lr"], 1e-3)
        self.assertEqual(opt.defaults["weight_decay"], 0.01)
        self.assertEqual(opt.defaults["clip_threshold"], 0.5)

    def test_dict_configuration(self) -> None:
        linear = nn.Linear(8, 8)
        opt = build_optimizer(
            linear.parameters(),
            optimizer={"name": "adafactor", "params": {"beta1": 0.9}},
            learning_rate=5e-4,
        )
        self.assertIsInstance(opt, Adafactor)
        self.assertEqual(opt.defaults["lr"], 5e-4)
        self.assertEqual(opt.defaults["beta1"], 0.9)

    def test_unsupported_optimizer(self) -> None:
        linear = nn.Linear(8, 8)
        with self.assertRaises(ValueError):
            build_optimizer(linear.parameters(), optimizer="unknown_opt")


class TestModelOptimizerIntegration(unittest.TestCase):
    def test_bdh_default_optimizer_is_adamw(self) -> None:
        model = ConfiguredBDH(vocab_size=64, context_length=32, n_layer=2, n_embd=64, n_head=2)
        opt = model.configure_optimizers()
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_bdh_configured_adafactor(self) -> None:
        model = ConfiguredBDH(
            vocab_size=64,
            context_length=32,
            n_layer=2,
            n_embd=64,
            n_head=2,
            optimizer="adafactor",
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        opt = model.configure_optimizers()
        self.assertIsInstance(opt, Adafactor)
        self.assertEqual(opt.defaults["lr"], 1e-3)
        self.assertEqual(opt.defaults["weight_decay"], 0.01)

    def test_bdh_cq_configured_adafactor(self) -> None:
        model = ConfiguredBDHCQ(
            vocab_size=64,
            context_length=32,
            n_layer=2,
            n_embd=64,
            n_head=2,
            optimizer="adafactor",
            learning_rate=2e-3,
        )
        opt = model.configure_optimizers()
        self.assertIsInstance(opt, Adafactor)
        self.assertEqual(opt.defaults["lr"], 2e-3)

    def test_gpt_configured_adafactor(self) -> None:
        model = GPTModel(
            vocab_size=64,
            context_length=32,
            d_model=64,
            n_heads=2,
            n_layers=2,
            optimizer="adafactor",
        )
        opt = model.configure_optimizers()
        self.assertIsInstance(opt, Adafactor)


if __name__ == "__main__":
    unittest.main()
