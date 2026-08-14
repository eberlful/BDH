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

    def test_contextual_memory_shape_and_constant_footprint(self):
        config = BDHCQConfig(
            n_layer=3,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        # Short demonstrations
        demo_short = torch.randint(0, 32, (2, 4))
        mem_short = model.encode_contextual_memory(demo_short)
        self.assertEqual(len(mem_short), 3)
        for rho in mem_short:
            self.assertEqual(rho.shape, (2, nh, N, D))

        # Long demonstrations
        demo_long = torch.randint(0, 32, (2, 48))
        mem_long = model.encode_contextual_memory(demo_long)
        self.assertEqual(len(mem_long), 3)
        for rho in mem_long:
            self.assertEqual(rho.shape, (2, nh, N, D))

    def test_contextual_memory_alters_downstream_readout(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)

        torch.manual_seed(42)
        demo_a = torch.randint(0, 32, (1, 8))
        demo_b = torch.randint(0, 32, (1, 8))
        query = torch.randint(0, 32, (1, 4))

        seq_a = torch.cat([demo_a, query], dim=1)
        seq_b = torch.cat([demo_b, query], dim=1)

        logits_a, _ = model(seq_a, demo_len=8)
        logits_b, _ = model(seq_b, demo_len=8)

        # Query logits should differ because demonstrations differ
        query_logits_a = logits_a[:, 8:, :]
        query_logits_b = logits_b[:, 8:, :]
        self.assertFalse(torch.allclose(query_logits_a, query_logits_b, atol=1e-4))

    def test_contextual_memory_precomputed_equivalence(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)
        model.eval()

        demo = torch.randint(0, 32, (1, 6))
        query = torch.randint(0, 32, (1, 4))
        seq = torch.cat([demo, query], dim=1)

        # Forward with demo_len
        logits_full, _ = model(seq, demo_len=6)
        query_logits_full = logits_full[:, 6:, :]

        # Forward with precomputed contextual memory
        mem = model.encode_contextual_memory(demo)
        logits_query, _ = model(query, contextual_memory=mem)

        self.assertTrue(torch.allclose(query_logits_full, logits_query, atol=1e-5))

    def test_gradient_flow_through_contextual_memory(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
        )
        model = BDHCQ(config)

        seq = torch.randint(0, 32, (2, 10))
        targets = torch.randint(0, 32, (2, 10))

        logits, loss = model(seq, targets=targets, demo_len=6)
        loss.backward()

        self.assertIsNotNone(model.encoder.grad)
        self.assertIsNotNone(model.encoder_v.grad)
        self.assertIsNotNone(model.decoder.grad)
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertIsNotNone(model.lm_head.grad)
        self.assertTrue(torch.count_nonzero(model.encoder.grad) > 0)
        self.assertTrue(torch.count_nonzero(model.encoder_v.grad) > 0)
        self.assertTrue(torch.count_nonzero(model.decoder.grad) > 0)


    def test_dynamic_reasoning_steps_override(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
            latent_reasoning_steps=1,
        )
        model = BDHCQ(config)
        model.eval()

        input_ids = torch.randint(0, 32, (1, 8))

        # Test default R=1
        logits_r1, _ = model(input_ids)

        # Dynamically evaluate with R=3
        logits_r3, _ = model(input_ids, latent_reasoning_steps=3)

        self.assertEqual(logits_r1.shape, (1, 8, 32))
        self.assertEqual(logits_r3.shape, (1, 8, 32))
        # R=3 applies additional recurrent passes so output should differ from R=1
        self.assertFalse(torch.allclose(logits_r1, logits_r3, atol=1e-4))

        # Invalid R raises ValueError
        with self.assertRaises(ValueError):
            model(input_ids, latent_reasoning_steps=0)

    def test_iterative_latent_refinement_and_intermediate_logits(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
            latent_reasoning_steps=3,
        )
        model = BDHCQ(config)
        model.eval()

        demo = torch.randint(0, 32, (1, 6))
        query = torch.randint(0, 32, (1, 4))
        seq = torch.cat([demo, query], dim=1)

        # Forward returning intermediate logits across R=3 passes
        logits, _, intermediate_logits = model(
            seq, demo_len=6, return_intermediate_logits=True
        )

        self.assertEqual(len(intermediate_logits), 3)
        for r_step, step_logits in enumerate(intermediate_logits):
            self.assertEqual(step_logits.shape, (1, 10, 32))

        # The final step logits should match the primary return logits
        self.assertTrue(torch.allclose(logits, intermediate_logits[-1], atol=1e-5))

        # Successive passes refine query representation, so intermediate query logits should differ
        query_step0 = intermediate_logits[0][:, 6:, :]
        query_step1 = intermediate_logits[1][:, 6:, :]
        query_step2 = intermediate_logits[2][:, 6:, :]

        self.assertFalse(torch.allclose(query_step0, query_step1, atol=1e-4))
        self.assertFalse(torch.allclose(query_step1, query_step2, atol=1e-4))

        # Demonstration logits are computed during ingestion and remain stable
        demo_step0 = intermediate_logits[0][:, :6, :]
        demo_step1 = intermediate_logits[1][:, :6, :]
        self.assertTrue(torch.allclose(demo_step0, demo_step1, atol=1e-5))

    def test_hybrid_attention_with_precomputed_memory_and_recurrent_steps(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
            latent_reasoning_steps=2,
        )
        model = BDHCQ(config)
        model.eval()

        demo = torch.randint(0, 32, (1, 6))
        query = torch.randint(0, 32, (1, 4))
        seq = torch.cat([demo, query], dim=1)

        # Full sequence forward
        logits_full, _ = model(seq, demo_len=6, latent_reasoning_steps=2)
        query_logits_full = logits_full[:, 6:, :]

        # Precomputed memory forward with same R
        mem = model.encode_contextual_memory(demo)
        query_logits_mem, _ = model(
            query, contextual_memory=mem, latent_reasoning_steps=2
        )

        self.assertTrue(
            torch.allclose(query_logits_full, query_logits_mem, atol=1e-5)
        )

    def test_gradient_flow_through_multiple_recurrent_reasoning_steps(self):
        config = BDHCQConfig(
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            vocab_size=32,
            dropout=0.0,
            latent_reasoning_steps=3,
        )
        model = BDHCQ(config)

        seq = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))

        logits, loss = model(
            seq, targets=targets, demo_len=4, latent_reasoning_steps=3
        )
        loss.backward()

        self.assertIsNotNone(model.encoder.grad)
        self.assertIsNotNone(model.encoder_v.grad)
        self.assertIsNotNone(model.decoder.grad)
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertIsNotNone(model.lm_head.grad)
        self.assertTrue(torch.count_nonzero(model.encoder.grad) > 0)
        self.assertTrue(torch.count_nonzero(model.encoder_v.grad) > 0)
        self.assertTrue(torch.count_nonzero(model.decoder.grad) > 0)

    def test_autoregressive_generation_with_recurrent_latent_reasoning(self):
        model = ConfiguredBDHCQ(
            vocab_size=32,
            context_length=32,
            n_layer=2,
            n_embd=64,
            n_head=2,
            mlp_internal_dim_multiplier=16,
            latent_reasoning_steps=2,
        )

        # Generation without demonstrations but with R=2
        prompt = torch.randint(0, 32, (1, 4))
        generated_r2 = model.generate(
            prompt, max_new_tokens=4, latent_reasoning_steps=2
        )
        self.assertEqual(generated_r2.shape, (1, 8))

        # Generation with demonstrations and R=3
        demo_and_query = torch.randint(0, 32, (1, 8))  # 5 demo + 3 query
        generated_demo = model.generate(
            demo_and_query, max_new_tokens=4, demo_len=5, latent_reasoning_steps=3
        )
        self.assertEqual(generated_demo.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()


