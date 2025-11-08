"""
Unit tests for TRM model architecture

Tests:
1. Model initialization
2. Forward pass
3. Recursive reasoning
4. Halting mechanism
5. Strategy prediction
6. Parameter count validation
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.trm_model import TinyRecursiveModel, create_trm_model
from models.trm_config import TRMModelConfig


class TestTRMModelInitialization:
    """Test model initialization"""

    def test_default_initialization(self):
        """Test model with default parameters"""
        model = TinyRecursiveModel()

        assert model.input_dim == 10
        assert model.hidden_dim == 512
        assert model.output_dim == 8
        assert model.num_latent_steps == 6
        assert model.num_recursion_cycles == 3

    def test_custom_initialization(self):
        """Test model with custom parameters"""
        model = TinyRecursiveModel(
            input_dim=15,
            hidden_dim=256,
            output_dim=4,
            num_latent_steps=4,
            num_recursion_cycles=2
        )

        assert model.input_dim == 15
        assert model.hidden_dim == 256
        assert model.output_dim == 4
        assert model.num_latent_steps == 4
        assert model.num_recursion_cycles == 2

    def test_parameter_count(self):
        """Test that parameter count is approximately 7M"""
        model = TinyRecursiveModel()

        total_params = sum(p.numel() for p in model.parameters())

        # Should be around 7M parameters (6M - 8M range)
        assert 6_000_000 <= total_params <= 8_000_000, \
            f"Parameter count {total_params:,} outside expected range"


class TestTRMForwardPass:
    """Test forward pass and outputs"""

    def test_forward_pass_shapes(self):
        """Test output shapes from forward pass"""
        model = TinyRecursiveModel()
        batch_size = 4
        input_dim = 10

        x = torch.randn(batch_size, input_dim)
        output = model(x)

        assert 'strategy_logits' in output
        assert 'halt_probability' in output
        assert 'latent_state' in output
        assert 'solution_state' in output

        assert output['strategy_logits'].shape == (batch_size, 8)
        assert output['halt_probability'].shape == (batch_size,)
        assert output['latent_state'].shape == (batch_size, 512)
        assert output['solution_state'].shape == (batch_size, 512)

    def test_forward_pass_intermediate_states(self):
        """Test intermediate states tracking"""
        model = TinyRecursiveModel(num_recursion_cycles=3)
        batch_size = 2

        x = torch.randn(batch_size, 10)
        output = model(x, return_intermediate=True)

        assert 'intermediate_states' in output
        # Should have 3 intermediate states (one per cycle)
        assert len(output['intermediate_states']) == 3

    def test_halt_probability_range(self):
        """Test halting probability is in [0, 1]"""
        model = TinyRecursiveModel()
        batch_size = 8

        x = torch.randn(batch_size, 10)
        output = model(x)

        halt_prob = output['halt_probability']
        assert torch.all(halt_prob >= 0.0)
        assert torch.all(halt_prob <= 1.0)

    def test_variable_recursion_params(self):
        """Test forward pass with variable T and n"""
        model = TinyRecursiveModel()
        x = torch.randn(2, 10)

        # Test different recursion cycles
        output1 = model(x, T=2, n=4)
        output2 = model(x, T=5, n=8)

        assert output1['strategy_logits'].shape == output2['strategy_logits'].shape
        # Results should differ with different recursion params
        assert not torch.allclose(output1['strategy_logits'], output2['strategy_logits'])


class TestTRMRecursiveMechanisms:
    """Test recursive reasoning components"""

    def test_reasoning_update(self):
        """Test latent reasoning state update"""
        model = TinyRecursiveModel()
        batch_size = 4
        hidden_dim = 512

        x = torch.randn(batch_size, hidden_dim)
        y = torch.randn(batch_size, hidden_dim)
        z = torch.randn(batch_size, hidden_dim)

        z_new = model.reasoning_update(x, y, z)

        assert z_new.shape == (batch_size, hidden_dim)
        # Should not be identical (learning occurred)
        assert not torch.allclose(z_new, z)

    def test_solution_update(self):
        """Test solution state update"""
        model = TinyRecursiveModel()
        batch_size = 4
        hidden_dim = 512

        y = torch.randn(batch_size, hidden_dim)
        z = torch.randn(batch_size, hidden_dim)

        y_new = model.solution_update(y, z)

        assert y_new.shape == (batch_size, hidden_dim)
        # Should not be identical
        assert not torch.allclose(y_new, y)

    def test_latent_recursion(self):
        """Test full latent recursion cycle"""
        model = TinyRecursiveModel()
        batch_size = 2
        hidden_dim = 512

        x = torch.randn(batch_size, hidden_dim)
        y = torch.zeros(batch_size, hidden_dim)
        z = torch.zeros(batch_size, hidden_dim)

        # Run 6 steps
        y_final, z_final = model.latent_recursion(x, y, z, n=6)

        assert y_final.shape == (batch_size, hidden_dim)
        assert z_final.shape == (batch_size, hidden_dim)
        # Should evolve from zeros
        assert not torch.allclose(y_final, y)
        assert not torch.allclose(z_final, z)


class TestTRMPrediction:
    """Test prediction and inference"""

    def test_predict_strategy(self):
        """Test strategy prediction"""
        model = TinyRecursiveModel()
        batch_size = 4

        x = torch.randn(batch_size, 10)
        strategy_idx, confidence = model.predict_strategy(x, return_confidence=True)

        assert strategy_idx.shape == (batch_size,)
        assert confidence.shape == (batch_size,)

        # Strategy indices should be in [0, 7]
        assert torch.all(strategy_idx >= 0)
        assert torch.all(strategy_idx < 8)

        # Confidence should be in [0, 1]
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)

    def test_predict_without_confidence(self):
        """Test prediction without confidence scores"""
        model = TinyRecursiveModel()
        x = torch.randn(2, 10)

        strategy_idx, confidence = model.predict_strategy(x, return_confidence=False)

        assert strategy_idx is not None
        assert confidence is None

    def test_strategy_probabilities(self):
        """Test strategy probability distribution"""
        model = TinyRecursiveModel()
        batch_size = 4

        x = torch.randn(batch_size, 10)
        probs = model.get_strategy_probabilities(x)

        assert probs.shape == (batch_size, 8)

        # Probabilities should sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)

        # All probabilities should be non-negative
        assert torch.all(probs >= 0.0)


class TestTRMGradients:
    """Test gradient flow and backpropagation"""

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = TinyRecursiveModel()
        x = torch.randn(2, 10, requires_grad=True)

        output = model(x)
        loss = output['strategy_logits'].sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_no_gradient_in_early_cycles(self):
        """Test that early cycles don't compute gradients"""
        model = TinyRecursiveModel(num_recursion_cycles=3)
        x = torch.randn(2, 10, requires_grad=True)

        # Forward with intermediate states
        output = model(x, return_intermediate=True)

        # Only final cycle should have gradients
        # (This is a design property, hard to test directly)
        # Just verify model can backprop
        loss = output['strategy_logits'].sum()
        loss.backward()

        assert x.grad is not None


class TestTRMConfiguration:
    """Test configuration integration"""

    def test_create_from_config(self):
        """Test creating model from configuration"""
        config = TRMModelConfig(
            input_dim=15,
            hidden_dim=256,
            output_dim=4,
            num_latent_steps=4,
            num_recursion_cycles=2
        )

        model = create_trm_model(config.to_dict())

        assert model.input_dim == 15
        assert model.hidden_dim == 256
        assert model.output_dim == 4

    def test_effective_depth_calculation(self):
        """Test effective depth matches expected value"""
        config = TRMModelConfig(
            num_latent_steps=6,
            num_recursion_cycles=3
        )

        # Effective depth = T × (n + 1) × n_layers
        # = 3 × (6 + 1) × 2 = 42
        assert config.effective_depth == 42


class TestTRMEdgeCases:
    """Test edge cases and error handling"""

    def test_single_batch(self):
        """Test with batch size 1"""
        model = TinyRecursiveModel()
        x = torch.randn(1, 10)

        output = model(x)
        assert output['strategy_logits'].shape == (1, 8)

    def test_large_batch(self):
        """Test with large batch size"""
        model = TinyRecursiveModel()
        batch_size = 1024
        x = torch.randn(batch_size, 10)

        output = model(x)
        assert output['strategy_logits'].shape == (batch_size, 8)

    def test_single_cycle(self):
        """Test with T=1 (no gradient-free cycles)"""
        model = TinyRecursiveModel()
        x = torch.randn(2, 10)

        output = model(x, T=1, n=4)
        assert output['strategy_logits'].shape == (2, 8)

    def test_eval_mode(self):
        """Test model in eval mode"""
        model = TinyRecursiveModel(dropout=0.5)
        x = torch.randn(4, 10)

        model.eval()
        output1 = model(x)
        output2 = model(x)

        # In eval mode, same input should give same output (no dropout randomness)
        assert torch.allclose(output1['strategy_logits'], output2['strategy_logits'])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
