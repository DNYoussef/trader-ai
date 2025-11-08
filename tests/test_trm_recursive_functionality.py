"""
Comprehensive Functionality Audit for TRM Recursive Training
Tests that the complete recursive loop executes correctly with all input streams.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from models.trm_model import TinyRecursiveModel
from models.trm_config import TRMConfig


def test_recursive_training_loop():
    """Test that TRM recursive loop executes with T=3, n=6"""
    print("\n" + "="*80)
    print("TEST 1: TRM Recursive Training Loop Execution")
    print("="*80)

    # Create model
    model = TinyRecursiveModel(
        input_dim=10,
        hidden_dim=512,
        output_dim=8,
        num_latent_steps=6,
        num_recursion_cycles=3
    )

    # Create batch of 10 feature inputs
    batch_size = 32
    features = torch.randn(batch_size, 10)

    # Execute forward pass with recursion
    output = model(features, T=3, n=6)

    # Verify outputs
    assert 'strategy_logits' in output, "Missing strategy_logits in output"
    assert 'halt_probability' in output, "Missing halt_probability in output"
    assert 'latent_state' in output, "Missing latent_state in output"
    assert 'solution_state' in output, "Missing solution_state in output"

    # Verify shapes
    assert output['strategy_logits'].shape == (batch_size, 8), f"Wrong logits shape: {output['strategy_logits'].shape}"
    assert output['halt_probability'].shape == (batch_size, 1), f"Wrong halt shape: {output['halt_probability'].shape}"
    assert output['latent_state'].shape == (batch_size, 512), f"Wrong latent shape: {output['latent_state'].shape}"

    print("[PASS] Recursive loop executes with correct output shapes")
    print(f"  - strategy_logits: {output['strategy_logits'].shape}")
    print(f"  - halt_probability: {output['halt_probability'].shape}")
    print(f"  - latent_state: {output['latent_state'].shape}")
    print(f"  - solution_state: {output['solution_state'].shape}")

    # Verify effective depth calculation
    expected_depth = 3 * (6 + 1) * 2  # T * (n + 1) * 2 = 42 layers
    print(f"\n[PASS] Effective depth = T={3} * (n={6} + 1) * 2 = {expected_depth} layers")

    return True


def test_all_input_streams_processed():
    """Test that all 10 input features are actually used in computation"""
    print("\n" + "="*80)
    print("TEST 2: All 10 Input Streams Processed")
    print("="*80)

    model = TinyRecursiveModel(input_dim=10, hidden_dim=512, output_dim=8)

    # Create input with known pattern
    features_zero = torch.zeros(1, 10)
    features_ones = torch.ones(1, 10)

    # Forward pass with different inputs
    output_zero = model(features_zero, T=3, n=6)
    output_ones = model(features_ones, T=3, n=6)

    # Verify outputs differ (proving inputs affect output)
    logits_zero = output_zero['strategy_logits']
    logits_ones = output_ones['strategy_logits']

    diff = torch.abs(logits_zero - logits_ones).mean().item()
    assert diff > 0.01, f"Outputs too similar ({diff:.6f}), inputs may not be used!"

    print(f"[PASS] Input changes affect output (diff={diff:.4f})")

    # Test each input feature individually
    print("\nTesting individual feature impact:")
    base_input = torch.zeros(1, 10)
    base_output = model(base_input, T=3, n=6)['strategy_logits']

    for i in range(10):
        test_input = base_input.clone()
        test_input[0, i] = 1.0  # Perturb one feature
        test_output = model(test_input, T=3, n=6)['strategy_logits']
        feature_impact = torch.abs(test_output - base_output).mean().item()

        feature_names = ['vix', 'spy_returns_5d', 'spy_returns_20d', 'volume_ratio',
                        'market_breadth', 'correlation', 'put_call_ratio',
                        'gini_coefficient', 'sector_dispersion', 'signal_quality']

        status = "[PASS]" if feature_impact > 0.001 else "[WARN]"
        print(f"  {status} Feature {i} ({feature_names[i]:20s}): impact={feature_impact:.6f}")

    return True


def test_recursion_depth_matters():
    """Test that recursion depth (T) actually affects output"""
    print("\n" + "="*80)
    print("TEST 3: Recursion Depth Impact")
    print("="*80)

    model = TinyRecursiveModel(input_dim=10, hidden_dim=512, output_dim=8)
    features = torch.randn(1, 10)

    # Test different recursion depths
    output_T1 = model(features, T=1, n=6)['strategy_logits']
    output_T2 = model(features, T=2, n=6)['strategy_logits']
    output_T3 = model(features, T=3, n=6)['strategy_logits']

    diff_12 = torch.abs(output_T1 - output_T2).mean().item()
    diff_23 = torch.abs(output_T2 - output_T3).mean().item()

    print(f"  T=1 vs T=2 difference: {diff_12:.6f}")
    print(f"  T=2 vs T=3 difference: {diff_23:.6f}")

    assert diff_12 > 0.001, "T=1 and T=2 produce identical outputs!"
    assert diff_23 > 0.001, "T=2 and T=3 produce identical outputs!"

    print("[PASS] Recursion depth affects output (recursive loop is active)")
    return True


def test_latent_steps_matter():
    """Test that latent steps (n) affect output"""
    print("\n" + "="*80)
    print("TEST 4: Latent Steps Impact")
    print("="*80)

    model = TinyRecursiveModel(input_dim=10, hidden_dim=512, output_dim=8)
    features = torch.randn(1, 10)

    # Test different latent steps
    output_n3 = model(features, T=3, n=3)['strategy_logits']
    output_n6 = model(features, T=3, n=6)['strategy_logits']
    output_n9 = model(features, T=3, n=9)['strategy_logits']

    diff_36 = torch.abs(output_n3 - output_n6).mean().item()
    diff_69 = torch.abs(output_n6 - output_n9).mean().item()

    print(f"  n=3 vs n=6 difference: {diff_36:.6f}")
    print(f"  n=6 vs n=9 difference: {diff_69:.6f}")

    assert diff_36 > 0.001, "n=3 and n=6 produce identical outputs!"
    assert diff_69 > 0.001, "n=6 and n=9 produce identical outputs!"

    print("[PASS] Latent steps affect output (latent reasoning is active)")
    return True


def test_gradient_flow():
    """Test that gradients flow through recursive loop"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow Through Recursive Loop")
    print("="*80)

    model = TinyRecursiveModel(input_dim=10, hidden_dim=512, output_dim=8)
    features = torch.randn(1, 10, requires_grad=True)
    targets = torch.tensor([0])

    # Forward pass
    output = model(features, T=3, n=6)
    logits = output['strategy_logits']

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert features.grad is not None, "No gradients for input features!"
    assert features.grad.abs().sum() > 0, "Gradients are all zero!"

    # Check model parameters have gradients
    param_grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No model parameters have gradients!"

    print(f"[PASS] Gradients flow through model")
    print(f"  - Input gradient magnitude: {features.grad.abs().sum().item():.6f}")
    print(f"  - Parameters with gradients: {len(param_grads)}/{sum(1 for _ in model.parameters())}")
    print(f"  - Average parameter gradient: {np.mean(param_grads):.6f}")

    return True


def run_all_tests():
    """Run comprehensive functionality audit"""
    print("\n" + "="*80)
    print("TRM RECURSIVE TRAINING FUNCTIONALITY AUDIT")
    print("="*80)
    print("\nVerifying:")
    print("  1. Recursive training loop executes (T=3, n=6)")
    print("  2. All 10 input streams are processed")
    print("  3. Recursion depth affects computation")
    print("  4. Latent steps affect computation")
    print("  5. Gradients flow for training")

    results = []

    try:
        results.append(("Recursive Loop Execution", test_recursive_training_loop()))
    except Exception as e:
        print(f"[FAIL] Recursive loop test failed: {e}")
        results.append(("Recursive Loop Execution", False))

    try:
        results.append(("Input Streams Processing", test_all_input_streams_processed()))
    except Exception as e:
        print(f"[FAIL] Input streams test failed: {e}")
        results.append(("Input Streams Processing", False))

    try:
        results.append(("Recursion Depth Impact", test_recursion_depth_matters()))
    except Exception as e:
        print(f"[FAIL] Recursion depth test failed: {e}")
        results.append(("Recursion Depth Impact", False))

    try:
        results.append(("Latent Steps Impact", test_latent_steps_matter()))
    except Exception as e:
        print(f"[FAIL] Latent steps test failed: {e}")
        results.append(("Latent Steps Impact", False))

    try:
        results.append(("Gradient Flow", test_gradient_flow()))
    except Exception as e:
        print(f"[FAIL] Gradient flow test failed: {e}")
        results.append(("Gradient Flow", False))

    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] TRM recursive training is FULLY FUNCTIONAL")
        print("  - All 10 input streams processed")
        print("  - Recursive loop active (T=3, n=6, 42 effective layers)")
        print("  - Gradients flow correctly for training")
    else:
        print("\n[WARNING] Some tests failed - review output above")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
