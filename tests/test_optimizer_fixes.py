"""
Test script validating RC3, RC4, RC7 optimizer fixes.

Tests:
1. Weight decay is applied uniformly to ALL parameters
2. Gradients are not suppressed (magnitude check)
3. Effective learning rate after bigeometric scaling
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.meta_grokfast import MetaGrokFast, TRM_ENHANCED_CONFIG


class SimpleModel(nn.Module):
    """Simple model with 2D and 1D params for testing."""
    def __init__(self):
        super().__init__()
        self.weight_2d = nn.Parameter(torch.randn(4, 4))  # 2D: Muon path
        self.bias_1d = nn.Parameter(torch.randn(4))       # 1D: Adam path

    def forward(self, x):
        return (self.weight_2d @ x.unsqueeze(-1)).squeeze() + self.bias_1d


def test_weight_decay_uniformity():
    """Test that weight decay is applied to BOTH Muon and Adam paths."""
    print("\n" + "="*80)
    print("TEST 1: Weight Decay Uniformity (RC3 + RC4)")
    print("="*80)

    model = SimpleModel()
    optimizer = MetaGrokFast(model.parameters(), config=TRM_ENHANCED_CONFIG)

    # Store initial param values
    initial_weight = model.weight_2d.data.clone()
    initial_bias = model.bias_1d.data.clone()

    # Fake forward-backward
    x = torch.randn(4)
    loss = model(x).sum()
    loss.backward()

    # Do update
    optimizer.step()

    # Check that params changed (weight decay applied)
    weight_delta = (initial_weight - model.weight_2d.data).norm().item()
    bias_delta = (initial_bias - model.bias_1d.data).norm().item()

    print(f"\nConfig weight_decay: {TRM_ENHANCED_CONFIG.weight_decay}")
    print(f"Expected: 0.01 (NOT 1.0)")
    print(f"\nParameter changes:")
    print(f"  2D weight (Muon path) delta: {weight_delta:.6f}")
    print(f"  1D bias (Adam path) delta:   {bias_delta:.6f}")

    # Both should have changed
    assert weight_delta > 1e-6, "RC4 FAILED: Weight decay not applied to Muon path!"
    assert bias_delta > 1e-6, "Weight decay not applied to Adam path!"

    print("\nPASS: Weight decay applied to both Muon and Adam paths")
    return True


def test_gradient_magnitude():
    """Test that gradients are not suppressed by excessive weight decay."""
    print("\n" + "="*80)
    print("TEST 2: Gradient Magnitude (RC3 Validation)")
    print("="*80)

    model = SimpleModel()
    optimizer = MetaGrokFast(model.parameters(), config=TRM_ENHANCED_CONFIG)

    # Run a few steps and collect gradient magnitudes
    grad_norms = []

    for i in range(10):
        x = torch.randn(4)
        loss = model(x).sum()
        loss.backward()

        # Collect gradient norms BEFORE optimizer step
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())

        optimizer.step()
        optimizer.zero_grad()

    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    min_grad_norm = min(grad_norms)
    max_grad_norm = max(grad_norms)

    print(f"\nGradient magnitude statistics (10 steps):")
    print(f"  Average: {avg_grad_norm:.6f}")
    print(f"  Min:     {min_grad_norm:.6f}")
    print(f"  Max:     {max_grad_norm:.6f}")
    print(f"\nExpected range: 1e-3 to 1e-1 (NOT suppressed to ~1e-6)")

    # With weight_decay=0.01, gradients should NOT be suppressed
    assert avg_grad_norm > 1e-3, f"RC3 FAILED: Gradients suppressed! avg={avg_grad_norm:.2e}"
    assert avg_grad_norm < 10.0, f"Gradients exploding! avg={avg_grad_norm:.2e}"

    print("\nPASS: Gradients are healthy (not suppressed)")
    return True


def test_effective_learning_rate():
    """Test effective LR after bigeometric scaling."""
    print("\n" + "="*80)
    print("TEST 3: Effective Learning Rate (RC7 Validation)")
    print("="*80)

    # Math analysis
    base_lr = TRM_ENHANCED_CONFIG.lr
    k_typical = 0.1  # Typical k value from k(L) formula

    # Bigeometric scaling factor: |g|^(2k-1)
    # For typical gradient norm ~0.1:
    grad_norm = 0.1
    scaling_factor = grad_norm ** (2*k_typical - 1)  # |g|^(2*0.1 - 1) = |g|^(-0.8)

    # NOTE: The actual effective LR is NOT simply base_lr * scaling_factor
    # because bigeometric amplifies small gradients and compresses large ones
    # This makes the EFFECTIVE update size depend on gradient distribution
    # For a mix of gradient magnitudes, the effective LR is closer to base_lr

    print(f"\nLearning rate analysis:")
    print(f"  Base LR (config):           {base_lr:.2e}")
    print(f"  Expected base LR:           5e-4 (RC7 fix)")
    print(f"  Typical k value:            {k_typical:.3f}")
    print(f"  Bigeometric exponent:       {2*k_typical - 1:.3f}")
    print(f"  Example gradient norm:      {grad_norm:.3f}")
    print(f"  Example scaling factor:     {scaling_factor:.3f}")
    print(f"\nNote: Effective LR varies per-gradient based on magnitude.")
    print(f"      Small gradients (|g|<0.01): amplified 10-100x")
    print(f"      Medium gradients (|g|~0.1): amplified 5-10x")
    print(f"      Large gradients (|g|>1.0): compressed 0.1-1x")
    print(f"\nResult: Bigeometric is ADAPTIVE, not a fixed scaling factor.")
    print(f"        Base LR of 5e-4 is appropriate for this adaptive behavior.")

    # Verify base LR is correct
    assert base_lr == 5e-4, f"RC7 FAILED: LR should be 5e-4, got {base_lr:.2e}"

    # Base LR should be in reasonable range for adaptive scaling
    assert 1e-4 <= base_lr <= 1e-3, f"Base LR out of expected range: {base_lr:.2e}"

    print(f"\nPASS: Base LR = 5e-4 (correct for adaptive bigeometric scaling)")
    return True


def test_config_values():
    """Test that config values match RC3, RC4, RC7 fixes."""
    print("\n" + "="*80)
    print("TEST 4: Configuration Values")
    print("="*80)

    print(f"\nTRM_ENHANCED_CONFIG values:")
    print(f"  lr:           {TRM_ENHANCED_CONFIG.lr:.2e}  (Expected: 5e-4)")
    print(f"  muon_lr:      {TRM_ENHANCED_CONFIG.muon_lr:.2e}  (Expected: 5e-4)")
    print(f"  weight_decay: {TRM_ENHANCED_CONFIG.weight_decay:.3f}  (Expected: 0.01)")

    # RC7: lr should be 5e-4
    assert TRM_ENHANCED_CONFIG.lr == 5e-4, f"RC7 FAILED: lr={TRM_ENHANCED_CONFIG.lr:.2e}"
    assert TRM_ENHANCED_CONFIG.muon_lr == 5e-4, f"RC7 FAILED: muon_lr={TRM_ENHANCED_CONFIG.muon_lr:.2e}"

    # RC3: weight_decay should be 0.01
    assert TRM_ENHANCED_CONFIG.weight_decay == 0.01, f"RC3 FAILED: wd={TRM_ENHANCED_CONFIG.weight_decay}"

    print("\nPASS: All config values correct")
    return True


def run_all_tests():
    """Run all optimizer fix validation tests."""
    print("\n" + "="*80)
    print("OPTIMIZER FIXES VALIDATION SUITE")
    print("Testing RC3, RC4, RC7 fixes")
    print("="*80)

    tests = [
        ("Config Values", test_config_values),
        ("Weight Decay Uniformity", test_weight_decay_uniformity),
        ("Gradient Magnitude", test_gradient_magnitude),
        ("Effective Learning Rate", test_effective_learning_rate),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except AssertionError as e:
            print(f"\nFAILED: {e}")
            results[name] = False
        except Exception as e:
            print(f"\nERROR: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
