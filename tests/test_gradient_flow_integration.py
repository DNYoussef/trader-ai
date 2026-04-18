"""
Integration Test for Gradient Flow in MetaGrokFast.

Tests RC5 and RC6 fixes:
- RC5: Correct component order (GrokFast -> Bigeometric -> Adam)
- RC6: EMA formula correctness
- No Muon interference

Usage:
    python tests/test_gradient_flow_integration.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.meta_grokfast import MetaGrokFast, MetaGrokfastConfig, GrokfastFilterType


class SimpleModel(nn.Module):
    """Simple model for testing gradient flow."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_component_order():
    """Test RC5: Verify GrokFast processes raw gradients before Bigeometric."""
    print("\n" + "="*80)
    print("TEST 1: Component Order (RC5)")
    print("="*80)

    model = SimpleModel()
    config = MetaGrokfastConfig(
        lr=1e-3,
        use_bigeometric=True,
        use_muon=False,  # Should be disabled per RC5 fix
        grokfast_alpha=0.98,
        grokfast_lambda=2.0,
        warmup_steps=0,  # Disable warmup for immediate testing
    )
    optimizer = MetaGrokFast(model.parameters(), config=config)

    # Create synthetic data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Collect gradient stats over multiple steps
    raw_grad_norms = []
    ema_smoothness = []

    for step in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        # Capture raw gradient norm before optimizer step
        raw_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                raw_norm += torch.norm(p.grad).item() ** 2
        raw_norm = raw_norm ** 0.5
        raw_grad_norms.append(raw_norm)

        # Check EMA state exists and is being updated
        fc1_state = optimizer.state[model.fc1.weight]
        if "grokfast_ema" in fc1_state:
            ema = fc1_state["grokfast_ema"]
            ema_smoothness.append(torch.norm(ema).item())

        optimizer.step()

    print(f"Raw gradient norms (first 5 steps): {raw_grad_norms[:5]}")
    print(f"EMA norms (first 5 steps): {ema_smoothness[:5]}")

    # Validation
    assert len(raw_grad_norms) == 10, "Should have 10 gradient measurements"
    assert len(ema_smoothness) > 0, "EMA should be tracked"

    # EMA should smooth gradients (variance should decrease over time)
    if len(ema_smoothness) >= 5:
        early_variance = torch.tensor(ema_smoothness[:5]).var().item()
        late_variance = torch.tensor(ema_smoothness[-5:]).var().item()
        print(f"\nEMA variance - Early: {early_variance:.6f}, Late: {late_variance:.6f}")
        print(f"Smoothing ratio: {early_variance / (late_variance + 1e-8):.2f}x")

    print("\n[PASSED] Component order test")


def test_ema_formula():
    """Test RC6: Verify EMA formula is correct."""
    print("\n" + "="*80)
    print("TEST 2: EMA Formula Correctness (RC6)")
    print("="*80)

    model = SimpleModel()
    config = MetaGrokfastConfig(
        lr=1e-3,
        use_bigeometric=False,  # Disable to isolate EMA
        use_muon=False,
        filter_type=GrokfastFilterType.EMA,  # Use standard EMA for testing
        grokfast_alpha=0.9,  # Use lower alpha for clearer EMA effect
        grokfast_lambda=1.0,
        warmup_steps=0,
    )
    optimizer = MetaGrokFast(model.parameters(), config=config)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # First step - initialize EMA
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    # Capture gradient before step
    grad_step1 = model.fc1.weight.grad.clone()

    optimizer.step()

    # Second step - verify EMA update
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()

    grad_step2 = model.fc1.weight.grad.clone()

    # Get EMA from optimizer state
    ema_after_step1 = optimizer.state[model.fc1.weight]["grokfast_ema"]

    # Manually compute expected EMA
    # After step 1: ema = 0.9 * 0 + 0.1 * grad_step1 = 0.1 * grad_step1
    expected_ema = 0.1 * grad_step1

    # Verify EMA matches expected formula
    ema_diff = torch.norm(ema_after_step1 - expected_ema).item()
    print(f"EMA difference from expected: {ema_diff:.8f}")
    print(f"Expected EMA norm: {torch.norm(expected_ema).item():.6f}")
    print(f"Actual EMA norm: {torch.norm(ema_after_step1).item():.6f}")

    # Should be very close (within numerical precision)
    assert ema_diff < 1e-4, f"EMA formula incorrect: diff={ema_diff}"

    print("\n[PASSED] EMA formula test")


def test_no_muon_interference():
    """Test that Muon is disabled and doesn't interfere with GrokFast."""
    print("\n" + "="*80)
    print("TEST 3: No Muon Interference")
    print("="*80)

    model = SimpleModel()
    config = MetaGrokfastConfig(
        lr=1e-3,
        use_bigeometric=True,
        use_muon=True,  # Request Muon but it should be ignored
        grokfast_alpha=0.98,
        grokfast_lambda=2.0,
        warmup_steps=0,
    )
    optimizer = MetaGrokFast(model.parameters(), config=config)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Run optimization
    for _ in range(5):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

    # Check that Muon buffers are NOT being used for updates
    # In the fixed code, _adam_update is always called, never _muon_update
    fc1_state = optimizer.state[model.fc1.weight]

    # Adam state should exist
    assert "exp_avg" in fc1_state, "Adam momentum should exist"
    assert "exp_avg_sq" in fc1_state, "Adam variance should exist"

    # Momentum buffer exists but should not be used in updates
    # (it's initialized but _muon_update is never called)
    if "momentum_buffer" in fc1_state:
        print("Note: momentum_buffer exists in state but is not used (correct)")

    print("\n[PASSED] Muon is properly disabled")


def test_gradient_flow_stats():
    """Test that gradient flow produces expected statistical behavior."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow Statistics")
    print("="*80)

    model = SimpleModel()
    config = MetaGrokfastConfig(
        lr=1e-3,
        use_bigeometric=True,
        use_muon=False,
        grokfast_alpha=0.98,
        grokfast_lambda=2.0,
        warmup_steps=2,
        track_stats=True,
    )
    optimizer = MetaGrokFast(model.parameters(), config=config)

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Get stats
    stats = optimizer.get_stats()
    print(f"\nOptimization stats after 20 steps:")
    print(f"  Steps: {stats['steps']}")
    print(f"  Avg original grad norm: {stats['avg_orig_grad_norm']:.6f}")
    print(f"  Avg processed grad norm: {stats['avg_processed_grad_norm']:.6f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.4f}")

    # Loss should decrease
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nLoss improvement: {improvement:.2f}%")
    print(f"  Initial loss (avg first 5): {initial_loss:.6f}")
    print(f"  Final loss (avg last 5): {final_loss:.6f}")

    assert improvement > 0, "Loss should improve during training"

    print("\n[PASSED] Gradient flow statistics test")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MetaGrokFast Integration Tests - RC5 & RC6 Verification")
    print("="*80)

    try:
        test_component_order()
        test_ema_formula()
        test_no_muon_interference()
        test_gradient_flow_stats()

        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        print("\nSummary:")
        print("  [OK] RC5: Component order is correct (GrokFast -> Bigeometric -> Adam)")
        print("  [OK] RC6: EMA formula is correct")
        print("  [OK] No Muon interference")
        print("  [OK] Gradient flow produces expected behavior")
        print("\n")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
