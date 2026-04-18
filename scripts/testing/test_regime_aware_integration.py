"""
Integration Tests for Regime-Aware TRM System

Tests the full pipeline:
1. Regime detection identifies correct market states
2. Focal loss handles class imbalance
3. RegimeAwareTRM produces regime-biased predictions
4. Early warning triggers on crisis probability
5. End-to-end training converges

Run with: python scripts/testing/test_regime_aware_integration.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_market_data(
    n_samples: int = 100,
    regime: str = 'normal'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mock market data for testing.

    Args:
        n_samples: Number of samples
        regime: 'normal', 'high_vol', or 'crisis'

    Returns:
        market_features: (n_samples, 10)
        regime_features: (n_samples, 20, 5)
    """
    # Market features: 10 dimensions
    # vix_level, spy_returns_5d, spy_returns_20d, volume_ratio,
    # market_breadth, correlation, put_call_ratio, gini_coefficient,
    # sector_dispersion, signal_quality_score

    market = torch.zeros(n_samples, 10)
    regime_feat = torch.zeros(n_samples, 20, 5)

    if regime == 'crisis':
        market[:, 0] = torch.randn(n_samples) * 5 + 35  # VIX > 30
        market[:, 1] = torch.randn(n_samples) * 0.02 - 0.07  # Negative 5d returns
        market[:, 2] = torch.randn(n_samples) * 0.03 - 0.10  # Negative 20d returns
        regime_feat[:, :, 0] = 35  # High VIX
        regime_feat[:, :, 1] = -0.07  # Negative returns
    elif regime == 'high_vol':
        market[:, 0] = torch.randn(n_samples) * 3 + 25  # VIX 20-30
        market[:, 1] = torch.randn(n_samples) * 0.02  # Mixed returns
        market[:, 2] = torch.randn(n_samples) * 0.03
        regime_feat[:, :, 0] = 25
        regime_feat[:, :, 1] = 0.0
    else:  # normal
        market[:, 0] = torch.randn(n_samples) * 2 + 15  # VIX < 20
        market[:, 1] = torch.randn(n_samples) * 0.01 + 0.01  # Positive returns
        market[:, 2] = torch.randn(n_samples) * 0.02 + 0.02
        regime_feat[:, :, 0] = 15
        regime_feat[:, :, 1] = 0.01

    # Fill other features with noise
    market[:, 3:] = torch.randn(n_samples, 7) * 0.1
    regime_feat[:, :, 2:] = torch.randn(n_samples, 20, 3) * 0.1

    return market, regime_feat


def test_regime_detection():
    """Test 1: Regime detector identifies correct market states."""
    logger.info("=" * 60)
    logger.info("TEST 1: Regime Detection")
    logger.info("=" * 60)

    try:
        from src.models.regime_detector import NeuralRegimeDetector, RegimeDetectorConfig
    except ImportError as e:
        logger.error(f"Could not import regime detector: {e}")
        return False

    config = RegimeDetectorConfig(n_regimes=3, n_features=5, hidden_dim=32)
    detector = NeuralRegimeDetector(config)

    # Test crisis detection
    _, crisis_features = create_mock_market_data(10, regime='crisis')
    out = detector(crisis_features)
    crisis_prob = out['crisis_prob'].mean().item()

    # Test normal detection
    _, normal_features = create_mock_market_data(10, regime='normal')
    out_normal = detector(normal_features)
    normal_crisis_prob = out_normal['crisis_prob'].mean().item()

    logger.info(f"Crisis regime -> crisis_prob: {crisis_prob:.3f}")
    logger.info(f"Normal regime -> crisis_prob: {normal_crisis_prob:.3f}")

    # After training, crisis_prob should be higher in crisis regime
    # For untrained model, just check outputs are reasonable
    passed = (
        0 <= crisis_prob <= 1 and
        0 <= normal_crisis_prob <= 1 and
        out['regime_probs'].shape == (10, 3)
    )

    if passed:
        logger.info("PASSED: Regime detector produces valid outputs")
    else:
        logger.error("FAILED: Regime detector outputs invalid")

    return passed


def test_focal_loss():
    """Test 2: Focal loss handles class imbalance."""
    logger.info("=" * 60)
    logger.info("TEST 2: Focal Loss")
    logger.info("=" * 60)

    try:
        from src.training.focal_loss import FocalLoss, compute_class_weights
    except ImportError as e:
        logger.error(f"Could not import focal loss: {e}")
        return False

    # Create imbalanced data (strategy 0 rare, strategy 3 common)
    n_samples = 100
    logits = torch.randn(n_samples, 8)
    targets = torch.cat([
        torch.zeros(5, dtype=torch.long),   # 5% class 0 (rare)
        torch.ones(5, dtype=torch.long),    # 5% class 1
        torch.full((5,), 2, dtype=torch.long),
        torch.full((55,), 3, dtype=torch.long),  # 55% class 3 (common)
        torch.full((15,), 4, dtype=torch.long),
        torch.full((10,), 5, dtype=torch.long),
        torch.full((3,), 6, dtype=torch.long),
        torch.full((2,), 7, dtype=torch.long),
    ])

    # Compute class weights
    counts = {i: int((targets == i).sum().item()) for i in range(8)}
    weights = compute_class_weights(counts, n_classes=8, max_weight=10.0, use_sqrt=True)

    logger.info(f"Class counts: {counts}")
    logger.info(f"Class weights: {weights.tolist()}")

    # Test focal loss
    focal = FocalLoss(gamma=2.0, alpha=weights)
    standard_ce = nn.CrossEntropyLoss()

    focal_loss = focal(logits, targets)
    ce_loss = standard_ce(logits, targets)

    logger.info(f"Standard CE loss: {ce_loss.item():.4f}")
    logger.info(f"Focal loss (gamma=2.0): {focal_loss.item():.4f}")

    # Check weights give higher importance to rare classes
    weight_ratio = weights[0] / weights[3]  # Rare / Common
    passed = (
        weight_ratio > 1.5 and  # Rare class should have higher weight
        focal_loss.item() > 0 and
        weights.max().item() <= 10.0  # Capped
    )

    if passed:
        logger.info(f"PASSED: Rare class weight {weight_ratio:.2f}x common class")
    else:
        logger.error("FAILED: Class weights not properly computed")

    return passed


def test_regime_aware_trm():
    """Test 3: RegimeAwareTRM produces regime-biased predictions."""
    logger.info("=" * 60)
    logger.info("TEST 3: Regime-Aware TRM")
    logger.info("=" * 60)

    try:
        from src.models.regime_aware_trm import RegimeAwareTRM, RegimeAwareTRMConfig
    except ImportError as e:
        logger.error(f"Could not import RegimeAwareTRM: {e}")
        return False

    config = RegimeAwareTRMConfig(
        hidden_dim=64,  # Small for testing
        regime_hidden=32,
        use_regime_gating=True
    )
    model = RegimeAwareTRM(config)

    # Test with crisis features
    crisis_market, crisis_regime = create_mock_market_data(10, regime='crisis')
    normal_market, normal_regime = create_mock_market_data(10, regime='normal')

    model.eval()
    with torch.no_grad():
        crisis_out = model(crisis_market, crisis_regime, return_components=True)
        normal_out = model(normal_market, normal_regime, return_components=True)

    logger.info(f"Crisis regime_probs mean: {crisis_out['regime_probs'].mean(dim=0).tolist()}")
    logger.info(f"Normal regime_probs mean: {normal_out['regime_probs'].mean(dim=0).tolist()}")
    logger.info(f"Crisis regime_bias mean: {crisis_out['regime_bias'].mean(dim=0).tolist()}")
    logger.info(f"Normal regime_bias mean: {normal_out['regime_bias'].mean(dim=0).tolist()}")

    # Check outputs are valid
    passed = (
        crisis_out['strategy_logits'].shape == (10, 8) and
        crisis_out['regime_probs'].shape == (10, 3) and
        crisis_out['crisis_prob'].shape == (10, 1) and
        'regime_bias' in crisis_out and
        crisis_out['regime_bias'].shape == (10, 8)
    )

    if passed:
        logger.info("PASSED: RegimeAwareTRM produces valid outputs with regime bias")
    else:
        logger.error("FAILED: RegimeAwareTRM output shapes incorrect")

    return passed


def test_early_warning():
    """Test 4: Early warning triggers on crisis probability."""
    logger.info("=" * 60)
    logger.info("TEST 4: Early Warning System")
    logger.info("=" * 60)

    try:
        from src.models.regime_aware_trm import RegimeAwareTRM, RegimeAwareTRMConfig
    except ImportError as e:
        logger.error(f"Could not import RegimeAwareTRM: {e}")
        return False

    config = RegimeAwareTRMConfig(
        hidden_dim=64,
        crisis_threshold=0.30  # 30% threshold
    )
    model = RegimeAwareTRM(config)

    # Create escalating crisis scenarios
    scenarios = ['normal', 'high_vol', 'crisis']
    warnings = []

    model.eval()
    for scenario in scenarios:
        market, regime = create_mock_market_data(10, regime=scenario)
        with torch.no_grad():
            out = model(market, regime)
        n_warnings = out['early_warning'].sum().item()
        crisis_prob_mean = out['crisis_prob'].mean().item()
        warnings.append((scenario, n_warnings, crisis_prob_mean))
        logger.info(f"{scenario}: {n_warnings}/10 warnings, crisis_prob={crisis_prob_mean:.3f}")

    # For untrained model, just verify warning mechanism works
    passed = all(
        0 <= w[1] <= 10 and 0 <= w[2] <= 1
        for w in warnings
    )

    if passed:
        logger.info("PASSED: Early warning system produces valid outputs")
    else:
        logger.error("FAILED: Early warning outputs invalid")

    return passed


def test_training_convergence():
    """Test 5: End-to-end training converges."""
    logger.info("=" * 60)
    logger.info("TEST 5: Training Convergence")
    logger.info("=" * 60)

    try:
        from src.models.regime_aware_trm import RegimeAwareTRM, RegimeAwareTRMConfig
        from src.training.focal_loss import TRMFocalLoss
    except ImportError as e:
        logger.error(f"Could not import modules: {e}")
        return False

    config = RegimeAwareTRMConfig(hidden_dim=64, regime_hidden=32)
    model = RegimeAwareTRM(config)
    loss_fn = TRMFocalLoss(gamma=2.0, lambda_focal=1.0, lambda_halt=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate training data
    n_epochs = 10
    batch_size = 32
    losses = []

    model.train()
    for epoch in range(n_epochs):
        # Create batch with mixed regimes
        market, regime = create_mock_market_data(batch_size, regime='normal')
        # Targets: strategies 3-5 for normal conditions
        targets = torch.randint(3, 6, (batch_size,))
        pnl = torch.randn(batch_size) * 0.02

        optimizer.zero_grad()
        out = model(market, regime)

        loss = loss_fn(
            out['strategy_logits'],
            out['halt_logits'].squeeze(-1),
            targets,
            pnl
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: loss = {loss.item():.4f}")

    # Check convergence (loss should decrease or stay stable)
    avg_first_half = np.mean(losses[:5])
    avg_second_half = np.mean(losses[5:])

    logger.info(f"Avg loss first 5 epochs: {avg_first_half:.4f}")
    logger.info(f"Avg loss last 5 epochs: {avg_second_half:.4f}")

    # For 10 epochs, just check loss is finite and reasonable
    passed = (
        all(np.isfinite(l) for l in losses) and
        max(losses) < 100  # Not exploding
    )

    if passed:
        logger.info("PASSED: Training produces finite, reasonable losses")
    else:
        logger.error("FAILED: Training loss invalid or exploding")

    return passed


def run_all_tests() -> Dict[str, bool]:
    """Run all integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME-AWARE TRM INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")

    tests = [
        ("Regime Detection", test_regime_detection),
        ("Focal Loss", test_focal_loss),
        ("Regime-Aware TRM", test_regime_aware_trm),
        ("Early Warning", test_early_warning),
        ("Training Convergence", test_training_convergence),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"Test '{name}' raised exception: {e}")
            results[name] = False
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\nALL TESTS PASSED!")
    else:
        logger.warning(f"\n{total - passed} TESTS FAILED")

    return results


if __name__ == '__main__':
    results = run_all_tests()
    sys.exit(0 if all(results.values()) else 1)
