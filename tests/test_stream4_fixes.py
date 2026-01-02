"""
STREAM 4 FIXES - Validation Tests

Tests for RC8, RC9, RC10 mathematical fixes.
Run with: pytest tests/test_stream4_fixes.py -v
"""

import numpy as np
import torch
import pytest


class TestRC8BigeometricDocs:
    """Test RC8: Bigeometric documentation now matches actual behavior."""

    def test_amplification_k_greater_than_half(self):
        """Verify k > 0.5 AMPLIFIES large gradients (per corrected docs)."""
        from src.training.bigeometric import BigeometricTransform

        transform = BigeometricTransform()
        grad = torch.tensor([10.0])
        k = 0.7

        grad_meta = transform.transform(grad, k)

        # Exponent: 2*0.7 - 1 = 0.4
        # Scale: 10^0.4 = 2.512
        # Expected: 10 * 2.512 = 25.12
        assert grad_meta.item() > grad.item(), "k=0.7 should AMPLIFY large gradients"
        assert abs(grad_meta.item() - 25.12) < 0.1, \
            f"Expected ~25.12, got {grad_meta.item()}"

    def test_dampening_k_less_than_half(self):
        """Verify k < 0.5 DAMPENS large gradients (per corrected docs)."""
        from src.training.bigeometric import BigeometricTransform

        transform = BigeometricTransform()
        grad = torch.tensor([10.0])
        k = 0.3

        grad_meta = transform.transform(grad, k)

        # Exponent: 2*0.3 - 1 = -0.4
        # Scale: 10^-0.4 = 0.398
        # Expected: 10 * 0.398 = 3.98
        assert grad_meta.item() < grad.item(), "k=0.3 should DAMPEN large gradients"
        assert abs(grad_meta.item() - 3.98) < 0.1, \
            f"Expected ~3.98, got {grad_meta.item()}"

    def test_identity_k_equals_half(self):
        """Verify k = 0.5 is identity transform (per corrected docs)."""
        from src.training.bigeometric import BigeometricTransform

        transform = BigeometricTransform()
        grad = torch.tensor([10.0])
        k = 0.5

        grad_meta = transform.transform(grad, k)

        # Exponent: 2*0.5 - 1 = 0.0
        # Scale: 10^0.0 = 1.0
        # Expected: 10 * 1.0 = 10.0
        assert abs(grad_meta.item() - grad.item()) < 1e-6, \
            "k=0.5 should be IDENTITY"

    def test_mathematical_property(self):
        """Verify mathematical property: scale = |g|^(2k-1)."""
        from src.training.bigeometric import BigeometricTransform

        transform = BigeometricTransform()

        test_cases = [
            (5.0, 0.2),   # Small k
            (5.0, 0.5),   # Identity k
            (5.0, 0.8),   # Large k
            (100.0, 0.3), # Large gradient, small k
            (100.0, 0.7), # Large gradient, large k
        ]

        for g_val, k_val in test_cases:
            grad = torch.tensor([g_val])
            k = k_val

            grad_meta = transform.transform(grad, k)

            # Expected scale: |g|^(2k-1)
            exponent = 2 * k - 1
            expected_scale = abs(g_val) ** exponent
            expected_meta = g_val * expected_scale

            assert abs(grad_meta.item() - expected_meta) < 1e-4, \
                f"g={g_val}, k={k_val}: expected {expected_meta}, got {grad_meta.item()}"


class TestRC9KFormulaWarnings:
    """Test RC9: k(L) formula has appropriate warnings about domain transfer."""

    def test_uses_log10_not_ln(self):
        """Verify formula uses log10 (not ln) to prevent 35% error."""
        from src.training.k_formula import compute_k

        L = 100.0
        k = compute_k(L)

        # With log10: k = -0.0137 * log10(100) + 0.1593
        #               = -0.0137 * 2.0 + 0.1593
        #               = 0.1319
        k_expected_log10 = -0.0137 * np.log10(L) + 0.1593

        # With ln (WRONG): k = -0.0137 * ln(100) + 0.1593
        #                    = -0.0137 * 4.605 + 0.1593
        #                    = 0.0962
        k_wrong_ln = -0.0137 * np.log(L) + 0.1593

        assert abs(k - k_expected_log10) < 1e-6, "Should use log10"
        assert abs(k - k_wrong_ln) > 0.03, \
            f"Should NOT use ln (difference should be >3%)"

    def test_documentation_has_warnings(self):
        """Verify k_formula.py contains warnings about domain transfer."""
        import inspect
        from src.training import k_formula

        docstring = inspect.getdoc(k_formula)

        # Should contain warning about domain transfer
        assert "WARNING" in docstring, \
            "Module docstring should contain WARNING"

        # Should mention validation status
        assert any(phrase in docstring for phrase in [
            "NOT YET VALIDATED",
            "NOT VALIDATED",
            "not validated"
        ]), "Should warn about lack of ML validation"

        # Should mention log10 vs ln issue
        assert "log10" in docstring or "ln" in docstring, \
            "Should mention logarithm base issue"

    def test_k_decreases_with_L(self):
        """Verify k(L) has inverse relationship (higher L -> lower k)."""
        from src.training.k_formula import compute_k

        L_small = 0.01
        L_medium = 1.0
        L_large = 100.0

        k_small = compute_k(L_small)
        k_medium = compute_k(L_medium)
        k_large = compute_k(L_large)

        # k should decrease as L increases (inverse relationship)
        assert k_small > k_medium > k_large, \
            f"k should decrease with L: {k_small} > {k_medium} > {k_large}"

    def test_k_stays_in_bounds(self):
        """Verify k(L) is always clamped to [0, 1]."""
        from src.training.k_formula import compute_k

        # Test extreme L values
        L_values = [1e-10, 1e-5, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e10]

        for L in L_values:
            k = compute_k(L)
            assert 0.0 <= k <= 1.0, \
                f"k should be in [0,1], got {k} for L={L}"


class TestRC10ProfitFactorFinite:
    """Test RC10: Profit factor returns finite values (not inf)."""

    def test_no_losses_returns_finite(self):
        """Verify PF returns finite 1e6 (not inf) when losses=0."""
        from src.intelligence.validation.objectives_numba import profit_factor_core

        # Perfect strategy: all wins, no losses
        returns_perfect = np.array([0.01, 0.02, 0.0, 0.03, 0.01])
        pf = profit_factor_core(returns_perfect)

        assert np.isfinite(pf), "PF should be finite (not inf)"
        assert pf == 1e6, f"PF should be exactly 1e6, got {pf}"

    def test_no_trades_returns_zero(self):
        """Verify PF returns 0 when no trades (all zeros)."""
        from src.intelligence.validation.objectives_numba import profit_factor_core

        returns_none = np.array([0.0, 0.0, 0.0])
        pf = profit_factor_core(returns_none)

        assert pf == 0.0, f"PF should be 0.0 for no trades, got {pf}"

    def test_normal_calculation_correct(self):
        """Verify PF calculates correctly for normal case (gains and losses)."""
        from src.intelligence.validation.objectives_numba import profit_factor_core

        # Gains: 0.01 + 0.02 + 0.015 = 0.045
        # Losses: 0.005 + 0.01 = 0.015
        # PF: 0.045 / 0.015 = 3.0
        returns_normal = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        pf = profit_factor_core(returns_normal)

        expected_pf = 3.0
        assert abs(pf - expected_pf) < 1e-6, \
            f"Expected PF={expected_pf}, got {pf}"

    def test_optimization_safe(self):
        """Verify PF values work safely in optimization algorithms."""
        from src.intelligence.validation.objectives_numba import profit_factor_core

        # Mix of strategies including perfect one
        strategies = [
            np.array([0.01, -0.005, 0.02, -0.01]),  # PF ~2.0
            np.array([0.01, 0.02, 0.03]),            # PF = 1e6 (perfect)
            np.array([0.005, -0.01, 0.015, -0.005]), # PF ~1.33
        ]

        pf_values = [profit_factor_core(s) for s in strategies]

        # All finite
        assert all(np.isfinite(pf) for pf in pf_values), \
            "All PF values should be finite"

        # Can sort without issues
        sorted_pf = sorted(pf_values)
        assert sorted_pf[0] < sorted_pf[1] < sorted_pf[2], \
            "Should be sortable"

        # Can compute gradients without NaN
        gradient = np.gradient(pf_values)
        assert all(np.isfinite(g) for g in gradient), \
            "Gradients should be finite (no NaN propagation)"

        # Can compute mean without pollution
        mean_pf = np.mean(pf_values)
        assert np.isfinite(mean_pf), "Mean should be finite"

        # Perfect strategy should rank highest
        assert pf_values[1] == max(pf_values), \
            "Perfect strategy (no losses) should have highest PF"

    def test_edge_case_tiny_losses(self):
        """Verify PF handles very small (but nonzero) losses correctly."""
        from src.intelligence.validation.objectives_numba import profit_factor_core

        # Very small losses (below 1e-10 threshold triggers cap)
        returns_tiny_losses = np.array([0.01, -1e-11, 0.02])
        pf = profit_factor_core(returns_tiny_losses)

        # Should trigger the "losses < 1e-10" cap
        assert pf == 1e6, \
            f"Tiny losses (<1e-10) should trigger cap, got PF={pf}"

        # Slightly larger losses (above threshold)
        returns_small_losses = np.array([0.01, -1e-9, 0.02])
        pf2 = profit_factor_core(returns_small_losses)

        # Should calculate normally (not capped)
        assert pf2 < 1e6, \
            f"Small losses (>1e-10) should calculate normally, got PF={pf2}"
        assert np.isfinite(pf2), "Should still be finite"


class TestIntegrationStreamFixes:
    """Integration tests verifying all fixes work together."""

    def test_bigeometric_with_adaptive_k(self):
        """Test bigeometric transform with k from k_formula."""
        from src.training.bigeometric import BigeometricTransform
        from src.training.k_formula import compute_k

        transform = BigeometricTransform()

        # Large gradient -> low k -> dampening
        grad_large = torch.tensor([100.0])
        L_large = 100.0
        k_large = compute_k(L_large)  # Should be ~0.13

        grad_meta_large = transform.transform(grad_large, k_large)

        # k ~0.13 < 0.5, so should dampen
        assert grad_meta_large.item() < grad_large.item(), \
            f"Large gradient with k={k_large} should dampen"

        # Small gradient -> high k -> still dampens (but less)
        grad_small = torch.tensor([1.0])
        L_small = 1.0
        k_small = compute_k(L_small)  # Should be ~0.16

        grad_meta_small = transform.transform(grad_small, k_small)

        # k ~0.16 < 0.5, still dampening (but less than k=0.13)
        assert grad_meta_small.item() < grad_small.item(), \
            f"Small gradient with k={k_small} should dampen"

    def test_profit_factor_in_objective_wrapper(self):
        """Test profit_factor wrapper function (not just numba core)."""
        from src.intelligence.validation.objectives import profit_factor

        # Test through the main API
        returns_perfect = np.array([0.01, 0.02, 0.03])
        pf = profit_factor(returns_perfect)

        assert np.isfinite(pf), "API wrapper should return finite PF"
        assert pf == 1e6, f"API wrapper should cap at 1e6, got {pf}"

    def test_all_fixes_documented(self):
        """Verify STREAM4-FIXES.md exists and contains all fixes."""
        import os

        docs_path = "D:\\Projects\\trader-ai\\docs\\STREAM4-FIXES.md"
        assert os.path.exists(docs_path), \
            "STREAM4-FIXES.md should exist"

        with open(docs_path, 'r') as f:
            content = f.read()

        # Should document all three RCs
        assert "RC8" in content, "Should document RC8"
        assert "RC9" in content, "Should document RC9"
        assert "RC10" in content, "Should document RC10"

        # Should have mathematical proofs
        assert "Proof" in content or "proof" in content, \
            "Should contain mathematical proofs"

        # Should have test recommendations
        assert "test" in content.lower(), \
            "Should contain test recommendations"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
