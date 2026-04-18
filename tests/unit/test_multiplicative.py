"""
Unit tests for src/utils/multiplicative.py

Tests all NNC formulas (F1-F7) and utility classes.
Verifies numerical stability, edge cases, and formula correctness.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1
"""

import pytest
import numpy as np
from typing import List

# Import the module under test
import sys
sys.path.insert(0, str(__file__).replace('tests/unit/test_multiplicative.py', ''))
from src.utils.multiplicative import (
    # Constants
    NUMERICAL_EPSILON,
    MAX_EXP_ARG,
    # Classes
    GeometricDerivative,
    BigeometricDerivative,
    MetaFriedmannFormulas,
    KEvolution,
    PrelecWeighting,
    GeometricOperations,
    MultiplicativeRisk,
    BoundedNNC,
    BoundedNNCConfig,
    # Convenience functions
    geometric_mean,
    geometric_return,
    prelec_weight,
    k_for_capital,
    compound_survival,
)


# =============================================================================
# FORMULA F1: GEOMETRIC DERIVATIVE TESTS
# =============================================================================

class TestGeometricDerivative:
    """Test F1: D_G[f](a) = exp(f'(a) / f(a))"""

    def test_constant_function(self):
        """D_G[constant] = 1 (since f' = 0)"""
        geo = GeometricDerivative()
        x = np.linspace(1, 10, 100)
        f = np.ones_like(x) * 5.0  # constant function f(x) = 5

        result = geo(f, x)
        # f' = 0, so D_G = exp(0) = 1
        assert np.allclose(result, 1.0, atol=0.01)

    def test_exponential_function(self):
        """D_G[e^(kx)] = e^k (CONSTANT, independent of x)"""
        geo = GeometricDerivative()
        k = 0.5
        x = np.linspace(1, 10, 100)
        f = np.exp(k * x)

        result = geo(f, x)
        expected = np.exp(k)  # Should be constant ~1.649

        # Skip boundary points (gradient is less accurate there)
        assert np.allclose(result[10:-10], expected, rtol=0.05)

    def test_power_function(self):
        """D_G[x^n] = exp(n/x) for x > 0"""
        geo = GeometricDerivative()
        n = 2.0
        x = np.linspace(1, 10, 100)
        f = x ** n

        result = geo(f, x)
        expected = np.exp(n / x)  # D_G[x^2] = exp(2/x)

        assert np.allclose(result[10:-10], expected[10:-10], rtol=0.05)

    def test_near_zero_handling(self):
        """Should handle f(x) near zero without division by zero"""
        geo = GeometricDerivative()
        x = np.linspace(-1, 1, 100)
        f = x ** 2  # f(0) = 0

        result = geo(f, x)
        # Should not contain inf or nan
        assert np.all(np.isfinite(result))

    def test_single_point(self):
        """Test at_point method"""
        geo = GeometricDerivative()
        f_prime = 1.0
        f_val = 2.0

        result = geo.at_point(f_prime, f_val)
        expected = np.exp(f_prime / f_val)

        assert np.isclose(result, expected)


# =============================================================================
# FORMULA F2: BIGEOMETRIC DERIVATIVE TESTS
# =============================================================================

class TestBigeometricDerivative:
    """Test F2: D_BG[f](a) = exp(a * f'(a) / f(a))"""

    def test_power_law_theorem(self):
        """D_BG[x^n] = e^n (CONSTANT for all x > 0!)"""
        bigeo = BigeometricDerivative()
        n = 3.0
        x = np.linspace(1, 10, 100)
        f = x ** n

        result = bigeo(f, x)
        expected = np.exp(n)  # Should be constant ~20.09

        # This is the key property! It's constant regardless of x
        assert np.allclose(result[10:-10], expected, rtol=0.1)

    def test_exponential_function(self):
        """D_BG[e^x] = exp(x * 1) = e^x"""
        bigeo = BigeometricDerivative()
        x = np.linspace(0.1, 3, 100)  # Avoid x=0
        f = np.exp(x)

        result = bigeo(f, x)
        expected = np.exp(x)

        assert np.allclose(result[10:-10], expected[10:-10], rtol=0.1)

    def test_elasticity_calculation(self):
        """Test elasticity = x * f'(x) / f(x)"""
        bigeo = BigeometricDerivative()
        n = 2.0
        x = np.linspace(1, 10, 100)
        f = x ** n

        elasticity = bigeo.elasticity(f, x)

        # For f = x^n, elasticity = n (constant)
        assert np.allclose(elasticity[10:-10], n, rtol=0.1)

    def test_positive_x_required(self):
        """x values should be positive for bigeometric"""
        bigeo = BigeometricDerivative()
        x = np.linspace(1, 10, 100)
        f = x ** 2

        result = bigeo(f, x)
        assert np.all(np.isfinite(result))

    def test_at_point(self):
        """Test at_point method"""
        bigeo = BigeometricDerivative()
        x = 2.0
        f_prime = 4.0  # f = x^2, f' = 2x = 4 at x=2
        f_val = 4.0     # f(2) = 4

        result = bigeo.at_point(x, f_prime, f_val)
        # elasticity = 2 * 4 / 4 = 2
        # D_BG = e^2 ~ 7.389
        expected = np.exp(2)

        assert np.isclose(result, expected)


# =============================================================================
# FORMULAS F4, F5: META-FRIEDMANN TESTS
# =============================================================================

class TestMetaFriedmannFormulas:
    """Test F4 and F5: Expansion and density exponents"""

    def test_expansion_exponent_classical(self):
        """k=0 gives classical formula: n = 2 / (3 * (1+w))"""
        # Matter-dominated (w=0): n = 2/3
        n = MetaFriedmannFormulas.expansion_exponent(k=0, w=0)
        assert np.isclose(n, 2/3)

        # Radiation-dominated (w=1/3): n = 1/2
        n = MetaFriedmannFormulas.expansion_exponent(k=0, w=1/3)
        assert np.isclose(n, 0.5)

    def test_expansion_exponent_k_equals_1(self):
        """k=1 gives n=0 (no growth)"""
        n = MetaFriedmannFormulas.expansion_exponent(k=1, w=0)
        assert np.isclose(n, 0)

    def test_expansion_exponent_w_negative_one(self):
        """w=-1 is a singularity (inf)"""
        n = MetaFriedmannFormulas.expansion_exponent(k=0, w=-1)
        assert n == float('inf')

    def test_density_exponent_k_zero(self):
        """k=0: m=2 (classical singularity)"""
        m = MetaFriedmannFormulas.density_exponent(k=0)
        assert np.isclose(m, 2)

    def test_density_exponent_k_one(self):
        """k=1: m=0 (constant density, NO singularity)"""
        m = MetaFriedmannFormulas.density_exponent(k=1)
        assert np.isclose(m, 0)

    def test_density_exponent_k_greater_than_one(self):
        """k>1: m<0 (density vanishes at t=0)"""
        m = MetaFriedmannFormulas.density_exponent(k=1.5)
        assert m < 0
        assert np.isclose(m, -1)

    def test_meta_hubble_positive_t(self):
        """H_meta = n * t^(k-1)"""
        n = 0.5
        k = 0.5
        t = 2.0

        h = MetaFriedmannFormulas.meta_hubble(n, k, t)
        expected = n * t ** (k - 1)

        assert np.isclose(h, expected)

    def test_meta_hubble_t_zero(self):
        """Edge cases at t=0"""
        # k < 1: inf
        assert MetaFriedmannFormulas.meta_hubble(n=1, k=0.5, t=0) == float('inf')
        # k > 1: 0
        assert MetaFriedmannFormulas.meta_hubble(n=1, k=1.5, t=0) == 0
        # k = 1: n
        assert MetaFriedmannFormulas.meta_hubble(n=0.5, k=1, t=0) == 0.5


# =============================================================================
# FORMULA F6: k(L) SPATIAL PATTERN TESTS
# =============================================================================

class TestKEvolution:
    """Test F6: k(L) = -0.0137 * log10(L) + 0.1593"""

    def test_formula_correctness(self):
        """Verify the formula matches expected values"""
        # At L = 10^10: k = -0.0137 * 10 + 0.1593 = 0.0223
        k = KEvolution.k_spatial(1e10)
        expected = -0.0137 * 10 + 0.1593
        assert np.isclose(k, expected)

        # At L = 1: k = -0.0137 * 0 + 0.1593 = 0.1593
        k = KEvolution.k_spatial(1)
        assert np.isclose(k, 0.1593)

    def test_k_decreases_with_scale(self):
        """k should decrease as L increases"""
        k_small = KEvolution.k_spatial(1e3)
        k_large = KEvolution.k_spatial(1e12)

        assert k_small > k_large

    def test_k_bounded(self):
        """k should be clipped to [0, 1]"""
        # Very small L might give k > 1
        k = KEvolution.k_spatial(1e-100)
        assert 0 <= k <= 1

        # Very large L might give k < 0
        k = KEvolution.k_spatial(1e100)
        assert 0 <= k <= 1

    def test_k_for_gate(self):
        """k varies appropriately by gate capital"""
        k_g0 = KEvolution.k_for_gate(200)      # G0: $200
        k_g12 = KEvolution.k_for_gate(10_000_000)  # G12: $10M

        # G0 should have higher k (more classical, conservative)
        # G12 should have lower k (more meta, aggressive)
        assert k_g0 > k_g12

    def test_invalid_input(self):
        """Handle L <= 0"""
        k = KEvolution.k_spatial(0)
        assert k == KEvolution.K_MAX

        k = KEvolution.k_spatial(-10)
        assert k == KEvolution.K_MAX


# =============================================================================
# FORMULA F7: PRELEC WEIGHTING TESTS
# =============================================================================

class TestPrelecWeighting:
    """Test F7: w(p) = exp(-(-ln(p))^alpha)"""

    def test_boundary_cases(self):
        """p=0 -> 0, p=1 -> 1"""
        assert PrelecWeighting.weight(0) == 0
        assert PrelecWeighting.weight(1) == 1

    def test_identity_at_alpha_1(self):
        """alpha=1 gives w(p) = p"""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w = PrelecWeighting.weight(p, alpha=1.0)
            assert np.isclose(w, p, rtol=0.01)

    def test_overweight_small_probabilities(self):
        """alpha < 1 overweights small probabilities"""
        p = 0.1
        w = PrelecWeighting.weight(p, alpha=0.65)
        # Small p should be overweighted (w > p)
        assert w > p

    def test_underweight_small_probabilities(self):
        """alpha > 1 underweights small probabilities"""
        p = 0.1
        w = PrelecWeighting.weight(p, alpha=1.5)
        # Small p should be underweighted (w < p)
        assert w < p

    def test_inverse_roundtrip(self):
        """Inverse should recover original probability"""
        p_original = 0.3
        alpha = 0.65

        w = PrelecWeighting.weight(p_original, alpha)
        p_recovered = PrelecWeighting.inverse(w, alpha)

        assert np.isclose(p_original, p_recovered, rtol=0.01)


# =============================================================================
# GEOMETRIC OPERATIONS TESTS
# =============================================================================

class TestGeometricOperations:
    """Test geometric mean and related operations"""

    def test_geometric_mean_basic(self):
        """Geometric mean of [2, 8] = 4"""
        result = GeometricOperations.geometric_mean([2, 8])
        assert np.isclose(result, 4)

    def test_geometric_mean_single_value(self):
        """Geometric mean of [5] = 5"""
        result = GeometricOperations.geometric_mean([5])
        assert np.isclose(result, 5)

    def test_geometric_mean_empty(self):
        """Empty list returns 1"""
        result = GeometricOperations.geometric_mean([])
        assert result == 1.0

    def test_geometric_mean_vs_arithmetic(self):
        """Geometric mean <= arithmetic mean (AM-GM inequality)"""
        values = [1, 2, 3, 4, 5]
        geo = GeometricOperations.geometric_mean(values)
        arith = np.mean(values)

        assert geo <= arith

    def test_geometric_mean_return(self):
        """Test return calculation"""
        returns = [0.10, -0.05, 0.03]  # 10%, -5%, 3%

        result = GeometricOperations.geometric_mean_return(returns)

        # Manual calculation
        # (1.10 * 0.95 * 1.03)^(1/3) - 1
        multipliers = [1.10, 0.95, 1.03]
        expected = np.prod(multipliers) ** (1/3) - 1

        assert np.isclose(result, expected)

    def test_cagr(self):
        """Test CAGR calculation"""
        start = 1000
        end = 2000
        years = 5

        cagr = GeometricOperations.cagr(start, end, years)
        expected = (2000/1000) ** (1/5) - 1  # ~14.87%

        assert np.isclose(cagr, expected)


# =============================================================================
# MULTIPLICATIVE RISK TESTS
# =============================================================================

class TestMultiplicativeRisk:
    """Test multiplicative risk compounding"""

    def test_compound_survival(self):
        """Survival = product of factors"""
        factors = [0.99, 0.95, 0.98]
        result = MultiplicativeRisk.compound_survival(factors)
        expected = 0.99 * 0.95 * 0.98

        assert np.isclose(result, expected)

    def test_compound_survival_empty(self):
        """Empty list returns 1"""
        result = MultiplicativeRisk.compound_survival([])
        assert result == 1.0

    def test_amplification_factor(self):
        """1% daily risk over 100 days"""
        survival = MultiplicativeRisk.amplification_factor(0.01, 100)
        expected = 0.99 ** 100  # ~0.366

        assert np.isclose(survival, expected)

        # Contrast with (wrong) additive: 1 - 100*0.01 = 0
        additive_wrong = 1 - 100 * 0.01
        assert additive_wrong < survival  # Additive overestimates risk

    def test_effective_daily_risk(self):
        """Calculate daily risk for target survival"""
        target = 0.5  # 50% survival
        days = 100

        daily_risk = MultiplicativeRisk.effective_daily_risk(target, days)

        # Verify: (1 - daily_risk)^100 should equal ~0.5
        survival = (1 - daily_risk) ** days
        assert np.isclose(survival, target, rtol=0.01)

    def test_risk_validation(self):
        """Invalid inputs should raise errors"""
        with pytest.raises(ValueError):
            MultiplicativeRisk.amplification_factor(-0.1, 10)

        with pytest.raises(ValueError):
            MultiplicativeRisk.amplification_factor(1.5, 10)

        with pytest.raises(ValueError):
            MultiplicativeRisk.effective_daily_risk(0, 100)

        with pytest.raises(ValueError):
            MultiplicativeRisk.effective_daily_risk(0.5, -10)


# =============================================================================
# BOUNDED NNC TESTS
# =============================================================================

class TestBoundedNNC:
    """Test safe wrapper for NNC operations"""

    def test_safe_exp_no_overflow(self):
        """Large inputs should not overflow"""
        nnc = BoundedNNC()
        result = nnc.safe_exp(10000)  # Would normally overflow

        assert np.isfinite(result)
        assert result <= np.exp(MAX_EXP_ARG)

    def test_safe_log_no_underflow(self):
        """Zero/negative inputs should be handled"""
        nnc = BoundedNNC()
        result = nnc.safe_log(0)

        assert np.isfinite(result)

    def test_safe_log_negative(self):
        """Negative input handled gracefully"""
        nnc = BoundedNNC()
        result = nnc.safe_log(-5)

        assert np.isfinite(result)

    def test_star_derivative_method_selection(self):
        """star_derivative method selection"""
        nnc = BoundedNNC()
        x = np.linspace(1, 10, 50)
        f = x ** 2

        geo = nnc.star_derivative(f, x, method='geometric')
        bigeo = nnc.star_derivative(f, x, method='bigeometric')

        # Results should be different
        assert not np.allclose(geo, bigeo)

        # Invalid method raises error
        with pytest.raises(ValueError):
            nnc.star_derivative(f, x, method='invalid')

    def test_custom_config(self):
        """Custom configuration works"""
        config = BoundedNNCConfig(
            epsilon=1e-6,
            max_exp_arg=500
        )
        nnc = BoundedNNC(config)

        assert nnc.config.epsilon == 1e-6
        assert nnc.config.max_exp_arg == 500


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_geometric_mean_shorthand(self):
        result = geometric_mean([2, 8])
        assert np.isclose(result, 4)

    def test_geometric_return_shorthand(self):
        result = geometric_return([0.10, -0.05])
        assert np.isfinite(result)

    def test_prelec_weight_shorthand(self):
        result = prelec_weight(0.1)
        assert 0 < result < 1

    def test_k_for_capital_shorthand(self):
        result = k_for_capital(1000)
        assert 0 <= result <= 1

    def test_compound_survival_shorthand(self):
        result = compound_survival([0.9, 0.9])
        assert np.isclose(result, 0.81)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test numerical stability edge cases"""

    def test_nav_zero_handling(self):
        """NAV = 0 should not crash"""
        nnc = BoundedNNC()

        # Log of zero
        result = nnc.safe_log(0)
        assert np.isfinite(result)

        # Geometric mean with zero
        result = geometric_mean([0, 1, 2])
        assert np.isfinite(result)

    def test_returns_negative_100_percent(self):
        """Handle -100% return (total loss)"""
        returns = [0.10, -1.0, 0.05]  # -100% = total loss
        result = geometric_return(returns)

        # Should handle gracefully (multiplier = 0)
        assert np.isfinite(result)

    def test_extreme_probabilities(self):
        """Handle p very close to 0 or 1"""
        result = prelec_weight(1e-300)
        assert np.isfinite(result)

        result = prelec_weight(1 - 1e-15)
        assert np.isfinite(result)

    def test_large_number_of_periods(self):
        """Handle many periods in risk calculation"""
        survival = MultiplicativeRisk.amplification_factor(0.001, 10000)
        assert np.isfinite(survival)
        assert survival > 0

    def test_array_operations_stability(self):
        """Test stability with array operations"""
        geo = GeometricDerivative()
        x = np.linspace(0.01, 10, 1000)
        f = np.exp(-x)  # Decaying function

        result = geo(f, x)
        assert np.all(np.isfinite(result))


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
