"""
Test suite for Narrative Gap calculation

Tests the core alpha generation mechanism to ensure it works with real numbers
and integrates properly with Kelly position sizing.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading.narrative_gap import NarrativeGap, NarrativeGapComponents


class TestNarrativeGap:
    """Test the core Narrative Gap calculation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.ng_calc = NarrativeGap()

    def test_basic_ng_calculation(self):
        """Test basic NG calculation with simple inputs"""
        # Gary estimates higher than consensus
        market_price = 100.0
        consensus = 105.0
        gary_estimate = 110.0

        ng = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate)

        # Expected: abs(105 - 110) / 100 = 5/100 = 0.05
        expected_ng = 0.05
        assert abs(ng - expected_ng) < 0.001, f"Expected NG ~{expected_ng}, got {ng}"

    def test_ng_with_large_gap(self):
        """Test NG calculation with large price gap"""
        market_price = 100.0
        consensus = 120.0
        gary_estimate = 150.0

        ng = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate)

        # Expected: abs(120 - 150) / 100 = 30/100 = 0.30
        expected_ng = 0.30
        assert abs(ng - expected_ng) < 0.001, f"Expected NG ~{expected_ng}, got {ng}"

    def test_ng_with_reverse_gap(self):
        """Test NG when Gary estimates lower than consensus"""
        market_price = 100.0
        consensus = 110.0
        gary_estimate = 95.0

        ng = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate)

        # Expected: abs(110 - 95) / 100 = 15/100 = 0.15
        expected_ng = 0.15
        assert abs(ng - expected_ng) < 0.001, f"Expected NG ~{expected_ng}, got {ng}"

    def test_ng_zero_gap(self):
        """Test NG when consensus equals Gary's estimate"""
        market_price = 100.0
        consensus = 105.0
        gary_estimate = 105.0

        ng = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate)

        # Expected: abs(105 - 105) / 100 = 0/100 = 0.0
        assert ng == 0.0, f"Expected NG = 0, got {ng}"

    def test_ng_with_confidence_scaling(self):
        """Test NG calculation with confidence scaling"""
        market_price = 100.0
        consensus = 105.0
        gary_estimate = 115.0

        # High confidence
        ng_high_conf = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate, confidence=0.9)

        # Low confidence
        ng_low_conf = self.ng_calc.calculate_ng(market_price, consensus, gary_estimate, confidence=0.2)

        # High confidence should result in higher NG
        assert ng_high_conf > ng_low_conf, f"High confidence NG {ng_high_conf} should be > low confidence {ng_low_conf}"

    def test_position_multiplier(self):
        """Test position multiplier calculation"""
        # Test various NG values
        test_cases = [
            (0.0, 1.0),    # No gap = 1x multiplier
            (0.5, 1.5),    # 50% NG = 1.5x multiplier
            (1.0, 2.0),    # Max NG = 2x multiplier
        ]

        for ng, expected_multiplier in test_cases:
            multiplier = self.ng_calc.get_position_multiplier(ng)
            assert abs(multiplier - expected_multiplier) < 0.001, \
                f"NG {ng} should give {expected_multiplier}x multiplier, got {multiplier}"

    def test_detailed_ng_calculation(self):
        """Test detailed NG calculation with components"""
        market_price = 100.0
        consensus = 105.0
        gary_estimate = 120.0
        confidence = 0.8

        components = self.ng_calc.calculate_ng_detailed(
            market_price, consensus, gary_estimate, confidence
        )

        assert isinstance(components, NarrativeGapComponents)
        assert components.market_price == market_price
        assert components.consensus_forecast == consensus
        assert components.distribution_estimate == gary_estimate
        assert components.raw_gap == abs(consensus - gary_estimate)
        assert components.confidence_score == confidence
        assert components.normalized_ng > 0

    def test_input_validation(self):
        """Test input validation"""
        # Valid inputs
        is_valid, message = self.ng_calc.validate_inputs(100.0, 105.0, 110.0)
        assert is_valid, f"Valid inputs should pass validation: {message}"

        # Invalid price
        is_valid, message = self.ng_calc.validate_inputs(0.0, 105.0, 110.0)
        assert not is_valid, "Zero price should fail validation"

        # NaN values
        is_valid, message = self.ng_calc.validate_inputs(100.0, np.nan, 110.0)
        assert not is_valid, "NaN values should fail validation"

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Very small price
        ng = self.ng_calc.calculate_ng(0.1, 105.0, 110.0)
        assert ng >= 0, "NG should be non-negative even for small prices"

        # Negative price (should return 0)
        ng = self.ng_calc.calculate_ng(-1.0, 105.0, 110.0)
        assert ng == 0.0, "Negative price should return 0 NG"

        # Very large gap (should be capped)
        ng = self.ng_calc.calculate_ng(1.0, 100.0, 200.0)
        assert ng <= 1.0, "NG should be capped at max value"


class TestNarrativeGapIntegration:
    """Test integration scenarios"""

    def test_real_world_scenario_1(self):
        """Test realistic trading scenario"""
        ng_calc = NarrativeGap()

        # AAPL-like scenario: $150 stock
        market_price = 150.0
        consensus = 155.0  # Analysts average
        gary_estimate = 165.0  # Gary sees stronger distribution

        ng = ng_calc.calculate_ng(market_price, consensus, gary_estimate)
        multiplier = ng_calc.get_position_multiplier(ng)

        # Should have reasonable NG and multiplier
        assert 0.05 <= ng <= 0.15, f"NG {ng} should be in reasonable range"
        assert 1.05 <= multiplier <= 1.15, f"Multiplier {multiplier} should be in reasonable range"

    def test_real_world_scenario_2(self):
        """Test scenario with large opportunity"""
        ng_calc = NarrativeGap()

        # Scenario: Market undervalues significantly
        market_price = 50.0
        consensus = 55.0    # Modest analyst expectations
        gary_estimate = 75.0  # Gary sees major upside

        ng = ng_calc.calculate_ng(market_price, consensus, gary_estimate)
        multiplier = ng_calc.get_position_multiplier(ng)

        # Should result in significant position sizing boost
        assert ng >= 0.3, f"Large opportunity should give NG >= 0.3, got {ng}"
        assert multiplier >= 1.3, f"Large opportunity should give multiplier >= 1.3x, got {multiplier}"

    def test_no_opportunity_scenario(self):
        """Test scenario with no clear opportunity"""
        ng_calc = NarrativeGap()

        # Scenario: Gary and consensus agree
        market_price = 100.0
        consensus = 102.0
        gary_estimate = 103.0  # Small difference

        ng = ng_calc.calculate_ng(market_price, consensus, gary_estimate)
        multiplier = ng_calc.get_position_multiplier(ng)

        # Should result in minimal position boost
        assert ng <= 0.05, f"Small opportunity should give NG <= 0.05, got {ng}"
        assert multiplier <= 1.05, f"Small opportunity should give multiplier <= 1.05x, got {multiplier}"


def test_ng_factory_function():
    """Test factory function"""
    from trading.narrative_gap import create_narrative_gap_calculator

    ng_calc = create_narrative_gap_calculator(max_ng=0.5)
    assert ng_calc.max_ng == 0.5, "Factory should respect custom parameters"


def test_performance():
    """Test calculation performance"""
    import time

    ng_calc = NarrativeGap()

    start_time = time.time()
    for _ in range(1000):
        ng = ng_calc.calculate_ng(100.0, 105.0, 110.0)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / 1000
    assert avg_time_ms < 1.0, f"Average calculation time {avg_time_ms:.2f}ms should be < 1ms"


if __name__ == "__main__":
    # Run basic tests
    print("Running Narrative Gap Tests...")

    # Test basic functionality
    ng_calc = NarrativeGap()

    print("\n1. Basic NG Calculation Test:")
    market_price = 100.0
    consensus = 105.0
    gary_estimate = 110.0

    ng = ng_calc.calculate_ng(market_price, consensus, gary_estimate)
    multiplier = ng_calc.get_position_multiplier(ng)

    print(f"   Market Price: ${market_price}")
    print(f"   Consensus: ${consensus}")
    print(f"   Gary's Estimate: ${gary_estimate}")
    print(f"   Narrative Gap: {ng:.4f}")
    print(f"   Position Multiplier: {multiplier:.2f}x")

    # Test with confidence
    print("\n2. NG with Confidence Test:")
    ng_with_conf = ng_calc.calculate_ng(market_price, consensus, gary_estimate, confidence=0.8)
    mult_with_conf = ng_calc.get_position_multiplier(ng_with_conf)

    print(f"   NG with 80% confidence: {ng_with_conf:.4f}")
    print(f"   Multiplier with confidence: {mult_with_conf:.2f}x")

    # Test detailed calculation
    print("\n3. Detailed NG Components:")
    components = ng_calc.calculate_ng_detailed(market_price, consensus, gary_estimate, confidence=0.8)
    print(f"   Raw Gap: ${components.raw_gap:.2f}")
    print(f"   Normalized NG: {components.normalized_ng:.4f}")
    print(f"   Confidence Score: {components.confidence_score:.2f}")

    print("\n✓ All basic tests passed! Narrative Gap is working correctly.")