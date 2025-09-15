"""
Tests for Gary's Distributional Pressure Index Calculator

Validates the core DPI calculations and integration with WeeklyCycle
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.strategies.dpi_calculator import (
    DistributionalPressureIndex,
    DPIWeeklyCycleIntegrator,
    DistributionalRegime,
    DPIComponents,
    NarrativeGapAnalysis,
    PositionSizingOutput
)


class TestDistributionalPressureIndex:
    """Test Gary's DPI calculation engine"""

    @pytest.fixture
    def dpi_calculator(self):
        """Create DPI calculator for testing"""
        return DistributionalPressureIndex(lookback_periods=10, confidence_threshold=0.5)

    def test_dpi_calculation(self, dpi_calculator):
        """Test DPI calculation for a symbol"""
        # Mock market data
        with patch.object(dpi_calculator, '_fetch_market_data') as mock_fetch:
            # Create sample market data
            import pandas as pd
            from datetime import datetime, timedelta

            dates = pd.date_range(start=datetime.now() - timedelta(days=15),
                                end=datetime.now(), freq='D')

            # Realistic market data with trend
            returns = np.random.normal(0.002, 0.015, len(dates))  # Slight bullish bias
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(10, 0.3, len(dates))

            mock_data = pd.DataFrame({
                'Open': prices * 0.995,
                'High': prices * 1.008,
                'Low': prices * 0.992,
                'Close': prices,
                'Volume': volumes
            }, index=dates)

            mock_fetch.return_value = mock_data

            # Test DPI calculation
            dpi_score, components = dpi_calculator.calculate_dpi('ULTY')

            # Verify results
            assert isinstance(dpi_score, float)
            assert -1.0 <= dpi_score <= 1.0, f"DPI score {dpi_score} out of range [-1, 1]"
            assert isinstance(components, DPIComponents)
            assert hasattr(components, 'order_flow_pressure')
            assert hasattr(components, 'volume_weighted_skew')
            assert hasattr(components, 'price_momentum_bias')
            assert hasattr(components, 'volatility_clustering')

    def test_narrative_gap_calculation(self, dpi_calculator):
        """Test narrative gap analysis"""
        with patch.object(dpi_calculator, '_fetch_market_data') as mock_fetch:
            import pandas as pd
            from datetime import datetime, timedelta

            # Create market data with clear trend
            dates = pd.date_range(start=datetime.now() - timedelta(days=10),
                                end=datetime.now(), freq='D')

            # Strong uptrend
            returns = np.array([0.02] * len(dates))  # 2% daily gains
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(10, 0.2, len(dates))

            mock_data = pd.DataFrame({
                'Open': prices * 0.98,
                'High': prices * 1.01,
                'Low': prices * 0.99,
                'Close': prices,
                'Volume': volumes
            }, index=dates)

            mock_fetch.return_value = mock_data

            # Test narrative gap
            ng_analysis = dpi_calculator.detect_narrative_gap('ULTY')

            # Verify results
            assert isinstance(ng_analysis, NarrativeGapAnalysis)
            assert isinstance(ng_analysis.sentiment_score, float)
            assert isinstance(ng_analysis.price_action_score, float)
            assert isinstance(ng_analysis.narrative_gap, float)
            assert ng_analysis.gap_direction in ['bullish', 'bearish', 'neutral']
            assert 0 <= ng_analysis.confidence <= 1

    def test_position_sizing(self, dpi_calculator):
        """Test position sizing calculation"""
        # Test parameters
        symbol = 'ULTY'
        dpi_score = 0.6  # Strong bullish signal
        ng_score = -0.2  # Slight bearish sentiment (contrarian opportunity)
        available_cash = 1000.0

        sizing_output = dpi_calculator.determine_position_size(
            symbol, dpi_score, ng_score, available_cash
        )

        # Verify results
        assert isinstance(sizing_output, PositionSizingOutput)
        assert 0 <= sizing_output.recommended_size <= available_cash
        assert 0 <= sizing_output.risk_adjusted_size <= available_cash
        assert sizing_output.risk_adjusted_size <= sizing_output.max_position_size
        assert 0 <= sizing_output.confidence_factor <= 1
        assert sizing_output.dpi_contribution >= 0
        assert sizing_output.ng_contribution >= 0

    def test_distributional_regime_detection(self, dpi_calculator):
        """Test distributional regime classification"""
        with patch.object(dpi_calculator, 'calculate_dpi') as mock_dpi:
            from src.strategies.dpi_calculator import DPIComponents

            # High pressure bullish regime
            mock_dpi.return_value = (0.7, DPIComponents(
                order_flow_pressure=0.6,
                volume_weighted_skew=0.4,
                price_momentum_bias=0.5,
                volatility_clustering=0.3,
                raw_score=0.7,
                normalized_score=0.7
            ))

            regime = dpi_calculator.get_distributional_regime('ULTY')
            assert regime == DistributionalRegime.HIGH_PRESSURE_BULLISH

    def test_dpi_summary_multiple_symbols(self, dpi_calculator):
        """Test DPI summary for multiple symbols"""
        symbols = ['ULTY', 'AMDY', 'IAU']

        with patch.object(dpi_calculator, 'calculate_dpi') as mock_dpi, \
             patch.object(dpi_calculator, 'detect_narrative_gap') as mock_ng, \
             patch.object(dpi_calculator, 'get_distributional_regime') as mock_regime:

            # Mock responses
            mock_dpi.return_value = (0.3, Mock())
            mock_ng.return_value = Mock(narrative_gap=0.1, confidence=0.7)
            mock_regime.return_value = DistributionalRegime.LOW_PRESSURE_BALANCED

            summary = dpi_calculator.get_dpi_summary(symbols)

            # Verify summary structure
            assert 'timestamp' in summary
            assert 'symbols' in summary
            assert 'market_regime' in summary
            assert len(summary['symbols']) == len(symbols)

            for symbol in symbols:
                assert symbol in summary['symbols']
                assert 'dpi_score' in summary['symbols'][symbol]
                assert 'narrative_gap' in summary['symbols'][symbol]
                assert 'regime' in summary['symbols'][symbol]
                assert 'confidence' in summary['symbols'][symbol]


class TestDPIWeeklyCycleIntegrator:
    """Test DPI integration with WeeklyCycle"""

    @pytest.fixture
    def dpi_calculator(self):
        return DistributionalPressureIndex()

    @pytest.fixture
    def dpi_integrator(self, dpi_calculator):
        return DPIWeeklyCycleIntegrator(dpi_calculator)

    def test_dpi_enhanced_allocations(self, dpi_integrator):
        """Test DPI-enhanced allocation calculation"""
        symbols = ['ULTY', 'AMDY']
        available_cash = 1000.0
        base_allocations = {'ULTY': 70.0, 'AMDY': 30.0}

        with patch.object(dpi_integrator.dpi_calculator, 'calculate_dpi') as mock_dpi, \
             patch.object(dpi_integrator.dpi_calculator, 'detect_narrative_gap') as mock_ng:

            # Mock strong DPI signal for ULTY, weak for AMDY
            mock_dpi.side_effect = [(0.5, Mock()), (-0.2, Mock())]
            mock_ng.side_effect = [
                Mock(narrative_gap=0.1, confidence=0.8),
                Mock(narrative_gap=-0.3, confidence=0.6)
            ]

            enhanced_allocations = dpi_integrator.get_dpi_enhanced_allocations(
                symbols, available_cash, base_allocations
            )

            # Verify enhanced allocations
            assert isinstance(enhanced_allocations, dict)
            assert set(enhanced_allocations.keys()) == set(symbols)

            # Allocations should sum to approximately 100%
            total_allocation = sum(enhanced_allocations.values())
            assert abs(total_allocation - 100.0) < 1.0

            # ULTY should get more allocation due to strong DPI signal
            assert enhanced_allocations['ULTY'] > base_allocations['ULTY']


class TestDPIEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_market_data(self):
        """Test behavior with empty market data"""
        dpi_calculator = DistributionalPressureIndex()

        with patch.object(dpi_calculator, '_fetch_market_data') as mock_fetch:
            import pandas as pd
            mock_fetch.return_value = pd.DataFrame()  # Empty data

            # Should handle gracefully
            with pytest.raises(ValueError, match="No market data"):
                dpi_calculator.calculate_dpi('INVALID')

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        dpi_calculator = DistributionalPressureIndex()

        with patch.object(dpi_calculator, '_fetch_market_data') as mock_fetch:
            import pandas as pd
            from datetime import datetime

            # Only 1 data point
            mock_data = pd.DataFrame({
                'Open': [100],
                'High': [101],
                'Low': [99],
                'Close': [100.5],
                'Volume': [1000]
            }, index=[datetime.now()])

            mock_fetch.return_value = mock_data

            # Should handle gracefully
            dpi_score, components = dpi_calculator.calculate_dpi('TEST')
            assert isinstance(dpi_score, float)
            assert -1.0 <= dpi_score <= 1.0

    def test_extreme_position_sizing(self):
        """Test position sizing with extreme values"""
        dpi_calculator = DistributionalPressureIndex()

        # Test with very high DPI and NG
        sizing = dpi_calculator.determine_position_size(
            'TEST', dpi=0.95, ng=0.8, available_cash=10.0  # Low cash
        )

        # Should cap at reasonable size
        assert sizing.risk_adjusted_size == 0.0  # Too small, should be 0

    def test_confidence_threshold_enforcement(self):
        """Test confidence threshold enforcement"""
        dpi_calculator = DistributionalPressureIndex(confidence_threshold=0.8)

        # Low confidence scenario
        sizing = dpi_calculator.determine_position_size(
            'TEST', dpi=0.3, ng=0.5, available_cash=1000.0  # Weak signals
        )

        # Should have reduced confidence
        assert sizing.confidence_factor < 0.8


if __name__ == '__main__':
    # Run specific tests for debugging
    test_dpi = TestDistributionalPressureIndex()
    test_dpi.test_dpi_calculation(test_dpi.dpi_calculator())
    print("DPI Calculator tests passed!")