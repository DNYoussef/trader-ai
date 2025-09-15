"""
Comprehensive Test Suite for Kelly Criterion Position Sizing System

Tests cover:
1. Kelly calculation accuracy with DPI integration
2. Overleverage prevention (Kelly > 1.0)
3. Gate system compliance validation
4. Real-time performance requirements (<50ms)
5. Risk constraint integration
6. Dynamic position sizing optimization

Test Categories:
- Unit tests: Individual component testing
- Integration tests: System component interaction
- Performance tests: Latency and throughput validation
- Compliance tests: Gate system integration
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from unittest.mock import Mock, patch, MagicMock

# Import system under test
from src.risk.kelly_criterion import (
    KellyCriterionCalculator,
    PositionSizeRecommendation,
    KellyComponents,
    KellyRiskMetrics,
    KellyRegime
)
from src.risk.dynamic_position_sizing import (
    DynamicPositionSizer,
    PortfolioAllocation,
    DynamicSizingConfig
)
from src.strategies.dpi_calculator import (
    DistributionalPressureIndex,
    DPIComponents
)
from src.gates.gate_manager import GateManager, GateLevel


class TestKellyCriterionCalculator:
    """Unit tests for Kelly Criterion Calculator"""

    @pytest.fixture
    def mock_dpi_calculator(self):
        """Mock DPI calculator for testing"""
        dpi_calc = Mock(spec=DistributionalPressureIndex)
        dpi_calc.calculate_dpi.return_value = (
            0.3,  # DPI score
            DPIComponents(
                order_flow_pressure=0.2,
                volume_weighted_skew=0.1,
                price_momentum_bias=0.15,
                volatility_clustering=0.05,
                raw_score=0.25,
                normalized_score=0.3
            )
        )
        dpi_calc._fetch_market_data.return_value = self._create_mock_market_data()
        return dpi_calc

    @pytest.fixture
    def mock_gate_manager(self):
        """Mock gate manager for testing"""
        gate_mgr = Mock(spec=GateManager)
        gate_mgr.current_gate = GateLevel.G1
        gate_mgr.validate_trade.return_value = Mock(is_valid=True)
        return gate_mgr

    @pytest.fixture
    def kelly_calculator(self, mock_dpi_calculator, mock_gate_manager):
        """Kelly calculator instance for testing"""
        return KellyCriterionCalculator(
            dpi_calculator=mock_dpi_calculator,
            gate_manager=mock_gate_manager,
            max_kelly=0.25,
            min_edge=0.01
        )

    def _create_mock_market_data(self):
        """Create realistic mock market data"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)  # Reproducible results

        # Generate price series with realistic characteristics
        returns = np.random.normal(0.0008, 0.015, len(dates))  # 20% annual vol
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 0.3, len(dates))

        return pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Volume': volumes
        }, index=dates)

    def test_kelly_calculation_basic(self, kelly_calculator):
        """Test basic Kelly calculation functionality"""
        result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )

        assert isinstance(result, PositionSizeRecommendation)
        assert result.symbol == "ULTY"
        assert 0 <= result.kelly_percentage <= 1.0  # Valid percentage range
        assert result.dollar_amount >= 0
        assert result.share_quantity >= 0
        assert 0 <= result.confidence_score <= 1.0

    def test_overleverage_prevention(self, kelly_calculator):
        """Test that Kelly never exceeds 100% (overleverage protection)"""
        # Mock extreme DPI that might cause overleverage
        kelly_calculator.dpi_calculator.calculate_dpi.return_value = (
            0.95,  # Very high DPI score
            DPIComponents(0.8, 0.7, 0.9, 0.6, 0.9, 0.95)
        )

        result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY",
            current_price=10.0,
            available_capital=10000.0
        )

        # Should be capped at configured maximum (25% in this case)
        assert result.kelly_percentage <= kelly_calculator.max_kelly
        assert result.kelly_percentage <= 1.0  # Never exceed 100%

    def test_performance_latency_requirement(self, kelly_calculator):
        """Test <50ms latency requirement"""
        start_time = time.time()

        result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )

        execution_time_ms = (time.time() - start_time) * 1000

        # Performance requirement: <50ms
        assert execution_time_ms < 50.0, f"Execution took {execution_time_ms:.1f}ms, exceeds 50ms limit"
        assert result.execution_time_ms < 50.0

    def test_dpi_integration(self, kelly_calculator):
        """Test DPI integration affects Kelly calculation"""
        # Test with high DPI score
        kelly_calculator.dpi_calculator.calculate_dpi.return_value = (
            0.7, DPIComponents(0.6, 0.5, 0.8, 0.4, 0.6, 0.7)
        )

        high_dpi_result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY", current_price=50.0, available_capital=1000.0
        )

        # Test with low DPI score
        kelly_calculator.dpi_calculator.calculate_dpi.return_value = (
            0.1, DPIComponents(0.1, 0.0, 0.1, 0.1, 0.1, 0.1)
        )

        low_dpi_result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY", current_price=50.0, available_capital=1000.0
        )

        # High DPI should result in larger position (assuming positive edge)
        if high_dpi_result.kelly_percentage > 0 and low_dpi_result.kelly_percentage > 0:
            assert high_dpi_result.kelly_percentage >= low_dpi_result.kelly_percentage

    def test_minimum_edge_requirement(self, kelly_calculator):
        """Test minimum edge requirement prevents positions"""
        # Mock very low/no edge scenario
        kelly_calculator.dpi_calculator.calculate_dpi.return_value = (
            0.05,  # Very low DPI
            DPIComponents(0.01, 0.01, 0.01, 0.01, 0.02, 0.05)
        )

        result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )

        # Should result in no position due to insufficient edge
        assert result.kelly_percentage == 0.0
        assert result.dollar_amount == 0.0
        assert result.share_quantity == 0

    def test_risk_metrics_calculation(self, kelly_calculator):
        """Test risk metrics are properly calculated"""
        result = kelly_calculator.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )

        risk_metrics = result.risk_metrics
        assert isinstance(risk_metrics, KellyRiskMetrics)
        assert 0 <= risk_metrics.max_drawdown_risk <= 1.0
        assert 0 <= risk_metrics.volatility_adjustment <= 1.0
        assert len(risk_metrics.confidence_interval) == 2
        assert risk_metrics.confidence_interval[0] <= risk_metrics.confidence_interval[1]

    def test_kelly_regime_classification(self, kelly_calculator):
        """Test Kelly regime classification"""
        regimes = [
            (0.0, KellyRegime.NO_BET),
            (0.03, KellyRegime.MINIMAL),
            (0.08, KellyRegime.CONSERVATIVE),
            (0.15, KellyRegime.MODERATE),
            (0.30, KellyRegime.AGGRESSIVE)
        ]

        for kelly_pct, expected_regime in regimes:
            regime = kelly_calculator.get_regime_classification(kelly_pct)
            assert regime == expected_regime


class TestDynamicPositionSizer:
    """Unit tests for Dynamic Position Sizing System"""

    @pytest.fixture
    def mock_kelly_calculator(self):
        """Mock Kelly calculator for testing"""
        kelly_calc = Mock(spec=KellyCriterionCalculator)
        kelly_calc.calculate_kelly_position.return_value = PositionSizeRecommendation(
            symbol="ULTY",
            kelly_percentage=0.10,
            dollar_amount=100.0,
            share_quantity=2,
            confidence_score=0.8,
            risk_metrics=Mock(),
            gate_compliant=True,
            execution_time_ms=25.0
        )
        return kelly_calc

    @pytest.fixture
    def mock_dpi_calculator(self):
        """Mock DPI calculator"""
        return Mock(spec=DistributionalPressureIndex)

    @pytest.fixture
    def mock_gate_manager(self):
        """Mock gate manager"""
        gate_mgr = Mock(spec=GateManager)
        gate_mgr.current_gate = GateLevel.G1
        gate_mgr.get_current_config.return_value = Mock(
            level=GateLevel.G1,
            allowed_assets={'ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP'},
            cash_floor_pct=0.60,
            max_position_pct=0.22
        )
        return gate_mgr

    @pytest.fixture
    def position_sizer(self, mock_kelly_calculator, mock_dpi_calculator, mock_gate_manager):
        """Dynamic position sizer instance"""
        config = DynamicSizingConfig(
            max_positions=3,
            max_position_risk=0.02,
            total_risk_budget=0.06
        )
        return DynamicPositionSizer(
            kelly_calculator=mock_kelly_calculator,
            dpi_calculator=mock_dpi_calculator,
            gate_manager=mock_gate_manager,
            config=config
        )

    @pytest.mark.asyncio
    async def test_portfolio_allocation_calculation(self, position_sizer):
        """Test portfolio allocation calculation"""
        symbols = ['ULTY', 'AMDY', 'IAU']
        prices = {'ULTY': 50.0, 'AMDY': 25.0, 'IAU': 40.0}
        total_capital = 10000.0

        allocation = await position_sizer.calculate_portfolio_allocation(
            symbols=symbols,
            current_prices=prices,
            total_capital=total_capital
        )

        assert isinstance(allocation, PortfolioAllocation)
        assert allocation.total_capital == total_capital
        assert 0 <= allocation.total_allocated_pct <= 1.0
        assert allocation.cash_reserve_pct >= 0
        assert allocation.compliance_status in ["COMPLIANT", "NON_COMPLIANT", "SAFE_DEFAULT"]

    @pytest.mark.asyncio
    async def test_gate_filtering(self, position_sizer):
        """Test that positions are filtered by gate allowances"""
        # Include symbols not allowed in G1 gate
        symbols = ['ULTY', 'SPY', 'QQQ']  # SPY/QQQ not allowed in G1
        prices = {'ULTY': 50.0, 'SPY': 400.0, 'QQQ': 300.0}
        total_capital = 10000.0

        allocation = await position_sizer.calculate_portfolio_allocation(
            symbols=symbols,
            current_prices=prices,
            total_capital=total_capital
        )

        # Should only include ULTY (allowed in G1)
        if allocation.allocations:
            allowed_symbols = set(allocation.allocations.keys())
            gate_allowed = position_sizer.gate_manager.get_current_config().allowed_assets
            assert allowed_symbols.issubset(gate_allowed)

    @pytest.mark.asyncio
    async def test_risk_budget_enforcement(self, position_sizer):
        """Test risk budget is properly enforced"""
        symbols = ['ULTY', 'AMDY', 'IAU']
        prices = {'ULTY': 50.0, 'AMDY': 25.0, 'IAU': 40.0}
        total_capital = 10000.0

        allocation = await position_sizer.calculate_portfolio_allocation(
            symbols=symbols,
            current_prices=prices,
            total_capital=total_capital
        )

        # Total risk should not exceed configured budget
        assert allocation.risk_budget_used <= position_sizer.config.total_risk_budget

    def test_rebalancing_detection(self, position_sizer):
        """Test rebalancing trigger detection"""
        # Create mock allocations with significant differences
        current_allocation = PortfolioAllocation(
            total_capital=10000.0,
            allocations={
                'ULTY': Mock(kelly_percentage=0.15),
                'AMDY': Mock(kelly_percentage=0.05)
            }
        )

        target_allocation = PortfolioAllocation(
            total_capital=10000.0,
            allocations={
                'ULTY': Mock(kelly_percentage=0.05),  # Significant change
                'AMDY': Mock(kelly_percentage=0.15)   # Significant change
            }
        )

        should_rebalance = position_sizer.should_rebalance(
            current_allocation, target_allocation
        )

        assert should_rebalance  # Significant changes should trigger rebalance


class TestGateIntegration:
    """Integration tests for Kelly system with gate constraints"""

    @pytest.fixture
    def integrated_system(self):
        """Complete integrated system for testing"""
        dpi_calc = DistributionalPressureIndex()
        gate_mgr = GateManager(data_dir="./test_data/gates")
        kelly_calc = KellyCriterionCalculator(
            dpi_calculator=dpi_calc,
            gate_manager=gate_mgr
        )
        position_sizer = DynamicPositionSizer(
            kelly_calculator=kelly_calc,
            dpi_calculator=dpi_calc,
            gate_manager=gate_mgr
        )
        return {
            'kelly': kelly_calc,
            'sizer': position_sizer,
            'gates': gate_mgr,
            'dpi': dpi_calc
        }

    def test_gate_compliance_validation(self, integrated_system):
        """Test gate compliance is properly validated"""
        system = integrated_system

        # Set specific gate level
        system['gates'].current_gate = GateLevel.G0  # Restrictive gate

        result = system['kelly'].calculate_kelly_position(
            symbol="ULTY",  # Allowed in G0
            current_price=50.0,
            available_capital=300.0  # Within G0 range
        )

        assert result.gate_compliant

        # Test with non-allowed symbol
        result_invalid = system['kelly'].calculate_kelly_position(
            symbol="SPY",  # Not allowed in G0
            current_price=400.0,
            available_capital=300.0
        )

        assert not result_invalid.gate_compliant

    def test_position_size_limits(self, integrated_system):
        """Test position size limits are enforced"""
        system = integrated_system
        gate_config = system['gates'].get_current_config()

        result = system['kelly'].calculate_kelly_position(
            symbol="ULTY",
            current_price=10.0,
            available_capital=1000.0
        )

        # Kelly percentage should not exceed gate limit
        assert result.kelly_percentage <= gate_config.max_position_pct

    def test_cash_floor_compliance(self, integrated_system):
        """Test cash floor requirements are maintained"""
        system = integrated_system
        gate_config = system['gates'].get_current_config()

        # Test with limited capital that would violate cash floor
        total_capital = 500.0
        available_after_floor = total_capital * (1 - gate_config.cash_floor_pct)

        result = system['kelly'].calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=available_after_floor
        )

        # Position should fit within available capital after cash floor
        assert result.dollar_amount <= available_after_floor


class TestPerformanceRequirements:
    """Performance and latency tests"""

    def test_batch_calculation_performance(self):
        """Test performance with multiple simultaneous calculations"""
        # Mock components for performance testing
        dpi_calc = Mock(spec=DistributionalPressureIndex)
        dpi_calc.calculate_dpi.return_value = (0.2, Mock())
        dpi_calc._fetch_market_data.return_value = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 900, 1200, 1050]
        })

        gate_mgr = Mock(spec=GateManager)
        gate_mgr.validate_trade.return_value = Mock(is_valid=True)

        kelly_calc = KellyCriterionCalculator(
            dpi_calculator=dpi_calc,
            gate_manager=gate_mgr
        )

        symbols = ['ULTY', 'AMDY', 'IAU', 'GLDM', 'VTIP']

        start_time = time.time()

        results = []
        for symbol in symbols:
            result = kelly_calc.calculate_kelly_position(
                symbol=symbol,
                current_price=50.0,
                available_capital=1000.0
            )
            results.append(result)

        total_time = (time.time() - start_time) * 1000

        # All calculations should complete within reasonable time
        assert total_time < 200.0  # 200ms for 5 calculations
        assert len(results) == len(symbols)

        # Each individual calculation should meet latency requirement
        for result in results:
            assert result.execution_time_ms < 50.0

    def test_cache_effectiveness(self):
        """Test caching improves performance on repeated calls"""
        dpi_calc = Mock(spec=DistributionalPressureIndex)
        dpi_calc.calculate_dpi.return_value = (0.2, Mock())
        dpi_calc._fetch_market_data.return_value = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 900, 1200, 1050]
        })

        gate_mgr = Mock(spec=GateManager)
        gate_mgr.validate_trade.return_value = Mock(is_valid=True)

        kelly_calc = KellyCriterionCalculator(
            dpi_calculator=dpi_calc,
            gate_manager=gate_mgr
        )

        # First calculation (cache miss)
        start_time = time.time()
        result1 = kelly_calc.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )
        first_time = (time.time() - start_time) * 1000

        # Second identical calculation (cache hit)
        start_time = time.time()
        result2 = kelly_calc.calculate_kelly_position(
            symbol="ULTY",
            current_price=50.0,
            available_capital=1000.0
        )
        second_time = (time.time() - start_time) * 1000

        # Cached call should be significantly faster
        assert second_time < first_time
        assert second_time < 10.0  # Should be very fast with cache


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])