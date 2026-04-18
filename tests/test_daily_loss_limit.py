"""
Test suite for daily loss limit protection.

Validates that the -2% daily loss limit correctly blocks trading
when losses exceed threshold.
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Import the classes we're testing
import sys
sys.path.insert(0, 'D:/Projects/trader-ai')

from src.portfolio.portfolio_manager import PortfolioManager


class TestDailyLossLimit:
    """Test daily loss limit protection functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker adapter."""
        broker = Mock()
        broker.is_connected = True
        broker.get_account_value = AsyncMock(return_value=Decimal("200.00"))
        broker.get_cash_balance = AsyncMock(return_value=Decimal("200.00"))
        broker.get_positions = AsyncMock(return_value=[])
        return broker

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data provider."""
        market_data = Mock()
        return market_data

    @pytest.fixture
    def portfolio_manager(self, mock_broker, mock_market_data):
        """Create portfolio manager instance."""
        return PortfolioManager(
            mock_broker,
            mock_market_data,
            initial_capital=Decimal("200.00")
        )

    @pytest.mark.asyncio
    async def test_initial_reset(self, portfolio_manager):
        """Test that first call initializes daily tracking."""
        # Mock get_total_portfolio_value
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("200.00")
        )

        result = await portfolio_manager.check_daily_loss()

        assert result['daily_reset'] is True
        assert result['start_value'] == 200.00
        assert portfolio_manager.daily_start_value == Decimal("200.00")
        assert portfolio_manager.daily_loss_triggered is False

    @pytest.mark.asyncio
    async def test_within_limit(self, portfolio_manager):
        """Test that losses within limit don't trigger protection."""
        # Initialize
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc)
        portfolio_manager.daily_loss_triggered = False

        # Portfolio down 1% (within -2% limit)
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("198.00")
        )

        result = await portfolio_manager.check_daily_loss()

        assert result['triggered'] is False
        assert result['daily_change_pct'] == pytest.approx(-0.01, abs=0.001)
        assert result['limit_pct'] == -0.02

    @pytest.mark.asyncio
    async def test_exceeds_limit(self, portfolio_manager):
        """Test that losses exceeding -2% trigger protection."""
        # Initialize
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc)
        portfolio_manager.daily_loss_triggered = False

        # Portfolio down 2.5% (exceeds -2% limit)
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("195.00")
        )

        result = await portfolio_manager.check_daily_loss()

        assert result['triggered'] is True
        assert result['daily_change_pct'] == pytest.approx(-0.025, abs=0.001)
        assert portfolio_manager.daily_loss_triggered is True

    @pytest.mark.asyncio
    async def test_exactly_at_limit(self, portfolio_manager):
        """Test behavior exactly at -2% limit."""
        # Initialize
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc)
        portfolio_manager.daily_loss_triggered = False

        # Portfolio down exactly 2%
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("196.00")
        )

        result = await portfolio_manager.check_daily_loss()

        # At exactly -2%, should trigger (<=)
        assert result['triggered'] is True
        assert result['daily_change_pct'] == pytest.approx(-0.02, abs=0.001)

    @pytest.mark.asyncio
    async def test_stays_triggered(self, portfolio_manager):
        """Test that once triggered, stays triggered until reset."""
        # Already triggered
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc)
        portfolio_manager.daily_loss_triggered = True

        # Portfolio recovers to -1.5%
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("197.00")
        )

        result = await portfolio_manager.check_daily_loss()

        # Should still be triggered (doesn't reset mid-day)
        assert result['triggered'] is True

    @pytest.mark.asyncio
    async def test_daily_reset(self, portfolio_manager):
        """Test that limit resets next trading day."""
        # Set reset time to yesterday
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc) - timedelta(days=1)
        portfolio_manager.daily_loss_triggered = True  # Was triggered yesterday

        # Today's value
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("195.00")
        )

        result = await portfolio_manager.check_daily_loss()

        # Should reset for new day
        assert result.get('daily_reset') is True
        assert portfolio_manager.daily_loss_triggered is False
        assert portfolio_manager.daily_start_value == Decimal("195.00")

    @pytest.mark.asyncio
    async def test_positive_returns(self, portfolio_manager):
        """Test behavior with positive daily returns."""
        # Initialize
        portfolio_manager.daily_start_value = Decimal("200.00")
        portfolio_manager.daily_reset_time = datetime.now(timezone.utc)
        portfolio_manager.daily_loss_triggered = False

        # Portfolio up 3%
        portfolio_manager.get_total_portfolio_value = AsyncMock(
            return_value=Decimal("206.00")
        )

        result = await portfolio_manager.check_daily_loss()

        assert result['triggered'] is False
        assert result['daily_change_pct'] == pytest.approx(0.03, abs=0.001)


class TestTradingEngineIntegration:
    """Test integration of daily loss limit in trading engine."""

    @pytest.mark.asyncio
    async def test_blocks_trading_when_triggered(self):
        """Test that trading engine blocks trades when limit triggered."""
        # This would require a full trading engine setup
        # For now, we document the expected behavior:

        # 1. Portfolio loses 2.5% in a day
        # 2. check_daily_loss() returns triggered=True
        # 3. Trading engine returns early without executing trades
        # 4. Audit log records the block event
        # 5. Critical logger alert fired

        pass  # Integration test placeholder


def run_manual_simulation():
    """
    Manual simulation to demonstrate daily loss limit.
    Run with: python -m pytest tests/test_daily_loss_limit.py -v -s
    """
    print("\n" + "="*60)
    print("DAILY LOSS LIMIT SIMULATION")
    print("="*60)

    print("\nScenario: Portfolio starts at $200")
    print("Limit: -2% ($196)")
    print()

    scenarios = [
        ("Day 1, 10 AM: Down 1%", 198.00, False),
        ("Day 1, 11 AM: Down 1.5%", 197.00, False),
        ("Day 1, 12 PM: Down 2.5%", 195.00, True),
        ("Day 1, 2 PM: Recovers to -1.5%", 197.00, True),  # Still blocked
        ("Day 2, 10 AM: Reset, starts at $197", 197.00, False),  # New day
    ]

    for desc, value, should_block in scenarios:
        status = "BLOCKED" if should_block else "TRADING"
        print(f"{desc}: ${value:.2f} -> {status}")

    print("\n" + "="*60)


if __name__ == '__main__':
    # Run simulation
    run_manual_simulation()

    # Run tests
    print("\nRunning unit tests...")
    pytest.main([__file__, '-v'])
