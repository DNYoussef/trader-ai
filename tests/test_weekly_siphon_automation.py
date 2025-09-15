"""
Comprehensive tests for Weekly Siphon Automation System

Tests the core functionality:
- ProfitCalculator with 50/50 split logic
- Capital protection safeguards
- WeeklySiphonAutomator scheduling and execution
- Integration with existing WeeklyCycle
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
import pytz

from src.cycles.profit_calculator import ProfitCalculator, ProfitStatus
from src.cycles.weekly_siphon_automator import WeeklySiphonAutomator, SiphonStatus
from src.cycles.weekly_cycle import WeeklyCycle


class TestProfitCalculator:
    """Test the ProfitCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a ProfitCalculator with $200 base capital."""
        return ProfitCalculator(initial_capital=Decimal("200.00"))

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Mock portfolio manager."""
        portfolio = Mock()
        portfolio.get_total_portfolio_value.return_value = Decimal("250.00")
        portfolio.get_nav_at_date.return_value = Decimal("200.00")
        portfolio.get_deposits_in_period.return_value = Decimal("0.00")
        portfolio.get_withdrawals_in_period.return_value = Decimal("0.00")
        return portfolio

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.base_capital == Decimal("200.00")
        assert calculator.total_withdrawals == Decimal("0.00")
        assert calculator.total_deposits_beyond_base == Decimal("0.00")
        assert len(calculator.profit_history) == 0

    def test_profit_calculation_with_50_dollar_gain(self, calculator, mock_portfolio_manager):
        """Test profit calculation when portfolio gained $50."""
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("250.00")

        calculation = calculator.calculate_weekly_profit(mock_portfolio_manager)

        assert calculation.profit_status == ProfitStatus.PROFIT_AVAILABLE
        assert calculation.total_profit == Decimal("50.00")
        assert calculation.reinvestment_amount == Decimal("25.00")
        assert calculation.withdrawal_amount == Decimal("25.00")
        assert calculation.remaining_capital == Decimal("225.00")
        assert calculation.base_capital == Decimal("200.00")

    def test_profit_calculation_no_profit(self, calculator, mock_portfolio_manager):
        """Test calculation when there's no profit."""
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("200.00")

        calculation = calculator.calculate_weekly_profit(mock_portfolio_manager)

        assert calculation.profit_status == ProfitStatus.NO_PROFIT
        assert calculation.total_profit == Decimal("0.00")
        assert calculation.reinvestment_amount == Decimal("0.00")
        assert calculation.withdrawal_amount == Decimal("0.00")
        assert calculation.remaining_capital == Decimal("200.00")

    def test_profit_calculation_with_loss(self, calculator, mock_portfolio_manager):
        """Test calculation when portfolio has a loss."""
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("180.00")

        calculation = calculator.calculate_weekly_profit(mock_portfolio_manager)

        assert calculation.profit_status == ProfitStatus.LOSS
        assert calculation.total_profit == Decimal("-20.00")
        assert calculation.reinvestment_amount == Decimal("0.00")
        assert calculation.withdrawal_amount == Decimal("0.00")
        assert calculation.remaining_capital == Decimal("180.00")

    def test_capital_protection_safeguard(self, calculator, mock_portfolio_manager):
        """Test that capital protection prevents base capital withdrawal."""
        # Scenario: Portfolio at $210, should only allow $10 withdrawal max
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("210.00")

        calculation = calculator.calculate_weekly_profit(mock_portfolio_manager)

        # Should have $10 profit, but split would give $5 each
        assert calculation.total_profit == Decimal("10.00")
        assert calculation.withdrawal_amount <= Decimal("10.00")  # Never more than total profit
        assert calculation.remaining_capital >= calculator.base_capital  # Never below base

    def test_withdrawal_safety_validation(self, calculator):
        """Test withdrawal safety validation."""
        # Safe withdrawal
        is_safe, reason = calculator.validate_withdrawal_safety(
            Decimal("10.00"), Decimal("220.00")
        )
        assert is_safe is True
        assert "safe" in reason.lower()

        # Unsafe withdrawal (would breach base capital)
        is_safe, reason = calculator.validate_withdrawal_safety(
            Decimal("50.00"), Decimal("210.00")
        )
        assert is_safe is False
        assert "breach" in reason.lower()

    def test_additional_deposits_tracking(self, calculator, mock_portfolio_manager):
        """Test tracking of additional deposits beyond base capital."""
        # Record additional deposit of $100
        calculator.record_additional_deposit(Decimal("100.00"))

        # Portfolio now worth $350 (base $200 + deposit $100 + profit $50)
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("350.00")

        calculation = calculator.calculate_weekly_profit(mock_portfolio_manager)

        # Should only consider $50 as profit (not the additional deposit)
        assert calculation.total_profit == Decimal("50.00")
        assert calculation.withdrawal_amount == Decimal("25.00")

    def test_profit_history_tracking(self, calculator, mock_portfolio_manager):
        """Test that profit calculations are stored in history."""
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("250.00")

        calculation1 = calculator.calculate_weekly_profit(mock_portfolio_manager)
        calculation2 = calculator.calculate_weekly_profit(mock_portfolio_manager)

        assert len(calculator.profit_history) == 2
        assert calculator.profit_history[0].total_profit == Decimal("50.00")
        assert calculator.profit_history[1].total_profit == Decimal("50.00")

    def test_get_capital_protection_status(self, calculator):
        """Test capital protection status reporting."""
        current_value = Decimal("250.00")
        status = calculator.get_capital_protection_status(current_value)

        assert status['base_capital'] == Decimal("200.00")
        assert status['current_value'] == Decimal("250.00")
        assert status['buffer_amount'] == Decimal("50.00")
        assert status['buffer_percent'] == Decimal("25.00")  # 50/200 * 100
        assert status['is_protected'] is True

    def test_profit_summary(self, calculator, mock_portfolio_manager):
        """Test profit summary generation."""
        mock_portfolio_manager.get_total_portfolio_value.return_value = Decimal("250.00")

        # Generate some profit history
        for _ in range(3):
            calculator.calculate_weekly_profit(mock_portfolio_manager)

        summary = calculator.get_profit_summary(weeks=3)

        assert summary['weeks_analyzed'] == 3
        assert summary['total_profit'] == Decimal("150.00")  # 3 * $50
        assert summary['successful_siphons'] == 0  # No withdrawals recorded yet


class TestWeeklySiphonAutomator:
    """Test the WeeklySiphonAutomator class."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Mock portfolio manager."""
        portfolio = AsyncMock()
        portfolio.sync_with_broker.return_value = True
        portfolio.get_total_portfolio_value.return_value = Decimal("250.00")
        portfolio.record_transaction = AsyncMock()
        return portfolio

    @pytest.fixture
    def mock_broker_adapter(self):
        """Mock broker adapter."""
        broker = AsyncMock()
        broker.withdraw_funds.return_value = True
        broker.get_last_withdrawal_id.return_value = "WD123456"
        return broker

    @pytest.fixture
    def profit_calculator(self):
        """Create profit calculator."""
        return ProfitCalculator(Decimal("200.00"))

    @pytest.fixture
    def mock_holiday_calendar(self):
        """Mock holiday calendar."""
        calendar = Mock()
        calendar.is_market_holiday.return_value = False
        return calendar

    @pytest.fixture
    def siphon_automator(self, mock_portfolio_manager, mock_broker_adapter,
                        profit_calculator, mock_holiday_calendar):
        """Create siphon automator with mocked dependencies."""
        return WeeklySiphonAutomator(
            portfolio_manager=mock_portfolio_manager,
            broker_adapter=mock_broker_adapter,
            profit_calculator=profit_calculator,
            holiday_calendar=mock_holiday_calendar,
            enable_auto_execution=True
        )

    def test_initialization(self, siphon_automator):
        """Test siphon automator initialization."""
        assert siphon_automator.enable_auto_execution is True
        assert siphon_automator.min_withdrawal_amount == Decimal("10.00")
        assert siphon_automator.is_running is False
        assert len(siphon_automator.execution_history) == 0

    def test_should_execute_siphon_conditions(self, siphon_automator):
        """Test siphon execution condition checking."""
        # Mock current time to Friday 6:30pm ET
        et_tz = pytz.timezone('US/Eastern')
        mock_friday = datetime(2024, 3, 8, 18, 30).replace(tzinfo=et_tz)  # Friday 6:30pm

        with pytest.mock.patch.object(siphon_automator, 'get_current_et_time', return_value=mock_friday):
            should_execute, reason = siphon_automator.should_execute_siphon()
            # Would return True if no previous execution this week
            assert isinstance(should_execute, bool)
            assert isinstance(reason, str)

    @pytest.mark.asyncio
    async def test_manual_siphon_execution_with_profit(self, siphon_automator):
        """Test manual siphon execution when there's profit."""
        # Setup profit calculation to return $50 profit
        mock_calc = Mock()
        mock_calc.profit_status = ProfitStatus.PROFIT_AVAILABLE
        mock_calc.withdrawal_amount = Decimal("25.00")
        mock_calc.current_value = Decimal("250.00")

        siphon_automator.profit_calculator.calculate_weekly_profit = Mock(return_value=mock_calc)
        siphon_automator.profit_calculator.validate_withdrawal_safety = Mock(return_value=(True, "Safe withdrawal"))

        # Force execution to bypass timing checks
        result = await siphon_automator.execute_manual_siphon(force=True)

        assert result.status == SiphonStatus.SUCCESS
        assert result.withdrawal_amount == Decimal("25.00")
        assert result.withdrawal_success is True
        assert "Safe withdrawal" in result.safety_checks

    @pytest.mark.asyncio
    async def test_manual_siphon_execution_no_profit(self, siphon_automator):
        """Test manual siphon execution when there's no profit."""
        # Setup profit calculation to return no profit
        mock_calc = Mock()
        mock_calc.profit_status = ProfitStatus.NO_PROFIT
        mock_calc.withdrawal_amount = Decimal("0.00")

        siphon_automator.profit_calculator.calculate_weekly_profit = Mock(return_value=mock_calc)

        result = await siphon_automator.execute_manual_siphon(force=True)

        assert result.status == SiphonStatus.NO_PROFIT
        assert result.withdrawal_amount == Decimal("0.00")
        assert result.withdrawal_success is True  # No withdrawal needed

    @pytest.mark.asyncio
    async def test_safety_block_on_unsafe_withdrawal(self, siphon_automator):
        """Test that unsafe withdrawals are blocked."""
        # Setup profit calculation with unsafe withdrawal
        mock_calc = Mock()
        mock_calc.profit_status = ProfitStatus.PROFIT_AVAILABLE
        mock_calc.withdrawal_amount = Decimal("60.00")
        mock_calc.current_value = Decimal("210.00")

        siphon_automator.profit_calculator.calculate_weekly_profit = Mock(return_value=mock_calc)
        siphon_automator.profit_calculator.validate_withdrawal_safety = Mock(
            return_value=(False, "Would breach base capital")
        )

        result = await siphon_automator.execute_manual_siphon(force=True)

        assert result.status == SiphonStatus.SAFETY_BLOCK
        assert result.withdrawal_amount == Decimal("0.00")
        assert result.withdrawal_success is False
        assert "Would breach base capital" in result.errors[0]

    @pytest.mark.asyncio
    async def test_minimum_withdrawal_threshold(self, siphon_automator):
        """Test minimum withdrawal threshold enforcement."""
        # Setup profit calculation with small withdrawal
        mock_calc = Mock()
        mock_calc.profit_status = ProfitStatus.PROFIT_AVAILABLE
        mock_calc.withdrawal_amount = Decimal("5.00")  # Below $10 minimum
        mock_calc.current_value = Decimal("205.00")

        siphon_automator.profit_calculator.calculate_weekly_profit = Mock(return_value=mock_calc)
        siphon_automator.profit_calculator.validate_withdrawal_safety = Mock(return_value=(True, "Safe"))

        result = await siphon_automator.execute_manual_siphon(force=True)

        assert result.status == SiphonStatus.NO_PROFIT  # Treated as no profit due to threshold
        assert result.withdrawal_amount == Decimal("0.00")
        assert "Below minimum withdrawal threshold" in result.safety_checks

    def test_scheduler_control(self, siphon_automator):
        """Test scheduler start/stop functionality."""
        # Start scheduler
        success = siphon_automator.start_scheduler()
        assert success is True
        assert siphon_automator.is_running is True

        # Stop scheduler
        siphon_automator.stop_scheduler()
        assert siphon_automator.is_running is False

    def test_execution_history_tracking(self, siphon_automator):
        """Test that execution history is properly tracked."""
        # Add some mock results to history
        from src.cycles.weekly_siphon_automator import SiphonResult

        result1 = SiphonResult(
            timestamp=datetime.now(),
            status=SiphonStatus.SUCCESS,
            profit_calculation=None,
            withdrawal_amount=Decimal("25.00"),
            withdrawal_success=True,
            broker_confirmation="WD123",
            safety_checks=["All checks passed"],
            errors=[]
        )

        siphon_automator.execution_history.append(result1)

        history = siphon_automator.get_execution_history(limit=5)
        assert len(history) == 1
        assert history[0].withdrawal_amount == Decimal("25.00")

    def test_status_reporting(self, siphon_automator):
        """Test status reporting functionality."""
        status = siphon_automator.get_status()

        assert 'is_running' in status
        assert 'auto_execution_enabled' in status
        assert 'current_time_et' in status
        assert 'should_execute' in status
        assert 'total_executions' in status
        assert 'successful_executions' in status


@pytest.mark.asyncio
async def test_integration_weekly_cycle_with_siphon():
    """Integration test for WeeklyCycle with siphon automation."""

    # Mock all dependencies
    mock_portfolio = AsyncMock()
    mock_portfolio.sync_with_broker.return_value = True
    mock_portfolio.get_total_portfolio_value.return_value = Decimal("250.00")
    mock_portfolio.get_gate_positions.return_value = {}

    mock_trade_executor = Mock()
    mock_trade_executor.broker = AsyncMock()
    mock_trade_executor.broker.withdraw_funds.return_value = True

    mock_market_data = Mock()
    mock_holiday_calendar = Mock()
    mock_holiday_calendar.is_market_holiday.return_value = False

    # Create WeeklyCycle with siphon automation enabled
    weekly_cycle = WeeklyCycle(
        portfolio_manager=mock_portfolio,
        trade_executor=mock_trade_executor,
        market_data=mock_market_data,
        holiday_calendar=mock_holiday_calendar,
        enable_dpi=False,  # Disable DPI for simpler testing
        enable_siphon_automation=True,
        initial_capital=Decimal("200.00")
    )

    # Test that siphon automator is initialized
    assert weekly_cycle.siphon_automator is not None
    assert weekly_cycle.profit_calculator is not None

    # Test enhanced siphon phase execution
    siphon_result = await weekly_cycle.execute_siphon_phase("G0")

    assert 'profit_withdrawal' in siphon_result
    assert siphon_result['success'] is True

    # Test status includes siphon information
    status = weekly_cycle.get_cycle_status()
    assert 'siphon_automation' in status
    assert 'profit_summary' in status

    # Cleanup
    weekly_cycle.stop_siphon_automation()


def test_profit_calculator_edge_cases():
    """Test edge cases in profit calculation."""
    calculator = ProfitCalculator(Decimal("200.00"))
    mock_portfolio = Mock()

    # Test with exactly base capital (no profit/loss)
    mock_portfolio.get_total_portfolio_value.return_value = Decimal("200.00")
    mock_portfolio.get_nav_at_date.return_value = Decimal("200.00")
    mock_portfolio.get_deposits_in_period.return_value = Decimal("0.00")
    mock_portfolio.get_withdrawals_in_period.return_value = Decimal("0.00")

    calculation = calculator.calculate_weekly_profit(mock_portfolio)
    assert calculation.profit_status == ProfitStatus.NO_PROFIT

    # Test with very small profit (odd number for split testing)
    mock_portfolio.get_total_portfolio_value.return_value = Decimal("200.01")

    calculation = calculator.calculate_weekly_profit(mock_portfolio)
    assert calculation.total_profit == Decimal("0.01")
    # Test that split is handled correctly (should be 0.01 and 0.00 or 0.00 and 0.01)
    assert calculation.reinvestment_amount + calculation.withdrawal_amount == Decimal("0.01")


if __name__ == "__main__":
    """Run the tests."""
    pytest.main([__file__, "-v"])