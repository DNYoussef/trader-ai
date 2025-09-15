"""
Tests for the WeeklyCycle system

Tests all major functionality including:
- Buy/siphon phase execution timing
- Market holiday handling
- Gate allocation logic  
- Weekly delta calculations
- Timezone handling
"""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch
import pytz

from src.cycles.weekly_cycle import (
    WeeklyCycle, 
    CyclePhase, 
    GateAllocation, 
    WeeklyDelta
)
from src.utils.holiday_calendar import MarketHolidayCalendar


class TestGateAllocation:
    """Test gate allocation configuration"""
    
    def test_valid_allocation(self):
        """Test valid allocation sums to 100%"""
        allocation = GateAllocation(ulty_pct=70.0, amdy_pct=30.0)
        allocation.validate()  # Should not raise
    
    def test_invalid_allocation(self):
        """Test invalid allocation raises error"""
        allocation = GateAllocation(ulty_pct=70.0, amdy_pct=25.0)
        with pytest.raises(ValueError, match="must sum to 100%"):
            allocation.validate()
    
    def test_four_asset_allocation(self):
        """Test G1 four-asset allocation"""
        allocation = GateAllocation(
            ulty_pct=50.0, 
            amdy_pct=20.0, 
            iau_pct=15.0, 
            vtip_pct=15.0
        )
        allocation.validate()  # Should not raise


class TestWeeklyCycle:
    """Test WeeklyCycle functionality"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        portfolio_manager = Mock()
        trade_executor = Mock()
        market_data = Mock()
        holiday_calendar = Mock(spec=MarketHolidayCalendar)
        
        return {
            'portfolio_manager': portfolio_manager,
            'trade_executor': trade_executor,
            'market_data': market_data,
            'holiday_calendar': holiday_calendar
        }
    
    @pytest.fixture
    def weekly_cycle(self, mock_dependencies):
        """Create WeeklyCycle instance with mocks"""
        return WeeklyCycle(**mock_dependencies)
    
    def test_initialization(self, mock_dependencies):
        """Test WeeklyCycle initialization"""
        cycle = WeeklyCycle(**mock_dependencies)
        
        assert cycle.portfolio_manager is not None
        assert cycle.trade_executor is not None
        assert cycle.market_data is not None
        assert cycle.holiday_calendar is not None
        assert len(cycle.GATE_ALLOCATIONS) == 2
        assert 'G0' in cycle.GATE_ALLOCATIONS
        assert 'G1' in cycle.GATE_ALLOCATIONS
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_buy_friday_correct_time(self, mock_datetime, weekly_cycle):
        """Test buy execution on Friday at correct time"""
        # Friday 4:15 PM ET
        friday_415pm = datetime(2024, 1, 5, 16, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_415pm
        
        weekly_cycle.holiday_calendar.is_market_holiday.return_value = False
        
        assert weekly_cycle.should_execute_buy() == True
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_buy_friday_early(self, mock_datetime, weekly_cycle):
        """Test no buy execution on Friday before 4:10 PM"""
        # Friday 4:00 PM ET
        friday_400pm = datetime(2024, 1, 5, 16, 0, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_400pm
        
        assert weekly_cycle.should_execute_buy() == False
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_buy_not_friday(self, mock_datetime, weekly_cycle):
        """Test no buy execution on non-Friday"""
        # Thursday 4:15 PM ET
        thursday_415pm = datetime(2024, 1, 4, 16, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = thursday_415pm
        
        assert weekly_cycle.should_execute_buy() == False
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_buy_market_holiday(self, mock_datetime, weekly_cycle):
        """Test no buy execution on market holiday"""
        # Friday 4:15 PM ET (but market holiday)
        friday_415pm = datetime(2024, 1, 5, 16, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_415pm
        
        weekly_cycle.holiday_calendar.is_market_holiday.return_value = True
        
        assert weekly_cycle.should_execute_buy() == False
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_siphon_friday_correct_time(self, mock_datetime, weekly_cycle):
        """Test siphon execution on Friday at correct time"""
        # Friday 6:15 PM ET
        friday_615pm = datetime(2024, 1, 5, 18, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_615pm
        
        # Set up buy execution earlier this week
        friday_415pm = datetime(2024, 1, 5, 16, 15, tzinfo=WeeklyCycle.ET)
        weekly_cycle._last_buy_execution = friday_415pm
        
        weekly_cycle.holiday_calendar.is_market_holiday.return_value = False
        
        assert weekly_cycle.should_execute_siphon() == True
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_should_execute_siphon_no_buy_execution(self, mock_datetime, weekly_cycle):
        """Test no siphon execution without prior buy execution"""
        # Friday 6:15 PM ET
        friday_615pm = datetime(2024, 1, 5, 18, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_615pm
        
        weekly_cycle.holiday_calendar.is_market_holiday.return_value = False
        
        assert weekly_cycle.should_execute_siphon() == False
    
    def test_execute_buy_phase_g0(self, weekly_cycle):
        """Test buy phase execution for G0 gate"""
        available_cash = 10000.0
        
        # Mock successful trade execution
        weekly_cycle.trade_executor.buy_market_order.return_value = {
            'success': True,
            'shares': 100,
            'price': 50.0
        }
        
        result = weekly_cycle.execute_buy_phase('G0', available_cash)
        
        assert result['success'] == True
        assert result['gate'] == 'G0'
        assert result['phase'] == CyclePhase.BUY.value
        assert len(result['trades']) == 2  # ULTY and AMDY
        
        # Verify ULTY allocation (70%)
        ulty_trade = next(t for t in result['trades'] if t['symbol'] == 'ULTY')
        assert ulty_trade['amount'] == 7000.0
        
        # Verify AMDY allocation (30%)
        amdy_trade = next(t for t in result['trades'] if t['symbol'] == 'AMDY')
        assert amdy_trade['amount'] == 3000.0
    
    def test_execute_buy_phase_g1(self, weekly_cycle):
        """Test buy phase execution for G1 gate"""
        available_cash = 10000.0
        
        # Mock successful trade execution
        weekly_cycle.trade_executor.buy_market_order.return_value = {
            'success': True,
            'shares': 100,
            'price': 50.0
        }
        
        result = weekly_cycle.execute_buy_phase('G1', available_cash)
        
        assert result['success'] == True
        assert result['gate'] == 'G1'
        assert len(result['trades']) == 4  # ULTY, AMDY, IAU, VTIP
        
        # Verify allocations
        symbols_amounts = {t['symbol']: t['amount'] for t in result['trades']}
        
        assert symbols_amounts['ULTY'] == 5000.0  # 50%
        assert symbols_amounts['AMDY'] == 2000.0  # 20%
        assert symbols_amounts['IAU'] == 1500.0   # 15%
        assert symbols_amounts['VTIP'] == 1500.0  # 15%
    
    def test_execute_buy_phase_invalid_gate(self, weekly_cycle):
        """Test buy phase with invalid gate"""
        with pytest.raises(ValueError, match="Unknown gate"):
            weekly_cycle.execute_buy_phase('INVALID', 1000.0)
    
    def test_execute_siphon_phase(self, weekly_cycle):
        """Test siphon phase execution (50/50 rebalancing)"""
        # Mock portfolio positions
        mock_positions = {
            'ULTY': Mock(market_value=6000.0),  # Overweight
            'AMDY': Mock(market_value=4000.0),  # Underweight
        }
        
        weekly_cycle.portfolio_manager.get_gate_positions.return_value = mock_positions
        
        # Mock successful trade execution
        weekly_cycle.trade_executor.sell_market_order.return_value = {'success': True}
        weekly_cycle.trade_executor.buy_market_order.return_value = {'success': True}
        
        result = weekly_cycle.execute_siphon_phase('G0')
        
        assert result['success'] == True
        assert result['gate'] == 'G0'
        assert result['phase'] == CyclePhase.SIPHON.value
        
        # Should have operations for both positions (target = 5000 each)
        # ULTY: sell 1000, AMDY: buy 1000
        assert len(result['operations']) == 2
    
    def test_calculate_weekly_delta(self, weekly_cycle):
        """Test weekly delta calculation"""
        week_start = datetime(2024, 1, 1, tzinfo=WeeklyCycle.ET)
        
        # Mock NAV and cash flow data
        weekly_cycle.portfolio_manager.get_nav_at_date.side_effect = [100000.0, 105000.0]
        weekly_cycle.portfolio_manager.get_deposits_in_period.return_value = 2000.0
        weekly_cycle.portfolio_manager.get_withdrawals_in_period.return_value = 0.0
        
        delta = weekly_cycle.calculate_weekly_delta(week_start)
        
        assert delta.nav_start == 100000.0
        assert delta.nav_end == 105000.0
        assert delta.deposits == 2000.0
        assert delta.withdrawals == 0.0
        # Delta = (105000 - 100000) - (2000 - 0) = 3000
        assert delta.delta == 3000.0
        assert delta.delta_pct == 3.0  # 3000/100000 * 100
    
    @patch('src.cycles.weekly_cycle.datetime')
    def test_get_cycle_status(self, mock_datetime, weekly_cycle):
        """Test cycle status retrieval"""
        # Friday 4:15 PM ET
        friday_415pm = datetime(2024, 1, 5, 16, 15, tzinfo=WeeklyCycle.ET)
        mock_datetime.now.return_value = friday_415pm
        
        weekly_cycle.holiday_calendar.is_market_holiday.return_value = False
        
        status = weekly_cycle.get_cycle_status()
        
        assert 'current_time_et' in status
        assert 'current_phase' in status
        assert 'should_execute_buy' in status
        assert 'should_execute_siphon' in status
        assert status['should_execute_buy'] == True
    
    def test_handle_market_holiday(self, weekly_cycle):
        """Test market holiday handling"""
        holiday_date = date(2024, 7, 4)  # July 4th
        next_trading_day = date(2024, 7, 5)
        
        weekly_cycle.holiday_calendar.get_next_trading_day.return_value = next_trading_day
        
        result = weekly_cycle.handle_market_holiday(holiday_date)
        
        assert result['holiday_date'] == holiday_date
        assert result['next_trading_day'] == next_trading_day
        assert result['action'] == 'defer_execution'
    
    def test_get_weekly_performance(self, weekly_cycle):
        """Test weekly performance data retrieval"""
        # Add some mock weekly deltas
        for i in range(5):
            delta = WeeklyDelta(
                week_start=datetime(2024, 1, 1 + i*7, tzinfo=WeeklyCycle.ET),
                week_end=datetime(2024, 1, 8 + i*7, tzinfo=WeeklyCycle.ET),
                nav_start=100000.0,
                nav_end=101000.0,
                deposits=0.0,
                withdrawals=0.0,
                delta=1000.0,
                delta_pct=1.0
            )
            weekly_cycle._weekly_deltas.append(delta)
        
        # Get last 3 weeks
        recent_performance = weekly_cycle.get_weekly_performance(weeks=3)
        
        assert len(recent_performance) == 3
        assert all(isinstance(delta, WeeklyDelta) for delta in recent_performance)


class TestWeeklyDelta:
    """Test WeeklyDelta data class"""
    
    def test_weekly_delta_creation(self):
        """Test WeeklyDelta creation and attributes"""
        week_start = datetime(2024, 1, 1, tzinfo=WeeklyCycle.ET)
        week_end = datetime(2024, 1, 8, tzinfo=WeeklyCycle.ET)
        
        delta = WeeklyDelta(
            week_start=week_start,
            week_end=week_end,
            nav_start=100000.0,
            nav_end=102000.0,
            deposits=1000.0,
            withdrawals=0.0,
            delta=1000.0,  # 2000 gain - 1000 deposit
            delta_pct=1.0
        )
        
        assert delta.week_start == week_start
        assert delta.week_end == week_end
        assert delta.nav_start == 100000.0
        assert delta.nav_end == 102000.0
        assert delta.deposits == 1000.0
        assert delta.withdrawals == 0.0
        assert delta.delta == 1000.0
        assert delta.delta_pct == 1.0