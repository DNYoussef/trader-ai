"""Tests for weekly buy/siphon cycle"""
import unittest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch
import pytz

from src.cycles.weekly_cycle import WeeklyCycle, WeeklyDelta
from src.brokers.broker_interface import Position

class TestWeeklyCycle(unittest.TestCase):
    """Test weekly cycle functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_broker = Mock()
        self.mock_gate_manager = Mock()
        self.weekly_cycle = WeeklyCycle(self.mock_broker, self.mock_gate_manager)

    def test_friday_detection(self):
        """Test Friday 4:10pm ET detection"""
        et_tz = pytz.timezone('America/New_York')

        # Test Friday 4:10pm ET - should trigger
        friday_410 = datetime(2024, 1, 5, 16, 10, tzinfo=et_tz)  # Friday
        with patch('src.cycles.weekly_cycle.datetime') as mock_datetime:
            mock_datetime.now.return_value = friday_410
            self.assertTrue(self.weekly_cycle.should_execute_buy())

        # Test Friday 6:00pm ET - should trigger siphon
        friday_600 = datetime(2024, 1, 5, 18, 0, tzinfo=et_tz)  # Friday
        with patch('src.cycles.weekly_cycle.datetime') as mock_datetime:
            mock_datetime.now.return_value = friday_600
            self.assertTrue(self.weekly_cycle.should_execute_siphon())

        # Test Thursday - should not trigger
        thursday = datetime(2024, 1, 4, 16, 10, tzinfo=et_tz)  # Thursday
        with patch('src.cycles.weekly_cycle.datetime') as mock_datetime:
            mock_datetime.now.return_value = thursday
            self.assertFalse(self.weekly_cycle.should_execute_buy())

    def test_weekly_delta_calculation(self):
        """Test weekly delta calculation with 50/50 split"""
        # Set up mock data
        self.weekly_cycle.last_week_nav = Decimal('200.00')
        self.mock_broker.get_account_value.return_value = Decimal('220.00')
        self.weekly_cycle.weekly_deposits = Decimal('0')
        self.weekly_cycle.weekly_withdrawals = Decimal('0')

        # Calculate delta
        delta = self.weekly_cycle.calculate_weekly_delta()

        # Verify calculations
        self.assertEqual(delta.delta, Decimal('20.00'))  # $20 profit
        self.assertEqual(delta.reinvest_amount, Decimal('10.00'))  # 50% reinvest
        self.assertEqual(delta.siphon_amount, Decimal('10.00'))  # 50% siphon

    def test_cash_floor_enforcement(self):
        """Test that cash floor is maintained"""
        # Set up gate config
        mock_config = Mock()
        mock_config.cash_floor_pct = Decimal('0.50')
        mock_config.allowed_symbols = ['ULTY', 'AMDY']
        mock_config.max_ticket_size = Decimal('25')
        self.mock_gate_manager.get_current_gate_config.return_value = mock_config

        # Set up broker state
        self.mock_broker.get_account_value.return_value = Decimal('200')
        self.mock_broker.get_cash_balance.return_value = Decimal('100')
        self.mock_broker.get_positions.return_value = []
        self.mock_broker.get_market_price.return_value = Decimal('5.57')

        # Mock validation to always pass for this test
        self.mock_gate_manager.validate_trade.return_value = (True, {})

        # Execute buy phase
        result = self.weekly_cycle.execute_buy_phase()

        # Should not invest more than 50% (cash floor)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'insufficient_cash')

    def test_fractional_shares_rounding(self):
        """Test fractional shares are rounded to 6 decimal places"""
        # Set up gate config
        mock_config = Mock()
        mock_config.cash_floor_pct = Decimal('0.50')
        mock_config.allowed_symbols = ['ULTY', 'AMDY']
        mock_config.max_ticket_size = Decimal('25')
        self.mock_gate_manager.get_current_gate_config.return_value = mock_config
        self.mock_gate_manager.current_gate = 'G0'

        # Set up broker state with sufficient cash
        self.mock_broker.get_account_value.return_value = Decimal('200')
        self.mock_broker.get_cash_balance.return_value = Decimal('150')  # More than 50%
        self.mock_broker.get_positions.return_value = []
        self.mock_broker.get_market_price.side_effect = [Decimal('5.57'), Decimal('7.72')]
        self.mock_broker.submit_order.return_value = 'test_order_id'

        # Mock validation to always pass
        self.mock_gate_manager.validate_trade.return_value = (True, {})

        # Execute buy phase
        result = self.weekly_cycle.execute_buy_phase()

        # Verify orders were placed
        self.assertEqual(result['status'], 'executed')
        self.assertEqual(len(result['orders']), 2)  # ULTY and AMDY

        # Check that quantities have max 6 decimal places
        for order in result['orders']:
            quantity_str = str(order['quantity'])
            if '.' in quantity_str:
                decimal_places = len(quantity_str.split('.')[1])
                self.assertLessEqual(decimal_places, 6)

    def test_market_holiday_handling(self):
        """Test that market holidays are handled"""
        # Test New Year's Day
        et_tz = pytz.timezone('America/New_York')
        new_years = datetime(2024, 1, 1, 16, 10, tzinfo=et_tz)

        with patch('src.cycles.weekly_cycle.datetime') as mock_datetime:
            mock_datetime.now.return_value = new_years
            self.assertTrue(self.weekly_cycle.handle_market_holiday())

        # Test regular day
        regular_day = datetime(2024, 1, 5, 16, 10, tzinfo=et_tz)
        with patch('src.cycles.weekly_cycle.datetime') as mock_datetime:
            mock_datetime.now.return_value = regular_day
            self.assertFalse(self.weekly_cycle.handle_market_holiday())

if __name__ == '__main__':
    unittest.main()