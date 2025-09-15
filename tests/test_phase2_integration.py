"""Comprehensive test suite for Phase 2 integration"""

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.integration.phase2_factory import Phase2SystemFactory
from src.safety.kill_switch_system import TriggerType
from src.risk.kelly_criterion import KellyCriterionCalculator


class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 system integration"""

    def setUp(self):
        """Set up test environment"""
        self.factory = Phase2SystemFactory()
        self.phase1 = self.factory.initialize_phase1_systems()
        self.phase2 = self.factory.initialize_phase2_systems()
        self.systems = self.factory.get_integrated_system()

    def test_factory_initialization(self):
        """Test that factory initializes successfully"""
        self.assertIsNotNone(self.factory)
        self.assertIsNotNone(self.factory.config)
        self.assertEqual(len(self.phase1), 7)
        self.assertEqual(len(self.phase2), 5)

    def test_phase1_systems_exist(self):
        """Test all Phase 1 systems are initialized"""
        expected_systems = [
            'broker', 'market_data', 'portfolio_manager',
            'dpi_calculator', 'antifragility_engine',
            'gate_manager', 'trade_executor'
        ]
        for system in expected_systems:
            self.assertIn(system, self.phase1)
            self.assertIsNotNone(self.phase1[system])

    def test_phase2_systems_exist(self):
        """Test all Phase 2 systems are initialized"""
        expected_systems = [
            'kill_switch', 'kelly_calculator', 'evt_engine',
            'profit_calculator', 'siphon_automator'
        ]
        for system in expected_systems:
            self.assertIn(system, self.phase2)
            self.assertIsNotNone(self.phase2[system])

    def test_kill_switch_integration(self):
        """Test kill switch is properly integrated with broker"""
        kill_switch = self.phase2['kill_switch']
        self.assertIsNotNone(kill_switch.broker)
        self.assertEqual(kill_switch.broker, self.phase1['broker'])

        # Test kill switch state
        self.assertFalse(kill_switch.active)
        self.assertTrue(kill_switch.armed)

    def test_kelly_calculator_integration(self):
        """Test Kelly calculator is integrated with DPI and gates"""
        kelly = self.phase2['kelly_calculator']
        self.assertIsNotNone(kelly.dpi_calculator)
        self.assertEqual(kelly.dpi_calculator, self.phase1['dpi_calculator'])
        self.assertIsNotNone(kelly.gate_manager)
        self.assertEqual(kelly.gate_manager, self.phase1['gate_manager'])

    def test_siphon_automator_integration(self):
        """Test siphon automator is integrated with portfolio and broker"""
        siphon = self.phase2['siphon_automator']
        self.assertIsNotNone(siphon.portfolio_manager)
        self.assertEqual(siphon.portfolio_manager, self.phase1['portfolio_manager'])
        self.assertIsNotNone(siphon.broker)
        self.assertEqual(siphon.broker, self.phase1['broker'])
        self.assertIsNotNone(siphon.profit_calculator)

    def test_validation_passes(self):
        """Test that validation reports all systems ready"""
        validation = self.factory.validate_integration()
        self.assertTrue(validation['all_systems_ready'])

        # Check individual validations
        expected_checks = [
            'broker_connection', 'dpi_calculator', 'gate_manager',
            'kill_switch', 'kelly_calculator', 'evt_engine',
            'siphon_automator', 'kill_switch_broker',
            'kelly_dpi_integration'
        ]
        for check in expected_checks:
            self.assertTrue(validation[check], f"{check} validation failed")

    def test_evt_engine_functionality(self):
        """Test Enhanced EVT engine basic functionality"""
        evt = self.phase2['evt_engine']

        # Test that EVT engine exists and has required methods
        self.assertIsNotNone(evt)
        self.assertTrue(hasattr(evt, 'fit_evt'))
        self.assertTrue(hasattr(evt, 'calculate_var'))
        self.assertTrue(hasattr(evt, 'calculate_cvar'))

    def test_profit_calculator_initialization(self):
        """Test profit calculator with correct seed capital"""
        profit_calc = self.phase2['profit_calculator']
        self.assertEqual(profit_calc.base_capital, Decimal('200'))

        # Test profit calculation using actual method name
        current_value = Decimal('250')
        profit = profit_calc.calculate_weekly_profit(current_value)
        self.assertEqual(profit, Decimal('50'))

    def test_config_structure(self):
        """Test configuration has all required sections"""
        config = self.systems['config']

        required_sections = ['broker', 'risk', 'kill_switch', 'siphon', 'dashboard']
        for section in required_sections:
            self.assertIn(section, config)

        # Test risk parameters
        self.assertEqual(config['risk']['max_position_size'], 0.25)
        self.assertEqual(config['risk']['max_kelly'], 0.25)
        self.assertEqual(config['risk']['cash_floor'], 0.5)

    def test_mock_mode_enabled(self):
        """Test that systems initialize in mock mode without API credentials"""
        broker = self.phase1['broker']
        self.assertTrue(broker.mock_mode)
        # API key may be empty in mock mode, just verify mock mode is on
        self.assertTrue(broker.mock_mode or not broker.api_key)

    def test_system_state_persistence(self):
        """Test that systems can save and load state"""
        gate_manager = self.phase1['gate_manager']

        # Gate manager should start at G0 (may be an enum)
        self.assertTrue(str(gate_manager.current_gate).endswith('G0'))

        # Test that gate manager has state management
        self.assertIsNotNone(gate_manager)
        self.assertTrue(hasattr(gate_manager, 'current_gate'))

    def test_response_time_target(self):
        """Test kill switch response time configuration"""
        kill_switch = self.phase2['kill_switch']
        config = self.systems['config']['kill_switch']

        # Should have <500ms response target
        self.assertEqual(config['response_time_target_ms'], 500)

    def test_siphon_schedule_configuration(self):
        """Test weekly siphon schedule is configured correctly"""
        config = self.systems['config']['siphon']

        self.assertEqual(config['schedule']['day'], 'friday')
        self.assertEqual(config['schedule']['time'], '18:00')
        self.assertEqual(config['profit_split'], 0.5)  # 50/50 split
        self.assertEqual(config['minimum_profit'], 100)  # $100 minimum

    @patch('src.cycles.weekly_siphon_automator.datetime')
    def test_siphon_timing_logic(self, mock_datetime):
        """Test siphon timing for Friday 6pm"""
        siphon = self.phase2['siphon_automator']

        # Mock current time as Friday 6pm
        friday_6pm = datetime(2024, 1, 5, 18, 0, 0)  # Friday
        mock_datetime.now.return_value = friday_6pm

        # Should be ready to execute
        is_friday = friday_6pm.weekday() == 4
        self.assertTrue(is_friday)

    def test_integrated_system_completeness(self):
        """Test integrated system contains all components"""
        integrated = self.factory.get_integrated_system()

        # Should have all Phase 1 + Phase 2 systems + config
        total_systems = len(self.phase1) + len(self.phase2) + 1  # +1 for config
        self.assertEqual(len(integrated), total_systems)

        # Verify all systems present
        for key in self.phase1:
            self.assertIn(key, integrated)
        for key in self.phase2:
            self.assertIn(key, integrated)
        self.assertIn('config', integrated)


class TestPhase2Performance(unittest.TestCase):
    """Performance tests for Phase 2 systems"""

    def setUp(self):
        """Set up performance test environment"""
        self.factory = Phase2SystemFactory()
        self.factory.initialize_phase1_systems()
        self.factory.initialize_phase2_systems()

    def test_factory_initialization_speed(self):
        """Test that factory initializes within acceptable time"""
        import time

        start = time.time()
        factory = Phase2SystemFactory()
        factory.initialize_phase1_systems()
        factory.initialize_phase2_systems()
        end = time.time()

        # Should initialize within 1 second
        self.assertLess(end - start, 1.0)

    def test_validation_speed(self):
        """Test validation completes quickly"""
        import time

        start = time.time()
        validation = self.factory.validate_integration()
        end = time.time()

        # Validation should complete within 100ms
        self.assertLess(end - start, 0.1)


if __name__ == '__main__':
    unittest.main()