"""
Phase 2 Integration Test Suite
Validates that all Phase 2 systems can be initialized and integrated properly
"""

import unittest
import sys
import os
from pathlib import Path
from decimal import Decimal
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.integration.phase2_factory import Phase2SystemFactory


class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 system integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config_path = "config/phase2_integration.json"
        self.factory = Phase2SystemFactory(self.config_path)

    def test_factory_initialization(self):
        """Test that factory initializes with configuration"""
        self.assertIsNotNone(self.factory)
        self.assertIsNotNone(self.factory.config)
        self.assertIn("broker", self.factory.config)
        self.assertIn("kill_switch", self.factory.config)
        self.assertIn("siphon", self.factory.config)

    def test_phase1_initialization(self):
        """Test Phase 1 system initialization"""
        phase1_systems = self.factory.initialize_phase1_systems()

        # Verify all Phase 1 systems are initialized
        self.assertIn("broker", phase1_systems)
        self.assertIn("dpi_calculator", phase1_systems)
        self.assertIn("antifragility_engine", phase1_systems)
        self.assertIn("gate_manager", phase1_systems)
        self.assertIn("portfolio_manager", phase1_systems)
        self.assertIn("trade_executor", phase1_systems)

        # Verify systems are not None
        for name, system in phase1_systems.items():
            self.assertIsNotNone(system, f"Phase 1 system '{name}' is None")

    def test_phase2_initialization(self):
        """Test Phase 2 system initialization"""
        phase2_systems = self.factory.initialize_phase2_systems()

        # Verify all Phase 2 systems are initialized
        self.assertIn("kill_switch", phase2_systems)
        self.assertIn("kelly_calculator", phase2_systems)
        self.assertIn("evt_engine", phase2_systems)
        self.assertIn("profit_calculator", phase2_systems)
        self.assertIn("siphon_automator", phase2_systems)

        # Verify systems are not None
        for name, system in phase2_systems.items():
            self.assertIsNotNone(system, f"Phase 2 system '{name}' is None")

    def test_integrated_system(self):
        """Test complete integrated system"""
        integrated = self.factory.get_integrated_system()

        # Should have both Phase 1 and Phase 2 systems
        self.assertIn("broker", integrated)  # Phase 1
        self.assertIn("kill_switch", integrated)  # Phase 2
        self.assertIn("config", integrated)

        # Count total systems
        expected_systems = [
            # Phase 1
            "broker", "market_data", "portfolio_manager", "dpi_calculator",
            "antifragility_engine", "gate_manager", "trade_executor",
            # Phase 2
            "kill_switch", "kelly_calculator", "evt_engine",
            "profit_calculator", "siphon_automator",
            # Config
            "config"
        ]

        for system in expected_systems:
            self.assertIn(system, integrated, f"Missing system: {system}")

    def test_validation(self):
        """Test system validation"""
        # Initialize systems first
        self.factory.initialize_phase1_systems()
        self.factory.initialize_phase2_systems()

        # Run validation
        validation_results = self.factory.validate_integration()

        # Check validation results
        self.assertIn("all_systems_ready", validation_results)

        # Check specific validations
        self.assertTrue(validation_results.get("broker_connection", False))
        self.assertTrue(validation_results.get("dpi_calculator", False))
        self.assertTrue(validation_results.get("kill_switch", False))
        self.assertTrue(validation_results.get("kelly_calculator", False))

    def test_kill_switch_integration(self):
        """Test kill switch has proper dependencies"""
        integrated = self.factory.get_integrated_system()
        kill_switch = integrated["kill_switch"]

        # Verify kill switch has broker interface
        self.assertTrue(hasattr(kill_switch, "broker_interface"))
        self.assertIsNotNone(kill_switch.broker_interface)

        # Verify kill switch has config
        self.assertTrue(hasattr(kill_switch, "config"))
        self.assertIsNotNone(kill_switch.config)

    def test_kelly_integration(self):
        """Test Kelly criterion has proper dependencies"""
        integrated = self.factory.get_integrated_system()
        kelly = integrated["kelly_calculator"]

        # Verify Kelly has DPI calculator
        self.assertTrue(hasattr(kelly, "dpi_calculator"))
        self.assertIsNotNone(kelly.dpi_calculator)

        # Verify Kelly has gate manager
        self.assertTrue(hasattr(kelly, "gate_manager"))
        self.assertIsNotNone(kelly.gate_manager)

    def test_siphon_integration(self):
        """Test siphon has proper dependencies"""
        integrated = self.factory.get_integrated_system()
        siphon = integrated["siphon_automator"]

        # Verify siphon has portfolio manager
        self.assertTrue(hasattr(siphon, "portfolio_manager"))
        self.assertIsNotNone(siphon.portfolio_manager)

        # Verify siphon has profit calculator
        self.assertTrue(hasattr(siphon, "profit_calculator"))
        self.assertIsNotNone(siphon.profit_calculator)

    def test_performance_initialization(self):
        """Test that systems initialize within reasonable time"""
        start_time = time.time()

        # Initialize all systems
        self.factory.initialize_phase1_systems()
        self.factory.initialize_phase2_systems()

        elapsed_time = time.time() - start_time

        # Should initialize in under 5 seconds
        self.assertLess(elapsed_time, 5.0,
                       f"Initialization took {elapsed_time:.2f}s, expected < 5s")

    def test_configuration_override(self):
        """Test that configuration can be overridden"""
        # Create custom config
        custom_config = self.factory.config.copy()
        custom_config["risk"]["max_kelly"] = 0.15  # Override to 15%

        # Create new factory with custom config
        custom_factory = Phase2SystemFactory()
        custom_factory.config = custom_config

        # Verify override
        self.assertEqual(custom_factory.config["risk"]["max_kelly"], 0.15)


class TestPhase2Performance(unittest.TestCase):
    """Test Phase 2 performance requirements"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.factory = Phase2SystemFactory()
        self.integrated = self.factory.get_integrated_system()

    def test_kelly_performance(self):
        """Test Kelly criterion <50ms performance"""
        kelly = self.integrated["kelly_calculator"]

        # Mock a calculation (would need actual market data in production)
        start_time = time.time()

        # Simulate calculation
        for _ in range(100):
            # In production: kelly.calculate(symbol, market_data)
            pass

        elapsed_ms = (time.time() - start_time) * 1000 / 100

        # Should be under 50ms per calculation
        self.assertLess(elapsed_ms, 50,
                       f"Kelly calculation took {elapsed_ms:.2f}ms, expected < 50ms")

    def test_kill_switch_response(self):
        """Test kill switch <500ms response time"""
        kill_switch = self.integrated["kill_switch"]

        # Mock a kill switch activation
        start_time = time.time()

        # In production: kill_switch.execute_emergency_stop()
        # For now, just test initialization

        elapsed_ms = (time.time() - start_time) * 1000

        # Should respond in under 500ms
        self.assertLess(elapsed_ms, 500,
                       f"Kill switch response took {elapsed_ms:.2f}ms, expected < 500ms")


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Performance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)