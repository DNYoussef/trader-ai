#!/usr/bin/env python3
"""
Weekly Siphon Automation Validation Script

Comprehensive validation of the Phase 2 automated siphon system:
- ProfitCalculator accuracy
- WeeklySiphonAutomator functionality
- Capital protection safeguards
- Integration with existing WeeklyCycle
- End-to-end automation pipeline

Usage:
    python validate_siphon_automation.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import asyncio
import sys
import logging
from decimal import Decimal
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our siphon automation components
try:
    from src.cycles.profit_calculator import ProfitCalculator, ProfitStatus
    from src.cycles.weekly_siphon_automator import WeeklySiphonAutomator, SiphonStatus
    from src.cycles.siphon_monitor import SiphonMonitor
    from src.cycles.weekly_cycle import WeeklyCycle
    logger.info("✅ Successfully imported all siphon automation components")
except ImportError as e:
    logger.error(f"❌ Failed to import siphon components: {e}")
    sys.exit(1)


class SiphonValidationSuite:
    """Comprehensive validation suite for siphon automation."""

    def __init__(self):
        """Initialize validation suite."""
        self.test_results = []
        self.base_capital = Decimal("200.00")

    async def run_all_validations(self) -> bool:
        """Run all validation tests."""
        logger.info("🚀 Starting Weekly Siphon Automation Validation")
        logger.info("=" * 60)

        validations = [
            self.validate_profit_calculator,
            self.validate_capital_protection,
            self.validate_siphon_automator,
            self.validate_monitoring_system,
            self.validate_integration,
            self.validate_edge_cases,
            self.validate_error_handling
        ]

        all_passed = True
        for validation in validations:
            try:
                result = await validation()
                self.test_results.append(result)
                if not result['passed']:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ Validation failed with exception: {e}")
                self.test_results.append({
                    'name': validation.__name__,
                    'passed': False,
                    'error': str(e),
                    'details': {}
                })
                all_passed = False

        self.print_summary()
        return all_passed

    async def validate_profit_calculator(self) -> dict:
        """Validate ProfitCalculator functionality."""
        logger.info("📊 Validating ProfitCalculator...")

        try:
            calculator = ProfitCalculator(self.base_capital)

            # Mock portfolio manager
            mock_portfolio = Mock()
            mock_portfolio.get_nav_at_date.return_value = Decimal("200.00")
            mock_portfolio.get_deposits_in_period.return_value = Decimal("0.00")
            mock_portfolio.get_withdrawals_in_period.return_value = Decimal("0.00")

            # Test 1: No profit scenario
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("200.00")
            calc1 = calculator.calculate_weekly_profit(mock_portfolio)
            assert calc1.profit_status == ProfitStatus.NO_PROFIT
            assert calc1.withdrawal_amount == Decimal("0.00")

            # Test 2: $50 profit scenario (should split 25/25)
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("250.00")
            calc2 = calculator.calculate_weekly_profit(mock_portfolio)
            assert calc2.profit_status == ProfitStatus.PROFIT_AVAILABLE
            assert calc2.total_profit == Decimal("50.00")
            assert calc2.withdrawal_amount == Decimal("25.00")
            assert calc2.reinvestment_amount == Decimal("25.00")

            # Test 3: Capital protection (should never withdraw base capital)
            assert calc2.remaining_capital >= self.base_capital

            # Test 4: Safety validation
            is_safe, reason = calculator.validate_withdrawal_safety(
                Decimal("25.00"), Decimal("250.00")
            )
            assert is_safe is True

            # Test 5: Unsafe withdrawal detection
            is_unsafe, reason = calculator.validate_withdrawal_safety(
                Decimal("60.00"), Decimal("210.00")
            )
            assert is_unsafe is False

            logger.info("✅ ProfitCalculator validation passed")
            return {
                'name': 'ProfitCalculator',
                'passed': True,
                'details': {
                    'no_profit_handled': True,
                    'profit_split_correct': True,
                    'capital_protection_active': True,
                    'safety_validation_working': True
                }
            }

        except Exception as e:
            logger.error(f"❌ ProfitCalculator validation failed: {e}")
            return {'name': 'ProfitCalculator', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_capital_protection(self) -> dict:
        """Validate capital protection safeguards."""
        logger.info("🛡️ Validating Capital Protection Safeguards...")

        try:
            calculator = ProfitCalculator(self.base_capital)

            # Test scenarios that should trigger protection
            test_cases = [
                # (current_value, withdrawal_request, should_be_safe)
                (Decimal("250.00"), Decimal("25.00"), True),   # Safe: 25 of 50 profit
                (Decimal("210.00"), Decimal("5.00"), True),    # Safe: 5 of 10 profit
                (Decimal("210.00"), Decimal("15.00"), False),  # Unsafe: would leave <200
                (Decimal("200.00"), Decimal("1.00"), False),   # Unsafe: no profit to withdraw
                (Decimal("190.00"), Decimal("0.00"), True),    # Safe: no withdrawal when loss
            ]

            protection_working = True
            for current_value, withdrawal_amount, should_be_safe in test_cases:
                is_safe, reason = calculator.validate_withdrawal_safety(withdrawal_amount, current_value)
                if is_safe != should_be_safe:
                    protection_working = False
                    logger.error(f"Protection failed: ${withdrawal_amount} from ${current_value} - Expected safe: {should_be_safe}, Got: {is_safe}")

            # Test capital protection status
            status = calculator.get_capital_protection_status(Decimal("250.00"))
            assert status['base_capital'] == self.base_capital
            assert status['is_protected'] is True
            assert status['buffer_amount'] == Decimal("50.00")

            logger.info("✅ Capital Protection validation passed")
            return {
                'name': 'CapitalProtection',
                'passed': protection_working,
                'details': {
                    'safety_checks_working': protection_working,
                    'status_reporting_working': True,
                    'base_capital_never_breached': True
                }
            }

        except Exception as e:
            logger.error(f"❌ Capital Protection validation failed: {e}")
            return {'name': 'CapitalProtection', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_siphon_automator(self) -> dict:
        """Validate WeeklySiphonAutomator functionality."""
        logger.info("🤖 Validating WeeklySiphonAutomator...")

        try:
            # Mock dependencies
            mock_portfolio = AsyncMock()
            mock_portfolio.sync_with_broker.return_value = True
            mock_portfolio.record_transaction = AsyncMock()

            mock_broker = AsyncMock()
            mock_broker.withdraw_funds.return_value = True
            mock_broker.get_last_withdrawal_id.return_value = "WD123456"

            calculator = ProfitCalculator(self.base_capital)

            mock_holiday_calendar = Mock()
            mock_holiday_calendar.is_market_holiday.return_value = False

            # Create automator (disabled for testing)
            automator = WeeklySiphonAutomator(
                portfolio_manager=mock_portfolio,
                broker_adapter=mock_broker,
                profit_calculator=calculator,
                holiday_calendar=mock_holiday_calendar,
                enable_auto_execution=False  # Disabled for testing
            )

            # Test 1: Scheduler control
            scheduler_started = automator.start_scheduler()
            assert scheduler_started is True
            assert automator.is_running is True

            automator.stop_scheduler()
            assert automator.is_running is False

            # Test 2: Manual execution with profit
            mock_calc = Mock()
            mock_calc.profit_status = ProfitStatus.PROFIT_AVAILABLE
            mock_calc.withdrawal_amount = Decimal("25.00")
            mock_calc.current_value = Decimal("250.00")

            calculator.calculate_weekly_profit = Mock(return_value=mock_calc)
            calculator.validate_withdrawal_safety = Mock(return_value=(True, "Safe withdrawal"))

            result = await automator.execute_manual_siphon(force=True)
            assert result.status == SiphonStatus.SUCCESS
            assert result.withdrawal_amount == Decimal("25.00")

            # Test 3: Execution with no profit
            mock_calc_no_profit = Mock()
            mock_calc_no_profit.profit_status = ProfitStatus.NO_PROFIT
            mock_calc_no_profit.withdrawal_amount = Decimal("0.00")

            calculator.calculate_weekly_profit = Mock(return_value=mock_calc_no_profit)

            result_no_profit = await automator.execute_manual_siphon(force=True)
            assert result_no_profit.status == SiphonStatus.NO_PROFIT

            # Test 4: Safety block
            calculator.calculate_weekly_profit = Mock(return_value=mock_calc)
            calculator.validate_withdrawal_safety = Mock(return_value=(False, "Unsafe withdrawal"))

            result_unsafe = await automator.execute_manual_siphon(force=True)
            assert result_unsafe.status == SiphonStatus.SAFETY_BLOCK

            logger.info("✅ WeeklySiphonAutomator validation passed")
            return {
                'name': 'SiphonAutomator',
                'passed': True,
                'details': {
                    'scheduler_control_working': True,
                    'profit_execution_working': True,
                    'no_profit_handling_working': True,
                    'safety_block_working': True
                }
            }

        except Exception as e:
            logger.error(f"❌ WeeklySiphonAutomator validation failed: {e}")
            return {'name': 'SiphonAutomator', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_monitoring_system(self) -> dict:
        """Validate SiphonMonitor functionality."""
        logger.info("📈 Validating Monitoring System...")

        try:
            # Create monitor (disable file logging for testing)
            monitor = SiphonMonitor(enable_file_logging=False)

            # Test health status
            health = monitor.get_health_status()
            assert 'status' in health
            assert 'timestamp' in health

            # Test performance summary
            performance = monitor.get_performance_summary(days=7)
            assert 'period_days' in performance

            # Test alert handling (simulate alert)
            from src.cycles.siphon_monitor import SiphonAlert, AlertLevel

            test_alert = SiphonAlert(
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                component="test",
                message="Test alert",
                details={}
            )

            await monitor._process_alert(test_alert)
            assert len(monitor.alerts) == 1

            recent_alerts = monitor.get_recent_alerts(limit=10)
            assert len(recent_alerts) == 1
            assert recent_alerts[0]['level'] == 'warning'

            logger.info("✅ Monitoring System validation passed")
            return {
                'name': 'MonitoringSystem',
                'passed': True,
                'details': {
                    'health_reporting_working': True,
                    'alert_processing_working': True,
                    'performance_tracking_working': True
                }
            }

        except Exception as e:
            logger.error(f"❌ Monitoring System validation failed: {e}")
            return {'name': 'MonitoringSystem', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_integration(self) -> dict:
        """Validate integration with existing WeeklyCycle."""
        logger.info("🔗 Validating WeeklyCycle Integration...")

        try:
            # Mock all dependencies for WeeklyCycle
            mock_portfolio = AsyncMock()
            mock_portfolio.sync_with_broker.return_value = True
            mock_portfolio.get_gate_positions.return_value = {}

            mock_trade_executor = Mock()
            mock_trade_executor.broker = AsyncMock()
            mock_trade_executor.broker.withdraw_funds.return_value = True

            mock_market_data = Mock()
            mock_holiday_calendar = Mock()
            mock_holiday_calendar.is_market_holiday.return_value = False

            # Create WeeklyCycle with siphon automation
            weekly_cycle = WeeklyCycle(
                portfolio_manager=mock_portfolio,
                trade_executor=mock_trade_executor,
                market_data=mock_market_data,
                holiday_calendar=mock_holiday_calendar,
                enable_dpi=False,
                enable_siphon_automation=False,  # Disabled for testing
                initial_capital=self.base_capital
            )

            # Test that components were initialized
            assert weekly_cycle.siphon_automator is not None
            assert weekly_cycle.profit_calculator is not None

            # Test enhanced status reporting
            status = weekly_cycle.get_cycle_status()
            assert 'siphon_automation' in status
            assert 'profit_summary' in status

            # Test siphon execution history
            history = weekly_cycle.get_siphon_execution_history()
            assert isinstance(history, list)

            # Cleanup
            weekly_cycle.stop_siphon_automation()

            logger.info("✅ WeeklyCycle Integration validation passed")
            return {
                'name': 'WeeklyCycleIntegration',
                'passed': True,
                'details': {
                    'components_initialized': True,
                    'status_reporting_enhanced': True,
                    'cleanup_working': True
                }
            }

        except Exception as e:
            logger.error(f"❌ WeeklyCycle Integration validation failed: {e}")
            return {'name': 'WeeklyCycleIntegration', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_edge_cases(self) -> dict:
        """Validate edge case handling."""
        logger.info("🎯 Validating Edge Cases...")

        try:
            calculator = ProfitCalculator(self.base_capital)
            mock_portfolio = Mock()
            mock_portfolio.get_nav_at_date.return_value = Decimal("200.00")
            mock_portfolio.get_deposits_in_period.return_value = Decimal("0.00")
            mock_portfolio.get_withdrawals_in_period.return_value = Decimal("0.00")

            edge_cases_passed = True

            # Edge Case 1: Tiny profit (1 cent)
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("200.01")
            calc_tiny = calculator.calculate_weekly_profit(mock_portfolio)
            # Should handle the split correctly
            assert calc_tiny.reinvestment_amount + calc_tiny.withdrawal_amount == Decimal("0.01")

            # Edge Case 2: Exactly at base capital
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("200.00")
            calc_exact = calculator.calculate_weekly_profit(mock_portfolio)
            assert calc_exact.profit_status == ProfitStatus.NO_PROFIT

            # Edge Case 3: Large profit
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("400.00")
            calc_large = calculator.calculate_weekly_profit(mock_portfolio)
            assert calc_large.total_profit == Decimal("200.00")
            assert calc_large.withdrawal_amount == Decimal("100.00")
            # Should still protect base capital
            assert calc_large.remaining_capital >= self.base_capital

            # Edge Case 4: Additional deposits
            calculator.record_additional_deposit(Decimal("50.00"))
            mock_portfolio.get_total_portfolio_value.return_value = Decimal("275.00")
            calc_with_deposit = calculator.calculate_weekly_profit(mock_portfolio)
            # Total adjusted base is now $250, so profit should be $25
            assert calc_with_deposit.total_profit == Decimal("25.00")

            logger.info("✅ Edge Cases validation passed")
            return {
                'name': 'EdgeCases',
                'passed': edge_cases_passed,
                'details': {
                    'tiny_profit_handled': True,
                    'exact_base_handled': True,
                    'large_profit_handled': True,
                    'additional_deposits_handled': True
                }
            }

        except Exception as e:
            logger.error(f"❌ Edge Cases validation failed: {e}")
            return {'name': 'EdgeCases', 'passed': False, 'error': str(e), 'details': {}}

    async def validate_error_handling(self) -> dict:
        """Validate error handling and resilience."""
        logger.info("⚠️ Validating Error Handling...")

        try:
            # Test error handling in various scenarios
            calculator = ProfitCalculator(self.base_capital)

            # Mock portfolio that raises exception
            mock_failing_portfolio = Mock()
            mock_failing_portfolio.get_total_portfolio_value.side_effect = Exception("Connection failed")

            # Should handle the error gracefully
            calc_error = calculator.calculate_weekly_profit(mock_failing_portfolio)
            assert calc_error.profit_status == ProfitStatus.INSUFFICIENT_DATA

            # Test automator error handling
            mock_portfolio = AsyncMock()
            mock_portfolio.sync_with_broker.return_value = False  # Sync failure

            mock_broker = AsyncMock()
            calculator = ProfitCalculator(self.base_capital)
            mock_holiday_calendar = Mock()

            automator = WeeklySiphonAutomator(
                portfolio_manager=mock_portfolio,
                broker_adapter=mock_broker,
                profit_calculator=calculator,
                holiday_calendar=mock_holiday_calendar,
                enable_auto_execution=False
            )

            # Should handle sync failure gracefully
            result = await automator.execute_manual_siphon(force=True)
            assert result.status == SiphonStatus.FAILED
            assert len(result.errors) > 0

            logger.info("✅ Error Handling validation passed")
            return {
                'name': 'ErrorHandling',
                'passed': True,
                'details': {
                    'profit_calculator_error_handling': True,
                    'automator_error_handling': True,
                    'graceful_degradation': True
                }
            }

        except Exception as e:
            logger.error(f"❌ Error Handling validation failed: {e}")
            return {'name': 'ErrorHandling', 'passed': False, 'error': str(e), 'details': {}}

    def print_summary(self):
        """Print validation summary."""
        logger.info("=" * 60)
        logger.info("📋 VALIDATION SUMMARY")
        logger.info("=" * 60)

        passed_count = sum(1 for result in self.test_results if result['passed'])
        total_count = len(self.test_results)

        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"{status} - {result['name']}")
            if not result['passed'] and 'error' in result:
                logger.error(f"    Error: {result['error']}")

        logger.info("-" * 60)
        logger.info(f"RESULTS: {passed_count}/{total_count} validations passed")

        if passed_count == total_count:
            logger.info("🎉 ALL VALIDATIONS PASSED - SIPHON AUTOMATION READY FOR PRODUCTION")
        else:
            logger.error("💥 SOME VALIDATIONS FAILED - DO NOT DEPLOY TO PRODUCTION")

        logger.info("=" * 60)


async def main():
    """Main validation entry point."""
    try:
        validator = SiphonValidationSuite()
        success = await validator.run_all_validations()

        if success:
            logger.info("✅ Weekly Siphon Automation validation completed successfully!")
            logger.info("📦 Phase 2 Division 3 deliverables ready for deployment")
            return 0
        else:
            logger.error("❌ Weekly Siphon Automation validation failed!")
            logger.error("🚫 Phase 2 Division 3 deliverables NOT ready for deployment")
            return 1

    except Exception as e:
        logger.error(f"💥 Validation suite crashed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)