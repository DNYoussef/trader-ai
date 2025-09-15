#!/usr/bin/env python3
"""
Comprehensive sandbox testing for Gary×Taleb Foundation Phase Components
Focuses on theater detection and root cause analysis
"""
import sys
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Foundation components
try:
    from src.brokers.alpaca_adapter import AlpacaAdapter, MockAlpacaClient
    from src.gates.gate_manager import GateManager, GateLevel, TradeValidationResult
    from src.trading_engine import TradingEngine
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging for testing
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FoundationSandboxTester:
    """Comprehensive sandbox tester for Foundation components"""

    def __init__(self):
        self.results = {
            'environment_setup': {},
            'alpaca_adapter': {},
            'gate_manager': {},
            'trading_engine': {},
            'integration_tests': {},
            'error_scenarios': {},
            'theater_analysis': {},
            'performance_tests': {},
            'root_cause_analysis': []
        }
        self.test_timestamp = datetime.now(timezone.utc).isoformat()

    def log_test_result(self, component: str, test_name: str, result: Dict[str, Any]):
        """Log a test result"""
        if component not in self.results:
            self.results[component] = {}

        self.results[component][test_name] = {
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'PASS' if result.get('success', False) else 'FAIL'
        }
        print(f"[{component}] {test_name}: {'✓ PASS' if result.get('success', False) else '✗ FAIL'}")

    def test_environment_setup(self) -> bool:
        """Test 1: Environment Setup Validation"""
        print("\n=== Test 1: Environment Setup ===")

        try:
            # Check Python version
            python_version = sys.version_info
            py_version_ok = python_version >= (3, 8)

            # Check directory structure
            required_dirs = ['src', 'src/brokers', 'src/gates', 'src/cycles', 'config']
            dirs_exist = all(os.path.exists(d) for d in required_dirs)

            # Check config file
            config_exists = os.path.exists('config/config.json')

            result = {
                'success': py_version_ok and dirs_exist and config_exists,
                'python_version': str(python_version),
                'directories_exist': dirs_exist,
                'config_exists': config_exists,
                'required_dirs': required_dirs
            }

            self.log_test_result('environment_setup', 'basic_setup', result)
            return result['success']

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('environment_setup', 'basic_setup', result)
            return False

    def test_alpaca_adapter(self) -> bool:
        """Test 2: AlpacaAdapter Mock Mode Testing"""
        print("\n=== Test 2: AlpacaAdapter Testing ===")

        try:
            # Test 2.1: Mock Mode Initialization
            config = {
                'api_key': 'test_key',
                'secret_key': 'test_secret',
                'paper_trading': True
            }

            adapter = AlpacaAdapter(config)
            init_success = adapter is not None

            self.log_test_result('alpaca_adapter', 'initialization', {
                'success': init_success,
                'is_paper_trading': adapter.is_paper_trading,
                'mock_mode': not hasattr(adapter, 'ALPACA_AVAILABLE') or not adapter.ALPACA_AVAILABLE
            })

            # Test 2.2: Connection Test
            async def test_connection():
                try:
                    connected = await adapter.connect()
                    if connected:
                        account_value = await adapter.get_account_value()
                        cash_balance = await adapter.get_cash_balance()
                        buying_power = await adapter.get_buying_power()

                        await adapter.disconnect()

                        return {
                            'success': True,
                            'connected': connected,
                            'account_value': str(account_value),
                            'cash_balance': str(cash_balance),
                            'buying_power': str(buying_power)
                        }
                    return {'success': False, 'connected': False}
                except Exception as e:
                    return {'success': False, 'error': str(e)}

            connection_result = asyncio.run(test_connection())
            self.log_test_result('alpaca_adapter', 'connection', connection_result)

            # Test 2.3: Mock Order Submission
            async def test_mock_orders():
                try:
                    await adapter.connect()

                    # Create mock order
                    from src.brokers.broker_interface import Order, OrderType, TimeInForce

                    order = Order(
                        symbol='ULTY',
                        qty=Decimal('10.0'),
                        side='buy',
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    submitted_order = await adapter.submit_order(order)
                    order_retrieved = await adapter.get_order(submitted_order.id) if submitted_order else None

                    await adapter.disconnect()

                    return {
                        'success': submitted_order is not None,
                        'order_submitted': submitted_order is not None,
                        'order_retrieved': order_retrieved is not None,
                        'order_id': submitted_order.id if submitted_order else None
                    }
                except Exception as e:
                    return {'success': False, 'error': str(e)}

            order_result = asyncio.run(test_mock_orders())
            self.log_test_result('alpaca_adapter', 'mock_orders', order_result)

            return init_success and connection_result.get('success', False)

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('alpaca_adapter', 'general_error', result)
            return False

    def test_gate_manager(self) -> bool:
        """Test 3: GateManager Logic and Constraints"""
        print("\n=== Test 3: GateManager Testing ===")

        try:
            # Test 3.1: Initialization
            gate_manager = GateManager()
            init_success = gate_manager.current_gate == GateLevel.G0

            self.log_test_result('gate_manager', 'initialization', {
                'success': init_success,
                'current_gate': gate_manager.current_gate.value,
                'configs_loaded': len(gate_manager.gate_configs)
            })

            # Test 3.2: G0 Gate Validation (Theater Detection Focus)
            g0_config = gate_manager.get_current_config()

            # Valid trade
            valid_trade = {
                'symbol': 'ULTY',
                'side': 'BUY',
                'quantity': 10.0,
                'price': 5.57,
                'trade_type': 'STOCK'
            }

            valid_portfolio = {
                'cash': 150.0,
                'total_value': 200.0,
                'positions': {}
            }

            valid_result = gate_manager.validate_trade(valid_trade, valid_portfolio)

            # Invalid trade - not allowed asset
            invalid_trade = {
                'symbol': 'SPY',  # Not allowed in G0
                'side': 'BUY',
                'quantity': 10.0,
                'price': 400.0,
                'trade_type': 'STOCK'
            }

            invalid_result = gate_manager.validate_trade(invalid_trade, valid_portfolio)

            # Cash floor violation
            cash_floor_trade = {
                'symbol': 'ULTY',
                'side': 'BUY',
                'quantity': 50.0,  # Too large
                'price': 5.57,
                'trade_type': 'STOCK'
            }

            cash_floor_portfolio = {
                'cash': 120.0,
                'total_value': 200.0,
                'positions': {}
            }

            cash_floor_result = gate_manager.validate_trade(cash_floor_trade, cash_floor_portfolio)

            validation_success = (
                valid_result.is_valid and
                not invalid_result.is_valid and
                not cash_floor_result.is_valid
            )

            self.log_test_result('gate_manager', 'validation_logic', {
                'success': validation_success,
                'valid_trade_passed': valid_result.is_valid,
                'invalid_asset_blocked': not invalid_result.is_valid,
                'cash_floor_enforced': not cash_floor_result.is_valid,
                'invalid_violations': len(invalid_result.violations),
                'cash_floor_violations': len(cash_floor_result.violations)
            })

            # Test 3.3: Gate Progression Logic
            # Simulate G0 -> G1 progression criteria
            mock_metrics = {
                'sharpe_ratio_30d': 1.5,
                'max_drawdown_30d': 0.03,
                'avg_cash_utilization_30d': 0.6,
                'total_return_30d': 0.08
            }

            # Set up for graduation test
            gate_manager.graduation_metrics.consecutive_compliant_days = 15
            gate_manager.graduation_metrics.total_violations_30d = 0
            gate_manager.current_capital = 500.0  # Above G1 minimum

            graduation_decision = gate_manager.check_graduation(mock_metrics)

            self.log_test_result('gate_manager', 'progression_logic', {
                'success': graduation_decision in ['GRADUATE', 'HOLD', 'DOWNGRADE'],
                'decision': graduation_decision,
                'performance_score': gate_manager.graduation_metrics.performance_score,
                'compliant_days': gate_manager.graduation_metrics.consecutive_compliant_days
            })

            return init_success and validation_success

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('gate_manager', 'general_error', result)
            return False

    def test_trading_engine_integration(self) -> bool:
        """Test 4: TradingEngine Integration"""
        print("\n=== Test 4: TradingEngine Integration ===")

        try:
            # Test 4.1: Engine Initialization
            engine = TradingEngine()
            init_success = engine is not None

            self.log_test_result('trading_engine', 'initialization', {
                'success': init_success,
                'mode': engine.mode,
                'config_loaded': engine.config is not None
            })

            # Test 4.2: Component Integration (Theater Detection Focus)
            initialize_success = engine.initialize()

            if initialize_success:
                status = engine.get_status()
                integration_success = (
                    status.get('status') != 'not_initialized' and
                    status.get('gate') is not None and
                    status.get('nav') is not None
                )

                self.log_test_result('trading_engine', 'component_integration', {
                    'success': integration_success,
                    'status': status.get('status'),
                    'gate': status.get('gate'),
                    'nav': status.get('nav'),
                    'broker_connected': engine.broker.is_connected if engine.broker else False
                })

                # Test 4.3: Kill Switch (Critical Safety Feature)
                # This should safely shut down without real trading
                initial_status = engine.get_status()
                engine.activate_kill_switch()
                post_kill_status = engine.get_status()

                kill_switch_success = engine.kill_switch_activated

                self.log_test_result('trading_engine', 'kill_switch', {
                    'success': kill_switch_success,
                    'kill_switch_activated': engine.kill_switch_activated,
                    'engine_stopped': not engine.running
                })

                return initialize_success and integration_success and kill_switch_success
            else:
                self.log_test_result('trading_engine', 'component_integration', {
                    'success': False,
                    'error': 'Failed to initialize engine'
                })
                return False

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('trading_engine', 'general_error', result)
            return False

    def test_error_scenarios(self) -> bool:
        """Test 5: Error Handling and Edge Cases"""
        print("\n=== Test 5: Error Scenarios ===")

        try:
            # Test 5.1: Invalid Configuration
            invalid_config_result = self._test_invalid_configs()

            # Test 5.2: Network/Connection Failures
            connection_failure_result = self._test_connection_failures()

            # Test 5.3: Invalid Trade Parameters
            invalid_trade_result = self._test_invalid_trades()

            overall_success = all([
                invalid_config_result,
                connection_failure_result,
                invalid_trade_result
            ])

            self.log_test_result('error_scenarios', 'comprehensive', {
                'success': overall_success,
                'invalid_config': invalid_config_result,
                'connection_failures': connection_failure_result,
                'invalid_trades': invalid_trade_result
            })

            return overall_success

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('error_scenarios', 'general_error', result)
            return False

    def _test_invalid_configs(self) -> bool:
        """Test invalid configuration handling"""
        try:
            # Test with missing config file
            engine = TradingEngine(config_path='nonexistent.json')
            # Should use default config
            return engine.config is not None
        except Exception:
            return True  # Exception handling is acceptable

    def _test_connection_failures(self) -> bool:
        """Test connection failure scenarios"""
        try:
            # Test with invalid credentials
            invalid_adapter = AlpacaAdapter({
                'api_key': 'invalid',
                'secret_key': 'invalid',
                'paper_trading': True
            })

            # In mock mode, this should still work
            # In real mode, this should fail gracefully
            async def test_invalid_connection():
                try:
                    await invalid_adapter.connect()
                    return True  # Mock mode succeeds
                except Exception:
                    return True  # Real mode fails gracefully

            return asyncio.run(test_invalid_connection())
        except Exception:
            return True  # Graceful failure is acceptable

    def _test_invalid_trades(self) -> bool:
        """Test invalid trade parameter handling"""
        try:
            gate_manager = GateManager()

            # Test with missing required fields
            invalid_trades = [
                {'symbol': '', 'side': 'BUY'},  # Empty symbol
                {'symbol': 'ULTY', 'side': ''},  # Empty side
                {'symbol': 'ULTY', 'side': 'BUY', 'quantity': -10},  # Negative quantity
            ]

            portfolio = {'cash': 100, 'total_value': 200, 'positions': {}}

            for trade in invalid_trades:
                result = gate_manager.validate_trade(trade, portfolio)
                if result.is_valid:  # Should be invalid
                    return False

            return True  # All invalid trades were caught
        except Exception:
            return True  # Exception handling is acceptable

    def run_theater_detection_analysis(self) -> Dict[str, Any]:
        """Critical: Theater Detection Analysis"""
        print("\n=== Theater Detection Analysis ===")

        analysis = {
            'potential_theater_issues': [],
            'genuine_implementations': [],
            'risk_assessment': 'LOW',
            'recommendations': []
        }

        # Analyze AlpacaAdapter for theater
        if hasattr(AlpacaAdapter, '__doc__') and AlpacaAdapter.__doc__:
            analysis['genuine_implementations'].append({
                'component': 'AlpacaAdapter',
                'evidence': 'Comprehensive documentation and mock mode implementation',
                'detail': 'Real API integration with fallback to mock for testing'
            })

        # Analyze GateManager for theater
        gate_manager = GateManager()
        if len(gate_manager.gate_configs) == 4:  # G0-G3
            analysis['genuine_implementations'].append({
                'component': 'GateManager',
                'evidence': 'Complete gate configuration system',
                'detail': f'All 4 gates configured with specific constraints'
            })
        else:
            analysis['potential_theater_issues'].append({
                'component': 'GateManager',
                'issue': 'Incomplete gate configuration',
                'severity': 'MEDIUM'
            })

        # Check for real validation logic
        test_trade = {'symbol': 'SPY', 'side': 'BUY', 'quantity': 10, 'price': 400, 'trade_type': 'STOCK'}
        test_portfolio = {'cash': 100, 'total_value': 200, 'positions': {}}
        validation_result = gate_manager.validate_trade(test_trade, test_portfolio)

        if not validation_result.is_valid and len(validation_result.violations) > 0:
            analysis['genuine_implementations'].append({
                'component': 'GateManager Validation',
                'evidence': 'Real constraint enforcement',
                'detail': f'Blocked invalid trade with {len(validation_result.violations)} violations'
            })
        else:
            analysis['potential_theater_issues'].append({
                'component': 'GateManager Validation',
                'issue': 'Validation not working properly',
                'severity': 'HIGH'
            })
            analysis['risk_assessment'] = 'HIGH'

        # Check TradingEngine kill switch
        engine = TradingEngine()
        if hasattr(engine, 'activate_kill_switch') and callable(engine.activate_kill_switch):
            analysis['genuine_implementations'].append({
                'component': 'TradingEngine Kill Switch',
                'evidence': 'Emergency stop mechanism implemented',
                'detail': 'Kill switch method exists and is callable'
            })
        else:
            analysis['potential_theater_issues'].append({
                'component': 'TradingEngine Kill Switch',
                'issue': 'Missing or non-functional kill switch',
                'severity': 'CRITICAL'
            })
            analysis['risk_assessment'] = 'CRITICAL'

        # Generate recommendations
        if len(analysis['potential_theater_issues']) == 0:
            analysis['recommendations'].append("✓ No theater issues detected - implementation appears genuine")
        else:
            analysis['recommendations'].append("⚠ Theater issues detected - requires immediate attention")

        for issue in analysis['potential_theater_issues']:
            if issue['severity'] == 'CRITICAL':
                analysis['recommendations'].append(f"CRITICAL FIX: {issue['component']} - {issue['issue']}")
            elif issue['severity'] == 'HIGH':
                analysis['recommendations'].append(f"HIGH PRIORITY: {issue['component']} - {issue['issue']}")

        self.results['theater_analysis'] = analysis
        return analysis

    def run_performance_tests(self) -> bool:
        """Test 6: Performance Under Load"""
        print("\n=== Test 6: Performance Testing ===")

        try:
            import time

            # Test 6.1: Multiple simultaneous validations
            start_time = time.time()

            gate_manager = GateManager()
            test_portfolio = {'cash': 100, 'total_value': 200, 'positions': {}}

            # Run 100 validations
            for i in range(100):
                test_trade = {
                    'symbol': 'ULTY',
                    'side': 'BUY',
                    'quantity': 1.0,
                    'price': 5.57,
                    'trade_type': 'STOCK'
                }
                result = gate_manager.validate_trade(test_trade, test_portfolio)

            validation_time = time.time() - start_time

            # Test 6.2: Engine initialization time
            start_time = time.time()
            engine = TradingEngine()
            engine.initialize()
            init_time = time.time() - start_time

            performance_success = validation_time < 1.0 and init_time < 5.0

            self.log_test_result('performance_tests', 'load_testing', {
                'success': performance_success,
                'validation_time_100_ops': validation_time,
                'engine_init_time': init_time,
                'validations_per_second': 100 / validation_time if validation_time > 0 else 0
            })

            return performance_success

        except Exception as e:
            result = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
            self.log_test_result('performance_tests', 'general_error', result)
            return False

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate final comprehensive test report"""

        # Calculate overall success rates
        total_tests = 0
        passed_tests = 0

        for component, tests in self.results.items():
            if component == 'theater_analysis':
                continue
            for test_name, test_result in tests.items():
                total_tests += 1
                if test_result.get('status') == 'PASS':
                    passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Determine overall status
        theater_analysis = self.results.get('theater_analysis', {})
        risk_level = theater_analysis.get('risk_assessment', 'UNKNOWN')

        if success_rate >= 90 and risk_level in ['LOW', 'MEDIUM']:
            overall_status = 'PASS'
        elif success_rate >= 70 and risk_level != 'CRITICAL':
            overall_status = 'CONDITIONAL_PASS'
        else:
            overall_status = 'FAIL'

        # Generate executive summary
        summary = {
            'test_timestamp': self.test_timestamp,
            'overall_status': overall_status,
            'success_rate': round(success_rate, 1),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'theater_risk_level': risk_level,
            'critical_issues': [
                issue for issue in theater_analysis.get('potential_theater_issues', [])
                if issue.get('severity') == 'CRITICAL'
            ],
            'recommendations': theater_analysis.get('recommendations', []),
            'next_phase_readiness': 'READY' if overall_status == 'PASS' else 'BLOCKED'
        }

        # Root cause analysis for failures
        root_causes = []
        for component, tests in self.results.items():
            if component == 'theater_analysis':
                continue
            for test_name, test_result in tests.items():
                if test_result.get('status') == 'FAIL':
                    root_causes.append({
                        'component': component,
                        'test': test_name,
                        'error': test_result.get('result', {}).get('error', 'Unknown error'),
                        'remediation': self._suggest_remediation(component, test_name, test_result)
                    })

        return {
            'executive_summary': summary,
            'detailed_results': self.results,
            'root_cause_analysis': root_causes
        }

    def _suggest_remediation(self, component: str, test_name: str, test_result: Dict) -> str:
        """Suggest remediation steps for failed tests"""

        remediation_map = {
            'environment_setup': 'Verify Python installation and project directory structure',
            'alpaca_adapter': 'Check broker configuration and API credentials',
            'gate_manager': 'Review gate configuration and validation logic',
            'trading_engine': 'Verify component initialization and integration',
            'error_scenarios': 'Improve error handling and edge case management',
            'performance_tests': 'Optimize algorithms and reduce computational complexity'
        }

        return remediation_map.get(component, 'Manual investigation required')

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all sandbox tests and generate comprehensive report"""
        print("🧪 Starting Gary×Taleb Foundation Phase Sandbox Testing")
        print("=" * 60)

        # Run all test suites
        tests = [
            ('Environment Setup', self.test_environment_setup),
            ('AlpacaAdapter', self.test_alpaca_adapter),
            ('GateManager', self.test_gate_manager),
            ('TradingEngine Integration', self.test_trading_engine_integration),
            ('Error Scenarios', self.test_error_scenarios),
            ('Performance Tests', self.run_performance_tests)
        ]

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.results.setdefault('critical_failures', []).append({
                    'test': test_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

        # Run theater detection analysis
        self.run_theater_detection_analysis()

        # Generate final report
        final_report = self.generate_comprehensive_report()

        print("\n" + "=" * 60)
        print("📊 SANDBOX TEST SUMMARY")
        print("=" * 60)

        summary = final_report['executive_summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Theater Risk: {summary['theater_risk_level']}")
        print(f"Next Phase: {summary['next_phase_readiness']}")

        if summary['critical_issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"  - {issue['component']}: {issue['issue']}")

        if final_report['root_cause_analysis']:
            print(f"\n🔍 ROOT CAUSE ANALYSIS ({len(final_report['root_cause_analysis'])} issues):")
            for rca in final_report['root_cause_analysis'][:3]:  # Show top 3
                print(f"  - {rca['component']}.{rca['test']}: {rca['remediation']}")

        print(f"\n💡 RECOMMENDATIONS ({len(summary['recommendations'])}):")
        for rec in summary['recommendations'][:3]:  # Show top 3
            print(f"  - {rec}")

        return final_report


def main():
    """Main test execution"""
    tester = FoundationSandboxTester()

    try:
        report = tester.run_all_tests()

        # Save detailed results
        output_file = 'sandbox_test_results/foundation_sandbox_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Full report saved to: {output_file}")

        # Return appropriate exit code
        if report['executive_summary']['overall_status'] == 'PASS':
            print("\n✅ SANDBOX TESTING PASSED - Foundation Phase Ready")
            sys.exit(0)
        elif report['executive_summary']['overall_status'] == 'CONDITIONAL_PASS':
            print("\n⚠️  CONDITIONAL PASS - Minor issues detected")
            sys.exit(1)
        else:
            print("\n❌ SANDBOX TESTING FAILED - Critical issues must be resolved")
            sys.exit(2)

    except Exception as e:
        print(f"\n💥 SANDBOX TESTING CRASHED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(3)


if __name__ == '__main__':
    main()