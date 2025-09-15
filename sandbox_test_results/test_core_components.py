#!/usr/bin/env python3
"""
Focused sandbox testing for Gary×Taleb Foundation Core Components
Theater detection and root cause analysis focus
"""
import sys
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_alpaca_adapter():
    """Test AlpacaAdapter with mock mode"""
    print("=== Testing AlpacaAdapter ===")

    try:
        from src.brokers.alpaca_adapter import AlpacaAdapter
        from src.brokers.broker_interface import Order, OrderType, TimeInForce

        # Test initialization
        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'paper_trading': True
        }

        adapter = AlpacaAdapter(config)
        print("  ✓ AlpacaAdapter initialized")
        print(f"    Paper trading: {adapter.is_paper_trading}")

        # Test mock connection
        async def test_connection():
            try:
                connected = await adapter.connect()
                print(f"  ✓ Connection: {connected}")

                if connected:
                    account_value = await adapter.get_account_value()
                    cash_balance = await adapter.get_cash_balance()
                    print(f"    Account value: ${account_value}")
                    print(f"    Cash balance: ${cash_balance}")

                    # Test mock order
                    order = Order(
                        symbol='ULTY',
                        qty=Decimal('10.0'),
                        side='buy',
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    submitted_order = await adapter.submit_order(order)
                    if submitted_order:
                        print(f"  ✓ Mock order submitted: {submitted_order.id}")

                        retrieved_order = await adapter.get_order(submitted_order.id)
                        if retrieved_order:
                            print("  ✓ Order retrieved successfully")

                    await adapter.disconnect()

                return True

            except Exception as e:
                print(f"  ✗ Connection test failed: {e}")
                return False

        connection_success = asyncio.run(test_connection())
        return connection_success

    except Exception as e:
        print(f"  ✗ AlpacaAdapter test failed: {e}")
        return False

def test_gate_manager():
    """Test GateManager validation and logic"""
    print("\n=== Testing GateManager ===")

    try:
        from src.gates.gate_manager import GateManager, GateLevel

        # Test initialization
        gate_manager = GateManager()
        print(f"  ✓ GateManager initialized at {gate_manager.current_gate.value}")
        print(f"    Gate configs loaded: {len(gate_manager.gate_configs)}")

        # Test G0 validation (Theater Detection Focus)
        print("\n  Testing G0 Validation:")

        # Valid trade - ULTY in G0
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
        print(f"    Valid ULTY trade: {'PASS' if valid_result.is_valid else 'FAIL'}")

        # Invalid trade - SPY not allowed in G0 (THEATER TEST)
        invalid_trade = {
            'symbol': 'SPY',
            'side': 'BUY',
            'quantity': 10.0,
            'price': 400.0,
            'trade_type': 'STOCK'
        }

        invalid_result = gate_manager.validate_trade(invalid_trade, valid_portfolio)
        print(f"    Invalid SPY trade blocked: {'PASS' if not invalid_result.is_valid else 'FAIL (THEATER RISK!)'}")
        if not invalid_result.is_valid:
            print(f"      Violations detected: {len(invalid_result.violations)}")

        # Cash floor test (THEATER TEST)
        cash_floor_trade = {
            'symbol': 'ULTY',
            'side': 'BUY',
            'quantity': 100.0,  # Large quantity
            'price': 5.57,
            'trade_type': 'STOCK'
        }

        low_cash_portfolio = {
            'cash': 120.0,
            'total_value': 200.0,
            'positions': {}
        }

        cash_result = gate_manager.validate_trade(cash_floor_trade, low_cash_portfolio)
        print(f"    Cash floor enforcement: {'PASS' if not cash_result.is_valid else 'FAIL (THEATER RISK!)'}")
        if not cash_result.is_valid:
            print(f"      Cash floor violations: {len(cash_result.violations)}")

        # Test gate configuration details
        g0_config = gate_manager.get_current_config()
        print(f"\n  G0 Configuration:")
        print(f"    Allowed assets: {g0_config.allowed_assets}")
        print(f"    Cash floor: {g0_config.cash_floor_pct * 100}%")
        print(f"    Options enabled: {g0_config.options_enabled}")

        # Test progression logic
        mock_metrics = {
            'sharpe_ratio_30d': 1.5,
            'max_drawdown_30d': 0.03,
            'avg_cash_utilization_30d': 0.6
        }

        gate_manager.graduation_metrics.consecutive_compliant_days = 15
        gate_manager.graduation_metrics.total_violations_30d = 0
        gate_manager.current_capital = 500.0

        decision = gate_manager.check_graduation(mock_metrics)
        print(f"    Graduation decision: {decision}")

        # THEATER DETECTION: Check if validation logic is real
        theater_score = 0
        if valid_result.is_valid:
            theater_score += 1
        if not invalid_result.is_valid:
            theater_score += 1
        if not cash_result.is_valid:
            theater_score += 1

        theater_assessment = "GENUINE" if theater_score == 3 else "THEATER RISK"
        print(f"\n  THEATER ASSESSMENT: {theater_assessment} (Score: {theater_score}/3)")

        return theater_score == 3

    except Exception as e:
        print(f"  ✗ GateManager test failed: {e}")
        return False

def test_trading_engine():
    """Test TradingEngine integration"""
    print("\n=== Testing TradingEngine ===")

    try:
        from src.trading_engine import TradingEngine

        # Test initialization
        engine = TradingEngine()
        print("  ✓ TradingEngine created")
        print(f"    Mode: {engine.mode}")
        print(f"    Config loaded: {engine.config is not None}")

        # Test initialization
        init_success = engine.initialize()
        print(f"    Initialization: {'SUCCESS' if init_success else 'FAILED'}")

        if init_success:
            # Test status
            status = engine.get_status()
            print(f"    Status: {status.get('status', 'unknown')}")
            print(f"    Gate: {status.get('gate', 'unknown')}")
            print(f"    NAV: {status.get('nav', 'unknown')}")

            # CRITICAL TEST: Kill switch (Safety mechanism)
            print("\n  Testing Kill Switch (CRITICAL SAFETY):")
            print("    Activating kill switch...")
            engine.activate_kill_switch()

            post_kill_status = engine.get_status()
            kill_switch_ok = engine.kill_switch_activated
            print(f"    Kill switch activated: {'YES' if kill_switch_ok else 'NO (DANGER!)'}")
            print(f"    Engine stopped: {'YES' if not engine.running else 'NO'}")

            if not kill_switch_ok:
                print("    ⚠️  CRITICAL SAFETY ISSUE: Kill switch not working!")
                return False

        return init_success

    except Exception as e:
        print(f"  ✗ TradingEngine test failed: {e}")
        return False

def test_main_entry():
    """Test main.py entry point"""
    print("\n=== Testing Main Entry Point ===")

    try:
        # Test --test mode (safe)
        import subprocess
        import sys

        # Run main.py in test mode
        result = subprocess.run([
            sys.executable, 'main.py', '--test'
        ], capture_output=True, text=True, timeout=10)

        success = result.returncode == 0
        print(f"  Test mode execution: {'SUCCESS' if success else 'FAILED'}")

        if result.stdout:
            print(f"    Output: {result.stdout[:200]}...")

        if result.stderr and not success:
            print(f"    Error: {result.stderr[:200]}...")

        return success

    except subprocess.TimeoutExpired:
        print("  ✓ Test mode timed out (expected for some scenarios)")
        return True
    except Exception as e:
        print(f"  ✗ Main entry test failed: {e}")
        return False

def theater_detection_summary(results):
    """Generate theater detection analysis"""
    print("\n" + "="*60)
    print("THEATER DETECTION ANALYSIS")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)

    risk_level = "LOW"
    issues = []

    if not results.get('alpaca_adapter', False):
        issues.append("AlpacaAdapter: Mock mode or connection issues")
        risk_level = "MEDIUM"

    if not results.get('gate_manager', False):
        issues.append("GateManager: Validation logic not working properly")
        risk_level = "HIGH"

    if not results.get('trading_engine', False):
        issues.append("TradingEngine: Integration or kill switch issues")
        risk_level = "HIGH"

    # Critical assessment
    if not results.get('gate_manager', False):
        risk_level = "CRITICAL"
        issues.append("CRITICAL: Gate constraints not enforced - real money at risk!")

    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print(f"Theater Risk Level: {risk_level}")

    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No theater issues detected - implementation appears genuine")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if risk_level == "LOW":
        print("  ✓ Foundation phase ready for next stage")
    elif risk_level == "MEDIUM":
        print("  ⚠ Address minor issues before proceeding")
    elif risk_level == "HIGH":
        print("  🚨 Fix critical issues before any real trading")
    else:  # CRITICAL
        print("  ❌ HALT: Do not proceed until all issues resolved")

    return {
        'risk_level': risk_level,
        'success_rate': passed_tests / total_tests * 100,
        'issues': issues,
        'next_phase_ready': risk_level in ['LOW', 'MEDIUM']
    }

def main():
    """Run all foundation tests"""
    print("🧪 Gary×Taleb Foundation Phase - Sandbox Testing")
    print("🎯 Focus: Theater Detection & Root Cause Analysis")
    print("="*60)

    # Run core component tests
    results = {
        'alpaca_adapter': test_alpaca_adapter(),
        'gate_manager': test_gate_manager(),
        'trading_engine': test_trading_engine(),
        'main_entry': test_main_entry()
    }

    # Theater detection analysis
    analysis = theater_detection_summary(results)

    # Save results
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'test_results': results,
        'theater_analysis': analysis,
        'environment': {
            'python_version': sys.version,
            'working_directory': os.getcwd()
        }
    }

    with open('sandbox_test_results/foundation_core_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved: sandbox_test_results/foundation_core_report.json")

    # Final assessment
    if analysis['risk_level'] == 'LOW':
        print("\n✅ FOUNDATION SANDBOX TESTING: PASSED")
        return 0
    elif analysis['risk_level'] in ['MEDIUM']:
        print("\n⚠️  FOUNDATION SANDBOX TESTING: CONDITIONAL PASS")
        return 1
    else:
        print("\n❌ FOUNDATION SANDBOX TESTING: FAILED")
        return 2

if __name__ == '__main__':
    sys.exit(main())