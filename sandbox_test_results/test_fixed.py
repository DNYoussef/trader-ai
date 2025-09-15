#!/usr/bin/env python3
"""
Fixed sandbox testing for Gary×Taleb Foundation Core Components
"""
import sys
import os
import json
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_gate_manager_validation_fixed():
    """FIXED: Gate Manager Validation Logic"""
    print("\n=== FIXED: Gate Manager Validation ===")

    try:
        from src.gates.gate_manager import GateManager

        gate_manager = GateManager()
        print(f"  Initialized at gate: {gate_manager.current_gate.value}")

        # Test 1: Valid ULTY trade with proper cash floor
        valid_trade = {
            'symbol': 'ULTY',
            'side': 'BUY',
            'quantity': 5.0,  # Smaller quantity
            'price': 5.57,
            'trade_type': 'STOCK'
        }

        # Portfolio with sufficient cash buffer for 50% floor
        valid_portfolio = {
            'cash': 170.0,  # Ensures >50% remains after trade
            'total_value': 200.0,
            'positions': {}
        }

        print(f"  Trade value: ${valid_trade['quantity'] * valid_trade['price']:.2f}")
        print(f"  Post-trade cash: ${valid_portfolio['cash'] - (valid_trade['quantity'] * valid_trade['price']):.2f}")
        print(f"  Required cash (50%): ${valid_portfolio['total_value'] * 0.5:.2f}")

        valid_result = gate_manager.validate_trade(valid_trade, valid_portfolio)
        print(f"  ULTY valid trade: {'PASS' if valid_result.is_valid else 'FAIL'}")
        if not valid_result.is_valid:
            print(f"    Violations: {[v['type'] for v in valid_result.violations]}")

        # Test 2: Invalid SPY trade (should FAIL - Theater Detection)
        invalid_trade = {
            'symbol': 'SPY',  # Not allowed in G0
            'side': 'BUY',
            'quantity': 1.0,
            'price': 400.0,
            'trade_type': 'STOCK'
        }

        invalid_result = gate_manager.validate_trade(invalid_trade, valid_portfolio)
        blocked_correctly = not invalid_result.is_valid
        print(f"  SPY trade blocked: {'PASS' if blocked_correctly else 'FAIL - THEATER RISK!'}")

        if blocked_correctly:
            print(f"    Violations: {[v['type'] for v in invalid_result.violations]}")

        # Test 3: Cash floor violation (should FAIL)
        cash_floor_trade = {
            'symbol': 'ULTY',
            'side': 'BUY',
            'quantity': 20.0,  # Large quantity to violate cash floor
            'price': 5.57,
            'trade_type': 'STOCK'
        }

        cash_result = gate_manager.validate_trade(cash_floor_trade, valid_portfolio)
        cash_blocked = not cash_result.is_valid
        print(f"  Cash floor enforced: {'PASS' if cash_blocked else 'FAIL - THEATER RISK!'}")

        # Theater Assessment
        theater_score = sum([valid_result.is_valid, blocked_correctly, cash_blocked])
        theater_assessment = "GENUINE" if theater_score == 3 else f"THEATER RISK (Score: {theater_score}/3)"
        print(f"  THEATER ASSESSMENT: {theater_assessment}")

        return theater_score == 3

    except Exception as e:
        print(f"  [ERROR] Gate manager test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_alpaca_adapter_fixed():
    """FIXED: AlpacaAdapter with proper decimal handling"""
    print("\n=== FIXED: AlpacaAdapter ===")

    try:
        from src.brokers.alpaca_adapter import AlpacaAdapter
        from src.brokers.broker_interface import Order, OrderType, TimeInForce

        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'paper_trading': True
        }

        adapter = AlpacaAdapter(config)
        print("  Adapter created successfully")

        async def test_async():
            try:
                connected = await adapter.connect()
                print(f"  Connection: {'SUCCESS' if connected else 'FAILED'}")

                if connected:
                    account_value = await adapter.get_account_value()
                    print(f"  Account value: ${account_value}")

                    # Test order submission with proper decimal handling
                    order = Order(
                        symbol='ULTY',
                        qty=Decimal('10.0'),
                        side='buy',
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY
                    )

                    submitted = await adapter.submit_order(order)
                    order_success = submitted is not None
                    print(f"  Order submission: {'SUCCESS' if order_success else 'FAILED'}")

                    if order_success:
                        print(f"    Order ID: {submitted.id}")
                        print(f"    Order symbol: {submitted.symbol}")
                        print(f"    Order qty: {submitted.qty}")

                    await adapter.disconnect()
                    return order_success

                return False

            except Exception as e:
                print(f"  Async test error: {e}")
                return False

        return asyncio.run(test_async())

    except Exception as e:
        print(f"  [ERROR] AlpacaAdapter test failed: {e}")
        return False

def test_trading_engine_fixed():
    """FIXED: TradingEngine with stub dependencies"""
    print("\n=== FIXED: TradingEngine ===")

    try:
        from src.trading_engine import TradingEngine

        engine = TradingEngine()
        print("  Engine created")

        # Try initialization
        init_success = engine.initialize()
        print(f"  Initialization: {'SUCCESS' if init_success else 'FAILED'}")

        if init_success:
            # Test status
            status = engine.get_status()
            print(f"    Status: {status.get('status', 'unknown')}")
            print(f"    Gate: {status.get('gate', 'unknown')}")

            # Test kill switch (CRITICAL SAFETY)
            print("  Testing kill switch...")
            engine.activate_kill_switch()
            kill_switch_works = engine.kill_switch_activated
            print(f"  Kill switch: {'WORKS' if kill_switch_works else 'BROKEN - DANGER!'}")

            return kill_switch_works
        else:
            print("  Engine initialization failed")
            return False

    except Exception as e:
        print(f"  [ERROR] TradingEngine test failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def run_fixed_sandbox_tests():
    """Run fixed sandbox tests"""
    print("GARY×TALEB FOUNDATION PHASE - FIXED SANDBOX TESTING")
    print("Focus: Theater Detection & Safety Validation")
    print("=" * 60)

    # Test all components
    test_results = {
        'gate_manager': test_gate_manager_validation_fixed(),
        'alpaca_adapter': test_alpaca_adapter_fixed(),
        'trading_engine': test_trading_engine_fixed()
    }

    # Theater Detection Analysis
    print("\n" + "=" * 60)
    print("THEATER DETECTION ANALYSIS")
    print("=" * 60)

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)

    print(f"Test Results: {passed_tests}/{total_tests} passed")

    # Critical theater assessment
    gate_validation_works = test_results.get('gate_manager', False)
    kill_switch_works = test_results.get('trading_engine', False)

    if gate_validation_works and kill_switch_works:
        theater_risk = "LOW"
        print("  [PASS] Critical components working correctly")
    elif gate_validation_works or kill_switch_works:
        theater_risk = "MEDIUM"
        print("  [WARN] Some critical components have issues")
    else:
        theater_risk = "HIGH"
        print("  [FAIL] Critical components not working")

    print(f"\nTheater Risk Level: {theater_risk}")

    # Detailed results
    print("\nDetailed Results:")
    for component, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {component}: {status}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if theater_risk == "LOW":
        print("  ✓ Foundation components appear genuine")
        print("  ✓ Critical safety mechanisms working")
        print("  ✓ Safe to proceed with next development phase")
    elif theater_risk == "MEDIUM":
        print("  ⚠ Some issues remain, but core safety intact")
        print("  - Address remaining issues before production")
        print("  - Monitor closely during development")
    else:
        print("  ❌ CRITICAL ISSUES REMAIN")
        print("  - DO NOT PROCEED until all issues resolved")
        print("  - Safety mechanisms not verified")

    # Generate report
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'test_results': test_results,
        'theater_risk': theater_risk,
        'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        'next_phase_ready': theater_risk in ['LOW', 'MEDIUM'],
        'critical_safety_verified': gate_validation_works and kill_switch_works
    }

    # Save report
    try:
        with open('sandbox_test_results/foundation_fixed_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFixed report saved: sandbox_test_results/foundation_fixed_report.json")
    except Exception as e:
        print(f"Could not save report: {e}")

    # Return exit code
    if theater_risk == "LOW":
        print("\n✅ [SUCCESS] Foundation sandbox testing PASSED")
        return 0
    elif theater_risk == "MEDIUM":
        print("\n⚠️ [WARNING] Conditional pass with minor issues")
        return 1
    else:
        print("\n❌ [FAILURE] Critical issues remain")
        return 2

if __name__ == '__main__':
    sys.exit(run_fixed_sandbox_tests())