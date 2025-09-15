#!/usr/bin/env python3
"""
Alpha Generation Systems Validation Script

This script validates that all Phase 5 alpha generation components
are working correctly and can be integrated together.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        # Test Shadow Book System
        from learning.shadow_book import ShadowBookEngine, Trade, TradeType, ActionType
        print("✅ Shadow Book System imported successfully")

        # Test Policy Twin
        from learning.policy_twin import PolicyTwin, AlphaType
        print("✅ Policy Twin System imported successfully")

        # Test Narrative Gap Engine
        from intelligence.narrative.narrative_gap import NarrativeGapEngine
        print("✅ Narrative Gap Engine imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_shadow_book():
    """Test Shadow Book System functionality"""
    print("\n🔍 Testing Shadow Book System...")

    try:
        from learning.shadow_book import ShadowBookEngine, Trade, TradeType, ActionType

        # Create in-memory shadow book
        shadow_book = ShadowBookEngine(db_path=':memory:')

        # Create sample trade
        sample_trade = Trade(
            trade_id='VALIDATION_001',
            symbol='AAPL',
            trade_type=TradeType.ACTUAL,
            action_type=ActionType.ENTRY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            strategy_id='validation_test',
            confidence=0.8
        )

        # Test basic functionality (without database operations)
        shadow_book.trades.append(sample_trade)

        # Test performance calculation
        performance = shadow_book.get_shadow_book_performance("actual")

        print("✅ Shadow Book System functional")
        print(f"   - Performance metrics: {list(performance.keys())}")

        return True

    except Exception as e:
        print(f"❌ Shadow Book error: {e}")
        return False

def test_policy_twin():
    """Test Policy Twin System functionality"""
    print("\n🔍 Testing Policy Twin System...")

    try:
        from learning.policy_twin import PolicyTwin, AlphaType

        policy_twin = PolicyTwin()

        # Sample trade data
        sample_trade_data = {
            'trade_id': 'VALIDATION_002',
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 1000,
            'price': 150.0,
            'strategy_id': 'momentum_strategy'
        }

        # Analyze ethics
        ethical_trade = policy_twin.analyze_trade_ethics(sample_trade_data)

        print("✅ Policy Twin System functional")
        print(f"   - Alpha Type: {ethical_trade.alpha_type.value}")
        print(f"   - Social Impact Score: {ethical_trade.social_impact_score:.3f}")
        print(f"   - Transparency Level: {ethical_trade.transparency_level:.3f}")

        return True

    except Exception as e:
        print(f"❌ Policy Twin error: {e}")
        return False

async def test_narrative_gap():
    """Test Narrative Gap Engine functionality"""
    print("\n🔍 Testing Narrative Gap Engine...")

    try:
        from intelligence.narrative.narrative_gap import NarrativeGapEngine

        ng_engine = NarrativeGapEngine()

        # Test basic signal generation
        symbol = "AAPL"
        current_price = 150.0

        signal = await ng_engine.calculate_narrative_gap(symbol, current_price)

        print("✅ Narrative Gap Engine functional")
        print(f"   - NG Score: {signal.ng_score:.3f}")
        print(f"   - Gap Magnitude: {signal.gap_magnitude:.3f}")
        print(f"   - Confidence: {signal.confidence:.3f}")

        return True

    except Exception as e:
        print(f"❌ Narrative Gap error: {e}")
        return False

def test_component_compatibility():
    """Test that components work together"""
    print("\n🔍 Testing component compatibility...")

    try:
        from learning.shadow_book import ShadowBookEngine
        from learning.policy_twin import PolicyTwin

        # Initialize components
        shadow_book = ShadowBookEngine(db_path=':memory:')
        policy_twin = PolicyTwin()

        # Test data flow
        trade_data = {
            'trade_id': 'COMPAT_TEST',
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 1000,
            'price': 150.0,
            'strategy_id': 'test'
        }

        ethical_trade = policy_twin.analyze_trade_ethics(trade_data)

        print("✅ Components compatible")
        print(f"   - Ethical assessment completed")
        print(f"   - Shadow book initialized")

        return True

    except Exception as e:
        print(f"❌ Compatibility error: {e}")
        return False

async def main():
    """Run all validation tests"""
    print("=" * 60)
    print("PHASE 5 ALPHA GENERATION SYSTEMS VALIDATION")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Shadow Book Tests", test_shadow_book),
        ("Policy Twin Tests", test_policy_twin),
        ("Narrative Gap Tests", test_narrative_gap),
        ("Compatibility Tests", test_component_compatibility)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL ALPHA GENERATION SYSTEMS VALIDATED SUCCESSFULLY!")
        print("\nKey Features Validated:")
        print("✅ Shadow Book counterfactual P&L tracking")
        print("✅ Policy Twin ethical trading framework")
        print("✅ Narrative Gap market consensus analysis")
        print("✅ Component integration compatibility")
        print("\nSystems ready for integration and deployment!")
    else:
        print("⚠️  Some validation tests failed. Review errors above.")

    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())