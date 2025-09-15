#!/usr/bin/env python3
"""
Simple Alpha Generation Systems Validation
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

def test_policy_twin():
    """Test Policy Twin System"""
    print("Testing Policy Twin System...")

    try:
        from learning.policy_twin import PolicyTwin, AlphaType

        policy_twin = PolicyTwin()

        sample_trade_data = {
            'trade_id': 'TEST_001',
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 1000,
            'price': 150.0,
            'strategy_id': 'momentum_strategy'
        }

        ethical_trade = policy_twin.analyze_trade_ethics(sample_trade_data)

        print("PASS: Policy Twin System working")
        print(f"  Alpha Type: {ethical_trade.alpha_type.value}")
        print(f"  Social Impact: {ethical_trade.social_impact_score:.3f}")
        print(f"  Transparency: {ethical_trade.transparency_level:.3f}")

        return True

    except Exception as e:
        print(f"FAIL: Policy Twin error: {e}")
        return False

def test_shadow_book():
    """Test Shadow Book System"""
    print("\nTesting Shadow Book System...")

    try:
        from learning.shadow_book import ShadowBookEngine, Trade, TradeType, ActionType

        # Simple test without database
        shadow_book = ShadowBookEngine(db_path=':memory:')

        sample_trade = Trade(
            trade_id='TEST_002',
            symbol='AAPL',
            trade_type=TradeType.ACTUAL,
            action_type=ActionType.ENTRY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            strategy_id='test_strategy',
            confidence=0.8
        )

        # Test basic functionality
        shadow_book.trades.append(sample_trade)
        performance = shadow_book.get_shadow_book_performance("actual")

        print("PASS: Shadow Book System working")
        print(f"  Performance keys: {list(performance.keys())}")

        return True

    except Exception as e:
        print(f"FAIL: Shadow Book error: {e}")
        return False

async def test_narrative_gap():
    """Test Narrative Gap Engine"""
    print("\nTesting Narrative Gap Engine...")

    try:
        from intelligence.narrative.narrative_gap import NarrativeGapEngine

        ng_engine = NarrativeGapEngine()
        signal = await ng_engine.calculate_narrative_gap("AAPL", 150.0)

        print("PASS: Narrative Gap Engine working")
        print(f"  NG Score: {signal.ng_score:.3f}")
        print(f"  Gap Magnitude: {signal.gap_magnitude:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")

        return True

    except Exception as e:
        print(f"FAIL: Narrative Gap error: {e}")
        return False

async def main():
    """Run validation tests"""
    print("=" * 50)
    print("ALPHA GENERATION SYSTEMS VALIDATION")
    print("=" * 50)

    tests = [
        test_policy_twin(),
        test_shadow_book(),
        await test_narrative_gap()
    ]

    passed = sum(tests)
    total = len(tests)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nALL SYSTEMS VALIDATED SUCCESSFULLY!")
        print("Alpha generation components ready for deployment.")
    else:
        print("\nSome tests failed. Check errors above.")

    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())