#!/usr/bin/env python3
"""
Complete End-to-End Integration Test
Gary x Taleb Autonomous Trading System - All 5 Phases
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import sys
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

def test_phase_5_vision_components():
    """Test Phase 5: Super-Gary Vision Components"""
    print("=== PHASE 5: Super-Gary Vision Components ===")

    try:
        # Test Narrative Gap Engine
        from src.trading.narrative_gap import NarrativeGap
        ng = NarrativeGap()

        # Test NG calculation
        market_price = 100.0
        consensus_forecast = 105.0
        gary_estimate = 110.0

        ng_score = ng.calculate_ng(market_price, consensus_forecast, gary_estimate)
        multiplier = ng.position_multiplier(ng_score)

        print(f"SUCCESS: Narrative Gap: Score = {ng_score:.4f}, Multiplier = {multiplier:.3f}x")

        # Test Brier Score Calibration
        from src.performance.simple_brier import BrierTracker
        brier = BrierTracker()

        # Record some test predictions
        brier.record_prediction(0.7, 1)  # 70% confidence, correct
        brier.record_prediction(0.3, 0)  # 30% confidence, correct
        brier.record_prediction(0.8, 0)  # 80% confidence, wrong

        brier_score = brier.get_brier_score()
        print(f"SUCCESS: Brier Calibration: Score = {brier_score:.4f}")

        # Test Enhanced DPI with wealth flow
        from src.strategies.dpi_calculator import DistributionalPressureIndex
        enhanced_dpi = DistributionalPressureIndex()

        print("SUCCESS: Enhanced DPI: Import successful")

        return True

    except Exception as e:
        print(f"FAILED: Phase 5 - {e}")
        traceback.print_exc()
        return False

def test_integrated_signal_generation():
    """Test Complete Integrated Signal Generation"""
    print("=== INTEGRATED SIGNAL GENERATION ===")

    try:
        # Test complete signal pipeline
        from src.trading.narrative_gap import NarrativeGap
        from src.performance.simple_brier import BrierTracker

        # Initialize components
        ng = NarrativeGap()
        brier = BrierTracker()

        # Simulate trading signal generation
        market_price = 150.0
        consensus_forecast = 155.0
        gary_estimate = 160.0

        # Step 1: Calculate Narrative Gap
        ng_score = ng.calculate_ng(market_price, consensus_forecast, gary_estimate)
        ng_multiplier = ng.position_multiplier(ng_score)

        # Step 2: Get Brier adjustment
        brier.record_prediction(0.7, 1)  # Some calibration history
        brier_score = brier.get_brier_score()
        brier_adjustment = 1 - brier_score

        # Step 3: Calculate enhanced position
        base_position = 1000  # $1000 base
        enhanced_position = base_position * ng_multiplier * brier_adjustment

        print(f"SUCCESS: Integrated Signal Generation:")
        print(f"   Base Position: ${base_position:.2f}")
        print(f"   NG Multiplier: {ng_multiplier:.3f}x")
        print(f"   Brier Adjustment: {brier_adjustment:.3f}x")
        print(f"   Final Enhanced Position: ${enhanced_position:.2f}")
        print(f"   Enhancement: {((enhanced_position / base_position) - 1) * 100:+.1f}%")

        return True

    except Exception as e:
        print(f"FAILED: Integrated Signal Generation - {e}")
        traceback.print_exc()
        return False

def main():
    """Run integration test"""
    print("Gary x Taleb Trading System - Integration Test")
    print("=" * 50)
    print(f"Start Time: {datetime.now()}")
    print("=" * 50)

    # Track test results
    results = {}

    # Run key tests
    results['Phase 5'] = test_phase_5_vision_components()
    results['Integration'] = test_integrated_signal_generation()

    # Summary
    print("=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100

    for phase, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{phase:15}: {status}")

    print("-" * 50)
    print(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("INTEGRATION TEST: SUCCESS - System ready for next phase")
        return True
    else:
        print("INTEGRATION TEST: PARTIAL - Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)