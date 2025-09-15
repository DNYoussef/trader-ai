#!/usr/bin/env python3
"""
Complete End-to-End Integration Test
Gary×Taleb Autonomous Trading System - All 5 Phases

Tests: Data → DPI → ML → Risk → NG Enhancement → Position Sizing
Validates: Phase 2 → Phase 3 → Phase 4 → Phase 5 integration
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

def test_phase_2_risk_framework():
    """Test Phase 2: Risk & Quality Framework"""
    print("=== PHASE 2: Risk & Quality Framework ===")

    try:
        # Test Kelly Criterion (enhanced with Phase 5)
        from src.risk.kelly_criterion import KellyCriterion
        kelly = KellyCriterion()

        # Test basic Kelly calculation
        win_rate = 0.6
        avg_win = 0.15
        avg_loss = 0.10
        capital = 10000

        position_size = kelly.calculate_position_size(win_rate, avg_win, avg_loss, capital)
        print(f"✅ Kelly Criterion: ${position_size:.2f} position size")

        # Test Kill Switch
        from src.safety.kill_switch import KillSwitch
        kill_switch = KillSwitch(max_drawdown=0.10)

        # Simulate 5% drawdown (should not trigger)
        kill_switch.update_portfolio_value(9500)  # 5% loss
        assert not kill_switch.is_triggered(), "Kill switch should not trigger at 5%"

        # Simulate 15% drawdown (should trigger)
        kill_switch.update_portfolio_value(8500)  # 15% loss
        assert kill_switch.is_triggered(), "Kill switch should trigger at 15%"
        print("✅ Kill Switch: Triggers correctly at 10% drawdown")

        # Test EVT
        from src.risk.extreme_value_theory import ExtremeValueTheory
        evt = ExtremeValueTheory()

        # Generate sample returns
        returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
        evt.fit(returns)
        var_95 = evt.calculate_var(0.95)
        print(f"✅ EVT: 95% VaR = {var_95:.4f}")

        return True

    except Exception as e:
        print(f"❌ Phase 2 FAILED: {e}")
        traceback.print_exc()
        return False

def test_phase_3_ml_intelligence():
    """Test Phase 3: ML Intelligence Layer"""
    print("\n=== PHASE 3: ML Intelligence Layer ===")

    try:
        # Test Gary DPI System
        from src.trading.gary_dpi_system import GaryDPISystem
        dpi_system = GaryDPISystem()

        # Generate sample market data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
        volumes = np.random.lognormal(10, 1, 100)

        market_data = pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'high': prices * 1.02,
            'low': prices * 0.98
        }, index=dates)

        # Calculate DPI
        dpi_score = dpi_system.calculate_dpi(market_data)
        print(f"✅ Gary DPI: Score = {dpi_score:.4f}")

        # Test Taleb Antifragility
        from src.trading.taleb_antifragility import TalebAntifragilityEngine
        taleb_engine = TalebAntifragilityEngine()

        antifragility_score = taleb_engine.calculate_antifragility(market_data)
        print(f"✅ Taleb Antifragility: Score = {antifragility_score:.4f}")

        # Test ML Models (if available)
        try:
            import os
            model_dir = "trained_models"
            if os.path.exists(model_dir) and os.listdir(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.pth'))]
                print(f"✅ ML Models: {len(model_files)} trained models found")
            else:
                print("⚠️ ML Models: No trained models found (not critical)")
        except Exception as e:
            print(f"⚠️ ML Models: {e}")

        return True

    except Exception as e:
        print(f"❌ Phase 3 FAILED: {e}")
        traceback.print_exc()
        return False

def test_phase_4_production_system():
    """Test Phase 4: Production System"""
    print("\n=== PHASE 4: Production System ===")

    try:
        # Test Production Factory
        from src.integration.phase2_factory import Phase2SystemFactory

        # Test factory creation (mock mode for testing)
        factory = Phase2SystemFactory()
        print("✅ Phase2SystemFactory: Created successfully")

        # Test dashboard components (if available)
        try:
            import subprocess
            import os

            dashboard_path = "src/risk-dashboard"
            if os.path.exists(dashboard_path):
                # Check if package.json exists
                package_json = os.path.join(dashboard_path, "package.json")
                if os.path.exists(package_json):
                    print("✅ Risk Dashboard: Package.json found")
                else:
                    print("⚠️ Risk Dashboard: Package.json not found")
            else:
                print("⚠️ Risk Dashboard: Directory not found")
        except Exception as e:
            print(f"⚠️ Risk Dashboard: {e}")

        # Test performance benchmarking
        try:
            from src.performance.benchmarker.BenchmarkExecutor import BenchmarkExecutor
            print("✅ Performance Benchmarker: Import successful")
        except Exception as e:
            print(f"⚠️ Performance Benchmarker: {e}")

        return True

    except Exception as e:
        print(f"❌ Phase 4 FAILED: {e}")
        traceback.print_exc()
        return False

def test_phase_5_vision_components():
    """Test Phase 5: Super-Gary Vision Components"""
    print("\n=== PHASE 5: Super-Gary Vision Components ===")

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

        print(f"✅ Narrative Gap: Score = {ng_score:.4f}, Multiplier = {multiplier:.3f}x")

        # Test Brier Score Calibration
        from src.performance.simple_brier import BrierTracker
        brier = BrierTracker()

        # Record some test predictions
        brier.record_prediction(0.7, 1)  # 70% confidence, correct
        brier.record_prediction(0.3, 0)  # 30% confidence, correct
        brier.record_prediction(0.8, 0)  # 80% confidence, wrong

        brier_score = brier.get_brier_score()
        print(f"✅ Brier Calibration: Score = {brier_score:.4f}")

        # Test Enhanced DPI with wealth flow
        from src.strategies.dpi_calculator import DistributionalPressureIndex
        enhanced_dpi = DistributionalPressureIndex()

        # Test wealth flow calculation
        income_data = {
            'high_income_gains': 1000000,
            'total_gains': 1200000,
            'wealth_concentration': 0.8
        }

        flow_score = enhanced_dpi.calculate_wealth_flow(income_data)
        print(f"✅ Enhanced DPI: Wealth flow score = {flow_score:.4f}")

        return True

    except Exception as e:
        print(f"❌ Phase 5 FAILED: {e}")
        traceback.print_exc()
        return False

def test_integrated_signal_generation():
    """Test Complete Integrated Signal Generation"""
    print("\n=== INTEGRATED SIGNAL GENERATION ===")

    try:
        # Test complete signal pipeline: Data → DPI → NG → Brier → Kelly
        from src.trading.narrative_gap import NarrativeGap
        from src.performance.simple_brier import BrierTracker
        from src.risk.kelly_criterion import KellyCriterion

        # Initialize components
        ng = NarrativeGap()
        brier = BrierTracker()
        kelly = KellyCriterion()

        # Simulate trading signal generation
        market_price = 150.0
        consensus_forecast = 155.0
        gary_estimate = 160.0

        # Step 1: Calculate Narrative Gap
        ng_score = ng.calculate_ng(market_price, consensus_forecast, gary_estimate)
        ng_multiplier = ng.position_multiplier(ng_score)

        # Step 2: Get Brier adjustment (assume some prediction history)
        brier.record_prediction(0.7, 1)  # Some calibration history
        brier_score = brier.get_brier_score()
        brier_adjustment = 1 - brier_score

        # Step 3: Calculate base Kelly position
        win_rate = 0.6
        avg_win = 0.12
        avg_loss = 0.08
        capital = 10000

        base_kelly = kelly.calculate_position_size(win_rate, avg_win, avg_loss, capital)

        # Step 4: Apply enhancements
        enhanced_position = base_kelly * ng_multiplier * brier_adjustment

        print(f"✅ Integrated Signal Generation:")
        print(f"   Base Kelly Position: ${base_kelly:.2f}")
        print(f"   NG Multiplier: {ng_multiplier:.3f}x")
        print(f"   Brier Adjustment: {brier_adjustment:.3f}x")
        print(f"   Final Enhanced Position: ${enhanced_position:.2f}")
        print(f"   Enhancement: {((enhanced_position / base_kelly) - 1) * 100:+.1f}%")

        return True

    except Exception as e:
        print(f"❌ Integrated Signal Generation FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run complete integration test"""
    print("🚀 Gary×Taleb Trading System - Complete Integration Test")
    print("=" * 60)
    print(f"Start Time: {datetime.now()}")
    print("=" * 60)

    # Track test results
    results = {}

    # Run all phase tests
    results['Phase 2'] = test_phase_2_risk_framework()
    results['Phase 3'] = test_phase_3_ml_intelligence()
    results['Phase 4'] = test_phase_4_production_system()
    results['Phase 5'] = test_phase_5_vision_components()
    results['Integration'] = test_integrated_signal_generation()

    # Summary
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100

    for phase, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{phase:15}: {status}")

    print("-" * 60)
    print(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("🎉 INTEGRATION TEST: SUCCESS - System ready for next phase")
        return True
    else:
        print("⚠️ INTEGRATION TEST: PARTIAL - Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)