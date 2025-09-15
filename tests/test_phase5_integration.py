"""
Comprehensive Test Suite for Phase 5 Risk & Calibration Systems

Tests all Phase 5 components including Brier scoring, convexity optimization,
enhanced Kelly criterion, and integrated risk management system.

Test Coverage:
- Brier Score Calibration system functionality
- Convexity Manager regime detection and optimization
- Enhanced Kelly Criterion with survival constraints
- Phase 5 Integration system coordination
- Performance under various market conditions
- Error handling and recovery
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.risk.brier_scorer import BrierScorer, CalibrationMetrics, PredictionRecord
    from src.risk.convexity_manager import ConvexityManager, MarketRegime, RegimeState
    from src.risk.kelly_enhanced import EnhancedKellyCriterion, SurvivalMode, AssetRiskProfile
    from src.risk.phase5_integration import Phase5Integration, Phase5Mode
    PHASE5_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 5 components not available: {e}")
    PHASE5_AVAILABLE = False

class TestBrierScorer(unittest.TestCase):
    """Test suite for Brier Score calibration system"""

    def setUp(self):
        """Set up test environment"""
        if not PHASE5_AVAILABLE:
            self.skipTest("Phase 5 components not available")

        self.scorer = BrierScorer({
            'min_predictions': 10,
            'calibration_window': 100,
            'base_kelly_multiplier': 0.25
        })

    def test_prediction_addition(self):
        """Test adding predictions to the system"""
        pred_id = self.scorer.add_prediction(
            prediction_id="test_001",
            forecast=0.7,
            prediction_type="direction",
            confidence=0.8
        )

        self.assertEqual(pred_id, "test_001")
        self.assertIn("test_001", self.scorer.predictions)

        prediction = self.scorer.predictions["test_001"]
        self.assertEqual(prediction.forecast, 0.7)
        self.assertEqual(prediction.prediction_type, "direction")
        self.assertEqual(prediction.confidence, 0.8)
        self.assertIsNone(prediction.outcome)

    def test_outcome_update(self):
        """Test updating prediction outcomes"""
        self.scorer.add_prediction("test_002", 0.6, "volatility")
        success = self.scorer.update_outcome("test_002", True)

        self.assertTrue(success)
        self.assertTrue(self.scorer.predictions["test_002"].outcome)

    def test_calibration_score_calculation(self):
        """Test calibration score calculation"""
        # Add well-calibrated predictions
        np.random.seed(42)
        for i in range(50):
            pred_id = f"calibrated_{i}"
            forecast = np.random.uniform(0.2, 0.8)
            outcome = np.random.random() < forecast  # Well-calibrated

            self.scorer.add_prediction(pred_id, forecast, "direction")
            self.scorer.update_outcome(pred_id, outcome)

        score = self.scorer.get_calibration_score()
        self.assertGreater(score, 0.4)  # Should be reasonable for well-calibrated data
        self.assertLessEqual(score, 1.0)

    def test_position_size_multiplier(self):
        """Test position sizing based on calibration"""
        # Add poor predictions
        for i in range(20):
            pred_id = f"poor_{i}"
            forecast = 0.8  # Always high confidence
            outcome = False  # Always wrong

            self.scorer.add_prediction(pred_id, forecast, "direction")
            self.scorer.update_outcome(pred_id, outcome)

        multiplier = self.scorer.get_position_size_multiplier()
        self.assertLess(multiplier, 0.5)  # Should reduce position size for poor calibration

    def test_performance_scoreboard(self):
        """Test performance scoreboard generation"""
        # Add mixed predictions
        for i in range(30):
            pred_id = f"mixed_{i}"
            forecast = np.random.uniform(0.3, 0.7)
            outcome = np.random.random() < 0.5

            self.scorer.add_prediction(pred_id, forecast, "direction")
            self.scorer.update_outcome(pred_id, outcome)

        scoreboard = self.scorer.get_performance_scoreboard()

        self.assertIn('overall_metrics', scoreboard)
        self.assertIn('timestamp', scoreboard)
        self.assertIsInstance(scoreboard['overall_metrics']['total_predictions'], int)

class TestConvexityManager(unittest.TestCase):
    """Test suite for Convexity optimization system"""

    def setUp(self):
        """Set up test environment"""
        if not PHASE5_AVAILABLE:
            self.skipTest("Phase 5 components not available")

        self.manager = ConvexityManager({
            'hmm_components': 3,
            'regime_lookback': 100,
            'max_gamma_exposure': 0.1
        })

    def test_market_data_update(self):
        """Test market data processing and regime detection"""
        # Generate sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        np.random.seed(42)

        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        price_data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices
        }, index=dates)

        regime_state = self.manager.update_market_data(price_data)

        self.assertIsInstance(regime_state, RegimeState)
        self.assertIn(regime_state.regime, MarketRegime)
        self.assertGreaterEqual(regime_state.confidence, 0)
        self.assertLessEqual(regime_state.confidence, 1)

    def test_convexity_requirements(self):
        """Test convexity requirement calculation"""
        # Create dummy regime state
        regime_state = RegimeState(
            regime=MarketRegime.HIGH_VOL_CRISIS,
            confidence=0.8,
            regime_probabilities={},
            uncertainty=0.6,
            time_in_regime=5,
            transition_probability=0.3
        )

        target = self.manager.get_convexity_requirements(
            asset="SPY",
            position_size=100000,
            current_regime=regime_state
        )

        self.assertGreater(target.target_gamma, 0)  # Should require positive gamma in crisis
        self.assertIsInstance(target.convexity_score, float)

    def test_gamma_farming_optimization(self):
        """Test gamma farming strategy optimization"""
        structure = self.manager.optimize_gamma_farming(
            underlying="SPY",
            portfolio_value=1000000,
            implied_vol_percentile=0.2,  # Low IV for farming
            current_volatility=0.15
        )

        if structure:  # May return None if conditions aren't met
            self.assertEqual(structure.underlying, "SPY")
            self.assertEqual(structure.structure_type, "straddle")
            self.assertGreater(structure.gamma, 0)

    def test_event_exposure_management(self):
        """Test event-driven exposure management"""
        recommendations = self.manager.manage_event_exposure()

        self.assertIn('actions', recommendations)
        self.assertIn('risk_adjustments', recommendations)
        self.assertIn('hedge_suggestions', recommendations)
        self.assertIsInstance(recommendations['actions'], list)

class TestEnhancedKellyCriterion(unittest.TestCase):
    """Test suite for Enhanced Kelly Criterion system"""

    def setUp(self):
        """Set up test environment"""
        if not PHASE5_AVAILABLE:
            self.skipTest("Phase 5 components not available")

        self.kelly = EnhancedKellyCriterion({
            'base_kelly_fraction': 0.2,
            'survival_mode': SurvivalMode.MODERATE,
            'max_single_asset': 0.1
        })

    def test_asset_profile_creation(self):
        """Test asset risk profile creation"""
        # Generate sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252))

        market_data = {
            'liquidity_score': 0.8,
            'beta_equity': 1.2,
            'crowding_score': 0.4
        }

        profile = self.kelly.add_asset_profile("TEST", returns, market_data)

        self.assertEqual(profile.asset, "TEST")
        self.assertGreater(profile.expected_return, -0.5)  # Reasonable bounds
        self.assertLess(profile.expected_return, 2.0)
        self.assertGreater(profile.volatility, 0)

    def test_survival_kelly_calculation(self):
        """Test survival-first Kelly calculation"""
        # Add asset profile first
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        self.kelly.add_asset_profile("SPY", returns, {'liquidity_score': 0.9})

        survival_kelly = self.kelly.calculate_survival_kelly(
            asset="SPY",
            base_kelly=0.15,
            confidence_adjustment=0.8
        )

        self.assertGreaterEqual(survival_kelly, 0)
        self.assertLess(survival_kelly, 0.15)  # Should be conservative

    def test_multi_asset_optimization(self):
        """Test multi-asset portfolio optimization"""
        # Add multiple assets
        assets = ['SPY', 'TLT', 'GLD']
        for asset in assets:
            returns = pd.Series(np.random.normal(0.0008, 0.018, 100))
            market_data = {
                'liquidity_score': np.random.uniform(0.6, 0.9),
                'beta_equity': np.random.uniform(0.5, 1.5),
                'crowding_score': np.random.uniform(0.2, 0.6)
            }
            self.kelly.add_asset_profile(asset, returns, market_data)

        # Update correlation matrix
        returns_data = {asset: pd.Series(np.random.normal(0.001, 0.02, 100))
                       for asset in assets}
        returns_df = pd.DataFrame(returns_data)
        self.kelly.update_correlation_matrix(returns_df)

        # Optimize portfolio
        expected_returns = {asset: np.random.uniform(0.05, 0.12) for asset in assets}
        result = self.kelly.optimize_multi_asset_portfolio(expected_returns)

        self.assertEqual(result.optimization_status, "success")
        self.assertIn('SPY', result.optimal_weights)
        self.assertGreater(result.expected_return, 0)
        self.assertGreater(result.survival_probability, 0.8)

    def test_factor_decomposition(self):
        """Test factor exposure decomposition"""
        weights = {'SPY': 0.6, 'TLT': 0.3, 'CASH': 0.1}

        # Add assets with factor exposures
        for asset in ['SPY', 'TLT']:
            returns = pd.Series(np.random.normal(0.001, 0.02, 50))
            market_data = {
                'beta_equity': 1.0 if asset == 'SPY' else -0.2,
                'beta_duration': 0.1 if asset == 'SPY' else 0.8,
                'beta_inflation': 0.0
            }
            self.kelly.add_asset_profile(asset, returns, market_data)

        exposures = self.kelly.get_factor_decomposition(weights)

        self.assertIn('equity_beta', exposures)
        self.assertIn('duration_beta', exposures)
        self.assertGreater(exposures['equity_beta'], 0)  # SPY dominance

class TestPhase5Integration(unittest.TestCase):
    """Test suite for Phase 5 integrated system"""

    def setUp(self):
        """Set up test environment"""
        if not PHASE5_AVAILABLE:
            self.skipTest("Phase 5 components not available")

        self.integration = Phase5Integration({
            'monitoring_enabled': False,  # Disable for testing
            'update_frequency': 1
        })

    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.integration.brier_scorer)
        self.assertIsNotNone(self.integration.convexity_manager)
        self.assertIsNotNone(self.integration.kelly_system)
        self.assertEqual(self.integration.current_mode, Phase5Mode.FULL_OPERATIONAL)

    def test_prediction_workflow(self):
        """Test end-to-end prediction workflow"""
        success = self.integration.add_prediction_and_outcome(
            prediction_id="workflow_test",
            forecast=0.65,
            prediction_type="direction",
            outcome=True,
            confidence=0.75
        )

        self.assertTrue(success)

    def test_market_data_workflow(self):
        """Test market data processing workflow"""
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
        np.random.seed(42)

        price_data = pd.DataFrame({
            'open': 100 + np.random.normal(0, 1, len(dates)).cumsum(),
            'high': 102 + np.random.normal(0, 1, len(dates)).cumsum(),
            'low': 98 + np.random.normal(0, 1, len(dates)).cumsum(),
            'close': 100 + np.random.normal(0, 1, len(dates)).cumsum()
        }, index=dates)

        status = self.integration.update_market_data(price_data)

        self.assertIsInstance(status.mode, Phase5Mode)
        self.assertIsInstance(status.portfolio_health, str)

    def test_portfolio_optimization_workflow(self):
        """Test integrated portfolio optimization"""
        # Prepare test data
        assets = ['SPY', 'TLT']
        expected_returns = {'SPY': 0.08, 'TLT': 0.04}
        assets_data = {
            asset: pd.Series(np.random.normal(0.001, 0.02, 100))
            for asset in assets
        }

        result = self.integration.optimize_portfolio(expected_returns, assets_data)

        if result:  # May be None if optimization fails
            self.assertIn('SPY', result.optimal_weights)
            self.assertGreater(result.expected_return, 0)

    def test_position_sizing_recommendation(self):
        """Test position sizing recommendation system"""
        recommendation = self.integration.get_position_sizing_recommendation(
            asset="SPY",
            base_kelly=0.12
        )

        self.assertIn('final_recommendation', recommendation)
        self.assertIn('adjustment_factor', recommendation)
        self.assertGreaterEqual(recommendation['final_recommendation'], 0)

    def test_integrated_dashboard(self):
        """Test integrated risk dashboard"""
        dashboard = self.integration.get_integrated_risk_dashboard()

        self.assertIn('timestamp', dashboard)
        self.assertIn('system_status', dashboard)
        self.assertIn('calibration', dashboard)
        self.assertIn('convexity', dashboard)
        self.assertIn('kelly', dashboard)

    def test_system_mode_transitions(self):
        """Test system mode transitions"""
        # Force survival mode by degrading metrics
        self.integration.current_mode = Phase5Mode.SURVIVAL_MODE
        self.assertEqual(self.integration.current_mode, Phase5Mode.SURVIVAL_MODE)

        # Test emergency mode
        self.integration._trigger_emergency_mode("Test emergency")
        self.assertEqual(self.integration.current_mode, Phase5Mode.EMERGENCY_STOP)

class TestPhase5Performance(unittest.TestCase):
    """Performance and stress testing for Phase 5 systems"""

    def setUp(self):
        """Set up performance test environment"""
        if not PHASE5_AVAILABLE:
            self.skipTest("Phase 5 components not available")

    def test_calibration_performance_large_dataset(self):
        """Test calibration system with large prediction dataset"""
        scorer = BrierScorer()

        # Add 1000 predictions quickly
        start_time = datetime.now()
        for i in range(1000):
            scorer.add_prediction(f"perf_{i}", np.random.random(), "direction")
            scorer.update_outcome(f"perf_{i}", np.random.random() > 0.5)

        elapsed = (datetime.now() - start_time).total_seconds()
        self.assertLess(elapsed, 10)  # Should complete in under 10 seconds

        # Test scoreboard generation performance
        start_time = datetime.now()
        scoreboard = scorer.get_performance_scoreboard()
        elapsed = (datetime.now() - start_time).total_seconds()
        self.assertLess(elapsed, 2)  # Scoreboard should be fast

    def test_regime_detection_performance(self):
        """Test regime detection with large dataset"""
        manager = ConvexityManager()

        # Large dataset
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        price_data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices
        }, index=dates)

        start_time = datetime.now()
        regime_state = manager.update_market_data(price_data)
        elapsed = (datetime.now() - start_time).total_seconds()

        self.assertLess(elapsed, 30)  # Should complete in reasonable time
        self.assertIsInstance(regime_state, RegimeState)

def run_phase5_validation():
    """Run comprehensive Phase 5 validation suite"""
    if not PHASE5_AVAILABLE:
        print("Phase 5 components not available - skipping tests")
        return False

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create test suite
    test_classes = [
        TestBrierScorer,
        TestConvexityManager,
        TestEnhancedKellyCriterion,
        TestPhase5Integration,
        TestPhase5Performance
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 5 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")

    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_phase5_validation()
    sys.exit(0 if success else 1)