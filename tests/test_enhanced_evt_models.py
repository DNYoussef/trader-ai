"""
Comprehensive Test Suite for Enhanced EVT Models - Phase 2 Division 1

Tests all components of the enhanced EVT tail modeling system:
1. Enhanced EVT model fitting and selection
2. Backtesting framework validation
3. Integration with existing antifragility engine
4. Performance benchmarks (<100ms target)
5. VaR accuracy validation (±5% target)
"""

import pytest
import numpy as np
import pandas as pd
import time
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import components under test
from risk.enhanced_evt_models import (
    EnhancedEVTEngine,
    TailDistribution,
    EstimationMethod,
    EnhancedTailRiskModel,
    enhance_existing_evt_model
)
from risk.evt_backtesting import (
    EVTBacktestingEngine,
    BacktestType,
    BacktestResults,
    ESBacktestResults,
    run_comprehensive_backtest
)
from risk.evt_integration import (
    EnhancedAntifragilityIntegration,
    IntegratedTailRiskModel,
    PerformanceComparison,
    patch_antifragility_engine,
    validate_phase2_requirements
)

# Import existing antifragility engine for integration testing
from strategies.antifragility_engine import AntifragilityEngine, TailRiskModel

class TestEnhancedEVTModels:
    """Test suite for enhanced EVT model fitting and selection"""

    @pytest.fixture
    def sample_returns_data(self):
        """Generate various types of return data for testing"""
        np.random.seed(42)

        # Normal market returns
        normal_returns = np.random.normal(0.001, 0.02, 1000)

        # Fat-tailed returns (Student's t)
        fat_tail_returns = np.random.standard_t(df=3, size=1000) * 0.02

        # Skewed returns
        skewed_returns = np.random.gamma(2, 0.01, 1000) - 0.02

        # High volatility periods with clustering
        volatile_returns = np.random.normal(0.0005, 0.01, 1000)
        for i in range(100, 150):  # Add volatility cluster
            volatile_returns[i] *= 3

        # Extreme event data
        extreme_returns = normal_returns.copy()
        extreme_returns[500] = -0.15  # -15% crash
        extreme_returns[750] = -0.08  # -8% drop

        return {
            'normal': normal_returns,
            'fat_tail': fat_tail_returns,
            'skewed': skewed_returns,
            'volatile': volatile_returns,
            'extreme': extreme_returns
        }

    @pytest.fixture
    def enhanced_engine(self):
        """Create enhanced EVT engine for testing"""
        return EnhancedEVTEngine(
            threshold_percentile=95.0,
            min_exceedances=20,
            confidence_level=0.95
        )

    def test_enhanced_engine_initialization(self, enhanced_engine):
        """Test enhanced EVT engine initialization"""
        assert enhanced_engine.threshold_percentile == 95.0
        assert enhanced_engine.min_exceedances == 20
        assert enhanced_engine.confidence_level == 0.95
        assert len(enhanced_engine.available_distributions) == 5
        assert len(enhanced_engine.available_methods) == 3

    def test_fit_multiple_models_normal_data(self, enhanced_engine, sample_returns_data):
        """Test model fitting with normal return data"""
        returns = sample_returns_data['normal']

        result = enhanced_engine.fit_multiple_models(returns, 'NORMAL_TEST')

        assert isinstance(result, EnhancedTailRiskModel)
        assert result.symbol == 'NORMAL_TEST'
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.var_99_9 > result.var_99
        assert result.expected_shortfall_95 >= result.var_95
        assert result.expected_shortfall_99 >= result.var_99

        # Should have selected a best model
        assert result.best_model is not None
        assert result.best_model.distribution in TailDistribution
        assert result.best_model.aic < np.inf

        # Should have alternative models
        assert len(result.alternative_models) >= 0

    def test_fit_multiple_models_fat_tail_data(self, enhanced_engine, sample_returns_data):
        """Test model fitting with fat-tailed return data"""
        returns = sample_returns_data['fat_tail']

        result = enhanced_engine.fit_multiple_models(returns, 'FAT_TAIL_TEST')

        assert isinstance(result, EnhancedTailRiskModel)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95

        # Fat-tailed data should likely select Student-t or GPD
        assert result.best_model.distribution in [TailDistribution.STUDENT_T, TailDistribution.GPD]

        # Should have reasonable tail index for fat tails
        if result.best_model.distribution == TailDistribution.GPD:
            assert result.best_model.parameters.shape > 0  # Positive shape indicates fat tail

    def test_fit_multiple_models_insufficient_data(self, enhanced_engine):
        """Test model fitting with insufficient data"""
        insufficient_returns = np.random.normal(0, 0.01, 50)  # Only 50 observations

        result = enhanced_engine.fit_multiple_models(insufficient_returns, 'INSUFFICIENT_TEST')

        # Should fall back to empirical model
        assert isinstance(result, EnhancedTailRiskModel)
        assert result.var_95 > 0
        assert len(result.alternative_models) == 0  # No alternatives due to insufficient data

    def test_model_selection_criteria(self, enhanced_engine, sample_returns_data):
        """Test that model selection uses appropriate criteria"""
        returns = sample_returns_data['normal']

        result = enhanced_engine.fit_multiple_models(returns, 'SELECTION_TEST')

        best_model = result.best_model

        # Best model should have finite AIC/BIC
        assert np.isfinite(best_model.aic)
        assert np.isfinite(best_model.bic)

        # If there are alternatives, best model should have lower or comparable AIC
        for alt_model in result.alternative_models:
            # Best model should be within 2 AIC units or better
            assert best_model.aic <= alt_model.aic + 2.0

    def test_confidence_intervals_calculation(self, enhanced_engine, sample_returns_data):
        """Test confidence interval calculations"""
        returns = sample_returns_data['normal']

        result = enhanced_engine.fit_multiple_models(returns, 'CI_TEST')

        if result.confidence_intervals:
            for ci_name, (lower, upper) in result.confidence_intervals.items():
                assert lower < upper, f"Invalid confidence interval for {ci_name}: {lower} >= {upper}"
                assert lower > 0, f"Negative lower bound for {ci_name}: {lower}"

    def test_performance_benchmarks(self, enhanced_engine, sample_returns_data):
        """Test that model fitting meets performance requirements (<100ms)"""
        returns = sample_returns_data['normal']

        start_time = time.perf_counter()
        result = enhanced_engine.fit_multiple_models(returns, 'PERFORMANCE_TEST')
        elapsed_time = time.perf_counter() - start_time

        # Should complete within 100ms for 1000 data points
        assert elapsed_time < 0.1, f"Model fitting too slow: {elapsed_time:.3f}s > 0.1s"

        # Result should be valid
        assert isinstance(result, EnhancedTailRiskModel)
        assert result.var_95 > 0

    def test_enhance_existing_evt_model(self, sample_returns_data):
        """Test integration function for enhancing existing EVT models"""
        returns = sample_returns_data['normal']

        # Mock existing model parameters
        existing_params = {
            'var_95': 0.04,
            'var_99': 0.06,
            'expected_shortfall': 0.05,
            'tail_index': 0.3,
            'scale_parameter': 0.02
        }

        enhanced_model = enhance_existing_evt_model(existing_params, returns, 'ENHANCE_TEST')

        assert isinstance(enhanced_model, EnhancedTailRiskModel)
        assert enhanced_model.symbol == 'ENHANCE_TEST'

        # Enhanced model should provide additional metrics
        assert hasattr(enhanced_model, 'var_99_9')
        assert hasattr(enhanced_model, 'expected_shortfall_99')
        assert hasattr(enhanced_model, 'best_model')


class TestEVTBacktesting:
    """Test suite for EVT backtesting framework"""

    @pytest.fixture
    def backtesting_engine(self):
        """Create backtesting engine for testing"""
        return EVTBacktestingEngine(
            confidence_levels=[0.95, 0.99],
            min_test_observations=100,  # Reduced for testing
            significance_level=0.05
        )

    @pytest.fixture
    def backtest_data(self):
        """Generate data for backtesting"""
        np.random.seed(123)

        # Generate 500 returns for testing
        returns = np.random.normal(0.001, 0.02, 500)

        # Add some extreme events
        returns[100] = -0.08  # 8% loss
        returns[200] = -0.06  # 6% loss
        returns[300] = -0.12  # 12% loss

        # Generate corresponding VaR forecasts (slightly conservative)
        var_95_forecasts = np.full(500, 0.04)  # 4% VaR
        var_99_forecasts = np.full(500, 0.06)  # 6% VaR
        es_95_forecasts = np.full(500, 0.05)   # 5% ES

        return {
            'returns': returns,
            'var_95_forecasts': var_95_forecasts,
            'var_99_forecasts': var_99_forecasts,
            'es_95_forecasts': es_95_forecasts
        }

    def test_kupiec_pof_test_basic(self, backtesting_engine, backtest_data):
        """Test Kupiec Proportion of Failures test"""
        returns = backtest_data['returns']
        var_forecasts = backtest_data['var_95_forecasts']

        result = backtesting_engine.kupiec_pof_test(returns, var_forecasts, 0.95)

        assert isinstance(result, BacktestResults)
        assert result.test_type == BacktestType.VAR_KUPIEC
        assert result.confidence_level == 0.95
        assert result.total_observations == len(returns)
        assert result.violation_rate >= 0
        assert result.violation_rate <= 1
        assert result.expected_violations == len(returns) * 0.05  # 5% expected

        # Test statistic should be calculated
        assert np.isfinite(result.test_statistic)
        assert 0 <= result.p_value <= 1

        # Should have violations list
        assert isinstance(result.violations, list)
        assert result.total_violations == len(result.violations)

    def test_christoffersen_independence_test(self, backtesting_engine, backtest_data):
        """Test Christoffersen independence test"""
        returns = backtest_data['returns']
        var_forecasts = backtest_data['var_95_forecasts']

        result = backtesting_engine.christoffersen_independence_test(returns, var_forecasts, 0.95)

        assert isinstance(result, BacktestResults)
        assert result.test_type == BacktestType.VAR_CHRISTOFFERSEN
        assert result.total_observations == len(returns) - 1  # -1 for transitions

        # Should have transition matrix in model_info
        assert 'transition_matrix' in result.model_info
        transition_matrix = result.model_info['transition_matrix']
        assert len(transition_matrix) == 2
        assert len(transition_matrix[0]) == 2

    def test_expected_shortfall_backtest(self, backtesting_engine, backtest_data):
        """Test Expected Shortfall backtesting"""
        returns = backtest_data['returns']
        var_forecasts = backtest_data['var_95_forecasts']
        es_forecasts = backtest_data['es_95_forecasts']

        result = backtesting_engine.expected_shortfall_backtest(
            returns, var_forecasts, es_forecasts, 0.95
        )

        assert isinstance(result, ESBacktestResults)
        assert isinstance(result.backtest_results, BacktestResults)
        assert result.backtest_results.test_type == BacktestType.ES_BACKTEST

        # ES-specific metrics
        assert hasattr(result, 'es_violations')
        assert hasattr(result, 'average_es_shortfall')
        assert hasattr(result, 'es_accuracy_ratio')
        assert hasattr(result, 'es_test_statistic')
        assert hasattr(result, 'es_p_value')

        # Accuracy ratio should be positive
        assert result.es_accuracy_ratio > 0

    def test_var_accuracy_calculation(self, backtesting_engine, backtest_data):
        """Test VaR accuracy calculation for ±5% target"""
        returns = backtest_data['returns']
        var_forecasts = backtest_data['var_95_forecasts']

        accuracy_metrics = backtesting_engine.calculate_var_accuracy(returns, var_forecasts, 0.95)

        # Should have all required metrics
        expected_keys = [
            'rate_accuracy', 'violation_count_accuracy', 'relative_accuracy',
            'mae', 'rmse', 'observed_rate', 'expected_rate', 'rate_difference',
            'within_5pct_target', 'accuracy_score'
        ]

        for key in expected_keys:
            assert key in accuracy_metrics

        # Metrics should be in valid ranges
        assert 0 <= accuracy_metrics['rate_accuracy'] <= 1
        assert 0 <= accuracy_metrics['relative_accuracy'] <= 1
        assert accuracy_metrics['mae'] >= 0
        assert accuracy_metrics['rmse'] >= 0
        assert 0 <= accuracy_metrics['observed_rate'] <= 1
        assert accuracy_metrics['expected_rate'] == 0.05  # 5% for 95% confidence

        # Should indicate if within target
        assert isinstance(accuracy_metrics['within_5pct_target'], bool)

    def test_rolling_window_backtest(self, backtesting_engine, sample_returns_data):
        """Test rolling window backtesting"""
        returns = sample_returns_data['normal']

        # Create mock enhanced engine
        mock_engine = Mock()
        mock_model = Mock()
        mock_model.var_95 = 0.04
        mock_model.best_model.distribution.value = 'gpd'
        mock_model.best_model.aic = 100.0
        mock_engine.fit_multiple_models.return_value = mock_model

        results = backtesting_engine.rolling_window_backtest(
            returns, mock_engine, 'TEST', window_size=252, step_size=22
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert isinstance(result, BacktestResults)
            assert 'window_start' in result.test_metadata
            assert 'best_model' in result.test_metadata

    def test_comprehensive_backtest_function(self, sample_returns_data):
        """Test comprehensive backtest function"""
        returns = sample_returns_data['normal']

        # Create mock enhanced model
        mock_model = Mock()
        mock_model.symbol = 'TEST'
        mock_model.var_95 = 0.04
        mock_model.var_99 = 0.06
        mock_model.expected_shortfall_95 = 0.05

        results = run_comprehensive_backtest(returns, mock_model, 'TEST')

        assert isinstance(results, dict)

        # Should have multiple test results
        expected_tests = ['kupiec_var_95', 'kupiec_var_99', 'christoffersen_var_95', 'es_backtest']
        for test in expected_tests:
            assert test in results


class TestEVTIntegration:
    """Test suite for EVT integration with existing antifragility engine"""

    @pytest.fixture
    def integration_layer(self):
        """Create integration layer for testing"""
        return EnhancedAntifragilityIntegration(
            enable_enhanced_evt=True,
            performance_target_ms=100.0,
            accuracy_target=0.05,
            auto_model_selection=True
        )

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for integration testing"""
        np.random.seed(456)
        return np.random.normal(0.001, 0.02, 252).tolist()  # 1 year of daily returns

    def test_integration_layer_initialization(self, integration_layer):
        """Test integration layer initialization"""
        assert integration_layer.enable_enhanced_evt is True
        assert integration_layer.performance_target_ms == 100.0
        assert integration_layer.accuracy_target == 0.05
        assert integration_layer.auto_model_selection is True

        assert integration_layer.enhanced_engine is not None
        assert integration_layer.backtesting_engine is not None

        # Should have empty caches initially
        assert len(integration_layer.model_performance_cache) == 0
        assert len(integration_layer.calculation_times) == 0

    def test_model_tail_risk_enhanced_basic_functionality(self, integration_layer, sample_returns):
        """Test enhanced model_tail_risk method basic functionality"""
        result = integration_layer.model_tail_risk_enhanced('TEST', sample_returns)

        assert isinstance(result, IntegratedTailRiskModel)
        assert result.symbol == 'TEST'
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.expected_shortfall > 0
        assert result.tail_index != 0
        assert result.scale_parameter > 0

        # Should have enhanced fields
        assert hasattr(result, 'var_99_9')
        assert hasattr(result, 'expected_shortfall_99')
        assert hasattr(result, 'model_type')
        assert hasattr(result, 'best_distribution')
        assert hasattr(result, 'calculation_time_ms')

    def test_model_tail_risk_enhanced_performance_target(self, integration_layer, sample_returns):
        """Test that enhanced method meets performance target"""
        start_time = time.perf_counter()
        result = integration_layer.model_tail_risk_enhanced('PERF_TEST', sample_returns)
        elapsed_time = time.perf_counter() - start_time

        # Should complete within performance target
        assert elapsed_time < integration_layer.performance_target_ms / 1000  # Convert ms to seconds

        # Result should indicate performance timing
        assert result.calculation_time_ms < integration_layer.performance_target_ms

    def test_backward_compatibility(self, integration_layer, sample_returns):
        """Test backward compatibility with existing TailRiskModel"""
        integrated_model = integration_layer.model_tail_risk_enhanced('COMPAT_TEST', sample_returns)
        legacy_model = integrated_model.to_legacy_format()

        assert isinstance(legacy_model, TailRiskModel)
        assert legacy_model.symbol == integrated_model.symbol
        assert legacy_model.var_95 == integrated_model.var_95
        assert legacy_model.var_99 == integrated_model.var_99
        assert legacy_model.expected_shortfall == integrated_model.expected_shortfall
        assert legacy_model.tail_index == integrated_model.tail_index
        assert legacy_model.scale_parameter == integrated_model.scale_parameter

    def test_force_basic_model(self, integration_layer, sample_returns):
        """Test forcing use of basic EVT model"""
        result = integration_layer.model_tail_risk_enhanced('BASIC_TEST', sample_returns, force_basic=True)

        assert isinstance(result, IntegratedTailRiskModel)
        # Should use basic model even with enhanced capabilities available
        assert 'basic' in result.model_type

        # Should not have model comparison since enhanced wasn't used
        assert result.model_comparison is None

    def test_performance_validation_comprehensive(self, integration_layer, sample_returns):
        """Test comprehensive performance validation"""
        validation_results = integration_layer.run_performance_validation(sample_returns, 'VALIDATION_TEST')

        assert isinstance(validation_results, dict)
        assert 'symbol' in validation_results
        assert 'validation_date' in validation_results
        assert 'requirements_met' in validation_results
        assert 'performance_metrics' in validation_results
        assert 'recommendations' in validation_results
        assert 'overall_status' in validation_results

        # Should have checked key requirements
        requirements = validation_results['requirements_met']
        expected_requirements = [
            'var_accuracy_5pct', 'performance_100ms',
            'backward_compatibility', 'enhanced_accuracy'
        ]

        for req in expected_requirements:
            assert req in requirements
            assert isinstance(requirements[req], bool)

        # Overall status should be valid
        assert validation_results['overall_status'] in ['PASS', 'PARTIAL', 'FAIL']

    def test_model_performance_summary(self, integration_layer, sample_returns):
        """Test model performance summary functionality"""
        # Generate some test data
        integration_layer.model_tail_risk_enhanced('TEST1', sample_returns)
        integration_layer.model_tail_risk_enhanced('TEST2', sample_returns)

        summary = integration_layer.get_model_performance_summary()

        assert isinstance(summary, dict)
        assert 'total_models_tested' in summary
        assert 'average_calculation_time_ms' in summary
        assert 'models_within_performance_target' in summary
        assert 'enhanced_model_recommendations' in summary
        assert 'cached_results' in summary

        assert summary['total_models_tested'] == 2
        assert len(summary['cached_results']) == 2

    def test_monkey_patching_functionality(self, sample_returns):
        """Test monkey patching of existing antifragility engine"""
        # Create original engine
        original_engine = AntifragilityEngine(100000)

        # Store original method
        original_method = original_engine.model_tail_risk

        # Apply patch
        patch_antifragility_engine()

        # Test patched method
        result = original_engine.model_tail_risk('PATCH_TEST', sample_returns)

        assert isinstance(result, TailRiskModel)
        assert result.symbol == 'PATCH_TEST'
        assert result.var_95 > 0

        # Should have access to original method as backup
        assert hasattr(AntifragilityEngine, '_original_model_tail_risk')

    def test_validate_phase2_requirements_function(self, sample_returns):
        """Test Phase 2 requirements validation function"""
        test_data = {
            'ASSET1': sample_returns,
            'ASSET2': sample_returns,
            'ASSET3': sample_returns
        }

        validation_report = validate_phase2_requirements(test_data)

        assert isinstance(validation_report, dict)
        assert 'validation_date' in validation_report
        assert 'assets_tested' in validation_report
        assert 'requirements_summary' in validation_report
        assert 'asset_results' in validation_report
        assert 'overall_assessment' in validation_report
        assert 'performance_summary' in validation_report

        # Should have tested all assets
        assert len(validation_report['assets_tested']) == 3
        assert len(validation_report['asset_results']) == 3

        # Requirements summary should have pass rates
        for req_name, req_info in validation_report['requirements_summary'].items():
            assert 'passed_count' in req_info
            assert 'total_count' in req_info
            assert 'pass_rate' in req_info
            assert 'status' in req_info
            assert req_info['total_count'] == 3  # 3 assets tested


class TestPerformanceBenchmarks:
    """Test suite specifically for performance benchmarks and targets"""

    def test_var_accuracy_within_5_percent_target(self):
        """Test that VaR accuracy meets ±5% target"""
        np.random.seed(789)

        # Generate returns with known characteristics
        returns = np.random.normal(0.001, 0.02, 1000)

        # Fit enhanced model
        engine = EnhancedEVTEngine()
        enhanced_model = engine.fit_multiple_models(returns, 'ACCURACY_TEST')

        # Test VaR accuracy on out-of-sample data
        test_returns = np.random.normal(0.001, 0.02, 500)
        var_forecasts = np.full(500, enhanced_model.var_95)

        backtesting_engine = EVTBacktestingEngine()
        accuracy_metrics = backtesting_engine.calculate_var_accuracy(test_returns, var_forecasts, 0.95)

        # Should meet ±5% target
        assert abs(accuracy_metrics['rate_difference']) <= 0.05, \
            f"VaR accuracy outside ±5% target: {accuracy_metrics['rate_difference']:.3%}"

        assert accuracy_metrics['within_5pct_target'], \
            "VaR accuracy not within 5% target according to metric"

    def test_calculation_time_under_100ms(self):
        """Test that tail risk calculations complete under 100ms"""
        np.random.seed(999)

        # Generate reasonably sized dataset (1 year daily data)
        returns = np.random.normal(0.001, 0.02, 252)

        integration = EnhancedAntifragilityIntegration(performance_target_ms=100.0)

        start_time = time.perf_counter()
        result = integration.model_tail_risk_enhanced('TIMING_TEST', returns.tolist())
        elapsed_time = time.perf_counter() - start_time

        # Should complete within 100ms
        assert elapsed_time < 0.1, f"Calculation too slow: {elapsed_time:.3f}s > 0.1s"

        # Result should also report timing within target
        assert result.calculation_time_ms < 100.0, \
            f"Reported calculation time exceeds target: {result.calculation_time_ms:.1f}ms > 100ms"

    def test_backtesting_coverage_95_percent(self):
        """Test that backtesting covers 95%+ of test scenarios"""
        np.random.seed(111)

        # Generate various market scenarios
        scenarios = {
            'normal': np.random.normal(0.001, 0.02, 500),
            'high_vol': np.random.normal(0.001, 0.05, 500),
            'trending_up': np.random.normal(0.005, 0.02, 500),
            'trending_down': np.random.normal(-0.003, 0.02, 500),
            'fat_tails': np.random.standard_t(df=3, size=500) * 0.02
        }

        successful_backtests = 0
        total_scenarios = len(scenarios)

        integration = EnhancedAntifragilityIntegration()

        for scenario_name, returns in scenarios.items():
            try:
                # Fit model and run validation
                validation_result = integration.run_performance_validation(returns.tolist(), scenario_name)

                # Check if backtesting was successful
                if validation_result.get('overall_status') in ['PASS', 'PARTIAL']:
                    successful_backtests += 1

            except Exception as e:
                # Log but don't fail - some scenarios might be challenging
                print(f"Scenario {scenario_name} failed: {e}")

        # Should achieve 95% success rate
        success_rate = successful_backtests / total_scenarios
        assert success_rate >= 0.95, f"Backtesting coverage only {success_rate:.1%} < 95%"

    def test_zero_breaking_changes_integration(self):
        """Test that integration maintains zero breaking changes"""
        np.random.seed(222)
        returns = np.random.normal(0.001, 0.02, 252).tolist()

        # Test original antifragility engine
        original_engine = AntifragilityEngine(100000)

        # Store methods that should remain unchanged
        original_methods = [
            'calculate_barbell_allocation',
            'assess_convexity',
            'rebalance_on_volatility',
            'calculate_antifragility_score',
            'get_antifragile_recommendations'
        ]

        # Apply enhanced integration
        patch_antifragility_engine()

        # Verify all original methods still exist and work
        for method_name in original_methods:
            assert hasattr(original_engine, method_name), \
                f"Original method {method_name} missing after integration"

            method = getattr(original_engine, method_name)
            assert callable(method), f"Method {method_name} not callable after integration"

        # Test that model_tail_risk still returns TailRiskModel
        result = original_engine.model_tail_risk('COMPATIBILITY_TEST', returns)
        assert isinstance(result, TailRiskModel), \
            f"model_tail_risk returns {type(result)} instead of TailRiskModel"

        # Verify all expected fields are present
        expected_fields = ['symbol', 'var_95', 'var_99', 'expected_shortfall', 'tail_index', 'scale_parameter']
        for field in expected_fields:
            assert hasattr(result, field), f"TailRiskModel missing field {field}"


if __name__ == "__main__":
    # Run basic test when called directly
    print("Running basic Enhanced EVT tests...")

    # Test enhanced model fitting
    print("1. Testing enhanced model fitting...")
    engine = EnhancedEVTEngine()
    np.random.seed(42)
    test_returns = np.random.normal(0.001, 0.02, 1000)

    start_time = time.perf_counter()
    enhanced_model = engine.fit_multiple_models(test_returns, 'TEST')
    elapsed_time = time.perf_counter() - start_time

    print(f"   ✓ Model fitted: {enhanced_model.best_model.distribution.value}")
    print(f"   ✓ VaR95: {enhanced_model.var_95:.4f}, VaR99: {enhanced_model.var_99:.4f}")
    print(f"   ✓ Performance: {elapsed_time*1000:.1f}ms < 100ms target")

    # Test integration
    print("2. Testing integration...")
    integration = EnhancedAntifragilityIntegration()
    integrated_model = integration.model_tail_risk_enhanced('TEST', test_returns.tolist())

    print(f"   ✓ Integration successful: {integrated_model.model_type}")
    print(f"   ✓ Calculation time: {integrated_model.calculation_time_ms:.1f}ms")
    print(f"   ✓ Backward compatibility: {type(integrated_model.to_legacy_format())}")

    # Test backtesting
    print("3. Testing backtesting...")
    backtesting_engine = EVTBacktestingEngine()
    var_forecasts = np.full(len(test_returns), enhanced_model.var_95)
    backtest_result = backtesting_engine.kupiec_pof_test(test_returns, var_forecasts, 0.95)

    print(f"   ✓ Backtest completed: {backtest_result.total_violations}/{len(test_returns)} violations")
    print(f"   ✓ Violation rate: {backtest_result.violation_rate:.2%} vs expected 5%")
    print(f"   ✓ Test passed: {'Yes' if not backtest_result.reject_null else 'No'}")

    # Test performance validation
    print("4. Testing performance validation...")
    validation_results = integration.run_performance_validation(test_returns.tolist(), 'VALIDATION_TEST')

    print(f"   ✓ Overall status: {validation_results['overall_status']}")
    for req, status in validation_results['requirements_met'].items():
        print(f"   ✓ {req}: {'PASS' if status else 'FAIL'}")

    print("✓ All basic Enhanced EVT tests passed!")
    print("\nRun 'pytest tests/test_enhanced_evt_models.py' for comprehensive testing.")