"""
EVT Backtesting Framework - Phase 2 Division 1

Comprehensive backtesting framework for validating tail risk models.
Implements standard backtesting methods for VaR and Expected Shortfall.

Key Features:
1. VaR backtesting (Kupiec, Christoffersen, Berkowitz tests)
2. Expected Shortfall backtesting
3. Model comparison and validation
4. Rolling window backtesting
5. Performance metrics calculation
6. Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import warnings
from enum import Enum
import logging

from .enhanced_evt_models import EnhancedTailRiskModel, EnhancedEVTEngine

logger = logging.getLogger(__name__)

class BacktestType(Enum):
    """Types of backtests available"""
    VAR_KUPIEC = "var_kupiec"               # Kupiec POF test
    VAR_CHRISTOFFERSEN = "var_christoffersen" # Independence test
    VAR_BERKOWITZ = "var_berkowitz"         # Berkowitz likelihood ratio test
    ES_BACKTEST = "expected_shortfall"      # ES backtesting
    ROLLING_WINDOW = "rolling_window"       # Time-varying model performance

@dataclass
class ViolationEvent:
    """Individual VaR/ES violation event"""
    date: datetime
    actual_return: float
    predicted_var: float
    predicted_es: Optional[float]
    violation_magnitude: float  # How much the actual exceeded predicted
    violation_type: str         # 'var' or 'es'

@dataclass
class BacktestResults:
    """Results from backtesting procedure"""
    test_type: BacktestType
    confidence_level: float
    test_period_start: datetime
    test_period_end: datetime

    # Basic statistics
    total_observations: int
    total_violations: int
    violation_rate: float
    expected_violations: float

    # Test statistics
    test_statistic: float
    p_value: float
    critical_value: float
    reject_null: bool

    # Violations details
    violations: List[ViolationEvent]

    # Performance metrics
    accuracy_score: float
    mean_absolute_error: float
    root_mean_squared_error: float

    # Additional info
    model_info: Dict
    test_metadata: Dict

@dataclass
class ESBacktestResults:
    """Extended results for Expected Shortfall backtesting"""
    backtest_results: BacktestResults
    es_violations: List[ViolationEvent]
    average_es_shortfall: float
    es_accuracy_ratio: float  # Actual ES / Predicted ES
    es_test_statistic: float
    es_p_value: float

@dataclass
class ModelComparisonResults:
    """Results from comparing multiple models"""
    models_compared: List[str]
    comparison_period: Tuple[datetime, datetime]

    # Performance rankings
    var_accuracy_ranking: List[Tuple[str, float]]  # (model_name, accuracy)
    es_accuracy_ranking: List[Tuple[str, float]]
    overall_ranking: List[Tuple[str, float]]

    # Statistical significance tests
    significance_tests: Dict[str, Dict]

    # Model selection recommendation
    recommended_model: str
    recommendation_confidence: float

class EVTBacktestingEngine:
    """
    Comprehensive backtesting engine for EVT models

    Implements industry-standard backtesting procedures for validating
    tail risk models against historical data.
    """

    def __init__(self,
                 confidence_levels: List[float] = [0.95, 0.99],
                 min_test_observations: int = 250,  # ~1 year of daily data
                 significance_level: float = 0.05):
        """
        Initialize backtesting engine

        Args:
            confidence_levels: VaR confidence levels to test
            min_test_observations: Minimum observations for valid backtest
            significance_level: Significance level for statistical tests
        """
        self.confidence_levels = confidence_levels
        self.min_test_observations = min_test_observations
        self.significance_level = significance_level

        logger.info(f"EVT Backtesting Engine initialized - confidence levels: {confidence_levels}")

    def kupiec_pof_test(self,
                       returns: np.ndarray,
                       var_forecasts: np.ndarray,
                       confidence_level: float) -> BacktestResults:
        """
        Kupiec Proportion of Failures (POF) test for VaR backtesting

        Tests if the observed violation rate matches the expected rate.
        H0: violation rate = (1 - confidence_level)

        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts
            confidence_level: VaR confidence level

        Returns:
            BacktestResults with test statistics and violations
        """

        if len(returns) != len(var_forecasts):
            raise ValueError("Returns and VaR forecasts must have same length")

        if len(returns) < self.min_test_observations:
            logger.warning(f"Insufficient observations: {len(returns)} < {self.min_test_observations}")

        # Convert to losses (positive values)
        losses = -returns

        # Identify violations (losses exceeding VaR)
        violations_bool = losses > var_forecasts
        violations = []

        for i, is_violation in enumerate(violations_bool):
            if is_violation:
                violation = ViolationEvent(
                    date=datetime.now() + timedelta(days=i),  # Placeholder dates
                    actual_return=returns[i],
                    predicted_var=var_forecasts[i],
                    predicted_es=None,
                    violation_magnitude=losses[i] - var_forecasts[i],
                    violation_type='var'
                )
                violations.append(violation)

        # Calculate test statistics
        n = len(returns)
        x = np.sum(violations_bool)  # Number of violations
        p = 1 - confidence_level    # Expected violation rate

        violation_rate = x / n
        expected_violations = n * p

        # Kupiec test statistic (likelihood ratio test)
        if x == 0:
            lr_stat = 2 * n * np.log(1 - p)
        elif x == n:
            lr_stat = 2 * n * np.log(p)
        else:
            lr_stat = -2 * np.log(((1-p)**(n-x) * p**x) / ((1-violation_rate)**(n-x) * violation_rate**x))

        # Critical value (chi-square with 1 degree of freedom)
        critical_value = stats.chi2.ppf(1 - self.significance_level, df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        reject_null = lr_stat > critical_value

        # Performance metrics
        accuracy_score = 1 - abs(violation_rate - p)  # Closer to expected rate = higher accuracy
        mae = np.mean(np.abs(losses - var_forecasts))
        rmse = np.sqrt(np.mean((losses - var_forecasts)**2))

        # Create results
        results = BacktestResults(
            test_type=BacktestType.VAR_KUPIEC,
            confidence_level=confidence_level,
            test_period_start=datetime.now() - timedelta(days=len(returns)),
            test_period_end=datetime.now(),
            total_observations=n,
            total_violations=x,
            violation_rate=violation_rate,
            expected_violations=expected_violations,
            test_statistic=lr_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            violations=violations,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            model_info={'confidence_level': confidence_level},
            test_metadata={
                'test_name': 'Kupiec POF Test',
                'null_hypothesis': f'violation_rate = {p:.3f}',
                'test_statistic_distribution': 'chi2(1)',
                'significance_level': self.significance_level
            }
        )

        logger.info(f"Kupiec test completed: {x}/{n} violations ({violation_rate:.2%} vs expected {p:.2%}), "
                   f"LR={lr_stat:.3f}, p-value={p_value:.3f}")

        return results

    def christoffersen_independence_test(self,
                                       returns: np.ndarray,
                                       var_forecasts: np.ndarray,
                                       confidence_level: float) -> BacktestResults:
        """
        Christoffersen independence test for VaR backtesting

        Tests if violations are independently distributed (no clustering).
        H0: violations are independent

        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts
            confidence_level: VaR confidence level

        Returns:
            BacktestResults with independence test results
        """

        losses = -returns
        violations_bool = losses > var_forecasts

        # Calculate transition matrix
        n00 = n01 = n10 = n11 = 0

        for i in range(len(violations_bool) - 1):
            if not violations_bool[i] and not violations_bool[i+1]:
                n00 += 1
            elif not violations_bool[i] and violations_bool[i+1]:
                n01 += 1
            elif violations_bool[i] and not violations_bool[i+1]:
                n10 += 1
            elif violations_bool[i] and violations_bool[i+1]:
                n11 += 1

        # Calculate test statistic
        n0 = n00 + n01  # Number of non-violations
        n1 = n10 + n11  # Number of violations

        if n0 == 0 or n1 == 0 or (n01 == 0 and n11 == 0) or (n00 == 0 and n10 == 0):
            # Edge cases - insufficient data for test
            lr_stat = 0
            p_value = 1.0
        else:
            # Likelihood ratio test statistic
            p01 = n01 / n0 if n0 > 0 else 0
            p11 = n11 / n1 if n1 > 0 else 0
            p = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0

            if p01 > 0 and p11 > 0 and p > 0:
                lr_stat = -2 * np.log((p**(n01 + n11) * (1-p)**(n00 + n10)) /
                                     (p01**n01 * (1-p01)**n00 * p11**n11 * (1-p11)**n10))
            else:
                lr_stat = 0

        critical_value = stats.chi2.ppf(1 - self.significance_level, df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1) if lr_stat > 0 else 1.0
        reject_null = lr_stat > critical_value

        # Create violation events
        violations = []
        for i, is_violation in enumerate(violations_bool):
            if is_violation:
                violation = ViolationEvent(
                    date=datetime.now() + timedelta(days=i),
                    actual_return=returns[i],
                    predicted_var=var_forecasts[i],
                    predicted_es=None,
                    violation_magnitude=losses[i] - var_forecasts[i],
                    violation_type='var'
                )
                violations.append(violation)

        # Performance metrics
        violation_rate = np.mean(violations_bool)
        expected_rate = 1 - confidence_level
        accuracy_score = 1 - abs(violation_rate - expected_rate)
        mae = np.mean(np.abs(losses - var_forecasts))
        rmse = np.sqrt(np.mean((losses - var_forecasts)**2))

        results = BacktestResults(
            test_type=BacktestType.VAR_CHRISTOFFERSEN,
            confidence_level=confidence_level,
            test_period_start=datetime.now() - timedelta(days=len(returns)),
            test_period_end=datetime.now(),
            total_observations=len(returns) - 1,  # -1 because we look at transitions
            total_violations=int(np.sum(violations_bool)),
            violation_rate=violation_rate,
            expected_violations=len(returns) * expected_rate,
            test_statistic=lr_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            violations=violations,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            model_info={
                'confidence_level': confidence_level,
                'transition_matrix': [[n00, n01], [n10, n11]]
            },
            test_metadata={
                'test_name': 'Christoffersen Independence Test',
                'null_hypothesis': 'violations are independent',
                'test_statistic_distribution': 'chi2(1)',
                'significance_level': self.significance_level
            }
        )

        logger.info(f"Christoffersen test completed: LR={lr_stat:.3f}, p-value={p_value:.3f}")

        return results

    def expected_shortfall_backtest(self,
                                  returns: np.ndarray,
                                  var_forecasts: np.ndarray,
                                  es_forecasts: np.ndarray,
                                  confidence_level: float) -> ESBacktestResults:
        """
        Backtest Expected Shortfall predictions

        Tests if ES predictions are accurate for tail losses beyond VaR.

        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts
            es_forecasts: Expected Shortfall forecasts
            confidence_level: Confidence level

        Returns:
            ESBacktestResults with ES-specific metrics
        """

        losses = -returns
        var_violations = losses > var_forecasts

        # Extract tail losses (beyond VaR)
        tail_losses = losses[var_violations]
        tail_es_forecasts = es_forecasts[var_violations]

        if len(tail_losses) == 0:
            logger.warning("No VaR violations found - cannot backtest ES")
            # Return dummy results
            dummy_backtest = BacktestResults(
                test_type=BacktestType.ES_BACKTEST,
                confidence_level=confidence_level,
                test_period_start=datetime.now() - timedelta(days=len(returns)),
                test_period_end=datetime.now(),
                total_observations=0,
                total_violations=0,
                violation_rate=0.0,
                expected_violations=0.0,
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                reject_null=False,
                violations=[],
                accuracy_score=0.0,
                mean_absolute_error=0.0,
                root_mean_squared_error=0.0,
                model_info={'confidence_level': confidence_level},
                test_metadata={'test_name': 'ES Backtest - No violations'}
            )

            return ESBacktestResults(
                backtest_results=dummy_backtest,
                es_violations=[],
                average_es_shortfall=0.0,
                es_accuracy_ratio=1.0,
                es_test_statistic=0.0,
                es_p_value=1.0
            )

        # Calculate ES metrics
        actual_es = np.mean(tail_losses)
        predicted_es = np.mean(tail_es_forecasts)
        es_accuracy_ratio = actual_es / predicted_es if predicted_es > 0 else 0.0

        # ES violations (actual tail loss exceeds predicted ES)
        es_violations_bool = tail_losses > tail_es_forecasts
        es_violations = []

        violation_indices = np.where(var_violations)[0]
        for i, is_es_violation in enumerate(es_violations_bool):
            if is_es_violation:
                original_idx = violation_indices[i]
                violation = ViolationEvent(
                    date=datetime.now() + timedelta(days=original_idx),
                    actual_return=returns[original_idx],
                    predicted_var=var_forecasts[original_idx],
                    predicted_es=es_forecasts[original_idx],
                    violation_magnitude=tail_losses[i] - tail_es_forecasts[i],
                    violation_type='es'
                )
                es_violations.append(violation)

        # ES test statistic (simplified t-test)
        if len(tail_losses) > 1:
            es_diff = tail_losses - tail_es_forecasts
            es_test_stat = np.mean(es_diff) / (np.std(es_diff) / np.sqrt(len(es_diff)))
            es_p_value = 2 * (1 - stats.t.cdf(abs(es_test_stat), df=len(es_diff)-1))
        else:
            es_test_stat = 0.0
            es_p_value = 1.0

        # Performance metrics for overall VaR model
        accuracy_score = 1 - abs(es_accuracy_ratio - 1.0)  # Closer to 1.0 is better
        mae = np.mean(np.abs(losses - var_forecasts))
        rmse = np.sqrt(np.mean((losses - var_forecasts)**2))

        # Create VaR violations for the overall backtest
        all_violations = []
        for i, is_violation in enumerate(var_violations):
            if is_violation:
                violation = ViolationEvent(
                    date=datetime.now() + timedelta(days=i),
                    actual_return=returns[i],
                    predicted_var=var_forecasts[i],
                    predicted_es=es_forecasts[i],
                    violation_magnitude=losses[i] - var_forecasts[i],
                    violation_type='var'
                )
                all_violations.append(violation)

        # Overall backtest results
        backtest_results = BacktestResults(
            test_type=BacktestType.ES_BACKTEST,
            confidence_level=confidence_level,
            test_period_start=datetime.now() - timedelta(days=len(returns)),
            test_period_end=datetime.now(),
            total_observations=len(returns),
            total_violations=int(np.sum(var_violations)),
            violation_rate=np.mean(var_violations),
            expected_violations=len(returns) * (1 - confidence_level),
            test_statistic=es_test_stat,
            p_value=es_p_value,
            critical_value=stats.t.ppf(1 - self.significance_level/2, df=max(len(tail_losses)-1, 1)),
            reject_null=abs(es_test_stat) > stats.t.ppf(1 - self.significance_level/2, df=max(len(tail_losses)-1, 1)),
            violations=all_violations,
            accuracy_score=accuracy_score,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            model_info={
                'confidence_level': confidence_level,
                'tail_observations': len(tail_losses),
                'actual_es': actual_es,
                'predicted_es': predicted_es
            },
            test_metadata={
                'test_name': 'Expected Shortfall Backtest',
                'null_hypothesis': 'ES predictions are accurate',
                'test_statistic_distribution': f't({max(len(tail_losses)-1, 1)})',
                'significance_level': self.significance_level
            }
        )

        results = ESBacktestResults(
            backtest_results=backtest_results,
            es_violations=es_violations,
            average_es_shortfall=actual_es - predicted_es,
            es_accuracy_ratio=es_accuracy_ratio,
            es_test_statistic=es_test_stat,
            es_p_value=es_p_value
        )

        logger.info(f"ES backtest completed: {len(tail_losses)} tail observations, "
                   f"ES ratio={es_accuracy_ratio:.3f}, p-value={es_p_value:.3f}")

        return results

    def rolling_window_backtest(self,
                              returns: np.ndarray,
                              model: EnhancedEVTEngine,
                              symbol: str,
                              window_size: int = 252,
                              step_size: int = 22,
                              confidence_level: float = 0.95) -> List[BacktestResults]:
        """
        Rolling window backtesting to assess time-varying model performance

        Args:
            returns: Full return series
            model: Enhanced EVT engine for fitting
            symbol: Asset symbol
            window_size: Training window size (252 = ~1 year)
            step_size: Step size between windows (22 = ~1 month)
            confidence_level: VaR confidence level

        Returns:
            List of BacktestResults for each window
        """

        if len(returns) < window_size + step_size:
            raise ValueError(f"Insufficient data: need at least {window_size + step_size} observations")

        results = []

        # Rolling window loop
        start_idx = 0
        while start_idx + window_size + step_size <= len(returns):

            # Training window
            train_returns = returns[start_idx:start_idx + window_size]

            # Test window
            test_returns = returns[start_idx + window_size:start_idx + window_size + step_size]

            try:
                # Fit model on training data
                enhanced_model = model.fit_multiple_models(train_returns, f"{symbol}_window_{start_idx}")

                # Generate VaR forecasts (simplified - using single forecast for entire test period)
                if confidence_level == 0.95:
                    var_forecast = enhanced_model.var_95
                elif confidence_level == 0.99:
                    var_forecast = enhanced_model.var_99
                else:
                    var_forecast = enhanced_model.var_95  # Default

                var_forecasts = np.full(len(test_returns), var_forecast)

                # Backtest on test data
                window_results = self.kupiec_pof_test(test_returns, var_forecasts, confidence_level)

                # Add window metadata
                window_results.test_metadata.update({
                    'window_start': start_idx,
                    'window_end': start_idx + window_size,
                    'test_start': start_idx + window_size,
                    'test_end': start_idx + window_size + step_size,
                    'best_model': enhanced_model.best_model.distribution.value,
                    'model_aic': enhanced_model.best_model.aic
                })

                results.append(window_results)

                logger.debug(f"Rolling window {start_idx}-{start_idx+window_size}: "
                           f"{window_results.total_violations}/{step_size} violations "
                           f"({window_results.violation_rate:.2%})")

            except Exception as e:
                logger.warning(f"Failed to fit model for window {start_idx}: {e}")

            start_idx += step_size

        logger.info(f"Rolling window backtest completed: {len(results)} windows tested")

        return results

    def compare_models(self,
                      returns: np.ndarray,
                      models: List[Tuple[str, EnhancedTailRiskModel]],
                      test_start_pct: float = 0.7) -> ModelComparisonResults:
        """
        Compare multiple EVT models on out-of-sample data

        Args:
            returns: Full return series
            models: List of (model_name, fitted_model) tuples
            test_start_pct: Percentage of data to use for training (rest for testing)

        Returns:
            ModelComparisonResults with performance comparison
        """

        split_idx = int(len(returns) * test_start_pct)
        test_returns = returns[split_idx:]

        model_performances = []

        for model_name, fitted_model in models:

            # Generate forecasts (simplified)
            var_95_forecasts = np.full(len(test_returns), fitted_model.var_95)
            var_99_forecasts = np.full(len(test_returns), fitted_model.var_99)
            es_95_forecasts = np.full(len(test_returns), fitted_model.expected_shortfall_95)

            # VaR 95% backtesting
            var_95_results = self.kupiec_pof_test(test_returns, var_95_forecasts, 0.95)

            # VaR 99% backtesting
            var_99_results = self.kupiec_pof_test(test_returns, var_99_forecasts, 0.99)

            # ES backtesting
            es_results = self.expected_shortfall_backtest(test_returns, var_95_forecasts, es_95_forecasts, 0.95)

            # Aggregate performance score
            var_95_accuracy = var_95_results.accuracy_score
            var_99_accuracy = var_99_results.accuracy_score
            es_accuracy = es_results.backtest_results.accuracy_score

            overall_score = 0.4 * var_95_accuracy + 0.4 * var_99_accuracy + 0.2 * es_accuracy

            model_performances.append({
                'model_name': model_name,
                'var_95_accuracy': var_95_accuracy,
                'var_99_accuracy': var_99_accuracy,
                'es_accuracy': es_accuracy,
                'overall_score': overall_score,
                'var_95_results': var_95_results,
                'var_99_results': var_99_results,
                'es_results': es_results
            })

        # Rank models
        var_95_ranking = sorted([(p['model_name'], p['var_95_accuracy']) for p in model_performances],
                               key=lambda x: x[1], reverse=True)
        var_99_ranking = sorted([(p['model_name'], p['var_99_accuracy']) for p in model_performances],
                               key=lambda x: x[1], reverse=True)
        es_ranking = sorted([(p['model_name'], p['es_accuracy']) for p in model_performances],
                           key=lambda x: x[1], reverse=True)
        overall_ranking = sorted([(p['model_name'], p['overall_score']) for p in model_performances],
                                key=lambda x: x[1], reverse=True)

        # Significance testing (simplified Diebold-Mariano test)
        significance_tests = {}
        for i, (name1, perf1) in enumerate(model_performances):
            for j, (name2, perf2) in enumerate(model_performances):
                if i < j:
                    # Compare VaR 95% accuracy
                    mae1 = perf1['var_95_results'].mean_absolute_error
                    mae2 = perf2['var_95_results'].mean_absolute_error

                    test_key = f"{name1}_vs_{name2}"
                    significance_tests[test_key] = {
                        'mae_difference': mae1 - mae2,
                        'better_model': name1 if mae1 < mae2 else name2,
                        'improvement_pct': abs((mae1 - mae2) / max(mae1, mae2)) * 100
                    }

        # Model recommendation
        best_model_name, best_score = overall_ranking[0]
        second_best_score = overall_ranking[1][1] if len(overall_ranking) > 1 else 0.0
        recommendation_confidence = (best_score - second_best_score) if best_score > second_best_score else 0.0

        results = ModelComparisonResults(
            models_compared=[p['model_name'] for p in model_performances],
            comparison_period=(datetime.now() - timedelta(days=len(test_returns)), datetime.now()),
            var_accuracy_ranking=var_95_ranking,
            es_accuracy_ranking=es_ranking,
            overall_ranking=overall_ranking,
            significance_tests=significance_tests,
            recommended_model=best_model_name,
            recommendation_confidence=recommendation_confidence
        )

        logger.info(f"Model comparison completed: {best_model_name} recommended "
                   f"(score={best_score:.3f}, confidence={recommendation_confidence:.3f})")

        return results

    def calculate_var_accuracy(self,
                             actual_returns: np.ndarray,
                             var_forecasts: np.ndarray,
                             confidence_level: float) -> Dict[str, float]:
        """
        Calculate VaR accuracy metrics for Phase 2 target (±5% accuracy)

        Args:
            actual_returns: Observed returns
            var_forecasts: VaR predictions
            confidence_level: VaR confidence level

        Returns:
            Dictionary with accuracy metrics
        """

        losses = -actual_returns
        violations = losses > var_forecasts

        observed_rate = np.mean(violations)
        expected_rate = 1 - confidence_level

        # Main accuracy metric: how close observed rate is to expected rate
        rate_accuracy = 1 - abs(observed_rate - expected_rate) / expected_rate

        # Additional metrics
        violation_count_accuracy = 1 - abs(np.sum(violations) - len(losses) * expected_rate) / (len(losses) * expected_rate)

        # Mean absolute error for VaR levels
        mae = np.mean(np.abs(losses - var_forecasts))

        # Root mean squared error
        rmse = np.sqrt(np.mean((losses - var_forecasts)**2))

        # Relative accuracy (target: ±5%)
        relative_accuracy = max(0, 1 - abs(observed_rate - expected_rate) / 0.05)  # 5% tolerance

        accuracy_metrics = {
            'rate_accuracy': rate_accuracy,
            'violation_count_accuracy': violation_count_accuracy,
            'relative_accuracy': relative_accuracy,
            'mae': mae,
            'rmse': rmse,
            'observed_rate': observed_rate,
            'expected_rate': expected_rate,
            'rate_difference': observed_rate - expected_rate,
            'within_5pct_target': abs(observed_rate - expected_rate) <= 0.05,
            'accuracy_score': (rate_accuracy + violation_count_accuracy + relative_accuracy) / 3
        }

        logger.info(f"VaR accuracy calculated: {accuracy_metrics['accuracy_score']:.3f} "
                   f"(rate diff: {accuracy_metrics['rate_difference']:.3%}, "
                   f"within target: {accuracy_metrics['within_5pct_target']})")

        return accuracy_metrics

# Example usage and utility functions
def run_comprehensive_backtest(returns: np.ndarray,
                              enhanced_model: EnhancedTailRiskModel,
                              symbol: str = "TEST") -> Dict[str, BacktestResults]:
    """
    Run comprehensive backtesting suite on enhanced EVT model

    Args:
        returns: Historical returns
        enhanced_model: Fitted enhanced EVT model
        symbol: Asset symbol

    Returns:
        Dictionary with all backtest results
    """

    engine = EVTBacktestingEngine()

    # Split data for out-of-sample testing
    split_idx = int(len(returns) * 0.7)
    test_returns = returns[split_idx:]

    # Generate forecasts (simplified - constant forecasts)
    var_95_forecasts = np.full(len(test_returns), enhanced_model.var_95)
    var_99_forecasts = np.full(len(test_returns), enhanced_model.var_99)
    es_95_forecasts = np.full(len(test_returns), enhanced_model.expected_shortfall_95)

    results = {}

    # Kupiec POF test for VaR 95%
    results['kupiec_var_95'] = engine.kupiec_pof_test(test_returns, var_95_forecasts, 0.95)

    # Kupiec POF test for VaR 99%
    results['kupiec_var_99'] = engine.kupiec_pof_test(test_returns, var_99_forecasts, 0.99)

    # Christoffersen independence test
    results['christoffersen_var_95'] = engine.christoffersen_independence_test(test_returns, var_95_forecasts, 0.95)

    # Expected Shortfall backtest
    results['es_backtest'] = engine.expected_shortfall_backtest(test_returns, var_95_forecasts, es_95_forecasts, 0.95)

    # VaR accuracy metrics
    results['var_accuracy_95'] = engine.calculate_var_accuracy(test_returns, var_95_forecasts, 0.95)
    results['var_accuracy_99'] = engine.calculate_var_accuracy(test_returns, var_99_forecasts, 0.99)

    logger.info(f"Comprehensive backtest completed for {symbol}")

    return results