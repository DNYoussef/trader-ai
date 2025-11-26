"""
Statistical Testing and Validation Framework
Comprehensive validation system for trading strategy performance and statistical significance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import (
    jarque_bera, shapiro, anderson, ttest_1samp, levene, kruskal
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox
)
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from sklearn.utils import resample
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    test_type: str  # normality, stationarity, independence, etc.
    statistic: float
    p_value: float
    critical_values: Optional[Dict[str, float]] = None
    confidence_level: float = 0.95
    is_significant: bool = False
    null_hypothesis: str = ""
    interpretation: str = ""
    recommendation: str = ""

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    strategy_name: str
    validation_type: str
    tests: List[StatisticalTest]
    overall_score: float
    confidence_level: float
    validation_period: Tuple[pd.Timestamp, pd.Timestamp]
    sample_size: int
    power_analysis: Dict[str, float]
    bootstrap_results: Optional[Dict] = None

class StatisticalValidator:
    """Comprehensive statistical validation framework"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.validation_cache = {}

    def validate_strategy_performance(self, returns: pd.Series,
                                    equity_curve: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None,
                                    strategy_name: str = "Strategy") -> ValidationResult:
        """Comprehensive statistical validation of strategy performance"""

        print(f"Running comprehensive statistical validation for {strategy_name}...")

        validation_tests = []

        # 1. Distribution Tests
        distribution_tests = self._test_return_distribution(returns)
        validation_tests.extend(distribution_tests)

        # 2. Time Series Properties
        time_series_tests = self._test_time_series_properties(returns)
        validation_tests.extend(time_series_tests)

        # 3. Independence Tests
        independence_tests = self._test_independence(returns)
        validation_tests.extend(independence_tests)

        # 4. Stationarity Tests
        stationarity_tests = self._test_stationarity(returns)
        validation_tests.extend(stationarity_tests)

        # 5. Heteroscedasticity Tests
        heteroscedasticity_tests = self._test_heteroscedasticity(returns)
        validation_tests.extend(heteroscedasticity_tests)

        # 6. Performance Significance Tests
        performance_tests = self._test_performance_significance(returns, benchmark_returns)
        validation_tests.extend(performance_tests)

        # 7. Risk Model Validation
        risk_tests = self._test_risk_model_validity(returns, equity_curve)
        validation_tests.extend(risk_tests)

        # 8. Regime Stability Tests
        regime_tests = self._test_regime_stability(returns)
        validation_tests.extend(regime_tests)

        # Bootstrap validation
        bootstrap_results = self._bootstrap_validation(returns, equity_curve)

        # Power analysis
        power_analysis = self._perform_power_analysis(returns)

        # Calculate overall validation score
        overall_score = self._calculate_validation_score(validation_tests)

        return ValidationResult(
            strategy_name=strategy_name,
            validation_type="Comprehensive Statistical Validation",
            tests=validation_tests,
            overall_score=overall_score,
            confidence_level=self.confidence_level,
            validation_period=(returns.index[0], returns.index[-1]),
            sample_size=len(returns),
            power_analysis=power_analysis,
            bootstrap_results=bootstrap_results
        )

    def _test_return_distribution(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test return distribution properties"""

        tests = []

        # 1. Normality Tests
        # Jarque-Bera Test
        jb_stat, jb_p = jarque_bera(returns.dropna())
        tests.append(StatisticalTest(
            test_name="Jarque-Bera Normality Test",
            test_type="normality",
            statistic=jb_stat,
            p_value=jb_p,
            confidence_level=self.confidence_level,
            is_significant=jb_p < self.alpha,
            null_hypothesis="Returns follow normal distribution",
            interpretation="Significant result suggests non-normal distribution" if jb_p < self.alpha else "Cannot reject normality",
            recommendation="Consider non-parametric methods" if jb_p < self.alpha else "Parametric methods appropriate"
        ))

        # Shapiro-Wilk Test (for smaller samples)
        if len(returns) <= 5000:
            sw_stat, sw_p = shapiro(returns.dropna())
            tests.append(StatisticalTest(
                test_name="Shapiro-Wilk Normality Test",
                test_type="normality",
                statistic=sw_stat,
                p_value=sw_p,
                confidence_level=self.confidence_level,
                is_significant=sw_p < self.alpha,
                null_hypothesis="Returns follow normal distribution",
                interpretation="Significant result suggests non-normal distribution" if sw_p < self.alpha else "Cannot reject normality",
                recommendation="Use robust statistical methods" if sw_p < self.alpha else "Standard methods applicable"
            ))

        # Anderson-Darling Test
        ad_result = anderson(returns.dropna(), dist='norm')
        ad_significant = ad_result.statistic > ad_result.critical_values[2]  # 5% level
        tests.append(StatisticalTest(
            test_name="Anderson-Darling Normality Test",
            test_type="normality",
            statistic=ad_result.statistic,
            p_value=0.05 if ad_significant else 0.1,  # Approximate
            critical_values={f"{sl}%": cv for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)},
            confidence_level=self.confidence_level,
            is_significant=ad_significant,
            null_hypothesis="Returns follow normal distribution",
            interpretation="Strong evidence against normality" if ad_significant else "Weak evidence against normality",
            recommendation="Transform data or use non-parametric methods" if ad_significant else "Normal distribution assumption reasonable"
        ))

        # 2. Distribution Characteristics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Skewness test
        skew_stat = abs(skewness) / np.sqrt(6/len(returns))  # Approximate test
        skew_p = 2 * (1 - stats.norm.cdf(abs(skew_stat)))
        tests.append(StatisticalTest(
            test_name="Skewness Test",
            test_type="distribution",
            statistic=skewness,
            p_value=skew_p,
            confidence_level=self.confidence_level,
            is_significant=skew_p < self.alpha,
            null_hypothesis="Returns have zero skewness",
            interpretation=f"Returns are {'significantly' if skew_p < self.alpha else 'not significantly'} skewed",
            recommendation="Account for asymmetry in risk models" if skew_p < self.alpha else "Symmetric distribution assumption valid"
        ))

        # Excess kurtosis test
        excess_kurtosis = kurtosis
        kurt_stat = abs(excess_kurtosis) / np.sqrt(24/len(returns))  # Approximate test
        kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))
        tests.append(StatisticalTest(
            test_name="Excess Kurtosis Test",
            test_type="distribution",
            statistic=excess_kurtosis,
            p_value=kurt_p,
            confidence_level=self.confidence_level,
            is_significant=kurt_p < self.alpha,
            null_hypothesis="Returns have normal kurtosis",
            interpretation=f"Returns show {'significant' if kurt_p < self.alpha else 'normal'} tail behavior",
            recommendation="Use fat-tailed distributions for modeling" if kurt_p < self.alpha else "Normal tail assumption reasonable"
        ))

        return tests

    def _test_time_series_properties(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test time series properties"""

        tests = []

        # 1. Serial Correlation Tests
        # Ljung-Box Test
        lb_stat, lb_p = acorr_ljungbox(returns.dropna(), lags=10, return_df=False)
        tests.append(StatisticalTest(
            test_name="Ljung-Box Test (Serial Correlation)",
            test_type="independence",
            statistic=float(lb_stat[9]) if len(lb_stat) > 9 else float(lb_stat[-1]),
            p_value=float(lb_p[9]) if len(lb_p) > 9 else float(lb_p[-1]),
            confidence_level=self.confidence_level,
            is_significant=float(lb_p[9] if len(lb_p) > 9 else lb_p[-1]) < self.alpha,
            null_hypothesis="No serial correlation in returns",
            interpretation="Significant serial correlation detected" if float(lb_p[9] if len(lb_p) > 9 else lb_p[-1]) < self.alpha else "No significant serial correlation",
            recommendation="Model autocorrelation structure" if float(lb_p[9] if len(lb_p) > 9 else lb_p[-1]) < self.alpha else "Independent returns assumption valid"
        ))

        # Durbin-Watson Test
        dw_stat = durbin_watson(returns.dropna())
        dw_significant = dw_stat < 1.5 or dw_stat > 2.5  # Rough guideline
        tests.append(StatisticalTest(
            test_name="Durbin-Watson Test",
            test_type="independence",
            statistic=dw_stat,
            p_value=0.05 if dw_significant else 0.1,  # Approximate
            confidence_level=self.confidence_level,
            is_significant=dw_significant,
            null_hypothesis="No first-order autocorrelation",
            interpretation="Evidence of autocorrelation" if dw_significant else "No strong autocorrelation evidence",
            recommendation="Consider AR models" if dw_significant else "No autocorrelation correction needed"
        ))

        return tests

    def _test_independence(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test independence assumptions"""

        tests = []
        clean_returns = returns.dropna()

        # Runs Test for randomness
        def runs_test(x):
            median = np.median(x)
            binary = (x > median).astype(int)
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i-1]:
                    runs += 1

            n1 = np.sum(binary)
            n2 = len(binary) - n1

            if n1 == 0 or n2 == 0:
                return 0, 1.0

            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

            if variance <= 0:
                return 0, 1.0

            z_stat = (runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            return z_stat, p_value

        runs_stat, runs_p = runs_test(clean_returns)
        tests.append(StatisticalTest(
            test_name="Runs Test for Randomness",
            test_type="independence",
            statistic=runs_stat,
            p_value=runs_p,
            confidence_level=self.confidence_level,
            is_significant=runs_p < self.alpha,
            null_hypothesis="Returns are random",
            interpretation="Non-random pattern detected" if runs_p < self.alpha else "Returns appear random",
            recommendation="Investigate systematic patterns" if runs_p < self.alpha else "Independence assumption supported"
        ))

        return tests

    def _test_stationarity(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test stationarity properties"""

        tests = []
        clean_returns = returns.dropna()

        # Augmented Dickey-Fuller Test
        try:
            adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(clean_returns, autolag='AIC')
            tests.append(StatisticalTest(
                test_name="Augmented Dickey-Fuller Test",
                test_type="stationarity",
                statistic=adf_stat,
                p_value=adf_p,
                critical_values=adf_crit,
                confidence_level=self.confidence_level,
                is_significant=adf_p < self.alpha,
                null_hypothesis="Series has unit root (non-stationary)",
                interpretation="Series is stationary" if adf_p < self.alpha else "Series may have unit root",
                recommendation="Series suitable for modeling" if adf_p < self.alpha else "Consider differencing or detrending"
            ))
        except Exception as e:
            print(f"ADF test failed: {str(e)}")

        # KPSS Test
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(clean_returns, regression='c')
            tests.append(StatisticalTest(
                test_name="KPSS Test",
                test_type="stationarity",
                statistic=kpss_stat,
                p_value=kpss_p,
                critical_values=kpss_crit,
                confidence_level=self.confidence_level,
                is_significant=kpss_p < self.alpha,
                null_hypothesis="Series is stationary",
                interpretation="Series is non-stationary" if kpss_p < self.alpha else "Series is stationary",
                recommendation="Consider trend removal" if kpss_p < self.alpha else "Stationarity assumption valid"
            ))
        except Exception as e:
            print(f"KPSS test failed: {str(e)}")

        return tests

    def _test_heteroscedasticity(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test for heteroscedasticity (changing variance)"""

        tests = []
        clean_returns = returns.dropna()

        if len(clean_returns) < 20:
            return tests

        # Split into periods for variance comparison
        mid_point = len(clean_returns) // 2
        first_half = clean_returns[:mid_point]
        second_half = clean_returns[mid_point:]

        # Levene Test for equal variances
        levene_stat, levene_p = levene(first_half, second_half)
        tests.append(StatisticalTest(
            test_name="Levene Test (Variance Equality)",
            test_type="heteroscedasticity",
            statistic=levene_stat,
            p_value=levene_p,
            confidence_level=self.confidence_level,
            is_significant=levene_p < self.alpha,
            null_hypothesis="Variances are equal across time periods",
            interpretation="Variance changes over time" if levene_p < self.alpha else "Constant variance assumption valid",
            recommendation="Use GARCH models" if levene_p < self.alpha else "Homoscedasticity assumption reasonable"
        ))

        # ARCH Test for conditional heteroscedasticity
        try:
            # Simple ARCH test using squared returns
            squared_returns = clean_returns ** 2
            squared_returns.mean()

            # Test if squared returns are serially correlated
            lb_stat_sq, lb_p_sq = acorr_ljungbox(squared_returns, lags=5, return_df=False)
            arch_p = float(lb_p_sq[4]) if len(lb_p_sq) > 4 else float(lb_p_sq[-1])

            tests.append(StatisticalTest(
                test_name="ARCH Effect Test",
                test_type="heteroscedasticity",
                statistic=float(lb_stat_sq[4]) if len(lb_stat_sq) > 4 else float(lb_stat_sq[-1]),
                p_value=arch_p,
                confidence_level=self.confidence_level,
                is_significant=arch_p < self.alpha,
                null_hypothesis="No ARCH effects",
                interpretation="ARCH effects present" if arch_p < self.alpha else "No significant ARCH effects",
                recommendation="Model volatility clustering" if arch_p < self.alpha else "Constant volatility assumption valid"
            ))
        except Exception as e:
            print(f"ARCH test failed: {str(e)}")

        return tests

    def _test_performance_significance(self, returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None) -> List[StatisticalTest]:
        """Test statistical significance of performance"""

        tests = []
        clean_returns = returns.dropna()

        # Test if mean return is significantly different from zero
        t_stat, t_p = ttest_1samp(clean_returns, 0)
        tests.append(StatisticalTest(
            test_name="One-Sample t-Test (Mean Return)",
            test_type="performance",
            statistic=t_stat,
            p_value=t_p,
            confidence_level=self.confidence_level,
            is_significant=t_p < self.alpha,
            null_hypothesis="Mean return equals zero",
            interpretation="Significant positive/negative returns" if t_p < self.alpha else "Returns not significantly different from zero",
            recommendation="Performance is statistically significant" if t_p < self.alpha else "Performance may be due to chance"
        ))

        # Test if Sharpe ratio is significantly different from zero
        sharpe_ratio = clean_returns.mean() / clean_returns.std() * np.sqrt(252) if clean_returns.std() > 0 else 0
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2) / len(clean_returns))  # Approximate standard error
        sharpe_t = sharpe_ratio / sharpe_se if sharpe_se > 0 else 0
        sharpe_p = 2 * (1 - stats.norm.cdf(abs(sharpe_t)))

        tests.append(StatisticalTest(
            test_name="Sharpe Ratio Significance Test",
            test_type="performance",
            statistic=sharpe_t,
            p_value=sharpe_p,
            confidence_level=self.confidence_level,
            is_significant=sharpe_p < self.alpha,
            null_hypothesis="Sharpe ratio equals zero",
            interpretation="Risk-adjusted returns are significant" if sharpe_p < self.alpha else "Risk-adjusted performance not significant",
            recommendation="Strong risk-adjusted performance" if sharpe_p < self.alpha else "Performance may not persist"
        ))

        # Benchmark comparison if available
        if benchmark_returns is not None:
            common_index = clean_returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 10:
                strategy_aligned = clean_returns.loc[common_index]
                benchmark_aligned = benchmark_returns.loc[common_index]

                # Paired t-test for excess returns
                excess_returns = strategy_aligned - benchmark_aligned
                excess_t, excess_p = ttest_1samp(excess_returns, 0)

                tests.append(StatisticalTest(
                    test_name="Excess Return Significance Test",
                    test_type="performance",
                    statistic=excess_t,
                    p_value=excess_p,
                    confidence_level=self.confidence_level,
                    is_significant=excess_p < self.alpha,
                    null_hypothesis="No excess returns over benchmark",
                    interpretation="Significant outperformance" if excess_p < self.alpha and excess_t > 0 else "No significant outperformance",
                    recommendation="Strategy adds value" if excess_p < self.alpha and excess_t > 0 else "Reconsider strategy vs benchmark"
                ))

        return tests

    def _test_risk_model_validity(self, returns: pd.Series, equity_curve: pd.Series) -> List[StatisticalTest]:
        """Test validity of risk model assumptions"""

        tests = []
        clean_returns = returns.dropna()

        # VaR Backtesting (Kupiec Test)
        var_95 = np.percentile(clean_returns, 5)
        violations = (clean_returns < var_95).sum()
        len(clean_returns) * 0.05

        # Binomial test for VaR violations
        binom_p = stats.binom_test(violations, len(clean_returns), 0.05)
        tests.append(StatisticalTest(
            test_name="VaR Backtest (Kupiec Test)",
            test_type="risk_model",
            statistic=violations / len(clean_returns),
            p_value=binom_p,
            confidence_level=self.confidence_level,
            is_significant=binom_p < self.alpha,
            null_hypothesis="VaR model is correctly calibrated",
            interpretation="VaR model is miscalibrated" if binom_p < self.alpha else "VaR model appears well-calibrated",
            recommendation="Recalibrate VaR model" if binom_p < self.alpha else "VaR model is acceptable"
        ))

        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Test if maximum drawdown is within expected range
        # Using approximate normal distribution for max drawdown
        expected_max_dd = -2 * clean_returns.std() * np.sqrt(len(clean_returns))
        dd_z_score = (max_drawdown - expected_max_dd) / (clean_returns.std() * 0.5)
        dd_p = 2 * (1 - stats.norm.cdf(abs(dd_z_score)))

        tests.append(StatisticalTest(
            test_name="Maximum Drawdown Test",
            test_type="risk_model",
            statistic=dd_z_score,
            p_value=dd_p,
            confidence_level=self.confidence_level,
            is_significant=dd_p < self.alpha,
            null_hypothesis="Drawdown is within expected range",
            interpretation="Unusually large drawdown" if dd_p < self.alpha else "Drawdown within expected range",
            recommendation="Review risk management" if dd_p < self.alpha else "Risk model appears adequate"
        ))

        return tests

    def _test_regime_stability(self, returns: pd.Series) -> List[StatisticalTest]:
        """Test stability across different market regimes"""

        tests = []
        clean_returns = returns.dropna()

        if len(clean_returns) < 60:
            return tests

        # Split into thirds for regime analysis
        n = len(clean_returns)
        third = n // 3

        regime1 = clean_returns[:third]
        regime2 = clean_returns[third:2*third]
        regime3 = clean_returns[2*third:]

        # Test for equal means across regimes
        f_stat, f_p = stats.f_oneway(regime1, regime2, regime3)
        tests.append(StatisticalTest(
            test_name="One-Way ANOVA (Regime Stability)",
            test_type="stability",
            statistic=f_stat,
            p_value=f_p,
            confidence_level=self.confidence_level,
            is_significant=f_p < self.alpha,
            null_hypothesis="Performance is stable across time periods",
            interpretation="Performance varies significantly across periods" if f_p < self.alpha else "Stable performance across periods",
            recommendation="Investigate regime-dependent effects" if f_p < self.alpha else "Performance appears consistent"
        ))

        # Kruskal-Wallis test (non-parametric alternative)
        kw_stat, kw_p = kruskal(regime1, regime2, regime3)
        tests.append(StatisticalTest(
            test_name="Kruskal-Wallis Test (Non-parametric Stability)",
            test_type="stability",
            statistic=kw_stat,
            p_value=kw_p,
            confidence_level=self.confidence_level,
            is_significant=kw_p < self.alpha,
            null_hypothesis="Distributions are identical across periods",
            interpretation="Significant distribution changes across periods" if kw_p < self.alpha else "Consistent distributions across periods",
            recommendation="Model regime-dependent behavior" if kw_p < self.alpha else "Single model may suffice"
        ))

        return tests

    def _bootstrap_validation(self, returns: pd.Series, equity_curve: pd.Series) -> Dict:
        """Bootstrap validation of performance metrics"""

        clean_returns = returns.dropna()
        n_bootstrap = 1000

        # Bootstrap samples
        bootstrap_metrics = {
            'total_returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = resample(clean_returns, n_samples=len(clean_returns))
            bootstrap_series = pd.Series(bootstrap_sample)

            # Calculate metrics
            total_return = (1 + bootstrap_series).prod() - 1
            sharpe_ratio = bootstrap_series.mean() / bootstrap_series.std() * np.sqrt(252) if bootstrap_series.std() > 0 else 0

            bootstrap_equity = (1 + bootstrap_series).cumprod() * 200
            peak = bootstrap_equity.expanding().max()
            drawdown = (bootstrap_equity - peak) / peak
            max_drawdown = drawdown.min()

            win_rate = (bootstrap_series > 0).mean()

            bootstrap_metrics['total_returns'].append(total_return)
            bootstrap_metrics['sharpe_ratios'].append(sharpe_ratio)
            bootstrap_metrics['max_drawdowns'].append(max_drawdown)
            bootstrap_metrics['win_rates'].append(win_rate)

        # Calculate confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            confidence_intervals[metric] = (ci_lower, ci_upper)

        return {
            'bootstrap_metrics': bootstrap_metrics,
            'confidence_intervals': confidence_intervals,
            'n_bootstrap': n_bootstrap
        }

    def _perform_power_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """Perform statistical power analysis"""

        clean_returns = returns.dropna()
        n = len(clean_returns)

        # Power analysis for mean return test
        effect_size = abs(clean_returns.mean()) / clean_returns.std() if clean_returns.std() > 0 else 0

        # Approximate power calculation
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = effect_size * np.sqrt(n) - z_alpha
        power = stats.norm.cdf(z_beta)

        # Minimum sample size for 80% power
        if effect_size > 0:
            required_n_80 = ((z_alpha + stats.norm.ppf(0.8)) / effect_size) ** 2
        else:
            required_n_80 = float('inf')

        return {
            'current_sample_size': n,
            'effect_size': effect_size,
            'statistical_power': power,
            'required_n_for_80_percent_power': required_n_80,
            'power_adequate': power >= 0.8
        }

    def _calculate_validation_score(self, tests: List[StatisticalTest]) -> float:
        """Calculate overall validation score"""

        if not tests:
            return 0.0

        # Weight different test types
        weights = {
            'normality': 0.15,
            'independence': 0.20,
            'stationarity': 0.15,
            'heteroscedasticity': 0.15,
            'performance': 0.25,
            'risk_model': 0.20,
            'stability': 0.15,
            'distribution': 0.10
        }

        total_score = 0
        total_weight = 0

        for test in tests:
            test_weight = weights.get(test.test_type, 0.1)

            # Score based on test result appropriateness
            if test.test_type in ['performance']:
                # For performance tests, significant is good
                test_score = 1.0 if test.is_significant else 0.5
            elif test.test_type in ['normality', 'independence', 'stationarity']:
                # For assumption tests, non-significant is often preferred
                test_score = 0.8 if not test.is_significant else 0.6
            elif test.test_type in ['risk_model', 'stability']:
                # For model validation, non-significant is good
                test_score = 1.0 if not test.is_significant else 0.4
            else:
                test_score = 0.7  # Neutral score

            total_score += test_score * test_weight
            total_weight += test_weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """Generate comprehensive validation report"""

        report = []
        report.append("=" * 80)
        report.append("STATISTICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Strategy: {validation_result.strategy_name}")
        report.append(f"Validation Period: {validation_result.validation_period[0].strftime('%Y-%m-%d')} to {validation_result.validation_period[1].strftime('%Y-%m-%d')}")
        report.append(f"Sample Size: {validation_result.sample_size}")
        report.append(f"Confidence Level: {validation_result.confidence_level:.1%}")
        report.append(f"Overall Validation Score: {validation_result.overall_score:.3f}")
        report.append("")

        # Group tests by type
        test_groups = {}
        for test in validation_result.tests:
            if test.test_type not in test_groups:
                test_groups[test.test_type] = []
            test_groups[test.test_type].append(test)

        # Report by test category
        for test_type, tests in test_groups.items():
            report.append(f"{test_type.upper().replace('_', ' ')} TESTS")
            report.append("-" * (len(test_type) + 6))

            for test in tests:
                significance = "✓" if test.is_significant else "✗"
                report.append(f"\n{significance} {test.test_name}")
                report.append(f"   Statistic: {test.statistic:.4f}")
                report.append(f"   P-value: {test.p_value:.4f}")
                report.append(f"   Interpretation: {test.interpretation}")
                report.append(f"   Recommendation: {test.recommendation}")

            report.append("")

        # Power Analysis
        if validation_result.power_analysis:
            power = validation_result.power_analysis
            report.append("POWER ANALYSIS")
            report.append("-" * 15)
            report.append(f"Effect Size: {power['effect_size']:.3f}")
            report.append(f"Statistical Power: {power['statistical_power']:.3f}")
            report.append(f"Power Adequate (≥80%): {'Yes' if power['power_adequate'] else 'No'}")
            if power['required_n_for_80_percent_power'] != float('inf'):
                report.append(f"Required Sample Size for 80% Power: {power['required_n_for_80_percent_power']:.0f}")
            report.append("")

        # Bootstrap Results
        if validation_result.bootstrap_results:
            bootstrap = validation_result.bootstrap_results
            report.append("BOOTSTRAP VALIDATION")
            report.append("-" * 20)
            report.append(f"Bootstrap Samples: {bootstrap['n_bootstrap']}")
            report.append("95% Confidence Intervals:")

            for metric, (ci_lower, ci_upper) in bootstrap['confidence_intervals'].items():
                metric_name = metric.replace('_', ' ').title()
                report.append(f"  {metric_name}: [{ci_lower:.4f}, {ci_upper:.4f}]")
            report.append("")

        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 20)

        if validation_result.overall_score >= 0.8:
            report.append("✅ EXCELLENT: Strategy passes most statistical validation tests")
        elif validation_result.overall_score >= 0.7:
            report.append("✅ GOOD: Strategy shows solid statistical properties")
        elif validation_result.overall_score >= 0.6:
            report.append("⚠️  ACCEPTABLE: Some concerns but generally valid")
        elif validation_result.overall_score >= 0.5:
            report.append("⚠️  MARGINAL: Several statistical issues identified")
        else:
            report.append("❌ POOR: Significant statistical problems detected")

        # Key recommendations
        report.append("\nKEY RECOMMENDATIONS:")

        significant_performance_tests = [t for t in validation_result.tests if t.test_type == 'performance' and t.is_significant]
        if significant_performance_tests:
            report.append("• Performance is statistically significant")
        else:
            report.append("• Performance significance is questionable")

        normality_violations = [t for t in validation_result.tests if t.test_type == 'normality' and t.is_significant]
        if normality_violations:
            report.append("• Use non-parametric methods due to non-normal returns")

        independence_violations = [t for t in validation_result.tests if t.test_type == 'independence' and t.is_significant]
        if independence_violations:
            report.append("• Account for serial correlation in models")

        stability_issues = [t for t in validation_result.tests if t.test_type == 'stability' and t.is_significant]
        if stability_issues:
            report.append("• Monitor for regime changes and parameter stability")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def create_validation_visualization(self, validation_result: ValidationResult,
                                      returns: pd.Series) -> go.Figure:
        """Create comprehensive validation visualization"""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Return Distribution', 'Q-Q Plot (Normality)',
                          'Autocorrelation Function', 'Test Results Summary',
                          'Rolling Statistics', 'Bootstrap Confidence Intervals'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        clean_returns = returns.dropna()

        # 1. Return distribution histogram
        fig.add_trace(
            go.Histogram(x=clean_returns, nbinsx=50, name='Return Distribution'),
            row=1, col=1
        )

        # Add normal distribution overlay
        x_norm = np.linspace(clean_returns.min(), clean_returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, clean_returns.mean(), clean_returns.std())
        y_norm_scaled = y_norm * len(clean_returns) * (clean_returns.max() - clean_returns.min()) / 50

        fig.add_trace(
            go.Scatter(x=x_norm, y=y_norm_scaled, mode='lines',
                      name='Normal Distribution', line=dict(color='red')),
            row=1, col=1
        )

        # 2. Q-Q Plot for normality
        (osm, osr), (slope, intercept, r) = stats.probplot(clean_returns, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
            row=1, col=2
        )

        # Add reference line
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        fig.add_trace(
            go.Scatter(x=line_x, y=line_y, mode='lines',
                      name='Reference Line', line=dict(color='red')),
            row=1, col=2
        )

        # 3. Autocorrelation function
        max_lags = min(40, len(clean_returns) // 4)
        lags = range(1, max_lags + 1)
        autocorrs = [clean_returns.autocorr(lag=lag) for lag in lags]

        fig.add_trace(
            go.Scatter(x=list(lags), y=autocorrs, mode='lines+markers',
                      name='Autocorrelation'),
            row=2, col=1
        )

        # Add significance bounds
        bound = 1.96 / np.sqrt(len(clean_returns))
        fig.add_hline(y=bound, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-bound, line_dash="dash", line_color="red", row=2, col=1)

        # 4. Test results summary
        test_types = list(set(test.test_type for test in validation_result.tests))
        significance_counts = []

        for test_type in test_types:
            type_tests = [t for t in validation_result.tests if t.test_type == test_type]
            significant_count = sum(1 for t in type_tests if t.is_significant)
            significance_counts.append(significant_count / len(type_tests) if type_tests else 0)

        fig.add_trace(
            go.Bar(x=test_types, y=significance_counts,
                  name='Significance Rate', marker_color='lightblue'),
            row=2, col=2
        )

        # 5. Rolling statistics
        window = min(30, len(clean_returns) // 4)
        rolling_mean = clean_returns.rolling(window).mean()
        rolling_std = clean_returns.rolling(window).std()

        fig.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean,
                      mode='lines', name='Rolling Mean'),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=rolling_std.index, y=rolling_std,
                      mode='lines', name='Rolling Std', yaxis='y2'),
            row=3, col=1
        )

        # 6. Bootstrap confidence intervals (if available)
        if validation_result.bootstrap_results:
            bootstrap = validation_result.bootstrap_results
            metrics = list(bootstrap['confidence_intervals'].keys())
            ci_widths = []

            for metric in metrics:
                ci_lower, ci_upper = bootstrap['confidence_intervals'][metric]
                ci_widths.append(ci_upper - ci_lower)

            fig.add_trace(
                go.Bar(x=metrics, y=ci_widths,
                      name='CI Width', marker_color='orange'),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Statistical Validation Dashboard",
            height=900,
            showlegend=True
        )

        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Simulate returns with some interesting properties
    returns = np.random.normal(0.0005, 0.015, len(dates))

    # Add some autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]

    # Add some heteroscedasticity
    vol_regime = np.sin(np.arange(len(returns)) / 50) * 0.005 + 0.015
    returns = returns * vol_regime / 0.015

    returns_series = pd.Series(returns, index=dates)
    equity_curve = (1 + returns_series).cumprod() * 200

    # Initialize validator
    validator = StatisticalValidator(confidence_level=0.95)

    # Perform validation
    validation_result = validator.validate_strategy_performance(
        returns_series, equity_curve, strategy_name="Gary×Taleb Strategy"
    )

    # Generate report
    report = validator.generate_validation_report(validation_result)
    print(report)

    # Create visualization
    fig = validator.create_validation_visualization(validation_result, returns_series)
    fig.show()

    print("\nValidation completed!")
    print(f"Overall Score: {validation_result.overall_score:.3f}")
    print(f"Tests Performed: {len(validation_result.tests)}")
    print(f"Statistical Power: {validation_result.power_analysis['statistical_power']:.3f}")