"""
Strategy Comparison Engine with Statistical Significance Testing
Advanced comparison framework for trading strategy validation and ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, jarque_bera, normaltest
import warnings
from sklearn.utils import resample
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_level: float
    is_significant: bool
    interpretation: str

@dataclass
class ComparisonResult:
    """Comprehensive strategy comparison result"""
    strategy_a: str
    strategy_b: str
    tests: List[StatisticalTest]
    performance_difference: Dict[str, float]
    confidence_interval: Tuple[float, float]
    effect_size: float
    recommendation: str
    risk_assessment: str

class StrategyComparison:
    """Advanced strategy comparison engine with statistical testing"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.comparison_cache = {}

    def compare_strategies(self, strategy_data: Dict[str, Dict]) -> Dict[str, ComparisonResult]:
        """Comprehensive comparison of multiple strategies"""

        strategies = list(strategy_data.keys())
        comparison_results = {}

        print(f"Comparing {len(strategies)} strategies with statistical significance testing...")

        # Pairwise comparisons
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                strategy_a = strategies[i]
                strategy_b = strategies[j]

                print(f"Comparing {strategy_a} vs {strategy_b}...")

                result = self._compare_two_strategies(
                    strategy_a, strategy_data[strategy_a],
                    strategy_b, strategy_data[strategy_b]
                )

                comparison_key = f"{strategy_a}_vs_{strategy_b}"
                comparison_results[comparison_key] = result

        return comparison_results

    def _compare_two_strategies(self, name_a: str, data_a: Dict,
                              name_b: str, data_b: Dict) -> ComparisonResult:
        """Detailed comparison between two strategies"""

        returns_a = data_a['returns'].dropna()
        returns_b = data_b['returns'].dropna()

        # Align returns to same period
        common_index = returns_a.index.intersection(returns_b.index)
        if len(common_index) < 30:
            # Handle different time periods
            min_len = min(len(returns_a), len(returns_b))
            returns_a = returns_a.iloc[-min_len:]
            returns_b = returns_b.iloc[-min_len:]
        else:
            returns_a = returns_a.loc[common_index]
            returns_b = returns_b.loc[common_index]

        # Statistical tests
        tests = []

        # 1. Mean return difference test (t-test)
        t_stat, t_p_value = ttest_ind(returns_a, returns_b)
        tests.append(StatisticalTest(
            test_name="T-Test (Mean Returns)",
            statistic=t_stat,
            p_value=t_p_value,
            critical_value=stats.t.ppf(1 - self.alpha/2, len(returns_a) + len(returns_b) - 2),
            confidence_level=self.confidence_level,
            is_significant=t_p_value < self.alpha,
            interpretation=f"{'Significant' if t_p_value < self.alpha else 'Not significant'} difference in mean returns"
        ))

        # 2. Mann-Whitney U test (non-parametric)
        mw_stat, mw_p_value = mannwhitneyu(returns_a, returns_b, alternative='two-sided')
        tests.append(StatisticalTest(
            test_name="Mann-Whitney U Test",
            statistic=mw_stat,
            p_value=mw_p_value,
            critical_value=None,
            confidence_level=self.confidence_level,
            is_significant=mw_p_value < self.alpha,
            interpretation=f"{'Significant' if mw_p_value < self.alpha else 'Not significant'} difference in return distributions"
        ))

        # 3. Kolmogorov-Smirnov test (distribution similarity)
        ks_stat, ks_p_value = ks_2samp(returns_a, returns_b)
        tests.append(StatisticalTest(
            test_name="Kolmogorov-Smirnov Test",
            statistic=ks_stat,
            p_value=ks_p_value,
            critical_value=None,
            confidence_level=self.confidence_level,
            is_significant=ks_p_value < self.alpha,
            interpretation=f"Distributions are {'different' if ks_p_value < self.alpha else 'similar'}"
        ))

        # 4. Variance test (F-test)
        f_stat = np.var(returns_a, ddof=1) / np.var(returns_b, ddof=1)
        f_p_value = 2 * min(stats.f.cdf(f_stat, len(returns_a)-1, len(returns_b)-1),
                           1 - stats.f.cdf(f_stat, len(returns_a)-1, len(returns_b)-1))
        tests.append(StatisticalTest(
            test_name="F-Test (Variance Equality)",
            statistic=f_stat,
            p_value=f_p_value,
            critical_value=stats.f.ppf(1 - self.alpha/2, len(returns_a)-1, len(returns_b)-1),
            confidence_level=self.confidence_level,
            is_significant=f_p_value < self.alpha,
            interpretation=f"Variances are {'significantly different' if f_p_value < self.alpha else 'not significantly different'}"
        ))

        # 5. Sharpe ratio comparison
        sharpe_a = self._calculate_sharpe(returns_a)
        sharpe_b = self._calculate_sharpe(returns_b)
        sharpe_diff_stat, sharpe_p_value = self._sharpe_ratio_test(returns_a, returns_b)

        tests.append(StatisticalTest(
            test_name="Sharpe Ratio Difference",
            statistic=sharpe_diff_stat,
            p_value=sharpe_p_value,
            critical_value=1.96,  # 95% confidence
            confidence_level=self.confidence_level,
            is_significant=abs(sharpe_diff_stat) > 1.96,
            interpretation=f"Sharpe ratios are {'significantly different' if abs(sharpe_diff_stat) > 1.96 else 'not significantly different'}"
        ))

        # Performance differences
        performance_diff = {
            'mean_return_diff': returns_a.mean() - returns_b.mean(),
            'volatility_diff': returns_a.std() - returns_b.std(),
            'sharpe_diff': sharpe_a - sharpe_b,
            'max_drawdown_diff': self._calculate_max_drawdown(returns_a) - self._calculate_max_drawdown(returns_b),
            'skewness_diff': returns_a.skew() - returns_b.skew(),
            'kurtosis_diff': returns_a.kurtosis() - returns_b.kurtosis()
        }

        # Confidence interval for mean difference
        pooled_std = np.sqrt(((len(returns_a)-1)*np.var(returns_a, ddof=1) +
                             (len(returns_b)-1)*np.var(returns_b, ddof=1)) /
                            (len(returns_a) + len(returns_b) - 2))
        standard_error = pooled_std * np.sqrt(1/len(returns_a) + 1/len(returns_b))
        t_critical = stats.t.ppf(1 - self.alpha/2, len(returns_a) + len(returns_b) - 2)
        margin_error = t_critical * standard_error
        mean_diff = returns_a.mean() - returns_b.mean()
        confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)

        # Effect size (Cohen's d)
        effect_size = mean_diff / pooled_std

        # Generate recommendation
        recommendation = self._generate_recommendation(tests, performance_diff, effect_size)

        # Risk assessment
        risk_assessment = self._assess_risk_characteristics(returns_a, returns_b, name_a, name_b)

        return ComparisonResult(
            strategy_a=name_a,
            strategy_b=name_b,
            tests=tests,
            performance_difference=performance_diff,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            recommendation=recommendation,
            risk_assessment=risk_assessment
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_return = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_return / volatility if volatility > 0 else 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _sharpe_ratio_test(self, returns_a: pd.Series, returns_b: pd.Series) -> Tuple[float, float]:
        """Test for significant difference in Sharpe ratios"""

        # Calculate Sharpe ratios
        sharpe_a = self._calculate_sharpe(returns_a)
        sharpe_b = self._calculate_sharpe(returns_b)

        # Bootstrap confidence intervals
        n_bootstrap = 1000
        sharpe_diffs = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_a = resample(returns_a, n_samples=len(returns_a))
            sample_b = resample(returns_b, n_samples=len(returns_b))

            sharpe_diff = self._calculate_sharpe(pd.Series(sample_a)) - self._calculate_sharpe(pd.Series(sample_b))
            sharpe_diffs.append(sharpe_diff)

        sharpe_diffs = np.array(sharpe_diffs)

        # Calculate test statistic
        observed_diff = sharpe_a - sharpe_b
        z_score = (observed_diff - np.mean(sharpe_diffs)) / np.std(sharpe_diffs)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return z_score, p_value

    def _generate_recommendation(self, tests: List[StatisticalTest],
                               performance_diff: Dict[str, float],
                               effect_size: float) -> str:
        """Generate strategy recommendation based on test results"""

        significant_tests = [test for test in tests if test.is_significant]
        total_tests = len(tests)

        if len(significant_tests) >= total_tests * 0.6:  # Majority of tests significant
            if performance_diff['sharpe_diff'] > 0:
                if abs(effect_size) > 0.8:  # Large effect size
                    return f"STRONG RECOMMENDATION: Strategy A significantly outperforms Strategy B (Large effect size: {effect_size:.2f})"
                elif abs(effect_size) > 0.5:  # Medium effect size
                    return f"MODERATE RECOMMENDATION: Strategy A outperforms Strategy B (Medium effect size: {effect_size:.2f})"
                else:
                    return f"WEAK RECOMMENDATION: Strategy A slightly outperforms Strategy B (Small effect size: {effect_size:.2f})"
            else:
                if abs(effect_size) > 0.8:
                    return f"STRONG RECOMMENDATION: Strategy B significantly outperforms Strategy A (Large effect size: {abs(effect_size):.2f})"
                elif abs(effect_size) > 0.5:
                    return f"MODERATE RECOMMENDATION: Strategy B outperforms Strategy A (Medium effect size: {abs(effect_size):.2f})"
                else:
                    return f"WEAK RECOMMENDATION: Strategy B slightly outperforms Strategy A (Small effect size: {abs(effect_size):.2f})"
        else:
            return f"NO CLEAR RECOMMENDATION: No statistically significant difference found ({len(significant_tests)}/{total_tests} tests significant)"

    def _assess_risk_characteristics(self, returns_a: pd.Series, returns_b: pd.Series,
                                   name_a: str, name_b: str) -> str:
        """Assess risk characteristics of both strategies"""

        risk_metrics_a = {
            'volatility': returns_a.std() * np.sqrt(252),
            'max_drawdown': abs(self._calculate_max_drawdown(returns_a)),
            'var_95': np.percentile(returns_a, 5),
            'skewness': returns_a.skew(),
            'kurtosis': returns_a.kurtosis()
        }

        risk_metrics_b = {
            'volatility': returns_b.std() * np.sqrt(252),
            'max_drawdown': abs(self._calculate_max_drawdown(returns_b)),
            'var_95': np.percentile(returns_b, 5),
            'skewness': returns_b.skew(),
            'kurtosis': returns_b.kurtosis()
        }

        assessment = []

        # Volatility comparison
        if risk_metrics_a['volatility'] < risk_metrics_b['volatility'] * 0.9:
            assessment.append(f"{name_a} has significantly lower volatility")
        elif risk_metrics_a['volatility'] > risk_metrics_b['volatility'] * 1.1:
            assessment.append(f"{name_a} has significantly higher volatility")

        # Drawdown comparison
        if risk_metrics_a['max_drawdown'] < risk_metrics_b['max_drawdown'] * 0.8:
            assessment.append(f"{name_a} has much better drawdown control")
        elif risk_metrics_a['max_drawdown'] > risk_metrics_b['max_drawdown'] * 1.2:
            assessment.append(f"{name_a} has worse drawdown control")

        # Tail risk comparison
        if risk_metrics_a['var_95'] > risk_metrics_b['var_95'] * 1.2:
            assessment.append(f"{name_a} has better tail risk profile")
        elif risk_metrics_a['var_95'] < risk_metrics_b['var_95'] * 0.8:
            assessment.append(f"{name_a} has worse tail risk profile")

        # Distribution characteristics
        if abs(risk_metrics_a['skewness']) < abs(risk_metrics_b['skewness']) * 0.8:
            assessment.append(f"{name_a} has more symmetric returns")

        if risk_metrics_a['kurtosis'] > risk_metrics_b['kurtosis'] * 1.5:
            assessment.append(f"{name_a} has higher tail risk (excess kurtosis)")

        return "; ".join(assessment) if assessment else "Similar risk characteristics"

    def run_robustness_tests(self, strategy_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Run robustness tests for all strategies"""

        robustness_results = {}

        for strategy_name, data in strategy_data.items():
            returns = data['returns'].dropna()

            tests = {}

            # 1. Normality test
            jb_stat, jb_p_value = jarque_bera(returns)
            sw_stat, sw_p_value = normaltest(returns)

            tests['normality'] = {
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p_value},
                'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_p_value},
                'is_normal': jb_p_value > 0.05 and sw_p_value > 0.05
            }

            # 2. Stationarity test
            adf_stat, adf_p_value, _, _, critical_values, _ = adfuller(returns, autolag='AIC')

            tests['stationarity'] = {
                'adf_statistic': adf_stat,
                'adf_p_value': adf_p_value,
                'critical_values': critical_values,
                'is_stationary': adf_p_value < 0.05
            }

            # 3. Serial correlation test
            lb_stat, lb_p_value = acorr_ljungbox(returns, lags=10, return_df=False)

            tests['serial_correlation'] = {
                'ljung_box_statistic': float(lb_stat[9]) if len(lb_stat) > 9 else float(lb_stat[-1]),
                'ljung_box_p_value': float(lb_p_value[9]) if len(lb_p_value) > 9 else float(lb_p_value[-1]),
                'has_correlation': float(lb_p_value[9] if len(lb_p_value) > 9 else lb_p_value[-1]) < 0.05
            }

            # 4. Stability over time
            window_size = max(60, len(returns) // 4)
            rolling_sharpe = []

            for i in range(window_size, len(returns)):
                window_returns = returns.iloc[i-window_size:i]
                sharpe = self._calculate_sharpe(window_returns)
                rolling_sharpe.append(sharpe)

            tests['stability'] = {
                'sharpe_volatility': np.std(rolling_sharpe),
                'is_stable': np.std(rolling_sharpe) < 0.5  # Threshold for stability
            }

            # 5. Regime change detection (simplified)
            # Using rolling correlation with market
            market_proxy = np.random.normal(0.0003, 0.015, len(returns))  # Simplified market proxy
            rolling_correlation = returns.rolling(30).corr(pd.Series(market_proxy, index=returns.index))

            tests['regime_sensitivity'] = {
                'correlation_volatility': rolling_correlation.std(),
                'is_regime_sensitive': rolling_correlation.std() > 0.3
            }

            robustness_results[strategy_name] = tests

        return robustness_results

    def generate_comparison_report(self, comparison_results: Dict[str, ComparisonResult],
                                 robustness_results: Dict[str, Dict]) -> str:
        """Generate comprehensive comparison report"""

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE STRATEGY COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Confidence Level: {self.confidence_level:.1%}")
        report.append("")

        # Summary of comparisons
        report.append("PAIRWISE COMPARISON SUMMARY")
        report.append("-" * 40)

        for comp_key, result in comparison_results.items():
            report.append(f"\n{result.strategy_a} vs {result.strategy_b}:")
            report.append(f"  Recommendation: {result.recommendation}")
            report.append(f"  Effect Size: {result.effect_size:.3f}")
            report.append(f"  Significant Tests: {sum(1 for test in result.tests if test.is_significant)}/{len(result.tests)}")

            # Key performance differences
            perf_diff = result.performance_difference
            report.append("  Performance Differences:")
            report.append(f"    Mean Return: {perf_diff['mean_return_diff']:.4f}")
            report.append(f"    Sharpe Ratio: {perf_diff['sharpe_diff']:.3f}")
            report.append(f"    Max Drawdown: {perf_diff['max_drawdown_diff']:.2%}")

        # Detailed statistical tests
        report.append("\n\nDETAILED STATISTICAL TEST RESULTS")
        report.append("-" * 45)

        for comp_key, result in comparison_results.items():
            report.append(f"\n{result.strategy_a} vs {result.strategy_b}:")

            for test in result.tests:
                significance = "✓" if test.is_significant else "✗"
                report.append(f"  {significance} {test.test_name}:")
                report.append(f"    Statistic: {test.statistic:.4f}")
                report.append(f"    P-value: {test.p_value:.4f}")
                report.append(f"    {test.interpretation}")

        # Robustness analysis
        report.append("\n\nROBUSTNESS ANALYSIS")
        report.append("-" * 25)

        for strategy_name, tests in robustness_results.items():
            report.append(f"\n{strategy_name}:")

            # Normality
            is_normal = "✓" if tests['normality']['is_normal'] else "✗"
            report.append(f"  {is_normal} Normal Distribution (JB p-value: {tests['normality']['jarque_bera']['p_value']:.4f})")

            # Stationarity
            is_stationary = "✓" if tests['stationarity']['is_stationary'] else "✗"
            report.append(f"  {is_stationary} Stationary (ADF p-value: {tests['stationarity']['adf_p_value']:.4f})")

            # Serial correlation
            has_correlation = "✗" if tests['serial_correlation']['has_correlation'] else "✓"
            report.append(f"  {has_correlation} No Serial Correlation (LB p-value: {tests['serial_correlation']['ljung_box_p_value']:.4f})")

            # Stability
            is_stable = "✓" if tests['stability']['is_stable'] else "✗"
            report.append(f"  {is_stable} Performance Stability (Sharpe volatility: {tests['stability']['sharpe_volatility']:.3f})")

        # Overall rankings
        report.append("\n\nOVERALL STRATEGY RANKING")
        report.append("-" * 30)

        # Calculate overall scores (simplified)
        strategy_scores = {}
        for comp_key, result in comparison_results.items():
            strategy_a = result.strategy_a
            strategy_b = result.strategy_b

            if strategy_a not in strategy_scores:
                strategy_scores[strategy_a] = 0
            if strategy_b not in strategy_scores:
                strategy_scores[strategy_b] = 0

            if result.performance_difference['sharpe_diff'] > 0:
                strategy_scores[strategy_a] += 1
            else:
                strategy_scores[strategy_b] += 1

        ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (strategy, score) in enumerate(ranked_strategies, 1):
            report.append(f"{i}. {strategy} (Score: {score})")

        # Recommendations
        report.append("\n\nFINAL RECOMMENDATIONS")
        report.append("-" * 25)

        if ranked_strategies:
            best_strategy = ranked_strategies[0][0]
            report.append(f"• Best Overall Strategy: {best_strategy}")

            # Find Gary×Taleb strategy if present
            gary_taleb_rank = None
            for i, (strategy, _) in enumerate(ranked_strategies):
                if 'gary' in strategy.lower() or 'taleb' in strategy.lower():
                    gary_taleb_rank = i + 1
                    break

            if gary_taleb_rank:
                if gary_taleb_rank == 1:
                    report.append("• Gary×Taleb strategy is the top performer!")
                elif gary_taleb_rank <= 2:
                    report.append(f"• Gary×Taleb strategy ranks #{gary_taleb_rank} - Strong performance")
                else:
                    report.append(f"• Gary×Taleb strategy ranks #{gary_taleb_rank} - Consider optimization")

        report.append("• Focus on strategies with high statistical significance")
        report.append("• Consider robustness characteristics for live trading")
        report.append("• Monitor performance stability over time")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def plot_comparison_results(self, comparison_results: Dict[str, ComparisonResult],
                              strategy_data: Dict[str, Dict]) -> go.Figure:
        """Create interactive comparison visualization"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Strategy Performance Comparison', 'Risk-Return Profile',
                          'Statistical Significance Heatmap', 'Effect Size Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Extract strategy names and metrics
        strategies = list(strategy_data.keys())
        metrics = {}

        for strategy in strategies:
            returns = strategy_data[strategy]['returns']
            metrics[strategy] = {
                'total_return': (strategy_data[strategy]['equity_curve'].iloc[-1] /
                               strategy_data[strategy]['equity_curve'].iloc[0] - 1),
                'sharpe_ratio': self._calculate_sharpe(returns),
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': abs(self._calculate_max_drawdown(returns))
            }

        # 1. Performance comparison bar chart
        for i, strategy in enumerate(strategies):
            fig.add_trace(
                go.Bar(name=strategy,
                      x=['Total Return', 'Sharpe Ratio', 'Max Drawdown'],
                      y=[metrics[strategy]['total_return'],
                         metrics[strategy]['sharpe_ratio'],
                         metrics[strategy]['max_drawdown']]),
                row=1, col=1
            )

        # 2. Risk-Return scatter
        fig.add_trace(
            go.Scatter(
                x=[metrics[s]['volatility'] for s in strategies],
                y=[metrics[s]['total_return'] for s in strategies],
                mode='markers+text',
                text=strategies,
                textposition="top center",
                marker=dict(size=10, color=range(len(strategies)), colorscale='viridis'),
                name='Strategies'
            ),
            row=1, col=2
        )

        # 3. Statistical significance heatmap
        sig_matrix = np.zeros((len(strategies), len(strategies)))
        for comp_key, result in comparison_results.items():
            try:
                i = strategies.index(result.strategy_a)
                j = strategies.index(result.strategy_b)
                sig_tests = sum(1 for test in result.tests if test.is_significant)
                sig_matrix[i, j] = sig_tests / len(result.tests)
                sig_matrix[j, i] = sig_matrix[i, j]
            except ValueError:
                continue

        fig.add_trace(
            go.Heatmap(
                z=sig_matrix,
                x=strategies,
                y=strategies,
                colorscale='RdYlBu_r',
                showscale=True
            ),
            row=2, col=1
        )

        # 4. Effect size comparison
        effect_sizes = []
        comparison_names = []
        for comp_key, result in comparison_results.items():
            effect_sizes.append(abs(result.effect_size))
            comparison_names.append(f"{result.strategy_a[:10]} vs {result.strategy_b[:10]}")

        fig.add_trace(
            go.Bar(
                x=comparison_names,
                y=effect_sizes,
                name='Effect Size',
                marker_color='lightblue'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Strategy Comparison Dashboard",
            height=800,
            showlegend=True
        )

        fig.update_xaxes(title_text="Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Total Return", row=1, col=2)
        fig.update_xaxes(title_text="Comparisons", row=2, col=2)
        fig.update_yaxes(title_text="Effect Size", row=2, col=2)

        return fig

# Example usage
if __name__ == "__main__":
    # Create sample strategy data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    strategies_data = {}

    # Strategy A (Gary×Taleb)
    returns_a = np.random.normal(0.0008, 0.018, len(dates))
    strategies_data['Gary×Taleb'] = {
        'returns': pd.Series(returns_a, index=dates),
        'equity_curve': (1 + pd.Series(returns_a, index=dates)).cumprod() * 200
    }

    # Strategy B (Buy & Hold)
    returns_b = np.random.normal(0.0005, 0.020, len(dates))
    strategies_data['Buy & Hold'] = {
        'returns': pd.Series(returns_b, index=dates),
        'equity_curve': (1 + pd.Series(returns_b, index=dates)).cumprod() * 200
    }

    # Strategy C (Moving Average)
    returns_c = np.random.normal(0.0003, 0.022, len(dates))
    strategies_data['Moving Average'] = {
        'returns': pd.Series(returns_c, index=dates),
        'equity_curve': (1 + pd.Series(returns_c, index=dates)).cumprod() * 200
    }

    # Initialize comparison engine
    comparator = StrategyComparison(confidence_level=0.95)

    # Run comparisons
    comparison_results = comparator.compare_strategies(strategies_data)
    robustness_results = comparator.run_robustness_tests(strategies_data)

    # Generate report
    report = comparator.generate_comparison_report(comparison_results, robustness_results)
    print(report)

    # Create visualization
    fig = comparator.plot_comparison_results(comparison_results, strategies_data)
    fig.show()