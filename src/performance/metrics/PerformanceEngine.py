"""
Advanced Performance Metrics Calculation Engine
Specialized engine for trading strategy performance analysis with Sharpe optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Comprehensive risk assessment metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration_avg: float
    drawdown_duration_max: int
    ulcer_index: float
    pain_index: float
    burke_ratio: float
    martin_ratio: float
    downside_deviation: float
    semi_variance: float

@dataclass
class ReturnMetrics:
    """Return-based performance metrics"""
    total_return: float
    annualized_return: float
    cagr: float
    arithmetic_mean: float
    geometric_mean: float
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float
    positive_months: float
    gain_to_pain_ratio: float
    lake_ratio: float
    mountain_ratio: float

@dataclass
class RatioMetrics:
    """Performance ratio metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    excess_return_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    tracking_error: float
    omega_ratio: float
    kappa_three: float
    gain_loss_ratio: float
    payoff_ratio: float

@dataclass
class TalebMetrics:
    """Taleb-inspired antifragility and robustness metrics"""
    antifragility_score: float
    convexity_measure: float
    asymmetry_ratio: float
    tail_expectation_ratio: float
    black_swan_protection: float
    optionality_ratio: float
    fragility_index: float
    robustness_score: float
    barbell_efficiency: float
    via_negativa_score: float

@dataclass
class DPIMetrics:
    """Gary's Dynamic Performance Indicator metrics"""
    dpi_score: float
    momentum_component: float
    stability_component: float
    consistency_component: float
    adaptability_score: float
    signal_quality: float
    execution_efficiency: float
    market_alignment: float
    persistence_factor: float

class PerformanceEngine:
    """Advanced performance calculation engine with optimization capabilities"""

    def __init__(self, risk_free_rate: float = 0.02, benchmark_return: float = 0.10):
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        self.calculation_cache = {}

    def calculate_comprehensive_metrics(self, returns: pd.Series,
                                      equity_curve: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate all performance metrics comprehensively"""

        # Clean data
        returns = returns.dropna()
        equity_curve = equity_curve.dropna()

        if len(returns) == 0 or len(equity_curve) == 0:
            return self._empty_metrics()

        # Calculate core metrics
        risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
        return_metrics = self._calculate_return_metrics(returns, equity_curve)
        ratio_metrics = self._calculate_ratio_metrics(returns, benchmark_returns)
        taleb_metrics = self._calculate_taleb_metrics(returns)
        dpi_metrics = self._calculate_dpi_metrics(returns, equity_curve)

        return {
            'risk': risk_metrics,
            'returns': return_metrics,
            'ratios': ratio_metrics,
            'taleb': taleb_metrics,
            'dpi': dpi_metrics,
            'overall_score': self._calculate_overall_score(ratio_metrics, taleb_metrics, dpi_metrics)
        }

    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""

        # Value at Risk calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0

        # Drawdown calculations
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        drawdown_series = drawdown[drawdown < 0]

        max_drawdown = drawdown.min()
        avg_drawdown = drawdown_series.mean() if len(drawdown_series) > 0 else 0

        # Drawdown duration
        drawdown_durations = []
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            drawdown_durations.append(current_duration)

        drawdown_duration_avg = np.mean(drawdown_durations) if drawdown_durations else 0
        drawdown_duration_max = max(drawdown_durations) if drawdown_durations else 0

        # Ulcer Index
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())

        # Pain Index
        pain_index = np.sqrt(np.mean(drawdown_series ** 2)) if len(drawdown_series) > 0 else 0

        # Burke Ratio components
        burke_ratio = abs(np.sqrt(np.sum(drawdown_series ** 2))) if len(drawdown_series) > 0 else 0

        # Martin Ratio (Ulcer Performance Index)
        excess_return = returns.mean() - self.risk_free_rate / 252
        martin_ratio = excess_return / ulcer_index if ulcer_index > 0 else 0

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Semi-variance
        mean_return = returns.mean()
        negative_deviation = returns[returns < mean_return] - mean_return
        semi_variance = (negative_deviation ** 2).mean() if len(negative_deviation) > 0 else 0

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration_avg=drawdown_duration_avg,
            drawdown_duration_max=drawdown_duration_max,
            ulcer_index=ulcer_index,
            pain_index=pain_index,
            burke_ratio=burke_ratio,
            martin_ratio=martin_ratio,
            downside_deviation=downside_deviation,
            semi_variance=semi_variance
        )

    def _calculate_return_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> ReturnMetrics:
        """Calculate comprehensive return metrics"""

        # Basic return calculations
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        periods_per_year = 252  # Trading days
        years = len(returns) / periods_per_year

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years)) - 1 if years > 0 else 0

        # Mean calculations
        arithmetic_mean = returns.mean() * periods_per_year
        geometric_mean = ((1 + returns).prod() ** (1 / len(returns)) - 1) * periods_per_year

        # Monthly aggregations
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

        # Yearly aggregations
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        best_year = yearly_returns.max() if len(yearly_returns) > 0 else 0
        worst_year = yearly_returns.min() if len(yearly_returns) > 0 else 0

        # Gain to pain ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')

        # Lake and Mountain ratios
        peak = equity_curve.expanding().max()
        underwater = peak - equity_curve
        lake_ratio = underwater.sum() / (len(equity_curve) * equity_curve.iloc[-1])
        mountain_ratio = equity_curve.sum() / (len(equity_curve) * equity_curve.iloc[-1])

        return ReturnMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            arithmetic_mean=arithmetic_mean,
            geometric_mean=geometric_mean,
            best_month=best_month,
            worst_month=worst_month,
            best_year=best_year,
            worst_year=worst_year,
            positive_months=positive_months,
            gain_to_pain_ratio=gain_to_pain_ratio,
            lake_ratio=lake_ratio,
            mountain_ratio=mountain_ratio
        )

    def _calculate_ratio_metrics(self, returns: pd.Series,
                               benchmark_returns: Optional[pd.Series] = None) -> RatioMetrics:
        """Calculate performance ratio metrics"""

        # Annualized values
        excess_return = returns.mean() * 252 - self.risk_free_rate
        volatility = returns.std() * np.sqrt(252)

        # Basic ratios
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

        # Drawdown-based ratios
        peak = pd.Series(returns).cumsum().expanding().max()
        drawdown = pd.Series(returns).cumsum() - peak
        max_drawdown = abs(drawdown.min())

        calmar_ratio = (returns.mean() * 252) / max_drawdown if max_drawdown > 0 else 0
        sterling_ratio = calmar_ratio  # Simplified

        # Burke Ratio
        drawdown_squared = (drawdown ** 2).sum()
        burke_ratio = excess_return / np.sqrt(drawdown_squared) if drawdown_squared > 0 else 0

        # Benchmark-relative metrics
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_return = returns - benchmark_returns
            tracking_error = active_return.std() * np.sqrt(252)
            information_ratio = (active_return.mean() * 252) / tracking_error if tracking_error > 0 else 0

            # Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1

            treynor_ratio = excess_return / beta if beta > 0 else 0
            jensen_alpha = (returns.mean() * 252) - (self.risk_free_rate + beta * (self.benchmark_return - self.risk_free_rate))
        else:
            tracking_error = 0
            information_ratio = 0
            treynor_ratio = 0
            jensen_alpha = 0

        # Omega Ratio
        threshold = 0.0
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        omega_ratio = gains.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() < 0 else float('inf')

        # Kappa Three (similar to Omega but different threshold)
        kappa_threshold = returns.mean()
        kappa_gains = (returns - kappa_threshold)[returns > kappa_threshold]
        kappa_losses = (kappa_threshold - returns)[returns <= kappa_threshold]
        kappa_three = kappa_gains.sum() / kappa_losses.sum() if len(kappa_losses) > 0 and kappa_losses.sum() > 0 else float('inf')

        # Trade-based ratios
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        gain_loss_ratio = positive_returns.sum() / abs(negative_returns.sum()) if len(negative_returns) > 0 else float('inf')
        payoff_ratio = positive_returns.mean() / abs(negative_returns.mean()) if len(negative_returns) > 0 else float('inf')

        excess_return_ratio = excess_return / volatility if volatility > 0 else 0  # Same as Sharpe

        return RatioMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            excess_return_ratio=excess_return_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            omega_ratio=omega_ratio,
            kappa_three=kappa_three,
            gain_loss_ratio=gain_loss_ratio,
            payoff_ratio=payoff_ratio
        )

    def _calculate_taleb_metrics(self, returns: pd.Series) -> TalebMetrics:
        """Calculate Taleb-inspired antifragility metrics"""

        # Antifragility: benefits from volatility
        vol_periods = returns.rolling(20).std()
        return_periods = returns.rolling(20).mean()
        antifragility_correlation = vol_periods.corr(return_periods)
        antifragility_score = max(0, antifragility_correlation) if not np.isnan(antifragility_correlation) else 0

        # Convexity measure (Jensen's inequality)
        mean_return = returns.mean()
        convex_returns = np.exp(returns) - 1  # Exponential transformation
        convexity_measure = convex_returns.mean() - np.exp(mean_return) + 1

        # Asymmetry ratio (upside vs downside)
        upside_returns = returns[returns > returns.median()]
        downside_returns = returns[returns < returns.median()]
        asymmetry_ratio = (upside_returns.mean() / abs(downside_returns.mean())) if len(downside_returns) > 0 and downside_returns.mean() < 0 else 1

        # Tail expectation ratio
        upper_tail = returns[returns > np.percentile(returns, 95)]
        lower_tail = returns[returns < np.percentile(returns, 5)]
        tail_expectation_ratio = upper_tail.mean() / abs(lower_tail.mean()) if len(lower_tail) > 0 and lower_tail.mean() < 0 else 1

        # Black swan protection (negative skew preference in traditional sense, but positive for upside)
        skewness = returns.skew()
        black_swan_protection = max(0, skewness)  # Positive skew is good

        # Optionality ratio (non-linear payoffs)
        squared_returns = returns ** 2
        linear_returns = returns
        optionality_ratio = squared_returns.mean() / (linear_returns.mean() ** 2) if linear_returns.mean() != 0 else 0

        # Fragility index (sensitivity to large moves)
        large_moves = returns[abs(returns) > returns.std() * 2]
        fragility_index = len(large_moves) / len(returns) if len(returns) > 0 else 0

        # Robustness score (consistent performance across conditions)
        rolling_sharpe = []
        window = min(60, len(returns) // 4)  # Quarterly windows
        for i in range(window, len(returns)):
            period_returns = returns.iloc[i-window:i]
            period_sharpe = (period_returns.mean() * 252) / (period_returns.std() * np.sqrt(252)) if period_returns.std() > 0 else 0
            rolling_sharpe.append(period_sharpe)

        robustness_score = 1 - (np.std(rolling_sharpe) / np.mean(rolling_sharpe)) if len(rolling_sharpe) > 0 and np.mean(rolling_sharpe) != 0 else 0
        robustness_score = max(0, robustness_score)

        # Barbell efficiency (combination of safe and risky bets)
        conservative_returns = returns[(returns >= np.percentile(returns, 10)) & (returns <= np.percentile(returns, 90))]
        extreme_returns = returns[(returns < np.percentile(returns, 10)) | (returns > np.percentile(returns, 90))]

        conservative_ratio = len(conservative_returns) / len(returns)
        extreme_upside = extreme_returns[extreme_returns > 0].sum() if len(extreme_returns[extreme_returns > 0]) > 0 else 0
        extreme_downside = abs(extreme_returns[extreme_returns < 0].sum()) if len(extreme_returns[extreme_returns < 0]) > 0 else 0

        barbell_efficiency = (conservative_ratio * 0.5 + (extreme_upside / max(extreme_downside, 0.001)) * 0.5)

        # Via Negativa score (what you don't lose)
        max_single_loss = abs(returns.min())
        total_positive = returns[returns > 0].sum()
        via_negativa_score = 1 - (max_single_loss / total_positive) if total_positive > 0 else 0
        via_negativa_score = max(0, via_negativa_score)

        return TalebMetrics(
            antifragility_score=antifragility_score,
            convexity_measure=convexity_measure,
            asymmetry_ratio=asymmetry_ratio,
            tail_expectation_ratio=tail_expectation_ratio,
            black_swan_protection=black_swan_protection,
            optionality_ratio=optionality_ratio,
            fragility_index=fragility_index,
            robustness_score=robustness_score,
            barbell_efficiency=barbell_efficiency,
            via_negativa_score=via_negativa_score
        )

    def _calculate_dpi_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> DPIMetrics:
        """Calculate Gary's Dynamic Performance Indicator metrics"""

        # Momentum component
        rolling_returns = returns.rolling(20).mean()
        momentum_trend = rolling_returns.diff().mean()
        momentum_component = np.tanh(momentum_trend * 100)  # Normalize to [-1, 1]

        # Stability component
        rolling_volatility = returns.rolling(20).std()
        volatility_stability = 1 - (rolling_volatility.std() / rolling_volatility.mean()) if rolling_volatility.mean() > 0 else 0
        stability_component = max(0, min(1, volatility_stability))

        # Consistency component
        rolling_sharpe = []
        for i in range(20, len(returns)):
            period_returns = returns.iloc[i-20:i]
            period_sharpe = (period_returns.mean() * 252) / (period_returns.std() * np.sqrt(252)) if period_returns.std() > 0 else 0
            rolling_sharpe.append(period_sharpe)

        consistency_component = 1 - (np.std(rolling_sharpe) / (abs(np.mean(rolling_sharpe)) + 0.001)) if len(rolling_sharpe) > 0 else 0
        consistency_component = max(0, min(1, consistency_component))

        # Adaptability score (performance in different market conditions)
        high_vol_periods = returns[returns.rolling(20).std() > returns.rolling(60).std().mean()]
        low_vol_periods = returns[returns.rolling(20).std() <= returns.rolling(60).std().mean()]

        high_vol_sharpe = (high_vol_periods.mean() * 252) / (high_vol_periods.std() * np.sqrt(252)) if len(high_vol_periods) > 0 and high_vol_periods.std() > 0 else 0
        low_vol_sharpe = (low_vol_periods.mean() * 252) / (low_vol_periods.std() * np.sqrt(252)) if len(low_vol_periods) > 0 and low_vol_periods.std() > 0 else 0

        adaptability_score = min(high_vol_sharpe, low_vol_sharpe) / max(abs(high_vol_sharpe), abs(low_vol_sharpe), 0.001)
        adaptability_score = max(0, min(1, (adaptability_score + 1) / 2))  # Normalize to [0, 1]

        # Signal quality (hit rate and magnitude)
        positive_returns = returns[returns > 0]
        signal_quality = len(positive_returns) / len(returns) if len(returns) > 0 else 0

        # Execution efficiency (actual vs theoretical performance)
        theoretical_returns = returns.abs()  # Perfect timing
        actual_returns = returns
        execution_efficiency = actual_returns.mean() / theoretical_returns.mean() if theoretical_returns.mean() > 0 else 0
        execution_efficiency = max(0, min(1, execution_efficiency))

        # Market alignment (correlation with favorable conditions)
        market_trends = returns.rolling(10).mean().diff()
        strategy_alignment = returns.corr(market_trends) if not market_trends.isna().all() else 0
        market_alignment = max(0, (strategy_alignment + 1) / 2)  # Normalize to [0, 1]

        # Persistence factor (ability to maintain performance)
        equity_growth = equity_curve.pct_change().fillna(0)
        positive_streaks = []
        current_streak = 0

        for growth in equity_growth:
            if growth > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    positive_streaks.append(current_streak)
                current_streak = 0

        if current_streak > 0:
            positive_streaks.append(current_streak)

        avg_positive_streak = np.mean(positive_streaks) if positive_streaks else 0
        persistence_factor = min(1, avg_positive_streak / 10)  # Normalize to [0, 1]

        # Overall DPI score
        dpi_score = (momentum_component * 0.2 +
                    stability_component * 0.2 +
                    consistency_component * 0.15 +
                    adaptability_score * 0.15 +
                    signal_quality * 0.1 +
                    execution_efficiency * 0.1 +
                    market_alignment * 0.05 +
                    persistence_factor * 0.05)

        return DPIMetrics(
            dpi_score=dpi_score,
            momentum_component=momentum_component,
            stability_component=stability_component,
            consistency_component=consistency_component,
            adaptability_score=adaptability_score,
            signal_quality=signal_quality,
            execution_efficiency=execution_efficiency,
            market_alignment=market_alignment,
            persistence_factor=persistence_factor
        )

    def _calculate_overall_score(self, ratio_metrics: RatioMetrics,
                               taleb_metrics: TalebMetrics,
                               dpi_metrics: DPIMetrics) -> float:
        """Calculate overall performance score"""

        # Normalize Sharpe ratio (target: 2.0)
        normalized_sharpe = min(1, ratio_metrics.sharpe_ratio / 2.0) if ratio_metrics.sharpe_ratio > 0 else 0

        # Weight the components
        overall_score = (
            normalized_sharpe * 0.3 +           # 30% Sharpe ratio
            taleb_metrics.antifragility_score * 0.25 +  # 25% Antifragility
            dpi_metrics.dpi_score * 0.25 +      # 25% DPI score
            ratio_metrics.calmar_ratio / 5 * 0.1 +  # 10% Calmar ratio
            taleb_metrics.robustness_score * 0.1     # 10% Robustness
        )

        return max(0, min(1, overall_score))

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'risk': RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            'returns': ReturnMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            'ratios': RatioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            'taleb': TalebMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            'dpi': DPIMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0),
            'overall_score': 0.0
        }

    def optimize_sharpe_ratio(self, returns_history: pd.DataFrame,
                            strategy_params: Dict,
                            param_bounds: Dict) -> Tuple[Dict, float]:
        """Optimize strategy parameters for maximum Sharpe ratio"""

        def objective(params):
            # Update strategy with new parameters
            strategy_params.update(dict(zip(param_bounds.keys(), params)))

            # Simulate strategy with new parameters (simplified)
            # In practice, this would call the actual strategy backtest
            simulated_returns = self._simulate_strategy_returns(returns_history, strategy_params)

            # Calculate Sharpe ratio
            excess_return = simulated_returns.mean() * 252 - self.risk_free_rate
            volatility = simulated_returns.std() * np.sqrt(252)
            sharpe = excess_return / volatility if volatility > 0 else -999

            return -sharpe  # Minimize negative Sharpe

        # Set up optimization
        initial_params = list(param_bounds.values())
        bounds = [(min_val, max_val) for min_val, max_val in param_bounds.values()]

        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')

        # Extract optimized parameters
        optimized_params = dict(zip(param_bounds.keys(), result.x))
        optimized_sharpe = -result.fun

        return optimized_params, optimized_sharpe

    def _simulate_strategy_returns(self, price_data: pd.DataFrame, params: Dict) -> pd.Series:
        """Simulate strategy returns with given parameters (simplified implementation)"""

        # This is a simplified simulation - in practice, you'd call your actual strategy
        returns = price_data['Close'].pct_change().dropna()

        # Apply some parameter-based modifications
        momentum_factor = params.get('momentum_factor', 1.0)
        params.get('volatility_factor', 1.0)

        # Simple momentum strategy simulation
        momentum_signal = returns.rolling(20).mean()
        strategy_returns = momentum_signal.shift(1) * returns * momentum_factor

        return strategy_returns.dropna()

    def generate_optimization_report(self, original_params: Dict,
                                   optimized_params: Dict,
                                   original_sharpe: float,
                                   optimized_sharpe: float) -> str:
        """Generate optimization report"""

        improvement = ((optimized_sharpe / original_sharpe) - 1) * 100 if original_sharpe > 0 else 0

        report = []
        report.append("SHARPE RATIO OPTIMIZATION REPORT")
        report.append("=" * 40)
        report.append(f"Original Sharpe Ratio: {original_sharpe:.3f}")
        report.append(f"Optimized Sharpe Ratio: {optimized_sharpe:.3f}")
        report.append(f"Improvement: {improvement:.1f}%")
        report.append("")

        report.append("PARAMETER CHANGES:")
        report.append("-" * 20)
        for param, orig_value in original_params.items():
            opt_value = optimized_params.get(param, orig_value)
            change = ((opt_value / orig_value) - 1) * 100 if orig_value != 0 else 0
            report.append(f"{param}: {orig_value:.3f} -> {opt_value:.3f} ({change:+.1f}%)")

        report.append("")
        if improvement > 5:
            report.append("RECOMMENDATION: Apply optimized parameters")
        elif improvement > 0:
            report.append("RECOMMENDATION: Consider applying optimized parameters")
        else:
            report.append("RECOMMENDATION: Keep original parameters")

        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 200

    # Initialize engine
    engine = PerformanceEngine()

    # Calculate comprehensive metrics
    metrics = engine.calculate_comprehensive_metrics(
        pd.Series(returns, index=dates),
        equity_curve
    )

    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 40)
    print(f"Overall Score: {metrics['overall_score']:.3f}")
    print(f"Sharpe Ratio: {metrics['ratios'].sharpe_ratio:.3f}")
    print(f"Antifragility Score: {metrics['taleb'].antifragility_score:.3f}")
    print(f"DPI Score: {metrics['dpi'].dpi_score:.3f}")
    print(f"Max Drawdown: {metrics['risk'].max_drawdown:.2%}")