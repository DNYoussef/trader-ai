"""
Antifragility Metrics and DPI Analysis Integration
Comprehensive analysis system for Gary×Taleb strategy components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

warnings.filterwarnings('ignore')

@dataclass
class AntifragilityMetrics:
    """Comprehensive antifragility metrics"""
    # Core antifragility measures
    convexity_score: float
    volatility_benefit: float
    asymmetry_ratio: float
    tail_expectation_ratio: float
    stress_test_performance: float

    # Taleb-inspired measures
    barbell_efficiency: float
    optionality_ratio: float
    fragility_index: float
    via_negativa_score: float
    black_swan_protection: float

    # Dynamic measures
    regime_adaptability: float
    volatility_scaling: float
    antifragile_momentum: float
    resilience_factor: float
    antifragile_persistence: float

@dataclass
class DPIMetrics:
    """Gary's Dynamic Performance Indicator metrics"""
    # Core DPI components
    momentum_score: float
    stability_score: float
    consistency_score: float
    adaptability_score: float
    signal_quality: float

    # Advanced DPI measures
    execution_efficiency: float
    market_alignment: float
    persistence_factor: float
    coherence_measure: float
    predictive_power: float

    # Integration measures
    risk_adjusted_dpi: float
    volatility_normalized_dpi: float
    regime_sensitive_dpi: float
    forward_looking_dpi: float
    composite_dpi_score: float

@dataclass
class IntegratedAnalysis:
    """Integrated Gary×Taleb analysis result"""
    timestamp: pd.Timestamp
    antifragility_metrics: AntifragilityMetrics
    dpi_metrics: DPIMetrics
    synergy_score: float
    combined_effectiveness: float
    risk_return_profile: Dict[str, float]
    optimization_opportunities: List[str]
    strategic_recommendations: List[str]

class AntifragilityAnalyzer:
    """Advanced antifragility and DPI analysis system"""

    def __init__(self):
        self.analysis_cache = {}
        self.regime_detector = None
        self.antifragility_model = None
        self.dpi_model = None

    def calculate_antifragility_metrics(self, returns: pd.Series,
                                      volatility_data: Optional[pd.Series] = None,
                                      market_data: Optional[Dict[str, pd.Series]] = None) -> AntifragilityMetrics:
        """Calculate comprehensive antifragility metrics"""

        print("Calculating comprehensive antifragility metrics...")

        # Prepare data
        clean_returns = returns.dropna()
        if volatility_data is None:
            volatility_data = returns.rolling(20).std()

        # Core antifragility measures
        convexity_score = self._calculate_convexity_score(clean_returns)
        volatility_benefit = self._calculate_volatility_benefit(clean_returns, volatility_data)
        asymmetry_ratio = self._calculate_asymmetry_ratio(clean_returns)
        tail_expectation_ratio = self._calculate_tail_expectation_ratio(clean_returns)
        stress_test_performance = self._calculate_stress_test_performance(clean_returns)

        # Taleb-inspired measures
        barbell_efficiency = self._calculate_barbell_efficiency(clean_returns)
        optionality_ratio = self._calculate_optionality_ratio(clean_returns)
        fragility_index = self._calculate_fragility_index(clean_returns)
        via_negativa_score = self._calculate_via_negativa_score(clean_returns)
        black_swan_protection = self._calculate_black_swan_protection(clean_returns)

        # Dynamic measures
        regime_adaptability = self._calculate_regime_adaptability(clean_returns, market_data)
        volatility_scaling = self._calculate_volatility_scaling(clean_returns, volatility_data)
        antifragile_momentum = self._calculate_antifragile_momentum(clean_returns)
        resilience_factor = self._calculate_resilience_factor(clean_returns)
        antifragile_persistence = self._calculate_antifragile_persistence(clean_returns)

        return AntifragilityMetrics(
            convexity_score=convexity_score,
            volatility_benefit=volatility_benefit,
            asymmetry_ratio=asymmetry_ratio,
            tail_expectation_ratio=tail_expectation_ratio,
            stress_test_performance=stress_test_performance,
            barbell_efficiency=barbell_efficiency,
            optionality_ratio=optionality_ratio,
            fragility_index=fragility_index,
            via_negativa_score=via_negativa_score,
            black_swan_protection=black_swan_protection,
            regime_adaptability=regime_adaptability,
            volatility_scaling=volatility_scaling,
            antifragile_momentum=antifragile_momentum,
            resilience_factor=resilience_factor,
            antifragile_persistence=antifragile_persistence
        )

    def calculate_dpi_metrics(self, returns: pd.Series,
                            signal_data: Optional[Dict[str, pd.Series]] = None,
                            execution_data: Optional[Dict[str, pd.Series]] = None) -> DPIMetrics:
        """Calculate comprehensive DPI metrics"""

        print("Calculating comprehensive DPI metrics...")

        clean_returns = returns.dropna()

        # Core DPI components
        momentum_score = self._calculate_momentum_score(clean_returns)
        stability_score = self._calculate_stability_score(clean_returns)
        consistency_score = self._calculate_consistency_score(clean_returns)
        adaptability_score = self._calculate_adaptability_score(clean_returns)
        signal_quality = self._calculate_signal_quality(clean_returns, signal_data)

        # Advanced DPI measures
        execution_efficiency = self._calculate_execution_efficiency(clean_returns, execution_data)
        market_alignment = self._calculate_market_alignment(clean_returns)
        persistence_factor = self._calculate_persistence_factor(clean_returns)
        coherence_measure = self._calculate_coherence_measure(clean_returns)
        predictive_power = self._calculate_predictive_power(clean_returns)

        # Integration measures
        risk_adjusted_dpi = self._calculate_risk_adjusted_dpi(clean_returns)
        volatility_normalized_dpi = self._calculate_volatility_normalized_dpi(clean_returns)
        regime_sensitive_dpi = self._calculate_regime_sensitive_dpi(clean_returns)
        forward_looking_dpi = self._calculate_forward_looking_dpi(clean_returns)

        # Composite score
        composite_dpi_score = self._calculate_composite_dpi_score([
            momentum_score, stability_score, consistency_score,
            adaptability_score, signal_quality, execution_efficiency
        ])

        return DPIMetrics(
            momentum_score=momentum_score,
            stability_score=stability_score,
            consistency_score=consistency_score,
            adaptability_score=adaptability_score,
            signal_quality=signal_quality,
            execution_efficiency=execution_efficiency,
            market_alignment=market_alignment,
            persistence_factor=persistence_factor,
            coherence_measure=coherence_measure,
            predictive_power=predictive_power,
            risk_adjusted_dpi=risk_adjusted_dpi,
            volatility_normalized_dpi=volatility_normalized_dpi,
            regime_sensitive_dpi=regime_sensitive_dpi,
            forward_looking_dpi=forward_looking_dpi,
            composite_dpi_score=composite_dpi_score
        )

    def perform_integrated_analysis(self, returns: pd.Series,
                                  antifragility_metrics: AntifragilityMetrics,
                                  dpi_metrics: DPIMetrics,
                                  market_context: Optional[Dict] = None) -> IntegratedAnalysis:
        """Perform integrated Gary×Taleb analysis"""

        print("Performing integrated Gary×Taleb analysis...")

        timestamp = returns.index[-1] if not returns.empty else pd.Timestamp.now()

        # Calculate synergy between antifragility and DPI
        synergy_score = self._calculate_antifragility_dpi_synergy(antifragility_metrics, dpi_metrics)

        # Combined effectiveness
        combined_effectiveness = self._calculate_combined_effectiveness(
            antifragility_metrics, dpi_metrics, returns
        )

        # Risk-return profile analysis
        risk_return_profile = self._analyze_risk_return_profile(
            returns, antifragility_metrics, dpi_metrics
        )

        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            antifragility_metrics, dpi_metrics, returns
        )

        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            antifragility_metrics, dpi_metrics, market_context
        )

        return IntegratedAnalysis(
            timestamp=timestamp,
            antifragility_metrics=antifragility_metrics,
            dpi_metrics=dpi_metrics,
            synergy_score=synergy_score,
            combined_effectiveness=combined_effectiveness,
            risk_return_profile=risk_return_profile,
            optimization_opportunities=optimization_opportunities,
            strategic_recommendations=strategic_recommendations
        )

    # Antifragility calculation methods
    def _calculate_convexity_score(self, returns: pd.Series) -> float:
        """Calculate convexity score (Jensen's inequality benefit)"""
        if len(returns) < 20:
            return 0.5

        # Test for convex payoff: E[f(X)] > f(E[X])
        # Using exponential transformation as proxy for convex payoff
        mean_return = returns.mean()
        exp_returns = np.exp(returns) - 1
        mean_exp_return = exp_returns.mean()
        exp_mean_return = np.exp(mean_return) - 1

        convexity = (mean_exp_return - exp_mean_return) / abs(exp_mean_return) if exp_mean_return != 0 else 0
        return max(0, min(1, convexity + 0.5))

    def _calculate_volatility_benefit(self, returns: pd.Series, volatility_data: pd.Series) -> float:
        """Calculate how much the strategy benefits from volatility"""
        common_idx = returns.index.intersection(volatility_data.index)
        if len(common_idx) < 20:
            return 0.5

        aligned_returns = returns.loc[common_idx]
        aligned_vol = volatility_data.loc[common_idx]

        correlation = aligned_returns.corr(aligned_vol)
        return max(0, (correlation + 1) / 2) if not np.isnan(correlation) else 0.5

    def _calculate_asymmetry_ratio(self, returns: pd.Series) -> float:
        """Calculate upside vs downside asymmetry"""
        upside = returns[returns > 0]
        downside = returns[returns < 0]

        if len(upside) == 0 or len(downside) == 0:
            return 0.5

        upside_mean = upside.mean()
        downside_mean = abs(downside.mean())

        asymmetry = upside_mean / (upside_mean + downside_mean)
        return asymmetry

    def _calculate_tail_expectation_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio of positive to negative tail expectations"""
        upper_tail = returns[returns > returns.quantile(0.95)]
        lower_tail = returns[returns < returns.quantile(0.05)]

        if len(upper_tail) == 0 or len(lower_tail) == 0:
            return 0.5

        upper_expectation = upper_tail.mean()
        lower_expectation = abs(lower_tail.mean())

        tail_ratio = upper_expectation / (upper_expectation + lower_expectation)
        return tail_ratio

    def _calculate_stress_test_performance(self, returns: pd.Series) -> float:
        """Calculate performance during stress periods"""
        # Define stress as periods with high volatility
        vol_threshold = returns.rolling(20).std().quantile(0.8)
        stress_periods = returns.rolling(20).std() > vol_threshold

        if stress_periods.sum() == 0:
            return 0.5

        stress_returns = returns[stress_periods]
        normal_returns = returns[~stress_periods]

        if len(normal_returns) == 0:
            return 0.5

        stress_performance = stress_returns.mean()
        normal_performance = normal_returns.mean()

        # Relative performance during stress
        if normal_performance != 0:
            relative_performance = stress_performance / normal_performance
            return max(0, min(1, (relative_performance + 1) / 2))
        else:
            return 0.5

    def _calculate_barbell_efficiency(self, returns: pd.Series) -> float:
        """Calculate efficiency of barbell strategy (safe + risky combination)"""
        # Analyze distribution of returns for barbell pattern
        conservative_returns = returns[(returns >= returns.quantile(0.1)) & (returns <= returns.quantile(0.9))]
        extreme_returns = returns[(returns < returns.quantile(0.1)) | (returns > returns.quantile(0.9))]

        conservative_ratio = len(conservative_returns) / len(returns)
        extreme_positive = extreme_returns[extreme_returns > 0].sum() if len(extreme_returns) > 0 else 0
        extreme_negative = abs(extreme_returns[extreme_returns < 0].sum()) if len(extreme_returns) > 0 else 0.001

        barbell_score = (conservative_ratio * 0.5) + (extreme_positive / (extreme_positive + extreme_negative) * 0.5)
        return barbell_score

    def _calculate_optionality_ratio(self, returns: pd.Series) -> float:
        """Calculate optionality (non-linear payoffs)"""
        # Measure non-linearity in return distribution
        squared_returns = returns ** 2
        linear_returns = returns

        # Compare variance of squared vs linear returns
        if linear_returns.var() == 0:
            return 0.5

        optionality = squared_returns.var() / (linear_returns.var() ** 2)
        return min(1, optionality / 10)  # Normalize

    def _calculate_fragility_index(self, returns: pd.Series) -> float:
        """Calculate fragility index (sensitivity to large moves)"""
        large_moves = returns[abs(returns) > returns.std() * 2]
        fragility = len(large_moves) / len(returns)

        # Lower fragility is better for antifragile systems
        return 1 - min(1, fragility * 5)

    def _calculate_via_negativa_score(self, returns: pd.Series) -> float:
        """Calculate via negativa score (what you don't lose)"""
        max_single_loss = abs(returns.min())
        total_gains = returns[returns > 0].sum()

        if total_gains == 0:
            return 0

        via_negativa = 1 - (max_single_loss / total_gains)
        return max(0, via_negativa)

    def _calculate_black_swan_protection(self, returns: pd.Series) -> float:
        """Calculate protection against black swan events"""
        # Measure tail behavior and skewness
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Positive skew and moderate kurtosis indicate good tail protection
        skew_score = max(0, (skewness + 1) / 2)
        kurt_score = max(0, 1 - abs(kurtosis) / 10)

        protection_score = (skew_score * 0.7) + (kurt_score * 0.3)
        return protection_score

    def _calculate_regime_adaptability(self, returns: pd.Series, market_data: Optional[Dict] = None) -> float:
        """Calculate adaptability across different market regimes"""
        if len(returns) < 60:
            return 0.5

        # Divide into thirds for regime analysis
        n = len(returns)
        third = n // 3

        regime1 = returns[:third]
        regime2 = returns[third:2*third]
        regime3 = returns[2*third:]

        # Calculate Sharpe ratios for each regime
        sharpe_ratios = []
        for regime in [regime1, regime2, regime3]:
            if regime.std() > 0:
                sharpe = regime.mean() / regime.std()
                sharpe_ratios.append(sharpe)

        if len(sharpe_ratios) == 0:
            return 0.5

        # Consistent performance across regimes indicates good adaptability
        consistency = 1 - (np.std(sharpe_ratios) / (abs(np.mean(sharpe_ratios)) + 0.001))
        return max(0, min(1, consistency))

    def _calculate_volatility_scaling(self, returns: pd.Series, volatility_data: pd.Series) -> float:
        """Calculate how well strategy scales with volatility"""
        common_idx = returns.index.intersection(volatility_data.index)
        if len(common_idx) < 20:
            return 0.5

        aligned_returns = returns.loc[common_idx]
        aligned_vol = volatility_data.loc[common_idx]

        # Calculate rolling correlation
        correlation = aligned_returns.rolling(20).corr(aligned_vol).mean()
        return max(0, (correlation + 1) / 2) if not np.isnan(correlation) else 0.5

    def _calculate_antifragile_momentum(self, returns: pd.Series) -> float:
        """Calculate momentum in antifragile characteristics"""
        if len(returns) < 40:
            return 0.5

        # Compare recent vs historical antifragile properties
        recent_returns = returns.iloc[-20:]
        historical_returns = returns.iloc[:-20]

        recent_skew = recent_returns.skew()
        historical_skew = historical_returns.skew()

        momentum = (recent_skew - historical_skew) / (abs(historical_skew) + 0.001)
        return max(0, min(1, (momentum + 1) / 2))

    def _calculate_resilience_factor(self, returns: pd.Series) -> float:
        """Calculate resilience (recovery from drawdowns)"""
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        # Find recovery periods
        recovery_times = []
        in_drawdown = False
        drawdown_start = None

        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:  # Recovery
                in_drawdown = False
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)

        if not recovery_times:
            return 0.8  # No significant drawdowns

        avg_recovery = np.mean(recovery_times)
        # Faster recovery = higher resilience
        resilience = max(0, 1 - avg_recovery / 50)  # Normalize to 50 days
        return resilience

    def _calculate_antifragile_persistence(self, returns: pd.Series) -> float:
        """Calculate persistence of antifragile characteristics"""
        if len(returns) < 60:
            return 0.5

        window = 30
        antifragile_scores = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_skew = window_returns.skew()
            antifragile_scores.append(max(0, window_skew))

        if not antifragile_scores:
            return 0.5

        # Measure consistency of antifragile characteristics
        persistence = 1 - (np.std(antifragile_scores) / (np.mean(antifragile_scores) + 0.001))
        return max(0, min(1, persistence))

    # DPI calculation methods
    def _calculate_momentum_score(self, returns: pd.Series) -> float:
        """Calculate momentum component of DPI"""
        if len(returns) < 20:
            return 0.5

        short_momentum = returns.rolling(5).mean().iloc[-1]
        long_momentum = returns.rolling(20).mean().iloc[-1]

        momentum_trend = short_momentum - long_momentum
        momentum_score = np.tanh(momentum_trend * 1000)  # Scale and normalize

        return (momentum_score + 1) / 2  # Convert to [0, 1]

    def _calculate_stability_score(self, returns: pd.Series) -> float:
        """Calculate stability component of DPI"""
        if len(returns) < 20:
            return 0.5

        rolling_vol = returns.rolling(10).std()
        vol_stability = 1 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0

        return max(0, min(1, vol_stability))

    def _calculate_consistency_score(self, returns: pd.Series) -> float:
        """Calculate consistency component of DPI"""
        if len(returns) < 30:
            return 0.5

        # Rolling Sharpe ratios
        window = 15
        rolling_sharpe = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if window_returns.std() > 0:
                sharpe = window_returns.mean() / window_returns.std()
                rolling_sharpe.append(sharpe)

        if not rolling_sharpe:
            return 0.5

        consistency = 1 - (np.std(rolling_sharpe) / (abs(np.mean(rolling_sharpe)) + 0.001))
        return max(0, min(1, consistency))

    def _calculate_adaptability_score(self, returns: pd.Series) -> float:
        """Calculate adaptability component of DPI"""
        if len(returns) < 40:
            return 0.5

        # Performance in different volatility regimes
        vol_series = returns.rolling(20).std()
        vol_median = vol_series.median()

        low_vol_returns = returns[vol_series <= vol_median]
        high_vol_returns = returns[vol_series > vol_median]

        if len(low_vol_returns) == 0 or len(high_vol_returns) == 0:
            return 0.5

        low_vol_sharpe = low_vol_returns.mean() / low_vol_returns.std() if low_vol_returns.std() > 0 else 0
        high_vol_sharpe = high_vol_returns.mean() / high_vol_returns.std() if high_vol_returns.std() > 0 else 0

        # Good adaptability means positive performance in both regimes
        adaptability = min(low_vol_sharpe, high_vol_sharpe) / max(abs(low_vol_sharpe), abs(high_vol_sharpe), 0.001)
        return max(0, min(1, (adaptability + 1) / 2))

    def _calculate_signal_quality(self, returns: pd.Series, signal_data: Optional[Dict] = None) -> float:
        """Calculate signal quality component of DPI"""
        # Basic signal quality based on hit rate
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)

        hit_rate = positive_returns / total_returns if total_returns > 0 else 0.5

        # Enhanced with signal strength if available
        if signal_data and 'signal_strength' in signal_data:
            signal_strength = signal_data['signal_strength']
            common_idx = returns.index.intersection(signal_strength.index)

            if len(common_idx) > 10:
                strong_signals = signal_strength.loc[common_idx] > 0.7
                strong_signal_returns = returns.loc[common_idx][strong_signals]

                if len(strong_signal_returns) > 0:
                    strong_hit_rate = (strong_signal_returns > 0).sum() / len(strong_signal_returns)
                    hit_rate = (hit_rate + strong_hit_rate) / 2

        return hit_rate

    def _calculate_execution_efficiency(self, returns: pd.Series, execution_data: Optional[Dict] = None) -> float:
        """Calculate execution efficiency component of DPI"""
        # Basic efficiency based on return smoothness
        if len(returns) < 10:
            return 0.5

        # Measure how much actual returns deviate from smooth trend
        smooth_returns = returns.rolling(5, center=True).mean()
        deviations = abs(returns - smooth_returns).dropna()

        efficiency = 1 - (deviations.mean() / returns.std()) if returns.std() > 0 else 0.5
        return max(0, min(1, efficiency))

    def _calculate_market_alignment(self, returns: pd.Series) -> float:
        """Calculate market alignment component of DPI"""
        # Simplified market alignment using autocorrelation
        if len(returns) < 20:
            return 0.5

        # Positive autocorrelation suggests good trend following
        autocorr = returns.autocorr(lag=1)
        alignment = (autocorr + 1) / 2 if not np.isnan(autocorr) else 0.5

        return alignment

    def _calculate_persistence_factor(self, returns: pd.Series) -> float:
        """Calculate persistence factor component of DPI"""
        if len(returns) < 30:
            return 0.5

        # Measure persistence of positive performance
        positive_streaks = []
        current_streak = 0

        for ret in returns:
            if ret > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    positive_streaks.append(current_streak)
                current_streak = 0

        if current_streak > 0:
            positive_streaks.append(current_streak)

        avg_streak = np.mean(positive_streaks) if positive_streaks else 0
        persistence = min(1, avg_streak / 5)  # Normalize to 5-day streaks

        return persistence

    def _calculate_coherence_measure(self, returns: pd.Series) -> float:
        """Calculate coherence measure for DPI"""
        if len(returns) < 20:
            return 0.5

        # Measure coherence between short and long-term performance
        short_term = returns.rolling(5).mean()
        long_term = returns.rolling(20).mean()

        coherence = short_term.corr(long_term)
        return (coherence + 1) / 2 if not np.isnan(coherence) else 0.5

    def _calculate_predictive_power(self, returns: pd.Series) -> float:
        """Calculate predictive power component of DPI"""
        if len(returns) < 30:
            return 0.5

        # Simple predictive power using lagged correlation
        lagged_returns = returns.shift(1)
        predictive_corr = returns.corr(lagged_returns)

        # Transform correlation to [0, 1] scale
        return abs(predictive_corr) if not np.isnan(predictive_corr) else 0.5

    def _calculate_risk_adjusted_dpi(self, returns: pd.Series) -> float:
        """Calculate risk-adjusted DPI"""
        if returns.std() == 0:
            return 0.5

        sharpe_ratio = returns.mean() / returns.std()
        # Transform Sharpe to [0, 1] scale
        risk_adjusted = np.tanh(sharpe_ratio)
        return (risk_adjusted + 1) / 2

    def _calculate_volatility_normalized_dpi(self, returns: pd.Series) -> float:
        """Calculate volatility-normalized DPI"""
        if len(returns) < 20:
            return 0.5

        vol_adjusted_returns = returns / returns.rolling(20).std()
        vol_normalized = vol_adjusted_returns.mean()

        return np.tanh(vol_normalized * 10) / 2 + 0.5

    def _calculate_regime_sensitive_dpi(self, returns: pd.Series) -> float:
        """Calculate regime-sensitive DPI"""
        if len(returns) < 40:
            return 0.5

        # Calculate performance in different volatility regimes
        vol_series = returns.rolling(10).std()
        low_vol_mask = vol_series <= vol_series.quantile(0.5)

        low_vol_performance = returns[low_vol_mask].mean()
        high_vol_performance = returns[~low_vol_mask].mean()

        # Balanced performance across regimes
        regime_score = min(low_vol_performance, high_vol_performance) / max(abs(low_vol_performance), abs(high_vol_performance), 0.001)
        return max(0, min(1, (regime_score + 1) / 2))

    def _calculate_forward_looking_dpi(self, returns: pd.Series) -> float:
        """Calculate forward-looking DPI"""
        if len(returns) < 30:
            return 0.5

        # Use recent performance to predict future
        recent_performance = returns.iloc[-10:].mean()
        historical_performance = returns.iloc[:-10].mean()

        momentum = recent_performance - historical_performance
        forward_looking = np.tanh(momentum * 1000)

        return (forward_looking + 1) / 2

    def _calculate_composite_dpi_score(self, component_scores: List[float]) -> float:
        """Calculate composite DPI score from components"""
        if not component_scores:
            return 0.5

        # Weighted average of components
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]  # Adjust based on importance
        weights = weights[:len(component_scores)]

        if sum(weights) > 0:
            composite = sum(score * weight for score, weight in zip(component_scores, weights)) / sum(weights)
        else:
            composite = np.mean(component_scores)

        return composite

    # Integration methods
    def _calculate_antifragility_dpi_synergy(self, antifragility_metrics: AntifragilityMetrics,
                                           dpi_metrics: DPIMetrics) -> float:
        """Calculate synergy between antifragility and DPI"""

        # Key synergistic relationships
        volatility_synergy = antifragility_metrics.volatility_benefit * dpi_metrics.adaptability_score
        stability_synergy = antifragility_metrics.resilience_factor * dpi_metrics.stability_score
        momentum_synergy = antifragility_metrics.antifragile_momentum * dpi_metrics.momentum_score
        consistency_synergy = antifragility_metrics.antifragile_persistence * dpi_metrics.consistency_score

        synergy_score = np.mean([volatility_synergy, stability_synergy, momentum_synergy, consistency_synergy])
        return synergy_score

    def _calculate_combined_effectiveness(self, antifragility_metrics: AntifragilityMetrics,
                                        dpi_metrics: DPIMetrics,
                                        returns: pd.Series) -> float:
        """Calculate combined effectiveness of Gary×Taleb approach"""

        # Weight antifragility and DPI components
        antifragility_score = np.mean([
            antifragility_metrics.convexity_score,
            antifragility_metrics.volatility_benefit,
            antifragility_metrics.asymmetry_ratio,
            antifragility_metrics.black_swan_protection
        ])

        dpi_score = dpi_metrics.composite_dpi_score

        # Calculate performance metrics
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        performance_score = np.tanh(sharpe_ratio) / 2 + 0.5

        # Combined effectiveness
        combined = (antifragility_score * 0.4 + dpi_score * 0.4 + performance_score * 0.2)
        return combined

    def _analyze_risk_return_profile(self, returns: pd.Series,
                                   antifragility_metrics: AntifragilityMetrics,
                                   dpi_metrics: DPIMetrics) -> Dict[str, float]:
        """Analyze risk-return profile"""

        profile = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': (returns.mean() * 252) / abs(self._calculate_max_drawdown(returns)) if self._calculate_max_drawdown(returns) != 0 else 0,
            'antifragility_contribution': antifragility_metrics.convexity_score * 0.1,
            'dpi_contribution': dpi_metrics.composite_dpi_score * 0.1,
            'tail_risk_protection': antifragility_metrics.black_swan_protection,
            'consistency_factor': dpi_metrics.consistency_score
        }

        return profile

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def _identify_optimization_opportunities(self, antifragility_metrics: AntifragilityMetrics,
                                           dpi_metrics: DPIMetrics,
                                           returns: pd.Series) -> List[str]:
        """Identify optimization opportunities"""

        opportunities = []

        # Antifragility opportunities
        if antifragility_metrics.convexity_score < 0.6:
            opportunities.append("Enhance convexity through options-like position sizing")

        if antifragility_metrics.volatility_benefit < 0.5:
            opportunities.append("Improve volatility scaling mechanisms")

        if antifragility_metrics.barbell_efficiency < 0.7:
            opportunities.append("Optimize barbell strategy allocation")

        if antifragility_metrics.black_swan_protection < 0.6:
            opportunities.append("Strengthen tail risk protection")

        # DPI opportunities
        if dpi_metrics.momentum_score < 0.6:
            opportunities.append("Refine momentum detection algorithms")

        if dpi_metrics.stability_score < 0.7:
            opportunities.append("Improve stability through better risk management")

        if dpi_metrics.signal_quality < 0.6:
            opportunities.append("Enhance signal filtering and quality")

        if dpi_metrics.execution_efficiency < 0.7:
            opportunities.append("Optimize execution and reduce slippage")

        # Combined opportunities
        synergy_score = self._calculate_antifragility_dpi_synergy(antifragility_metrics, dpi_metrics)
        if synergy_score < 0.6:
            opportunities.append("Improve integration between antifragility and DPI components")

        return opportunities

    def _generate_strategic_recommendations(self, antifragility_metrics: AntifragilityMetrics,
                                          dpi_metrics: DPIMetrics,
                                          market_context: Optional[Dict] = None) -> List[str]:
        """Generate strategic recommendations"""

        recommendations = []

        # High-level strategic recommendations
        overall_antifragility = np.mean([
            antifragility_metrics.convexity_score,
            antifragility_metrics.volatility_benefit,
            antifragility_metrics.asymmetry_ratio
        ])

        overall_dpi = dpi_metrics.composite_dpi_score

        if overall_antifragility > 0.7 and overall_dpi > 0.7:
            recommendations.append("Maintain current Gary×Taleb integration - both components performing well")
        elif overall_antifragility > 0.7:
            recommendations.append("Focus on improving DPI components while maintaining antifragile characteristics")
        elif overall_dpi > 0.7:
            recommendations.append("Enhance antifragility features while preserving DPI performance")
        else:
            recommendations.append("Comprehensive review and optimization of both antifragility and DPI systems needed")

        # Specific tactical recommendations
        if antifragility_metrics.volatility_benefit > 0.8:
            recommendations.append("Leverage strong volatility benefits by increasing exposure during high-vol periods")

        if dpi_metrics.momentum_score > 0.8:
            recommendations.append("Capitalize on strong momentum detection with dynamic position sizing")

        if antifragility_metrics.fragility_index > 0.8:
            recommendations.append("System shows low fragility - consider gradually increasing position sizes")

        if dpi_metrics.consistency_score > 0.8:
            recommendations.append("High consistency detected - suitable for steady capital allocation")

        return recommendations

    def generate_comprehensive_report(self, integrated_analysis: IntegratedAnalysis,
                                    returns: pd.Series) -> str:
        """Generate comprehensive Gary×Taleb analysis report"""

        report = []
        report.append("=" * 80)
        report.append("GARY×TALEB COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {integrated_analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
        report.append(f"Sample Size: {len(returns)} observations")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Combined Effectiveness Score: {integrated_analysis.combined_effectiveness:.3f}")
        report.append(f"Antifragility-DPI Synergy: {integrated_analysis.synergy_score:.3f}")
        report.append(f"Overall Performance Rating: {'EXCELLENT' if integrated_analysis.combined_effectiveness > 0.8 else 'GOOD' if integrated_analysis.combined_effectiveness > 0.6 else 'MODERATE' if integrated_analysis.combined_effectiveness > 0.4 else 'NEEDS IMPROVEMENT'}")
        report.append("")

        # Antifragility Analysis
        af_metrics = integrated_analysis.antifragility_metrics
        report.append("ANTIFRAGILITY ANALYSIS")
        report.append("-" * 25)
        report.append("Core Measures:")
        report.append(f"  Convexity Score: {af_metrics.convexity_score:.3f}")
        report.append(f"  Volatility Benefit: {af_metrics.volatility_benefit:.3f}")
        report.append(f"  Asymmetry Ratio: {af_metrics.asymmetry_ratio:.3f}")
        report.append(f"  Black Swan Protection: {af_metrics.black_swan_protection:.3f}")
        report.append("")
        report.append("Taleb-Inspired Measures:")
        report.append(f"  Barbell Efficiency: {af_metrics.barbell_efficiency:.3f}")
        report.append(f"  Optionality Ratio: {af_metrics.optionality_ratio:.3f}")
        report.append(f"  Fragility Index: {af_metrics.fragility_index:.3f}")
        report.append(f"  Via Negativa Score: {af_metrics.via_negativa_score:.3f}")
        report.append("")

        # DPI Analysis
        dpi_metrics = integrated_analysis.dpi_metrics
        report.append("DYNAMIC PERFORMANCE INDICATOR (DPI) ANALYSIS")
        report.append("-" * 45)
        report.append("Core Components:")
        report.append(f"  Momentum Score: {dpi_metrics.momentum_score:.3f}")
        report.append(f"  Stability Score: {dpi_metrics.stability_score:.3f}")
        report.append(f"  Consistency Score: {dpi_metrics.consistency_score:.3f}")
        report.append(f"  Adaptability Score: {dpi_metrics.adaptability_score:.3f}")
        report.append(f"  Signal Quality: {dpi_metrics.signal_quality:.3f}")
        report.append("")
        report.append("Advanced Measures:")
        report.append(f"  Execution Efficiency: {dpi_metrics.execution_efficiency:.3f}")
        report.append(f"  Market Alignment: {dpi_metrics.market_alignment:.3f}")
        report.append(f"  Composite DPI Score: {dpi_metrics.composite_dpi_score:.3f}")
        report.append("")

        # Risk-Return Profile
        profile = integrated_analysis.risk_return_profile
        report.append("RISK-RETURN PROFILE")
        report.append("-" * 20)
        report.append(f"Total Return: {profile['total_return']:.2%}")
        report.append(f"Annualized Return: {profile['annualized_return']:.2%}")
        report.append(f"Volatility: {profile['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {profile['sharpe_ratio']:.3f}")
        report.append(f"Maximum Drawdown: {profile['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {profile['calmar_ratio']:.3f}")
        report.append("")

        # Optimization Opportunities
        if integrated_analysis.optimization_opportunities:
            report.append("OPTIMIZATION OPPORTUNITIES")
            report.append("-" * 30)
            for i, opportunity in enumerate(integrated_analysis.optimization_opportunities, 1):
                report.append(f"{i}. {opportunity}")
            report.append("")

        # Strategic Recommendations
        if integrated_analysis.strategic_recommendations:
            report.append("STRATEGIC RECOMMENDATIONS")
            report.append("-" * 30)
            for i, recommendation in enumerate(integrated_analysis.strategic_recommendations, 1):
                report.append(f"{i}. {recommendation}")
            report.append("")

        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 15)

        # Antifragility insights
        if af_metrics.volatility_benefit > 0.7:
            report.append("✓ Strong antifragile characteristics - strategy benefits from volatility")
        if af_metrics.convexity_score > 0.7:
            report.append("✓ Excellent convexity - strategy shows option-like payoffs")
        if af_metrics.black_swan_protection > 0.7:
            report.append("✓ Good tail risk protection against extreme events")

        # DPI insights
        if dpi_metrics.composite_dpi_score > 0.7:
            report.append("✓ Strong DPI performance - consistent dynamic adaptation")
        if dpi_metrics.consistency_score > 0.8:
            report.append("✓ Excellent consistency - reliable performance across periods")
        if dpi_metrics.adaptability_score > 0.7:
            report.append("✓ Good market adaptability - performs well across regimes")

        # Synergy insights
        if integrated_analysis.synergy_score > 0.7:
            report.append("✓ Excellent synergy between antifragility and DPI components")
        elif integrated_analysis.synergy_score < 0.5:
            report.append("⚠ Limited synergy - consider better integration of components")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def create_comprehensive_visualization(self, integrated_analysis: IntegratedAnalysis,
                                         returns: pd.Series) -> go.Figure:
        """Create comprehensive visualization dashboard"""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Antifragility Radar', 'DPI Components', 'Performance Timeline',
                          'Risk-Return Profile', 'Synergy Analysis', 'Regime Performance',
                          'Optimization Heatmap', 'Component Correlation', 'Strategic Dashboard'),
            specs=[[{"type": "scatterpolar"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}, {"type": "indicator"}]]
        )

        af_metrics = integrated_analysis.antifragility_metrics
        dpi_metrics = integrated_analysis.dpi_metrics

        # 1. Antifragility Radar Chart
        af_categories = ['Convexity', 'Volatility Benefit', 'Asymmetry', 'Tail Protection',
                        'Barbell Efficiency', 'Optionality', 'Resilience']
        af_values = [af_metrics.convexity_score, af_metrics.volatility_benefit,
                    af_metrics.asymmetry_ratio, af_metrics.black_swan_protection,
                    af_metrics.barbell_efficiency, af_metrics.optionality_ratio,
                    af_metrics.resilience_factor]

        fig.add_trace(
            go.Scatterpolar(r=af_values, theta=af_categories, fill='toself',
                           name='Antifragility Profile'),
            row=1, col=1
        )

        # 2. DPI Components Bar Chart
        dpi_components = ['Momentum', 'Stability', 'Consistency', 'Adaptability', 'Signal Quality']
        dpi_values = [dpi_metrics.momentum_score, dpi_metrics.stability_score,
                     dpi_metrics.consistency_score, dpi_metrics.adaptability_score,
                     dpi_metrics.signal_quality]

        fig.add_trace(
            go.Bar(x=dpi_components, y=dpi_values, name='DPI Components'),
            row=1, col=2
        )

        # 3. Performance Timeline
        equity_curve = (1 + returns).cumprod() * 200
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values,
                      mode='lines', name='Equity Curve'),
            row=1, col=3
        )

        # 4. Risk-Return Profile
        profile = integrated_analysis.risk_return_profile
        fig.add_trace(
            go.Scatter(x=[profile['volatility']], y=[profile['annualized_return']],
                      mode='markers', marker=dict(size=15), name='Strategy'),
            row=2, col=1
        )

        # 5. Synergy Analysis
        synergy_components = ['Volatility Synergy', 'Stability Synergy', 'Momentum Synergy']
        synergy_values = [0.8, 0.7, 0.9]  # Placeholder values

        fig.add_trace(
            go.Scatter(x=synergy_components, y=synergy_values,
                      mode='lines+markers', name='Synergy Scores'),
            row=2, col=2
        )

        # 6. Regime Performance (simplified)
        regimes = ['Low Vol', 'Medium Vol', 'High Vol']
        regime_performance = [0.7, 0.8, 0.9]  # Placeholder

        fig.add_trace(
            go.Bar(x=regimes, y=regime_performance, name='Regime Performance'),
            row=2, col=3
        )

        # 7. Optimization Heatmap
        optimization_matrix = np.random.rand(5, 5)  # Placeholder
        fig.add_trace(
            go.Heatmap(z=optimization_matrix, colorscale='RdYlBu_r'),
            row=3, col=1
        )

        # 8. Component Correlation
        fig.add_trace(
            go.Scatter(x=af_values, y=dpi_values,
                      mode='markers', name='AF vs DPI Correlation'),
            row=3, col=2
        )

        # 9. Strategic Dashboard (Indicator)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=integrated_analysis.combined_effectiveness * 100,
                title={'text': "Combined Effectiveness"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=3, col=3
        )

        # Update layout
        fig.update_layout(
            title_text="Gary×Taleb Comprehensive Analysis Dashboard",
            height=1000,
            showlegend=True
        )

        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Simulate Gary×Taleb strategy returns with antifragile characteristics
    base_returns = np.random.normal(0.0008, 0.015, len(dates))

    # Add volatility benefits (positive correlation with volatility)
    volatilities = np.random.lognormal(-3, 0.5, len(dates))
    vol_benefits = volatilities * 0.1
    antifragile_returns = base_returns + vol_benefits

    # Add some positive skew
    skew_component = np.random.gamma(2, 0.002, len(dates))
    final_returns = antifragile_returns + skew_component

    returns_series = pd.Series(final_returns, index=dates)

    # Initialize analyzer
    analyzer = AntifragilityAnalyzer()

    # Calculate antifragility metrics
    antifragility_metrics = analyzer.calculate_antifragility_metrics(returns_series)

    # Calculate DPI metrics
    dpi_metrics = analyzer.calculate_dpi_metrics(returns_series)

    # Perform integrated analysis
    integrated_analysis = analyzer.perform_integrated_analysis(
        returns_series, antifragility_metrics, dpi_metrics
    )

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(integrated_analysis, returns_series)
    print(report)

    # Create visualization
    fig = analyzer.create_comprehensive_visualization(integrated_analysis, returns_series)
    fig.show()

    print(f"\nGary×Taleb Analysis Complete!")
    print(f"Combined Effectiveness: {integrated_analysis.combined_effectiveness:.3f}")
    print(f"Synergy Score: {integrated_analysis.synergy_score:.3f}")
    print(f"Optimization Opportunities: {len(integrated_analysis.optimization_opportunities)}")