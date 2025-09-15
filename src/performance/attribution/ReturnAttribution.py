"""
Return Attribution Analysis System
Comprehensive analysis of return sources and performance drivers for trading strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

@dataclass
class AttributionComponent:
    """Individual attribution component"""
    name: str
    contribution: float
    percentage: float
    significance: float
    description: str
    period_performance: List[float] = field(default_factory=list)

@dataclass
class AttributionResult:
    """Complete attribution analysis result"""
    total_return: float
    components: List[AttributionComponent]
    unexplained_return: float
    r_squared: float
    analysis_period: Tuple[pd.Timestamp, pd.Timestamp]
    methodology: str

class ReturnAttribution:
    """Comprehensive return attribution analysis system"""

    def __init__(self):
        self.attribution_cache = {}
        self.factor_models = {}

    def analyze_return_attribution(self, returns: pd.Series,
                                 strategy_data: Dict,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 factor_data: Optional[Dict[str, pd.Series]] = None) -> AttributionResult:
        """Comprehensive return attribution analysis"""

        print("Performing comprehensive return attribution analysis...")

        # Prepare data
        analysis_start = returns.index[0]
        analysis_end = returns.index[-1]
        total_return = (1 + returns).prod() - 1

        # Multiple attribution methodologies
        components = []

        # 1. Gary×Taleb Strategy Attribution
        gary_taleb_components = self._analyze_gary_taleb_attribution(returns, strategy_data)
        components.extend(gary_taleb_components)

        # 2. Risk Factor Attribution
        risk_factor_components = self._analyze_risk_factor_attribution(returns, factor_data)
        components.extend(risk_factor_components)

        # 3. Timing and Selection Attribution
        timing_selection_components = self._analyze_timing_selection_attribution(returns, strategy_data)
        components.extend(timing_selection_components)

        # 4. Market Regime Attribution
        regime_components = self._analyze_regime_attribution(returns, factor_data)
        components.extend(regime_components)

        # 5. Behavioral Attribution
        behavioral_components = self._analyze_behavioral_attribution(returns, strategy_data)
        components.extend(behavioral_components)

        # Calculate unexplained return
        explained_return = sum(comp.contribution for comp in components)
        unexplained_return = total_return - explained_return

        # Calculate R-squared
        predicted_returns = self._reconstruct_returns(components, len(returns))
        r_squared = np.corrcoef(returns, predicted_returns)[0, 1] ** 2 if not np.isnan(np.corrcoef(returns, predicted_returns)[0, 1]) else 0

        return AttributionResult(
            total_return=total_return,
            components=components,
            unexplained_return=unexplained_return,
            r_squared=r_squared,
            analysis_period=(analysis_start, analysis_end),
            methodology="Multi-Factor Gary×Taleb Attribution"
        )

    def _analyze_gary_taleb_attribution(self, returns: pd.Series, strategy_data: Dict) -> List[AttributionComponent]:
        """Analyze attribution specific to Gary×Taleb strategy components"""

        components = []

        # DPI Component Attribution
        if 'dpi_scores' in strategy_data:
            dpi_scores = strategy_data['dpi_scores']
            dpi_aligned_returns = returns[dpi_scores > 0.6]  # High DPI periods

            if len(dpi_aligned_returns) > 0:
                dpi_contribution = dpi_aligned_returns.sum()
                dpi_percentage = (len(dpi_aligned_returns) / len(returns)) * 100

                components.append(AttributionComponent(
                    name="DPI High-Quality Signals",
                    contribution=dpi_contribution,
                    percentage=dpi_percentage,
                    significance=abs(dpi_contribution) / returns.std() if returns.std() > 0 else 0,
                    description="Returns from periods with high DPI scores (>0.6)",
                    period_performance=dpi_aligned_returns.tolist()
                ))

        # Antifragility Component
        if 'antifragility_scores' in strategy_data:
            antifragility_scores = strategy_data['antifragility_scores']

            # High volatility periods where antifragility should benefit
            vol_threshold = returns.rolling(20).std().quantile(0.75)
            high_vol_periods = returns.rolling(20).std() > vol_threshold
            antifragile_periods = (antifragility_scores > 0.5) & high_vol_periods

            if antifragile_periods.sum() > 0:
                antifragile_returns = returns[antifragile_periods]
                antifragile_contribution = antifragile_returns.sum()

                components.append(AttributionComponent(
                    name="Antifragility Benefits",
                    contribution=antifragile_contribution,
                    percentage=(antifragile_periods.sum() / len(returns)) * 100,
                    significance=abs(antifragile_contribution) / returns.std() if returns.std() > 0 else 0,
                    description="Returns from high-volatility periods with strong antifragility",
                    period_performance=antifragile_returns.tolist()
                ))

        # Momentum Component
        momentum_returns = returns[returns.rolling(5).mean() > 0]
        if len(momentum_returns) > 0:
            momentum_contribution = momentum_returns.sum()

            components.append(AttributionComponent(
                name="Momentum Capture",
                contribution=momentum_contribution,
                percentage=(len(momentum_returns) / len(returns)) * 100,
                significance=abs(momentum_contribution) / returns.std() if returns.std() > 0 else 0,
                description="Returns from positive momentum periods",
                period_performance=momentum_returns.tolist()
            ))

        # Mean Reversion Component
        mean_reversion_signal = returns.rolling(20).mean()
        oversold_periods = returns < mean_reversion_signal - returns.rolling(20).std()
        mr_returns = returns[oversold_periods.shift(1).fillna(False)]

        if len(mr_returns) > 0:
            mr_contribution = mr_returns.sum()

            components.append(AttributionComponent(
                name="Mean Reversion Alpha",
                contribution=mr_contribution,
                percentage=(len(mr_returns) / len(returns)) * 100,
                significance=abs(mr_contribution) / returns.std() if returns.std() > 0 else 0,
                description="Returns from mean reversion opportunities",
                period_performance=mr_returns.tolist()
            ))

        return components

    def _analyze_risk_factor_attribution(self, returns: pd.Series, factor_data: Optional[Dict[str, pd.Series]]) -> List[AttributionComponent]:
        """Analyze attribution based on risk factors"""

        components = []

        if factor_data is None:
            # Create synthetic factor data for demonstration
            factor_data = self._create_synthetic_factors(returns)

        # Market Beta Attribution
        if 'market' in factor_data:
            market_factor = factor_data['market']
            common_index = returns.index.intersection(market_factor.index)

            if len(common_index) > 20:
                aligned_returns = returns.loc[common_index]
                aligned_market = market_factor.loc[common_index]

                # Calculate beta
                covariance = np.cov(aligned_returns, aligned_market)[0, 1]
                market_variance = np.var(aligned_market)
                beta = covariance / market_variance if market_variance > 0 else 0

                # Beta contribution
                beta_contribution = beta * aligned_market.sum()

                components.append(AttributionComponent(
                    name="Market Beta",
                    contribution=beta_contribution,
                    percentage=abs(beta_contribution) / abs(returns.sum()) * 100 if returns.sum() != 0 else 0,
                    significance=abs(beta),
                    description=f"Market exposure (Beta: {beta:.2f})",
                    period_performance=(beta * aligned_market).tolist()
                ))

        # Volatility Factor Attribution
        volatility_factor = returns.rolling(20).std()
        high_vol_periods = volatility_factor > volatility_factor.quantile(0.75)
        high_vol_returns = returns[high_vol_periods]

        if len(high_vol_returns) > 0:
            vol_contribution = high_vol_returns.sum()

            components.append(AttributionComponent(
                name="Volatility Factor",
                contribution=vol_contribution,
                percentage=(len(high_vol_returns) / len(returns)) * 100,
                significance=abs(vol_contribution) / returns.std() if returns.std() > 0 else 0,
                description="Performance during high volatility periods",
                period_performance=high_vol_returns.tolist()
            ))

        # Size Factor (approximated by position sizing if available)
        if 'position_sizes' in returns.index.names or hasattr(returns, 'position_sizes'):
            # Placeholder for size factor analysis
            pass

        return components

    def _analyze_timing_selection_attribution(self, returns: pd.Series, strategy_data: Dict) -> List[AttributionComponent]:
        """Analyze timing and selection attribution"""

        components = []

        # Market Timing Attribution
        if 'trades' in strategy_data:
            trades = strategy_data['trades']

            # Analyze entry timing
            entry_returns = []
            for trade in trades:
                if trade['type'] == 'buy':
                    trade_date = pd.to_datetime(trade['date'])
                    if trade_date in returns.index:
                        # Look at next day return (timing quality)
                        next_day_idx = returns.index.get_loc(trade_date) + 1
                        if next_day_idx < len(returns):
                            entry_returns.append(returns.iloc[next_day_idx])

            if entry_returns:
                timing_contribution = sum(entry_returns)

                components.append(AttributionComponent(
                    name="Entry Timing",
                    contribution=timing_contribution,
                    percentage=(len(entry_returns) / len(returns)) * 100,
                    significance=abs(timing_contribution) / returns.std() if returns.std() > 0 else 0,
                    description="Quality of trade entry timing",
                    period_performance=entry_returns
                ))

        # Signal Quality Attribution
        positive_signal_returns = returns[returns > 0]
        negative_signal_returns = returns[returns < 0]

        signal_quality = len(positive_signal_returns) / len(returns) if len(returns) > 0 else 0
        signal_contribution = positive_signal_returns.sum() - abs(negative_signal_returns.sum())

        components.append(AttributionComponent(
            name="Signal Selection Quality",
            contribution=signal_contribution,
            percentage=signal_quality * 100,
            significance=signal_quality,
            description=f"Win rate: {signal_quality:.1%}",
            period_performance=[signal_quality]
        ))

        return components

    def _analyze_regime_attribution(self, returns: pd.Series, factor_data: Optional[Dict[str, pd.Series]]) -> List[AttributionComponent]:
        """Analyze performance across different market regimes"""

        components = []

        # Volatility Regime Analysis
        vol_window = 20
        rolling_vol = returns.rolling(vol_window).std()

        # Define regimes
        low_vol_threshold = rolling_vol.quantile(0.33)
        high_vol_threshold = rolling_vol.quantile(0.67)

        low_vol_periods = rolling_vol <= low_vol_threshold
        medium_vol_periods = (rolling_vol > low_vol_threshold) & (rolling_vol <= high_vol_threshold)
        high_vol_periods = rolling_vol > high_vol_threshold

        # Attribution by regime
        regimes = [
            ("Low Volatility Regime", low_vol_periods),
            ("Medium Volatility Regime", medium_vol_periods),
            ("High Volatility Regime", high_vol_periods)
        ]

        for regime_name, regime_mask in regimes:
            regime_returns = returns[regime_mask]
            if len(regime_returns) > 0:
                regime_contribution = regime_returns.sum()

                components.append(AttributionComponent(
                    name=regime_name,
                    contribution=regime_contribution,
                    percentage=(len(regime_returns) / len(returns)) * 100,
                    significance=abs(regime_contribution) / returns.std() if returns.std() > 0 else 0,
                    description=f"Performance in {regime_name.lower()}",
                    period_performance=regime_returns.tolist()
                ))

        # Trend Regime Analysis
        trend_window = 50
        price_proxy = (1 + returns).cumprod()
        moving_average = price_proxy.rolling(trend_window).mean()

        uptrend_periods = price_proxy > moving_average
        downtrend_periods = price_proxy <= moving_average

        trend_regimes = [
            ("Uptrend Regime", uptrend_periods),
            ("Downtrend Regime", downtrend_periods)
        ]

        for regime_name, regime_mask in trend_regimes:
            regime_returns = returns[regime_mask]
            if len(regime_returns) > 0:
                regime_contribution = regime_returns.sum()

                components.append(AttributionComponent(
                    name=regime_name,
                    contribution=regime_contribution,
                    percentage=(len(regime_returns) / len(returns)) * 100,
                    significance=abs(regime_contribution) / returns.std() if returns.std() > 0 else 0,
                    description=f"Performance during {regime_name.lower()}",
                    period_performance=regime_returns.tolist()
                ))

        return components

    def _analyze_behavioral_attribution(self, returns: pd.Series, strategy_data: Dict) -> List[AttributionComponent]:
        """Analyze behavioral and psychological factors in performance"""

        components = []

        # Streak Analysis
        streak_returns = []
        current_streak = 0
        streak_type = None

        for ret in returns:
            if ret > 0:
                if streak_type != 'positive':
                    current_streak = 1
                    streak_type = 'positive'
                else:
                    current_streak += 1
            else:
                if streak_type != 'negative':
                    current_streak = 1
                    streak_type = 'negative'
                else:
                    current_streak += 1

            # Analyze performance during streaks
            if current_streak >= 3:  # Extended streaks
                streak_returns.append(ret)

        if streak_returns:
            streak_contribution = sum(streak_returns)

            components.append(AttributionComponent(
                name="Streak Performance",
                contribution=streak_contribution,
                percentage=(len(streak_returns) / len(returns)) * 100,
                significance=abs(streak_contribution) / returns.std() if returns.std() > 0 else 0,
                description="Performance during extended winning/losing streaks",
                period_performance=streak_returns
            ))

        # Overconfidence/Underconfidence Analysis
        # Based on position sizing relative to signal strength
        if 'trades' in strategy_data:
            trades = strategy_data['trades']

            # Analyze position sizing decisions
            sizing_returns = []
            for trade in trades:
                if 'position_size' in trade and 'signal_strength' in trade:
                    trade_date = pd.to_datetime(trade['date'])
                    if trade_date in returns.index:
                        # Analyze if position size was appropriate for signal strength
                        expected_size = trade['signal_strength']
                        actual_size = trade['position_size']

                        # Performance when sizing was well-calibrated
                        if abs(actual_size - expected_size) < 0.2:
                            sizing_returns.append(returns.loc[trade_date])

            if sizing_returns:
                sizing_contribution = sum(sizing_returns)

                components.append(AttributionComponent(
                    name="Position Sizing Discipline",
                    contribution=sizing_contribution,
                    percentage=(len(sizing_returns) / len(returns)) * 100,
                    significance=abs(sizing_contribution) / returns.std() if returns.std() > 0 else 0,
                    description="Returns from well-calibrated position sizing",
                    period_performance=sizing_returns
                ))

        return components

    def _create_synthetic_factors(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Create synthetic factor data for demonstration"""

        # Market factor (correlated with returns but with noise)
        market_returns = returns * 0.7 + np.random.normal(0, 0.01, len(returns))

        # Volatility factor
        vol_factor = returns.rolling(10).std()

        # Momentum factor
        momentum_factor = returns.rolling(20).mean()

        return {
            'market': pd.Series(market_returns, index=returns.index),
            'volatility': vol_factor,
            'momentum': momentum_factor
        }

    def _reconstruct_returns(self, components: List[AttributionComponent], return_length: int) -> np.ndarray:
        """Reconstruct returns from attribution components"""

        reconstructed = np.zeros(return_length)

        for component in components:
            if component.period_performance:
                # Distribute component contribution across periods
                contribution_per_period = component.contribution / len(component.period_performance)
                # Simplified reconstruction - in practice this would be more sophisticated
                reconstructed += contribution_per_period / len(components)

        return reconstructed

    def perform_factor_analysis(self, returns: pd.Series,
                              factor_data: Dict[str, pd.Series]) -> Dict:
        """Perform factor analysis using multiple regression"""

        print("Performing factor regression analysis...")

        # Prepare data
        common_index = returns.index
        for factor_name, factor_series in factor_data.items():
            common_index = common_index.intersection(factor_series.index)

        if len(common_index) < 30:
            return {"error": "Insufficient overlapping data for factor analysis"}

        # Align data
        y = returns.loc[common_index].values
        X = np.column_stack([factor_data[name].loc[common_index].values
                           for name in factor_data.keys()])
        factor_names = list(factor_data.keys())

        # Multiple regression
        model = LinearRegression()
        model.fit(X, y)

        # Calculate statistics
        r_squared = model.score(X, y)
        coefficients = model.coef_
        intercept = model.intercept_

        # Statistical significance (simplified)
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals ** 2)

        results = {
            'r_squared': r_squared,
            'alpha': intercept,
            'factors': {
                factor_names[i]: {
                    'beta': coefficients[i],
                    'contribution': coefficients[i] * np.mean(X[:, i]),
                    'significance': abs(coefficients[i]) / np.std(X[:, i]) if np.std(X[:, i]) > 0 else 0
                }
                for i in range(len(factor_names))
            },
            'model_fit': {
                'mse': mse,
                'residual_std': np.std(residuals)
            }
        }

        return results

    def generate_attribution_report(self, attribution_result: AttributionResult,
                                  factor_analysis: Optional[Dict] = None) -> str:
        """Generate comprehensive attribution report"""

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE RETURN ATTRIBUTION ANALYSIS")
        report.append("=" * 80)
        report.append(f"Analysis Period: {attribution_result.analysis_period[0].strftime('%Y-%m-%d')} to {attribution_result.analysis_period[1].strftime('%Y-%m-%d')}")
        report.append(f"Methodology: {attribution_result.methodology}")
        report.append(f"Total Return: {attribution_result.total_return:.2%}")
        report.append(f"Model R²: {attribution_result.r_squared:.3f}")
        report.append("")

        # Sort components by absolute contribution
        sorted_components = sorted(attribution_result.components,
                                 key=lambda x: abs(x.contribution), reverse=True)

        report.append("RETURN ATTRIBUTION BREAKDOWN")
        report.append("-" * 35)

        cumulative_explained = 0
        for i, component in enumerate(sorted_components, 1):
            contribution_pct = (component.contribution / attribution_result.total_return * 100) if attribution_result.total_return != 0 else 0
            cumulative_explained += abs(component.contribution)

            report.append(f"\n{i}. {component.name}")
            report.append(f"   Contribution: {component.contribution:.4f} ({contribution_pct:.1f}% of total)")
            report.append(f"   Time Exposure: {component.percentage:.1f}%")
            report.append(f"   Significance: {component.significance:.3f}")
            report.append(f"   Description: {component.description}")

        # Unexplained return
        unexplained_pct = (attribution_result.unexplained_return / attribution_result.total_return * 100) if attribution_result.total_return != 0 else 0
        report.append(f"\nUnexplained Return: {attribution_result.unexplained_return:.4f} ({unexplained_pct:.1f}%)")

        # Factor analysis results
        if factor_analysis:
            report.append("\n\nFACTOR ANALYSIS RESULTS")
            report.append("-" * 25)
            report.append(f"Alpha (Excess Return): {factor_analysis['alpha']:.4f}")
            report.append(f"Model R²: {factor_analysis['r_squared']:.3f}")
            report.append("")

            report.append("Factor Exposures:")
            for factor_name, factor_data in factor_analysis['factors'].items():
                report.append(f"  {factor_name}:")
                report.append(f"    Beta: {factor_data['beta']:.3f}")
                report.append(f"    Contribution: {factor_data['contribution']:.4f}")
                report.append(f"    Significance: {factor_data['significance']:.3f}")

        # Key insights
        report.append("\n\nKEY INSIGHTS")
        report.append("-" * 15)

        # Find top contributors
        top_positive = [c for c in sorted_components if c.contribution > 0][:3]
        top_negative = [c for c in sorted_components if c.contribution < 0][:3]

        if top_positive:
            report.append("Top Positive Contributors:")
            for component in top_positive:
                report.append(f"  • {component.name}: {component.contribution:.4f}")

        if top_negative:
            report.append("Top Negative Contributors:")
            for component in top_negative:
                report.append(f"  • {component.name}: {component.contribution:.4f}")

        # Gary×Taleb specific insights
        gary_taleb_components = [c for c in attribution_result.components
                               if any(keyword in c.name.lower()
                                    for keyword in ['dpi', 'antifragility', 'gary', 'taleb'])]

        if gary_taleb_components:
            total_gary_taleb = sum(c.contribution for c in gary_taleb_components)
            gary_taleb_pct = (total_gary_taleb / attribution_result.total_return * 100) if attribution_result.total_return != 0 else 0

            report.append(f"\nGary×Taleb Strategy Components: {total_gary_taleb:.4f} ({gary_taleb_pct:.1f}% of total)")

            if gary_taleb_pct > 60:
                report.append("✓ Strong Gary×Taleb strategy effectiveness")
            elif gary_taleb_pct > 30:
                report.append("• Moderate Gary×Taleb strategy contribution")
            else:
                report.append("⚠ Limited Gary×Taleb strategy impact")

        # Recommendations
        report.append("\n\nRECOMMENDATIONS")
        report.append("-" * 20)

        if attribution_result.r_squared > 0.7:
            report.append("• Strong model explanatory power - strategy is well understood")
        elif attribution_result.r_squared > 0.4:
            report.append("• Moderate model fit - consider additional factors")
        else:
            report.append("• Low model fit - significant unexplained performance")

        # Component-specific recommendations
        dpi_components = [c for c in attribution_result.components if 'dpi' in c.name.lower()]
        if dpi_components and dpi_components[0].contribution > 0:
            report.append("• DPI signals showing positive contribution - maintain approach")
        elif dpi_components:
            report.append("• Review DPI calibration for improved performance")

        antifragile_components = [c for c in attribution_result.components if 'antifragility' in c.name.lower()]
        if antifragile_components and antifragile_components[0].contribution > 0:
            report.append("• Antifragility benefits detected - enhance volatility positioning")

        report.append("• Regular attribution analysis to track performance drivers")
        report.append("• Focus on enhancing top positive contributors")
        report.append("• Address or hedge top negative contributors")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def create_attribution_visualization(self, attribution_result: AttributionResult) -> go.Figure:
        """Create comprehensive attribution visualization"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attribution Waterfall', 'Component Contributions',
                          'Time-based Attribution', 'Significance vs Contribution'),
            specs=[[{"type": "waterfall"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        # 1. Waterfall chart
        components = attribution_result.components
        sorted_components = sorted(components, key=lambda x: x.contribution, reverse=True)

        waterfall_x = ['Starting'] + [c.name for c in sorted_components] + ['Unexplained', 'Total']
        waterfall_y = [0] + [c.contribution for c in sorted_components] + [attribution_result.unexplained_return, attribution_result.total_return]

        fig.add_trace(
            go.Waterfall(
                name="Attribution Waterfall",
                orientation="v",
                measure=['absolute'] + ['relative'] * len(sorted_components) + ['relative', 'total'],
                x=waterfall_x,
                y=waterfall_y,
                textposition="outside"
            ),
            row=1, col=1
        )

        # 2. Component contributions bar chart
        component_names = [c.name for c in sorted_components]
        component_values = [c.contribution for c in sorted_components]
        colors = ['green' if v > 0 else 'red' for v in component_values]

        fig.add_trace(
            go.Bar(
                x=component_names,
                y=component_values,
                marker_color=colors,
                name="Contributions"
            ),
            row=1, col=2
        )

        # 3. Time-based attribution (simplified)
        # Show cumulative contribution over time
        if len(components) > 0:
            time_periods = list(range(len(components)))
            cumulative_contrib = np.cumsum([c.contribution for c in components])

            fig.add_trace(
                go.Scatter(
                    x=time_periods,
                    y=cumulative_contrib,
                    mode='lines+markers',
                    name='Cumulative Attribution'
                ),
                row=2, col=1
            )

        # 4. Significance vs Contribution scatter
        significance_values = [c.significance for c in components]
        contribution_values = [c.contribution for c in components]
        component_labels = [c.name for c in components]

        fig.add_trace(
            go.Scatter(
                x=significance_values,
                y=contribution_values,
                mode='markers+text',
                text=component_labels,
                textposition="top center",
                marker=dict(size=10, opacity=0.7),
                name='Significance vs Contribution'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Return Attribution Analysis Dashboard",
            height=800,
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Components", row=1, col=2)
        fig.update_yaxes(title_text="Contribution", row=1, col=2)
        fig.update_xaxes(title_text="Time Period", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Contribution", row=2, col=1)
        fig.update_xaxes(title_text="Significance", row=2, col=2)
        fig.update_yaxes(title_text="Contribution", row=2, col=2)

        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Simulate strategy returns
    returns = pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates)

    # Create sample strategy data
    strategy_data = {
        'dpi_scores': pd.Series(np.random.uniform(0.3, 0.9, len(dates)), index=dates),
        'antifragility_scores': pd.Series(np.random.uniform(0.2, 0.8, len(dates)), index=dates),
        'trades': [
            {'type': 'buy', 'date': dates[i], 'signal_strength': np.random.uniform(0.4, 0.9), 'position_size': np.random.uniform(0.3, 1.0)}
            for i in range(0, len(dates), 10)
        ]
    }

    # Initialize analyzer
    analyzer = ReturnAttribution()

    # Perform attribution analysis
    attribution_result = analyzer.analyze_return_attribution(returns, strategy_data)

    # Perform factor analysis
    factor_data = analyzer._create_synthetic_factors(returns)
    factor_analysis = analyzer.perform_factor_analysis(returns, factor_data)

    # Generate report
    report = analyzer.generate_attribution_report(attribution_result, factor_analysis)
    print(report)

    # Create visualization
    fig = analyzer.create_attribution_visualization(attribution_result)
    fig.show()

    print(f"\nAttribution Analysis Complete!")
    print(f"Total Return: {attribution_result.total_return:.2%}")
    print(f"Explained Variance (R²): {attribution_result.r_squared:.1%}")
    print(f"Number of Components: {len(attribution_result.components)}")