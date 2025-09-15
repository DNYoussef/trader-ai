"""
Advanced Drawdown Analysis and Prevention System
Comprehensive risk management system for minimizing and preventing portfolio drawdowns
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')

@dataclass
class DrawdownEvent:
    """Individual drawdown event analysis"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trough_date: pd.Timestamp
    start_value: float
    trough_value: float
    end_value: float
    max_drawdown: float
    duration_days: int
    recovery_days: int
    total_days: int
    drawdown_rate: float  # Rate of decline per day
    recovery_rate: float  # Rate of recovery per day
    cause_analysis: str
    severity_level: str

@dataclass
class DrawdownStats:
    """Comprehensive drawdown statistics"""
    max_drawdown: float
    avg_drawdown: float
    drawdown_frequency: float
    avg_duration: float
    max_duration: int
    avg_recovery_time: float
    max_recovery_time: int
    pain_index: float
    ulcer_index: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    lake_ratio: float
    drawdown_at_risk: float  # 95th percentile
    worst_month: float
    worst_quarter: float

@dataclass
class RiskAlert:
    """Risk alert for potential drawdown"""
    timestamp: pd.Timestamp
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    current_drawdown: float
    predicted_max_drawdown: float
    confidence: float
    recommended_actions: List[str]
    risk_factors: Dict[str, float]

class DrawdownAnalysis:
    """Advanced drawdown analysis and prevention system"""

    def __init__(self, max_acceptable_drawdown: float = 0.15,
                 risk_alert_threshold: float = 0.05):
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.risk_alert_threshold = risk_alert_threshold
        self.analysis_cache = {}
        self.risk_model = None
        self.scaler = StandardScaler()

    def analyze_drawdowns(self, equity_curve: pd.Series,
                         returns: pd.Series) -> Tuple[List[DrawdownEvent], DrawdownStats]:
        """Comprehensive drawdown analysis"""

        print("Analyzing drawdown patterns and characteristics...")

        # Calculate drawdown series
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        # Identify drawdown events
        drawdown_events = self._identify_drawdown_events(equity_curve, drawdown, returns)

        # Calculate comprehensive statistics
        drawdown_stats = self._calculate_drawdown_stats(drawdown, equity_curve, returns)

        return drawdown_events, drawdown_stats

    def _identify_drawdown_events(self, equity_curve: pd.Series,
                                drawdown: pd.Series,
                                returns: pd.Series) -> List[DrawdownEvent]:
        """Identify and analyze individual drawdown events"""

        events = []
        in_drawdown = False
        start_idx = None
        min_idx = None
        min_value = None

        for i, (date, dd_value) in enumerate(drawdown.items()):
            if dd_value < 0 and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_idx = i
                min_idx = i
                min_value = dd_value

            elif dd_value < 0 and in_drawdown:
                # Continue drawdown, check if new minimum
                if dd_value < min_value:
                    min_idx = i
                    min_value = dd_value

            elif dd_value >= 0 and in_drawdown:
                # End of drawdown (recovery to peak)
                in_drawdown = False
                end_idx = i

                # Create drawdown event
                event = self._create_drawdown_event(
                    equity_curve, drawdown, returns,
                    start_idx, min_idx, end_idx
                )
                events.append(event)

        # Handle case where we end in a drawdown
        if in_drawdown:
            end_idx = len(equity_curve) - 1
            event = self._create_drawdown_event(
                equity_curve, drawdown, returns,
                start_idx, min_idx, end_idx
            )
            events.append(event)

        return events

    def _create_drawdown_event(self, equity_curve: pd.Series,
                             drawdown: pd.Series,
                             returns: pd.Series,
                             start_idx: int,
                             min_idx: int,
                             end_idx: int) -> DrawdownEvent:
        """Create detailed drawdown event analysis"""

        dates = equity_curve.index
        start_date = dates[start_idx]
        trough_date = dates[min_idx]
        end_date = dates[end_idx]

        start_value = equity_curve.iloc[start_idx]
        trough_value = equity_curve.iloc[min_idx]
        end_value = equity_curve.iloc[end_idx]

        max_drawdown = drawdown.iloc[min_idx]
        duration_days = (trough_date - start_date).days
        recovery_days = (end_date - trough_date).days
        total_days = (end_date - start_date).days

        # Rates of decline and recovery
        drawdown_rate = max_drawdown / max(duration_days, 1)
        recovery_rate = abs(max_drawdown) / max(recovery_days, 1) if recovery_days > 0 else 0

        # Cause analysis (simplified)
        period_returns = returns.loc[start_date:end_date]
        cause_analysis = self._analyze_drawdown_cause(period_returns)

        # Severity classification
        severity_level = self._classify_drawdown_severity(abs(max_drawdown))

        return DrawdownEvent(
            start_date=start_date,
            end_date=end_date,
            trough_date=trough_date,
            start_value=start_value,
            trough_value=trough_value,
            end_value=end_value,
            max_drawdown=max_drawdown,
            duration_days=duration_days,
            recovery_days=recovery_days,
            total_days=total_days,
            drawdown_rate=drawdown_rate,
            recovery_rate=recovery_rate,
            cause_analysis=cause_analysis,
            severity_level=severity_level
        )

    def _analyze_drawdown_cause(self, period_returns: pd.Series) -> str:
        """Analyze the cause of drawdown"""

        if len(period_returns) == 0:
            return "Insufficient data"

        # Analyze return characteristics
        volatility = period_returns.std()
        skewness = period_returns.skew()
        worst_day = period_returns.min()
        consecutive_losses = self._count_consecutive_losses(period_returns)

        if worst_day < -0.05:  # Single large loss
            return f"Single large loss event ({worst_day:.2%})"
        elif consecutive_losses > 5:
            return f"Extended losing streak ({consecutive_losses} consecutive losses)"
        elif volatility > 0.03:  # High volatility period
            return f"High volatility period (daily vol: {volatility:.2%})"
        elif skewness < -1:
            return "Negative tail event"
        else:
            return "Gradual decline"

    def _count_consecutive_losses(self, returns: pd.Series) -> int:
        """Count maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0

        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _classify_drawdown_severity(self, drawdown_magnitude: float) -> str:
        """Classify drawdown severity"""
        if drawdown_magnitude < 0.02:
            return "MINOR"
        elif drawdown_magnitude < 0.05:
            return "MODERATE"
        elif drawdown_magnitude < 0.10:
            return "SIGNIFICANT"
        elif drawdown_magnitude < 0.20:
            return "SEVERE"
        else:
            return "EXTREME"

    def _calculate_drawdown_stats(self, drawdown: pd.Series,
                                equity_curve: pd.Series,
                                returns: pd.Series) -> DrawdownStats:
        """Calculate comprehensive drawdown statistics"""

        # Basic drawdown metrics
        max_drawdown = drawdown.min()
        drawdown_series = drawdown[drawdown < 0]
        avg_drawdown = drawdown_series.mean() if len(drawdown_series) > 0 else 0

        # Duration analysis
        durations = []
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        avg_duration = np.mean(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # Recovery time analysis
        recovery_times = []
        in_drawdown = False
        drawdown_start = None

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)

        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        max_recovery_time = max(recovery_times) if recovery_times else 0

        # Frequency
        total_periods = len(drawdown)
        drawdown_periods = len(drawdown_series)
        drawdown_frequency = drawdown_periods / total_periods if total_periods > 0 else 0

        # Advanced metrics
        pain_index = np.sqrt(np.mean(drawdown_series ** 2)) if len(drawdown_series) > 0 else 0
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        # Risk-adjusted metrics
        annual_return = returns.mean() * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sterling_ratio = calmar_ratio  # Simplified
        burke_ratio = annual_return / np.sqrt(np.sum(drawdown_series ** 2)) if len(drawdown_series) > 0 else 0

        # Lake ratio (underwater equity)
        underwater = equity_curve.expanding().max() - equity_curve
        lake_ratio = underwater.sum() / (len(equity_curve) * equity_curve.iloc[-1])

        # Drawdown at Risk (95th percentile)
        drawdown_at_risk = np.percentile(drawdown_series, 5) if len(drawdown_series) > 0 else 0

        # Worst periods
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)

        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
        worst_quarter = quarterly_returns.min() if len(quarterly_returns) > 0 else 0

        return DrawdownStats(
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_frequency=drawdown_frequency,
            avg_duration=avg_duration,
            max_duration=max_duration,
            avg_recovery_time=avg_recovery_time,
            max_recovery_time=max_recovery_time,
            pain_index=pain_index,
            ulcer_index=ulcer_index,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            lake_ratio=lake_ratio,
            drawdown_at_risk=drawdown_at_risk,
            worst_month=worst_month,
            worst_quarter=worst_quarter
        )

    def build_risk_prediction_model(self, historical_data: Dict[str, pd.Series]):
        """Build machine learning model to predict drawdown risk"""

        print("Building drawdown risk prediction model...")

        # Prepare features
        features = []
        labels = []

        for strategy_name, returns in historical_data.items():
            equity_curve = (1 + returns).cumprod() * 200
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak

            # Create features for each time point
            for i in range(30, len(returns)):  # Need lookback period
                window_returns = returns.iloc[i-30:i]
                current_dd = drawdown.iloc[i]

                # Feature engineering
                feature_vector = [
                    window_returns.mean(),                    # Average return
                    window_returns.std(),                     # Volatility
                    window_returns.skew(),                    # Skewness
                    window_returns.kurtosis(),                # Kurtosis
                    (window_returns < 0).sum() / 30,         # Loss rate
                    window_returns.rolling(5).std().mean(),  # Short-term volatility
                    window_returns.rolling(20).std().mean(), # Long-term volatility
                    current_dd,                               # Current drawdown
                    abs(current_dd),                          # Drawdown magnitude
                    np.mean([abs(dd) for dd in drawdown.iloc[i-10:i] if dd < 0])  # Recent drawdown severity
                ]

                # Future maximum drawdown (prediction target)
                future_returns = returns.iloc[i:i+20] if i+20 < len(returns) else returns.iloc[i:]
                future_equity = (1 + future_returns).cumprod() * equity_curve.iloc[i]
                future_peak = future_equity.expanding().max()
                future_drawdown = (future_equity - future_peak) / future_peak
                max_future_dd = future_drawdown.min()

                features.append(feature_vector)
                labels.append(abs(max_future_dd))

        # Train model
        X = np.array(features)
        y = np.array(labels)

        # Handle missing values
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest model
        self.risk_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.risk_model.fit(X_scaled, y)

        print(f"Model trained on {len(X)} samples")
        print(f"Feature importance: {dict(zip(range(len(X[0])), self.risk_model.feature_importances_))}")

    def predict_drawdown_risk(self, current_returns: pd.Series,
                            current_equity: pd.Series) -> RiskAlert:
        """Predict future drawdown risk"""

        if self.risk_model is None:
            return RiskAlert(
                timestamp=current_returns.index[-1],
                alert_type="MODEL_ERROR",
                severity="LOW",
                current_drawdown=0,
                predicted_max_drawdown=0,
                confidence=0,
                recommended_actions=["Build risk model first"],
                risk_factors={}
            )

        # Calculate current state
        peak = current_equity.expanding().max()
        current_drawdown = (current_equity.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]

        # Prepare features (last 30 periods)
        window_returns = current_returns.iloc[-30:]

        features = [
            window_returns.mean(),
            window_returns.std(),
            window_returns.skew(),
            window_returns.kurtosis(),
            (window_returns < 0).sum() / 30,
            window_returns.rolling(5).std().mean(),
            window_returns.rolling(20).std().mean(),
            current_drawdown,
            abs(current_drawdown),
            np.mean([abs(dd) for dd in (current_equity - peak) / peak if dd < 0])
        ]

        # Handle missing values
        features = np.nan_to_num(features)

        # Scale and predict
        features_scaled = self.scaler.transform([features])
        predicted_max_dd = self.risk_model.predict(features_scaled)[0]

        # Calculate confidence (simplified)
        confidence = min(1.0, 1.0 - predicted_max_dd)

        # Determine severity
        if predicted_max_dd > self.max_acceptable_drawdown:
            severity = "CRITICAL"
        elif predicted_max_dd > self.risk_alert_threshold:
            severity = "HIGH"
        elif predicted_max_dd > self.risk_alert_threshold / 2:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            current_drawdown, predicted_max_dd, window_returns
        )

        # Risk factor analysis
        risk_factors = {
            'current_volatility': window_returns.std(),
            'recent_losses': (window_returns < 0).sum(),
            'current_drawdown': current_drawdown,
            'momentum': window_returns.mean(),
            'tail_risk': window_returns.quantile(0.05)
        }

        return RiskAlert(
            timestamp=current_returns.index[-1],
            alert_type="DRAWDOWN_PREDICTION",
            severity=severity,
            current_drawdown=current_drawdown,
            predicted_max_drawdown=predicted_max_dd,
            confidence=confidence,
            recommended_actions=recommendations,
            risk_factors=risk_factors
        )

    def _generate_risk_recommendations(self, current_dd: float,
                                     predicted_dd: float,
                                     recent_returns: pd.Series) -> List[str]:
        """Generate risk management recommendations"""

        recommendations = []

        if predicted_dd > self.max_acceptable_drawdown:
            recommendations.append("URGENT: Reduce position size immediately")
            recommendations.append("Consider closing all positions")
            recommendations.append("Implement emergency stop-loss")

        elif predicted_dd > self.risk_alert_threshold:
            recommendations.append("Reduce position size by 50%")
            recommendations.append("Tighten stop-loss levels")
            recommendations.append("Increase cash allocation")

        elif current_dd < -0.05:
            recommendations.append("Monitor position closely")
            recommendations.append("Prepare for position reduction")

        # Specific recommendations based on recent performance
        if recent_returns.std() > 0.025:
            recommendations.append("High volatility detected - consider volatility targeting")

        if (recent_returns < 0).sum() > 15:  # More than half negative
            recommendations.append("Extended losing streak - review strategy")

        if recent_returns.quantile(0.05) < -0.03:
            recommendations.append("High tail risk - implement tail hedging")

        return recommendations if recommendations else ["Continue monitoring"]

    def optimize_stop_loss_levels(self, returns: pd.Series,
                                equity_curve: pd.Series,
                                stop_loss_range: Tuple[float, float] = (0.02, 0.15)) -> Dict:
        """Optimize stop-loss levels to minimize drawdowns"""

        print("Optimizing stop-loss levels...")

        def simulate_stop_loss(stop_level: float) -> Dict:
            """Simulate trading with stop-loss"""
            equity = equity_curve.copy()
            returns_adj = returns.copy()
            peak = equity.iloc[0]
            stop_triggered = False

            for i in range(1, len(equity)):
                current_value = equity.iloc[i]
                peak = max(peak, current_value)

                # Check stop-loss condition
                drawdown = (current_value - peak) / peak
                if drawdown <= -stop_level:
                    # Stop-loss triggered - go to cash
                    stop_triggered = True
                    returns_adj.iloc[i:] = 0  # No more returns
                    break

                # Update peak only if not stopped out
                if not stop_triggered:
                    peak = max(peak, current_value)

            # Calculate performance metrics
            final_equity = (1 + returns_adj).cumprod() * 200
            max_dd = ((final_equity - final_equity.expanding().max()) / final_equity.expanding().max()).min()
            total_return = (final_equity.iloc[-1] / final_equity.iloc[0]) - 1
            sharpe = (returns_adj.mean() * 252) / (returns_adj.std() * np.sqrt(252)) if returns_adj.std() > 0 else 0

            return {
                'max_drawdown': max_dd,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'stop_triggered': stop_triggered,
                'final_equity': final_equity
            }

        # Test different stop-loss levels
        stop_levels = np.arange(stop_loss_range[0], stop_loss_range[1], 0.01)
        results = {}

        for stop_level in stop_levels:
            results[stop_level] = simulate_stop_loss(stop_level)

        # Find optimal stop-loss (minimize drawdown while maintaining returns)
        best_score = -np.inf
        optimal_stop = stop_levels[0]

        for stop_level, result in results.items():
            # Composite score: Sharpe ratio weighted by drawdown improvement
            drawdown_improvement = abs(result['max_drawdown']) - abs(((equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()).min())
            score = result['sharpe_ratio'] - abs(result['max_drawdown']) * 2

            if score > best_score:
                best_score = score
                optimal_stop = stop_level

        return {
            'optimal_stop_loss': optimal_stop,
            'optimization_results': results,
            'baseline_performance': {
                'max_drawdown': ((equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()).min(),
                'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            }
        }

    def generate_drawdown_report(self, events: List[DrawdownEvent],
                               stats: DrawdownStats,
                               risk_alerts: List[RiskAlert] = None) -> str:
        """Generate comprehensive drawdown analysis report"""

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE DRAWDOWN ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary statistics
        report.append("DRAWDOWN SUMMARY STATISTICS")
        report.append("-" * 35)
        report.append(f"Maximum Drawdown: {stats.max_drawdown:.2%}")
        report.append(f"Average Drawdown: {stats.avg_drawdown:.2%}")
        report.append(f"Drawdown Frequency: {stats.drawdown_frequency:.1%}")
        report.append(f"Average Duration: {stats.avg_duration:.1f} days")
        report.append(f"Maximum Duration: {stats.max_duration} days")
        report.append(f"Average Recovery Time: {stats.avg_recovery_time:.1f} days")
        report.append(f"Maximum Recovery Time: {stats.max_recovery_time} days")
        report.append("")

        # Risk metrics
        report.append("RISK METRICS")
        report.append("-" * 15)
        report.append(f"Pain Index: {stats.pain_index:.3f}")
        report.append(f"Ulcer Index: {stats.ulcer_index:.3f}")
        report.append(f"Calmar Ratio: {stats.calmar_ratio:.3f}")
        report.append(f"Burke Ratio: {stats.burke_ratio:.3f}")
        report.append(f"Lake Ratio: {stats.lake_ratio:.3f}")
        report.append(f"Drawdown at Risk (95%): {stats.drawdown_at_risk:.2%}")
        report.append(f"Worst Month: {stats.worst_month:.2%}")
        report.append(f"Worst Quarter: {stats.worst_quarter:.2%}")
        report.append("")

        # Individual drawdown events
        report.append("MAJOR DRAWDOWN EVENTS")
        report.append("-" * 25)

        significant_events = [event for event in events if abs(event.max_drawdown) > 0.02]
        significant_events.sort(key=lambda x: x.max_drawdown)

        for i, event in enumerate(significant_events[:10], 1):  # Top 10 worst
            report.append(f"\n{i}. {event.severity_level} Drawdown ({event.start_date.strftime('%Y-%m-%d')})")
            report.append(f"   Maximum Drawdown: {event.max_drawdown:.2%}")
            report.append(f"   Duration: {event.duration_days} days")
            report.append(f"   Recovery Time: {event.recovery_days} days")
            report.append(f"   Cause: {event.cause_analysis}")

        # Risk alerts (if provided)
        if risk_alerts:
            report.append("\n\nRISK ALERTS")
            report.append("-" * 15)

            for alert in risk_alerts[-5:]:  # Latest 5 alerts
                report.append(f"\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {alert.severity}")
                report.append(f"Current Drawdown: {alert.current_drawdown:.2%}")
                report.append(f"Predicted Max Drawdown: {alert.predicted_max_drawdown:.2%}")
                report.append(f"Confidence: {alert.confidence:.1%}")
                report.append("Recommendations:")
                for action in alert.recommended_actions:
                    report.append(f"  • {action}")

        # Recommendations
        report.append("\n\nRECOMMENDations")
        report.append("-" * 20)

        if stats.max_drawdown < -0.20:
            report.append("• CRITICAL: Implement emergency risk controls")
            report.append("• Consider position sizing optimization")

        if stats.avg_duration > 30:
            report.append("• Long drawdown periods detected - improve exit strategy")

        if stats.max_recovery_time > 90:
            report.append("• Slow recovery times - consider rebalancing mechanisms")

        if stats.drawdown_frequency > 0.3:
            report.append("• High drawdown frequency - review signal quality")

        report.append("• Implement dynamic position sizing based on volatility")
        report.append("• Consider stop-loss optimization")
        report.append("• Monitor correlation with market stress periods")
        report.append("• Regular review of risk parameters")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def plot_drawdown_analysis(self, equity_curve: pd.Series,
                             events: List[DrawdownEvent],
                             stats: DrawdownStats) -> go.Figure:
        """Create comprehensive drawdown visualization"""

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve with Drawdowns', 'Drawdown Timeline',
                          'Drawdown Distribution', 'Duration vs Magnitude',
                          'Recovery Analysis', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Equity curve with drawdown periods
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve,
                      name='Equity Curve', line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=peak.index, y=peak,
                      name='Peak', line=dict(color='green', dash='dash')),
            row=1, col=1
        )

        # 2. Drawdown timeline
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown,
                      name='Drawdown', fill='tonegative',
                      line=dict(color='red')),
            row=1, col=2
        )

        # 3. Drawdown distribution
        drawdown_magnitudes = [abs(event.max_drawdown) for event in events]
        fig.add_trace(
            go.Histogram(x=drawdown_magnitudes, nbinsx=20,
                        name='Drawdown Distribution'),
            row=2, col=1
        )

        # 4. Duration vs Magnitude scatter
        durations = [event.total_days for event in events]
        magnitudes = [abs(event.max_drawdown) for event in events]

        fig.add_trace(
            go.Scatter(x=durations, y=magnitudes, mode='markers',
                      name='Duration vs Magnitude',
                      marker=dict(size=8, opacity=0.6)),
            row=2, col=2
        )

        # 5. Recovery analysis
        recovery_times = [event.recovery_days for event in events if event.recovery_days > 0]
        recovery_rates = [event.recovery_rate for event in events if event.recovery_rate > 0]

        fig.add_trace(
            go.Scatter(x=recovery_times, y=recovery_rates, mode='markers',
                      name='Recovery Time vs Rate',
                      marker=dict(size=8, opacity=0.6, color='green')),
            row=3, col=1
        )

        # 6. Risk metrics radar (simplified bar chart)
        metrics = ['Max DD', 'Avg DD', 'Pain Index', 'Ulcer Index', 'Calmar Ratio']
        values = [abs(stats.max_drawdown), abs(stats.avg_drawdown),
                 stats.pain_index, stats.ulcer_index, stats.calmar_ratio]

        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Risk Metrics'),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Comprehensive Drawdown Analysis",
            height=900,
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Equity Value", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=1, col=2)
        fig.update_xaxes(title_text="Drawdown Magnitude", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Duration (days)", row=2, col=2)
        fig.update_yaxes(title_text="Magnitude", row=2, col=2)
        fig.update_xaxes(title_text="Recovery Time (days)", row=3, col=1)
        fig.update_yaxes(title_text="Recovery Rate", row=3, col=1)
        fig.update_xaxes(title_text="Metric", row=3, col=2)
        fig.update_yaxes(title_text="Value", row=3, col=2)

        return fig

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Simulate returns with some volatility clustering and drawdown periods
    returns = []
    for i in range(len(dates)):
        if i < 50 or i > 300:  # Normal periods
            ret = np.random.normal(0.0005, 0.015)
        else:  # Drawdown period
            ret = np.random.normal(-0.001, 0.025)
        returns.append(ret)

    returns = pd.Series(returns, index=dates)
    equity_curve = (1 + returns).cumprod() * 200

    # Initialize analyzer
    analyzer = DrawdownAnalysis(max_acceptable_drawdown=0.15)

    # Analyze drawdowns
    events, stats = analyzer.analyze_drawdowns(equity_curve, returns)

    # Build risk model
    historical_data = {'strategy': returns}
    analyzer.build_risk_prediction_model(historical_data)

    # Generate prediction
    risk_alert = analyzer.predict_drawdown_risk(returns, equity_curve)

    # Generate report
    report = analyzer.generate_drawdown_report(events, stats, [risk_alert])
    print(report)

    # Create visualization
    fig = analyzer.plot_drawdown_analysis(equity_curve, events, stats)
    fig.show()

    # Optimize stop-loss
    stop_optimization = analyzer.optimize_stop_loss_levels(returns, equity_curve)
    print(f"\nOptimal Stop-Loss Level: {stop_optimization['optimal_stop_loss']:.1%}")