"""
Real-Time Performance Tracking System
Live monitoring and optimization recommendations for trading strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import warnings
import threading
import time
import json
import asyncio
from queue import Queue, Empty
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import websocket
import sqlite3
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

@dataclass
class PerformanceSnapshot:
    """Real-time performance snapshot"""
    timestamp: pd.Timestamp
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    sharpe_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float
    current_position: float
    cash_balance: float
    exposure: float

    # Gary×Taleb specific metrics
    dpi_score: float
    antifragility_score: float
    signal_strength: float

    # Risk metrics
    var_95: float
    current_streak: int
    days_since_peak: int

@dataclass
class OptimizationSignal:
    """Real-time optimization recommendation"""
    timestamp: pd.Timestamp
    signal_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    current_metric: float
    target_metric: float
    confidence: float
    action: str
    parameters: Dict[str, float]
    expected_improvement: float
    risk_impact: str

@dataclass
class AlertCondition:
    """Performance alert condition"""
    name: str
    condition: Callable[[PerformanceSnapshot], bool]
    severity: str
    message_template: str
    cooldown_minutes: int = 30

class RealTimeTracker:
    """Real-time performance tracking and optimization system"""

    def __init__(self, initial_capital: float = 200.0,
                 update_frequency: int = 1,  # seconds
                 alert_config: Dict = None):
        self.initial_capital = initial_capital
        self.update_frequency = update_frequency
        self.alert_config = alert_config or {}

        # Data storage
        self.performance_history = deque(maxlen=10000)  # Keep last 10k snapshots
        self.optimization_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=500)

        # Current state
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        self.is_tracking = False
        self.tracking_thread = None

        # Alert conditions
        self.alert_conditions = self._initialize_alert_conditions()
        self.last_alert_times = {}

        # Optimization models
        self.optimization_models = {}
        self._initialize_optimization_models()

        # Data sources
        self.data_queue = Queue()
        self.db_engine = None

        # Performance targets
        self.targets = {
            'sharpe_ratio': 2.0,
            'max_drawdown': -0.10,
            'win_rate': 0.55,
            'profit_factor': 1.5,
            'monthly_return': 0.05
        }

    def _initialize_alert_conditions(self) -> List[AlertCondition]:
        """Initialize default alert conditions"""

        return [
            AlertCondition(
                name="High Drawdown",
                condition=lambda snap: snap.drawdown < -0.05,
                severity="HIGH",
                message_template="Drawdown exceeded 5%: {drawdown:.2%}",
                cooldown_minutes=15
            ),
            AlertCondition(
                name="Critical Drawdown",
                condition=lambda snap: snap.drawdown < -0.10,
                severity="CRITICAL",
                message_template="CRITICAL: Drawdown exceeded 10%: {drawdown:.2%}",
                cooldown_minutes=5
            ),
            AlertCondition(
                name="Low Sharpe Ratio",
                condition=lambda snap: snap.sharpe_ratio < 0.5,
                severity="MEDIUM",
                message_template="Sharpe ratio below target: {sharpe_ratio:.2f}",
                cooldown_minutes=60
            ),
            AlertCondition(
                name="High Volatility",
                condition=lambda snap: snap.volatility > 0.25,
                severity="MEDIUM",
                message_template="High volatility detected: {volatility:.2%}",
                cooldown_minutes=30
            ),
            AlertCondition(
                name="Extended Losing Streak",
                condition=lambda snap: snap.current_streak < -5,
                severity="HIGH",
                message_template="Extended losing streak: {current_streak} consecutive losses",
                cooldown_minutes=45
            ),
            AlertCondition(
                name="Long Time Since Peak",
                condition=lambda snap: snap.days_since_peak > 30,
                severity="MEDIUM",
                message_template="No new peak for {days_since_peak} days",
                cooldown_minutes=120
            ),
            AlertCondition(
                name="Low DPI Score",
                condition=lambda snap: snap.dpi_score < 0.3,
                severity="MEDIUM",
                message_template="DPI score below threshold: {dpi_score:.2f}",
                cooldown_minutes=60
            ),
            AlertCondition(
                name="Low Antifragility",
                condition=lambda snap: snap.antifragility_score < 0.2,
                severity="MEDIUM",
                message_template="Low antifragility score: {antifragility_score:.2f}",
                cooldown_minutes=90
            )
        ]

    def _initialize_optimization_models(self):
        """Initialize optimization models"""

        # Position sizing optimizer
        self.optimization_models['position_sizing'] = {
            'type': 'kelly_criterion',
            'parameters': {'confidence_threshold': 0.6, 'max_position': 1.0}
        }

        # Risk management optimizer
        self.optimization_models['risk_management'] = {
            'type': 'adaptive_stops',
            'parameters': {'base_stop': 0.02, 'volatility_multiplier': 2.0}
        }

        # Signal filtering optimizer
        self.optimization_models['signal_filtering'] = {
            'type': 'dynamic_threshold',
            'parameters': {'base_threshold': 0.6, 'adaptation_rate': 0.1}
        }

    def start_tracking(self, data_source: str = "live"):
        """Start real-time tracking"""

        if self.is_tracking:
            print("Tracking already active")
            return

        print(f"Starting real-time performance tracking (update frequency: {self.update_frequency}s)")

        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, args=(data_source,))
        self.tracking_thread.daemon = True
        self.tracking_thread.start()

        print("Real-time tracking started successfully")

    def stop_tracking(self):
        """Stop real-time tracking"""

        if not self.is_tracking:
            print("Tracking not active")
            return

        print("Stopping real-time tracking...")
        self.is_tracking = False

        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)

        print("Real-time tracking stopped")

    def _tracking_loop(self, data_source: str):
        """Main tracking loop"""

        while self.is_tracking:
            try:
                # Get latest data
                latest_data = self._get_latest_data(data_source)

                if latest_data:
                    # Calculate performance snapshot
                    snapshot = self._calculate_performance_snapshot(latest_data)

                    # Store snapshot
                    self.performance_history.append(snapshot)
                    self.current_snapshot = snapshot

                    # Check alerts
                    self._check_alerts(snapshot)

                    # Generate optimization recommendations
                    optimization_signals = self._generate_optimization_signals(snapshot)

                    for signal in optimization_signals:
                        self.optimization_history.append(signal)

                    # Log performance
                    self._log_performance(snapshot, optimization_signals)

                time.sleep(self.update_frequency)

            except Exception as e:
                print(f"Error in tracking loop: {str(e)}")
                time.sleep(self.update_frequency)

    def _get_latest_data(self, data_source: str) -> Optional[Dict]:
        """Get latest market and portfolio data"""

        if data_source == "simulation":
            # Simulated data for testing
            return self._generate_simulated_data()
        elif data_source == "queue":
            # Get from data queue
            try:
                return self.data_queue.get_nowait()
            except Empty:
                return None
        else:
            # Live data source (implement based on your data provider)
            return self._get_live_data()

    def _generate_simulated_data(self) -> Dict:
        """Generate simulated market data for testing"""

        current_time = pd.Timestamp.now()

        # Simulate market movement
        price_change = np.random.normal(0.0005, 0.015)
        current_price = 100 * (1 + price_change)

        # Simulate portfolio state
        position_size = np.random.uniform(0.5, 1.0)
        cash_balance = self.initial_capital * (1 - position_size)
        portfolio_value = cash_balance + (position_size * self.initial_capital * (current_price / 100))

        return {
            'timestamp': current_time,
            'price': current_price,
            'volume': np.random.randint(1000, 10000),
            'portfolio_value': portfolio_value,
            'position_size': position_size,
            'cash_balance': cash_balance,
            'daily_return': price_change,
            'signal_strength': np.random.uniform(0.3, 0.9)
        }

    def _get_live_data(self) -> Optional[Dict]:
        """Get live market data (implement based on your broker/data provider)"""

        # Placeholder for live data implementation
        # This would connect to your actual data source
        return None

    def _calculate_performance_snapshot(self, data: Dict) -> PerformanceSnapshot:
        """Calculate comprehensive performance snapshot"""

        timestamp = data['timestamp']
        portfolio_value = data['portfolio_value']

        # Calculate returns
        if len(self.performance_history) > 0:
            prev_value = self.performance_history[-1].portfolio_value
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0.0

        cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital

        # Calculate drawdown
        if len(self.performance_history) > 0:
            historical_values = [snap.portfolio_value for snap in self.performance_history]
            peak_value = max(historical_values + [portfolio_value])
        else:
            peak_value = portfolio_value

        drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0

        # Calculate performance metrics (rolling)
        if len(self.performance_history) >= 30:
            recent_returns = [snap.daily_return for snap in list(self.performance_history)[-30:]]
            recent_returns.append(daily_return)

            sharpe_ratio = self._calculate_rolling_sharpe(recent_returns)
            volatility = np.std(recent_returns) * np.sqrt(252)
            win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)

            positive_returns = [r for r in recent_returns if r > 0]
            negative_returns = [r for r in recent_returns if r < 0]

            profit_factor = (sum(positive_returns) / abs(sum(negative_returns))) if negative_returns else float('inf')
        else:
            sharpe_ratio = 0.0
            volatility = 0.0
            win_rate = 0.5
            profit_factor = 1.0

        # Calculate Gary×Taleb specific metrics
        dpi_score = self._calculate_dpi_score(data)
        antifragility_score = self._calculate_antifragility_score()

        # Risk metrics
        if len(self.performance_history) >= 20:
            recent_returns = [snap.daily_return for snap in list(self.performance_history)[-20:]]
            var_95 = np.percentile(recent_returns, 5)
        else:
            var_95 = 0.0

        # Streak calculation
        current_streak = self._calculate_current_streak()

        # Days since peak
        days_since_peak = self._calculate_days_since_peak()

        return PerformanceSnapshot(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            current_position=data.get('position_size', 0),
            cash_balance=data.get('cash_balance', 0),
            exposure=data.get('position_size', 0),
            dpi_score=dpi_score,
            antifragility_score=antifragility_score,
            signal_strength=data.get('signal_strength', 0),
            var_95=var_95,
            current_streak=current_streak,
            days_since_peak=days_since_peak
        )

    def _calculate_rolling_sharpe(self, returns: List[float]) -> float:
        """Calculate rolling Sharpe ratio"""

        if len(returns) < 5:
            return 0.0

        excess_return = np.mean(returns) * 252 - 0.02  # Assuming 2% risk-free rate
        volatility = np.std(returns) * np.sqrt(252)

        return excess_return / volatility if volatility > 0 else 0

    def _calculate_dpi_score(self, data: Dict) -> float:
        """Calculate Gary's Dynamic Performance Indicator score"""

        if len(self.performance_history) < 20:
            return 0.5  # Neutral score

        # Momentum component
        recent_returns = [snap.daily_return for snap in list(self.performance_history)[-20:]]
        momentum = np.mean(recent_returns[-5:]) / np.mean(recent_returns[-20:]) if np.mean(recent_returns[-20:]) != 0 else 1
        momentum_component = np.tanh(momentum)  # Normalize to [-1, 1]

        # Stability component
        volatility_stability = 1 - (np.std(recent_returns[-10:]) / np.std(recent_returns[-20:])) if np.std(recent_returns[-20:]) > 0 else 0
        stability_component = max(0, min(1, volatility_stability))

        # Signal strength component
        signal_strength = data.get('signal_strength', 0.5)

        # Combined DPI score
        dpi_score = (momentum_component * 0.4 + stability_component * 0.3 + signal_strength * 0.3)

        return max(0, min(1, (dpi_score + 1) / 2))  # Normalize to [0, 1]

    def _calculate_antifragility_score(self) -> float:
        """Calculate Taleb-inspired antifragility score"""

        if len(self.performance_history) < 20:
            return 0.5  # Neutral score

        recent_returns = [snap.daily_return for snap in list(self.performance_history)[-20:]]
        recent_volatilities = [abs(r) for r in recent_returns]

        # Correlation between volatility and returns (antifragile systems benefit from volatility)
        if len(recent_returns) > 5:
            correlation = np.corrcoef(recent_volatilities, recent_returns)[0, 1]
            antifragility_score = max(0, correlation) if not np.isnan(correlation) else 0
        else:
            antifragility_score = 0

        return antifragility_score

    def _calculate_current_streak(self) -> int:
        """Calculate current winning/losing streak"""

        if len(self.performance_history) == 0:
            return 0

        streak = 0
        last_positive = None

        for snap in reversed(list(self.performance_history)):
            if snap.daily_return > 0:
                if last_positive is None:
                    last_positive = True
                if last_positive:
                    streak += 1
                else:
                    break
            elif snap.daily_return < 0:
                if last_positive is None:
                    last_positive = False
                if not last_positive:
                    streak -= 1
                else:
                    break
            else:
                break  # Zero return breaks the streak

        return streak

    def _calculate_days_since_peak(self) -> int:
        """Calculate days since last equity peak"""

        if len(self.performance_history) == 0:
            return 0

        historical_values = [snap.portfolio_value for snap in self.performance_history]
        max_value = max(historical_values)

        # Find the last occurrence of max value
        for i, snap in enumerate(reversed(list(self.performance_history))):
            if snap.portfolio_value == max_value:
                return i

        return len(self.performance_history)

    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check alert conditions and trigger alerts"""

        current_time = snapshot.timestamp

        for condition in self.alert_conditions:
            # Check cooldown
            if condition.name in self.last_alert_times:
                time_since_last = (current_time - self.last_alert_times[condition.name]).total_seconds() / 60
                if time_since_last < condition.cooldown_minutes:
                    continue

            # Check condition
            if condition.condition(snapshot):
                alert_message = condition.message_template.format(**asdict(snapshot))

                alert = {
                    'timestamp': current_time,
                    'condition': condition.name,
                    'severity': condition.severity,
                    'message': alert_message,
                    'snapshot': snapshot
                }

                self.alert_history.append(alert)
                self.last_alert_times[condition.name] = current_time

                # Trigger alert action
                self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict):
        """Trigger alert action (logging, notifications, etc.)"""

        print(f"[{alert['timestamp']}] {alert['severity']}: {alert['message']}")

        # Here you could implement additional alert actions:
        # - Send email/SMS notifications
        # - Write to log files
        # - Send to monitoring systems
        # - Trigger automatic actions

    def _generate_optimization_signals(self, snapshot: PerformanceSnapshot) -> List[OptimizationSignal]:
        """Generate real-time optimization recommendations"""

        signals = []
        current_time = snapshot.timestamp

        # Position sizing optimization
        if snapshot.volatility > 0.20:  # High volatility
            optimal_position = self._calculate_optimal_position_size(snapshot)
            if abs(optimal_position - snapshot.current_position) > 0.1:
                signals.append(OptimizationSignal(
                    timestamp=current_time,
                    signal_type="position_sizing",
                    severity="MEDIUM",
                    current_metric=snapshot.current_position,
                    target_metric=optimal_position,
                    confidence=0.75,
                    action=f"Adjust position size to {optimal_position:.1%}",
                    parameters={'target_position': optimal_position},
                    expected_improvement=0.05,
                    risk_impact="Reduces volatility exposure"
                ))

        # Sharpe ratio optimization
        if snapshot.sharpe_ratio < self.targets['sharpe_ratio'] * 0.7:
            target_sharpe = self._calculate_target_sharpe(snapshot)
            signals.append(OptimizationSignal(
                timestamp=current_time,
                signal_type="sharpe_optimization",
                severity="HIGH",
                current_metric=snapshot.sharpe_ratio,
                target_metric=target_sharpe,
                confidence=0.65,
                action="Improve signal filtering and risk management",
                parameters={'signal_threshold': 0.7, 'risk_multiplier': 1.5},
                expected_improvement=0.15,
                risk_impact="Potential reduction in trade frequency"
            ))

        # DPI score optimization
        if snapshot.dpi_score < 0.4:
            signals.append(OptimizationSignal(
                timestamp=current_time,
                signal_type="dpi_optimization",
                severity="MEDIUM",
                current_metric=snapshot.dpi_score,
                target_metric=0.6,
                confidence=0.60,
                action="Recalibrate DPI parameters",
                parameters={'momentum_weight': 0.5, 'stability_weight': 0.3},
                expected_improvement=0.10,
                risk_impact="May increase signal sensitivity"
            ))

        # Drawdown management
        if snapshot.drawdown < -0.05:
            signals.append(OptimizationSignal(
                timestamp=current_time,
                signal_type="drawdown_management",
                severity="HIGH",
                current_metric=snapshot.drawdown,
                target_metric=-0.03,
                confidence=0.80,
                action="Implement defensive positioning",
                parameters={'stop_loss': 0.02, 'position_reduction': 0.5},
                expected_improvement=0.02,
                risk_impact="Reduces maximum loss potential"
            ))

        return signals

    def _calculate_optimal_position_size(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate optimal position size using Kelly criterion"""

        if snapshot.win_rate <= 0.5 or snapshot.profit_factor <= 1:
            return 0.2  # Conservative position

        # Simplified Kelly criterion
        win_prob = snapshot.win_rate
        loss_prob = 1 - win_prob
        avg_win = snapshot.daily_return if snapshot.daily_return > 0 else 0.01
        avg_loss = abs(snapshot.daily_return) if snapshot.daily_return < 0 else 0.01

        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win

        # Apply volatility adjustment
        vol_adjustment = max(0.5, 1 - snapshot.volatility)
        optimal_position = kelly_fraction * vol_adjustment

        return max(0.1, min(1.0, optimal_position))

    def _calculate_target_sharpe(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate achievable target Sharpe ratio"""

        base_target = self.targets['sharpe_ratio']

        # Adjust based on current market conditions
        if snapshot.volatility > 0.25:
            return base_target * 0.8  # Lower target in high vol environment
        elif snapshot.volatility < 0.15:
            return base_target * 1.2  # Higher target in low vol environment
        else:
            return base_target

    def _log_performance(self, snapshot: PerformanceSnapshot, signals: List[OptimizationSignal]):
        """Log performance data"""

        # Basic logging
        if len(self.performance_history) % 60 == 0:  # Log every 60 updates
            print(f"[{snapshot.timestamp}] Portfolio: ${snapshot.portfolio_value:.2f}, "
                  f"Return: {snapshot.cumulative_return:.2%}, DD: {snapshot.drawdown:.2%}, "
                  f"Sharpe: {snapshot.sharpe_ratio:.2f}, DPI: {snapshot.dpi_score:.2f}")

        # Log optimization signals
        for signal in signals:
            print(f"  OPTIMIZATION: {signal.action} (Confidence: {signal.confidence:.1%})")

    def get_performance_dashboard_data(self) -> Dict:
        """Get data for performance dashboard"""

        if not self.performance_history:
            return {}

        # Convert to DataFrame for easier analysis
        df_data = []
        for snapshot in self.performance_history:
            df_data.append(asdict(snapshot))

        df = pd.DataFrame(df_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Calculate additional metrics
        dashboard_data = {
            'current_performance': asdict(self.current_snapshot) if self.current_snapshot else {},
            'time_series': {
                'portfolio_value': df['portfolio_value'].to_dict(),
                'cumulative_return': df['cumulative_return'].to_dict(),
                'drawdown': df['drawdown'].to_dict(),
                'sharpe_ratio': df['sharpe_ratio'].to_dict(),
                'dpi_score': df['dpi_score'].to_dict(),
                'antifragility_score': df['antifragility_score'].to_dict()
            },
            'summary_stats': {
                'total_return': df['cumulative_return'].iloc[-1] if not df.empty else 0,
                'max_drawdown': df['drawdown'].min() if not df.empty else 0,
                'current_sharpe': df['sharpe_ratio'].iloc[-1] if not df.empty else 0,
                'avg_dpi': df['dpi_score'].mean() if not df.empty else 0,
                'win_rate': df['win_rate'].iloc[-1] if not df.empty else 0,
                'profit_factor': df['profit_factor'].iloc[-1] if not df.empty else 0
            },
            'recent_alerts': list(self.alert_history)[-10:],
            'recent_optimizations': list(self.optimization_history)[-10:],
            'targets': self.targets
        }

        return dashboard_data

    def create_live_dashboard(self) -> go.Figure:
        """Create live performance dashboard"""

        dashboard_data = self.get_performance_dashboard_data()

        if not dashboard_data:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Drawdown', 'Sharpe Ratio Evolution',
                          'DPI vs Antifragility', 'Performance Metrics', 'Alert Status'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        time_series = dashboard_data['time_series']
        timestamps = list(time_series['portfolio_value'].keys())

        # 1. Portfolio value
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(time_series['portfolio_value'].values()),
                      name='Portfolio Value', line=dict(color='blue')),
            row=1, col=1
        )

        # 2. Drawdown
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(time_series['drawdown'].values()),
                      name='Drawdown', fill='tonegative', line=dict(color='red')),
            row=1, col=2
        )

        # 3. Sharpe ratio
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(time_series['sharpe_ratio'].values()),
                      name='Sharpe Ratio', line=dict(color='green')),
            row=2, col=1
        )

        # Add target line
        fig.add_hline(y=self.targets['sharpe_ratio'], line_dash="dash",
                     line_color="gray", row=2, col=1)

        # 4. DPI vs Antifragility scatter
        fig.add_trace(
            go.Scatter(x=list(time_series['dpi_score'].values()),
                      y=list(time_series['antifragility_score'].values()),
                      mode='markers', name='DPI vs Antifragility',
                      marker=dict(color=timestamps, colorscale='viridis')),
            row=2, col=2
        )

        # 5. Performance metrics (current values)
        current_perf = dashboard_data['current_performance']
        metrics = ['Win Rate', 'Profit Factor', 'Volatility', 'Exposure']
        values = [current_perf.get('win_rate', 0), current_perf.get('profit_factor', 0),
                 current_perf.get('volatility', 0), current_perf.get('exposure', 0)]

        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Current Metrics'),
            row=3, col=1
        )

        # 6. Alert status (simplified)
        recent_alerts = dashboard_data['recent_alerts']
        alert_counts = {}
        for alert in recent_alerts:
            severity = alert['severity']
            alert_counts[severity] = alert_counts.get(severity, 0) + 1

        if alert_counts:
            fig.add_trace(
                go.Bar(x=list(alert_counts.keys()), y=list(alert_counts.values()),
                      name='Recent Alerts', marker_color='orange'),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Real-Time Performance Dashboard",
            height=800,
            showlegend=True
        )

        return fig

    def save_performance_data(self, filename: str = None):
        """Save performance data to file"""

        if not filename:
            filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data_to_save = {
            'performance_history': [asdict(snap) for snap in self.performance_history],
            'optimization_history': [asdict(signal) for signal in self.optimization_history],
            'alert_history': list(self.alert_history),
            'targets': self.targets,
            'config': {
                'initial_capital': self.initial_capital,
                'update_frequency': self.update_frequency
            }
        }

        # Convert timestamps to strings for JSON serialization
        for record in data_to_save['performance_history']:
            record['timestamp'] = record['timestamp'].isoformat()

        for record in data_to_save['optimization_history']:
            record['timestamp'] = record['timestamp'].isoformat()

        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)

        print(f"Performance data saved to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = RealTimeTracker(initial_capital=200.0, update_frequency=1)

    print("Starting real-time performance tracking demo...")

    # Start tracking with simulated data
    tracker.start_tracking(data_source="simulation")

    # Let it run for a demo period
    time.sleep(30)  # Run for 30 seconds

    # Get dashboard data
    dashboard_data = tracker.get_performance_dashboard_data()
    print("\nCurrent Performance Summary:")
    print(f"Portfolio Value: ${dashboard_data['summary_stats']['total_return']:.2f}")
    print(f"Total Return: {dashboard_data['summary_stats']['total_return']:.2%}")
    print(f"Max Drawdown: {dashboard_data['summary_stats']['max_drawdown']:.2%}")
    print(f"Current Sharpe: {dashboard_data['summary_stats']['current_sharpe']:.2f}")

    # Create dashboard visualization
    fig = tracker.create_live_dashboard()
    fig.show()

    # Save data
    tracker.save_performance_data()

    # Stop tracking
    tracker.stop_tracking()

    print("Demo completed successfully!")