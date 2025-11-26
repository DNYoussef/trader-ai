"""
Performance Feedback System for Gary×Taleb Trading System

Tracks actual vs predicted returns, calculates performance feedback loops,
and provides adaptive learning signals for continuous model improvement.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import sqlite3
from pathlib import Path
import threading
import time
from collections import deque
import statistics
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score

@dataclass
class FeedbackMetrics:
    """Performance feedback metrics"""
    model_id: str
    timestamp: datetime
    actual_return: float
    predicted_return: float
    prediction_error: float
    absolute_error: float
    squared_error: float
    direction_accuracy: bool
    confidence_score: float
    market_regime: str
    volatility_bucket: str
    trade_size: float
    holding_period: float
    transaction_costs: float
    slippage: float

@dataclass
class PerformanceWindow:
    """Performance metrics over a time window"""
    start_time: datetime
    end_time: datetime
    total_trades: int
    total_pnl: float
    predicted_pnl: float
    mse: float
    mae: float
    r2: float
    direction_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    gary_dpi: float
    taleb_antifragility: float
    correlation_actual_predicted: float
    bias_factor: float
    overconfidence_score: float

@dataclass
class FeedbackSignal:
    """Adaptive learning signal based on performance feedback"""
    signal_type: str  # 'retrain', 'adjust_parameters', 'change_strategy', 'rollback'
    urgency: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    reason: str
    suggested_actions: List[str]
    performance_delta: float
    affected_components: List[str]

class PerformanceFeedback:
    """
    Performance feedback system that tracks actual vs predicted returns
    and generates adaptive learning signals for model improvement.
    """

    def __init__(self, window_size_hours: int = 24, min_samples: int = 10):
        self.window_size_hours = window_size_hours
        self.min_samples = min_samples
        self.logger = self._setup_logging()

        # Database setup
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/performance_feedback.db")
        self._init_database()

        # Performance tracking
        self.feedback_buffer = deque(maxlen=1000)
        self.performance_windows: Dict[str, List[PerformanceWindow]] = {}
        self.feedback_signals: List[FeedbackSignal] = []

        # Threading for real-time feedback
        self.is_running = False
        self.feedback_thread = None

        # Market regime detection
        self.market_regimes = ['trending_up', 'trending_down', 'sideways', 'high_volatility', 'low_volatility']
        self.volatility_buckets = ['very_low', 'low', 'medium', 'high', 'very_high']

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance feedback"""
        logger = logging.getLogger('PerformanceFeedback')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/performance_feedback.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for feedback tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actual_return REAL,
                    predicted_return REAL,
                    prediction_error REAL,
                    absolute_error REAL,
                    squared_error REAL,
                    direction_accuracy BOOLEAN,
                    confidence_score REAL,
                    market_regime TEXT,
                    volatility_bucket TEXT,
                    trade_size REAL,
                    holding_period REAL,
                    transaction_costs REAL,
                    slippage REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    total_trades INTEGER,
                    total_pnl REAL,
                    predicted_pnl REAL,
                    mse REAL,
                    mae REAL,
                    r2 REAL,
                    direction_accuracy REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    gary_dpi REAL,
                    taleb_antifragility REAL,
                    correlation_actual_predicted REAL,
                    bias_factor REAL,
                    overconfidence_score REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    confidence REAL,
                    reason TEXT,
                    suggested_actions TEXT,
                    performance_delta REAL,
                    affected_components TEXT,
                    model_id TEXT
                )
            ''')

    def start_feedback_system(self):
        """Start the performance feedback system"""
        if self.is_running:
            self.logger.warning("Performance feedback system already running")
            return

        self.is_running = True
        self.logger.info("Starting performance feedback system")

        # Start background feedback processing
        self.feedback_thread = threading.Thread(target=self._process_feedback_loop, daemon=True)
        self.feedback_thread.start()

        self.logger.info("Performance feedback system started")

    def stop_feedback_system(self):
        """Stop the performance feedback system"""
        self.is_running = False
        self.logger.info("Performance feedback system stopped")

    def _process_feedback_loop(self):
        """Main feedback processing loop"""
        while self.is_running:
            try:
                # Process recent feedback data
                self._process_recent_feedback()

                # Generate adaptive signals
                self._generate_feedback_signals()

                # Update performance windows
                self._update_performance_windows()

                time.sleep(30)  # Process every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in feedback processing loop: {e}")
                time.sleep(60)

    def record_feedback(self, model_id: str, actual_return: float, predicted_return: float,
                       trade_data: Dict[str, Any]) -> FeedbackMetrics:
        """
        Record feedback for a completed trade

        Args:
            model_id: ID of the model that made the prediction
            actual_return: Actual return achieved
            predicted_return: Return predicted by the model
            trade_data: Additional trade information

        Returns:
            FeedbackMetrics object
        """
        try:
            # Calculate feedback metrics
            prediction_error = actual_return - predicted_return
            absolute_error = abs(prediction_error)
            squared_error = prediction_error ** 2

            # Direction accuracy
            actual_direction = 1 if actual_return > 0 else -1
            predicted_direction = 1 if predicted_return > 0 else -1
            direction_accuracy = actual_direction == predicted_direction

            # Market regime detection
            market_regime = self._detect_market_regime(trade_data)
            volatility_bucket = self._classify_volatility(trade_data)

            # Create feedback metrics
            feedback = FeedbackMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                actual_return=actual_return,
                predicted_return=predicted_return,
                prediction_error=prediction_error,
                absolute_error=absolute_error,
                squared_error=squared_error,
                direction_accuracy=direction_accuracy,
                confidence_score=trade_data.get('confidence', 0.5),
                market_regime=market_regime,
                volatility_bucket=volatility_bucket,
                trade_size=trade_data.get('size', 0.0),
                holding_period=trade_data.get('holding_period', 0.0),
                transaction_costs=trade_data.get('transaction_costs', 0.0),
                slippage=trade_data.get('slippage', 0.0)
            )

            # Store in buffer and database
            self.feedback_buffer.append(feedback)
            self._save_feedback_to_db(feedback)

            self.logger.info(f"Recorded feedback for model {model_id}: error={prediction_error:.4f}")

            return feedback

        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            raise

    def _detect_market_regime(self, trade_data: Dict[str, Any]) -> str:
        """Detect current market regime"""
        try:
            # Simple regime detection based on recent price movements
            # In practice, this would use more sophisticated techniques
            volatility = trade_data.get('volatility', 0.02)
            trend_strength = trade_data.get('trend_strength', 0.0)

            if volatility > 0.05:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'low_volatility'
            elif trend_strength > 0.02:
                return 'trending_up'
            elif trend_strength < -0.02:
                return 'trending_down'
            else:
                return 'sideways'

        except Exception:
            return 'unknown'

    def _classify_volatility(self, trade_data: Dict[str, Any]) -> str:
        """Classify current volatility level"""
        try:
            volatility = trade_data.get('volatility', 0.02)

            if volatility < 0.005:
                return 'very_low'
            elif volatility < 0.015:
                return 'low'
            elif volatility < 0.03:
                return 'medium'
            elif volatility < 0.06:
                return 'high'
            else:
                return 'very_high'

        except Exception:
            return 'medium'

    def _save_feedback_to_db(self, feedback: FeedbackMetrics):
        """Save feedback metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feedback_metrics (
                    model_id, timestamp, actual_return, predicted_return,
                    prediction_error, absolute_error, squared_error, direction_accuracy,
                    confidence_score, market_regime, volatility_bucket,
                    trade_size, holding_period, transaction_costs, slippage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.model_id,
                feedback.timestamp.isoformat(),
                feedback.actual_return,
                feedback.predicted_return,
                feedback.prediction_error,
                feedback.absolute_error,
                feedback.squared_error,
                feedback.direction_accuracy,
                feedback.confidence_score,
                feedback.market_regime,
                feedback.volatility_bucket,
                feedback.trade_size,
                feedback.holding_period,
                feedback.transaction_costs,
                feedback.slippage
            ))

    def _process_recent_feedback(self):
        """Process recent feedback to identify patterns"""
        try:
            if len(self.feedback_buffer) < self.min_samples:
                return

            # Analyze recent feedback patterns
            recent_errors = [f.prediction_error for f in list(self.feedback_buffer)[-20:]]
            recent_accuracy = [f.direction_accuracy for f in list(self.feedback_buffer)[-20:]]

            # Check for systematic bias
            mean_error = statistics.mean(recent_errors)
            if abs(mean_error) > 0.01:  # 1% systematic bias
                self._generate_bias_signal(mean_error)

            # Check for accuracy degradation
            accuracy_rate = sum(recent_accuracy) / len(recent_accuracy)
            if accuracy_rate < 0.4:  # Below 40% accuracy
                self._generate_accuracy_signal(accuracy_rate)

            # Check for error variance increase
            error_variance = statistics.variance(recent_errors)
            if error_variance > 0.01:  # High variance threshold
                self._generate_variance_signal(error_variance)

        except Exception as e:
            self.logger.error(f"Error processing recent feedback: {e}")

    def _generate_bias_signal(self, bias: float):
        """Generate signal for systematic bias"""
        urgency = 'high' if abs(bias) > 0.02 else 'medium'

        signal = FeedbackSignal(
            signal_type='adjust_parameters',
            urgency=urgency,
            confidence=0.8,
            reason=f"Systematic bias detected: {bias:.4f}",
            suggested_actions=[
                'Recalibrate model predictions',
                'Check feature scaling',
                'Analyze training data distribution'
            ],
            performance_delta=bias,
            affected_components=['prediction_model', 'feature_preprocessor']
        )

        self.feedback_signals.append(signal)
        self._save_signal_to_db(signal)

    def _generate_accuracy_signal(self, accuracy: float):
        """Generate signal for low accuracy"""
        urgency = 'critical' if accuracy < 0.3 else 'high'

        signal = FeedbackSignal(
            signal_type='retrain',
            urgency=urgency,
            confidence=0.9,
            reason=f"Direction accuracy dropped to {accuracy:.2%}",
            suggested_actions=[
                'Immediate model retraining',
                'Feature importance analysis',
                'Check for regime change'
            ],
            performance_delta=accuracy - 0.5,
            affected_components=['prediction_model', 'feature_engineering']
        )

        self.feedback_signals.append(signal)
        self._save_signal_to_db(signal)

    def _generate_variance_signal(self, variance: float):
        """Generate signal for high prediction variance"""
        signal = FeedbackSignal(
            signal_type='adjust_parameters',
            urgency='medium',
            confidence=0.7,
            reason=f"Prediction variance increased to {variance:.4f}",
            suggested_actions=[
                'Tune model regularization',
                'Review feature stability',
                'Consider ensemble methods'
            ],
            performance_delta=-variance,
            affected_components=['prediction_model']
        )

        self.feedback_signals.append(signal)
        self._save_signal_to_db(signal)

    def _generate_feedback_signals(self):
        """Generate comprehensive feedback signals"""
        try:
            # Get models with recent feedback
            models_with_feedback = set(f.model_id for f in self.feedback_buffer)

            for model_id in models_with_feedback:
                # Analyze model-specific performance
                model_feedback = [f for f in self.feedback_buffer if f.model_id == model_id]

                if len(model_feedback) >= self.min_samples:
                    performance_window = self._calculate_performance_window(model_id, model_feedback)

                    # Check for performance degradation
                    if performance_window.gary_dpi < -0.1:
                        self._generate_performance_degradation_signal(model_id, performance_window)

                    # Check for overconfidence
                    if performance_window.overconfidence_score > 0.3:
                        self._generate_overconfidence_signal(model_id, performance_window)

        except Exception as e:
            self.logger.error(f"Error generating feedback signals: {e}")

    def _generate_performance_degradation_signal(self, model_id: str, performance: PerformanceWindow):
        """Generate signal for performance degradation"""
        signal = FeedbackSignal(
            signal_type='rollback',
            urgency='high',
            confidence=0.85,
            reason=f"Model {model_id} Gary DPI degraded to {performance.gary_dpi:.4f}",
            suggested_actions=[
                'Rollback to previous model version',
                'Analyze recent market changes',
                'Consider alternative strategies'
            ],
            performance_delta=performance.gary_dpi,
            affected_components=[f'model_{model_id}']
        )

        self.feedback_signals.append(signal)
        self._save_signal_to_db(signal)

    def _generate_overconfidence_signal(self, model_id: str, performance: PerformanceWindow):
        """Generate signal for model overconfidence"""
        signal = FeedbackSignal(
            signal_type='adjust_parameters',
            urgency='medium',
            confidence=0.7,
            reason=f"Model {model_id} showing overconfidence: {performance.overconfidence_score:.4f}",
            suggested_actions=[
                'Calibrate prediction confidence',
                'Reduce position sizes',
                'Implement uncertainty quantification'
            ],
            performance_delta=-performance.overconfidence_score,
            affected_components=[f'model_{model_id}', 'position_sizing']
        )

        self.feedback_signals.append(signal)
        self._save_signal_to_db(signal)

    def _calculate_performance_window(self, model_id: str, feedback_list: List[FeedbackMetrics]) -> PerformanceWindow:
        """Calculate performance metrics for a window of feedback"""
        if not feedback_list:
            return None

        start_time = min(f.timestamp for f in feedback_list)
        end_time = max(f.timestamp for f in feedback_list)

        actual_returns = [f.actual_return for f in feedback_list]
        predicted_returns = [f.predicted_return for f in feedback_list]
        errors = [f.prediction_error for f in feedback_list]

        # Basic metrics
        total_trades = len(feedback_list)
        total_pnl = sum(actual_returns)
        predicted_pnl = sum(predicted_returns)

        # Error metrics
        mse = statistics.mean([f.squared_error for f in feedback_list])
        mae = statistics.mean([f.absolute_error for f in feedback_list])

        # R-squared
        r2 = r2_score(actual_returns, predicted_returns) if len(actual_returns) > 1 else 0

        # Direction accuracy
        direction_accuracy = sum(f.direction_accuracy for f in feedback_list) / total_trades

        # Trading metrics
        sharpe_ratio = np.mean(actual_returns) / np.std(actual_returns) if np.std(actual_returns) > 0 else 0

        # Max drawdown
        cumulative_returns = np.cumprod(1 + np.array(actual_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = sum(1 for r in actual_returns if r > 0) / total_trades

        # Profit factor
        gross_profit = sum(r for r in actual_returns if r > 0)
        gross_loss = abs(sum(r for r in actual_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Gary DPI
        gary_dpi = self._calculate_gary_dpi(actual_returns)

        # Taleb antifragility
        taleb_antifragility = self._calculate_taleb_antifragility(actual_returns)

        # Correlation between actual and predicted
        correlation_actual_predicted = np.corrcoef(actual_returns, predicted_returns)[0, 1] if len(actual_returns) > 1 else 0

        # Bias factor
        bias_factor = statistics.mean(errors)

        # Overconfidence score
        overconfidence_score = self._calculate_overconfidence(feedback_list)

        return PerformanceWindow(
            start_time=start_time,
            end_time=end_time,
            total_trades=total_trades,
            total_pnl=total_pnl,
            predicted_pnl=predicted_pnl,
            mse=mse,
            mae=mae,
            r2=r2,
            direction_accuracy=direction_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            gary_dpi=gary_dpi,
            taleb_antifragility=taleb_antifragility,
            correlation_actual_predicted=correlation_actual_predicted,
            bias_factor=bias_factor,
            overconfidence_score=overconfidence_score
        )

    def _calculate_gary_dpi(self, returns: List[float]) -> float:
        """Calculate Gary's DPI from returns"""
        if not returns:
            return 0.0

        avg_return = statistics.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0

        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        denominator = max_drawdown + volatility
        gary_dpi = (avg_return * win_rate) / denominator if denominator > 0 else 0

        return gary_dpi

    def _calculate_taleb_antifragility(self, returns: List[float]) -> float:
        """Calculate Taleb's antifragility score"""
        if len(returns) < 4:
            return 0.0

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)

        # Identify stress periods (high volatility)
        volatility_threshold = np.percentile(np.abs(returns_array - mean_return), 75)
        stress_periods = np.abs(returns_array - mean_return) > volatility_threshold

        if np.sum(stress_periods) == 0:
            return 0.0

        stress_returns = returns_array[stress_periods]
        normal_returns = returns_array[~stress_periods]

        stress_performance = np.mean(stress_returns) if len(stress_returns) > 0 else 0
        normal_performance = np.mean(normal_returns) if len(normal_returns) > 0 else 0

        antifragility = (stress_performance - normal_performance) / (volatility + 1e-8)

        return antifragility

    def _calculate_overconfidence(self, feedback_list: List[FeedbackMetrics]) -> float:
        """Calculate model overconfidence score"""
        if not feedback_list:
            return 0.0

        # Compare confidence with actual accuracy
        high_confidence_trades = [f for f in feedback_list if f.confidence_score > 0.7]

        if not high_confidence_trades:
            return 0.0

        high_confidence_accuracy = sum(f.direction_accuracy for f in high_confidence_trades) / len(high_confidence_trades)
        overall_accuracy = sum(f.direction_accuracy for f in feedback_list) / len(feedback_list)

        # Overconfidence if high confidence trades don't outperform overall
        overconfidence = max(0, overall_accuracy - high_confidence_accuracy)

        return overconfidence

    def _update_performance_windows(self):
        """Update performance windows for all models"""
        try:
            # Get unique model IDs from recent feedback
            models = set(f.model_id for f in self.feedback_buffer)

            for model_id in models:
                # Get recent feedback for this model
                cutoff_time = datetime.now() - timedelta(hours=self.window_size_hours)
                recent_feedback = [
                    f for f in self.feedback_buffer
                    if f.model_id == model_id and f.timestamp > cutoff_time
                ]

                if len(recent_feedback) >= self.min_samples:
                    performance_window = self._calculate_performance_window(model_id, recent_feedback)

                    if model_id not in self.performance_windows:
                        self.performance_windows[model_id] = []

                    self.performance_windows[model_id].append(performance_window)

                    # Save to database
                    self._save_performance_window_to_db(model_id, performance_window)

        except Exception as e:
            self.logger.error(f"Error updating performance windows: {e}")

    def _save_performance_window_to_db(self, model_id: str, window: PerformanceWindow):
        """Save performance window to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_windows (
                    model_id, start_time, end_time, total_trades, total_pnl, predicted_pnl,
                    mse, mae, r2, direction_accuracy, sharpe_ratio, max_drawdown,
                    win_rate, profit_factor, gary_dpi, taleb_antifragility,
                    correlation_actual_predicted, bias_factor, overconfidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                window.start_time.isoformat(),
                window.end_time.isoformat(),
                window.total_trades,
                window.total_pnl,
                window.predicted_pnl,
                window.mse,
                window.mae,
                window.r2,
                window.direction_accuracy,
                window.sharpe_ratio,
                window.max_drawdown,
                window.win_rate,
                window.profit_factor,
                window.gary_dpi,
                window.taleb_antifragility,
                window.correlation_actual_predicted,
                window.bias_factor,
                window.overconfidence_score
            ))

    def _save_signal_to_db(self, signal: FeedbackSignal):
        """Save feedback signal to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feedback_signals (
                    timestamp, signal_type, urgency, confidence, reason,
                    suggested_actions, performance_delta, affected_components, model_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                signal.signal_type,
                signal.urgency,
                signal.confidence,
                signal.reason,
                json.dumps(signal.suggested_actions),
                signal.performance_delta,
                json.dumps(signal.affected_components),
                signal.affected_components[0] if signal.affected_components else None
            ))

    def get_feedback_summary(self, model_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get feedback summary for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if model_id:
            recent_feedback = [
                f for f in self.feedback_buffer
                if f.model_id == model_id and f.timestamp > cutoff_time
            ]
        else:
            recent_feedback = [
                f for f in self.feedback_buffer
                if f.timestamp > cutoff_time
            ]

        if not recent_feedback:
            return {}

        summary = {
            'total_trades': len(recent_feedback),
            'average_error': statistics.mean([f.prediction_error for f in recent_feedback]),
            'average_absolute_error': statistics.mean([f.absolute_error for f in recent_feedback]),
            'direction_accuracy': sum(f.direction_accuracy for f in recent_feedback) / len(recent_feedback),
            'error_variance': statistics.variance([f.prediction_error for f in recent_feedback]) if len(recent_feedback) > 1 else 0,
            'models_active': len(set(f.model_id for f in recent_feedback)),
            'recent_signals': len([s for s in self.feedback_signals if s.urgency in ['high', 'critical']]),
            'performance_by_regime': self._summarize_by_regime(recent_feedback),
            'performance_by_volatility': self._summarize_by_volatility(recent_feedback)
        }

        return summary

    def _summarize_by_regime(self, feedback_list: List[FeedbackMetrics]) -> Dict[str, Dict[str, float]]:
        """Summarize performance by market regime"""
        regime_summary = {}

        for regime in self.market_regimes:
            regime_feedback = [f for f in feedback_list if f.market_regime == regime]

            if regime_feedback:
                regime_summary[regime] = {
                    'count': len(regime_feedback),
                    'average_error': statistics.mean([f.prediction_error for f in regime_feedback]),
                    'direction_accuracy': sum(f.direction_accuracy for f in regime_feedback) / len(regime_feedback)
                }

        return regime_summary

    def _summarize_by_volatility(self, feedback_list: List[FeedbackMetrics]) -> Dict[str, Dict[str, float]]:
        """Summarize performance by volatility bucket"""
        volatility_summary = {}

        for bucket in self.volatility_buckets:
            bucket_feedback = [f for f in feedback_list if f.volatility_bucket == bucket]

            if bucket_feedback:
                volatility_summary[bucket] = {
                    'count': len(bucket_feedback),
                    'average_error': statistics.mean([f.prediction_error for f in bucket_feedback]),
                    'direction_accuracy': sum(f.direction_accuracy for f in bucket_feedback) / len(bucket_feedback)
                }

        return volatility_summary

    def get_active_signals(self, urgency_filter: Optional[str] = None) -> List[FeedbackSignal]:
        """Get current active feedback signals"""
        if urgency_filter:
            return [s for s in self.feedback_signals if s.urgency == urgency_filter]
        else:
            return self.feedback_signals.copy()

    def clear_processed_signals(self):
        """Clear processed feedback signals"""
        self.feedback_signals.clear()
        self.logger.info("Cleared processed feedback signals")

if __name__ == "__main__":
    # Example usage
    feedback_system = PerformanceFeedback(window_size_hours=12, min_samples=5)
    feedback_system.start_feedback_system()

    print("Performance feedback system started...")
    print("Use feedback_system.record_feedback() to record trade outcomes")
    print("Use feedback_system.get_feedback_summary() to get performance summary")