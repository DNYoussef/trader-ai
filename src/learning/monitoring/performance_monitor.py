"""
Model Performance Monitor for Gary×Taleb Trading System

Real-time model performance monitoring with automatic degradation detection,
alert system, and comprehensive performance analytics dashboard.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path
import threading
import time
from collections import deque
import statistics
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    trend_window_minutes: int = 60
    consecutive_violations: int = 3

@dataclass
class PerformanceAlert:
    """Performance alert data"""
    alert_id: str
    model_id: str
    metric_name: str
    alert_type: str  # 'warning', 'critical', 'recovery'
    current_value: float
    threshold_value: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    timestamp: datetime
    description: str
    suggested_actions: List[str]
    severity_score: float

@dataclass
class ModelHealthStatus:
    """Overall model health status"""
    model_id: str
    timestamp: datetime
    health_score: float  # 0-100
    status: str  # 'healthy', 'warning', 'critical', 'offline'
    active_alerts: List[PerformanceAlert]
    performance_summary: Dict[str, float]
    last_prediction_time: datetime
    prediction_frequency: float  # predictions per minute
    error_rate: float
    drift_score: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    model_id: str
    timestamp: datetime

    # Prediction accuracy
    mae: float
    mse: float
    rmse: float
    r2: float
    direction_accuracy: float

    # Trading performance
    gary_dpi: float
    taleb_antifragility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Model behavior
    prediction_latency: float
    confidence_score: float
    feature_importance_stability: float
    prediction_variance: float

    # System metrics
    memory_usage: float
    cpu_usage: float
    throughput: float

class PerformanceMonitor:
    """
    Real-time model performance monitoring system with automatic
    degradation detection and comprehensive alerting.
    """

    def __init__(self, monitoring_interval_seconds: int = 30):
        self.monitoring_interval = monitoring_interval_seconds
        self.logger = self._setup_logging()

        # Database setup
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/performance_monitor.db")
        self._init_database()

        # Performance tracking
        self.metrics_buffer: Dict[str, deque] = {}  # model_id -> metrics deque
        self.active_alerts: Dict[str, List[PerformanceAlert]] = {}  # model_id -> alerts
        self.model_health: Dict[str, ModelHealthStatus] = {}  # model_id -> health status

        # Monitoring configuration
        self.performance_thresholds = self._initialize_thresholds()
        self.alert_callbacks: List[Callable] = []

        # Threading
        self.is_monitoring = False
        self.monitor_thread = None

        # Drift detection
        self.baseline_distributions: Dict[str, Dict[str, Any]] = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance monitor"""
        logger = logging.getLogger('PerformanceMonitor')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/performance_monitor.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for performance monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    r2 REAL,
                    direction_accuracy REAL,
                    gary_dpi REAL,
                    taleb_antifragility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    prediction_latency REAL,
                    confidence_score REAL,
                    feature_importance_stability REAL,
                    prediction_variance REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    throughput REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    current_value REAL,
                    threshold_value REAL,
                    trend_direction TEXT,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    suggested_actions_json TEXT,
                    severity_score REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_health_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    health_score REAL,
                    status TEXT NOT NULL,
                    active_alerts_count INTEGER,
                    last_prediction_time TEXT,
                    prediction_frequency REAL,
                    error_rate REAL,
                    drift_score REAL,
                    performance_summary_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_thresholds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    warning_threshold REAL,
                    critical_threshold REAL,
                    trend_window_minutes INTEGER,
                    consecutive_violations INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    drift_score REAL,
                    drift_type TEXT,
                    feature_drifts_json TEXT,
                    target_drift REAL,
                    statistical_test TEXT,
                    p_value REAL,
                    is_significant BOOLEAN
                )
            ''')

    def _initialize_thresholds(self) -> Dict[str, PerformanceThreshold]:
        """Initialize default performance thresholds"""
        thresholds = {
            'gary_dpi': PerformanceThreshold(
                metric_name='gary_dpi',
                warning_threshold=-0.05,
                critical_threshold=-0.15,
                trend_window_minutes=60,
                consecutive_violations=3
            ),
            'taleb_antifragility': PerformanceThreshold(
                metric_name='taleb_antifragility',
                warning_threshold=-0.1,
                critical_threshold=-0.25,
                trend_window_minutes=60,
                consecutive_violations=3
            ),
            'direction_accuracy': PerformanceThreshold(
                metric_name='direction_accuracy',
                warning_threshold=0.45,
                critical_threshold=0.35,
                trend_window_minutes=30,
                consecutive_violations=2
            ),
            'r2': PerformanceThreshold(
                metric_name='r2',
                warning_threshold=0.3,
                critical_threshold=0.1,
                trend_window_minutes=60,
                consecutive_violations=3
            ),
            'prediction_latency': PerformanceThreshold(
                metric_name='prediction_latency',
                warning_threshold=100.0,  # milliseconds
                critical_threshold=500.0,
                trend_window_minutes=15,
                consecutive_violations=5
            ),
            'error_rate': PerformanceThreshold(
                metric_name='error_rate',
                warning_threshold=0.05,
                critical_threshold=0.15,
                trend_window_minutes=30,
                consecutive_violations=3
            )
        }

        # Save to database
        self._save_thresholds_to_db(thresholds)

        return thresholds

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        self.logger.info("Starting performance monitoring")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update model health status
                self._update_model_health()

                # Check for threshold violations
                self._check_threshold_violations()

                # Detect performance drift
                self._detect_performance_drift()

                # Clean up old data
                self._cleanup_old_data()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def record_performance(self, model_id: str, metrics: PerformanceMetrics) -> bool:
        """
        Record performance metrics for a model

        Args:
            model_id: Model identifier
            metrics: Performance metrics to record

        Returns:
            success: Whether recording was successful
        """
        try:
            # Initialize model buffer if needed
            if model_id not in self.metrics_buffer:
                self.metrics_buffer[model_id] = deque(maxlen=1000)

            # Add to buffer
            self.metrics_buffer[model_id].append(metrics)

            # Save to database
            self._save_metrics_to_db(metrics)

            # Update baseline distributions if needed
            self._update_baseline_distributions(model_id, metrics)

            # Immediate threshold check for critical metrics
            self._immediate_threshold_check(model_id, metrics)

            return True

        except Exception as e:
            self.logger.error(f"Error recording performance for {model_id}: {e}")
            return False

    def _save_metrics_to_db(self, metrics: PerformanceMetrics):
        """Save performance metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_metrics (
                    model_id, timestamp, mae, mse, rmse, r2, direction_accuracy,
                    gary_dpi, taleb_antifragility, sharpe_ratio, max_drawdown,
                    win_rate, profit_factor, prediction_latency, confidence_score,
                    feature_importance_stability, prediction_variance,
                    memory_usage, cpu_usage, throughput
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.model_id,
                metrics.timestamp.isoformat(),
                metrics.mae,
                metrics.mse,
                metrics.rmse,
                metrics.r2,
                metrics.direction_accuracy,
                metrics.gary_dpi,
                metrics.taleb_antifragility,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.win_rate,
                metrics.profit_factor,
                metrics.prediction_latency,
                metrics.confidence_score,
                metrics.feature_importance_stability,
                metrics.prediction_variance,
                metrics.memory_usage,
                metrics.cpu_usage,
                metrics.throughput
            ))

    def _immediate_threshold_check(self, model_id: str, metrics: PerformanceMetrics):
        """Immediate check for critical threshold violations"""
        critical_metrics = {
            'gary_dpi': metrics.gary_dpi,
            'direction_accuracy': metrics.direction_accuracy,
            'error_rate': 1.0 - metrics.direction_accuracy,  # Convert to error rate
            'prediction_latency': metrics.prediction_latency
        }

        for metric_name, value in critical_metrics.items():
            if metric_name in self.performance_thresholds:
                threshold = self.performance_thresholds[metric_name]

                # Check critical threshold
                if self._is_critical_violation(metric_name, value, threshold):
                    alert = self._create_alert(
                        model_id, metric_name, 'critical', value,
                        threshold.critical_threshold, 'immediate'
                    )
                    self._trigger_alert(alert)

    def _is_critical_violation(self, metric_name: str, value: float, threshold: PerformanceThreshold) -> bool:
        """Check if value violates critical threshold"""
        if metric_name in ['gary_dpi', 'taleb_antifragility', 'direction_accuracy', 'r2']:
            # Lower is worse
            return value < threshold.critical_threshold
        else:
            # Higher is worse (latency, error rate)
            return value > threshold.critical_threshold

    def _update_model_health(self):
        """Update health status for all monitored models"""
        for model_id in self.metrics_buffer.keys():
            try:
                health_status = self._calculate_model_health(model_id)
                self.model_health[model_id] = health_status
                self._save_health_status_to_db(health_status)

            except Exception as e:
                self.logger.error(f"Error updating health for model {model_id}: {e}")

    def _calculate_model_health(self, model_id: str) -> ModelHealthStatus:
        """Calculate overall health status for a model"""
        try:
            if model_id not in self.metrics_buffer or len(self.metrics_buffer[model_id]) == 0:
                return ModelHealthStatus(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    health_score=0.0,
                    status='offline',
                    active_alerts=[],
                    performance_summary={},
                    last_prediction_time=datetime.now(),
                    prediction_frequency=0.0,
                    error_rate=1.0,
                    drift_score=0.0
                )

            recent_metrics = list(self.metrics_buffer[model_id])[-10:]  # Last 10 observations
            latest_metrics = recent_metrics[-1]

            # Calculate health score (0-100)
            health_components = {
                'gary_dpi': self._score_metric(latest_metrics.gary_dpi, 'gary_dpi'),
                'direction_accuracy': self._score_metric(latest_metrics.direction_accuracy, 'direction_accuracy'),
                'r2': self._score_metric(latest_metrics.r2, 'r2'),
                'latency': self._score_metric(latest_metrics.prediction_latency, 'prediction_latency'),
                'stability': latest_metrics.feature_importance_stability * 100
            }

            health_score = np.mean(list(health_components.values()))

            # Determine status
            if health_score >= 80:
                status = 'healthy'
            elif health_score >= 60:
                status = 'warning'
            else:
                status = 'critical'

            # Get active alerts
            active_alerts = self.active_alerts.get(model_id, [])

            # Calculate prediction frequency
            if len(recent_metrics) >= 2:
                time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60
                prediction_frequency = len(recent_metrics) / time_span if time_span > 0 else 0
            else:
                prediction_frequency = 0

            # Calculate error rate
            error_rate = 1.0 - latest_metrics.direction_accuracy

            # Calculate drift score
            drift_score = self._calculate_drift_score(model_id)

            # Performance summary
            performance_summary = {
                'gary_dpi': latest_metrics.gary_dpi,
                'taleb_antifragility': latest_metrics.taleb_antifragility,
                'direction_accuracy': latest_metrics.direction_accuracy,
                'r2': latest_metrics.r2,
                'latency_ms': latest_metrics.prediction_latency
            }

            return ModelHealthStatus(
                model_id=model_id,
                timestamp=datetime.now(),
                health_score=health_score,
                status=status,
                active_alerts=active_alerts,
                performance_summary=performance_summary,
                last_prediction_time=latest_metrics.timestamp,
                prediction_frequency=prediction_frequency,
                error_rate=error_rate,
                drift_score=drift_score
            )

        except Exception as e:
            self.logger.error(f"Error calculating health for model {model_id}: {e}")
            return ModelHealthStatus(
                model_id=model_id,
                timestamp=datetime.now(),
                health_score=0.0,
                status='error',
                active_alerts=[],
                performance_summary={},
                last_prediction_time=datetime.now(),
                prediction_frequency=0.0,
                error_rate=1.0,
                drift_score=0.0
            )

    def _score_metric(self, value: float, metric_name: str) -> float:
        """Score a metric value (0-100)"""
        if metric_name not in self.performance_thresholds:
            return 50.0  # Neutral score

        threshold = self.performance_thresholds[metric_name]

        if metric_name in ['gary_dpi', 'taleb_antifragility', 'direction_accuracy', 'r2']:
            # Higher is better
            if value >= 0.8:  # Excellent
                return 100.0
            elif value >= threshold.warning_threshold:
                return 80.0
            elif value >= threshold.critical_threshold:
                return 40.0
            else:
                return 10.0
        else:
            # Lower is better (latency, error rate)
            if value <= threshold.warning_threshold * 0.5:
                return 100.0
            elif value <= threshold.warning_threshold:
                return 80.0
            elif value <= threshold.critical_threshold:
                return 40.0
            else:
                return 10.0

    def _check_threshold_violations(self):
        """Check for threshold violations across all models"""
        for model_id in self.metrics_buffer.keys():
            try:
                self._check_model_thresholds(model_id)
            except Exception as e:
                self.logger.error(f"Error checking thresholds for model {model_id}: {e}")

    def _check_model_thresholds(self, model_id: str):
        """Check threshold violations for a specific model"""
        if len(self.metrics_buffer[model_id]) < 3:
            return  # Need minimum data

        recent_metrics = list(self.metrics_buffer[model_id])[-10:]

        for threshold_name, threshold in self.performance_thresholds.items():
            # Get metric values
            metric_values = self._extract_metric_values(recent_metrics, threshold_name)

            if len(metric_values) < threshold.consecutive_violations:
                continue

            # Check for consecutive violations
            recent_values = metric_values[-threshold.consecutive_violations:]

            warning_violations = sum(
                1 for v in recent_values
                if self._is_warning_violation(threshold_name, v, threshold)
            )

            critical_violations = sum(
                1 for v in recent_values
                if self._is_critical_violation(threshold_name, v, threshold)
            )

            # Generate alerts
            if critical_violations >= threshold.consecutive_violations:
                alert = self._create_alert(
                    model_id, threshold_name, 'critical',
                    recent_values[-1], threshold.critical_threshold,
                    self._calculate_trend(recent_values)
                )
                self._trigger_alert(alert)

            elif warning_violations >= threshold.consecutive_violations:
                alert = self._create_alert(
                    model_id, threshold_name, 'warning',
                    recent_values[-1], threshold.warning_threshold,
                    self._calculate_trend(recent_values)
                )
                self._trigger_alert(alert)

    def _extract_metric_values(self, metrics_list: List[PerformanceMetrics], metric_name: str) -> List[float]:
        """Extract specific metric values from metrics list"""
        values = []

        for metrics in metrics_list:
            if metric_name == 'gary_dpi':
                values.append(metrics.gary_dpi)
            elif metric_name == 'taleb_antifragility':
                values.append(metrics.taleb_antifragility)
            elif metric_name == 'direction_accuracy':
                values.append(metrics.direction_accuracy)
            elif metric_name == 'r2':
                values.append(metrics.r2)
            elif metric_name == 'prediction_latency':
                values.append(metrics.prediction_latency)
            elif metric_name == 'error_rate':
                values.append(1.0 - metrics.direction_accuracy)

        return values

    def _is_warning_violation(self, metric_name: str, value: float, threshold: PerformanceThreshold) -> bool:
        """Check if value violates warning threshold"""
        if metric_name in ['gary_dpi', 'taleb_antifragility', 'direction_accuracy', 'r2']:
            return value < threshold.warning_threshold
        else:
            return value > threshold.warning_threshold

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values"""
        if len(values) < 2:
            return 'stable'

        # Simple trend calculation
        slope = (values[-1] - values[0]) / len(values)

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _create_alert(self, model_id: str, metric_name: str, alert_type: str,
                     current_value: float, threshold_value: float, trend: str) -> PerformanceAlert:
        """Create performance alert"""
        alert_id = f"{model_id}_{metric_name}_{alert_type}_{int(time.time())}"

        # Generate description and suggestions
        description = f"Model {model_id} {metric_name} {alert_type}: {current_value:.4f} (threshold: {threshold_value:.4f})"

        suggested_actions = self._get_suggested_actions(metric_name, alert_type)

        # Calculate severity score
        severity_score = self._calculate_severity_score(metric_name, current_value, threshold_value, alert_type)

        return PerformanceAlert(
            alert_id=alert_id,
            model_id=model_id,
            metric_name=metric_name,
            alert_type=alert_type,
            current_value=current_value,
            threshold_value=threshold_value,
            trend_direction=trend,
            timestamp=datetime.now(),
            description=description,
            suggested_actions=suggested_actions,
            severity_score=severity_score
        )

    def _get_suggested_actions(self, metric_name: str, alert_type: str) -> List[str]:
        """Get suggested actions for alert"""
        actions = {
            'gary_dpi': [
                "Review position sizing strategy",
                "Analyze recent market conditions",
                "Check feature importance changes",
                "Consider model retraining"
            ],
            'direction_accuracy': [
                "Immediate model retraining recommended",
                "Check for regime change",
                "Review feature engineering",
                "Validate prediction confidence"
            ],
            'prediction_latency': [
                "Check system resources",
                "Optimize model inference",
                "Review feature computation",
                "Consider model simplification"
            ],
            'r2': [
                "Model performance degraded",
                "Check for overfitting",
                "Review training data quality",
                "Consider ensemble methods"
            ]
        }

        base_actions = actions.get(metric_name, ["Review model configuration"])

        if alert_type == 'critical':
            base_actions.insert(0, "IMMEDIATE ACTION REQUIRED")

        return base_actions

    def _calculate_severity_score(self, metric_name: str, current_value: float,
                                threshold_value: float, alert_type: str) -> float:
        """Calculate severity score (0-100)"""
        # Base severity by alert type
        base_severity = {'warning': 50, 'critical': 80}.get(alert_type, 30)

        # Adjust based on how far from threshold
        if metric_name in ['gary_dpi', 'taleb_antifragility', 'direction_accuracy', 'r2']:
            # Lower is worse
            distance = abs(current_value - threshold_value) / abs(threshold_value)
        else:
            # Higher is worse
            distance = abs(current_value - threshold_value) / threshold_value

        severity_adjustment = min(20, distance * 20)

        return min(100, base_severity + severity_adjustment)

    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger performance alert"""
        try:
            # Add to active alerts
            if alert.model_id not in self.active_alerts:
                self.active_alerts[alert.model_id] = []

            # Check if similar alert already exists
            existing_alert = next(
                (a for a in self.active_alerts[alert.model_id]
                 if a.metric_name == alert.metric_name and a.alert_type == alert.alert_type),
                None
            )

            if not existing_alert:
                self.active_alerts[alert.model_id].append(alert)

                # Save to database
                self._save_alert_to_db(alert)

                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")

                # Log alert
                self.logger.warning(f"ALERT: {alert.description}")

        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")

    def _detect_performance_drift(self):
        """Detect performance drift using statistical tests"""
        for model_id in self.metrics_buffer.keys():
            try:
                if model_id not in self.baseline_distributions:
                    continue

                drift_score = self._calculate_drift_score(model_id)

                if drift_score > 0.1:  # Significant drift threshold
                    self._save_drift_detection(model_id, drift_score)

                    # Generate drift alert
                    alert = PerformanceAlert(
                        alert_id=f"{model_id}_drift_{int(time.time())}",
                        model_id=model_id,
                        metric_name="performance_drift",
                        alert_type="warning" if drift_score < 0.2 else "critical",
                        current_value=drift_score,
                        threshold_value=0.1,
                        trend_direction="increasing",
                        timestamp=datetime.now(),
                        description=f"Performance drift detected for model {model_id}: {drift_score:.4f}",
                        suggested_actions=[
                            "Investigate data distribution changes",
                            "Consider model retraining",
                            "Review feature engineering",
                            "Check for regime changes"
                        ],
                        severity_score=drift_score * 100
                    )

                    self._trigger_alert(alert)

            except Exception as e:
                self.logger.error(f"Error detecting drift for model {model_id}: {e}")

    def _calculate_drift_score(self, model_id: str) -> float:
        """Calculate performance drift score"""
        try:
            if (model_id not in self.baseline_distributions or
                model_id not in self.metrics_buffer or
                len(self.metrics_buffer[model_id]) < 10):
                return 0.0

            # Get recent performance
            recent_metrics = list(self.metrics_buffer[model_id])[-10:]
            recent_gary_dpi = [m.gary_dpi for m in recent_metrics]

            # Compare with baseline
            baseline_mean = self.baseline_distributions[model_id]['gary_dpi_mean']
            baseline_std = self.baseline_distributions[model_id]['gary_dpi_std']

            recent_mean = np.mean(recent_gary_dpi)

            # Normalized drift score
            drift_score = abs(recent_mean - baseline_mean) / (baseline_std + 1e-8)

            return min(1.0, drift_score)

        except Exception as e:
            self.logger.error(f"Error calculating drift score: {e}")
            return 0.0

    def _update_baseline_distributions(self, model_id: str, metrics: PerformanceMetrics):
        """Update baseline distributions for drift detection"""
        try:
            if model_id not in self.baseline_distributions:
                # Initialize baseline with first 50 observations
                if len(self.metrics_buffer[model_id]) >= 50:
                    baseline_metrics = list(self.metrics_buffer[model_id])[:50]

                    self.baseline_distributions[model_id] = {
                        'gary_dpi_mean': np.mean([m.gary_dpi for m in baseline_metrics]),
                        'gary_dpi_std': np.std([m.gary_dpi for m in baseline_metrics]),
                        'direction_accuracy_mean': np.mean([m.direction_accuracy for m in baseline_metrics]),
                        'direction_accuracy_std': np.std([m.direction_accuracy for m in baseline_metrics]),
                        'created_at': datetime.now()
                    }

        except Exception as e:
            self.logger.error(f"Error updating baseline distributions: {e}")

    def _save_health_status_to_db(self, health_status: ModelHealthStatus):
        """Save model health status to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_health_status (
                    model_id, timestamp, health_score, status, active_alerts_count,
                    last_prediction_time, prediction_frequency, error_rate, drift_score,
                    performance_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health_status.model_id,
                health_status.timestamp.isoformat(),
                health_status.health_score,
                health_status.status,
                len(health_status.active_alerts),
                health_status.last_prediction_time.isoformat(),
                health_status.prediction_frequency,
                health_status.error_rate,
                health_status.drift_score,
                json.dumps(health_status.performance_summary)
            ))

    def _save_alert_to_db(self, alert: PerformanceAlert):
        """Save alert to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_alerts (
                    alert_id, model_id, metric_name, alert_type, current_value,
                    threshold_value, trend_direction, timestamp, description,
                    suggested_actions_json, severity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.model_id,
                alert.metric_name,
                alert.alert_type,
                alert.current_value,
                alert.threshold_value,
                alert.trend_direction,
                alert.timestamp.isoformat(),
                alert.description,
                json.dumps(alert.suggested_actions),
                alert.severity_score
            ))

    def _save_drift_detection(self, model_id: str, drift_score: float):
        """Save drift detection result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO drift_detection (
                    model_id, timestamp, drift_score, drift_type,
                    statistical_test, is_significant
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                datetime.now().isoformat(),
                drift_score,
                'performance_drift',
                'statistical_comparison',
                drift_score > 0.1
            ))

    def _save_thresholds_to_db(self, thresholds: Dict[str, PerformanceThreshold]):
        """Save thresholds to database"""
        with sqlite3.connect(self.db_path) as conn:
            for threshold in thresholds.values():
                conn.execute('''
                    INSERT OR REPLACE INTO performance_thresholds (
                        metric_name, warning_threshold, critical_threshold,
                        trend_window_minutes, consecutive_violations
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    threshold.metric_name,
                    threshold.warning_threshold,
                    threshold.critical_threshold,
                    threshold.trend_window_minutes,
                    threshold.consecutive_violations
                ))

    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)

            with sqlite3.connect(self.db_path) as conn:
                # Clean old metrics (keep last 7 days)
                conn.execute('''
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                ''', (cutoff_time.isoformat(),))

                # Clean resolved alerts (keep last 30 days)
                alert_cutoff = datetime.now() - timedelta(days=30)
                conn.execute('''
                    DELETE FROM performance_alerts
                    WHERE resolved = TRUE AND resolved_at < ?
                ''', (alert_cutoff.isoformat(),))

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)

    def get_model_health(self, model_id: str) -> Optional[ModelHealthStatus]:
        """Get current health status for a model"""
        return self.model_health.get(model_id)

    def get_active_alerts(self, model_id: Optional[str] = None) -> List[PerformanceAlert]:
        """Get active alerts"""
        if model_id:
            return self.active_alerts.get(model_id, [])
        else:
            all_alerts = []
            for alerts in self.active_alerts.values():
                all_alerts.extend(alerts)
            return all_alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        try:
            # Remove from active alerts
            for model_id, alerts in self.active_alerts.items():
                self.active_alerts[model_id] = [a for a in alerts if a.alert_id != alert_id]

            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE performance_alerts
                    SET resolved = TRUE, resolved_at = ?
                    WHERE alert_id = ?
                ''', (datetime.now().isoformat(), alert_id))

            return True

        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'system_status': {
                    'is_monitoring': self.is_monitoring,
                    'monitored_models': len(self.metrics_buffer),
                    'total_alerts': sum(len(alerts) for alerts in self.active_alerts.values()),
                    'critical_alerts': len([
                        alert for alerts in self.active_alerts.values()
                        for alert in alerts if alert.alert_type == 'critical'
                    ])
                }
            }

            # Add model data
            for model_id in self.metrics_buffer.keys():
                health_status = self.model_health.get(model_id)
                recent_metrics = list(self.metrics_buffer[model_id])[-1] if self.metrics_buffer[model_id] else None

                dashboard_data['models'][model_id] = {
                    'health_score': health_status.health_score if health_status else 0,
                    'status': health_status.status if health_status else 'unknown',
                    'active_alerts': len(self.active_alerts.get(model_id, [])),
                    'last_update': recent_metrics.timestamp.isoformat() if recent_metrics else None,
                    'performance_summary': health_status.performance_summary if health_status else {}
                }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor(monitoring_interval_seconds=30)

    # Add alert callback
    def alert_handler(alert: PerformanceAlert):
        print(f"ALERT: {alert.description}")
        print(f"Suggestions: {', '.join(alert.suggested_actions)}")

    monitor.add_alert_callback(alert_handler)

    # Start monitoring
    monitor.start_monitoring()

    print("Performance monitoring started...")
    print("Use monitor.record_performance() to record model performance")
    print("Use monitor.get_model_health() to get model health status")
    print("Use monitor.get_monitoring_dashboard_data() to get dashboard data")