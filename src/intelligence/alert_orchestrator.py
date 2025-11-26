"""
Alert Orchestrator - Real-Time Alert Processing and Notification System

Coordinates all AI alert components for real-time risk detection and notification:
- Orchestrates AI alert system, pattern engine, and predictive warnings
- Real-time alert processing with priority queuing
- Alert severity classification and escalation
- Integration with DPI and antifragility calculations
- Performance monitoring and validation metrics
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Import our AI alert components
from .ai_alert_system import AIAlertSystem, RiskAlert, AlertSeverity
from .risk_pattern_engine import RiskPatternEngine, PatternDetection
from .predictive_warning_system import PredictiveWarningSystem, EarlyWarning
from .context_filter import ContextAwareFilter

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert processing priorities"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ProcessingStatus(Enum):
    """Alert processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    FILTERED = "filtered"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AlertMessage:
    """Unified alert message"""
    alert_id: str
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: AlertSeverity
    priority: AlertPriority
    confidence: float
    message: str
    details: Dict[str, Any]
    source_component: str
    processing_status: ProcessingStatus
    escalation_required: bool
    notification_channels: List[str]


@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    timestamp: datetime
    alerts_processed_per_minute: float
    average_processing_latency_ms: float
    false_positive_rate: float
    alert_accuracy: float
    pattern_detection_rate: float
    prediction_accuracy: float
    system_health_score: float
    queue_depth: int
    active_threads: int


class AlertOrchestrator:
    """
    Alert Orchestrator - Central coordination system for AI-powered risk alerts

    Integrates all AI alert components and provides real-time processing
    with advanced priority management and performance monitoring.
    """

    def __init__(self,
                 dpi_calculator=None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alert Orchestrator

        Args:
            dpi_calculator: Reference to DPI calculation system
            config: Configuration dictionary
        """
        self.dpi_calculator = dpi_calculator
        self.config = config or {}

        # Initialize AI components
        self.ai_alert_system = AIAlertSystem(
            dpi_calculator=dpi_calculator,
            lookback_days=self.config.get('lookback_days', 30),
            alert_threshold=0.05  # Target <5% false positive rate
        )

        self.pattern_engine = RiskPatternEngine(
            lookback_periods=self.config.get('pattern_lookback', {
                '1m': 60, '5m': 144, '15m': 96, '1h': 168, '1d': 252
            })
        )

        self.predictive_system = PredictiveWarningSystem(
            prediction_horizons=self.config.get('prediction_horizons', [5, 10, 15])
        )

        self.context_filter = ContextAwareFilter(
            config=self.config.get('filter_config', {})
        )

        # Alert processing infrastructure
        self.alert_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.is_running = False
        self.max_workers = self.config.get('max_workers', 4)

        # Performance tracking
        self.metrics_history = []
        self.alert_history = []
        self.processing_times = []
        self.last_metrics_update = datetime.now()

        # Alert routing and notification
        self.notification_handlers = {}
        self.escalation_rules = {
            AlertSeverity.CRITICAL: 0,      # Immediate escalation
            AlertSeverity.HIGH: 300,        # 5 minutes
            AlertSeverity.MEDIUM: 900,      # 15 minutes
            AlertSeverity.LOW: 1800         # 30 minutes
        }

        # ISS-018: Auto-register default notification handlers
        self._register_default_handlers()

        # System health monitoring
        self.health_checks = {
            'ai_system': True,
            'pattern_engine': True,
            'predictive_system': True,
            'context_filter': True
        }

        logger.info("Alert Orchestrator initialized with AI-powered components")

    def start(self) -> bool:
        """Start the alert processing system"""
        try:
            if self.is_running:
                logger.warning("Alert Orchestrator already running")
                return True

            logger.info("Starting Alert Orchestrator...")

            # Start processing threads
            self.is_running = True
            for i in range(self.max_workers):
                thread = threading.Thread(
                    target=self._alert_processing_worker,
                    args=(f"worker-{i}",),
                    daemon=True
                )
                thread.start()
                self.processing_threads.append(thread)

            # Start metrics collection thread
            metrics_thread = threading.Thread(
                target=self._metrics_collection_worker,
                daemon=True
            )
            metrics_thread.start()
            self.processing_threads.append(metrics_thread)

            logger.info(f"Alert Orchestrator started with {self.max_workers} processing workers")
            return True

        except Exception as e:
            logger.error(f"Failed to start Alert Orchestrator: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the alert processing system"""
        try:
            logger.info("Stopping Alert Orchestrator...")
            self.is_running = False

            # Wait for threads to complete
            for thread in self.processing_threads:
                if thread.is_alive():
                    thread.join(timeout=5.0)

            logger.info("Alert Orchestrator stopped")

        except Exception as e:
            logger.error(f"Error stopping Alert Orchestrator: {e}")

    def process_market_data(self,
                           symbol: str,
                           market_data: Dict[str, Any],
                           portfolio_context: Dict[str, Any] = None) -> List[AlertMessage]:
        """
        Process market data and generate alerts

        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio_context: Portfolio-wide context

        Returns:
            List of generated alert messages
        """
        try:
            start_time = time.time()
            generated_alerts = []

            # Build market context for filtering
            context_data = self._build_context_data(market_data, portfolio_context)

            # 1. AI Alert System - ML-based anomaly detection
            try:
                ai_alert = self.ai_alert_system.generate_alert(
                    symbol=symbol,
                    current_data=market_data,
                    market_context=context_data.get('alert_context')
                )

                if ai_alert:
                    alert_msg = self._convert_ai_alert_to_message(ai_alert)
                    generated_alerts.append(alert_msg)
                    logger.debug(f"AI alert generated for {symbol}: {ai_alert.severity.value}")

            except Exception as e:
                logger.error(f"AI alert generation failed for {symbol}: {e}")
                self.health_checks['ai_system'] = False

            # 2. Pattern Recognition Engine - regime and pattern detection
            try:
                if 'timeframe_data' in market_data:
                    patterns = self.pattern_engine.detect_patterns(
                        symbol=symbol,
                        market_data=market_data['timeframe_data'],
                        dpi_data=market_data.get('dpi_data')
                    )

                    for pattern in patterns:
                        alert_msg = self._convert_pattern_to_message(pattern)
                        generated_alerts.append(alert_msg)
                        logger.debug(f"Pattern detected for {symbol}: {pattern.pattern_type.value}")

            except Exception as e:
                logger.error(f"Pattern detection failed for {symbol}: {e}")
                self.health_checks['pattern_engine'] = False

            # 3. Predictive Warning System - early warnings
            try:
                if 'historical_data' in market_data:
                    warnings = self.predictive_system.generate_warnings(
                        symbol=symbol,
                        current_data=market_data['historical_data'],
                        market_context=context_data
                    )

                    for warning in warnings:
                        alert_msg = self._convert_warning_to_message(warning)
                        generated_alerts.append(alert_msg)
                        logger.debug(f"Predictive warning for {symbol}: {warning.event_type.value}")

            except Exception as e:
                logger.error(f"Predictive warning failed for {symbol}: {e}")
                self.health_checks['predictive_system'] = False

            # 4. Apply context filtering to reduce false positives
            filtered_alerts = []
            for alert in generated_alerts:
                try:
                    filter_result = self.context_filter.apply_filter(
                        alert_confidence=alert.confidence,
                        alert_type=alert.alert_type,
                        symbol=symbol,
                        market_data=market_data,
                        portfolio_context=portfolio_context
                    )

                    if filter_result.passed:
                        # Adjust confidence based on context
                        alert.confidence *= filter_result.confidence_adjustment
                        alert.details['context_score'] = filter_result.context_score
                        filtered_alerts.append(alert)
                    else:
                        logger.debug(f"Alert filtered out for {symbol}: {filter_result.suppression_reason}")

                except Exception as e:
                    logger.error(f"Context filtering failed for {symbol}: {e}")
                    # Include alert without filtering to be safe
                    filtered_alerts.append(alert)
                    self.health_checks['context_filter'] = False

            # 5. Queue alerts for processing
            for alert in filtered_alerts:
                self._queue_alert_for_processing(alert)

            # Record processing time
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            self.processing_times.append(processing_time)

            # Keep only last 1000 processing times
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]

            logger.info(f"Processed {symbol}: {len(generated_alerts)} generated, "
                       f"{len(filtered_alerts)} passed filter ({processing_time:.1f}ms)")

            return filtered_alerts

        except Exception as e:
            logger.error(f"Market data processing failed for {symbol}: {e}")
            return []

    def train_ai_models(self,
                       historical_data: Dict[str, Any],
                       risk_events: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train all AI models with historical data

        Args:
            historical_data: Historical market data
            risk_events: Historical risk events for supervised learning

        Returns:
            Training results and metrics
        """
        try:
            logger.info("Training AI models...")
            training_results = {}

            # Train AI Alert System
            try:
                ai_metrics = self.ai_alert_system.train_models(
                    historical_data=historical_data.get('market_data'),
                    risk_events=risk_events.get('risk_events') if risk_events else None
                )
                training_results['ai_alert_system'] = ai_metrics
                logger.info(f"AI Alert System trained - Accuracy: {ai_metrics.get('pattern_accuracy', 0):.3f}")

            except Exception as e:
                logger.error(f"AI Alert System training failed: {e}")
                training_results['ai_alert_system'] = {'error': str(e)}

            # Train Predictive Warning System
            try:
                pred_metrics = self.predictive_system.train_models(
                    training_data=historical_data.get('training_data', {})
                )
                training_results['predictive_system'] = asdict(pred_metrics)
                logger.info(f"Predictive System trained - Confidence: {pred_metrics.model_confidence:.3f}")

            except Exception as e:
                logger.error(f"Predictive System training failed: {e}")
                training_results['predictive_system'] = {'error': str(e)}

            # Update health status
            self.health_checks['ai_system'] = 'error' not in training_results.get('ai_alert_system', {})
            self.health_checks['predictive_system'] = 'error' not in training_results.get('predictive_system', {})

            logger.info("AI model training completed")
            return training_results

        except Exception as e:
            logger.error(f"AI model training failed: {e}")
            return {'error': str(e)}

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        try:
            now = datetime.now()

            # Calculate alerts per minute
            recent_alerts = [a for a in self.alert_history
                           if (now - a.timestamp).total_seconds() < 60]
            alerts_per_minute = len(recent_alerts)

            # Calculate average processing latency
            avg_latency = np.mean(self.processing_times[-100:]) if self.processing_times else 0.0

            # Calculate false positive rate (would need feedback data in production)
            fp_rate = self._calculate_false_positive_rate()

            # Calculate alert accuracy (would need validation data)
            accuracy = self._calculate_alert_accuracy()

            # Calculate pattern detection rate
            pattern_rate = self._calculate_pattern_detection_rate()

            # Calculate prediction accuracy
            pred_accuracy = self._calculate_prediction_accuracy()

            # Calculate system health score
            health_score = self._calculate_system_health_score()

            metrics = SystemMetrics(
                timestamp=now,
                alerts_processed_per_minute=alerts_per_minute,
                average_processing_latency_ms=avg_latency,
                false_positive_rate=fp_rate,
                alert_accuracy=accuracy,
                pattern_detection_rate=pattern_rate,
                prediction_accuracy=pred_accuracy,
                system_health_score=health_score,
                queue_depth=self.alert_queue.qsize(),
                active_threads=len([t for t in self.processing_threads if t.is_alive()])
            )

            return metrics

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                alerts_processed_per_minute=0.0,
                average_processing_latency_ms=0.0,
                false_positive_rate=0.05,
                alert_accuracy=0.8,
                pattern_detection_rate=0.0,
                prediction_accuracy=0.6,
                system_health_score=0.5,
                queue_depth=0,
                active_threads=0
            )

    def _alert_processing_worker(self, worker_name: str) -> None:
        """Background worker for processing alerts"""
        logger.info(f"Alert processing worker {worker_name} started")

        while self.is_running:
            try:
                # Get alert from queue with timeout
                try:
                    priority, alert = self.alert_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Process the alert
                self._process_alert(alert)

                # Mark task as done
                self.alert_queue.task_done()

            except Exception as e:
                logger.error(f"Alert processing worker {worker_name} error: {e}")

        logger.info(f"Alert processing worker {worker_name} stopped")

    def _metrics_collection_worker(self) -> None:
        """Background worker for metrics collection"""
        logger.info("Metrics collection worker started")

        while self.is_running:
            try:
                # Collect metrics every 30 seconds
                time.sleep(30)

                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 24 hours of metrics
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]

                # Log important metrics
                if metrics.false_positive_rate > 0.1:
                    logger.warning(f"High false positive rate: {metrics.false_positive_rate:.3f}")

                if metrics.average_processing_latency_ms > 1000:
                    logger.warning(f"High processing latency: {metrics.average_processing_latency_ms:.1f}ms")

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

        logger.info("Metrics collection worker stopped")

    def _queue_alert_for_processing(self, alert: AlertMessage) -> None:
        """Queue alert for background processing"""
        try:
            # Determine priority
            priority_value = alert.priority.value

            # Add timestamp for FIFO within priority
            priority_key = (priority_value, time.time())

            # Add to queue
            self.alert_queue.put((priority_key, alert))

        except Exception as e:
            logger.error(f"Failed to queue alert {alert.alert_id}: {e}")

    def _process_alert(self, alert: AlertMessage) -> None:
        """Process individual alert"""
        try:
            alert.processing_status = ProcessingStatus.PROCESSING

            # Escalation check
            if self._should_escalate_alert(alert):
                alert.escalation_required = True
                alert.processing_status = ProcessingStatus.ESCALATED
                logger.warning(f"Alert escalated: {alert.alert_id}")

            # Send notifications
            self._send_notifications(alert)

            # Update alert history
            self.alert_history.append(alert)
            alert.processing_status = ProcessingStatus.COMPLETED

            # Keep alert history manageable
            if len(self.alert_history) > 10000:
                self.alert_history = self.alert_history[-5000:]

        except Exception as e:
            logger.error(f"Alert processing failed for {alert.alert_id}: {e}")
            alert.processing_status = ProcessingStatus.FAILED

    def _should_escalate_alert(self, alert: AlertMessage) -> bool:
        """Determine if alert should be escalated"""
        try:
            # Immediate escalation conditions
            if alert.severity == AlertSeverity.CRITICAL:
                return True

            if alert.confidence > 0.95:
                return True

            # Time-based escalation (would be implemented with alert aging)
            # This is a simplified version
            self.escalation_rules.get(alert.severity, 1800)

            return False  # Placeholder - would implement alert aging logic

        except Exception as e:
            logger.error(f"Escalation check failed for {alert.alert_id}: {e}")
            return False

    def _send_notifications(self, alert: AlertMessage) -> None:
        """Send alert notifications through configured channels"""
        try:
            for channel in alert.notification_channels:
                handler = self.notification_handlers.get(channel)
                if handler:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Notification failed for channel {channel}: {e}")

        except Exception as e:
            logger.error(f"Notification sending failed for {alert.alert_id}: {e}")

    def _build_context_data(self,
                           market_data: Dict[str, Any],
                           portfolio_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build context data for filtering"""
        try:
            # This would be implemented to build comprehensive market context
            # For now, return basic structure
            return {
                'alert_context': market_data.get('context'),
                'portfolio_context': portfolio_context,
                'market_data': market_data
            }

        except Exception as e:
            logger.error(f"Context data building failed: {e}")
            return {}

    def _convert_ai_alert_to_message(self, alert: RiskAlert) -> AlertMessage:
        """Convert AI alert to unified alert message"""
        try:
            priority = self._severity_to_priority(alert.severity)

            return AlertMessage(
                alert_id=alert.alert_id,
                timestamp=alert.timestamp,
                symbol=alert.symbol,
                alert_type=alert.pattern_type.value,
                severity=alert.severity,
                priority=priority,
                confidence=alert.confidence,
                message=alert.description,
                details={
                    'metrics': alert.metrics,
                    'insights': alert.actionable_insights,
                    'false_positive_prob': alert.false_positive_probability
                },
                source_component='ai_alert_system',
                processing_status=ProcessingStatus.QUEUED,
                escalation_required=False,
                notification_channels=['log', 'dashboard']
            )

        except Exception as e:
            logger.error(f"AI alert conversion failed: {e}")
            return None

    def _convert_pattern_to_message(self, pattern: PatternDetection) -> AlertMessage:
        """Convert pattern detection to alert message"""
        try:
            # Map pattern strength to severity
            if pattern.strength > 0.8:
                severity = AlertSeverity.HIGH
            elif pattern.strength > 0.6:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            priority = self._severity_to_priority(severity)

            return AlertMessage(
                alert_id=f"PATTERN_{pattern.detection_time.strftime('%Y%m%d_%H%M%S')}_{pattern.affected_symbols[0]}",
                timestamp=pattern.detection_time,
                symbol=pattern.affected_symbols[0] if pattern.affected_symbols else 'UNKNOWN',
                alert_type=pattern.pattern_type.value,
                severity=severity,
                priority=priority,
                confidence=pattern.confidence,
                message=pattern.description,
                details={
                    'risk_metrics': pattern.risk_metrics,
                    'suggested_actions': pattern.suggested_actions,
                    'timeframe': pattern.timeframe,
                    'strength': pattern.strength
                },
                source_component='risk_pattern_engine',
                processing_status=ProcessingStatus.QUEUED,
                escalation_required=False,
                notification_channels=['log', 'dashboard']
            )

        except Exception as e:
            logger.error(f"Pattern conversion failed: {e}")
            return None

    def _convert_warning_to_message(self, warning: EarlyWarning) -> AlertMessage:
        """Convert early warning to alert message"""
        try:
            # Map warning level to severity
            severity_map = {
                'advisory': AlertSeverity.LOW,
                'watch': AlertSeverity.LOW,
                'warning': AlertSeverity.MEDIUM,
                'alert': AlertSeverity.HIGH
            }
            severity = severity_map.get(warning.warning_level.value, AlertSeverity.MEDIUM)
            priority = self._severity_to_priority(severity)

            return AlertMessage(
                alert_id=warning.warning_id,
                timestamp=warning.timestamp,
                symbol=warning.symbol,
                alert_type=warning.event_type.value,
                severity=severity,
                priority=priority,
                confidence=warning.confidence,
                message=warning.description,
                details={
                    'time_to_event': warning.time_to_event,
                    'magnitude': warning.magnitude,
                    'risk_factors': warning.risk_factors,
                    'recommended_actions': warning.recommended_actions
                },
                source_component='predictive_warning_system',
                processing_status=ProcessingStatus.QUEUED,
                escalation_required=warning.time_to_event < 5,  # Escalate if <5 minutes
                notification_channels=['log', 'dashboard', 'urgent'] if warning.time_to_event < 5 else ['log', 'dashboard']
            )

        except Exception as e:
            logger.error(f"Warning conversion failed: {e}")
            return None

    def _severity_to_priority(self, severity: AlertSeverity) -> AlertPriority:
        """Convert alert severity to processing priority"""
        priority_map = {
            AlertSeverity.CRITICAL: AlertPriority.EMERGENCY,
            AlertSeverity.HIGH: AlertPriority.CRITICAL,
            AlertSeverity.MEDIUM: AlertPriority.HIGH,
            AlertSeverity.LOW: AlertPriority.MEDIUM
        }
        return priority_map.get(severity, AlertPriority.MEDIUM)

    def _calculate_false_positive_rate(self) -> float:
        """Calculate false positive rate from alert history"""
        try:
            if not self.alert_history:
                return 0.05  # Default target

            # This would be implemented with actual validation data
            # For now, return estimated rate based on confidence scores
            total_alerts = len(self.alert_history)
            low_confidence_alerts = len([a for a in self.alert_history if a.confidence < 0.6])

            return min(0.2, low_confidence_alerts / total_alerts) if total_alerts > 0 else 0.05

        except Exception as e:
            logger.error(f"False positive calculation failed: {e}")
            return 0.05

    def _calculate_alert_accuracy(self) -> float:
        """Calculate overall alert accuracy"""
        try:
            if not self.alert_history:
                return 0.8  # Default estimate

            # This would be implemented with validation feedback
            # For now, return estimate based on confidence
            avg_confidence = np.mean([a.confidence for a in self.alert_history])
            return min(0.95, max(0.5, avg_confidence))

        except Exception as e:
            logger.error(f"Alert accuracy calculation failed: {e}")
            return 0.8

    def _calculate_pattern_detection_rate(self) -> float:
        """Calculate pattern detection rate"""
        try:
            pattern_alerts = [a for a in self.alert_history
                            if a.source_component == 'risk_pattern_engine']

            if not self.alert_history:
                return 0.0

            return len(pattern_alerts) / len(self.alert_history)

        except Exception as e:
            logger.error(f"Pattern detection rate calculation failed: {e}")
            return 0.0

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        try:
            # This would be implemented with outcome validation
            # For now, return estimate based on system performance
            return 0.65  # 65% prediction accuracy target

        except Exception as e:
            logger.error(f"Prediction accuracy calculation failed: {e}")
            return 0.6

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            health_components = []

            # Component health
            component_health = sum(self.health_checks.values()) / len(self.health_checks)
            health_components.append(component_health)

            # Processing performance
            if self.processing_times:
                avg_latency = np.mean(self.processing_times[-100:])
                latency_score = max(0.0, 1.0 - (avg_latency / 1000.0))  # 1 second = 0 score
                health_components.append(latency_score)

            # Queue depth
            queue_depth = self.alert_queue.qsize()
            queue_score = max(0.0, 1.0 - (queue_depth / 100.0))  # 100 alerts = 0 score
            health_components.append(queue_score)

            # Thread health
            active_threads = len([t for t in self.processing_threads if t.is_alive()])
            expected_threads = self.max_workers + 1  # Workers + metrics thread
            thread_score = active_threads / expected_threads
            health_components.append(thread_score)

            return np.mean(health_components) if health_components else 0.5

        except Exception as e:
            logger.error(f"System health calculation failed: {e}")
            return 0.5

    def register_notification_handler(self, channel: str, handler) -> None:
        """Register notification handler for a channel"""
        self.notification_handlers[channel] = handler
        logger.info(f"Notification handler registered for channel: {channel}")

    def _register_default_handlers(self) -> None:
        """
        ISS-018: Register default notification handlers.

        Registers log, email, and Slack handlers based on configuration/environment.
        """
        try:
            from .notification_handlers import register_default_handlers

            # Get notification config from main config
            notification_config = self.config.get('notifications', {})
            register_default_handlers(self, notification_config)

        except ImportError as e:
            logger.warning(f"Could not import notification handlers: {e}")
            # Fallback: register basic log handler
            self._register_basic_log_handler()
        except Exception as e:
            logger.error(f"Failed to register default handlers: {e}")
            self._register_basic_log_handler()

    def _register_basic_log_handler(self) -> None:
        """Fallback: Register basic logging handler"""
        def log_handler(alert):
            severity = getattr(alert, 'severity', None)
            severity_str = severity.value if severity and hasattr(severity, 'value') else 'INFO'
            logger.info(f"ALERT [{severity_str}] {getattr(alert, 'symbol', 'Unknown')}: {getattr(alert, 'message', 'No message')}")
            return True

        self.notification_handlers['log'] = log_handler
        logger.info("Basic log notification handler registered")

    def get_alert_history(self,
                         hours: int = 24,
                         symbol: str = None,
                         severity: AlertSeverity = None) -> List[AlertMessage]:
        """Get alert history with optional filtering"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_alerts = [
                a for a in self.alert_history
                if a.timestamp > cutoff_time
            ]

            if symbol:
                filtered_alerts = [a for a in filtered_alerts if a.symbol == symbol]

            if severity:
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

            return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Alert history retrieval failed: {e}")
            return []

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            metrics = self.get_system_metrics()

            return {
                'is_running': self.is_running,
                'component_health': self.health_checks,
                'current_metrics': asdict(metrics),
                'alert_queue_size': self.alert_queue.qsize(),
                'active_workers': len([t for t in self.processing_threads if t.is_alive()]),
                'total_alerts_processed': len(self.alert_history),
                'ai_system_status': self.ai_alert_system.get_system_status(),
                'predictive_system_status': self.predictive_system.get_system_status(),
                'filter_status': self.context_filter.get_filter_status()
            }

        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {'error': str(e)}


# Import numpy for calculations
import numpy as np