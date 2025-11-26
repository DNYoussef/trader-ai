"""
Health Monitoring System - Comprehensive system health monitoring and alerting.

This module implements advanced health monitoring with real-time metrics,
anomaly detection, and intelligent alerting to ensure system reliability
and early problem detection.

Key Features:
- Multi-dimensional health metrics collection
- Real-time anomaly detection
- Intelligent alerting with escalation
- Performance trend analysis
- Resource utilization monitoring
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque, defaultdict
import psutil
import os

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"           # Monotonically increasing (requests, errors)
    GAUGE = "gauge"               # Point-in-time value (CPU, memory)
    HISTOGRAM = "histogram"       # Distribution of values (latency)
    RATE = "rate"                 # Rate of change (requests/sec)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics"""
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "gt"  # gt, lt, eq, ne
    evaluation_window_seconds: int = 60
    consecutive_violations: int = 3


@dataclass
class HealthMetric:
    """Individual health metric"""
    metric_id: str
    metric_type: MetricType
    component_id: str
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold: Optional[MetricThreshold] = None


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    component_id: str
    metric_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


@dataclass
class ComponentHealthSummary:
    """Summary of component health status"""
    component_id: str
    overall_status: HealthStatus
    last_update: datetime
    metrics_count: int
    active_alerts: int
    uptime_percentage: float
    response_time_avg: float
    error_rate: float
    resource_usage: Dict[str, float] = field(default_factory=dict)


class HealthCollector:
    """Base class for health metric collectors"""

    def __init__(self, component_id: str, collection_interval: int = 30):
        """
        Initialize health collector.

        Args:
            component_id: ID of component being monitored
            collection_interval: Seconds between collections
        """
        self.component_id = component_id
        self.collection_interval = collection_interval
        self.running = False
        self._collection_task: Optional[asyncio.Task] = None

    async def collect_metrics(self) -> List[HealthMetric]:
        """Collect health metrics. Override in subclasses."""
        raise NotImplementedError

    async def start(self):
        """Start metric collection."""
        if not self.running:
            self.running = True
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info(f"Started health collector for {self.component_id}")

    async def stop(self):
        """Stop metric collection."""
        self.running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health collector for {self.component_id}")

    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                metrics = await self.collect_metrics()
                # Send metrics to health monitor (implement callback mechanism)
                await self._send_metrics(metrics)
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health collection for {self.component_id}: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _send_metrics(self, metrics: List[HealthMetric]):
        """Send metrics to health monitor."""
        # This will be set by HealthMonitor when registering collector
        if hasattr(self, '_metrics_callback'):
            await self._metrics_callback(metrics)


class SystemResourceCollector(HealthCollector):
    """Collector for system resource metrics"""

    async def collect_metrics(self) -> List[HealthMetric]:
        """Collect system resource metrics."""
        timestamp = datetime.now()
        metrics = []

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                metric_id="cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                component_id=self.component_id,
                value=cpu_percent,
                timestamp=timestamp,
                threshold=MetricThreshold(warning_threshold=70.0, critical_threshold=90.0)
            ))

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                metric_id="memory_usage_percent",
                metric_type=MetricType.GAUGE,
                component_id=self.component_id,
                value=memory.percent,
                timestamp=timestamp,
                threshold=MetricThreshold(warning_threshold=80.0, critical_threshold=95.0)
            ))

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(HealthMetric(
                metric_id="disk_usage_percent",
                metric_type=MetricType.GAUGE,
                component_id=self.component_id,
                value=disk.percent,
                timestamp=timestamp,
                threshold=MetricThreshold(warning_threshold=85.0, critical_threshold=95.0)
            ))

            # Network metrics
            net_io = psutil.net_io_counters()
            metrics.extend([
                HealthMetric(
                    metric_id="network_bytes_sent",
                    metric_type=MetricType.COUNTER,
                    component_id=self.component_id,
                    value=net_io.bytes_sent,
                    timestamp=timestamp
                ),
                HealthMetric(
                    metric_id="network_bytes_recv",
                    metric_type=MetricType.COUNTER,
                    component_id=self.component_id,
                    value=net_io.bytes_recv,
                    timestamp=timestamp
                )
            ])

            # Process-specific metrics
            process = psutil.Process(os.getpid())
            metrics.extend([
                HealthMetric(
                    metric_id="process_memory_mb",
                    metric_type=MetricType.GAUGE,
                    component_id=self.component_id,
                    value=process.memory_info().rss / 1024 / 1024,
                    timestamp=timestamp,
                    threshold=MetricThreshold(warning_threshold=500.0, critical_threshold=1000.0)
                ),
                HealthMetric(
                    metric_id="process_cpu_percent",
                    metric_type=MetricType.GAUGE,
                    component_id=self.component_id,
                    value=process.cpu_percent(),
                    timestamp=timestamp,
                    threshold=MetricThreshold(warning_threshold=50.0, critical_threshold=80.0)
                )
            ])

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics


class TradingSystemCollector(HealthCollector):
    """Collector for trading system specific metrics"""

    def __init__(self, component_id: str, trading_engine, collection_interval: int = 30):
        """Initialize with trading engine reference."""
        super().__init__(component_id, collection_interval)
        self.trading_engine = trading_engine

    async def collect_metrics(self) -> List[HealthMetric]:
        """Collect trading system metrics."""
        timestamp = datetime.now()
        metrics = []

        try:
            if not self.trading_engine:
                return metrics

            # Connection status
            broker_connected = getattr(self.trading_engine.broker, 'is_connected', False) if self.trading_engine.broker else False
            metrics.append(HealthMetric(
                metric_id="broker_connection_status",
                metric_type=MetricType.GAUGE,
                component_id=self.component_id,
                value=1.0 if broker_connected else 0.0,
                timestamp=timestamp,
                threshold=MetricThreshold(warning_threshold=0.5, critical_threshold=0.5, comparison_operator="lt")
            ))

            # Portfolio metrics
            if hasattr(self.trading_engine, 'portfolio_manager') and self.trading_engine.portfolio_manager:
                portfolio = self.trading_engine.portfolio_manager

                # Account value
                if hasattr(portfolio, 'get_total_value'):
                    try:
                        total_value = await portfolio.get_total_value()
                        metrics.append(HealthMetric(
                            metric_id="portfolio_total_value",
                            metric_type=MetricType.GAUGE,
                            component_id=self.component_id,
                            value=float(total_value),
                            timestamp=timestamp
                        ))
                    except Exception as e:
                        logger.warning(f"Could not get portfolio value: {e}")

                # Position count
                if hasattr(portfolio, 'get_positions'):
                    try:
                        positions = await portfolio.get_positions()
                        metrics.append(HealthMetric(
                            metric_id="active_positions_count",
                            metric_type=MetricType.GAUGE,
                            component_id=self.component_id,
                            value=len(positions),
                            timestamp=timestamp
                        ))
                    except Exception as e:
                        logger.warning(f"Could not get positions: {e}")

            # Engine health
            if hasattr(self.trading_engine, 'running'):
                metrics.append(HealthMetric(
                    metric_id="engine_running_status",
                    metric_type=MetricType.GAUGE,
                    component_id=self.component_id,
                    value=1.0 if self.trading_engine.running else 0.0,
                    timestamp=timestamp,
                    threshold=MetricThreshold(warning_threshold=0.5, critical_threshold=0.5, comparison_operator="lt")
                ))

        except Exception as e:
            logger.error(f"Error collecting trading system metrics: {e}")

        return metrics


class HealthMonitor:
    """
    Comprehensive health monitoring system.

    Provides real-time monitoring, anomaly detection, and intelligent
    alerting for all system components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Health Monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.collectors: Dict[str, HealthCollector] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[HealthAlert] = []
        self.component_status: Dict[str, ComponentHealthSummary] = {}
        self._lock = threading.RLock()

        # Configuration
        self.alert_history_size = config.get('alert_history_size', 1000)
        self.metrics_retention_hours = config.get('metrics_retention_hours', 24)
        self.anomaly_detection_enabled = config.get('anomaly_detection_enabled', True)

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("Health Monitor initialized")

    async def start(self):
        """Start health monitoring."""
        if self.running:
            return

        self.running = True

        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Start all registered collectors
        for collector in self.collectors.values():
            await collector.start()

        logger.info("Health Monitor started")

    async def stop(self):
        """Stop health monitoring."""
        if not self.running:
            return

        self.running = False

        # Stop collectors
        for collector in self.collectors.values():
            await collector.stop()

        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Wait for tasks to complete
        for task in [self._monitoring_task, self._cleanup_task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Health Monitor stopped")

    def register_collector(self, collector: HealthCollector):
        """Register a health collector."""
        self.collectors[collector.component_id] = collector

        # Set callback for metrics
        collector._metrics_callback = self._receive_metrics

        logger.info(f"Registered health collector: {collector.component_id}")

    async def _receive_metrics(self, metrics: List[HealthMetric]):
        """Receive and process metrics from collectors."""
        with self._lock:
            for metric in metrics:
                # Store metric
                metric_key = f"{metric.component_id}.{metric.metric_id}"
                self.metrics_history[metric_key].append(metric)

                # Evaluate thresholds
                if metric.threshold:
                    await self._evaluate_threshold(metric)

                # Update component status
                await self._update_component_status(metric)

    async def _evaluate_threshold(self, metric: HealthMetric):
        """Evaluate metric against its threshold."""
        if not metric.threshold:
            return

        threshold = metric.threshold
        value = metric.value

        # Determine if threshold is violated
        violation = False
        if threshold.comparison_operator == "gt":
            violation = value > threshold.critical_threshold
            severity = AlertSeverity.CRITICAL
            if not violation and value > threshold.warning_threshold:
                violation = True
                severity = AlertSeverity.WARNING
        elif threshold.comparison_operator == "lt":
            violation = value < threshold.critical_threshold
            severity = AlertSeverity.CRITICAL
            if not violation and value < threshold.warning_threshold:
                violation = True
                severity = AlertSeverity.WARNING

        if violation:
            # Check consecutive violations
            metric_key = f"{metric.component_id}.{metric.metric_id}"
            recent_metrics = list(self.metrics_history[metric_key])[-threshold.consecutive_violations:]

            if len(recent_metrics) >= threshold.consecutive_violations:
                all_violations = True
                for recent_metric in recent_metrics:
                    if threshold.comparison_operator == "gt":
                        if recent_metric.value <= threshold.warning_threshold:
                            all_violations = False
                            break
                    elif threshold.comparison_operator == "lt":
                        if recent_metric.value >= threshold.warning_threshold:
                            all_violations = False
                            break

                if all_violations:
                    await self._create_alert(metric, severity)

    async def _create_alert(self, metric: HealthMetric, severity: AlertSeverity):
        """Create a new alert."""
        alert_id = f"{metric.component_id}_{metric.metric_id}_{int(time.time())}"

        # Check if similar alert already exists
        existing_alert = self._find_existing_alert(metric.component_id, metric.metric_id)
        if existing_alert and not existing_alert.resolved:
            return  # Don't create duplicate alerts

        alert = HealthAlert(
            alert_id=alert_id,
            component_id=metric.component_id,
            metric_id=metric.metric_id,
            severity=severity,
            message=f"{metric.metric_id} threshold violated: {metric.value}",
            timestamp=datetime.now(),
            details={
                'metric_value': metric.value,
                'threshold_warning': metric.threshold.warning_threshold if metric.threshold else None,
                'threshold_critical': metric.threshold.critical_threshold if metric.threshold else None,
                'comparison_operator': metric.threshold.comparison_operator if metric.threshold else None
            }
        )

        with self._lock:
            self.alerts.append(alert)

            # Limit alert history
            if len(self.alerts) > self.alert_history_size:
                self.alerts = self.alerts[-self.alert_history_size // 2:]

        # Execute alert callbacks
        await self._execute_alert_callbacks(alert)

        logger.warning(f"Health alert created: {alert.alert_id} - {alert.message}")

    def _find_existing_alert(self, component_id: str, metric_id: str) -> Optional[HealthAlert]:
        """Find existing unresolved alert for component/metric."""
        with self._lock:
            for alert in reversed(self.alerts):  # Check most recent first
                if (alert.component_id == component_id and
                    alert.metric_id == metric_id and
                    not alert.resolved):
                    return alert
        return None

    async def _update_component_status(self, metric: HealthMetric):
        """Update component health status based on metrics."""
        component_id = metric.component_id

        with self._lock:
            if component_id not in self.component_status:
                self.component_status[component_id] = ComponentHealthSummary(
                    component_id=component_id,
                    overall_status=HealthStatus.HEALTHY,
                    last_update=datetime.now(),
                    metrics_count=0,
                    active_alerts=0,
                    uptime_percentage=100.0,
                    response_time_avg=0.0,
                    error_rate=0.0
                )

            status = self.component_status[component_id]
            status.last_update = datetime.now()
            status.metrics_count += 1

            # Count active alerts for this component
            active_alerts = sum(1 for alert in self.alerts
                              if alert.component_id == component_id and not alert.resolved)
            status.active_alerts = active_alerts

            # Determine overall status based on alerts
            critical_alerts = sum(1 for alert in self.alerts
                                if alert.component_id == component_id
                                and not alert.resolved
                                and alert.severity == AlertSeverity.CRITICAL)

            warning_alerts = sum(1 for alert in self.alerts
                               if alert.component_id == component_id
                               and not alert.resolved
                               and alert.severity == AlertSeverity.WARNING)

            if critical_alerts > 0:
                status.overall_status = HealthStatus.CRITICAL
            elif warning_alerts > 0:
                status.overall_status = HealthStatus.WARNING
            else:
                status.overall_status = HealthStatus.HEALTHY

    async def _monitoring_loop(self):
        """Main monitoring loop for periodic tasks."""
        while self.running:
            try:
                # Perform anomaly detection
                if self.anomaly_detection_enabled:
                    await self._detect_anomalies()

                # Update system-wide health metrics
                await self._update_system_health()

                # Sleep until next monitoring cycle
                await asyncio.sleep(60)  # Run every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self):
        """Cleanup old metrics and resolved alerts."""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _detect_anomalies(self):
        """Detect anomalies in metric patterns."""
        # Simple anomaly detection based on statistical analysis
        with self._lock:
            for metric_key, metrics in self.metrics_history.items():
                if len(metrics) < 10:  # Need enough data points
                    continue

                recent_values = [m.value for m in list(metrics)[-10:]]

                # Calculate statistical thresholds
                mean_val = statistics.mean(recent_values)
                try:
                    stdev_val = statistics.stdev(recent_values)

                    # Check if latest value is more than 2 standard deviations from mean
                    latest_value = recent_values[-1]
                    z_score = abs(latest_value - mean_val) / stdev_val if stdev_val > 0 else 0

                    if z_score > 2.0:  # Potential anomaly
                        component_id, metric_id = metric_key.split('.', 1)
                        logger.info(f"Anomaly detected in {metric_key}: value={latest_value}, "
                                   f"mean={mean_val:.2f}, stdev={stdev_val:.2f}, z_score={z_score:.2f}")

                except statistics.StatisticsError:
                    # Not enough variance in data
                    continue

    async def _update_system_health(self):
        """Update system-wide health metrics."""
        with self._lock:
            total_components = len(self.component_status)
            if total_components == 0:
                return

            healthy_components = sum(1 for status in self.component_status.values()
                                   if status.overall_status == HealthStatus.HEALTHY)

            system_health_percentage = (healthy_components / total_components) * 100

            # Log system health summary
            logger.debug(f"System health: {healthy_components}/{total_components} components healthy "
                        f"({system_health_percentage:.1f}%)")

    async def _cleanup_old_data(self):
        """Clean up old metrics and resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)

        with self._lock:
            # Clean up old metrics
            for metric_key in self.metrics_history:
                metrics = self.metrics_history[metric_key]
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()

            # Clean up old resolved alerts
            self.alerts = [alert for alert in self.alerts
                          if not alert.resolved or
                          (alert.resolved_timestamp and
                           alert.resolved_timestamp > cutoff_time)]

        logger.debug("Completed health monitoring data cleanup")

    async def _execute_alert_callbacks(self, alert: HealthAlert):
        """Execute registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback registered")

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        with self._lock:
            total_components = len(self.component_status)
            if total_components == 0:
                return {'status': 'unknown', 'components': 0}

            status_counts = defaultdict(int)
            for status in self.component_status.values():
                status_counts[status.overall_status.value] += 1

            active_alerts_by_severity = defaultdict(int)
            for alert in self.alerts:
                if not alert.resolved:
                    active_alerts_by_severity[alert.severity.value] += 1

            return {
                'overall_status': self._determine_overall_status(),
                'total_components': total_components,
                'component_status_breakdown': dict(status_counts),
                'active_alerts': sum(active_alerts_by_severity.values()),
                'alerts_by_severity': dict(active_alerts_by_severity),
                'last_update': max(status.last_update for status in self.component_status.values()).isoformat(),
                'components': {
                    comp_id: {
                        'status': status.overall_status.value,
                        'active_alerts': status.active_alerts,
                        'last_update': status.last_update.isoformat(),
                        'uptime_percentage': status.uptime_percentage
                    }
                    for comp_id, status in self.component_status.items()
                }
            }

    def _determine_overall_status(self) -> str:
        """Determine overall system health status."""
        if not self.component_status:
            return HealthStatus.UNKNOWN.value

        critical_count = sum(1 for status in self.component_status.values()
                           if status.overall_status == HealthStatus.CRITICAL)
        warning_count = sum(1 for status in self.component_status.values()
                          if status.overall_status == HealthStatus.WARNING)

        if critical_count > 0:
            return HealthStatus.CRITICAL.value
        elif warning_count > 0:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value

    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = datetime.now()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False