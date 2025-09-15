"""
Monitoring and Logging System for Weekly Siphon Operations

Provides comprehensive monitoring, alerting, and logging for the automated
siphon system with performance metrics and health checks.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SiphonAlert:
    """Alert for siphon system issues"""
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]


@dataclass
class SiphonMetrics:
    """Performance metrics for siphon system"""
    timestamp: datetime
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_withdrawn: Decimal
    average_withdrawal: Decimal
    success_rate_percent: float
    capital_protection_violations: int
    last_execution_time: Optional[datetime]


class SiphonMonitor:
    """
    Monitoring system for weekly siphon automation.

    Features:
    - Real-time performance metrics
    - Automated alert generation
    - Health status monitoring
    - Audit log generation
    - Performance trend analysis
    """

    def __init__(self,
                 log_directory: str = "logs/siphon",
                 enable_file_logging: bool = True,
                 enable_alerts: bool = True):
        """
        Initialize siphon monitor.

        Args:
            log_directory: Directory for log files
            enable_file_logging: Enable logging to files
            enable_alerts: Enable alert generation
        """
        self.log_directory = log_directory
        self.enable_file_logging = enable_file_logging
        self.enable_alerts = enable_alerts

        # Alert tracking
        self.alerts: List[SiphonAlert] = []
        self.alert_handlers: List[Callable] = []

        # Performance metrics
        self.metrics_history: List[SiphonMetrics] = []

        # Setup logging
        if self.enable_file_logging:
            self._setup_file_logging()

        logger.info("SiphonMonitor initialized")

    def _setup_file_logging(self) -> None:
        """Setup file-based logging."""
        try:
            os.makedirs(self.log_directory, exist_ok=True)

            # Create siphon-specific logger
            siphon_logger = logging.getLogger('siphon_automation')
            siphon_logger.setLevel(logging.INFO)

            # File handler for siphon logs
            log_file = os.path.join(self.log_directory, 'siphon_automation.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)

            siphon_logger.addHandler(file_handler)

            logger.info(f"File logging setup complete: {log_file}")

        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")

    async def monitor_siphon_execution(self, siphon_result) -> None:
        """
        Monitor a siphon execution result.

        Args:
            siphon_result: SiphonResult from execution
        """
        try:
            # Log execution details
            self._log_execution(siphon_result)

            # Check for alerts
            await self._check_execution_alerts(siphon_result)

            # Update metrics
            await self._update_metrics()

            logger.info(f"Monitored siphon execution: {siphon_result.status.value}")

        except Exception as e:
            logger.error(f"Failed to monitor siphon execution: {e}")

    def _log_execution(self, siphon_result) -> None:
        """Log siphon execution details."""
        siphon_logger = logging.getLogger('siphon_automation')

        log_data = {
            'timestamp': siphon_result.timestamp.isoformat(),
            'status': siphon_result.status.value,
            'withdrawal_amount': float(siphon_result.withdrawal_amount),
            'withdrawal_success': siphon_result.withdrawal_success,
            'broker_confirmation': siphon_result.broker_confirmation,
            'safety_checks_count': len(siphon_result.safety_checks),
            'errors_count': len(siphon_result.errors)
        }

        if siphon_result.profit_calculation:
            log_data['profit_data'] = {
                'total_profit': float(siphon_result.profit_calculation.total_profit),
                'reinvestment_amount': float(siphon_result.profit_calculation.reinvestment_amount),
                'base_capital': float(siphon_result.profit_calculation.base_capital),
                'weekly_return_percent': float(siphon_result.profit_calculation.weekly_return_percent)
            }

        siphon_logger.info(f"SIPHON_EXECUTION: {json.dumps(log_data, indent=2)}")

        # Log errors if any
        if siphon_result.errors:
            for error in siphon_result.errors:
                siphon_logger.error(f"SIPHON_ERROR: {error}")

    async def _check_execution_alerts(self, siphon_result) -> None:
        """Check siphon execution for alert conditions."""
        alerts = []

        # Failed execution alert
        if siphon_result.status.value in ['failed', 'safety_block']:
            alerts.append(SiphonAlert(
                timestamp=datetime.now(),
                level=AlertLevel.ERROR,
                component="execution",
                message=f"Siphon execution failed: {siphon_result.status.value}",
                details={
                    'status': siphon_result.status.value,
                    'errors': siphon_result.errors,
                    'withdrawal_amount': float(siphon_result.withdrawal_amount)
                }
            ))

        # Capital protection violation alert
        if 'breach' in str(siphon_result.errors).lower() or 'safety' in str(siphon_result.errors).lower():
            alerts.append(SiphonAlert(
                timestamp=datetime.now(),
                level=AlertLevel.CRITICAL,
                component="capital_protection",
                message="Capital protection safeguard activated",
                details={
                    'safety_checks': siphon_result.safety_checks,
                    'errors': siphon_result.errors
                }
            ))

        # Large withdrawal alert (>5% of base capital)
        if siphon_result.profit_calculation and siphon_result.withdrawal_amount > 0:
            base_capital = siphon_result.profit_calculation.base_capital
            withdrawal_percent = (siphon_result.withdrawal_amount / base_capital) * Decimal("100")

            if withdrawal_percent > Decimal("5.0"):
                alerts.append(SiphonAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    component="large_withdrawal",
                    message=f"Large withdrawal detected: {withdrawal_percent:.1f}% of base capital",
                    details={
                        'withdrawal_amount': float(siphon_result.withdrawal_amount),
                        'withdrawal_percent': float(withdrawal_percent),
                        'base_capital': float(base_capital)
                    }
                ))

        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)

    async def _process_alert(self, alert: SiphonAlert) -> None:
        """Process and handle an alert."""
        self.alerts.append(alert)

        # Log the alert
        alert_logger = logging.getLogger('siphon_automation')
        alert_logger.log(
            self._get_log_level(alert.level),
            f"SIPHON_ALERT: {alert.level.value.upper()} - {alert.component} - {alert.message}"
        )

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Convert AlertLevel to logging level."""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)

    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        # This would typically query the siphon automator for current stats
        # For now, we'll create a placeholder implementation
        current_time = datetime.now()

        # Create basic metrics (would be populated from actual data)
        metrics = SiphonMetrics(
            timestamp=current_time,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            total_withdrawn=Decimal("0.00"),
            average_withdrawal=Decimal("0.00"),
            success_rate_percent=0.0,
            capital_protection_violations=0,
            last_execution_time=None
        )

        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of siphon system."""
        current_time = datetime.now()
        recent_alerts = [alert for alert in self.alerts
                        if alert.timestamp > current_time - timedelta(hours=24)]

        critical_alerts = [alert for alert in recent_alerts
                          if alert.level == AlertLevel.CRITICAL]
        error_alerts = [alert for alert in recent_alerts
                       if alert.level == AlertLevel.ERROR]

        # Determine overall health
        if critical_alerts:
            health_status = "CRITICAL"
        elif error_alerts:
            health_status = "WARNING"
        elif recent_alerts:
            health_status = "HEALTHY_WITH_ALERTS"
        else:
            health_status = "HEALTHY"

        return {
            'status': health_status,
            'timestamp': current_time,
            'alerts_24h': {
                'total': len(recent_alerts),
                'critical': len(critical_alerts),
                'error': len(error_alerts),
                'warning': len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
                'info': len([a for a in recent_alerts if a.level == AlertLevel.INFO])
            },
            'metrics_available': len(self.metrics_history) > 0,
            'logging_enabled': self.enable_file_logging,
            'alerts_enabled': self.enable_alerts
        }

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for specified period."""
        cutoff_time = datetime.now() - timedelta(days=days)

        # Filter metrics for the period
        period_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        period_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

        if not period_metrics:
            return {
                'period_days': days,
                'metrics_available': False,
                'message': 'No metrics available for the specified period'
            }

        latest_metrics = period_metrics[-1] if period_metrics else None

        return {
            'period_days': days,
            'metrics_available': True,
            'latest_metrics': {
                'total_executions': latest_metrics.total_executions if latest_metrics else 0,
                'success_rate_percent': latest_metrics.success_rate_percent if latest_metrics else 0.0,
                'total_withdrawn': float(latest_metrics.total_withdrawn) if latest_metrics else 0.0,
                'average_withdrawal': float(latest_metrics.average_withdrawal) if latest_metrics else 0.0,
                'capital_protection_violations': latest_metrics.capital_protection_violations if latest_metrics else 0
            },
            'alerts_period': {
                'total': len(period_alerts),
                'by_level': {
                    'critical': len([a for a in period_alerts if a.level == AlertLevel.CRITICAL]),
                    'error': len([a for a in period_alerts if a.level == AlertLevel.ERROR]),
                    'warning': len([a for a in period_alerts if a.level == AlertLevel.WARNING]),
                    'info': len([a for a in period_alerts if a.level == AlertLevel.INFO])
                }
            }
        }

    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        recent_alerts = sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [{
            'timestamp': alert.timestamp,
            'level': alert.level.value,
            'component': alert.component,
            'message': alert.message,
            'details': alert.details
        } for alert in recent_alerts]

    async def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        period_alerts = [a for a in self.alerts
                        if start_date <= a.timestamp <= end_date]
        period_metrics = [m for m in self.metrics_history
                         if start_date <= m.timestamp <= end_date]

        # Calculate summary statistics
        total_alerts = len(period_alerts)
        critical_alerts = len([a for a in period_alerts if a.level == AlertLevel.CRITICAL])

        latest_metrics = period_metrics[-1] if period_metrics else None

        audit_report = {
            'report_period': {
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days
            },
            'system_health': {
                'total_alerts': total_alerts,
                'critical_alerts': critical_alerts,
                'alert_rate_per_day': total_alerts / max(1, (end_date - start_date).days),
                'overall_status': 'CRITICAL' if critical_alerts > 0 else 'HEALTHY'
            },
            'performance_metrics': {
                'executions_tracked': len(period_metrics),
                'success_rate': latest_metrics.success_rate_percent if latest_metrics else 0.0,
                'total_withdrawn': float(latest_metrics.total_withdrawn) if latest_metrics else 0.0,
                'capital_violations': latest_metrics.capital_protection_violations if latest_metrics else 0
            },
            'alert_breakdown': {
                'critical': len([a for a in period_alerts if a.level == AlertLevel.CRITICAL]),
                'error': len([a for a in period_alerts if a.level == AlertLevel.ERROR]),
                'warning': len([a for a in period_alerts if a.level == AlertLevel.WARNING]),
                'info': len([a for a in period_alerts if a.level == AlertLevel.INFO])
            },
            'recommendations': self._generate_recommendations(period_alerts, latest_metrics)
        }

        # Save audit report if file logging is enabled
        if self.enable_file_logging:
            await self._save_audit_report(audit_report, start_date, end_date)

        return audit_report

    def _generate_recommendations(self, alerts: List[SiphonAlert],
                                latest_metrics: Optional[SiphonMetrics]) -> List[str]:
        """Generate recommendations based on alerts and metrics."""
        recommendations = []

        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("CRITICAL: Investigate capital protection violations immediately")

        # Check success rate
        if latest_metrics and latest_metrics.success_rate_percent < 90:
            recommendations.append("WARNING: Success rate below 90% - review execution conditions")

        # Check for frequent alerts
        if len(alerts) > 10:
            recommendations.append("INFO: High alert frequency - consider adjusting alert thresholds")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations

    async def _save_audit_report(self, report: Dict[str, Any],
                                start_date: datetime, end_date: datetime) -> None:
        """Save audit report to file."""
        try:
            audit_dir = os.path.join(self.log_directory, 'audits')
            os.makedirs(audit_dir, exist_ok=True)

            filename = f"siphon_audit_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            filepath = os.path.join(audit_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Audit report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save audit report: {e}")


# Example alert handlers
async def email_alert_handler(alert: SiphonAlert) -> None:
    """Example email alert handler (placeholder)."""
    if alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
        logger.info(f"EMAIL ALERT: {alert.level.value} - {alert.message}")
        # In real implementation, would send actual email

async def slack_alert_handler(alert: SiphonAlert) -> None:
    """Example Slack alert handler (placeholder)."""
    if alert.level == AlertLevel.CRITICAL:
        logger.info(f"SLACK ALERT: {alert.message}")
        # In real implementation, would send to Slack webhook