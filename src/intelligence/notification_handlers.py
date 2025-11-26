"""
ISS-018: Real Notification Handlers for Alert System.

Implements email (SMTP) and Slack webhook handlers for the AlertOrchestrator.
"""
import os
import logging
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, Optional, List
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class EmailNotificationHandler:
    """
    Email notification handler using SMTP.

    ISS-018: Implements real email notifications for trading alerts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize email handler.

        Args:
            config: Optional configuration dict. Falls back to environment variables.
        """
        config = config or {}

        # SMTP configuration from config or environment
        self.smtp_host = config.get('smtp_host') or os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(config.get('smtp_port') or os.getenv('SMTP_PORT', '587'))
        self.smtp_user = config.get('smtp_user') or os.getenv('SMTP_USER', '')
        self.smtp_password = config.get('smtp_password') or os.getenv('SMTP_PASSWORD', '')
        self.from_email = config.get('from_email') or os.getenv('ALERT_FROM_EMAIL', self.smtp_user)
        self.to_emails = config.get('to_emails') or os.getenv('ALERT_TO_EMAILS', '').split(',')
        self.use_tls = config.get('use_tls', True)

        # Rate limiting
        self._last_send_time = None
        self._min_interval_seconds = config.get('min_interval_seconds', 60)

        self.enabled = bool(self.smtp_user and self.smtp_password and any(self.to_emails))

        if self.enabled:
            logger.info(f"Email notification handler initialized: {self.smtp_host}:{self.smtp_port}")
        else:
            logger.warning("Email notification handler disabled - missing SMTP credentials")

    def __call__(self, alert: Any) -> bool:
        """
        Send email notification for an alert.

        Args:
            alert: AlertMessage object from AlertOrchestrator

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Email notifications disabled")
            return False

        # Rate limiting
        now = datetime.now()
        if self._last_send_time:
            elapsed = (now - self._last_send_time).total_seconds()
            if elapsed < self._min_interval_seconds:
                logger.debug(f"Rate limited - {self._min_interval_seconds - elapsed:.0f}s until next email")
                return False

        try:
            # Build email content
            subject = self._build_subject(alert)
            body = self._build_body(alert)

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)

            # Add plain text and HTML versions
            text_part = MIMEText(body, 'plain')
            html_part = MIMEText(self._build_html_body(alert), 'html')
            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())

            self._last_send_time = now
            logger.info(f"Email alert sent: {subject}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _build_subject(self, alert: Any) -> str:
        """Build email subject line"""
        severity = getattr(alert, 'severity', None)
        severity_str = severity.value.upper() if severity else 'ALERT'
        symbol = getattr(alert, 'symbol', 'UNKNOWN')
        alert_type = getattr(alert, 'alert_type', 'Alert')

        return f"[{severity_str}] Trader-AI: {alert_type} - {symbol}"

    def _build_body(self, alert: Any) -> str:
        """Build plain text email body"""
        lines = [
            f"Trader-AI Alert Notification",
            f"{'=' * 40}",
            f"",
            f"Severity: {getattr(alert, 'severity', 'Unknown').value if hasattr(getattr(alert, 'severity', None), 'value') else 'Unknown'}",
            f"Symbol: {getattr(alert, 'symbol', 'Unknown')}",
            f"Type: {getattr(alert, 'alert_type', 'Unknown')}",
            f"Time: {getattr(alert, 'timestamp', datetime.now())}",
            f"Confidence: {getattr(alert, 'confidence', 0):.1%}",
            f"",
            f"Message: {getattr(alert, 'message', 'No message')}",
            f"",
        ]

        details = getattr(alert, 'details', {})
        if details:
            lines.append("Details:")
            for key, value in details.items():
                lines.append(f"  - {key}: {value}")

        lines.extend([
            f"",
            f"Alert ID: {getattr(alert, 'alert_id', 'Unknown')}",
            f"Source: {getattr(alert, 'source_component', 'Unknown')}",
            f"",
            f"---",
            f"This is an automated alert from Trader-AI Risk Management System."
        ])

        return '\n'.join(lines)

    def _build_html_body(self, alert: Any) -> str:
        """Build HTML email body"""
        severity = getattr(alert, 'severity', None)
        severity_str = severity.value if severity and hasattr(severity, 'value') else 'unknown'

        # Color based on severity
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#17a2b8'
        }
        color = colors.get(severity_str.lower(), '#6c757d')

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">Trader-AI Alert: {getattr(alert, 'alert_type', 'Unknown')}</h2>
            </div>
            <div style="padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6;">
                <p><strong>Symbol:</strong> {getattr(alert, 'symbol', 'Unknown')}</p>
                <p><strong>Severity:</strong> {severity_str.upper()}</p>
                <p><strong>Confidence:</strong> {getattr(alert, 'confidence', 0):.1%}</p>
                <p><strong>Time:</strong> {getattr(alert, 'timestamp', datetime.now())}</p>
                <hr>
                <p><strong>Message:</strong></p>
                <p>{getattr(alert, 'message', 'No message')}</p>
            </div>
            <div style="padding: 10px; background: #e9ecef; font-size: 12px; color: #6c757d;">
                Alert ID: {getattr(alert, 'alert_id', 'Unknown')} |
                Source: {getattr(alert, 'source_component', 'Unknown')}
            </div>
        </body>
        </html>
        """


class SlackNotificationHandler:
    """
    Slack notification handler using webhooks.

    ISS-018: Implements real Slack notifications for trading alerts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Slack handler.

        Args:
            config: Optional configuration dict. Falls back to environment variables.
        """
        config = config or {}

        # Slack configuration
        self.webhook_url = config.get('webhook_url') or os.getenv('SLACK_WEBHOOK_URL', '')
        self.channel = config.get('channel') or os.getenv('SLACK_CHANNEL', '#trading-alerts')
        self.username = config.get('username', 'Trader-AI Alert Bot')
        self.icon_emoji = config.get('icon_emoji', ':chart_with_upwards_trend:')

        # Rate limiting
        self._last_send_time = None
        self._min_interval_seconds = config.get('min_interval_seconds', 30)

        self.enabled = bool(self.webhook_url)

        if self.enabled:
            logger.info(f"Slack notification handler initialized for channel: {self.channel}")
        else:
            logger.warning("Slack notification handler disabled - missing webhook URL")

    def __call__(self, alert: Any) -> bool:
        """
        Send Slack notification for an alert.

        Args:
            alert: AlertMessage object from AlertOrchestrator

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Slack notifications disabled")
            return False

        # Rate limiting
        now = datetime.now()
        if self._last_send_time:
            elapsed = (now - self._last_send_time).total_seconds()
            if elapsed < self._min_interval_seconds:
                logger.debug(f"Rate limited - {self._min_interval_seconds - elapsed:.0f}s until next Slack message")
                return False

        try:
            payload = self._build_payload(alert)

            # Send webhook POST
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    self._last_send_time = now
                    logger.info(f"Slack alert sent: {getattr(alert, 'alert_type', 'Alert')}")
                    return True
                else:
                    logger.error(f"Slack webhook failed: {response.status_code} - {response.text}")
                    return False

        except httpx.TimeoutException:
            logger.error("Slack webhook timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _build_payload(self, alert: Any) -> Dict[str, Any]:
        """Build Slack message payload"""
        severity = getattr(alert, 'severity', None)
        severity_str = severity.value if severity and hasattr(severity, 'value') else 'unknown'

        # Color based on severity
        colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#17a2b8'
        }
        color = colors.get(severity_str.lower(), '#6c757d')

        # Build attachment
        attachment = {
            'color': color,
            'title': f"{severity_str.upper()}: {getattr(alert, 'alert_type', 'Alert')}",
            'text': getattr(alert, 'message', 'No message'),
            'fields': [
                {
                    'title': 'Symbol',
                    'value': getattr(alert, 'symbol', 'Unknown'),
                    'short': True
                },
                {
                    'title': 'Confidence',
                    'value': f"{getattr(alert, 'confidence', 0):.1%}",
                    'short': True
                },
                {
                    'title': 'Source',
                    'value': getattr(alert, 'source_component', 'Unknown'),
                    'short': True
                },
                {
                    'title': 'Priority',
                    'value': getattr(alert, 'priority', 'Unknown').value if hasattr(getattr(alert, 'priority', None), 'value') else 'Unknown',
                    'short': True
                }
            ],
            'footer': f"Alert ID: {getattr(alert, 'alert_id', 'Unknown')}",
            'ts': int(getattr(alert, 'timestamp', datetime.now()).timestamp()) if hasattr(getattr(alert, 'timestamp', None), 'timestamp') else int(datetime.now().timestamp())
        }

        return {
            'channel': self.channel,
            'username': self.username,
            'icon_emoji': self.icon_emoji,
            'attachments': [attachment]
        }


class LogNotificationHandler:
    """
    Simple logging handler for alerts (always available).

    ISS-018: Provides fallback logging for all alerts.
    """

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.enabled = True
        logger.info("Log notification handler initialized")

    def __call__(self, alert: Any) -> bool:
        """Log the alert"""
        try:
            severity = getattr(alert, 'severity', None)
            severity_str = severity.value if severity and hasattr(severity, 'value') else 'INFO'

            # Map severity to log level
            log_levels = {
                'critical': logging.CRITICAL,
                'high': logging.ERROR,
                'medium': logging.WARNING,
                'low': logging.INFO
            }
            level = log_levels.get(severity_str.lower(), logging.INFO)

            message = (
                f"ALERT [{severity_str.upper()}] "
                f"Symbol={getattr(alert, 'symbol', 'Unknown')} "
                f"Type={getattr(alert, 'alert_type', 'Unknown')} "
                f"Confidence={getattr(alert, 'confidence', 0):.1%} "
                f"Message={getattr(alert, 'message', 'No message')}"
            )

            logger.log(level, message)
            return True

        except Exception as e:
            logger.error(f"Log notification failed: {e}")
            return False


class DashboardNotificationHandler:
    """
    Dashboard notification handler for real-time UI updates.

    ISS-018: Integrates with dashboard WebSocket connections.
    """

    def __init__(self, websocket_manager=None):
        """
        Initialize dashboard handler.

        Args:
            websocket_manager: Reference to WebSocket connection manager
        """
        self.websocket_manager = websocket_manager
        self.enabled = websocket_manager is not None
        logger.info(f"Dashboard notification handler initialized: {'enabled' if self.enabled else 'disabled'}")

    def __call__(self, alert: Any) -> bool:
        """Send alert to dashboard via WebSocket"""
        if not self.enabled or not self.websocket_manager:
            return False

        try:
            # Build dashboard message
            message = {
                'type': 'alert',
                'data': {
                    'alert_id': getattr(alert, 'alert_id', 'Unknown'),
                    'symbol': getattr(alert, 'symbol', 'Unknown'),
                    'alert_type': getattr(alert, 'alert_type', 'Unknown'),
                    'severity': getattr(alert, 'severity', 'Unknown').value if hasattr(getattr(alert, 'severity', None), 'value') else 'Unknown',
                    'confidence': getattr(alert, 'confidence', 0),
                    'message': getattr(alert, 'message', 'No message'),
                    'timestamp': str(getattr(alert, 'timestamp', datetime.now()))
                }
            }

            # Broadcast to all connected clients
            if hasattr(self.websocket_manager, 'broadcast'):
                import asyncio
                asyncio.create_task(self.websocket_manager.broadcast(json.dumps(message)))
                return True
            else:
                logger.warning("WebSocket manager does not support broadcast")
                return False

        except Exception as e:
            logger.error(f"Dashboard notification failed: {e}")
            return False


def register_default_handlers(orchestrator, config: Optional[Dict[str, Any]] = None):
    """
    Register default notification handlers with AlertOrchestrator.

    ISS-018: Convenience function for handler setup.

    Args:
        orchestrator: AlertOrchestrator instance
        config: Optional configuration dict
    """
    config = config or {}

    # Always register log handler
    log_handler = LogNotificationHandler()
    orchestrator.register_notification_handler('log', log_handler)

    # Register email handler if configured
    email_config = config.get('email', {})
    email_handler = EmailNotificationHandler(email_config)
    if email_handler.enabled:
        orchestrator.register_notification_handler('email', email_handler)
        orchestrator.register_notification_handler('urgent', email_handler)  # Urgent also uses email

    # Register Slack handler if configured
    slack_config = config.get('slack', {})
    slack_handler = SlackNotificationHandler(slack_config)
    if slack_handler.enabled:
        orchestrator.register_notification_handler('slack', slack_handler)

    logger.info(f"Registered {len(orchestrator.notification_handlers)} notification handlers")
