"""
Multi-Trigger Kill Switch System
Monitors multiple failure conditions and activates kill switch automatically
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import deque, defaultdict

from .kill_switch_system import TriggerType

logger = logging.getLogger(__name__)

class TriggerSeverity(Enum):
    """Trigger severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TriggerCondition:
    """Trigger condition configuration"""
    name: str
    trigger_type: TriggerType
    severity: TriggerSeverity
    threshold_value: float
    time_window_seconds: float
    consecutive_failures: int = 1
    enabled: bool = True

@dataclass
class TriggerStatus:
    """Current status of a trigger"""
    condition: TriggerCondition
    current_value: float
    last_check: float
    failure_count: int
    consecutive_failures: int
    triggered: bool
    last_trigger_time: Optional[float] = None

class APIHealthMonitor:
    """Monitor API health and connectivity"""

    def __init__(self, broker_interface):
        self.broker = broker_interface
        self.response_times = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.last_successful_call = time.time()
        self.monitoring = False

    async def check_api_health(self) -> Dict[str, Any]:
        """Check API health metrics"""
        start_time = time.time()

        try:
            # Test basic connectivity
            account_info = await self.broker.get_account()
            response_time = (time.time() - start_time) * 1000

            self.response_times.append(response_time)
            self.last_successful_call = time.time()

            return {
                'healthy': True,
                'response_time_ms': response_time,
                'account_accessible': account_info is not None,
                'time_since_last_success': 0
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            self.error_counts[error_type] += 1

            time_since_success = time.time() - self.last_successful_call

            logger.error(f"API health check failed: {e}")

            return {
                'healthy': False,
                'response_time_ms': response_time,
                'error_type': error_type,
                'error_count': self.error_counts[error_type],
                'time_since_last_success': time_since_success,
                'error_message': str(e)
            }

    def get_average_response_time(self) -> float:
        """Get average API response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def get_error_rate(self, time_window: float = 300) -> float:
        """Get error rate over time window"""
        total_errors = sum(self.error_counts.values())
        # Simplified error rate calculation
        return total_errors / (time_window / 60)  # errors per minute

class PositionMonitor:
    """Monitor position sizes and exposure"""

    def __init__(self, broker_interface):
        self.broker = broker_interface
        self.position_history = deque(maxlen=1000)
        self.max_position_value = 0
        self.total_exposure = 0

    async def check_position_limits(self) -> Dict[str, Any]:
        """Check current position sizes and limits"""
        try:
            positions = await self.broker.get_positions()
            account = await self.broker.get_account()

            total_long_value = 0
            total_short_value = 0
            largest_position = 0
            position_count = 0

            for position in positions:
                if position.qty != 0:
                    position_count += 1
                    market_value = abs(float(position.market_value or 0))

                    if position.qty > 0:
                        total_long_value += market_value
                    else:
                        total_short_value += market_value

                    largest_position = max(largest_position, market_value)

            total_exposure = total_long_value + total_short_value
            account_equity = float(account.equity or 0)

            # Calculate exposure ratios
            exposure_ratio = total_exposure / account_equity if account_equity > 0 else 0
            largest_position_ratio = largest_position / account_equity if account_equity > 0 else 0

            self.total_exposure = total_exposure
            self.max_position_value = largest_position

            return {
                'total_exposure': total_exposure,
                'exposure_ratio': exposure_ratio,
                'largest_position': largest_position,
                'largest_position_ratio': largest_position_ratio,
                'position_count': position_count,
                'long_value': total_long_value,
                'short_value': total_short_value,
                'account_equity': account_equity
            }

        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            return {
                'error': str(e),
                'total_exposure': 0,
                'exposure_ratio': 0
            }

class LossLimitMonitor:
    """Monitor account losses and drawdowns"""

    def __init__(self, broker_interface):
        self.broker = broker_interface
        self.equity_history = deque(maxlen=1000)
        self.daily_high_water_mark = 0
        self.session_start_equity = 0

    async def check_loss_limits(self) -> Dict[str, Any]:
        """Check current losses and drawdowns"""
        try:
            account = await self.broker.get_account()
            current_equity = float(account.equity or 0)

            if not self.equity_history:
                self.session_start_equity = current_equity
                self.daily_high_water_mark = current_equity

            self.equity_history.append({
                'timestamp': time.time(),
                'equity': current_equity
            })

            # Update high water mark
            if current_equity > self.daily_high_water_mark:
                self.daily_high_water_mark = current_equity

            # Calculate various loss metrics
            session_pnl = current_equity - self.session_start_equity
            drawdown_from_high = self.daily_high_water_mark - current_equity
            drawdown_percent = (drawdown_from_high / self.daily_high_water_mark) * 100

            # Calculate recent performance
            recent_equity = [e['equity'] for e in list(self.equity_history)[-10:]]
            if len(recent_equity) >= 2:
                recent_change = recent_equity[-1] - recent_equity[0]
                recent_change_percent = (recent_change / recent_equity[0]) * 100
            else:
                recent_change = 0
                recent_change_percent = 0

            return {
                'current_equity': current_equity,
                'session_pnl': session_pnl,
                'session_pnl_percent': (session_pnl / self.session_start_equity) * 100,
                'drawdown_from_high': drawdown_from_high,
                'drawdown_percent': drawdown_percent,
                'recent_change': recent_change,
                'recent_change_percent': recent_change_percent,
                'high_water_mark': self.daily_high_water_mark
            }

        except Exception as e:
            logger.error(f"Loss limit check failed: {e}")
            return {
                'error': str(e),
                'current_equity': 0,
                'session_pnl': 0,
                'drawdown_percent': 0
            }

class NetworkConnectivityMonitor:
    """Monitor network connectivity and latency"""

    def __init__(self):
        self.ping_history = deque(maxlen=100)
        self.connection_failures = 0

    async def check_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        import subprocess
        import platform

        try:
            # Ping a reliable server (Google DNS)
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', '8.8.8.8']

            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5
            )
            response_time = (time.time() - start_time) * 1000

            success = result.returncode == 0

            if success:
                self.ping_history.append(response_time)
            else:
                self.connection_failures += 1

            avg_ping = statistics.mean(self.ping_history) if self.ping_history else 0

            return {
                'connected': success,
                'response_time_ms': response_time,
                'average_ping_ms': avg_ping,
                'connection_failures': self.connection_failures,
                'ping_history_size': len(self.ping_history)
            }

        except Exception as e:
            self.connection_failures += 1
            return {
                'connected': False,
                'error': str(e),
                'connection_failures': self.connection_failures
            }

class MultiTriggerSystem:
    """Comprehensive multi-trigger monitoring system"""

    def __init__(self, kill_switch, broker_interface, config: Dict[str, Any]):
        self.kill_switch = kill_switch
        self.broker = broker_interface
        self.config = config

        # Initialize monitors
        self.api_monitor = APIHealthMonitor(broker_interface)
        self.position_monitor = PositionMonitor(broker_interface)
        self.loss_monitor = LossLimitMonitor(broker_interface)
        self.network_monitor = NetworkConnectivityMonitor()

        # Trigger conditions
        self.trigger_conditions = self._load_trigger_conditions()
        self.trigger_statuses = {}

        # Monitoring state
        self.monitoring = False
        self.monitor_threads = []
        self.last_check_times = {}

        # Event callbacks
        self.trigger_callbacks: Dict[TriggerType, List[Callable]] = defaultdict(list)

        logger.info("Multi-trigger system initialized")

    def _load_trigger_conditions(self) -> List[TriggerCondition]:
        """Load trigger conditions from config"""
        conditions = []

        trigger_config = self.config.get('triggers', {})

        # API failure triggers
        if 'api_failure' in trigger_config:
            api_config = trigger_config['api_failure']
            conditions.extend([
                TriggerCondition(
                    name="API Response Time",
                    trigger_type=TriggerType.API_FAILURE,
                    severity=TriggerSeverity.MEDIUM,
                    threshold_value=api_config.get('response_timeout', 5000),  # 5 second timeout
                    time_window_seconds=60,
                    consecutive_failures=3
                ),
                TriggerCondition(
                    name="API Connection Lost",
                    trigger_type=TriggerType.API_FAILURE,
                    severity=TriggerSeverity.HIGH,
                    threshold_value=api_config.get('connection_timeout', 30),  # 30 seconds offline
                    time_window_seconds=300,
                    consecutive_failures=1
                )
            ])

        # Loss limit triggers
        if 'loss_limits' in trigger_config:
            loss_config = trigger_config['loss_limits']
            conditions.extend([
                TriggerCondition(
                    name="Session Loss Limit",
                    trigger_type=TriggerType.LOSS_LIMIT,
                    severity=TriggerSeverity.CRITICAL,
                    threshold_value=loss_config.get('session_loss_percent', -5.0),  # -5% session loss
                    time_window_seconds=3600,  # 1 hour window
                    consecutive_failures=1
                ),
                TriggerCondition(
                    name="Drawdown Limit",
                    trigger_type=TriggerType.LOSS_LIMIT,
                    severity=TriggerSeverity.HIGH,
                    threshold_value=loss_config.get('drawdown_percent', -3.0),  # -3% drawdown
                    time_window_seconds=1800,  # 30 minute window
                    consecutive_failures=2
                )
            ])

        # Position limit triggers
        if 'position_limits' in trigger_config:
            pos_config = trigger_config['position_limits']
            conditions.extend([
                TriggerCondition(
                    name="Total Exposure Limit",
                    trigger_type=TriggerType.POSITION_LIMIT,
                    severity=TriggerSeverity.HIGH,
                    threshold_value=pos_config.get('max_exposure_ratio', 0.95),  # 95% of equity
                    time_window_seconds=300,
                    consecutive_failures=1
                ),
                TriggerCondition(
                    name="Single Position Limit",
                    trigger_type=TriggerType.POSITION_LIMIT,
                    severity=TriggerSeverity.MEDIUM,
                    threshold_value=pos_config.get('max_single_position_ratio', 0.4),  # 40% of equity
                    time_window_seconds=300,
                    consecutive_failures=2
                )
            ])

        # Network failure triggers
        if 'network' in trigger_config:
            net_config = trigger_config['network']
            conditions.append(
                TriggerCondition(
                    name="Network Connectivity",
                    trigger_type=TriggerType.NETWORK_FAILURE,
                    severity=TriggerSeverity.HIGH,
                    threshold_value=net_config.get('max_ping_ms', 2000),  # 2 second ping
                    time_window_seconds=60,
                    consecutive_failures=5
                )
            )

        return conditions

    def start_monitoring(self):
        """Start all monitoring threads"""
        if self.monitoring:
            logger.warning("Multi-trigger monitoring already active")
            return

        self.monitoring = True

        # Initialize trigger statuses
        for condition in self.trigger_conditions:
            if condition.enabled:
                self.trigger_statuses[condition.name] = TriggerStatus(
                    condition=condition,
                    current_value=0.0,
                    last_check=0.0,
                    failure_count=0,
                    consecutive_failures=0,
                    triggered=False
                )

        # Start monitor threads
        monitors = [
            ('api_health', self._api_health_monitor),
            ('position_limits', self._position_limit_monitor),
            ('loss_limits', self._loss_limit_monitor),
            ('network', self._network_monitor),
            ('trigger_evaluator', self._trigger_evaluator)
        ]

        for name, monitor_func in monitors:
            thread = threading.Thread(target=monitor_func, name=f'multitrigger_{name}')
            thread.daemon = True
            thread.start()
            self.monitor_threads.append(thread)

        logger.info(f"Multi-trigger monitoring started with {len(self.trigger_conditions)} conditions")

    def stop_monitoring(self):
        """Stop all monitoring threads"""
        self.monitoring = False

        # Wait for threads to finish
        for thread in self.monitor_threads:
            thread.join(timeout=1.0)

        self.monitor_threads.clear()
        logger.info("Multi-trigger monitoring stopped")

    def add_trigger_callback(self, trigger_type: TriggerType, callback: Callable):
        """Add callback for specific trigger type"""
        self.trigger_callbacks[trigger_type].append(callback)

    async def _evaluate_trigger(self, status: TriggerStatus, current_value: float) -> bool:
        """Evaluate if trigger condition is met"""
        condition = status.condition
        status.current_value = current_value
        status.last_check = time.time()

        # Check if threshold is breached
        threshold_breached = False

        if condition.trigger_type in [TriggerType.LOSS_LIMIT]:
            # For loss limits, trigger when value is BELOW threshold (more negative)
            threshold_breached = current_value < condition.threshold_value
        else:
            # For other limits, trigger when value is ABOVE threshold
            threshold_breached = current_value > condition.threshold_value

        if threshold_breached:
            status.failure_count += 1
            status.consecutive_failures += 1
        else:
            status.consecutive_failures = 0

        # Check if consecutive failure threshold is met
        should_trigger = (
            threshold_breached and
            status.consecutive_failures >= condition.consecutive_failures and
            not status.triggered
        )

        if should_trigger:
            status.triggered = True
            status.last_trigger_time = time.time()

            logger.critical(
                f"Trigger activated: {condition.name} - "
                f"Value: {current_value}, Threshold: {condition.threshold_value}, "
                f"Consecutive failures: {status.consecutive_failures}"
            )

            # Activate kill switch
            trigger_data = {
                'condition_name': condition.name,
                'current_value': current_value,
                'threshold_value': condition.threshold_value,
                'consecutive_failures': status.consecutive_failures,
                'severity': condition.severity.value
            }

            await self.kill_switch.trigger_kill_switch(
                condition.trigger_type,
                trigger_data
            )

            # Execute callbacks
            for callback in self.trigger_callbacks[condition.trigger_type]:
                try:
                    await callback(status, trigger_data)
                except Exception as e:
                    logger.error(f"Trigger callback error: {e}")

            return True

        return False

    def _api_health_monitor(self):
        """Monitor API health"""
        while self.monitoring:
            try:
                # Run async check
                health_data = asyncio.run(self.api_monitor.check_api_health())

                # Evaluate API response time trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "API Response Time":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            health_data.get('response_time_ms', 0)
                        ))

                # Evaluate API connection trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "API Connection Lost":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            health_data.get('time_since_last_success', 0)
                        ))

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"API health monitor error: {e}")
                time.sleep(5)

    def _position_limit_monitor(self):
        """Monitor position limits"""
        while self.monitoring:
            try:
                position_data = asyncio.run(self.position_monitor.check_position_limits())

                # Evaluate exposure limit trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "Total Exposure Limit":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            position_data.get('exposure_ratio', 0)
                        ))

                # Evaluate single position limit trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "Single Position Limit":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            position_data.get('largest_position_ratio', 0)
                        ))

                time.sleep(15)  # Check every 15 seconds

            except Exception as e:
                logger.error(f"Position limit monitor error: {e}")
                time.sleep(5)

    def _loss_limit_monitor(self):
        """Monitor loss limits"""
        while self.monitoring:
            try:
                loss_data = asyncio.run(self.loss_monitor.check_loss_limits())

                # Evaluate session loss trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "Session Loss Limit":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            loss_data.get('session_pnl_percent', 0)
                        ))

                # Evaluate drawdown trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "Drawdown Limit":
                        asyncio.run(self._evaluate_trigger(
                            status,
                            -loss_data.get('drawdown_percent', 0)  # Negative for loss
                        ))

                time.sleep(20)  # Check every 20 seconds

            except Exception as e:
                logger.error(f"Loss limit monitor error: {e}")
                time.sleep(5)

    def _network_monitor(self):
        """Monitor network connectivity"""
        while self.monitoring:
            try:
                network_data = asyncio.run(self.network_monitor.check_connectivity())

                # Evaluate network connectivity trigger
                for status in self.trigger_statuses.values():
                    if status.condition.name == "Network Connectivity":
                        # Use ping time or large value if disconnected
                        ping_value = network_data.get('response_time_ms', 9999) if network_data.get('connected') else 9999
                        asyncio.run(self._evaluate_trigger(status, ping_value))

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Network monitor error: {e}")
                time.sleep(10)

    def _trigger_evaluator(self):
        """General trigger evaluation and maintenance"""
        while self.monitoring:
            try:
                current_time = time.time()

                # Reset triggers that haven't fired recently
                for status in self.trigger_statuses.values():
                    if (status.triggered and
                        status.last_trigger_time and
                        current_time - status.last_trigger_time > status.condition.time_window_seconds):

                        status.triggered = False
                        status.consecutive_failures = 0
                        logger.info(f"Reset trigger: {status.condition.name}")

                time.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Trigger evaluator error: {e}")
                time.sleep(30)

    def get_trigger_status(self) -> Dict[str, Any]:
        """Get current status of all triggers"""
        status_data = {}

        for name, status in self.trigger_statuses.items():
            status_data[name] = {
                'condition': asdict(status.condition),
                'current_value': status.current_value,
                'last_check': status.last_check,
                'failure_count': status.failure_count,
                'consecutive_failures': status.consecutive_failures,
                'triggered': status.triggered,
                'last_trigger_time': status.last_trigger_time,
                'time_since_last_check': time.time() - status.last_check if status.last_check > 0 else 0
            }

        return {
            'monitoring': self.monitoring,
            'trigger_count': len(self.trigger_conditions),
            'active_triggers': sum(1 for s in self.trigger_statuses.values() if s.triggered),
            'triggers': status_data
        }

if __name__ == '__main__':
    # Test multi-trigger system
    async def test_system():
        from unittest.mock import Mock

        # Mock broker
        broker = Mock()
        broker.get_account = Mock(return_value=Mock(equity=10000))
        broker.get_positions = Mock(return_value=[])

        # Mock kill switch
        kill_switch = Mock()

        config = {
            'triggers': {
                'loss_limits': {
                    'session_loss_percent': -2.0,
                    'drawdown_percent': -1.5
                },
                'position_limits': {
                    'max_exposure_ratio': 0.8,
                    'max_single_position_ratio': 0.3
                },
                'api_failure': {
                    'response_timeout': 3000,
                    'connection_timeout': 20
                }
            }
        }

        multi_trigger = MultiTriggerSystem(kill_switch, broker, config)

        print("Multi-trigger system test:")
        print(f"Loaded {len(multi_trigger.trigger_conditions)} trigger conditions")

        for condition in multi_trigger.trigger_conditions:
            print(f"- {condition.name}: {condition.threshold_value} ({condition.severity.value})")

        # Test status
        status = multi_trigger.get_trigger_status()
        print(f"Status: {status}")

    asyncio.run(test_system())