"""
Safety Integration Layer - Integration of safety systems with trading engine.

This module provides the integration layer that connects all safety systems
with the main trading engine, ensuring seamless operation and proper
coordination of safety mechanisms.

Key Features:
- Safety system lifecycle management
- Trading engine integration hooks
- Emergency callback registration
- Configuration management
- Health monitoring integration
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from .safety_manager import SafetyManager, ComponentState
from ..redundancy.failover_manager import FailoverManager
from ..circuit_breakers.circuit_breaker import CircuitBreakerManager, CircuitType, CircuitBreakerConfig
from ..recovery.recovery_manager import RecoveryManager, ComponentPriority
from ..monitoring.health_monitor import HealthMonitor, SystemResourceCollector, TradingSystemCollector

logger = logging.getLogger(__name__)


class TradingSafetyIntegration:
    """
    Integration layer for safety systems and trading engine.

    Coordinates all safety components and provides a unified interface
    for the trading system to interact with safety mechanisms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety integration.

        Args:
            config: Complete safety system configuration
        """
        self.config = config
        self.initialized = False
        self.running = False

        # Safety system components
        self.safety_manager: Optional[SafetyManager] = None
        self.failover_manager: Optional[FailoverManager] = None
        self.circuit_manager: Optional[CircuitBreakerManager] = None
        self.recovery_manager: Optional[RecoveryManager] = None
        self.health_monitor: Optional[HealthMonitor] = None

        # Trading engine reference
        self.trading_engine: Optional[Any] = None

        # Emergency callbacks
        self._emergency_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []

        logger.info("Safety Integration initialized")

    async def initialize(self, trading_engine: Any) -> bool:
        """
        Initialize all safety systems.

        Args:
            trading_engine: Reference to the main trading engine

        Returns:
            True if initialization successful
        """
        try:
            self.trading_engine = trading_engine
            logger.info("Initializing safety systems...")

            # Initialize core safety manager
            safety_config = self.config.get('safety_manager', {})
            self.safety_manager = SafetyManager(safety_config)

            # Initialize failover manager
            failover_config = self.config.get('failover_manager', {})
            self.failover_manager = FailoverManager(failover_config)

            # Initialize circuit breaker manager
            self.circuit_manager = CircuitBreakerManager()

            # Initialize recovery manager
            recovery_config = self.config.get('recovery_manager', {})
            self.recovery_manager = RecoveryManager(recovery_config)

            # Initialize health monitor
            health_config = self.config.get('health_monitor', {})
            self.health_monitor = HealthMonitor(health_config)

            # Register emergency callbacks
            await self._setup_emergency_callbacks()

            # Register trading system components
            await self._register_trading_components()

            # Setup circuit breakers
            await self._setup_circuit_breakers()

            # Setup health monitoring
            await self._setup_health_monitoring()

            self.initialized = True
            logger.info("Safety systems initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize safety systems: {e}")
            return False

    async def start(self) -> bool:
        """
        Start all safety systems.

        Returns:
            True if start successful
        """
        if not self.initialized:
            logger.error("Safety systems not initialized")
            return False

        try:
            logger.info("Starting safety systems...")

            # Start safety manager
            if not await self.safety_manager.start():
                raise Exception("Failed to start safety manager")

            # Start health monitor
            if not await self.health_monitor.start():
                raise Exception("Failed to start health monitor")

            # Create initial system snapshot
            if not await self.recovery_manager.create_system_snapshot("system_startup"):
                logger.warning("Failed to create initial system snapshot")

            self.running = True
            logger.info("Safety systems started successfully")

            # Log system status
            await self._log_system_status()

            return True

        except Exception as e:
            logger.error(f"Failed to start safety systems: {e}")
            await self.stop()
            return False

    async def stop(self) -> bool:
        """
        Stop all safety systems gracefully.

        Returns:
            True if stop successful
        """
        try:
            logger.info("Stopping safety systems...")

            if self.health_monitor:
                await self.health_monitor.stop()

            if self.failover_manager:
                await self.failover_manager.shutdown()

            if self.circuit_manager:
                await self.circuit_manager.shutdown_all()

            if self.safety_manager:
                await self.safety_manager.shutdown(emergency=False)

            self.running = False
            logger.info("Safety systems stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping safety systems: {e}")
            return False

    async def emergency_shutdown(self) -> bool:
        """
        Perform emergency shutdown of all systems.

        Returns:
            True if emergency shutdown successful
        """
        try:
            logger.critical("EMERGENCY SHUTDOWN INITIATED")

            # Stop trading engine immediately
            if self.trading_engine and hasattr(self.trading_engine, 'activate_kill_switch'):
                try:
                    self.trading_engine.activate_kill_switch()
                except Exception as e:
                    logger.error(f"Failed to activate trading engine kill switch: {e}")

            # Emergency shutdown safety manager
            if self.safety_manager:
                await self.safety_manager.shutdown(emergency=True)

            # Stop all other systems
            await self.stop()

            logger.critical("EMERGENCY SHUTDOWN COMPLETED")
            return True

        except Exception as e:
            logger.critical(f"EMERGENCY SHUTDOWN FAILED: {e}")
            return False

    async def system_recovery(self, checkpoint_id: Optional[str] = None) -> bool:
        """
        Initiate system recovery.

        Args:
            checkpoint_id: Specific checkpoint to recover from

        Returns:
            True if recovery successful
        """
        if not self.recovery_manager:
            logger.error("Recovery manager not available")
            return False

        try:
            logger.info(f"Initiating system recovery (checkpoint: {checkpoint_id})")

            # Create pre-recovery snapshot
            await self.recovery_manager.create_system_snapshot("pre_recovery")

            # Perform recovery
            success = await self.recovery_manager.recover_system(checkpoint_id)

            if success:
                logger.info("System recovery completed successfully")
                await self._log_system_status()
            else:
                logger.error("System recovery failed")

            return success

        except Exception as e:
            logger.error(f"Error during system recovery: {e}")
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        try:
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unknown',
                'safety_systems': {},
                'component_health': {}
            }

            # Safety manager status
            if self.safety_manager:
                safety_health = self.safety_manager.get_system_health()
                health_data['safety_systems']['safety_manager'] = safety_health
                health_data['overall_status'] = safety_health.get('system_state', 'unknown')

            # Health monitor status
            if self.health_monitor:
                monitor_health = self.health_monitor.get_system_health_summary()
                health_data['safety_systems']['health_monitor'] = monitor_health

            # Circuit breaker status
            if self.circuit_manager:
                circuit_status = self.circuit_manager.get_system_status()
                health_data['safety_systems']['circuit_breakers'] = circuit_status

            # Failover manager status
            if self.failover_manager:
                failover_metrics = self.failover_manager.get_failover_metrics()
                health_data['safety_systems']['failover'] = failover_metrics

            # Recovery manager status
            if self.recovery_manager:
                recovery_status = self.recovery_manager.get_recovery_status()
                health_data['safety_systems']['recovery'] = recovery_status

            return health_data

        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

    async def _setup_emergency_callbacks(self):
        """Setup emergency shutdown callbacks."""
        async def emergency_stop_trading():
            """Emergency callback to stop trading operations."""
            logger.critical("Emergency callback: Stopping trading operations")
            if self.trading_engine:
                try:
                    if hasattr(self.trading_engine, 'activate_kill_switch'):
                        self.trading_engine.activate_kill_switch()
                    if hasattr(self.trading_engine, 'stop'):
                        await self.trading_engine.stop()
                except Exception as e:
                    logger.error(f"Error stopping trading engine: {e}")

        async def emergency_close_positions():
            """Emergency callback to close all positions."""
            logger.critical("Emergency callback: Closing all positions")
            if (self.trading_engine and
                hasattr(self.trading_engine, 'portfolio_manager') and
                self.trading_engine.portfolio_manager):
                try:
                    portfolio = self.trading_engine.portfolio_manager
                    if hasattr(portfolio, 'close_all_positions'):
                        await portfolio.close_all_positions()
                except Exception as e:
                    logger.error(f"Error closing positions: {e}")

        # Register callbacks
        if self.safety_manager:
            self.safety_manager.register_emergency_callback(emergency_stop_trading)
            self.safety_manager.register_emergency_callback(emergency_close_positions)

    async def _register_trading_components(self):
        """Register trading system components with safety systems."""
        components = [
            ("trading_engine", ComponentPriority.HIGH),
            ("safety_manager", ComponentPriority.CRITICAL),
            ("broker_adapter", ComponentPriority.HIGH),
            ("portfolio_manager", ComponentPriority.HIGH),
            ("market_data", ComponentPriority.MEDIUM),
            ("trade_executor", ComponentPriority.HIGH),
        ]

        # Register with safety manager
        for comp_id, _ in components:
            self.safety_manager.register_component(comp_id, ComponentState.OPERATIONAL)

        # Register with recovery manager (if components implement RecoverableComponent)
        # This would be done when components are actually created
        logger.info("Trading components registered with safety systems")

    async def _setup_circuit_breakers(self):
        """Setup circuit breakers for trading operations."""
        # Trading loss circuit breaker
        loss_config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_rate_threshold=0.8,
            open_timeout_seconds=300,  # 5 minutes
            exponential_backoff=True
        )

        self.circuit_manager.create_circuit_breaker(
            name="trading_loss_protection",
            circuit_type=CircuitType.TRADING_LOSS,
            config=loss_config
        )

        # Connection circuit breaker
        connection_config = CircuitBreakerConfig(
            failure_threshold=5,
            failure_rate_threshold=0.5,
            open_timeout_seconds=60,
            exponential_backoff=True
        )

        self.circuit_manager.create_circuit_breaker(
            name="broker_connection_protection",
            circuit_type=CircuitType.CONNECTION_FAILURE,
            config=connection_config
        )

        # Performance circuit breaker
        performance_config = CircuitBreakerConfig(
            failure_threshold=10,
            failure_rate_threshold=0.3,
            open_timeout_seconds=30
        )

        self.circuit_manager.create_circuit_breaker(
            name="performance_protection",
            circuit_type=CircuitType.PERFORMANCE_LATENCY,
            config=performance_config
        )

        logger.info("Circuit breakers configured")

    async def _setup_health_monitoring(self):
        """Setup health monitoring collectors."""
        # System resource collector
        system_collector = SystemResourceCollector("system_resources")
        self.health_monitor.register_collector(system_collector)

        # Trading system collector
        if self.trading_engine:
            trading_collector = TradingSystemCollector("trading_engine", self.trading_engine)
            self.health_monitor.register_collector(trading_collector)

        # Setup alert callback
        async def handle_health_alert(alert):
            """Handle health monitoring alerts."""
            logger.warning(f"Health alert: {alert.severity.value} - {alert.message}")

            if alert.severity.value in ['critical', 'emergency']:
                # Update component status in safety manager
                await self.safety_manager.update_component_health(
                    alert.component_id,
                    ComponentState.FAILED,
                    alert.message
                )

        self.health_monitor.register_alert_callback(handle_health_alert)
        logger.info("Health monitoring configured")

    async def _log_system_status(self):
        """Log current system status."""
        health = self.get_system_health()
        logger.info(f"System Status: {health.get('overall_status', 'unknown')}")

        # Log component status
        safety_systems = health.get('safety_systems', {})
        for system_name, system_data in safety_systems.items():
            if isinstance(system_data, dict):
                status = system_data.get('system_state') or system_data.get('overall_status', 'unknown')
                logger.info(f"  {system_name}: {status}")

    # Context manager support for easy usage
    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        if exc_type is not None:
            logger.error(f"Exception in safety context: {exc_type.__name__}: {exc_val}")
            await self.emergency_shutdown()


class SafetyDecorator:
    """Decorator for protecting critical trading functions with circuit breakers."""

    def __init__(self, circuit_name: str, integration: TradingSafetyIntegration):
        """
        Initialize safety decorator.

        Args:
            circuit_name: Name of circuit breaker to use
            integration: Safety integration instance
        """
        self.circuit_name = circuit_name
        self.integration = integration

    def __call__(self, func):
        """Decorate function with circuit breaker protection."""
        async def wrapper(*args, **kwargs):
            if (self.integration.circuit_manager and
                self.integration.running):

                circuit = self.integration.circuit_manager.get_circuit_breaker(self.circuit_name)
                if circuit:
                    return await circuit.call(func, *args, **kwargs)

            # Fallback to direct call if circuit breaker unavailable
            return await func(*args, **kwargs)

        return wrapper


# Utility functions for easy integration

def create_safety_integration(config_path: str) -> TradingSafetyIntegration:
    """
    Create and configure safety integration from config file.

    Args:
        config_path: Path to safety configuration file

    Returns:
        Configured safety integration instance
    """
    import json

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load safety config from {config_path}: {e}")
        # Use default configuration
        config = get_default_safety_config()

    return TradingSafetyIntegration(config)


def get_default_safety_config() -> Dict[str, Any]:
    """Get default safety system configuration."""
    return {
        'safety_manager': {
            'health_check_interval': 5.0,
            'component_timeout': 30.0,
            'max_consecutive_failures': 3,
            'degraded_threshold': 0.8,
            'critical_threshold': 0.6,
            'recovery_timeout': 60.0,
            'data_dir': './data/safety'
        },
        'failover_manager': {
            'default_failure_threshold': 3,
            'default_switch_timeout': 30,
            'health_check_interval': 10
        },
        'recovery_manager': {
            'target_recovery_time': 60.0,
            'critical_recovery_time': 30.0,
            'state_dir': './data/recovery'
        },
        'health_monitor': {
            'alert_history_size': 1000,
            'metrics_retention_hours': 24,
            'anomaly_detection_enabled': True
        }
    }


async def initialize_trading_safety(trading_engine: Any, config_path: Optional[str] = None) -> TradingSafetyIntegration:
    """
    Convenience function to initialize safety systems for trading engine.

    Args:
        trading_engine: Trading engine instance
        config_path: Optional path to safety configuration

    Returns:
        Initialized and started safety integration

    Raises:
        Exception: If initialization or startup fails
    """
    if config_path:
        integration = create_safety_integration(config_path)
    else:
        config = get_default_safety_config()
        integration = TradingSafetyIntegration(config)

    if not await integration.initialize(trading_engine):
        raise Exception("Failed to initialize safety systems")

    if not await integration.start():
        raise Exception("Failed to start safety systems")

    logger.info("Trading safety systems ready")
    return integration