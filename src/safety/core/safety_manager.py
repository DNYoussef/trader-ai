"""
Safety Manager - Central coordination for all safety systems.

This module implements the core safety management infrastructure for the
Gary×Taleb trading system, providing fail-safe coordination across all
system components with zero single points of failure.

Key Features:
- Central safety state management
- Component health monitoring
- Emergency shutdown coordination
- Recovery orchestration
- Safety metrics collection
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyState(Enum):
    """Overall system safety states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    RECOVERY = "recovery"


class ComponentState(Enum):
    """Individual component health states"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


@dataclass
class SafetyMetrics:
    """Safety system performance metrics"""
    uptime_seconds: float = 0.0
    availability_percentage: float = 0.0
    mean_time_to_recovery: float = 0.0
    circuit_breaker_trips: int = 0
    emergency_shutdowns: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    last_health_check: Optional[datetime] = None

    def calculate_availability(self, total_time: float) -> float:
        """Calculate availability percentage"""
        if total_time <= 0:
            return 0.0
        return (self.uptime_seconds / total_time) * 100.0


@dataclass
class ComponentHealth:
    """Health status for system components"""
    component_id: str
    state: ComponentState
    last_heartbeat: datetime
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyEvent:
    """Safety-related system events"""
    timestamp: datetime
    event_type: str
    severity: str  # INFO, WARNING, CRITICAL, EMERGENCY
    component_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SafetyManager:
    """
    Central Safety Manager implementing fail-safe architecture.

    Responsibilities:
    - Monitor all system components for health
    - Coordinate emergency shutdowns
    - Orchestrate system recovery
    - Maintain safety metrics and audit trail
    - Trigger circuit breakers and failover mechanisms
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Safety Manager.

        Args:
            config: Safety configuration including thresholds, timeouts, etc.
        """
        self.config = config
        self.state = SafetyState.HEALTHY
        self.components: Dict[str, ComponentHealth] = {}
        self.safety_events: List[SafetyEvent] = []
        self.metrics = SafetyMetrics()

        # Thread-safe state management
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._recovery_event = threading.Event()

        # Health monitoring
        self.health_check_interval = config.get('health_check_interval', 5.0)  # seconds
        self.component_timeout = config.get('component_timeout', 30.0)  # seconds
        self.max_consecutive_failures = config.get('max_consecutive_failures', 3)

        # Safety thresholds
        self.degraded_threshold = config.get('degraded_threshold', 0.8)  # 80% components healthy
        self.critical_threshold = config.get('critical_threshold', 0.6)  # 60% components healthy

        # Recovery settings
        self.recovery_timeout = config.get('recovery_timeout', 60.0)  # seconds
        self.max_recovery_attempts = config.get('max_recovery_attempts', 3)

        # Event callbacks
        self._emergency_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []

        # Persistence
        self.data_dir = Path(config.get('data_dir', './data/safety'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Start monitoring
        self._monitoring_task = None
        self._start_time = time.time()

        logger.info("Safety Manager initialized with fail-safe architecture")

    async def start(self) -> bool:
        """Start safety monitoring systems."""
        try:
            with self._lock:
                self._shutdown_event.clear()
                self._start_time = time.time()

                # Start health monitoring
                self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())

                self.state = SafetyState.HEALTHY
                await self._log_safety_event("SYSTEM_START", "INFO", "safety_manager",
                                            "Safety Manager started successfully")

                logger.info("Safety Manager monitoring started")
                return True

        except Exception as e:
            logger.error(f"Failed to start Safety Manager: {e}")
            return False

    async def shutdown(self, emergency: bool = False) -> bool:
        """
        Shutdown safety systems.

        Args:
            emergency: If True, perform emergency shutdown
        """
        try:
            with self._lock:
                if emergency:
                    self.state = SafetyState.EMERGENCY_SHUTDOWN
                    await self._log_safety_event("EMERGENCY_SHUTDOWN", "EMERGENCY",
                                                "safety_manager", "Emergency shutdown initiated")
                    self.metrics.emergency_shutdowns += 1
                else:
                    await self._log_safety_event("GRACEFUL_SHUTDOWN", "INFO",
                                                "safety_manager", "Graceful shutdown initiated")

                self._shutdown_event.set()

                # Wait for monitoring to stop
                if self._monitoring_task:
                    try:
                        await asyncio.wait_for(self._monitoring_task, timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning("Health monitoring task did not stop gracefully")

                # Execute emergency callbacks
                if emergency:
                    await self._execute_emergency_callbacks()

                # Save final metrics
                await self._save_metrics()

                logger.info(f"Safety Manager shutdown complete (emergency={emergency})")
                return True

        except Exception as e:
            logger.error(f"Error during safety manager shutdown: {e}")
            return False

    def register_component(self, component_id: str, initial_state: ComponentState = ComponentState.OPERATIONAL) -> bool:
        """Register a system component for monitoring."""
        try:
            with self._lock:
                self.components[component_id] = ComponentHealth(
                    component_id=component_id,
                    state=initial_state,
                    last_heartbeat=datetime.now()
                )

                logger.info(f"Component registered for safety monitoring: {component_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False

    async def update_component_health(self, component_id: str, state: ComponentState,
                                    error_msg: Optional[str] = None,
                                    metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Update health status for a component."""
        try:
            with self._lock:
                if component_id not in self.components:
                    logger.warning(f"Attempted to update unregistered component: {component_id}")
                    return False

                component = self.components[component_id]
                old_state = component.state

                # Update component health
                component.state = state
                component.last_heartbeat = datetime.now()

                if error_msg:
                    component.last_error = error_msg
                    component.error_count += 1
                    if state == ComponentState.FAILED:
                        component.consecutive_failures += 1
                else:
                    component.consecutive_failures = 0

                if metrics:
                    component.metrics.update(metrics)

                # Log state changes
                if old_state != state:
                    severity = "WARNING" if state == ComponentState.DEGRADED else \
                              "CRITICAL" if state == ComponentState.FAILED else "INFO"

                    await self._log_safety_event("COMPONENT_STATE_CHANGE", severity, component_id,
                                                f"Component state changed: {old_state.value} -> {state.value}",
                                                {"old_state": old_state.value, "new_state": state.value,
                                                 "error_msg": error_msg})

                # Check if component needs circuit breaker
                if component.consecutive_failures >= self.max_consecutive_failures:
                    await self._trigger_circuit_breaker(component_id)

                # Update overall system state
                await self._assess_system_health()

                return True

        except Exception as e:
            logger.error(f"Failed to update component health for {component_id}: {e}")
            return False

    async def heartbeat(self, component_id: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Receive heartbeat from component."""
        return await self.update_component_health(component_id, ComponentState.OPERATIONAL,
                                                 metrics=metrics)

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        with self._lock:
            healthy_components = sum(1 for c in self.components.values()
                                   if c.state == ComponentState.OPERATIONAL)
            total_components = len(self.components)
            health_percentage = (healthy_components / total_components * 100) if total_components > 0 else 0

            # Update metrics
            current_time = time.time()
            if self.state in [SafetyState.HEALTHY, SafetyState.DEGRADED]:
                self.metrics.uptime_seconds = current_time - self._start_time

            self.metrics.availability_percentage = self.metrics.calculate_availability(
                current_time - self._start_time
            )
            self.metrics.last_health_check = datetime.now()

            return {
                "system_state": self.state.value,
                "health_percentage": health_percentage,
                "healthy_components": healthy_components,
                "total_components": total_components,
                "uptime_seconds": self.metrics.uptime_seconds,
                "availability_percentage": self.metrics.availability_percentage,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "emergency_shutdowns": self.metrics.emergency_shutdowns,
                "component_details": {
                    comp_id: {
                        "state": comp.state.value,
                        "last_heartbeat": comp.last_heartbeat.isoformat(),
                        "error_count": comp.error_count,
                        "consecutive_failures": comp.consecutive_failures,
                        "last_error": comp.last_error
                    }
                    for comp_id, comp in self.components.items()
                }
            }

    def register_emergency_callback(self, callback: Callable) -> None:
        """Register callback for emergency situations."""
        self._emergency_callbacks.append(callback)
        logger.info("Emergency callback registered")

    def register_recovery_callback(self, callback: Callable) -> None:
        """Register callback for recovery procedures."""
        self._recovery_callbacks.append(callback)
        logger.info("Recovery callback registered")

    async def initiate_recovery(self, component_id: Optional[str] = None) -> bool:
        """
        Initiate recovery procedures.

        Args:
            component_id: Specific component to recover, or None for system-wide recovery
        """
        try:
            with self._lock:
                if self.state == SafetyState.RECOVERY:
                    logger.warning("Recovery already in progress")
                    return False

                old_state = self.state
                self.state = SafetyState.RECOVERY
                self._recovery_event.set()

            await self._log_safety_event("RECOVERY_INITIATED", "INFO",
                                        component_id or "system", "Recovery procedure initiated")

            # Execute recovery callbacks
            recovery_success = await self._execute_recovery_callbacks(component_id)

            if recovery_success:
                with self._lock:
                    self.state = SafetyState.HEALTHY
                    self.metrics.successful_recoveries += 1

                await self._log_safety_event("RECOVERY_SUCCESSFUL", "INFO",
                                            component_id or "system", "Recovery completed successfully")
                logger.info("System recovery completed successfully")
                return True
            else:
                with self._lock:
                    self.state = old_state  # Revert to previous state
                    self.metrics.failed_recoveries += 1

                await self._log_safety_event("RECOVERY_FAILED", "CRITICAL",
                                            component_id or "system", "Recovery procedure failed")
                logger.error("System recovery failed")
                return False

        except Exception as e:
            logger.error(f"Error during recovery initiation: {e}")
            with self._lock:
                self.metrics.failed_recoveries += 1
            return False
        finally:
            self._recovery_event.clear()

    # Private methods

    async def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        logger.info("Health monitoring loop started")

        try:
            while not self._shutdown_event.is_set():
                await self._check_component_timeouts()
                await self._assess_system_health()
                await asyncio.sleep(self.health_check_interval)

        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in health monitoring loop: {e}")
            await self._log_safety_event("MONITORING_ERROR", "CRITICAL", "safety_manager",
                                        f"Health monitoring error: {e}")

    async def _check_component_timeouts(self):
        """Check for component timeouts."""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.component_timeout)

        with self._lock:
            for component_id, component in self.components.items():
                if (current_time - component.last_heartbeat) > timeout_threshold:
                    if component.state != ComponentState.FAILED:
                        await self.update_component_health(
                            component_id,
                            ComponentState.FAILED,
                            f"Component timeout: {self.component_timeout}s"
                        )

    async def _assess_system_health(self):
        """Assess overall system health and update state."""
        with self._lock:
            if not self.components:
                return

            healthy_count = sum(1 for c in self.components.values()
                              if c.state == ComponentState.OPERATIONAL)
            total_count = len(self.components)
            health_ratio = healthy_count / total_count

            old_state = self.state

            # Determine new state based on health ratio
            if health_ratio >= self.degraded_threshold:
                new_state = SafetyState.HEALTHY
            elif health_ratio >= self.critical_threshold:
                new_state = SafetyState.DEGRADED
            else:
                new_state = SafetyState.CRITICAL

            # Don't override recovery or emergency states
            if self.state not in [SafetyState.RECOVERY, SafetyState.EMERGENCY_SHUTDOWN]:
                self.state = new_state

            # Log state changes
            if old_state != self.state:
                severity = "WARNING" if self.state == SafetyState.DEGRADED else \
                          "CRITICAL" if self.state == SafetyState.CRITICAL else "INFO"

                await self._log_safety_event("SYSTEM_STATE_CHANGE", severity, "safety_manager",
                                            f"System state changed: {old_state.value} -> {self.state.value}",
                                            {"health_ratio": health_ratio,
                                             "healthy_components": healthy_count,
                                             "total_components": total_count})

    async def _trigger_circuit_breaker(self, component_id: str):
        """Trigger circuit breaker for component."""
        self.metrics.circuit_breaker_trips += 1

        await self._log_safety_event("CIRCUIT_BREAKER_TRIP", "CRITICAL", component_id,
                                    f"Circuit breaker triggered for component: {component_id}")

        logger.critical(f"Circuit breaker triggered for component: {component_id}")

    async def _execute_emergency_callbacks(self):
        """Execute all registered emergency callbacks."""
        for callback in self._emergency_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error executing emergency callback: {e}")

    async def _execute_recovery_callbacks(self, component_id: Optional[str]) -> bool:
        """Execute recovery callbacks and return success status."""
        success = True

        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(component_id)
                else:
                    result = callback(component_id)

                if isinstance(result, bool) and not result:
                    success = False

            except Exception as e:
                logger.error(f"Error executing recovery callback: {e}")
                success = False

        return success

    async def _log_safety_event(self, event_type: str, severity: str, component_id: str,
                               message: str, details: Optional[Dict[str, Any]] = None):
        """Log a safety event."""
        event = SafetyEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            component_id=component_id,
            message=message,
            details=details or {}
        )

        self.safety_events.append(event)

        # Limit event history size
        if len(self.safety_events) > 10000:
            self.safety_events = self.safety_events[-5000:]

        # Log to standard logging
        log_level = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "CRITICAL": logging.CRITICAL,
            "EMERGENCY": logging.CRITICAL
        }.get(severity, logging.INFO)

        logger.log(log_level, f"[SAFETY] {event_type}: {message}")

    async def _save_metrics(self):
        """Save safety metrics to disk."""
        try:
            metrics_file = self.data_dir / 'safety_metrics.json'

            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": self.metrics.uptime_seconds,
                "availability_percentage": self.metrics.availability_percentage,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "emergency_shutdowns": self.metrics.emergency_shutdowns,
                "successful_recoveries": self.metrics.successful_recoveries,
                "failed_recoveries": self.metrics.failed_recoveries
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save safety metrics: {e}")