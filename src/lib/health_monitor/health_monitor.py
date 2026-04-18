"""
Async Health Monitor with Alerting

A reusable health monitoring component for polling service health
and triggering alerts when thresholds are exceeded.

LEGO Component: Uses httpx for async HTTP operations

Usage:
    from library.components.utilities.health_monitor import (
        HealthMonitor, HealthCheckConfig, HealthStatus
    )

    monitor = HealthMonitor()

    # Add HTTP health checks
    monitor.add_http_check("api", "https://api.example.com/health")

    # Add custom checks
    monitor.add_check("database", db_health_check)

    # Run health checks
    status = await monitor.check_all()
    if not status.healthy:
        print(f"Unhealthy services: {status.unhealthy_services}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional httpx import for HTTP health checks
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - HTTP health checks disabled. Install with: pip install httpx")


class HealthState(Enum):
    """Health states for services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    state: HealthState
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class HealthStatus:
    """Overall health status of the system."""
    healthy: bool
    state: HealthState
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def healthy_services(self) -> List[str]:
        """Get names of healthy services."""
        return [c.name for c in self.checks if c.state == HealthState.HEALTHY]

    @property
    def unhealthy_services(self) -> List[str]:
        """Get names of unhealthy services."""
        return [c.name for c in self.checks if c.state == HealthState.UNHEALTHY]

    @property
    def degraded_services(self) -> List[str]:
        """Get names of degraded services."""
        return [c.name for c in self.checks if c.state == HealthState.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "state": self.state.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total": len(self.checks),
                "healthy": len(self.healthy_services),
                "degraded": len(self.degraded_services),
                "unhealthy": len(self.unhealthy_services),
            }
        }


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""
    name: str
    check_fn: Optional[Callable[[], Union[bool, HealthCheckResult]]] = None
    url: Optional[str] = None
    timeout_seconds: float = 5.0
    expected_status_codes: List[int] = field(default_factory=lambda: [200])
    degraded_threshold_ms: float = 1000.0  # Response time for degraded state
    unhealthy_threshold_ms: float = 5000.0  # Response time for unhealthy state
    retry_count: int = 1
    retry_delay_seconds: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for health alerts."""
    enabled: bool = True
    alert_fn: Optional[Callable[[str, HealthState, str], None]] = None
    consecutive_failures_threshold: int = 2
    cooldown_seconds: float = 300.0  # Time between alerts for same service


class HealthMonitor:
    """
    Async health monitor with polling and alerting.

    Provides:
    - HTTP health checks with configurable timeouts
    - Custom health check functions
    - Consecutive failure tracking
    - Alert callbacks with cooldown
    - Periodic background monitoring

    Usage:
        monitor = HealthMonitor()

        # Add checks
        monitor.add_http_check("api", "https://api.example.com/health")
        monitor.add_check("database", async_db_check)

        # Run once
        status = await monitor.check_all()

        # Or run continuously
        await monitor.start_monitoring(interval_seconds=30)
    """

    def __init__(self, alert_config: Optional[AlertConfig] = None):
        """
        Initialize health monitor.

        Args:
            alert_config: Optional alert configuration
        """
        self._checks: Dict[str, HealthCheckConfig] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._alert_config = alert_config or AlertConfig()
        self._consecutive_failures: Dict[str, int] = {}
        self._last_alert_time: Dict[str, datetime] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("Health Monitor initialized")

    def add_check(
        self,
        name: str,
        check_fn: Callable[[], Union[bool, HealthCheckResult]],
        timeout_seconds: float = 5.0,
        **kwargs: Any
    ) -> None:
        """
        Add a custom health check function.

        Args:
            name: Unique name for this check
            check_fn: Async or sync function returning bool or HealthCheckResult
            timeout_seconds: Maximum time to wait for check
            **kwargs: Additional metadata
        """
        config = HealthCheckConfig(
            name=name,
            check_fn=check_fn,
            timeout_seconds=timeout_seconds,
            metadata=kwargs
        )
        self._checks[name] = config
        logger.info(f"Added health check: {name}")

    def add_http_check(
        self,
        name: str,
        url: str,
        timeout_seconds: float = 5.0,
        expected_status_codes: Optional[List[int]] = None,
        degraded_threshold_ms: float = 1000.0,
        unhealthy_threshold_ms: float = 5000.0,
        **kwargs: Any
    ) -> None:
        """
        Add an HTTP health check.

        Args:
            name: Unique name for this check
            url: URL to check
            timeout_seconds: Request timeout
            expected_status_codes: Valid status codes (default: [200])
            degraded_threshold_ms: Response time threshold for degraded state
            unhealthy_threshold_ms: Response time threshold for unhealthy state
            **kwargs: Additional metadata
        """
        if not HTTPX_AVAILABLE:
            logger.error(f"Cannot add HTTP check '{name}': httpx not available")
            return

        config = HealthCheckConfig(
            name=name,
            url=url,
            timeout_seconds=timeout_seconds,
            expected_status_codes=expected_status_codes or [200],
            degraded_threshold_ms=degraded_threshold_ms,
            unhealthy_threshold_ms=unhealthy_threshold_ms,
            metadata=kwargs
        )
        self._checks[name] = config
        logger.info(f"Added HTTP health check: {name} -> {url}")

    def remove_check(self, name: str) -> bool:
        """
        Remove a health check.

        Args:
            name: Name of check to remove

        Returns:
            True if check was removed, False if not found
        """
        if name in self._checks:
            del self._checks[name]
            self._results.pop(name, None)
            self._consecutive_failures.pop(name, None)
            logger.info(f"Removed health check: {name}")
            return True
        return False

    async def check(self, name: str) -> HealthCheckResult:
        """
        Run a single health check.

        Args:
            name: Name of check to run

        Returns:
            HealthCheckResult with state and details
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                state=HealthState.UNKNOWN,
                message=f"Health check '{name}' not found"
            )

        config = self._checks[name]
        result = await self._execute_check(config)
        self._results[name] = result
        await self._handle_result(result)
        return result

    async def check_all(self) -> HealthStatus:
        """
        Run all health checks concurrently.

        Returns:
            HealthStatus with overall state and individual results
        """
        if not self._checks:
            return HealthStatus(
                healthy=True,
                state=HealthState.HEALTHY,
                checks=[]
            )

        # Run all checks concurrently
        tasks = [self._execute_check(config) for config in self._checks.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        check_results = []
        for config, result in zip(self._checks.values(), results):
            if isinstance(result, Exception):
                result = HealthCheckResult(
                    name=config.name,
                    state=HealthState.UNHEALTHY,
                    message=f"Check error: {str(result)}"
                )
            check_results.append(result)
            self._results[config.name] = result
            await self._handle_result(result)

        # Calculate overall state
        overall_state = self._calculate_overall_state(check_results)
        healthy = overall_state == HealthState.HEALTHY

        return HealthStatus(
            healthy=healthy,
            state=overall_state,
            checks=check_results
        )

    async def _execute_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Execute a single health check with retry logic."""
        if not config.url and not config.check_fn:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNKNOWN,
                message="No check function or URL configured"
            )

        last_result = None
        check_fn = self._execute_http_check if config.url else self._execute_function_check
        for attempt in range(config.retry_count):
            try:
                result = await check_fn(config)

                # Return on success or last attempt
                if result.state == HealthState.HEALTHY or attempt == config.retry_count - 1:
                    return result

                last_result = result
                await asyncio.sleep(config.retry_delay_seconds)

            except Exception as e:
                last_result = HealthCheckResult(
                    name=config.name,
                    state=HealthState.UNHEALTHY,
                    message=f"Check failed: {str(e)}"
                )
                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay_seconds)

        return last_result or HealthCheckResult(
            name=config.name,
            state=HealthState.UNKNOWN,
            message="No check result"
        )

    async def _execute_http_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Execute an HTTP health check."""
        if not HTTPX_AVAILABLE or not config.url:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNKNOWN,
                message="httpx not available or URL not configured"
            )

        start_time = asyncio.get_event_loop().time()
        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.get(config.url)
                response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                # Check status code
                if response.status_code not in config.expected_status_codes:
                    return HealthCheckResult(
                        name=config.name,
                        state=HealthState.UNHEALTHY,
                        message=f"Unexpected status code: {response.status_code}",
                        response_time_ms=response_time_ms,
                        metadata={"status_code": response.status_code}
                    )

                # Determine state based on response time
                if response_time_ms >= config.unhealthy_threshold_ms:
                    state = HealthState.UNHEALTHY
                    message = f"Response too slow: {response_time_ms:.0f}ms"
                elif response_time_ms >= config.degraded_threshold_ms:
                    state = HealthState.DEGRADED
                    message = f"Response slow: {response_time_ms:.0f}ms"
                else:
                    state = HealthState.HEALTHY
                    message = f"OK ({response_time_ms:.0f}ms)"

                return HealthCheckResult(
                    name=config.name,
                    state=state,
                    message=message,
                    response_time_ms=response_time_ms,
                    metadata={"status_code": response.status_code, "url": config.url}
                )

        except httpx.TimeoutException:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNHEALTHY,
                message=f"Request timed out after {config.timeout_seconds}s",
                response_time_ms=config.timeout_seconds * 1000
            )
        except httpx.ConnectError as e:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNHEALTHY,
                message=f"Connection error: {str(e)}"
            )
        except Exception as e:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNHEALTHY,
                message=f"HTTP check failed: {str(e)}"
            )

    async def _execute_function_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Execute a function-based health check."""
        if not config.check_fn:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNKNOWN,
                message="No check function configured"
            )

        start_time = asyncio.get_event_loop().time()
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(config.check_fn):
                result = await asyncio.wait_for(
                    config.check_fn(),
                    timeout=config.timeout_seconds
                )
            else:
                result = config.check_fn()

            response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Handle different return types
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time_ms
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=config.name,
                    state=HealthState.HEALTHY if result else HealthState.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    response_time_ms=response_time_ms
                )
            else:
                return HealthCheckResult(
                    name=config.name,
                    state=HealthState.HEALTHY,
                    message=str(result),
                    response_time_ms=response_time_ms
                )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNHEALTHY,
                message=f"Check timed out after {config.timeout_seconds}s"
            )
        except Exception as e:
            return HealthCheckResult(
                name=config.name,
                state=HealthState.UNHEALTHY,
                message=f"Check error: {str(e)}"
            )

    async def _handle_result(self, result: HealthCheckResult) -> None:
        """Handle a check result, updating counters and triggering alerts."""
        name = result.name

        if result.state != HealthState.UNHEALTHY:
            # Reset failure counter on non-unhealthy state
            self._consecutive_failures[name] = 0
            return

        self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1

        # Check if alert should be triggered
        if self._should_alert(name):
            await self._send_alert(name, result.state, result.message)

    def _should_alert(self, name: str) -> bool:
        """Check if an alert should be triggered for a service."""
        if not self._alert_config.enabled:
            return False

        if not self._alert_config.alert_fn:
            return False

        failures = self._consecutive_failures.get(name, 0)
        if failures < self._alert_config.consecutive_failures_threshold:
            return False

        # Check cooldown
        last_alert = self._last_alert_time.get(name)
        if not last_alert:
            return True
        elapsed = (datetime.now(timezone.utc) - last_alert).total_seconds()
        return elapsed >= self._alert_config.cooldown_seconds

    async def _send_alert(self, name: str, state: HealthState, message: str) -> None:
        """Send an alert for a service."""
        if not self._alert_config.alert_fn:
            return

        try:
            if asyncio.iscoroutinefunction(self._alert_config.alert_fn):
                await self._alert_config.alert_fn(name, state, message)
            else:
                self._alert_config.alert_fn(name, state, message)
            self._last_alert_time[name] = datetime.now(timezone.utc)
            logger.warning(f"Alert sent for {name}: {state.value} - {message}")

        except Exception as e:
            logger.error(f"Failed to send alert for {name}: {e}")

    def _calculate_overall_state(self, results: List[HealthCheckResult]) -> HealthState:
        """Calculate overall health state from individual results."""
        if not results:
            return HealthState.HEALTHY

        states = [r.state for r in results]

        if any(s == HealthState.UNHEALTHY for s in states):
            return HealthState.UNHEALTHY
        if any(s == HealthState.DEGRADED for s in states):
            return HealthState.DEGRADED
        if any(s == HealthState.UNKNOWN for s in states):
            return HealthState.UNKNOWN

        return HealthState.HEALTHY

    async def start_monitoring(
        self,
        interval_seconds: float = 30.0,
        on_status_change: Optional[Callable[[HealthStatus], None]] = None
    ) -> None:
        """
        Start background health monitoring.

        Args:
            interval_seconds: Time between health checks
            on_status_change: Callback for status changes
        """
        if self._running:
            logger.warning("Monitoring already running")
            return

        self._running = True
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")

        async def monitoring_loop():
            last_status: Optional[HealthStatus] = None
            while self._running:
                try:
                    status = await self.check_all()

                    # Notify on status change
                    if on_status_change:
                        if last_status is None or status.state != last_status.state:
                            try:
                                if asyncio.iscoroutinefunction(on_status_change):
                                    await on_status_change(status)
                                else:
                                    on_status_change(status)
                            except Exception as e:
                                logger.error(f"Status change callback error: {e}")

                    last_status = status
                    await asyncio.sleep(interval_seconds)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(interval_seconds)

        self._monitoring_task = asyncio.create_task(monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("Health monitoring stopped")

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get the last result for a specific check."""
        return self._results.get(name)

    def get_all_results(self) -> Dict[str, HealthCheckResult]:
        """Get all last results."""
        return self._results.copy()

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        results = list(self._results.values())
        overall_state = self._calculate_overall_state(results)

        return {
            "overall_state": overall_state.value,
            "healthy": overall_state == HealthState.HEALTHY,
            "total_checks": len(self._checks),
            "healthy_count": sum(1 for r in results if r.state == HealthState.HEALTHY),
            "degraded_count": sum(1 for r in results if r.state == HealthState.DEGRADED),
            "unhealthy_count": sum(1 for r in results if r.state == HealthState.UNHEALTHY),
            "unknown_count": sum(1 for r in results if r.state == HealthState.UNKNOWN),
            "checks": {name: result.to_dict() for name, result in self._results.items()}
        }
