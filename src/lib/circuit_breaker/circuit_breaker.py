"""
Generic Circuit Breaker with Exponential Backoff

A reusable circuit breaker pattern implementation for async operations.
Provides fault tolerance and graceful degradation for external service calls.

LEGO Component: Imports shared types from library.common.types

States:
- CLOSED: Normal operation, all calls go through
- OPEN: Failures exceeded threshold, fast-fail mode
- HALF_OPEN: Testing recovery, allowing limited calls through

Usage:
    from library.components.utilities.circuit_breaker import (
        CircuitBreaker, CircuitBreakerConfig
    )
    from library.common.types import ValidationResult

    config = CircuitBreakerConfig(failure_threshold=5, timeout_duration=60)
    breaker = CircuitBreaker("api_client", config)

    try:
        result = await breaker.call(fetch_data, arg1, arg2)
    except CircuitBreakerError as e:
        print(f"Circuit open, retry after {e.retry_after}s")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, ParamSpec, TypeVar

# LEGO Import: Shared types from library.common.types
from library.common.types import ValidationResult

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"        # Normal operation - requests flow through
    OPEN = "open"            # Circuit tripped - blocking all requests
    HALF_OPEN = "half_open"  # Testing recovery - limited requests allowed


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocking calls."""

    def __init__(
        self,
        message: str,
        breaker_name: str,
        state: CircuitState,
        retry_after: Optional[float] = None
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state
        self.retry_after = retry_after


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures to trip circuit
        failure_rate_threshold: Failure rate (0.0-1.0) to trip
        success_threshold: Consecutive successes to close circuit
        failure_window_seconds: Time window for failure counting
        timeout_duration: Seconds to wait in OPEN state
        half_open_max_calls: Test calls allowed in HALF_OPEN
        exponential_backoff: Enable exponential backoff
        backoff_multiplier: Backoff multiplication factor
        max_backoff_seconds: Maximum backoff time
        min_requests_for_rate: Minimum requests before rate calculation
        name: Name for logging/monitoring
    """
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5
    success_threshold: int = 3
    failure_window_seconds: int = 60
    timeout_duration: int = 60
    half_open_max_calls: int = 3
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 300
    min_requests_for_rate: int = 10
    name: str = "default"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_trips: int = 0
    time_in_open_state: float = 0.0
    time_in_half_open_state: float = 0.0
    average_response_time: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_trip_time: Optional[datetime] = None
    last_recovery_time: Optional[datetime] = None


@dataclass
class CircuitBreakerStatus:
    """Status snapshot of circuit breaker state."""
    name: str
    state: CircuitState
    failure_count: int
    failure_threshold: int
    success_count: int
    success_threshold: int
    timeout_duration: int
    current_backoff: float
    last_failure_time: Optional[datetime]
    half_open_calls: int
    half_open_max_calls: int
    metrics: CircuitBreakerMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "success_count": self.success_count,
            "success_threshold": self.success_threshold,
            "timeout_duration": self.timeout_duration,
            "current_backoff": self.current_backoff,
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time else None
            ),
            "half_open_calls": self.half_open_calls,
            "half_open_max_calls": self.half_open_max_calls,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "circuit_trips": self.metrics.circuit_trips,
            "average_response_time": self.metrics.average_response_time,
        }

    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult for compatibility."""
        errors = []
        warnings = []

        if self.state == CircuitState.OPEN:
            errors.append(f"Circuit breaker '{self.name}' is OPEN")
        elif self.state == CircuitState.HALF_OPEN:
            warnings.append(f"Circuit breaker '{self.name}' is testing recovery")

        if self.failure_count > 0:
            warnings.append(f"Recent failures: {self.failure_count}")

        return ValidationResult(
            valid=self.state != CircuitState.OPEN,
            errors=errors,
            warnings=warnings,
            metadata=self.to_dict()
        )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation with exponential backoff.

    Protects against cascading failures by:
    - Tracking consecutive failures
    - Opening circuit after threshold exceeded
    - Fast-failing when circuit is open
    - Gradually testing recovery in half-open state
    - Using exponential backoff for recovery attempts

    Usage:
        config = CircuitBreakerConfig(failure_threshold=3, timeout_duration=30)
        breaker = CircuitBreaker("my_service", config)

        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        try:
            result = await breaker.call(fetch_data)
        except CircuitBreakerError as e:
            print(f"Circuit open, retry after {e.retry_after}s")
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            config: Configuration options (uses defaults if None)
        """
        self._name = name
        self._config = config or CircuitBreakerConfig(name=name)
        if self._config.name == "default":
            self._config.name = name

        # State management
        self._state = CircuitState.CLOSED
        self._state_changed_time = datetime.now(timezone.utc)
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Backoff tracking
        self._backoff_count = 0
        self._current_backoff = float(self._config.timeout_duration)

        # Metrics
        self._metrics = CircuitBreakerMetrics()
        self._recent_failures: deque = deque()
        self._request_times: deque = deque(maxlen=100)

        # Callbacks
        self._trip_callbacks: List[Callable] = []
        self._recovery_callbacks: List[Callable] = []

        logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._name

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration."""
        return self._config

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (fast-fail mode)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def call(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> T:
        """
        Execute function call through circuit breaker.

        Args:
            func: Async or sync function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function call

        Raises:
            CircuitBreakerError: If circuit is open or half-open limit reached
            Exception: Original exception from function (after recording failure)
        """
        start_time = time.time()

        async with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                retry_after = self._calculate_retry_after()
                logger.warning(
                    f"Circuit breaker '{self._name}' is OPEN, fast-failing"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self._name}' is OPEN. "
                    f"Will retry after {retry_after:.1f}s",
                    breaker_name=self._name,
                    state=self._state,
                    retry_after=retry_after
                )

            if (
                self._state == CircuitState.HALF_OPEN
                and self._half_open_calls >= self._config.half_open_max_calls
            ):
                logger.warning(
                    f"Circuit breaker '{self._name}' HALF_OPEN call limit reached"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self._name}' HALF_OPEN call limit reached",
                    breaker_name=self._name,
                    state=self._state,
                    retry_after=None
                )
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            response_time = time.time() - start_time
            await self._on_success(response_time)

            return result

        except CircuitBreakerError:
            raise  # Re-raise circuit breaker errors
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            await self._on_failure(str(e), response_time)
            raise

    def _check_state_transition(self) -> None:
        """Check if circuit should transition states based on timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            time_since_failure = datetime.now(timezone.utc) - self._last_failure_time
            if time_since_failure.total_seconds() >= self._current_backoff:
                logger.info(
                    f"Circuit breaker '{self._name}' transitioning "
                    "from OPEN to HALF_OPEN"
                )
                self._transition_to_half_open()

    def _calculate_retry_after(self) -> float:
        """Calculate seconds until retry is allowed."""
        if not self._last_failure_time:
            return self._current_backoff

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return max(0, self._current_backoff - elapsed)

    async def _on_success(self, response_time: float) -> None:
        """Handle successful call."""
        should_execute_callbacks = False

        async with self._lock:
            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._update_response_time(response_time)

            # Clean old failures
            self._clean_old_failures()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to_closed()
                    should_execute_callbacks = True

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

        if should_execute_callbacks:
            await self._execute_recovery_callbacks()

    async def _on_failure(self, error_message: str, response_time: float) -> None:
        """Handle failed call."""
        should_trip = False

        async with self._lock:
            current_time = datetime.now(timezone.utc)

            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = current_time
            self._last_failure_time = current_time
            self._update_response_time(response_time)

            # Track recent failures
            self._recent_failures.append(current_time)
            self._clean_old_failures()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens circuit
                self._transition_to_open()
                should_trip = True
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._should_trip():
                    should_trip = True

        if should_trip:
            await self._trip_circuit()

    def _should_trip(self) -> bool:
        """Determine if circuit should trip based on failures."""
        if self._state != CircuitState.CLOSED:
            return False

        # Check failure count threshold
        if self._failure_count >= self._config.failure_threshold:
            return True

        # Check failure rate threshold
        recent_count = len(self._recent_failures)
        total_recent = self._metrics.total_requests
        if (total_recent >= self._config.min_requests_for_rate and total_recent > 0):
            failure_rate = recent_count / total_recent
            if failure_rate >= self._config.failure_rate_threshold:
                return True

        return False

    async def _trip_circuit(self) -> None:
        """Trip the circuit breaker to open state."""
        logger.warning(f"Circuit breaker '{self._name}' tripping to OPEN state")

        async with self._lock:
            self._transition_to_open()
            self._metrics.circuit_trips += 1
            self._metrics.last_trip_time = datetime.now(timezone.utc)

            # Calculate backoff with exponential increase
            if self._config.exponential_backoff:
                self._current_backoff = min(
                    self._config.timeout_duration * (self._config.backoff_multiplier ** self._backoff_count),
                    self._config.max_backoff_seconds
                )
                self._backoff_count += 1
            else:
                self._current_backoff = float(self._config.timeout_duration)

        await self._execute_trip_callbacks()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        if self._state != CircuitState.OPEN:
            old_state = self._state
            now = datetime.now(timezone.utc)

            # Update time tracking
            if old_state == CircuitState.HALF_OPEN:
                time_in_half_open = (now - self._state_changed_time).total_seconds()
                self._metrics.time_in_half_open_state += time_in_half_open

            self._state = CircuitState.OPEN
            self._state_changed_time = now
            self._success_count = 0

            logger.info(f"Circuit breaker '{self._name}' -> OPEN (backoff: {self._current_backoff}s)")

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        if self._state != CircuitState.HALF_OPEN:
            old_state = self._state
            now = datetime.now(timezone.utc)

            # Update time tracking
            if old_state == CircuitState.OPEN:
                time_in_open = (now - self._state_changed_time).total_seconds()
                self._metrics.time_in_open_state += time_in_open

            self._state = CircuitState.HALF_OPEN
            self._state_changed_time = now
            self._success_count = 0
            self._half_open_calls = 0

            logger.info(f"Circuit breaker '{self._name}' -> HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state (recovered)."""
        if self._state != CircuitState.CLOSED:
            old_state = self._state
            now = datetime.now(timezone.utc)

            # Update time tracking
            if old_state == CircuitState.HALF_OPEN:
                time_in_half_open = (now - self._state_changed_time).total_seconds()
                self._metrics.time_in_half_open_state += time_in_half_open
            elif old_state == CircuitState.OPEN:
                time_in_open = (now - self._state_changed_time).total_seconds()
                self._metrics.time_in_open_state += time_in_open

            self._state = CircuitState.CLOSED
            self._state_changed_time = now
            self._success_count = 0
            self._failure_count = 0
            self._backoff_count = 0
            self._current_backoff = float(self._config.timeout_duration)
            self._metrics.last_recovery_time = now

            logger.info(f"Circuit breaker '{self._name}' -> CLOSED (recovered)")

    def _clean_old_failures(self) -> None:
        """Remove failures outside the failure window."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=self._config.failure_window_seconds)

        while self._recent_failures and self._recent_failures[0] < window_start:
            self._recent_failures.popleft()

    def _update_response_time(self, response_time: float) -> None:
        """Update rolling average response time."""
        self._request_times.append(response_time)
        if self._request_times:
            self._metrics.average_response_time = sum(self._request_times) / len(self._request_times)

    def register_trip_callback(self, callback: Callable[[str, CircuitState], None]) -> None:
        """
        Register callback for when circuit trips.

        Args:
            callback: Function called with (breaker_name, state) when tripped
        """
        self._trip_callbacks.append(callback)

    def register_recovery_callback(self, callback: Callable[[str, CircuitState], None]) -> None:
        """
        Register callback for when circuit recovers.

        Args:
            callback: Function called with (breaker_name, state) on recovery
        """
        self._recovery_callbacks.append(callback)

    async def _execute_trip_callbacks(self) -> None:
        """Execute all registered trip callbacks."""
        for callback in self._trip_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._name, self._state)
                else:
                    callback(self._name, self._state)
            except Exception as e:
                logger.error(f"Error executing trip callback: {e}")

    async def _execute_recovery_callbacks(self) -> None:
        """Execute all registered recovery callbacks."""
        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._name, self._state)
                else:
                    callback(self._name, self._state)
            except Exception as e:
                logger.error(f"Error executing recovery callback: {e}")

    def get_status(self) -> CircuitBreakerStatus:
        """
        Get circuit breaker status for monitoring.

        Returns:
            CircuitBreakerStatus with current state snapshot
        """
        return CircuitBreakerStatus(
            name=self._name,
            state=self._state,
            failure_count=self._failure_count,
            failure_threshold=self._config.failure_threshold,
            success_count=self._success_count,
            success_threshold=self._config.success_threshold,
            timeout_duration=self._config.timeout_duration,
            current_backoff=self._current_backoff,
            last_failure_time=self._last_failure_time,
            half_open_calls=self._half_open_calls,
            half_open_max_calls=self._config.half_open_max_calls,
            metrics=self._metrics
        )

    async def force_open(self) -> None:
        """Force circuit breaker to open state (manual intervention)."""
        logger.warning(f"Force opening circuit breaker '{self._name}'")
        async with self._lock:
            self._transition_to_open()
        await self._execute_trip_callbacks()

    async def force_close(self) -> None:
        """Force circuit breaker to closed state (manual intervention)."""
        logger.info(f"Force closing circuit breaker '{self._name}'")
        async with self._lock:
            self._transition_to_closed()
        await self._execute_recovery_callbacks()

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        logger.info(f"Circuit breaker '{self._name}' reset")
        self._state = CircuitState.CLOSED
        self._state_changed_time = datetime.now(timezone.utc)
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._backoff_count = 0
        self._current_backoff = float(self._config.timeout_duration)
        self._metrics = CircuitBreakerMetrics()
        self._recent_failures.clear()
        self._request_times.clear()


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized creation, monitoring, and coordination
    of circuit breakers across the application.

    Usage:
        manager = CircuitBreakerManager()

        # Create breakers
        api_breaker = manager.create("broker_api")
        db_breaker = manager.create("database", CircuitBreakerConfig(failure_threshold=3))

        # Get system-wide status
        status = manager.get_system_status()
        print(f"Open breakers: {status['open_breakers']}")
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        logger.info("Circuit Breaker Manager initialized")

    async def create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Create and register a new circuit breaker.

        Args:
            name: Unique name for the breaker
            config: Optional configuration

        Returns:
            The created CircuitBreaker instance

        Raises:
            ValueError: If a breaker with this name already exists
        """
        async with self._lock:
            if name in self._breakers:
                raise ValueError(f"Circuit breaker '{name}' already exists")

            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker

            logger.info(f"Created circuit breaker: {name}")
            return breaker

    async def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        async with self._lock:
            return self._breakers.get(name)

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing breaker or create new one."""
        async with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            breaker = CircuitBreaker(name, config)
            self._breakers[name] = breaker
            return breaker

    async def remove(self, name: str) -> bool:
        """Remove a circuit breaker by name."""
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                return True
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        statuses = {}
        for name, breaker in self._breakers.items():
            statuses[name] = breaker.get_status().to_dict()

        total = len(self._breakers)
        open_count = sum(1 for b in self._breakers.values() if b.is_open)
        half_open_count = sum(1 for b in self._breakers.values() if b.is_half_open)

        return {
            "total_circuit_breakers": total,
            "open_breakers": open_count,
            "half_open_breakers": half_open_count,
            "closed_breakers": total - open_count - half_open_count,
            "circuit_breakers": statuses
        }

    def get_open_breakers(self) -> List[str]:
        """Get list of names of currently open circuit breakers."""
        return [name for name, b in self._breakers.items() if b.is_open]

    def any_open(self) -> bool:
        """Check if any circuit breakers are currently open."""
        return any(b.is_open for b in self._breakers.values())

    async def shutdown_all(self) -> None:
        """Shutdown all circuit breakers gracefully."""
        logger.info("Shutting down all circuit breakers")
        for breaker in self._breakers.values():
            await breaker.force_close()
        logger.info("All circuit breakers shut down")

    def reset_all(self) -> None:
        """Reset all circuit breakers to initial state."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
