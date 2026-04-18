"""
Circuit Breaker Implementation for Memory MCP
Provides fault tolerance and graceful degradation for external service calls

States:
- CLOSED: Normal operation, all calls go through
- OPEN: Failures exceeded threshold, fast-fail mode
- HALF_OPEN: Testing recovery, allowing limited calls through

Example:
    >>> breaker = CircuitBreaker(failure_threshold=3, timeout_duration=30)
    >>> result = await breaker.call(async_function, arg1, arg2)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocking calls"""

    def __init__(self, message: str, state: CircuitBreakerState, retry_after: Optional[float] = None):
        super().__init__(message)
        self.state = state
        self.retry_after = retry_after


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for CircuitBreaker

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit
        timeout_duration: Seconds to wait before transitioning from OPEN to HALF_OPEN
        half_open_max_calls: Number of test calls allowed in HALF_OPEN state
        name: Optional name for logging/monitoring
    """
    failure_threshold: int = 5
    timeout_duration: int = 30
    half_open_max_calls: int = 3
    name: str = "default"


@dataclass
class CircuitBreakerStatus:
    """Status snapshot of circuit breaker state"""
    state: CircuitBreakerState
    failure_count: int
    failure_threshold: int
    timeout_duration: int
    last_failure_time: Optional[datetime]
    half_open_calls: int
    half_open_max_calls: int
    name: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "timeout_duration": self.timeout_duration,
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time else None
            ),
            "half_open_calls": self.half_open_calls,
            "half_open_max_calls": self.half_open_max_calls,
            "name": self.name
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for async calls

    Provides fault tolerance by:
    - Tracking consecutive failures
    - Opening circuit after threshold exceeded
    - Fast-failing when circuit is open
    - Gradually testing recovery in half-open state

    Example:
        >>> config = CircuitBreakerConfig(failure_threshold=3, timeout_duration=30)
        >>> breaker = CircuitBreaker(config)
        >>>
        >>> async def fetch_data():
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()
        >>>
        >>> try:
        ...     result = await breaker.call(fetch_data)
        ... except CircuitBreakerError as e:
        ...     print(f"Circuit open, retry after {e.retry_after}s")
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker

        Args:
            config: CircuitBreakerConfig instance, uses defaults if None
        """
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get circuit breaker configuration"""
        return self._config

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count"""
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (fast-fail mode)"""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)"""
        return self._state == CircuitBreakerState.HALF_OPEN

    async def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute function call through circuit breaker

        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function call

        Raises:
            CircuitBreakerError: If circuit is open or half-open limit reached
            Exception: If function call fails (re-raised after recording failure)
        """
        async with self._lock:
            self._check_state_transition()

            if self._state == CircuitBreakerState.OPEN:
                retry_after = self._calculate_retry_after()
                logger.warning(
                    f"Circuit breaker '{self._config.name}' is OPEN, fast-failing"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self._config.name}' is OPEN. "
                    f"Will retry after {retry_after:.1f}s",
                    state=self._state,
                    retry_after=retry_after
                )

            if (
                self._state == CircuitBreakerState.HALF_OPEN
                and self._half_open_calls >= self._config.half_open_max_calls
            ):
                logger.warning(
                    f"Circuit breaker '{self._config.name}' HALF_OPEN call limit reached"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self._config.name}' HALF_OPEN call limit reached",
                    state=self._state,
                    retry_after=None
                )
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            # Success - handle state transitions
            async with self._lock:
                self._on_success()

            return result

        except Exception as e:
            # Failure - handle state transitions
            async with self._lock:
                self._on_failure()
            raise

    def _check_state_transition(self) -> None:
        """Check if circuit should transition states based on timeout"""
        if self._state == CircuitBreakerState.OPEN and self._last_failure_time:
            time_since_failure = datetime.now(timezone.utc) - self._last_failure_time
            if time_since_failure.total_seconds() >= self._config.timeout_duration:
                logger.info(
                    f"Circuit breaker '{self._config.name}' transitioning "
                    "from OPEN to HALF_OPEN"
                )
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0

    def _calculate_retry_after(self) -> float:
        """Calculate seconds until retry is allowed"""
        if not self._last_failure_time:
            return self._config.timeout_duration

        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return max(0, self._config.timeout_duration - elapsed)

    def _on_success(self) -> None:
        """Handle successful call"""
        if self._state == CircuitBreakerState.HALF_OPEN:
            logger.info(
                f"Circuit breaker '{self._config.name}' transitioning "
                "from HALF_OPEN to CLOSED"
            )
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
        elif self._state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Any failure in HALF_OPEN reopens circuit
            logger.warning(
                f"Circuit breaker '{self._config.name}' failure in HALF_OPEN state, "
                "reopening circuit"
            )
            self._state = CircuitBreakerState.OPEN
            self._half_open_calls = 0

        elif self._state == CircuitBreakerState.CLOSED:
            if self._failure_count >= self._config.failure_threshold:
                logger.warning(
                    f"Circuit breaker '{self._config.name}' failure threshold "
                    f"({self._config.failure_threshold}) reached, opening circuit"
                )
                self._state = CircuitBreakerState.OPEN

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state"""
        logger.info(f"Circuit breaker '{self._config.name}' manually reset")
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_status(self) -> CircuitBreakerStatus:
        """
        Get circuit breaker status for monitoring

        Returns:
            CircuitBreakerStatus with current state snapshot
        """
        return CircuitBreakerStatus(
            state=self._state,
            failure_count=self._failure_count,
            failure_threshold=self._config.failure_threshold,
            timeout_duration=self._config.timeout_duration,
            last_failure_time=self._last_failure_time,
            half_open_calls=self._half_open_calls,
            half_open_max_calls=self._config.half_open_max_calls,
            name=self._config.name
        )
