"""
Circuit Breaker Component

Generic circuit breaker with exponential backoff for fault tolerance.
Includes HTTP client wrappers for protecting external API calls.

Usage:
    # Basic circuit breaker
    from library.components.utilities.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerError,
        CircuitBreakerManager,
        CircuitState,
    )

    # HTTP client with circuit breaker
    from library.components.utilities.circuit_breaker import (
        CircuitBreakerHttpClient,
        HttpClientManager,
        with_circuit_breaker,
    )
"""

from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerStatus,
    CircuitBreaker,
    CircuitBreakerManager,
)

from .http_wrapper import (
    HttpResponse,
    CircuitBreakerHttpClient,
    HttpClientManager,
    with_circuit_breaker,
)

__all__ = [
    # Core circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerMetrics",
    "CircuitBreakerStatus",
    "CircuitBreaker",
    "CircuitBreakerManager",
    # HTTP wrappers
    "HttpResponse",
    "CircuitBreakerHttpClient",
    "HttpClientManager",
    "with_circuit_breaker",
]
