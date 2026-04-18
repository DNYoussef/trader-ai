"""
HTTP Client with Circuit Breaker Integration

Provides circuit breaker wrappers for common HTTP clients (httpx, aiohttp).
Use these to protect external API calls from cascading failures.

LEGO Component: Extends CircuitBreaker with HTTP-specific functionality

Usage:
    from library.components.utilities.circuit_breaker import (
        CircuitBreakerHttpClient,
        CircuitBreakerConfig,
    )

    # Create protected HTTP client
    client = CircuitBreakerHttpClient(
        name="memory_mcp",
        base_url="http://localhost:8080",
        config=CircuitBreakerConfig(failure_threshold=3, timeout_duration=30)
    )

    # Make protected calls
    try:
        result = await client.get("/health")
        data = await client.post("/tools/memory_store", json={"text": "hello"})
    except CircuitBreakerError as e:
        logger.warning(f"Service unavailable, retry after {e.retry_after}s")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerManager,
    CircuitState,
)

logger = logging.getLogger(__name__)

# Optional imports - support both httpx and aiohttp
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


@dataclass
class HttpResponse:
    """Unified HTTP response wrapper."""
    status_code: int
    text: str
    json_data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None

    def json(self) -> Dict[str, Any]:
        """Get JSON data, parsing if needed."""
        if self.json_data is not None:
            return self.json_data
        import json as json_module
        return json_module.loads(self.text)

    @property
    def ok(self) -> bool:
        """Check if response was successful (2xx)."""
        return 200 <= self.status_code < 300


class CircuitBreakerHttpClient:
    """
    HTTP client with built-in circuit breaker protection.

    Wraps httpx.AsyncClient with circuit breaker pattern for fault tolerance.
    Falls back to aiohttp if httpx is not available.

    Usage:
        client = CircuitBreakerHttpClient(
            name="external_api",
            base_url="https://api.example.com",
            config=CircuitBreakerConfig(failure_threshold=5)
        )

        async with client:
            response = await client.get("/endpoint")
            if response.ok:
                data = response.json()
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        config: Optional[CircuitBreakerConfig] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP client with circuit breaker.

        Args:
            name: Unique name for this client's circuit breaker
            base_url: Base URL for all requests
            config: Circuit breaker configuration
            timeout: Default timeout for requests in seconds
            headers: Default headers to include in all requests
        """
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = headers or {}

        # Initialize circuit breaker
        self._breaker = CircuitBreaker(name, config)

        # HTTP client (created on enter)
        self._client: Optional[Any] = None
        self._session: Optional[Any] = None

        logger.info(f"CircuitBreakerHttpClient '{name}' initialized for {base_url}")

    @property
    def breaker(self) -> CircuitBreaker:
        """Get the underlying circuit breaker."""
        return self._breaker

    @property
    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        return not self._breaker.is_open

    async def __aenter__(self) -> "CircuitBreakerHttpClient":
        """Async context manager entry - create HTTP client."""
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self.default_headers,
            )
        elif AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.default_headers,
            )
        else:
            raise RuntimeError("No HTTP client available (install httpx or aiohttp)")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> HttpResponse:
        """
        Make HTTP request through circuit breaker.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: URL path (appended to base_url)
            **kwargs: Additional arguments passed to HTTP client

        Returns:
            HttpResponse wrapper

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If request fails (after recording failure)
        """
        url = f"{self.base_url}{path}" if not path.startswith("http") else path

        async def do_request():
            if self._client is not None:
                # httpx
                response = await self._client.request(method, path, **kwargs)
                json_data = None
                try:
                    json_data = response.json()
                except Exception:
                    pass
                return HttpResponse(
                    status_code=response.status_code,
                    text=response.text,
                    json_data=json_data,
                    headers=dict(response.headers),
                )
            elif self._session is not None:
                # aiohttp
                async with self._session.request(method, path, **kwargs) as response:
                    text = await response.text()
                    json_data = None
                    try:
                        json_data = await response.json()
                    except Exception:
                        pass
                    return HttpResponse(
                        status_code=response.status,
                        text=text,
                        json_data=json_data,
                        headers=dict(response.headers),
                    )
            else:
                # Create temporary client
                if HTTPX_AVAILABLE:
                    async with httpx.AsyncClient(
                        timeout=self.timeout,
                        headers=self.default_headers
                    ) as client:
                        response = await client.request(method, url, **kwargs)
                        json_data = None
                        try:
                            json_data = response.json()
                        except Exception:
                            pass
                        return HttpResponse(
                            status_code=response.status_code,
                            text=response.text,
                            json_data=json_data,
                            headers=dict(response.headers),
                        )
                else:
                    raise RuntimeError("No HTTP client available")

        # Execute through circuit breaker
        return await self._breaker.call(do_request)

    async def get(self, path: str, **kwargs) -> HttpResponse:
        """GET request through circuit breaker."""
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> HttpResponse:
        """POST request through circuit breaker."""
        return await self._request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs) -> HttpResponse:
        """PUT request through circuit breaker."""
        return await self._request("PUT", path, **kwargs)

    async def patch(self, path: str, **kwargs) -> HttpResponse:
        """PATCH request through circuit breaker."""
        return await self._request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> HttpResponse:
        """DELETE request through circuit breaker."""
        return await self._request("DELETE", path, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        status = self._breaker.get_status()
        return {
            "name": self.name,
            "base_url": self.base_url,
            "available": self.is_available,
            "circuit_state": status.state.value,
            "failure_count": status.failure_count,
            "success_count": status.success_count,
            "metrics": status.to_dict(),
        }


class HttpClientManager:
    """
    Manager for multiple circuit-breaker protected HTTP clients.

    Provides centralized management of external service clients
    with automatic circuit breaker coordination.

    Usage:
        manager = HttpClientManager()

        # Register services
        await manager.register("memory_mcp", "http://localhost:8080")
        await manager.register("broker_api", "https://api.broker.com",
                               config=CircuitBreakerConfig(failure_threshold=3))

        # Get client and make calls
        async with manager.get("memory_mcp") as client:
            result = await client.get("/health")

        # Check system status
        status = manager.get_system_status()
        if status["unavailable_services"]:
            logger.warning(f"Services down: {status['unavailable_services']}")
    """

    def __init__(self):
        """Initialize HTTP client manager."""
        self._clients: Dict[str, CircuitBreakerHttpClient] = {}
        self._breaker_manager = CircuitBreakerManager()
        logger.info("HttpClientManager initialized")

    async def register(
        self,
        name: str,
        base_url: str,
        config: Optional[CircuitBreakerConfig] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> CircuitBreakerHttpClient:
        """
        Register a new HTTP client with circuit breaker.

        Args:
            name: Unique service name
            base_url: Base URL for the service
            config: Circuit breaker configuration
            timeout: Request timeout
            headers: Default headers

        Returns:
            The registered CircuitBreakerHttpClient
        """
        if name in self._clients:
            raise ValueError(f"Client '{name}' already registered")

        client = CircuitBreakerHttpClient(
            name=name,
            base_url=base_url,
            config=config,
            timeout=timeout,
            headers=headers,
        )
        self._clients[name] = client

        logger.info(f"Registered HTTP client: {name} -> {base_url}")
        return client

    def get(self, name: str) -> CircuitBreakerHttpClient:
        """
        Get a registered HTTP client by name.

        Args:
            name: Service name

        Returns:
            CircuitBreakerHttpClient instance

        Raises:
            KeyError: If service not registered
        """
        if name not in self._clients:
            raise KeyError(f"Client '{name}' not registered")
        return self._clients[name]

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all registered HTTP clients."""
        statuses = {}
        unavailable = []

        for name, client in self._clients.items():
            status = client.get_status()
            statuses[name] = status
            if not client.is_available:
                unavailable.append(name)

        return {
            "total_services": len(self._clients),
            "available_services": len(self._clients) - len(unavailable),
            "unavailable_services": unavailable,
            "services": statuses,
        }

    def any_unavailable(self) -> bool:
        """Check if any services have open circuit breakers."""
        return any(not c.is_available for c in self._clients.values())

    def get_unavailable_services(self) -> list:
        """Get list of services with open circuit breakers."""
        return [name for name, c in self._clients.items() if not c.is_available]


# Decorator for adding circuit breaker to existing async functions
def with_circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Any] = None
):
    """
    Decorator to wrap async functions with circuit breaker.

    Usage:
        breaker = CircuitBreaker("my_service")

        @with_circuit_breaker(breaker, fallback={"status": "unavailable"})
        async def call_external_api():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com")

        # Calls will be protected by circuit breaker
        # Returns fallback value if circuit is open
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call(func, *args, **kwargs)
            except CircuitBreakerError:
                if fallback is not None:
                    return fallback
                raise
        return wrapper
    return decorator
