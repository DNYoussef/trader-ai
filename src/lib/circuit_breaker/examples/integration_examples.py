"""
Circuit Breaker Integration Examples

Shows how to wire circuit breakers to external API calls in existing code.
These patterns can be applied to any project that makes HTTP calls.

Example 1: Wrapping existing HTTP client methods
Example 2: Using CircuitBreakerHttpClient directly
Example 3: Using the decorator pattern
Example 4: Manager pattern for multiple services
"""

import asyncio
import logging
from typing import Any, Dict, Optional

# Import circuit breaker components
from library.components.utilities.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerHttpClient,
    HttpClientManager,
    with_circuit_breaker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Wrapping Existing Methods with Circuit Breaker
# =============================================================================

class MemoryMCPClientWithBreaker:
    """
    Example: Adding circuit breaker to existing MemoryMCPClient.

    Pattern: Wrap the internal HTTP calls with circuit breaker.
    This is the least invasive approach for existing code.

    Original: MemoryMCPClient from life-os-dashboard
    Modified: Added _breaker and wrapped async calls
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.timeout = 30

        # Add circuit breaker for Memory MCP calls
        self._breaker = CircuitBreaker(
            name="memory_mcp",
            config=CircuitBreakerConfig(
                failure_threshold=3,        # Trip after 3 failures
                timeout_duration=30,        # Wait 30s before retry
                success_threshold=2,        # Need 2 successes to recover
                exponential_backoff=True,   # Use exponential backoff
                max_backoff_seconds=300,    # Max 5 minutes backoff
            )
        )

    async def store(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Store with circuit breaker protection."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")

        async def _do_store():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tools/memory_store",
                    json={"text": text, "metadata": metadata or {}}
                )
                response.raise_for_status()
                return response.json()

        # Execute through circuit breaker
        return await self._breaker.call(_do_store)

    async def search(self, query: str, limit: int = 10) -> Dict:
        """Search with circuit breaker protection."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed")

        async def _do_search():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tools/vector_search",
                    json={"query": query, "limit": limit}
                )
                response.raise_for_status()
                return response.json()

        return await self._breaker.call(_do_search)

    @property
    def is_available(self) -> bool:
        """Check if service is available (circuit not open)."""
        return not self._breaker.is_open

    def get_circuit_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        status = self._breaker.get_status()
        return status.to_dict()


# =============================================================================
# Example 2: Using CircuitBreakerHttpClient Directly
# =============================================================================

class BrokerAPIClient:
    """
    Example: Using CircuitBreakerHttpClient for new code.

    Pattern: Use the provided HTTP wrapper for cleaner code.
    Best for new implementations or major refactors.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.broker.com"):
        self.api_key = api_key
        self._client = CircuitBreakerHttpClient(
            name="broker_api",
            base_url=base_url,
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_duration=60,
                failure_rate_threshold=0.3,  # Trip if 30% fail
            ),
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def get_account(self) -> Dict:
        """Get account info with automatic circuit breaker."""
        async with self._client:
            response = await self._client.get("/v1/account")
            if not response.ok:
                raise ValueError(f"API error: {response.status_code}")
            return response.json()

    async def submit_order(self, order: Dict) -> Dict:
        """Submit order with circuit breaker protection."""
        async with self._client:
            response = await self._client.post("/v1/orders", json=order)
            if not response.ok:
                raise ValueError(f"Order failed: {response.text}")
            return response.json()

    @property
    def is_available(self) -> bool:
        """Check if broker API is available."""
        return self._client.is_available


# =============================================================================
# Example 3: Decorator Pattern for Functions
# =============================================================================

# Create a shared circuit breaker for external analytics
analytics_breaker = CircuitBreaker(
    name="analytics_api",
    config=CircuitBreakerConfig(failure_threshold=3)
)


@with_circuit_breaker(analytics_breaker, fallback={"events": [], "error": "unavailable"})
async def track_event(event_name: str, properties: Dict) -> Dict:
    """
    Track analytics event with circuit breaker.

    If the analytics service is down, returns fallback instead of failing.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            "https://analytics.example.com/track",
            json={"event": event_name, "properties": properties}
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Example 4: Manager Pattern for Multiple Services
# =============================================================================

async def setup_service_clients() -> HttpClientManager:
    """
    Example: Using HttpClientManager for multiple services.

    Pattern: Centralized management of all external service clients.
    Best for applications with many external dependencies.
    """
    manager = HttpClientManager()

    # Register all external services
    await manager.register(
        "memory_mcp",
        "http://localhost:8080",
        config=CircuitBreakerConfig(failure_threshold=3, timeout_duration=30)
    )

    await manager.register(
        "broker_api",
        "https://api.broker.com",
        config=CircuitBreakerConfig(failure_threshold=5, timeout_duration=60)
    )

    await manager.register(
        "analytics",
        "https://analytics.example.com",
        config=CircuitBreakerConfig(
            failure_threshold=10,
            timeout_duration=10,
            exponential_backoff=False  # Fast retry for analytics
        )
    )

    return manager


async def use_managed_clients(manager: HttpClientManager):
    """Example of using managed clients."""
    # Check system health
    status = manager.get_system_status()
    logger.info(f"Services available: {status['available_services']}/{status['total_services']}")

    if manager.any_unavailable():
        logger.warning(f"Unavailable: {manager.get_unavailable_services()}")

    # Use specific client
    try:
        async with manager.get("memory_mcp") as client:
            response = await client.get("/health")
            if response.ok:
                logger.info("Memory MCP healthy")
    except CircuitBreakerError as e:
        logger.warning(f"Memory MCP circuit open: retry after {e.retry_after}s")
    except KeyError:
        logger.error("Memory MCP client not registered")


# =============================================================================
# Example 5: Integration with FastAPI
# =============================================================================

def create_fastapi_integration():
    """
    Example: Integrating circuit breakers with FastAPI.

    Shows how to expose circuit breaker status via health endpoints.
    """
    from fastapi import FastAPI, Depends
    from typing import Dict

    app = FastAPI()

    # Global HTTP client manager
    _manager: Optional[HttpClientManager] = None

    async def get_manager() -> HttpClientManager:
        global _manager
        if _manager is None:
            _manager = await setup_service_clients()
        return _manager

    @app.get("/health/services")
    async def service_health(manager: HttpClientManager = Depends(get_manager)) -> Dict:
        """Get health status of all external services."""
        return manager.get_system_status()

    @app.get("/health/circuit-breakers")
    async def circuit_breaker_status(manager: HttpClientManager = Depends(get_manager)) -> Dict:
        """Get detailed circuit breaker status."""
        return {
            "unavailable": manager.get_unavailable_services(),
            "any_open": manager.any_unavailable(),
        }

    return app


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate circuit breaker usage patterns."""
    print("Circuit Breaker Integration Examples")
    print("=" * 50)

    # Example 1: Wrapped client
    print("\n1. Wrapped existing client:")
    client = MemoryMCPClientWithBreaker()
    print(f"   Available: {client.is_available}")
    print(f"   Status: {client.get_circuit_status()}")

    # Example 3: Decorator pattern
    print("\n2. Decorator pattern:")
    result = await track_event("test", {"source": "example"})
    print(f"   Result: {result}")

    # Example 4: Manager pattern
    print("\n3. Manager pattern:")
    manager = await setup_service_clients()
    status = manager.get_system_status()
    print(f"   Total services: {status['total_services']}")
    print(f"   Available: {status['available_services']}")

    print("\n" + "=" * 50)
    print("See source code for complete integration patterns")


if __name__ == "__main__":
    asyncio.run(main())
