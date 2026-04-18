"""
Memory MCP Client with Circuit Breaker and Fallback
Production-ready wrapper for Memory MCP with resilience patterns

Features:
- Circuit breaker integration for fault tolerance
- WHO/WHEN/PROJECT/WHY tagging protocol
- Vector search with semantic similarity ranking
- Pluggable fallback storage (PostgreSQL, SQLite, etc.)
- Pluggable cache layer (Redis, in-memory, etc.)
- Health monitoring and degraded mode detection

Example:
    >>> from memory_mcp_client import (
    ...     MemoryMCPClient, MemoryMCPConfig,
    ...     TaggingProtocol, TaggingConfig, AgentCategory, Intent,
    ...     CircuitBreaker, CircuitBreakerConfig
    ... )
    >>>
    >>> # Configure components
    >>> tagger = TaggingProtocol(TaggingConfig(
    ...     agent_id="my-agent",
    ...     agent_category=AgentCategory.BACKEND,
    ...     capabilities=["api-dev"],
    ...     project_id="my-project",
    ...     project_name="My Project"
    ... ))
    >>> breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    >>>
    >>> # Create client (with optional fallback/cache)
    >>> client = MemoryMCPClient(MemoryMCPConfig(
    ...     tagger=tagger,
    ...     circuit_breaker=breaker,
    ...     mcp_endpoint="http://localhost:3000"
    ... ))
    >>>
    >>> # Store data
    >>> result = await client.store("My content", Intent.IMPLEMENTATION, task_id="TASK-001")
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable
import json

from .tagging_protocol import TaggingProtocol, Intent
from .circuit_breaker import CircuitBreaker, CircuitBreakerState, CircuitBreakerError

logger = logging.getLogger(__name__)


# Protocol definitions for pluggable backends

@runtime_checkable
class FallbackStorage(Protocol):
    """Protocol for fallback storage implementations"""

    async def store(self, payload: Dict[str, Any]) -> None:
        """Store data in fallback storage"""
        ...

    async def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search fallback storage"""
        ...

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task data from fallback storage"""
        ...


@runtime_checkable
class CacheLayer(Protocol):
    """Protocol for cache layer implementations"""

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Set cache value with TTL"""
        ...

    async def get(self, key: str) -> Optional[str]:
        """Get cache value"""
        ...


@runtime_checkable
class MCPTransport(Protocol):
    """Protocol for MCP transport implementations"""

    async def store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store data via MCP"""
        ...

    async def vector_search(
        self,
        query: str,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Vector search via MCP"""
        ...

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task via MCP"""
        ...

    async def ping(self) -> bool:
        """Health check ping"""
        ...


# Default implementations

class InMemoryCache:
    """Simple in-memory cache implementation with TTL support and cleanup"""

    def __init__(self, cleanup_interval: int = 100):
        """
        Initialize in-memory cache.

        Args:
            cleanup_interval: Number of get() operations between automatic cleanups.
                              Set to 0 to disable automatic cleanup.
        """
        self._cache: Dict[str, tuple[str, datetime]] = {}
        self._cleanup_interval = cleanup_interval
        self._access_count = 0

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expires)

    async def get(self, key: str) -> Optional[str]:
        # Periodic cleanup on access
        self._access_count += 1
        if self._cleanup_interval > 0 and self._access_count >= self._cleanup_interval:
            await self.cleanup()
            self._access_count = 0

        if key not in self._cache:
            return None
        value, expires = self._cache[key]
        if datetime.utcnow() > expires:
            del self._cache[key]
            return None
        return value

    async def cleanup(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, expires) in self._cache.items()
            if now > expires
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"InMemoryCache cleanup: removed {len(expired_keys)} expired entries")
        return len(expired_keys)


class InMemoryFallback:
    """
    Simple in-memory fallback storage implementation.

    WARNING: This implementation uses O(n) linear search complexity for the search()
    method, where n is the number of stored entries. This is acceptable for development,
    testing, or small-scale usage (<1000 entries), but NOT recommended for production
    workloads with large datasets.

    For production use, consider implementing FallbackStorage with:
    - PostgreSQL with full-text search (pg_trgm)
    - SQLite with FTS5
    - Elasticsearch
    - A proper vector database (ChromaDB, Pinecone, etc.)
    """

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}

    async def store(self, payload: Dict[str, Any]) -> None:
        task_id = payload.get("metadata", {}).get("project", {}).get("task_id", "unknown")
        self._storage[task_id] = payload

    async def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for matching entries using simple substring matching.

        Note: This method has O(n) complexity where n is the number of stored entries.
        It performs a full scan of all stored data. For production workloads with
        large datasets, use a proper database-backed FallbackStorage implementation.

        Args:
            query: Search query (case-insensitive substring match)
            project_id: Optional project filter
            task_type: Optional task type filter (intent value)
            limit: Maximum results to return

        Returns:
            List of matching entries (no semantic similarity ranking)
        """
        results = []
        query_lower = query.lower()
        for task_id, data in self._storage.items():
            content = data.get("content", "").lower()
            if query_lower not in content:
                continue
            if project_id:
                data_project = data.get("metadata", {}).get("project", {}).get("project_id")
                if data_project != project_id:
                    continue
            if task_type:
                data_intent = data.get("metadata", {}).get("why", {}).get("intent")
                if data_intent != task_type:
                    continue
            results.append(data)
            if len(results) >= limit:
                break
        return results

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(task_id)


class MockMCPTransport:
    """Mock MCP transport for testing"""

    def __init__(self, endpoint: str = "http://localhost:3000"):
        self.endpoint = endpoint
        self._storage: Dict[str, Dict[str, Any]] = {}

    async def store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = payload.get("metadata", {}).get("project", {}).get("task_id", "unknown")
        self._storage[task_id] = payload
        await asyncio.sleep(0.01)  # Simulate network latency
        return {
            "id": task_id,
            "stored_at": payload.get("metadata", {}).get("when", {}).get("iso_timestamp")
        }

    async def vector_search(
        self,
        query: str,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.02)  # Simulate network latency
        # Mock implementation returns empty (would be semantic search in real MCP)
        return []

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(0.01)
        return self._storage.get(task_id)

    async def ping(self) -> bool:
        await asyncio.sleep(0.001)
        return True


@dataclass
class MemoryMCPConfig:
    """
    Configuration for MemoryMCPClient

    Attributes:
        tagger: TaggingProtocol instance for metadata generation
        circuit_breaker: CircuitBreaker instance for fault tolerance
        mcp_endpoint: Memory MCP server endpoint URL
        mcp_transport: Optional custom MCP transport (uses mock if None)
        fallback_storage: Optional fallback storage (uses in-memory if None)
        cache_layer: Optional cache layer (uses in-memory if None)
        cache_ttl_seconds: TTL for cached data (default: 24 hours)
        health_check_interval_seconds: Interval between health checks (default: 30s)
    """
    tagger: TaggingProtocol
    circuit_breaker: CircuitBreaker
    mcp_endpoint: str = "http://localhost:3000"
    mcp_transport: Optional[MCPTransport] = None
    fallback_storage: Optional[FallbackStorage] = None
    cache_layer: Optional[CacheLayer] = None
    cache_ttl_seconds: int = 86400  # 24 hours
    health_check_interval_seconds: int = 30


@dataclass
class StoreResult:
    """Result of a store operation"""
    status: str  # "success" or "degraded"
    storage: str  # "memory_mcp" or "fallback"
    task_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    warning: Optional[str] = None

    def __repr__(self) -> str:
        warning_str = f", warning={self.warning!r}" if self.warning else ""
        return (
            f"StoreResult(status={self.status!r}, storage={self.storage!r}, "
            f"task_id={self.task_id!r}{warning_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status,
            "storage": self.storage,
            "task_id": self.task_id,
            "metadata": self.metadata
        }
        if self.warning:
            result["warning"] = self.warning
        return result


@dataclass
class TaskHistoryResult:
    """Result of a task history query"""
    task: Optional[Dict[str, Any]]
    related_tasks: List[Dict[str, Any]]
    source: str  # "memory_mcp", "cache", or "fallback"
    warning: Optional[str] = None

    def __repr__(self) -> str:
        task_preview = "present" if self.task else "None"
        warning_str = f", warning={self.warning!r}" if self.warning else ""
        return (
            f"TaskHistoryResult(task={task_preview}, "
            f"related_tasks=[{len(self.related_tasks)} items], "
            f"source={self.source!r}{warning_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task": self.task,
            "related_tasks": self.related_tasks,
            "source": self.source
        }
        if self.warning:
            result["warning"] = self.warning
        return result


@dataclass
class HealthStatus:
    """Health check result"""
    status: str  # "healthy", "degraded", or "cached"
    degraded_mode: bool
    circuit_breaker_state: str
    mcp_available: Optional[bool] = None
    fallback_available: Optional[bool] = None
    error: Optional[str] = None
    last_check: Optional[str] = None

    def __repr__(self) -> str:
        error_str = f", error={self.error!r}" if self.error else ""
        return (
            f"HealthStatus(status={self.status!r}, degraded_mode={self.degraded_mode}, "
            f"circuit_breaker_state={self.circuit_breaker_state!r}, "
            f"mcp_available={self.mcp_available}{error_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "degraded_mode": self.degraded_mode,
            "circuit_breaker_state": self.circuit_breaker_state,
            "mcp_available": self.mcp_available,
            "fallback_available": self.fallback_available,
            "error": self.error,
            "last_check": self.last_check
        }


class MemoryMCPClient:
    """
    Production-ready Memory MCP client with circuit breaker and fallback

    Implements:
    - Circuit breaker pattern for fault tolerance
    - Automatic tagging protocol (WHO/WHEN/PROJECT/WHY)
    - Vector search with ranking
    - Fallback to configurable storage
    - Health monitoring

    Example:
        >>> config = MemoryMCPConfig(
        ...     tagger=my_tagger,
        ...     circuit_breaker=my_breaker
        ... )
        >>> client = MemoryMCPClient(config)
        >>>
        >>> # Store with automatic tagging
        >>> result = await client.store(
        ...     content="Implemented auth flow",
        ...     intent=Intent.IMPLEMENTATION,
        ...     task_id="AUTH-001"
        ... )
        >>>
        >>> # Vector search
        >>> results = await client.vector_search("authentication", limit=5)
        >>>
        >>> # Health check
        >>> health = await client.health_check()
    """

    def __init__(self, config: MemoryMCPConfig):
        """
        Initialize Memory MCP client

        Args:
            config: MemoryMCPConfig with all dependencies
        """
        self._config = config
        self._tagger = config.tagger
        self._circuit_breaker = config.circuit_breaker

        # Initialize transport (mock if not provided)
        self._transport: MCPTransport = config.mcp_transport or MockMCPTransport(config.mcp_endpoint)

        # Initialize fallback storage
        self._fallback: FallbackStorage = config.fallback_storage or InMemoryFallback()

        # Initialize cache layer
        self._cache: CacheLayer = config.cache_layer or InMemoryCache()

        # State lock protects _degraded_mode and _last_health_check from race conditions
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._degraded_mode = False
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval = timedelta(seconds=config.health_check_interval_seconds)

    @property
    def config(self) -> MemoryMCPConfig:
        """Get client configuration"""
        return self._config

    @property
    def is_degraded(self) -> bool:
        """Check if client is in degraded mode"""
        return self._degraded_mode

    @property
    def circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self._circuit_breaker.state

    async def store(
        self,
        content: str,
        intent: Intent,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> StoreResult:
        """
        Store data in Memory MCP with automatic tagging

        Args:
            content: Content to store
            intent: Intent category (implementation, bugfix, etc.)
            user_id: Optional user identifier
            task_id: Optional task identifier
            additional_metadata: Optional additional metadata

        Returns:
            StoreResult with storage confirmation
        """
        # Generate tagged payload
        payload_obj = self._tagger.create_memory_store_payload(
            content=content,
            intent=intent,
            user_id=user_id,
            task_id=task_id,
            additional_metadata=additional_metadata
        )
        payload = payload_obj.to_dict()
        actual_task_id = payload["metadata"]["project"]["task_id"]

        # Attempt Memory MCP storage with circuit breaker
        try:
            await self._circuit_breaker.call(
                self._transport.store,
                payload
            )

            # Also cache for fallback
            await self._cache_payload(payload)

            logger.info(
                f"Stored in Memory MCP: task_id={actual_task_id}, "
                f"intent={intent.value}, agent={self._tagger.agent_id}"
            )

            return StoreResult(
                status="success",
                storage="memory_mcp",
                task_id=actual_task_id,
                metadata=payload["metadata"]
            )

        except (CircuitBreakerError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Memory MCP storage failed: {e}, using fallback")

            # Fallback storage
            await self._fallback.store(payload)
            await self._cache_payload(payload)

            async with self._state_lock:
                self._degraded_mode = True

            return StoreResult(
                status="degraded",
                storage="fallback",
                task_id=actual_task_id,
                metadata=payload["metadata"],
                warning="Memory MCP unavailable, using fallback storage"
            )

    async def vector_search(
        self,
        query: str,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Vector search for similar tasks/content

        Args:
            query: Search query
            project_id: Optional project filter
            task_type: Optional task type filter (intent value)
            limit: Maximum results to return

        Returns:
            List of results ranked by semantic similarity
        """
        try:
            results = await self._circuit_breaker.call(
                self._transport.vector_search,
                query,
                project_id,
                task_type,
                limit
            )

            # Rank by similarity score (included in Memory MCP response)
            ranked_results = sorted(
                results,
                key=lambda x: x.get("similarity_score", 0),
                reverse=True
            )

            logger.info(f"Vector search returned {len(ranked_results)} results")

            return ranked_results

        except (CircuitBreakerError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Memory MCP vector search failed: {e}, using fallback")

            # Fallback: text search (no semantic similarity)
            fallback_results = await self._fallback.search(
                query,
                project_id,
                task_type,
                limit
            )

            async with self._state_lock:
                self._degraded_mode = True

            return fallback_results

    async def get_task_history(
        self,
        task_id: str,
        include_related: bool = True
    ) -> TaskHistoryResult:
        """
        Get task history with optional related tasks

        Args:
            task_id: Task identifier
            include_related: Whether to include semantically related tasks

        Returns:
            TaskHistoryResult with task data and related tasks
        """
        try:
            # Get primary task data
            task_data = await self._circuit_breaker.call(
                self._transport.get_task,
                task_id
            )

            # Optionally get related tasks via vector search
            related_tasks = []
            if include_related and task_data:
                content = task_data.get("content", "")
                task_type = task_data.get("metadata", {}).get("why", {}).get("intent")
                if content:
                    related_tasks = await self.vector_search(
                        query=content,
                        task_type=task_type,
                        limit=5
                    )

            return TaskHistoryResult(
                task=task_data,
                related_tasks=related_tasks,
                source="memory_mcp"
            )

        except (CircuitBreakerError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"Failed to get task history from Memory MCP: {e}")

            # Try cache first
            cached_data = await self._get_from_cache(task_id)
            if cached_data:
                logger.info(f"Serving stale data from cache for task {task_id}")
                return TaskHistoryResult(
                    task=cached_data,
                    related_tasks=[],
                    source="cache",
                    warning="Serving stale data - Memory MCP unavailable"
                )

            # Last resort: fallback storage
            task_data = await self._fallback.get(task_id)
            return TaskHistoryResult(
                task=task_data,
                related_tasks=[],
                source="fallback",
                warning="Memory MCP unavailable, no semantic search available"
            )

    async def health_check(self) -> HealthStatus:
        """
        Check Memory MCP health and circuit breaker state

        Returns:
            HealthStatus with current health information
        """
        now = datetime.utcnow()

        # Rate-limit health checks (protected by lock to prevent concurrent bypass)
        async with self._state_lock:
            if (self._last_health_check and
                now - self._last_health_check < self._health_check_interval):
                return HealthStatus(
                    status="cached",
                    degraded_mode=self._degraded_mode,
                    circuit_breaker_state=self._circuit_breaker.state.value
                )
            # Claim this check slot before releasing lock
            self._last_health_check = now

        try:
            # Quick health check to Memory MCP (outside lock - may be slow)
            await self._circuit_breaker.call(self._transport.ping)

            async with self._state_lock:
                self._degraded_mode = False

            return HealthStatus(
                status="healthy",
                degraded_mode=False,
                circuit_breaker_state=CircuitBreakerState.CLOSED.value,
                mcp_available=True,
                fallback_available=True,
                last_check=now.isoformat()
            )

        except (CircuitBreakerError, ConnectionError, TimeoutError, OSError) as e:
            async with self._state_lock:
                self._degraded_mode = True

            return HealthStatus(
                status="degraded",
                degraded_mode=True,
                circuit_breaker_state=self._circuit_breaker.state.value,
                mcp_available=False,
                fallback_available=True,
                error=str(e),
                last_check=now.isoformat()
            )

    async def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state"""
        self._circuit_breaker.reset()
        async with self._state_lock:
            self._degraded_mode = False
        logger.info("Circuit breaker reset, exiting degraded mode")

    # Private helper methods

    async def _cache_payload(self, payload: Dict[str, Any]) -> None:
        """Cache payload for fallback"""
        task_id = payload["metadata"]["project"]["task_id"]
        cache_key = f"memory_mcp:task:{task_id}"

        await self._cache.set(
            cache_key,
            json.dumps(payload),
            self._config.cache_ttl_seconds
        )

        logger.debug(f"Cached payload: {cache_key}")

    async def _get_from_cache(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data"""
        cache_key = f"memory_mcp:task:{task_id}"

        cached = await self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {cache_key}")
            return json.loads(cached)

        return None


# Factory function for quick setup

def create_memory_mcp_client(
    project_id: str,
    project_name: str,
    agent_id: str = "default-agent",
    agent_category: str = "backend",
    capabilities: Optional[List[str]] = None,
    mcp_endpoint: str = "http://localhost:3000",
    failure_threshold: int = 3,
    timeout_duration: int = 60,
    mcp_transport: Optional[MCPTransport] = None,
    fallback_storage: Optional[FallbackStorage] = None,
    cache_layer: Optional[CacheLayer] = None
) -> MemoryMCPClient:
    """
    Factory function to create configured Memory MCP client

    Args:
        project_id: Project identifier
        project_name: Human-readable project name
        agent_id: Agent identifier
        agent_category: Agent category string
        capabilities: Agent capabilities list
        mcp_endpoint: Memory MCP server endpoint
        failure_threshold: Circuit breaker failure threshold
        timeout_duration: Circuit breaker timeout in seconds
        mcp_transport: Optional custom MCP transport
        fallback_storage: Optional fallback storage
        cache_layer: Optional cache layer

    Returns:
        Configured MemoryMCPClient instance
    """
    from .tagging_protocol import TaggingConfig, AgentCategory as AC
    from .circuit_breaker import CircuitBreakerConfig

    # Map string to AgentCategory using dynamic lookup with fallback
    # This handles all AgentCategory enum values without manual mapping maintenance
    category_upper = agent_category.upper().replace("-", "_")
    category = getattr(AC, category_upper, None)
    if category is None:
        # Fallback: try exact match on enum values
        for member in AC:
            if member.value == agent_category.lower() or member.name == category_upper:
                category = member
                break
        else:
            category = AC.CUSTOM

    # Create tagging protocol
    tagger = TaggingProtocol(TaggingConfig(
        agent_id=agent_id,
        agent_category=category,
        capabilities=capabilities or ["general"],
        project_id=project_id,
        project_name=project_name
    ))

    # Create circuit breaker
    circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_duration=timeout_duration,
        name=f"memory-mcp-{project_id}"
    ))

    # Create client config
    config = MemoryMCPConfig(
        tagger=tagger,
        circuit_breaker=circuit_breaker,
        mcp_endpoint=mcp_endpoint,
        mcp_transport=mcp_transport,
        fallback_storage=fallback_storage,
        cache_layer=cache_layer
    )

    return MemoryMCPClient(config)
