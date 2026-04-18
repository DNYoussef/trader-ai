"""
Memory MCP Client Library Component

A production-ready Memory MCP client with circuit breaker and fallback patterns.
Extracted from Life-OS Dashboard and generalized for reuse.

Features:
- Circuit breaker pattern for fault tolerance
- WHO/WHEN/PROJECT/WHY tagging protocol
- Vector search with semantic similarity ranking
- Pluggable fallback storage (PostgreSQL, SQLite, Redis, in-memory)
- Pluggable cache layer (Redis, in-memory)
- Health monitoring and degraded mode detection

Quick Start:
    >>> from memory_mcp_client import create_memory_mcp_client, Intent
    >>>
    >>> # Create client with defaults
    >>> client = create_memory_mcp_client(
    ...     project_id="my-project",
    ...     project_name="My Project"
    ... )
    >>>
    >>> # Store data
    >>> result = await client.store(
    ...     content="Implemented feature X",
    ...     intent=Intent.IMPLEMENTATION,
    ...     task_id="TASK-001"
    ... )
    >>>
    >>> # Vector search
    >>> results = await client.vector_search("feature X", limit=5)
    >>>
    >>> # Health check
    >>> health = await client.health_check()

Advanced Usage:
    >>> from memory_mcp_client import (
    ...     MemoryMCPClient, MemoryMCPConfig,
    ...     TaggingProtocol, TaggingConfig, AgentCategory,
    ...     CircuitBreaker, CircuitBreakerConfig,
    ...     Intent
    ... )
    >>>
    >>> # Custom configuration
    >>> tagger = TaggingProtocol(TaggingConfig(
    ...     agent_id="custom-agent",
    ...     agent_category=AgentCategory.BACKEND,
    ...     capabilities=["api-dev", "database"],
    ...     project_id="my-project",
    ...     project_name="My Project"
    ... ))
    >>>
    >>> breaker = CircuitBreaker(CircuitBreakerConfig(
    ...     failure_threshold=5,
    ...     timeout_duration=60,
    ...     name="my-breaker"
    ... ))
    >>>
    >>> client = MemoryMCPClient(MemoryMCPConfig(
    ...     tagger=tagger,
    ...     circuit_breaker=breaker,
    ...     mcp_endpoint="http://localhost:3000"
    ... ))
"""

__version__ = "1.0.0"
__author__ = "David Youssef"
__source__ = "life-os-dashboard"

# Circuit breaker exports
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStatus,
    CircuitBreakerError,
)

# Tagging protocol exports
from .tagging_protocol import (
    # Enums
    Intent,
    AgentCategory,
    INTENT_DESCRIPTIONS,
    # Config and Protocol
    TaggingConfig,
    TaggingProtocol,
    # Tag dataclasses
    WhoTag,
    WhenTag,
    ProjectTag,
    WhyTag,
    MemoryTags,
    MemoryStorePayload,
    # Factory functions
    create_backend_tagger,
    create_frontend_tagger,
    create_testing_tagger,
    create_custom_tagger,
)

# Memory MCP client exports
from .memory_mcp_client import (
    # Protocols for custom implementations
    FallbackStorage,
    CacheLayer,
    MCPTransport,
    # Default implementations
    InMemoryCache,
    InMemoryFallback,
    MockMCPTransport,
    # Config and Client
    MemoryMCPConfig,
    MemoryMCPClient,
    # Result types
    StoreResult,
    TaskHistoryResult,
    HealthStatus,
    # Factory function
    create_memory_mcp_client,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__source__",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreakerStatus",
    "CircuitBreakerError",
    # Tagging - Enums
    "Intent",
    "AgentCategory",
    "INTENT_DESCRIPTIONS",
    # Tagging - Config and Protocol
    "TaggingConfig",
    "TaggingProtocol",
    # Tagging - Tag dataclasses
    "WhoTag",
    "WhenTag",
    "ProjectTag",
    "WhyTag",
    "MemoryTags",
    "MemoryStorePayload",
    # Tagging - Factories
    "create_backend_tagger",
    "create_frontend_tagger",
    "create_testing_tagger",
    "create_custom_tagger",
    # MCP Client - Protocols
    "FallbackStorage",
    "CacheLayer",
    "MCPTransport",
    # MCP Client - Default implementations
    "InMemoryCache",
    "InMemoryFallback",
    "MockMCPTransport",
    # MCP Client - Config and Client
    "MemoryMCPConfig",
    "MemoryMCPClient",
    # MCP Client - Results
    "StoreResult",
    "TaskHistoryResult",
    "HealthStatus",
    # MCP Client - Factory
    "create_memory_mcp_client",
]
