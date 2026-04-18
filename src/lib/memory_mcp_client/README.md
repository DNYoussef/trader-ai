# Memory MCP Client

A production-ready Memory MCP client with circuit breaker and fallback patterns.

## Source

Extracted from: `D:\Projects\life-os-dashboard\backend\app\utils\memory_mcp_client.py`

## Features

- **Circuit Breaker Pattern**: Fault tolerance with automatic state transitions (CLOSED -> OPEN -> HALF_OPEN)
- **WHO/WHEN/PROJECT/WHY Tagging**: Automatic metadata generation for all memory operations
- **Vector Search**: Semantic similarity search with ranking
- **Pluggable Backends**: Custom fallback storage and cache implementations
- **Health Monitoring**: Degraded mode detection and health checks
- **Type Safety**: Full type hints and dataclass configurations

## Installation

Copy this component to your project or add to your library path:

```python
import sys
sys.path.insert(0, r"C:\Users\17175\.claude\library\components\memory")
from memory_mcp_client import create_memory_mcp_client, Intent
```

## Quick Start

```python
import asyncio
from memory_mcp_client import create_memory_mcp_client, Intent

async def main():
    # Create client with defaults (uses in-memory fallback and cache)
    client = create_memory_mcp_client(
        project_id="my-project",
        project_name="My Project",
        agent_id="my-agent"
    )

    # Store data with automatic tagging
    result = await client.store(
        content="Implemented user authentication flow",
        intent=Intent.IMPLEMENTATION,
        task_id="AUTH-001"
    )
    print(f"Stored: {result.status}, storage: {result.storage}")

    # Vector search
    results = await client.vector_search("authentication", limit=5)
    print(f"Found {len(results)} results")

    # Get task history
    history = await client.get_task_history("AUTH-001")
    print(f"Source: {history.source}")

    # Health check
    health = await client.health_check()
    print(f"Status: {health.status}, degraded: {health.degraded_mode}")

asyncio.run(main())
```

## Advanced Usage

### Custom Configuration

```python
from memory_mcp_client import (
    MemoryMCPClient,
    MemoryMCPConfig,
    TaggingProtocol,
    TaggingConfig,
    AgentCategory,
    CircuitBreaker,
    CircuitBreakerConfig,
    Intent
)

# Create tagging protocol
tagger = TaggingProtocol(TaggingConfig(
    agent_id="backend-dev",
    agent_category=AgentCategory.BACKEND,
    capabilities=["REST API", "PostgreSQL", "Redis"],
    project_id="my-project",
    project_name="My Project",
    default_user_id="system"
))

# Create circuit breaker
breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    timeout_duration=60,
    half_open_max_calls=3,
    name="memory-mcp"
))

# Create client
client = MemoryMCPClient(MemoryMCPConfig(
    tagger=tagger,
    circuit_breaker=breaker,
    mcp_endpoint="http://localhost:3000",
    cache_ttl_seconds=86400,  # 24 hours
    health_check_interval_seconds=30
))
```

### Custom Fallback Storage (Redis Example)

```python
from memory_mcp_client import FallbackStorage
import aioredis
import json

class RedisFallbackStorage:
    """Redis-based fallback storage"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url)
        return self._redis

    async def store(self, payload: dict) -> None:
        redis = await self._get_redis()
        task_id = payload["metadata"]["project"]["task_id"]
        await redis.set(f"memory:task:{task_id}", json.dumps(payload))

    async def search(self, query: str, project_id=None, task_type=None, limit=10) -> list:
        # Implement search logic
        return []

    async def get(self, task_id: str) -> dict | None:
        redis = await self._get_redis()
        data = await redis.get(f"memory:task:{task_id}")
        return json.loads(data) if data else None

# Use custom storage
client = create_memory_mcp_client(
    project_id="my-project",
    project_name="My Project",
    fallback_storage=RedisFallbackStorage("redis://localhost:6379")
)
```

### Custom MCP Transport (HTTP Example)

```python
from memory_mcp_client import MCPTransport
import aiohttp

class HTTPMCPTransport:
    """HTTP-based MCP transport"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def store(self, payload: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/store",
                json=payload
            ) as response:
                return await response.json()

    async def vector_search(self, query: str, project_id=None, task_type=None, limit=10) -> list:
        async with aiohttp.ClientSession() as session:
            params = {"query": query, "limit": limit}
            if project_id:
                params["project_id"] = project_id
            if task_type:
                params["task_type"] = task_type
            async with session.get(
                f"{self.endpoint}/search",
                params=params
            ) as response:
                return await response.json()

    async def get_task(self, task_id: str) -> dict | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.endpoint}/task/{task_id}") as response:
                if response.status == 404:
                    return None
                return await response.json()

    async def ping(self) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.endpoint}/health") as response:
                return response.status == 200
```

## API Reference

### Intent Enum

```python
class Intent(str, Enum):
    IMPLEMENTATION = "implementation"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    RESEARCH = "research"
```

### AgentCategory Enum

```python
class AgentCategory(str, Enum):
    CORE_DEVELOPMENT = "core-development"
    TESTING_VALIDATION = "testing-validation"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    DOCUMENTATION = "documentation"
    SWARM_COORDINATION = "swarm-coordination"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESEARCH = "research"
    ORCHESTRATION = "orchestration"
    DEVOPS = "devops"
    CUSTOM = "custom"
```

### CircuitBreakerState Enum

```python
class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Fast-fail mode
    HALF_OPEN = "half_open"  # Testing recovery
```

### MemoryMCPClient Methods

| Method | Description |
|--------|-------------|
| `store(content, intent, user_id?, task_id?, additional_metadata?)` | Store data with automatic tagging |
| `vector_search(query, project_id?, task_type?, limit?)` | Semantic similarity search |
| `get_task_history(task_id, include_related?)` | Get task with related tasks |
| `health_check()` | Check MCP and circuit breaker health |
| `reset_circuit_breaker()` | Manually reset circuit breaker |

### Result Types

- **StoreResult**: status, storage, task_id, metadata, warning
- **TaskHistoryResult**: task, related_tasks, source, warning
- **HealthStatus**: status, degraded_mode, circuit_breaker_state, mcp_available, fallback_available, error, last_check

## WHO/WHEN/PROJECT/WHY Tags

All stored data includes these mandatory tags:

```json
{
  "who": {
    "agent_id": "backend-dev",
    "agent_category": "backend",
    "capabilities": ["REST API", "PostgreSQL"],
    "user_id": "system"
  },
  "when": {
    "iso_timestamp": "2026-01-10T12:00:00+00:00",
    "unix_timestamp": 1768075200,
    "readable": "2026-01-10 12:00:00 UTC"
  },
  "project": {
    "project_id": "my-project",
    "project_name": "My Project",
    "task_id": "TASK-001"
  },
  "why": {
    "intent": "implementation",
    "description": "Implementing new feature or functionality"
  }
}
```

## Circuit Breaker Behavior

1. **CLOSED** (normal): All calls pass through
2. After N failures (default 5): Transitions to **OPEN**
3. **OPEN**: Fast-fails all calls, returns CircuitBreakerError
4. After timeout (default 30s): Transitions to **HALF_OPEN**
5. **HALF_OPEN**: Allows limited test calls
   - Success: Returns to CLOSED
   - Failure: Returns to OPEN

## Dependencies

- Python 3.11+
- asyncio (stdlib)
- dataclasses (stdlib)
- typing (stdlib)
- json (stdlib)
- uuid (stdlib)
- enum (stdlib)
- datetime (stdlib)
- logging (stdlib)

No external dependencies required for core functionality. Optional integrations:
- `aioredis` for Redis cache/fallback
- `aiohttp` for HTTP MCP transport
- `asyncpg` for PostgreSQL fallback

## License

MIT - Part of the Context Cascade ecosystem.
