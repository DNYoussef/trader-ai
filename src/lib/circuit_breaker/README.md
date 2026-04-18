# Circuit Breaker Component

Generic circuit breaker with exponential backoff for fault tolerance and graceful degradation.

## Features

- **Three-state pattern**: CLOSED, OPEN, HALF_OPEN
- **Failure threshold**: Trip after N consecutive failures
- **Failure rate**: Trip when failure rate exceeds threshold
- **Exponential backoff**: Progressive recovery timing
- **Success threshold**: Require N successes to close
- **Callbacks**: Trip and recovery notifications
- **Async-first**: Built for asyncio operations
- **Thread-safe**: Safe for concurrent use

## LEGO Integration

This component imports shared types from `library.common.types`:

```python
from library.common.types import ValidationResult
```

## Installation

```python
from library.components.utilities.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerManager,
)
```

## Usage

### Basic Usage

```python
import asyncio
from library.components.utilities.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
)

async def fetch_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    config = CircuitBreakerConfig(
        failure_threshold=3,
        timeout_duration=30
    )
    breaker = CircuitBreaker("api_client", config)

    try:
        result = await breaker.call(fetch_data, "https://api.example.com/data")
        print(result)
    except CircuitBreakerError as e:
        print(f"Circuit open, retry after {e.retry_after}s")

asyncio.run(main())
```

### With Exponential Backoff

```python
config = CircuitBreakerConfig(
    failure_threshold=5,
    timeout_duration=60,           # Initial backoff: 60s
    exponential_backoff=True,
    backoff_multiplier=2.0,        # 60s -> 120s -> 240s
    max_backoff_seconds=300,       # Cap at 5 minutes
)
breaker = CircuitBreaker("external_service", config)
```

### Using Callbacks

```python
async def on_trip(name: str, state: CircuitState):
    print(f"Alert: Circuit {name} tripped to {state.value}")
    await send_alert_to_ops(name)

async def on_recovery(name: str, state: CircuitState):
    print(f"Notice: Circuit {name} recovered to {state.value}")

breaker = CircuitBreaker("payment_gateway")
breaker.register_trip_callback(on_trip)
breaker.register_recovery_callback(on_recovery)
```

### Circuit Breaker Manager

```python
from library.components.utilities.circuit_breaker import CircuitBreakerManager

async def main():
    manager = CircuitBreakerManager()

    # Create multiple breakers
    api_breaker = await manager.create("api_gateway")
    db_breaker = await manager.create("database", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_duration=15
    ))

    # Check system health
    status = manager.get_system_status()
    print(f"Open breakers: {status['open_breakers']}")

    # Check if any are open
    if manager.any_open():
        open_list = manager.get_open_breakers()
        print(f"Warning: These circuits are open: {open_list}")

    # Get or create pattern
    breaker = await manager.get_or_create("cache_service")

asyncio.run(main())
```

### Getting Status

```python
status = breaker.get_status()
print(f"State: {status.state.value}")
print(f"Failures: {status.failure_count}/{status.failure_threshold}")
print(f"Current backoff: {status.current_backoff}s")
print(f"Total requests: {status.metrics.total_requests}")
print(f"Avg response time: {status.metrics.average_response_time:.2f}ms")

# Convert to dict for JSON
status_dict = status.to_dict()

# Convert to ValidationResult for pipeline integration
validation = status.to_validation_result()
if not validation.valid:
    print(f"Errors: {validation.errors}")
```

### Manual Control

```python
# Force open for maintenance
await breaker.force_open()

# Force close after manual fix
await breaker.force_close()

# Full reset
breaker.reset()
```

## API Reference

### CircuitBreakerConfig

```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5           # Failures to trip
    failure_rate_threshold: float = 0.5  # Rate to trip (0.0-1.0)
    success_threshold: int = 3           # Successes to close
    failure_window_seconds: int = 60     # Failure counting window
    timeout_duration: int = 60           # Base timeout in OPEN
    half_open_max_calls: int = 3         # Test calls in HALF_OPEN
    exponential_backoff: bool = True     # Enable backoff
    backoff_multiplier: float = 2.0      # Backoff factor
    max_backoff_seconds: int = 300       # Maximum backoff
    min_requests_for_rate: int = 10      # Min requests for rate calc
    name: str = "default"                # Name for logging
```

### CircuitBreaker

```python
class CircuitBreaker:
    # Properties
    name: str
    config: CircuitBreakerConfig
    state: CircuitState
    is_closed: bool
    is_open: bool
    is_half_open: bool

    # Methods
    async def call(func, *args, **kwargs) -> T
    def get_status() -> CircuitBreakerStatus
    def register_trip_callback(callback: Callable) -> None
    def register_recovery_callback(callback: Callable) -> None
    async def force_open() -> None
    async def force_close() -> None
    def reset() -> None
```

### CircuitBreakerManager

```python
class CircuitBreakerManager:
    async def create(name: str, config: Optional[Config]) -> CircuitBreaker
    async def get(name: str) -> Optional[CircuitBreaker]
    async def get_or_create(name: str, config: Optional[Config]) -> CircuitBreaker
    async def remove(name: str) -> bool
    def get_system_status() -> Dict[str, Any]
    def get_open_breakers() -> List[str]
    def any_open() -> bool
    async def shutdown_all() -> None
    def reset_all() -> None
```

## State Transitions

```
                    +----------+
                    | CLOSED   |
                    | (Normal) |
                    +----+-----+
                         |
            failure_threshold reached
                         |
                         v
                    +----------+
                    |   OPEN   |
                    | (Block)  |
                    +----+-----+
                         |
              timeout_duration elapsed
                         |
                         v
                    +-----------+
                    | HALF_OPEN |
                    | (Testing) |
                    +-----+-----+
                         |
            +------------+------------+
            |                         |
      success_threshold          any failure
            |                         |
            v                         v
       +----------+              +----------+
       | CLOSED   |              |   OPEN   |
       +----------+              +----------+
```

## Integration Example

```python
from library.components.utilities.circuit_breaker import CircuitBreaker
from library.components.utilities.health_monitor import HealthMonitor

# Circuit breaker for API calls
api_breaker = CircuitBreaker("api_service")

# Register with health monitor
async def check_circuit_health():
    status = api_breaker.get_status()
    return status.state != CircuitState.OPEN

monitor = HealthMonitor()
monitor.register_check("api_circuit", check_circuit_health)
```

## Source

Extracted and generalized from:
- `memory-mcp-triple-system` circuit breaker
- `trader-ai` circuit breaker patterns
