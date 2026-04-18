# Health Monitor Component

Async health polling with alerting for service monitoring.

## Features

- **HTTP health checks**: Configurable endpoints with status code validation
- **Custom check functions**: Support async and sync functions
- **Response time thresholds**: Degraded and unhealthy states based on latency
- **Retry logic**: Configurable retry count and delay
- **Alert callbacks**: With consecutive failure threshold and cooldown
- **Background monitoring**: Continuous polling with status change notifications
- **Async-first**: Built for asyncio with httpx

## Installation

Requires `httpx` for HTTP health checks:

```bash
pip install httpx
```

## Usage

### Basic Usage

```python
import asyncio
from library.components.utilities.health_monitor import (
    HealthMonitor,
    HealthState,
)

async def main():
    monitor = HealthMonitor()

    # Add HTTP health check
    monitor.add_http_check(
        "api",
        "https://api.example.com/health",
        timeout_seconds=5.0
    )

    # Run health checks
    status = await monitor.check_all()
    print(f"Healthy: {status.healthy}")
    print(f"State: {status.state.value}")

    for check in status.checks:
        print(f"  {check.name}: {check.state.value} - {check.message}")

asyncio.run(main())
```

### Custom Health Check Functions

```python
async def check_database():
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False

monitor = HealthMonitor()
monitor.add_check("database", check_database)
```

### With Alerting

```python
from library.components.utilities.health_monitor import AlertConfig

async def send_alert(service: str, state: HealthState, message: str):
    await slack.post(f"ALERT: {service} is {state.value}: {message}")

alert_config = AlertConfig(
    enabled=True,
    alert_fn=send_alert,
    consecutive_failures_threshold=3,
    cooldown_seconds=300.0,
)

monitor = HealthMonitor(alert_config=alert_config)
```

### Background Monitoring

```python
async def on_status_change(status: HealthStatus):
    if not status.healthy:
        print(f"System unhealthy: {status.unhealthy_services}")

await monitor.start_monitoring(
    interval_seconds=30.0,
    on_status_change=on_status_change
)

# Later, stop monitoring
await monitor.stop_monitoring()
```

## API Reference

### HealthMonitor

```python
class HealthMonitor:
    def add_check(name, check_fn, timeout_seconds=5.0, **kwargs) -> None
    def add_http_check(name, url, timeout_seconds=5.0, ...) -> None
    def remove_check(name) -> bool
    async def check(name) -> HealthCheckResult
    async def check_all() -> HealthStatus
    async def start_monitoring(interval_seconds=30.0, on_status_change=None) -> None
    async def stop_monitoring() -> None
    def get_last_result(name) -> Optional[HealthCheckResult]
    def get_all_results() -> Dict[str, HealthCheckResult]
    def get_status_summary() -> Dict[str, Any]
```
