"""
Metric Collector Component

Prometheus-compatible metrics collector for code analysis.

Features:
- Counter, Gauge, Histogram metric types
- Prometheus exposition format
- Thread-safe metric updates
- Pre-configured quality metrics

References:
- https://github.com/prometheus/client_python

Example:
    from library.components.analysis.metric_collector import (
        MetricCollector,
        create_quality_collector,
    )

    collector = create_quality_collector()
    collector.counter("violations_total").labels(
        severity="high",
        analyzer="connascence",
    ).inc()

    print(collector.export())
"""

from .collector import (
    MetricCollector,
    Counter,
    Gauge,
    Histogram,
    MetricType,
    MetricValue,
    create_quality_collector,
)

__all__ = [
    "MetricCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricType",
    "MetricValue",
    "create_quality_collector",
]
