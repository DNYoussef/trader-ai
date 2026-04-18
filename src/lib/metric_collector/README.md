# Metric Collector Component

Prometheus-compatible metrics collector for code analysis and quality tracking.

## Features

- Counter, Gauge, Histogram metric types
- Prometheus text exposition format
- Thread-safe metric updates
- Pre-configured quality metrics collector

## Usage

### Basic Metrics

```python
from library.components.analysis.metric_collector import (
    MetricCollector,
    Counter,
    Gauge,
    Histogram,
)

collector = MetricCollector()

# Create metrics
violations = collector.counter(
    "violations_total",
    "Total number of code violations",
    labels=["severity", "analyzer"],
)

quality_score = collector.gauge(
    "quality_score",
    "Current quality score",
    labels=["project"],
)

analysis_time = collector.histogram(
    "analysis_duration_seconds",
    "Analysis duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

# Record metrics
violations.labels(severity="high", analyzer="connascence").inc()
violations.labels(severity="medium", analyzer="nasa").inc(5)

quality_score.labels(project="trader-ai").set(85.5)

analysis_time.observe(1.23)

# Export in Prometheus format
print(collector.export())
```

### Output Format

```
# HELP violations_total Total number of code violations
# TYPE violations_total counter
violations_total{analyzer="connascence",severity="high"} 1
violations_total{analyzer="nasa",severity="medium"} 5

# HELP quality_score Current quality score
# TYPE quality_score gauge
quality_score{project="trader-ai"} 85.5

# HELP analysis_duration_seconds Analysis duration
# TYPE analysis_duration_seconds histogram
analysis_duration_seconds_bucket{le="0.1"} 0
analysis_duration_seconds_bucket{le="0.5"} 0
analysis_duration_seconds_bucket{le="1.0"} 0
analysis_duration_seconds_bucket{le="2.0"} 1
analysis_duration_seconds_bucket{le="5.0"} 1
analysis_duration_seconds_bucket{le="+Inf"} 1
analysis_duration_seconds_sum 1.23
analysis_duration_seconds_count 1
```

### Pre-configured Quality Collector

```python
from library.components.analysis.metric_collector import create_quality_collector

# Get pre-configured quality metrics
collector = create_quality_collector()

# Available metrics:
# - quality_violations_total (counter)
# - quality_score (gauge)
# - quality_sigma_level (gauge)
# - quality_dpmo (gauge)
# - quality_analysis_duration_seconds (histogram)
# - quality_files_analyzed (gauge)
# - quality_lines_of_code (gauge)

# Record
collector.counter("violations_total").labels(
    severity="high",
    analyzer="connascence",
    rule="CoI",
).inc()

collector.gauge("score").labels(analyzer="six_sigma").set(4.2)

# Export
print(collector.export())
```

### Export as Dictionary

```python
data = collector.to_dict()
# Returns:
# {
#   "counters": {"violations_total": [...]},
#   "gauges": {"quality_score": [...]},
#   "histograms": {"analysis_duration_seconds": [...]}
# }
```

## API Reference

### MetricCollector

Central registry for all metrics.

```python
collector = MetricCollector(namespace="myapp")

# Methods
collector.counter(name, description, labels=None) -> Counter
collector.gauge(name, description, labels=None) -> Gauge
collector.histogram(name, description, labels=None, buckets=None) -> Histogram
collector.export() -> str  # Prometheus format
collector.to_dict() -> Dict  # JSON-serializable dict
collector.reset()  # Clear all metrics
```

### Counter

Monotonically increasing counter.

```python
counter = Counter("errors_total", "Total errors", labels=["type"])

# Methods
counter.inc(amount=1.0, labels=None)  # Increment
counter.labels(type="timeout").inc()  # With labels
counter.get(labels=None) -> float     # Get current value
counter.collect() -> List[MetricValue]  # Collect all values
```

### Gauge

Value that can go up or down.

```python
gauge = Gauge("temperature", "Current temp")

# Methods
gauge.set(value, labels=None)    # Set value
gauge.inc(amount=1.0, labels=None)  # Increment
gauge.dec(amount=1.0, labels=None)  # Decrement
gauge.get(labels=None) -> float  # Get current value
```

### Histogram

Distribution of values in buckets.

```python
histogram = Histogram(
    "latency_seconds",
    "Request latency",
    buckets=[0.1, 0.5, 1.0, 2.0],
)

# Methods
histogram.observe(value, labels=None)  # Record a value
histogram.labels(endpoint="/api").observe(0.35)
```

## Sources

- [prometheus/client_python](https://github.com/prometheus/client_python)
- [Prometheus exposition format](https://prometheus.github.io/client_python/collector/custom/)
