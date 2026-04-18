"""
Metric Collector Component

Prometheus-compatible metrics collector for code analysis and quality tracking.

Based on prometheus_client patterns but simplified for our use case.

References:
- https://github.com/prometheus/client_python
- https://prometheus.github.io/client_python/collector/custom/

Features:
- Counter, Gauge, Histogram metric types
- Prometheus exposition format
- Thread-safe metric updates
- Custom collectors for analysis results

Example:
    from library.components.analysis.metric_collector import (
        MetricCollector,
        Counter,
        Gauge,
        Histogram,
    )

    collector = MetricCollector()

    # Define metrics
    violations_total = collector.counter(
        "violations_total",
        "Total number of violations",
        labels=["severity", "analyzer"],
    )

    # Record metrics
    violations_total.labels(severity="high", analyzer="connascence").inc()

    # Export
    print(collector.export())
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import threading


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None


class Counter:
    """
    Prometheus-style counter metric.

    Counters only go up (and reset on restart).

    Example:
        errors = Counter("errors_total", "Total errors", ["type"])
        errors.labels(type="timeout").inc()
        errors.labels(type="connection").inc(5)
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "CounterLabel":
        """Get a labeled counter instance."""
        return CounterLabel(self, kwargs)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the counter."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0)

    def collect(self) -> List[MetricValue]:
        """Collect all values."""
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(key),
                )
                for key, value in self._values.items()
            ]


class CounterLabel:
    """Helper for labeled counter access."""

    def __init__(self, counter: Counter, labels: Dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, amount: float = 1.0):
        """Increment the counter."""
        self._counter.inc(amount, self._labels)


class Gauge:
    """
    Prometheus-style gauge metric.

    Gauges can go up and down.

    Example:
        temp = Gauge("temperature_celsius", "Current temperature")
        temp.set(23.5)
        temp.inc(1.5)
        temp.dec(0.5)
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "GaugeLabel":
        """Get a labeled gauge instance."""
        return GaugeLabel(self, kwargs)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set the gauge value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment the gauge."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement the gauge."""
        self.inc(-amount, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._values.get(key, 0)

    def collect(self) -> List[MetricValue]:
        """Collect all values."""
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(key),
                )
                for key, value in self._values.items()
            ]


class GaugeLabel:
    """Helper for labeled gauge access."""

    def __init__(self, gauge: Gauge, labels: Dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float):
        self._gauge.set(value, self._labels)

    def inc(self, amount: float = 1.0):
        self._gauge.inc(amount, self._labels)

    def dec(self, amount: float = 1.0):
        self._gauge.dec(amount, self._labels)


class Histogram:
    """
    Prometheus-style histogram metric.

    Tracks distribution of values in buckets.

    Example:
        latency = Histogram(
            "request_latency_seconds",
            "Request latency",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        latency.observe(0.35)
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._bucket_counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = {}
        self._counts: Dict[tuple, int] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> "HistogramLabel":
        """Get a labeled histogram instance."""
        return HistogramLabel(self, kwargs)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value."""
        labels = labels or {}
        key = tuple(sorted(labels.items()))

        with self._lock:
            if key not in self._bucket_counts:
                self._bucket_counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0
                self._counts[key] = 0

            for bucket in self.buckets:
                if value > bucket:
                    continue
                self._bucket_counts[key][bucket] += 1

            self._sums[key] += value
            self._counts[key] += 1

    def collect(self) -> List[MetricValue]:
        """Collect all values as bucket metrics."""
        metrics = []

        with self._lock:
            for key, buckets in self._bucket_counts.items():
                labels = dict(key)
                cumulative = 0

                for bucket, count in sorted(buckets.items()):
                    cumulative += count
                    bucket_labels = {**labels, "le": str(bucket)}
                    metrics.append(MetricValue(
                        value=cumulative,
                        labels=bucket_labels,
                    ))

                # +Inf bucket
                metrics.append(MetricValue(
                    value=self._counts[key],
                    labels={**labels, "le": "+Inf"},
                ))

                # Sum and count
                metrics.append(MetricValue(
                    value=self._sums[key],
                    labels={**labels, "_type": "sum"},
                ))
                metrics.append(MetricValue(
                    value=self._counts[key],
                    labels={**labels, "_type": "count"},
                ))

        return metrics


class HistogramLabel:
    """Helper for labeled histogram access."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float):
        self._histogram.observe(value, self._labels)


class MetricCollector:
    """
    Central metrics collector and registry.

    Example:
        collector = MetricCollector()

        # Create metrics
        violations = collector.counter("violations_total", "Total violations")
        quality_score = collector.gauge("quality_score", "Current quality score")
        analysis_time = collector.histogram("analysis_seconds", "Analysis duration")

        # Record
        violations.inc()
        quality_score.set(85.5)
        analysis_time.observe(1.23)

        # Export in Prometheus format
        print(collector.export())
    """

    def __init__(self, namespace: str = ""):
        self.namespace = namespace
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        """Get full metric name with namespace."""
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._counters:
                self._counters[full_name] = Counter(full_name, description, labels)
            return self._counters[full_name]

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._gauges:
                self._gauges[full_name] = Gauge(full_name, description, labels)
            return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create or get a histogram metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(
                    full_name, description, labels, buckets
                )
            return self._histograms[full_name]

    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []

        # Counters
        for name, counter in sorted(self._counters.items()):
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            for value in counter.collect():
                labels_str = self._format_labels(value.labels)
                lines.append(f"{name}{labels_str} {value.value}")

        # Gauges
        for name, gauge in sorted(self._gauges.items()):
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            for value in gauge.collect():
                labels_str = self._format_labels(value.labels)
                lines.append(f"{name}{labels_str} {value.value}")

        # Histograms
        for name, histogram in sorted(self._histograms.items()):
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            for value in histogram.collect():
                suffix = ""
                labels = {k: v for k, v in value.labels.items() if k != "_type"}

                if "_type" in value.labels:
                    suffix = f"_{value.labels['_type']}"
                elif "le" in value.labels:
                    suffix = "_bucket"

                labels_str = self._format_labels(labels)
                lines.append(f"{name}{suffix}{labels_str} {value.value}")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as a dictionary."""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        for name, counter in self._counters.items():
            result["counters"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in counter.collect()
            ]

        for name, gauge in self._gauges.items():
            result["gauges"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in gauge.collect()
            ]

        for name, histogram in self._histograms.items():
            result["histograms"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in histogram.collect()
            ]

        return result

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Pre-configured quality metrics collector
def create_quality_collector() -> MetricCollector:
    """Create a pre-configured collector for code quality metrics."""
    collector = MetricCollector(namespace="quality")

    # Violation metrics
    collector.counter(
        "violations_total",
        "Total number of code violations",
        labels=["severity", "analyzer", "rule"],
    )

    # Quality scores
    collector.gauge(
        "score",
        "Current quality score (0-100)",
        labels=["analyzer"],
    )

    collector.gauge(
        "sigma_level",
        "Six Sigma quality level",
    )

    collector.gauge(
        "dpmo",
        "Defects per million opportunities",
    )

    # Analysis timing
    collector.histogram(
        "analysis_duration_seconds",
        "Duration of analysis runs",
        labels=["analyzer"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    # File metrics
    collector.gauge(
        "files_analyzed",
        "Number of files analyzed",
        labels=["language"],
    )

    collector.gauge(
        "lines_of_code",
        "Total lines of code",
        labels=["language"],
    )

    return collector
