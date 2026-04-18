from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = "components.analysis.metric_collector"


def _library_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / "catalog.json").exists() and (root / "components").exists():
            return root
        root = root.parent
    raise RuntimeError("Library root not found")


def _ensure_library_package() -> None:
    root = _library_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "library" not in sys.modules:
        library = types.ModuleType("library")
        library.__path__ = [str(root)]
        sys.modules["library"] = library


def _import_module():
    _ensure_library_package()
    try:
        return importlib.import_module(MODULE_PATH)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name in MODULE_PATH:
            raise
        pytest.skip(f"Missing dependency: {exc.name}")
    except ImportError as exc:
        pytest.skip(f"Import error: {exc}")


def test_counter_export_with_labels():
    module = _import_module()
    collector = module.MetricCollector()
    counter = collector.counter("requests_total", "Total requests", labels=["status"])
    counter.labels(status="ok").inc()
    counter.labels(status="ok").inc(2)
    output = collector.export()
    assert 'requests_total{status="ok"} 3' in output


def test_histogram_collects_buckets():
    module = _import_module()
    histogram = module.Histogram("latency_seconds", "Latency", buckets=[1, 2])
    histogram.observe(0.5)
    histogram.observe(1.5)
    values = {(tuple(sorted(v.labels.items()))): v.value for v in histogram.collect()}
    assert values[(("le", "1"),)] == 1
    assert values[(("le", "2"),)] == 3
    assert values[(("le", "+Inf"),)] == 2
    assert values[(("_type", "count"),)] == 2
    assert values[(("_type", "sum"),)] == 2.0
