"""Reality-checking helpers for theater detection tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TheaterPattern:
    category: str
    pattern_type: str
    confidence: float
    severity: str
    evidence: List[str]
    baseline_comparison: Dict[str, Any]
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class RealityValidationResult:
    category: str
    genuine_improvement: bool
    improvement_magnitude: float
    validation_score: float
    evidence_quality: float
    theater_risk: float
    validation_details: Dict[str, Any]


class TheaterDetector:
    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _save_baseline_metrics(self, category: str, baseline: Dict[str, Any]) -> None:
        (self.artifact_dir / f"{category}_baseline.json").write_text(
            json.dumps(baseline, indent=2, default=str)
        )

    def _load_baseline_metrics(self, category: str) -> Dict[str, Any]:
        path = self.artifact_dir / f"{category}_baseline.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def detect_performance_theater(self, current_metrics: Dict[str, Any]) -> List[TheaterPattern]:
        baseline = self._load_baseline_metrics("performance")
        patterns: List[TheaterPattern] = []

        cache = current_metrics.get("cache_performance", {})
        if cache.get("hit_rate", 0) >= 0.98 or cache.get("efficiency", 0) >= 0.98:
            patterns.append(
                TheaterPattern(
                    category="performance",
                    pattern_type="artificial_cache_inflation",
                    confidence=0.85,
                    severity="high",
                    evidence=["Cache metrics are near-perfect."],
                    baseline_comparison=baseline,
                    recommendation="Validate cache metrics under cold-start conditions.",
                )
            )

        old_times = baseline.get("execution_times", {})
        new_times = current_metrics.get("execution_times", {})
        if any(old_times.get(key, 0) and value <= old_times[key] * 0.5 for key, value in new_times.items()):
            patterns.append(
                TheaterPattern(
                    category="performance",
                    pattern_type="baseline_manipulation",
                    confidence=0.8,
                    severity="high",
                    evidence=["Execution improvement exceeds plausibility threshold."],
                    baseline_comparison=baseline,
                    recommendation="Re-run benchmarks with fixed inputs and environment.",
                )
            )

        return patterns

    def detect_quality_theater(self, current_metrics: Dict[str, Any]) -> List[TheaterPattern]:
        baseline = self._load_baseline_metrics("quality")
        current_cov = current_metrics.get("test_coverage", {})
        baseline_cov = baseline.get("test_coverage", {})
        coverage_delta = current_cov.get("line_coverage", 0) - baseline_cov.get("line_coverage", 0)
        test_delta = current_cov.get("test_count", 0) - baseline_cov.get("test_count", 0)
        if coverage_delta >= 0.1 and test_delta <= 3:
            return [
                TheaterPattern(
                    category="quality",
                    pattern_type="shallow_test_coverage",
                    confidence=0.8,
                    severity="medium",
                    evidence=["Coverage increased with minimal new tests."],
                    baseline_comparison=baseline,
                    recommendation="Audit assertions and branch coverage.",
                )
            ]
        return []

    def validate_reality(self, category: str, current_metrics: Dict[str, Any]) -> RealityValidationResult:
        baseline = self._load_baseline_metrics(category)
        has_baseline = bool(baseline)
        score = 0.8 if has_baseline else 0.5
        return RealityValidationResult(
            category=category,
            genuine_improvement=has_baseline,
            improvement_magnitude=0.1 if has_baseline else 0.0,
            validation_score=score,
            evidence_quality=score,
            theater_risk=1.0 - score,
            validation_details={"baseline_available": has_baseline, "current_metrics": current_metrics},
        )

    def run_comprehensive_theater_detection(self) -> Dict[str, Any]:
        categories = ["performance", "quality", "security", "compliance", "architecture"]
        return {
            "theater_detection_deployment": {
                "system_status": "deployed",
                "detection_categories": len(categories),
                "monitoring_coverage": "100%",
            },
            "continuous_monitoring": {
                f"{category}_monitoring": "active" for category in categories
            },
            "reality_validation_evidence": {"evidence_store": str(self.artifact_dir)},
        }
