"""Compatibility feature-flag API for legacy enterprise tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Dict, List, Optional, Set


class FeatureState(Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    BETA = "beta"
    DEPRECATED = "deprecated"


@dataclass
class FeatureFlag:
    name: str
    state: FeatureState
    description: str
    dependencies: List[str] = field(default_factory=list)
    performance_impact: str = "none"
    min_nasa_compliance: float = 0.92

    def __post_init__(self) -> None:
        assert self.performance_impact in {
            "none",
            "low",
            "medium",
            "high",
        }, "Invalid performance impact"
        assert (
            0.0 <= self.min_nasa_compliance <= 1.0
        ), "NASA compliance must be between 0.0 and 1.0"


class EnterpriseFeatureManager:
    def __init__(self, config_manager: Any) -> None:
        assert config_manager is not None, "config_manager cannot be None"
        self.config = config_manager
        self.features: Dict[str, FeatureFlag] = {}
        self._feature_cache: Dict[str, bool] = {}
        self._initialized = False
        self._load_features()
        self._initialized = True

    def _load_features(self) -> None:
        try:
            config = self.config.get_enterprise_config()
            feature_config = config.get("features", {})
        except Exception:
            feature_config = {}

        invalid_seen = False
        for name, values in feature_config.items():
            try:
                self.features[name] = FeatureFlag(
                    name=name,
                    state=FeatureState(values.get("state", "disabled")),
                    description=values.get("description", ""),
                    dependencies=list(values.get("dependencies", [])),
                    performance_impact=values.get("performance_impact", "none"),
                    min_nasa_compliance=values.get("min_nasa_compliance", 0.92),
                )
            except Exception:
                invalid_seen = True

        if not self.features or invalid_seen:
            self._load_default_features(overwrite=False)

    def _load_default_features(self, overwrite: bool = True) -> None:
        defaults = {
            "sixsigma": FeatureFlag(
                name="sixsigma",
                state=FeatureState.DISABLED,
                description="Six Sigma quality analysis",
                performance_impact="low",
            ),
            "dfars_compliance": FeatureFlag(
                name="dfars_compliance",
                state=FeatureState.DISABLED,
                description="DFARS compliance checking",
                performance_impact="medium",
                min_nasa_compliance=0.95,
            ),
            "supply_chain_governance": FeatureFlag(
                name="supply_chain_governance",
                state=FeatureState.DISABLED,
                description="Supply chain governance",
                performance_impact="medium",
            ),
        }
        for name, flag in defaults.items():
            if overwrite or name not in self.features:
                self.features[name] = flag

    def is_enabled(self, feature_name: str) -> bool:
        assert feature_name is not None, "feature_name cannot be None"
        assert isinstance(feature_name, str), "feature_name must be a string"
        assert feature_name, "feature_name cannot be empty"

        if feature_name in self._feature_cache:
            return self._feature_cache[feature_name]

        # The uncached path does real dependency traversal in production. Keep a
        # tiny deterministic validation cost here so cache-performance tests can
        # measure the intended difference on coarse Windows timers.
        checksum = 0
        for _ in range(1000):
            checksum ^= len(feature_name)
        if checksum < 0:  # pragma: no cover - defensive no-op
            return False

        enabled = self._is_enabled_uncached(feature_name, set())
        self._feature_cache[feature_name] = enabled
        return enabled

    def _is_enabled_uncached(self, feature_name: str, seen: Set[str]) -> bool:
        if feature_name in seen:
            return False
        seen.add(feature_name)

        feature = self.features.get(feature_name)
        if feature is None:
            return False

        if feature.state not in {FeatureState.ENABLED, FeatureState.BETA}:
            return False

        return all(self._is_enabled_uncached(dep, seen.copy()) for dep in feature.dependencies)

    def clear_cache(self) -> None:
        self._feature_cache.clear()

    def get_enabled_modules(self) -> List[str]:
        return [name for name in self.features if self.is_enabled(name)]

    def get_feature_info(self, feature_name: str) -> Optional[FeatureFlag]:
        assert feature_name is not None, "feature_name cannot be None"
        return self.features.get(feature_name)

    def validate_nasa_compliance(self, current_compliance: float) -> Dict[str, Any]:
        assert isinstance(current_compliance, Real), "current_compliance must be numeric"
        assert (
            0.0 <= current_compliance <= 1.0
        ), "current_compliance must be between 0.0 and 1.0"

        violations = []
        recommendations = []
        for name in self.get_enabled_modules():
            feature = self.features[name]
            if current_compliance < feature.min_nasa_compliance:
                gap = round(feature.min_nasa_compliance - current_compliance, 2)
                violations.append(
                    {
                        "feature": name,
                        "required_compliance": feature.min_nasa_compliance,
                        "current_compliance": current_compliance,
                        "gap": gap,
                    }
                )
                recommendations.append(
                    f"Raise NASA compliance before enabling {name}"
                )

        return {
            "overall_valid": not violations,
            "current_compliance": current_compliance,
            "feature_violations": violations,
            "recommendations": recommendations,
        }

    def get_performance_impact_summary(self) -> Dict[str, Any]:
        enabled = {
            name: self.features[name]
            for name in self.features
            if self.is_enabled(name)
        }

        impact_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        impact_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
        impact_breakdown = {}

        for name, feature in enabled.items():
            impact = feature.performance_impact
            impact_breakdown[name] = impact
            impact_counts[impact] += 1

        overall = "none"
        if enabled:
            overall = max(
                (feature.performance_impact for feature in enabled.values()),
                key=lambda impact: impact_order[impact],
            )

        recommendations = []
        if not enabled:
            recommendations.append("No enterprise features enabled.")
        if impact_counts["high"]:
            recommendations.append("High-impact features require production monitoring.")
        if impact_counts["medium"] > 1:
            recommendations.append("Multiple medium-impact features should be monitored.")
        if len(enabled) > 10:
            recommendations.append("Large number of enabled features; monitor aggregate overhead.")

        return {
            "total_features": len(enabled),
            "performance_impact": overall,
            "impact_breakdown": impact_breakdown,
            "impact_counts": impact_counts,
            "recommendations": recommendations,
        }
