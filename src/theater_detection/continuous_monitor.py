"""Continuous theater monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .theater_detector import RealityValidationResult, TheaterPattern


@dataclass
class MonitoringAlert:
    category: str
    severity: str
    message: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StakeholderUpdate:
    update_type: str
    confidence_level: str
    summary: str
    created_at: datetime = field(default_factory=datetime.now)


class ContinuousTheaterMonitor:
    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.alerts: List[MonitoringAlert] = []
        self.stakeholder_updates: List[StakeholderUpdate] = []
        self.config = {
            "monitoring_intervals": {
                "performance": 300,
                "quality": 600,
                "security": 300,
                "compliance": 900,
                "architecture": 1800,
            },
            "alert_thresholds": {
                "theater_patterns_detected": 1,
                "critical_theater_patterns": 1,
            },
        }

    def _process_category_alerts(
        self,
        category: str,
        patterns: List[TheaterPattern],
        reality_validation: RealityValidationResult,
    ) -> None:
        for pattern in patterns:
            self.alerts.append(
                MonitoringAlert(
                    category=category,
                    severity=pattern.severity,
                    message=f"{pattern.pattern_type}: {pattern.recommendation}",
                )
            )
        if reality_validation.theater_risk >= 0.5:
            self.alerts.append(
                MonitoringAlert(
                    category=category,
                    severity="critical" if reality_validation.theater_risk >= 0.5 else "warning",
                    message="Reality validation score below threshold.",
                )
            )

    def _generate_weekly_stakeholder_update(self) -> None:
        critical_count = sum(1 for alert in self.alerts if alert.severity == "critical")
        confidence = "high" if critical_count == 0 else "medium" if critical_count < 3 else "low"
        self.stakeholder_updates.append(
            StakeholderUpdate(
                update_type="weekly",
                confidence_level=confidence,
                summary=f"{len(self.alerts)} alerts tracked.",
            )
        )

    def get_monitoring_status(self) -> Dict[str, Any]:
        return {
            "system_active": True,
            "categories_monitored": len(self.config["monitoring_intervals"]),
            "alerts": len(self.alerts),
            "stakeholder_updates": len(self.stakeholder_updates),
        }
