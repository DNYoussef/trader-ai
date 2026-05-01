"""Theater detection compatibility package."""

from .continuous_monitor import ContinuousTheaterMonitor, MonitoringAlert, StakeholderUpdate
from .reality_validator import EvidenceItem, RealityAssessment, RealityValidationSystem
from .theater_detector import RealityValidationResult, TheaterDetector, TheaterPattern

__all__ = [
    "ContinuousTheaterMonitor",
    "EvidenceItem",
    "MonitoringAlert",
    "RealityAssessment",
    "RealityValidationResult",
    "RealityValidationSystem",
    "StakeholderUpdate",
    "TheaterDetector",
    "TheaterPattern",
]
