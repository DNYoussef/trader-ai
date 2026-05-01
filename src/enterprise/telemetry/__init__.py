"""
Six Sigma Telemetry Module

Provides enterprise-grade quality metrics including:
- DPMO (Defects Per Million Opportunities) calculations
- RTY (Rolled Throughput Yield) measurements
- Process capability analysis
- Quality gate enforcement
"""

from .six_sigma import SixSigmaTelemetry, SixSigmaMetrics

try:
    from .dpmo_calculator import DPMOCalculator
except ModuleNotFoundError:  # pragma: no cover - optional legacy module
    DPMOCalculator = None

try:
    from .rty_calculator import RTYCalculator
except ModuleNotFoundError:  # pragma: no cover - optional legacy module
    RTYCalculator = None

try:
    from .process_capability import ProcessCapabilityAnalyzer
except ModuleNotFoundError:  # pragma: no cover - optional legacy module
    ProcessCapabilityAnalyzer = None

__all__ = [
    "SixSigmaTelemetry",
    "SixSigmaMetrics",
    "DPMOCalculator", 
    "RTYCalculator",
    "ProcessCapabilityAnalyzer"
]
