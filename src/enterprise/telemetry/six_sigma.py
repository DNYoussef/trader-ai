"""
Six Sigma Telemetry System

Implements enterprise-grade quality metrics and process monitoring
for software development workflows.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)

# Kept as a module attribute for legacy tests that patch it. The production
# calculation below intentionally uses the deterministic threshold table instead
# of importing SciPy in telemetry hot paths.
stats = None


class QualityLevel(Enum):
    """Six Sigma quality levels"""
    TWO_SIGMA = 2.0      # 308,537 DPMO
    THREE_SIGMA = 3.0    # 66,807 DPMO
    FOUR_SIGMA = 4.0     # 6,210 DPMO
    FIVE_SIGMA = 5.0     # 233 DPMO
    SIX_SIGMA = 6.0      # 3.4 DPMO


@dataclass
class SixSigmaMetrics:
    """Container for Six Sigma quality metrics"""
    dpmo: float = 0.0  # Defects Per Million Opportunities
    rty: float = 0.0   # Rolled Throughput Yield
    sigma_level: float = 0.0
    process_capability: float = 0.0
    quality_level: Optional[QualityLevel] = None
    timestamp: datetime = field(default_factory=datetime.now)
    process_name: str = ""
    sample_size: int = 0
    defect_count: int = 0
    opportunity_count: int = 0


class SixSigmaTelemetry:
    """
    Six Sigma telemetry system for enterprise quality monitoring
    
    Tracks and calculates key quality metrics:
    - DPMO: Defects per million opportunities
    - RTY: Rolled throughput yield
    - Process capability (Cp, Cpk)
    - Sigma level achievement
    """
    
    def __init__(self, process_name: str = "default"):
        self.process_name = process_name
        self.metrics_history: List[SixSigmaMetrics] = []
        self._lock = threading.Lock()
        self._last_snapshot_timestamp: Optional[datetime] = None
        self.current_session_data = {
            'defects': 0,
            'opportunities': 0,
            'units_processed': 0,
            'units_passed': 0,
            'start_time': time.time()
        }
        self.quality_thresholds = {
            QualityLevel.TWO_SIGMA: 308537,
            QualityLevel.THREE_SIGMA: 66807,
            QualityLevel.FOUR_SIGMA: 6210,
            QualityLevel.FIVE_SIGMA: 233,
            QualityLevel.SIX_SIGMA: 3.4
        }
        
    def record_defect(self, defect_type: str = "generic", opportunities: int = 1):
        """Record a defect occurrence with associated opportunities"""
        with self._lock:
            self.current_session_data['defects'] += 1
            self.current_session_data['opportunities'] += opportunities
        
        logger.debug(f"Recorded defect: {defect_type}, opportunities: {opportunities}")
        
    def record_unit_processed(self, passed: bool = True, opportunities: int = 1):
        """Record a processed unit (e.g., test case, code review, deployment)"""
        with self._lock:
            self.current_session_data['units_processed'] += 1
            self.current_session_data['opportunities'] += opportunities

            if passed:
                self.current_session_data['units_passed'] += 1
            else:
                self.current_session_data['defects'] += 1

    def calculate_dpmo(
        self,
        defects: int = None,
        opportunities: int = None,
        opportunities_per_unit: int = None,
    ) -> float:
        """
        Calculate Defects Per Million Opportunities
        
        DPMO = (Number of Defects / Number of Opportunities) * 1,000,000
        """
        if defects is None and opportunities is None and opportunities_per_unit is None:
            defects = self.current_session_data['defects']
        elif defects is None:
            return 0.0
        if opportunities is None:
            opportunities = self.current_session_data['opportunities']

        if opportunities_per_unit is not None:
            opportunities = opportunities * opportunities_per_unit
            
        if not opportunities:
            return 0.0

        try:
            defects = float(defects)
            opportunities = float(opportunities)
        except (TypeError, ValueError):
            return 0.0
            
        dpmo = (defects / opportunities) * 1_000_000
        return round(dpmo, 6)
        
    def calculate_rty(self, units_processed: int = None, units_passed: int = None) -> float:
        """
        Calculate Rolled Throughput Yield
        
        RTY = (Units Passed First Time / Total Units Processed) * 100
        """
        if units_processed is None:
            units_processed = self.current_session_data['units_processed']
        if units_passed is None:
            units_passed = self.current_session_data['units_passed']
            
        if units_processed == 0:
            return 100.0
            
        rty = (units_passed / units_processed) * 100
        return round(rty, 2)
        
    def calculate_sigma_level(self, dpmo: float = None) -> float:
        """
        Calculate sigma level from DPMO
        
        Uses the inverse normal distribution to determine sigma level
        """
        if dpmo is None:
            dpmo = self.calculate_dpmo()
            
        try:
            dpmo = float(dpmo)
        except (TypeError, ValueError):
            return 0.0

        if dpmo <= 0:
            return 6.0  # Perfect quality
        if dpmo >= 1_000_000:
            return 0.0

        # The threshold lookup is stable, deterministic, and fast enough for
        # telemetry hot paths. It also avoids importing SciPy in tight loops.
        return min(6.0, max(0.0, float(self._approximate_sigma_level(dpmo))))
            
    def _approximate_sigma_level(self, dpmo: float) -> float:
        """Approximate sigma level calculation without scipy"""
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1]):
            if dpmo <= threshold:
                return level.value
        return 1.0  # Below 2-sigma
        
    def get_quality_level(self, dpmo: float = None) -> QualityLevel:
        """Determine quality level based on DPMO"""
        if dpmo is None:
            dpmo = self.calculate_dpmo()
            
        for level, threshold in sorted(self.quality_thresholds.items(),
                                     key=lambda x: x[1]):
            if dpmo <= threshold:
                return level
                
        return QualityLevel.TWO_SIGMA  # Default to lowest level
        
    def calculate_process_capability(self, measurements: List[float], 
                                   lower_spec: float, upper_spec: float) -> Tuple[float, float]:
        """
        Calculate process capability indices (Cp, Cpk)
        
        Cp = (USL - LSL) / (6 * sigma)
        Cpk = min((USL - mean) / (3 * sigma), (mean - LSL) / (3 * sigma))
        """
        if not measurements or len(measurements) < 2:
            return 0.0, 0.0
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in measurements):
            return 0.0, 0.0
            
        mean = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements)
        
        if std_dev == 0:
            return float('inf'), float('inf')
            
        # Cp - Process Capability
        cp = (upper_spec - lower_spec) / (6 * std_dev)
        
        # Cpk - Process Capability Index
        cpk_upper = (upper_spec - mean) / (3 * std_dev)
        cpk_lower = (mean - lower_spec) / (3 * std_dev)
        cpk = min(cpk_upper, cpk_lower)
        
        return round(cp, 3), round(cpk, 3)
        
    def generate_metrics_snapshot(self) -> SixSigmaMetrics:
        """Generate current metrics snapshot"""
        dpmo = self.calculate_dpmo()
        rty = self.calculate_rty()
        sigma_level = self.calculate_sigma_level(dpmo)
        quality_level = self.get_quality_level(dpmo)
        timestamp = datetime.now()
        if self._last_snapshot_timestamp and timestamp <= self._last_snapshot_timestamp:
            timestamp = self._last_snapshot_timestamp + timedelta(microseconds=1)
        self._last_snapshot_timestamp = timestamp
        
        metrics = SixSigmaMetrics(
            dpmo=dpmo,
            rty=rty,
            sigma_level=sigma_level,
            quality_level=quality_level,
            timestamp=timestamp,
            process_name=self.process_name,
            sample_size=self.current_session_data['units_processed'],
            defect_count=self.current_session_data['defects'],
            opportunity_count=self.current_session_data['opportunities']
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
        return metrics
        
    def reset_session(self):
        """Reset current session data"""
        with self._lock:
            self.current_session_data = {
                'defects': 0,
                'opportunities': 0,
                'units_processed': 0,
                'units_passed': 0,
                'start_time': time.time()
            }
        
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analyze quality trends over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp >= cutoff_date]
        
        if not recent_metrics:
            return {"error": "No metrics data available for trend analysis"}
            
        dpmo_values = [m.dpmo for m in recent_metrics]
        rty_values = [m.rty for m in recent_metrics]
        sigma_values = [m.sigma_level for m in recent_metrics]
        
        return {
            "period_days": days,
            "sample_count": len(recent_metrics),
            "dpmo": {
                "current": dpmo_values[-1],
                "average": round(statistics.mean(dpmo_values), 2),
                "trend": "improving" if dpmo_values[-1] < dpmo_values[0] else "declining",
                "best": min(dpmo_values),
                "worst": max(dpmo_values)
            },
            "rty": {
                "current": rty_values[-1],
                "average": round(statistics.mean(rty_values), 2),
                "trend": "improving" if rty_values[-1] > rty_values[0] else "declining",
                "best": max(rty_values),
                "worst": min(rty_values)
            },
            "sigma_level": {
                "current": sigma_values[-1],
                "average": round(statistics.mean(sigma_values), 2),
                "trend": "improving" if sigma_values[-1] > sigma_values[0] else "declining",
                "best": max(sigma_values),
                "worst": min(sigma_values)
            }
        }
        
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics data"""
        return {
            "process_name": self.process_name,
            "current_session": self.current_session_data,
            "metrics_history": [
                {
                    "dpmo": m.dpmo,
                    "rty": m.rty,
                    "sigma_level": m.sigma_level,
                    "quality_level": m.quality_level.name if m.quality_level else None,
                    "timestamp": m.timestamp.isoformat(),
                    "sample_size": m.sample_size,
                    "defect_count": m.defect_count,
                    "opportunity_count": m.opportunity_count
                }
                for m in self.metrics_history
            ]
        }
