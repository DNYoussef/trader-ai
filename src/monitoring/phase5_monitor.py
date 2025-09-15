#!/usr/bin/env python3
"""
Phase 5 Enhanced Monitoring System
Gary x Taleb Trading System

Monitors Super-Gary Vision Components:
- Narrative Gap signal frequency and effectiveness
- Brier score degradation alerts
- Enhanced DPI performance tracking
- Integrated signal quality metrics
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading

@dataclass
class NGSignalMetrics:
    """Narrative Gap signal metrics"""
    timestamp: datetime
    ng_score: float
    position_multiplier: float
    market_price: float
    consensus_forecast: float
    gary_estimate: float
    signal_strength: str  # "weak", "moderate", "strong"

@dataclass
class BrierMetrics:
    """Brier score calibration metrics"""
    timestamp: datetime
    brier_score: float
    prediction_count: int
    recent_accuracy: float
    risk_adjustment: float
    alert_level: str  # "normal", "warning", "critical"

@dataclass
class EnhancedDPIMetrics:
    """Enhanced DPI with wealth flow tracking"""
    timestamp: datetime
    base_dpi: float
    wealth_flow_score: float
    enhanced_dpi: float
    regime_detected: str
    confidence: float

class Phase5Monitor:
    """Enhanced monitoring for Phase 5 vision components"""

    def __init__(self, alert_threshold_brier=0.4, alert_threshold_ng_frequency=0.1):
        self.alert_threshold_brier = alert_threshold_brier
        self.alert_threshold_ng_frequency = alert_threshold_ng_frequency

        # Metrics storage
        self.ng_signals: List[NGSignalMetrics] = []
        self.brier_history: List[BrierMetrics] = []
        self.dpi_history: List[EnhancedDPIMetrics] = []

        # Alert system
        self.alerts: List[Dict] = []
        self.monitoring_active = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase5_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("Phase5Monitor")

    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.logger.info("Phase 5 monitoring started")

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.logger.info("Phase 5 monitoring stopped")

    def record_ng_signal(self, ng_score: float, multiplier: float,
                        market_price: float, consensus: float, gary_estimate: float):
        """Record Narrative Gap signal"""

        # Determine signal strength
        if ng_score < 0.02:
            strength = "weak"
        elif ng_score < 0.05:
            strength = "moderate"
        else:
            strength = "strong"

        metrics = NGSignalMetrics(
            timestamp=datetime.now(),
            ng_score=ng_score,
            position_multiplier=multiplier,
            market_price=market_price,
            consensus_forecast=consensus,
            gary_estimate=gary_estimate,
            signal_strength=strength
        )

        self.ng_signals.append(metrics)

        # Check for alerts
        self._check_ng_alerts(metrics)

        self.logger.info(f"NG Signal recorded: {ng_score:.4f} ({strength})")

    def record_brier_update(self, brier_score: float, prediction_count: int,
                           recent_accuracy: float, risk_adjustment: float):
        """Record Brier score update"""

        # Determine alert level
        if brier_score > self.alert_threshold_brier:
            alert_level = "critical"
        elif brier_score > self.alert_threshold_brier * 0.7:
            alert_level = "warning"
        else:
            alert_level = "normal"

        metrics = BrierMetrics(
            timestamp=datetime.now(),
            brier_score=brier_score,
            prediction_count=prediction_count,
            recent_accuracy=recent_accuracy,
            risk_adjustment=risk_adjustment,
            alert_level=alert_level
        )

        self.brier_history.append(metrics)

        # Check for alerts
        self._check_brier_alerts(metrics)

        self.logger.info(f"Brier update: {brier_score:.4f} ({alert_level})")

    def record_enhanced_dpi(self, base_dpi: float, flow_score: float,
                           enhanced_dpi: float, regime: str, confidence: float):
        """Record Enhanced DPI metrics"""

        metrics = EnhancedDPIMetrics(
            timestamp=datetime.now(),
            base_dpi=base_dpi,
            wealth_flow_score=flow_score,
            enhanced_dpi=enhanced_dpi,
            regime_detected=regime,
            confidence=confidence
        )

        self.dpi_history.append(metrics)

        self.logger.info(f"Enhanced DPI: {enhanced_dpi:.4f} (regime: {regime})")

    def _check_ng_alerts(self, metrics: NGSignalMetrics):
        """Check for Narrative Gap alerts"""

        # Check signal frequency (last hour)
        recent_signals = [s for s in self.ng_signals
                         if s.timestamp > datetime.now() - timedelta(hours=1)]

        if len(recent_signals) < self.alert_threshold_ng_frequency * 60:  # Assuming 1 signal/minute target
            self._create_alert("low_ng_frequency",
                             f"Low NG signal frequency: {len(recent_signals)} in last hour")

        # Check for extreme multipliers
        if metrics.position_multiplier > 1.2:
            self._create_alert("high_ng_multiplier",
                             f"High NG multiplier: {metrics.position_multiplier:.3f}x")

    def _check_brier_alerts(self, metrics: BrierMetrics):
        """Check for Brier score alerts"""

        if metrics.alert_level == "critical":
            self._create_alert("brier_critical",
                             f"Critical Brier score: {metrics.brier_score:.4f}")
        elif metrics.alert_level == "warning":
            self._create_alert("brier_warning",
                             f"Warning Brier score: {metrics.brier_score:.4f}")

        # Check for rapid degradation
        if len(self.brier_history) >= 5:
            recent_scores = [b.brier_score for b in self.brier_history[-5:]]
            if all(recent_scores[i] < recent_scores[i+1] for i in range(4)):
                self._create_alert("brier_degrading",
                                 "Brier score rapidly degrading over last 5 updates")

    def _create_alert(self, alert_type: str, message: str):
        """Create system alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": "high" if "critical" in alert_type else "medium"
        }

        self.alerts.append(alert)
        self.logger.warning(f"ALERT [{alert_type}]: {message}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Generate periodic reports
                self._generate_periodic_report()

                # Cleanup old data (keep last 24 hours)
                cutoff = datetime.now() - timedelta(hours=24)
                self.ng_signals = [s for s in self.ng_signals if s.timestamp > cutoff]
                self.brier_history = [b for b in self.brier_history if b.timestamp > cutoff]
                self.dpi_history = [d for d in self.dpi_history if d.timestamp > cutoff]
                self.alerts = [a for a in self.alerts
                              if datetime.fromisoformat(a["timestamp"]) > cutoff]

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def _generate_periodic_report(self):
        """Generate periodic status report"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)

        # NG signals in last hour
        recent_ng = [s for s in self.ng_signals if s.timestamp > last_hour]

        # Latest Brier score
        latest_brier = self.brier_history[-1] if self.brier_history else None

        # Latest DPI
        latest_dpi = self.dpi_history[-1] if self.dpi_history else None

        # Recent alerts
        recent_alerts = [a for a in self.alerts
                        if datetime.fromisoformat(a["timestamp"]) > last_hour]

        report = {
            "timestamp": now.isoformat(),
            "ng_signals_last_hour": len(recent_ng),
            "ng_avg_multiplier": sum(s.position_multiplier for s in recent_ng) / len(recent_ng) if recent_ng else 0,
            "current_brier_score": latest_brier.brier_score if latest_brier else None,
            "current_brier_alert": latest_brier.alert_level if latest_brier else None,
            "current_enhanced_dpi": latest_dpi.enhanced_dpi if latest_dpi else None,
            "current_regime": latest_dpi.regime_detected if latest_dpi else None,
            "recent_alerts": len(recent_alerts),
            "system_health": "good" if len(recent_alerts) == 0 else "warning"
        }

        # Save report
        with open("phase5_status_report.json", "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Status report generated: {report['system_health']} health")

    def get_current_status(self) -> Dict:
        """Get current system status"""
        latest_brier = self.brier_history[-1] if self.brier_history else None
        latest_dpi = self.dpi_history[-1] if self.dpi_history else None
        recent_ng = [s for s in self.ng_signals
                    if s.timestamp > datetime.now() - timedelta(hours=1)]

        return {
            "monitoring_active": self.monitoring_active,
            "ng_signals_last_hour": len(recent_ng),
            "current_brier_score": latest_brier.brier_score if latest_brier else None,
            "brier_alert_level": latest_brier.alert_level if latest_brier else None,
            "current_enhanced_dpi": latest_dpi.enhanced_dpi if latest_dpi else None,
            "active_alerts": len([a for a in self.alerts
                                if datetime.fromisoformat(a["timestamp"]) >
                                   datetime.now() - timedelta(hours=1)]),
            "total_ng_signals": len(self.ng_signals),
            "total_brier_updates": len(self.brier_history)
        }

    def export_metrics(self, filename: str = None):
        """Export all metrics to JSON file"""
        if filename is None:
            filename = f"phase5_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "ng_signals": [asdict(s) for s in self.ng_signals],
            "brier_history": [asdict(b) for b in self.brier_history],
            "dpi_history": [asdict(d) for d in self.dpi_history],
            "alerts": self.alerts
        }

        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Process data to convert datetime objects
        for signal in data["ng_signals"]:
            signal["timestamp"] = signal["timestamp"].isoformat()
        for brier in data["brier_history"]:
            brier["timestamp"] = brier["timestamp"].isoformat()
        for dpi in data["dpi_history"]:
            dpi["timestamp"] = dpi["timestamp"].isoformat()

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Metrics exported to {filename}")
        return filename

# Example usage and testing
if __name__ == "__main__":
    # Create monitor
    monitor = Phase5Monitor()

    # Start monitoring
    monitor.start_monitoring()

    print("Phase 5 monitoring system initialized")
    print("Monitor status:", monitor.get_current_status())

    # Simulate some activity
    print("\\nSimulating Phase 5 activity...")

    # Simulate NG signals
    monitor.record_ng_signal(0.05, 1.05, 100.0, 105.0, 110.0)
    monitor.record_ng_signal(0.03, 1.03, 150.0, 152.0, 155.0)

    # Simulate Brier updates
    monitor.record_brier_update(0.25, 10, 0.75, 0.75)
    monitor.record_brier_update(0.30, 12, 0.70, 0.70)

    # Simulate Enhanced DPI
    monitor.record_enhanced_dpi(0.6, 0.3, 0.78, "bullish", 0.85)

    print("Current status:", monitor.get_current_status())

    # Export metrics
    filename = monitor.export_metrics()
    print(f"Metrics exported to: {filename}")

    print("\\nPhase 5 monitoring system ready for production use")