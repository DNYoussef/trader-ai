#!/usr/bin/env python3
"""
Telemetry Configuration System for Six Sigma Metrics
Real-time DPMO/RTY collection and monitoring
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import logging
from collections import deque, defaultdict


@dataclass
class TelemetryDataPoint:
    """Individual telemetry data point"""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class AlertRule:
    """Alert configuration"""
    metric_name: str
    threshold: float
    direction: str  # "above", "below"
    severity: str   # "info", "warning", "critical"
    callback: Optional[Callable] = None


class TelemetryCollector:
    """
    Collects and aggregates Six Sigma telemetry data
    Provides real-time monitoring and alerting
    """
    
    def __init__(self, config: Dict[str, Any], buffer_size: int = 1000):
        """Initialize telemetry collector"""
        self.config = config
        self.buffer_size = buffer_size
        
        # Data storage
        self.data_points: deque = deque(maxlen=buffer_size)
        self.aggregated_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Threading
        self.collection_queue: queue.Queue = queue.Queue()
        self.running = False
        self.collector_thread: Optional[threading.Thread] = None
        
        # Alerts
        self.alert_rules: List[AlertRule] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize from config
        self._setup_from_config()
    
    def _setup_from_config(self):
        """Setup collector from configuration"""
        telemetry_config = self.config.get('telemetry', {})
        
        # Setup alert rules from config
        for metric_name, metric_config in telemetry_config.get('metrics', {}).items():
            if 'alerts' in metric_config:
                for alert_config in metric_config['alerts']:
                    self.add_alert_rule(
                        metric_name=metric_name,
                        threshold=alert_config['threshold'],
                        direction=alert_config.get('direction', 'above'),
                        severity=alert_config['severity']
                    )
    
    def start_collection(self) -> None:
        """Start telemetry collection thread"""
        if not self.running:
            self.running = True
            self.collector_thread = threading.Thread(target=self._collection_worker)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            self.logger.info("Telemetry collection started")
    
    def stop_collection(self) -> None:
        """Stop telemetry collection thread"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join()
        self.logger.info("Telemetry collection stopped")
    
    def _collection_worker(self) -> None:
        """Background worker for processing telemetry data"""
        collection_interval = self.config.get('telemetry', {}).get('collection_interval', 300)
        
        while self.running:
            try:
                # Process queued data points
                while not self.collection_queue.empty():
                    data_point = self.collection_queue.get_nowait()
                    self._process_data_point(data_point)
                
                # Perform periodic aggregation
                self._perform_aggregation()
                
                # Check alert conditions
                self._check_alerts()
                
                time.sleep(collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in telemetry collection: {e}")
    
    def collect_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Collect a metric data point"""
        data_point = TelemetryDataPoint(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        # Add to queue for processing
        self.collection_queue.put(data_point)
        
        # Also add directly for immediate access
        self.data_points.append(data_point)
    
    def _process_data_point(self, data_point: TelemetryDataPoint) -> None:
        """Process individual data point"""
        # Add to aggregation buckets
        now = data_point.timestamp
        
        # Hourly aggregation
        hour_key = now.strftime("%Y-%m-%d-%H")
        self.aggregated_data[data_point.metric_name]["hourly"].append(data_point.value)
        
        # Daily aggregation
        day_key = now.strftime("%Y-%m-%d")
        self.aggregated_data[data_point.metric_name]["daily"].append(data_point.value)
        
        # Weekly aggregation
        week_start = now - timedelta(days=now.weekday())
        week_key = week_start.strftime("%Y-W%U")
        self.aggregated_data[data_point.metric_name]["weekly"].append(data_point.value)
    
    def _perform_aggregation(self) -> None:
        """Perform periodic aggregation of collected data"""
        # This could be expanded to calculate moving averages, percentiles, etc.
        # For now, we maintain the raw data for aggregation queries
        pass
    
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      direction: str = "above", severity: str = "warning",
                      callback: Optional[Callable] = None) -> None:
        """Add alert rule for a metric"""
        alert_rule = AlertRule(
            metric_name=metric_name,
            threshold=threshold,
            direction=direction,
            severity=severity,
            callback=callback
        )
        self.alert_rules.append(alert_rule)
    
    def _check_alerts(self) -> None:
        """Check alert conditions"""
        if not self.data_points:
            return
        
        # Get latest values for each metric
        latest_values = {}
        for data_point in reversed(self.data_points):
            if data_point.metric_name not in latest_values:
                latest_values[data_point.metric_name] = data_point.value
        
        # Check alert rules
        for rule in self.alert_rules:
            if rule.metric_name in latest_values:
                value = latest_values[rule.metric_name]
                triggered = False
                
                if rule.direction == "above" and value > rule.threshold:
                    triggered = True
                elif rule.direction == "below" and value < rule.threshold:
                    triggered = True
                
                if triggered:
                    alert = {
                        "metric_name": rule.metric_name,
                        "value": value,
                        "threshold": rule.threshold,
                        "direction": rule.direction,
                        "severity": rule.severity,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.alert_history.append(alert)
                    self.logger.warning(f"Alert triggered: {alert}")
                    
                    if rule.callback:
                        rule.callback(alert)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_data = [
            dp for dp in self.data_points 
            if dp.timestamp >= cutoff_time
        ]
        
        summary = {}
        
        # Group by metric
        metrics = defaultdict(list)
        for dp in recent_data:
            metrics[dp.metric_name].append(dp.value)
        
        # Calculate statistics
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "latest": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing"
                }
        
        return summary
    
    def get_aggregated_data(self, metric_name: str, aggregation: str = "hourly") -> List[float]:
        """Get aggregated data for a metric"""
        return self.aggregated_data.get(metric_name, {}).get(aggregation, [])
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]
    
    def export_data(self, output_dir: str = ".claude/.artifacts/sixsigma") -> str:
        """Export telemetry data to files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export raw data points
        raw_file = Path(output_dir) / f"telemetry_raw_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump([dp.to_dict() for dp in self.data_points], f, indent=2)
        
        # Export metrics summary
        summary_file = Path(output_dir) / f"telemetry_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "summary": self.get_metrics_summary(),
                "alerts": self.get_recent_alerts(),
                "export_timestamp": datetime.now().isoformat(),
                "total_data_points": len(self.data_points)
            }, f, indent=2)
        
        return str(summary_file)


class SixSigmaTelemetryManager:
    """
    Specialized telemetry manager for Six Sigma metrics
    Integrates with SixSigmaScorer for continuous monitoring
    """
    
    def __init__(self, config_path: str = "config/checks.yaml"):
        """Initialize with configuration"""
        from .sixsigma_scorer import SixSigmaScorer
        
        self.scorer = SixSigmaScorer(config_path)
        self.collector = TelemetryCollector(self.scorer.config)
        
        # Setup specific Six Sigma alerts
        self._setup_sixsigma_alerts()
    
    def _setup_sixsigma_alerts(self) -> None:
        """Setup Six Sigma specific alerts"""
        # DPMO alerts
        self.collector.add_alert_rule("dpmo", 10000, "above", "warning")
        self.collector.add_alert_rule("dpmo", 50000, "above", "critical")
        
        # RTY alerts
        self.collector.add_alert_rule("rty", 0.85, "below", "warning")
        self.collector.add_alert_rule("rty", 0.75, "below", "critical")
        
        # Sigma level alerts
        self.collector.add_alert_rule("sigma_level", 3.5, "below", "warning")
        self.collector.add_alert_rule("sigma_level", 3.0, "below", "critical")
    
    def start_monitoring(self) -> None:
        """Start continuous Six Sigma monitoring"""
        self.collector.start_collection()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.collector.stop_collection()
    
    def collect_sixsigma_metrics(self) -> None:
        """Collect current Six Sigma metrics"""
        metrics = self.scorer.calculate_comprehensive_metrics()
        
        # Collect core metrics
        self.collector.collect_metric("dpmo", metrics.dpmo, {"source": "sixsigma_scorer"})
        self.collector.collect_metric("rty", metrics.rty, {"source": "sixsigma_scorer"})
        self.collector.collect_metric("sigma_level", metrics.sigma_level, {"source": "sixsigma_scorer"})
        self.collector.collect_metric("process_capability", metrics.process_capability, {"source": "sixsigma_scorer"})
        
        # Collect stage-specific metrics
        for stage_name, yield_rate in metrics.stage_yields.items():
            self.collector.collect_metric(
                f"stage_yield_{stage_name.lower()}", 
                yield_rate, 
                {"stage": stage_name, "source": "sixsigma_scorer"}
            )
        
        # Collect defect category counts
        for category, count in metrics.defect_categories.items():
            self.collector.collect_metric(
                f"defects_{category}", 
                count, 
                {"category": category, "source": "sixsigma_scorer"}
            )
    
    def generate_telemetry_report(self) -> str:
        """Generate comprehensive telemetry report"""
        # Collect current metrics
        self.collect_sixsigma_metrics()
        
        # Export telemetry data
        return self.collector.export_data()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for Six Sigma dashboard"""
        return {
            "metrics_summary": self.collector.get_metrics_summary(),
            "recent_alerts": self.collector.get_recent_alerts(),
            "current_sixsigma": self.scorer.calculate_comprehensive_metrics().to_dict(),
            "trend_data": {
                "dpmo": self.collector.get_aggregated_data("dpmo", "hourly"),
                "rty": self.collector.get_aggregated_data("rty", "hourly"),
                "sigma_level": self.collector.get_aggregated_data("sigma_level", "hourly")
            }
        }


if __name__ == "__main__":
    # Test telemetry system
    print("Six Sigma Telemetry System Test")
    print("=" * 40)
    
    manager = SixSigmaTelemetryManager()
    
    # Start monitoring
    manager.start_monitoring()
    
    # Simulate some data collection
    for i in range(10):
        manager.collect_sixsigma_metrics()
        time.sleep(1)
    
    # Generate report
    report_file = manager.generate_telemetry_report()
    print(f"Telemetry report generated: {report_file}")
    
    # Get dashboard data
    dashboard_data = manager.get_dashboard_data()
    print(f"Dashboard metrics: {len(dashboard_data['metrics_summary'])} metrics")
    
    # Stop monitoring
    manager.stop_monitoring()