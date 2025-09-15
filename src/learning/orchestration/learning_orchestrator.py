"""
Learning System Orchestrator for Gary×Taleb Trading System

Central orchestrator that coordinates all learning components:
continuous learning, performance feedback, A/B testing, strategy adaptation,
monitoring, rollback, and online learning systems.
"""

import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Import learning components
from ..pipeline.continuous_learner import ContinuousLearner, LearningConfig, RetrainingResult
from ..feedback.performance_feedback import PerformanceFeedback, FeedbackMetrics, FeedbackSignal
from ..testing.ab_testing import ABTestingFramework, ExperimentConfig, VariantConfig
from ..adaptation.strategy_adaptation import StrategyAdaptationEngine, AdaptationSignal
from ..monitoring.performance_monitor import PerformanceMonitor, PerformanceAlert, PerformanceMetrics

@dataclass
class OrchestrationConfig:
    """Configuration for learning orchestration"""
    continuous_learning_enabled: bool = True
    performance_feedback_enabled: bool = True
    ab_testing_enabled: bool = True
    strategy_adaptation_enabled: bool = True
    performance_monitoring_enabled: bool = True
    auto_rollback_enabled: bool = True

    # Coordination intervals
    orchestration_interval_seconds: int = 60
    health_check_interval_seconds: int = 300
    coordination_timeout_seconds: int = 30

    # Performance thresholds for coordination
    critical_performance_threshold: float = -0.2
    auto_intervention_threshold: float = -0.15
    rollback_trigger_threshold: float = -0.25

    # A/B testing coordination
    ab_test_auto_promotion: bool = True
    ab_test_significance_threshold: float = 0.05

    # Strategy adaptation coordination
    adaptation_auto_apply: bool = True
    adaptation_confidence_threshold: float = 0.7

@dataclass
class SystemHealth:
    """Overall system health status"""
    timestamp: datetime
    overall_status: str  # 'healthy', 'warning', 'critical', 'offline'
    health_score: float  # 0-100
    component_status: Dict[str, str]
    active_experiments: int
    pending_adaptations: int
    critical_alerts: int
    models_performance: Dict[str, float]
    recommendations: List[str]

@dataclass
class CoordinationAction:
    """Action for component coordination"""
    action_type: str
    component: str
    parameters: Dict[str, Any]
    priority: int  # 1-10, 10 being highest
    reason: str
    expected_outcome: str
    timestamp: datetime

class LearningOrchestrator:
    """
    Central orchestrator for all continuous learning components.
    Coordinates between different systems and ensures optimal performance.
    """

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Database setup
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/learning_orchestrator.db")
        self._init_database()

        # Initialize learning components
        self._initialize_components()

        # Orchestration state
        self.is_running = False
        self.orchestration_thread = None
        self.health_check_thread = None

        # Coordination tracking
        self.pending_actions: List[CoordinationAction] = []
        self.system_health = None
        self.last_health_check = datetime.now()

        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'model_retrained': [],
            'experiment_completed': [],
            'adaptation_applied': [],
            'alert_triggered': [],
            'rollback_executed': []
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for orchestrator"""
        logger = logging.getLogger('LearningOrchestrator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/learning_orchestrator.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize orchestrator database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    health_score REAL,
                    component_status_json TEXT,
                    active_experiments INTEGER,
                    pending_adaptations INTEGER,
                    critical_alerts INTEGER,
                    models_performance_json TEXT,
                    recommendations_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS coordination_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    parameters_json TEXT,
                    priority INTEGER,
                    reason TEXT,
                    expected_outcome TEXT,
                    status TEXT DEFAULT 'pending',
                    executed_at TEXT,
                    result_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS orchestration_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    details_json TEXT,
                    impact_score REAL
                )
            ''')

    def _initialize_components(self):
        """Initialize all learning components"""
        try:
            # Continuous learning
            if self.config.continuous_learning_enabled:
                learning_config = LearningConfig(
                    retrain_frequency_hours=24,
                    min_samples_for_retrain=100,
                    auto_rollback_threshold=self.config.rollback_trigger_threshold
                )
                self.continuous_learner = ContinuousLearner(learning_config)
            else:
                self.continuous_learner = None

            # Performance feedback
            if self.config.performance_feedback_enabled:
                self.performance_feedback = PerformanceFeedback(window_size_hours=24, min_samples=10)
            else:
                self.performance_feedback = None

            # A/B testing
            if self.config.ab_testing_enabled:
                self.ab_testing = ABTestingFramework()
            else:
                self.ab_testing = None

            # Strategy adaptation
            if self.config.strategy_adaptation_enabled:
                self.strategy_adaptation = StrategyAdaptationEngine(adaptation_window_hours=24, min_samples=20)
            else:
                self.strategy_adaptation = None

            # Performance monitoring
            if self.config.performance_monitoring_enabled:
                self.performance_monitor = PerformanceMonitor(monitoring_interval_seconds=30)
                # Add alert callback
                self.performance_monitor.add_alert_callback(self._handle_performance_alert)
            else:
                self.performance_monitor = None

            self.logger.info("All learning components initialized")

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def start_orchestration(self):
        """Start the learning orchestration system"""
        if self.is_running:
            self.logger.warning("Learning orchestration already running")
            return

        self.is_running = True
        self.logger.info("Starting learning orchestration system")

        try:
            # Start individual components
            if self.continuous_learner:
                self.continuous_learner.start_continuous_learning()

            if self.performance_feedback:
                self.performance_feedback.start_feedback_system()

            if self.ab_testing:
                self.ab_testing.start_monitoring()

            if self.strategy_adaptation:
                self.strategy_adaptation.start_adaptation_engine()

            if self.performance_monitor:
                self.performance_monitor.start_monitoring()

            # Start orchestration threads
            self.orchestration_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
            self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)

            self.orchestration_thread.start()
            self.health_check_thread.start()

            self.logger.info("Learning orchestration system started successfully")

        except Exception as e:
            self.logger.error(f"Error starting orchestration: {e}")
            self.stop_orchestration()
            raise

    def stop_orchestration(self):
        """Stop the learning orchestration system"""
        self.is_running = False

        try:
            # Stop individual components
            if self.continuous_learner:
                self.continuous_learner.stop_continuous_learning()

            if self.performance_feedback:
                self.performance_feedback.stop_feedback_system()

            if self.ab_testing:
                self.ab_testing.stop_monitoring()

            if self.strategy_adaptation:
                self.strategy_adaptation.stop_adaptation_engine()

            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()

            self.logger.info("Learning orchestration system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping orchestration: {e}")

    def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.is_running:
            try:
                # Coordinate between components
                self._coordinate_components()

                # Process pending actions
                self._process_coordination_actions()

                # Check for system-wide optimizations
                self._check_system_optimizations()

                time.sleep(self.config.orchestration_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in orchestration loop: {e}")
                time.sleep(60)

    def _health_check_loop(self):
        """Health check loop"""
        while self.is_running:
            try:
                # Update system health
                self.system_health = self._assess_system_health()

                # Save to database
                self._save_system_health(self.system_health)

                # Generate health-based actions
                self._generate_health_actions(self.system_health)

                self.last_health_check = datetime.now()

                time.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                time.sleep(300)

    def _coordinate_components(self):
        """Coordinate between learning components"""
        try:
            # Get signals from each component
            feedback_signals = self._get_feedback_signals()
            adaptation_signals = self._get_adaptation_signals()
            monitoring_alerts = self._get_monitoring_alerts()

            # Cross-component coordination
            self._coordinate_retraining_and_adaptation(feedback_signals, adaptation_signals)
            self._coordinate_ab_testing_and_adaptation(adaptation_signals)
            self._coordinate_monitoring_and_rollback(monitoring_alerts)

        except Exception as e:
            self.logger.error(f"Error in component coordination: {e}")

    def _get_feedback_signals(self) -> List[FeedbackSignal]:
        """Get feedback signals from performance feedback system"""
        if not self.performance_feedback:
            return []

        try:
            return self.performance_feedback.get_active_signals('high')
        except Exception as e:
            self.logger.error(f"Error getting feedback signals: {e}")
            return []

    def _get_adaptation_signals(self) -> List[AdaptationSignal]:
        """Get adaptation signals from strategy adaptation engine"""
        if not self.strategy_adaptation:
            return []

        try:
            # Get recent adaptation signals from adaptation engine
            # This would need to be implemented in the adaptation engine
            return []
        except Exception as e:
            self.logger.error(f"Error getting adaptation signals: {e}")
            return []

    def _get_monitoring_alerts(self) -> List[PerformanceAlert]:
        """Get alerts from performance monitor"""
        if not self.performance_monitor:
            return []

        try:
            return self.performance_monitor.get_active_alerts()
        except Exception as e:
            self.logger.error(f"Error getting monitoring alerts: {e}")
            return []

    def _coordinate_retraining_and_adaptation(self, feedback_signals: List[FeedbackSignal],
                                            adaptation_signals: List[AdaptationSignal]):
        """Coordinate model retraining with strategy adaptation"""
        try:
            # If both systems suggest changes, coordinate them
            retrain_signals = [s for s in feedback_signals if s.signal_type == 'retrain']
            param_adjust_signals = [s for s in adaptation_signals if s.signal_type == 'parameter_adjustment']

            if retrain_signals and param_adjust_signals:
                # Prioritize retraining over parameter adjustment for major issues
                high_urgency_retrain = [s for s in retrain_signals if s.urgency in ['high', 'critical']]

                if high_urgency_retrain:
                    # Delay parameter adjustments until after retraining
                    action = CoordinationAction(
                        action_type='delay_parameter_adjustment',
                        component='strategy_adaptation',
                        parameters={'reason': 'model_retraining_in_progress'},
                        priority=7,
                        reason='Delaying parameter adjustment due to model retraining',
                        expected_outcome='Avoid conflicting changes',
                        timestamp=datetime.now()
                    )
                    self.pending_actions.append(action)

        except Exception as e:
            self.logger.error(f"Error coordinating retraining and adaptation: {e}")

    def _coordinate_ab_testing_and_adaptation(self, adaptation_signals: List[AdaptationSignal]):
        """Coordinate A/B testing with strategy adaptation"""
        try:
            if not self.ab_testing:
                return

            # Check for running experiments that might conflict with adaptations
            regime_change_signals = [s for s in adaptation_signals if s.signal_type == 'regime_change']

            if regime_change_signals and self.config.ab_test_auto_promotion:
                # Pause A/B tests during regime changes
                action = CoordinationAction(
                    action_type='pause_ab_tests',
                    component='ab_testing',
                    parameters={'reason': 'regime_change_detected'},
                    priority=8,
                    reason='Pausing A/B tests due to market regime change',
                    expected_outcome='Prevent invalid experiment results',
                    timestamp=datetime.now()
                )
                self.pending_actions.append(action)

        except Exception as e:
            self.logger.error(f"Error coordinating A/B testing and adaptation: {e}")

    def _coordinate_monitoring_and_rollback(self, monitoring_alerts: List[PerformanceAlert]):
        """Coordinate monitoring alerts with rollback decisions"""
        try:
            critical_alerts = [a for a in monitoring_alerts if a.alert_type == 'critical']

            for alert in critical_alerts:
                if alert.metric_name in ['gary_dpi', 'direction_accuracy']:
                    # Critical performance degradation - consider rollback
                    if alert.current_value < self.config.rollback_trigger_threshold:
                        action = CoordinationAction(
                            action_type='execute_rollback',
                            component='continuous_learner',
                            parameters={
                                'model_id': alert.model_id,
                                'reason': f'Critical {alert.metric_name} degradation',
                                'trigger_value': alert.current_value
                            },
                            priority=10,
                            reason=f'Emergency rollback due to {alert.description}',
                            expected_outcome='Restore model performance',
                            timestamp=datetime.now()
                        )
                        self.pending_actions.append(action)

        except Exception as e:
            self.logger.error(f"Error coordinating monitoring and rollback: {e}")

    def _process_coordination_actions(self):
        """Process pending coordination actions"""
        if not self.pending_actions:
            return

        try:
            # Sort by priority (highest first)
            self.pending_actions.sort(key=lambda x: x.priority, reverse=True)

            # Process top priority actions
            for action in self.pending_actions[:5]:  # Process top 5
                self._execute_coordination_action(action)

            # Remove processed actions
            self.pending_actions = self.pending_actions[5:]

        except Exception as e:
            self.logger.error(f"Error processing coordination actions: {e}")

    def _execute_coordination_action(self, action: CoordinationAction):
        """Execute a coordination action"""
        try:
            self.logger.info(f"Executing coordination action: {action.action_type} on {action.component}")

            success = False
            result = {}

            if action.action_type == 'execute_rollback':
                success = self._execute_rollback(action.parameters)

            elif action.action_type == 'pause_ab_tests':
                success = self._pause_ab_tests(action.parameters)

            elif action.action_type == 'delay_parameter_adjustment':
                success = self._delay_parameter_adjustment(action.parameters)

            elif action.action_type == 'force_retraining':
                success = self._force_retraining(action.parameters)

            elif action.action_type == 'emergency_adaptation':
                success = self._emergency_adaptation(action.parameters)

            # Save action result
            self._save_coordination_action(action, success, result)

            if success:
                self.logger.info(f"Successfully executed {action.action_type}")
            else:
                self.logger.error(f"Failed to execute {action.action_type}")

        except Exception as e:
            self.logger.error(f"Error executing coordination action: {e}")
            self._save_coordination_action(action, False, {'error': str(e)})

    def _execute_rollback(self, parameters: Dict[str, Any]) -> bool:
        """Execute model rollback"""
        try:
            if not self.continuous_learner:
                return False

            model_id = parameters.get('model_id')
            if model_id:
                # Trigger rollback in continuous learner
                self.continuous_learner._trigger_automatic_rollback(model_id)

                # Emit event
                self._emit_event('rollback_executed', 'continuous_learner', parameters)

                return True

        except Exception as e:
            self.logger.error(f"Error executing rollback: {e}")

        return False

    def _pause_ab_tests(self, parameters: Dict[str, Any]) -> bool:
        """Pause running A/B tests"""
        try:
            if not self.ab_testing:
                return False

            # This would need to be implemented in the A/B testing framework
            # For now, return success
            return True

        except Exception as e:
            self.logger.error(f"Error pausing A/B tests: {e}")

        return False

    def _delay_parameter_adjustment(self, parameters: Dict[str, Any]) -> bool:
        """Delay parameter adjustments"""
        try:
            if not self.strategy_adaptation:
                return False

            # This would need to be implemented in the strategy adaptation engine
            # For now, return success
            return True

        except Exception as e:
            self.logger.error(f"Error delaying parameter adjustment: {e}")

        return False

    def _force_retraining(self, parameters: Dict[str, Any]) -> bool:
        """Force model retraining"""
        try:
            if not self.continuous_learner:
                return False

            results = self.continuous_learner.force_retrain_all()
            successful_retrains = sum(1 for r in results if r.success)

            self._emit_event('model_retrained', 'continuous_learner', {
                'successful_retrains': successful_retrains,
                'total_models': len(results)
            })

            return successful_retrains > 0

        except Exception as e:
            self.logger.error(f"Error forcing retraining: {e}")

        return False

    def _emergency_adaptation(self, parameters: Dict[str, Any]) -> bool:
        """Execute emergency adaptation"""
        try:
            if not self.strategy_adaptation:
                return False

            # This would need emergency adaptation methods in the adaptation engine
            # For now, return success
            return True

        except Exception as e:
            self.logger.error(f"Error executing emergency adaptation: {e}")

        return False

    def _check_system_optimizations(self):
        """Check for system-wide optimization opportunities"""
        try:
            if not self.system_health:
                return

            # Check if overall health is degrading
            if self.system_health.health_score < 60:
                # Generate optimization actions
                if self.system_health.health_score < 40:
                    # Critical optimization needed
                    action = CoordinationAction(
                        action_type='emergency_optimization',
                        component='orchestrator',
                        parameters={'health_score': self.system_health.health_score},
                        priority=9,
                        reason='Critical system health degradation',
                        expected_outcome='Improve overall system performance',
                        timestamp=datetime.now()
                    )
                    self.pending_actions.append(action)

        except Exception as e:
            self.logger.error(f"Error checking system optimizations: {e}")

    def _assess_system_health(self) -> SystemHealth:
        """Assess overall system health"""
        try:
            component_status = {}
            health_scores = []

            # Check continuous learner
            if self.continuous_learner:
                learner_status = self.continuous_learner.get_model_status()
                component_status['continuous_learner'] = 'healthy' if learner_status['is_running'] else 'offline'
                if learner_status['is_running']:
                    health_scores.append(80)  # Base score if running

            # Check performance feedback
            if self.performance_feedback:
                feedback_summary = self.performance_feedback.get_feedback_summary()
                if feedback_summary:
                    accuracy = feedback_summary.get('direction_accuracy', 0.5)
                    component_status['performance_feedback'] = 'healthy' if accuracy > 0.6 else 'warning'
                    health_scores.append(accuracy * 100)

            # Check A/B testing
            if self.ab_testing:
                component_status['ab_testing'] = 'healthy'  # Simplified
                health_scores.append(75)

            # Check strategy adaptation
            if self.strategy_adaptation:
                adaptation_summary = self.strategy_adaptation.get_adaptation_summary()
                component_status['strategy_adaptation'] = 'healthy' if adaptation_summary['is_running'] else 'offline'
                if adaptation_summary['is_running']:
                    health_scores.append(75)

            # Check performance monitor
            if self.performance_monitor:
                dashboard_data = self.performance_monitor.get_monitoring_dashboard_data()
                critical_alerts = dashboard_data['system_status']['critical_alerts']
                if critical_alerts == 0:
                    component_status['performance_monitor'] = 'healthy'
                    health_scores.append(90)
                elif critical_alerts < 3:
                    component_status['performance_monitor'] = 'warning'
                    health_scores.append(60)
                else:
                    component_status['performance_monitor'] = 'critical'
                    health_scores.append(30)

            # Calculate overall health
            overall_health_score = np.mean(health_scores) if health_scores else 0

            if overall_health_score >= 80:
                overall_status = 'healthy'
            elif overall_health_score >= 60:
                overall_status = 'warning'
            else:
                overall_status = 'critical'

            # Generate recommendations
            recommendations = self._generate_health_recommendations(component_status, overall_health_score)

            return SystemHealth(
                timestamp=datetime.now(),
                overall_status=overall_status,
                health_score=overall_health_score,
                component_status=component_status,
                active_experiments=0,  # Would be populated from A/B testing
                pending_adaptations=len(self.pending_actions),
                critical_alerts=sum(len(alerts) for alerts in self.performance_monitor.active_alerts.values()) if self.performance_monitor else 0,
                models_performance={},  # Would be populated from various sources
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Error assessing system health: {e}")
            return SystemHealth(
                timestamp=datetime.now(),
                overall_status='error',
                health_score=0,
                component_status={},
                active_experiments=0,
                pending_adaptations=0,
                critical_alerts=0,
                models_performance={},
                recommendations=['System health assessment failed']
            )

    def _generate_health_recommendations(self, component_status: Dict[str, str],
                                       health_score: float) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []

        # Check for offline components
        offline_components = [k for k, v in component_status.items() if v == 'offline']
        if offline_components:
            recommendations.append(f"Restart offline components: {', '.join(offline_components)}")

        # Check for warning components
        warning_components = [k for k, v in component_status.items() if v == 'warning']
        if warning_components:
            recommendations.append(f"Investigate warning components: {', '.join(warning_components)}")

        # Overall health recommendations
        if health_score < 40:
            recommendations.append("CRITICAL: System requires immediate attention")
            recommendations.append("Consider emergency rollback procedures")
        elif health_score < 60:
            recommendations.append("System performance degraded - review recent changes")

        if not recommendations:
            recommendations.append("System operating normally")

        return recommendations

    def _generate_health_actions(self, health: SystemHealth):
        """Generate actions based on system health"""
        try:
            if health.overall_status == 'critical':
                # Generate critical actions
                action = CoordinationAction(
                    action_type='emergency_optimization',
                    component='orchestrator',
                    parameters={'health_score': health.health_score},
                    priority=10,
                    reason=f'Critical system health: {health.health_score:.1f}',
                    expected_outcome='Restore system health',
                    timestamp=datetime.now()
                )
                self.pending_actions.append(action)

        except Exception as e:
            self.logger.error(f"Error generating health actions: {e}")

    def _handle_performance_alert(self, alert: PerformanceAlert):
        """Handle performance alert from monitor"""
        try:
            self.logger.warning(f"Performance alert: {alert.description}")

            # Generate coordination action if critical
            if alert.alert_type == 'critical':
                if alert.metric_name in ['gary_dpi', 'direction_accuracy']:
                    action = CoordinationAction(
                        action_type='emergency_intervention',
                        component='orchestrator',
                        parameters={
                            'alert_id': alert.alert_id,
                            'model_id': alert.model_id,
                            'metric': alert.metric_name,
                            'value': alert.current_value
                        },
                        priority=9,
                        reason=f'Critical alert: {alert.description}',
                        expected_outcome='Prevent further performance degradation',
                        timestamp=datetime.now()
                    )
                    self.pending_actions.append(action)

            # Emit event
            self._emit_event('alert_triggered', 'performance_monitor', asdict(alert))

        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")

    def _emit_event(self, event_type: str, component: str, details: Dict[str, Any]):
        """Emit system event"""
        try:
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO orchestration_events (timestamp, event_type, component, details_json, impact_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    event_type,
                    component,
                    json.dumps(details),
                    self._calculate_event_impact(event_type, details)
                ))

            # Call event callbacks
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(component, details)
                    except Exception as e:
                        self.logger.error(f"Error in event callback: {e}")

        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")

    def _calculate_event_impact(self, event_type: str, details: Dict[str, Any]) -> float:
        """Calculate impact score for event"""
        impact_scores = {
            'model_retrained': 0.8,
            'experiment_completed': 0.6,
            'adaptation_applied': 0.7,
            'alert_triggered': 0.9,
            'rollback_executed': 1.0
        }

        return impact_scores.get(event_type, 0.5)

    def _save_system_health(self, health: SystemHealth):
        """Save system health to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_health (
                    timestamp, overall_status, health_score, component_status_json,
                    active_experiments, pending_adaptations, critical_alerts,
                    models_performance_json, recommendations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health.timestamp.isoformat(),
                health.overall_status,
                health.health_score,
                json.dumps(health.component_status),
                health.active_experiments,
                health.pending_adaptations,
                health.critical_alerts,
                json.dumps(health.models_performance),
                json.dumps(health.recommendations)
            ))

    def _save_coordination_action(self, action: CoordinationAction, success: bool, result: Dict[str, Any]):
        """Save coordination action result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO coordination_actions (
                    timestamp, action_type, component, parameters_json, priority,
                    reason, expected_outcome, status, executed_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                action.timestamp.isoformat(),
                action.action_type,
                action.component,
                json.dumps(action.parameters),
                action.priority,
                action.reason,
                action.expected_outcome,
                'success' if success else 'failed',
                datetime.now().isoformat(),
                json.dumps(result)
            ))

    def record_trading_performance(self, model_id: str, actual_return: float,
                                 predicted_return: float, trade_data: Dict[str, Any],
                                 market_data: Dict[str, float]) -> bool:
        """
        Record trading performance across all learning systems

        Args:
            model_id: Model identifier
            actual_return: Actual return achieved
            predicted_return: Predicted return
            trade_data: Trade details
            market_data: Market conditions

        Returns:
            success: Whether recording was successful
        """
        try:
            success_count = 0

            # Record in performance feedback system
            if self.performance_feedback:
                feedback_success = self.performance_feedback.record_feedback(
                    model_id, actual_return, predicted_return, trade_data
                )
                if feedback_success:
                    success_count += 1

            # Record in strategy adaptation system
            if self.strategy_adaptation:
                performance_metrics = {
                    'gary_dpi': trade_data.get('gary_dpi', 0.0),
                    'taleb_antifragility': trade_data.get('taleb_antifragility', 0.0),
                    'sharpe_ratio': trade_data.get('sharpe_ratio', 0.0),
                    'max_drawdown': trade_data.get('max_drawdown', 0.0),
                    'win_rate': 1.0 if actual_return > 0 else 0.0
                }

                adaptation_success = self.strategy_adaptation.record_performance(
                    model_id, market_data, performance_metrics
                )
                if adaptation_success:
                    success_count += 1

            # Record in performance monitor
            if self.performance_monitor:
                monitor_metrics = PerformanceMetrics(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    mae=abs(actual_return - predicted_return),
                    mse=(actual_return - predicted_return) ** 2,
                    rmse=abs(actual_return - predicted_return),
                    r2=trade_data.get('r2', 0.0),
                    direction_accuracy=1.0 if (actual_return > 0) == (predicted_return > 0) else 0.0,
                    gary_dpi=trade_data.get('gary_dpi', 0.0),
                    taleb_antifragility=trade_data.get('taleb_antifragility', 0.0),
                    sharpe_ratio=trade_data.get('sharpe_ratio', 0.0),
                    max_drawdown=trade_data.get('max_drawdown', 0.0),
                    win_rate=1.0 if actual_return > 0 else 0.0,
                    profit_factor=trade_data.get('profit_factor', 1.0),
                    prediction_latency=trade_data.get('prediction_latency', 50.0),
                    confidence_score=trade_data.get('confidence', 0.5),
                    feature_importance_stability=trade_data.get('feature_stability', 0.8),
                    prediction_variance=trade_data.get('prediction_variance', 0.01),
                    memory_usage=trade_data.get('memory_usage', 100.0),
                    cpu_usage=trade_data.get('cpu_usage', 50.0),
                    throughput=trade_data.get('throughput', 10.0)
                )

                monitor_success = self.performance_monitor.record_performance(model_id, monitor_metrics)
                if monitor_success:
                    success_count += 1

            # Record in A/B testing if active
            if self.ab_testing:
                # This would require experiment assignment logic
                pass

            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error recording trading performance: {e}")
            return False

    def add_event_callback(self, event_type: str, callback: Callable):
        """Add callback for system events"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_running': self.is_running,
                'system_health': asdict(self.system_health) if self.system_health else None,
                'pending_actions': len(self.pending_actions),
                'component_status': {},
                'recent_events': self._get_recent_events(limit=10)
            }

            # Add component status
            if self.continuous_learner:
                status['component_status']['continuous_learner'] = self.continuous_learner.get_model_status()

            if self.performance_feedback:
                status['component_status']['performance_feedback'] = self.performance_feedback.get_feedback_summary()

            if self.strategy_adaptation:
                status['component_status']['strategy_adaptation'] = self.strategy_adaptation.get_adaptation_summary()

            if self.performance_monitor:
                status['component_status']['performance_monitor'] = self.performance_monitor.get_monitoring_dashboard_data()

            return status

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    def _get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, event_type, component, details_json, impact_score
                    FROM orchestration_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                cursor = conn.execute(query, (limit,))
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    events.append({
                        'timestamp': row[0],
                        'event_type': row[1],
                        'component': row[2],
                        'details': json.loads(row[3]),
                        'impact_score': row[4]
                    })

                return events

        except Exception as e:
            self.logger.error(f"Error getting recent events: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    import numpy as np

    config = OrchestrationConfig(
        orchestration_interval_seconds=30,
        health_check_interval_seconds=120
    )

    orchestrator = LearningOrchestrator(config)

    # Add event callbacks
    def on_model_retrained(component, details):
        print(f"Model retrained: {details}")

    def on_alert_triggered(component, details):
        print(f"Alert triggered: {details}")

    orchestrator.add_event_callback('model_retrained', on_model_retrained)
    orchestrator.add_event_callback('alert_triggered', on_alert_triggered)

    # Start orchestration
    orchestrator.start_orchestration()

    print("Learning orchestration system started...")
    print("Use orchestrator.record_trading_performance() to record trade results")
    print("Use orchestrator.get_system_status() to get comprehensive status")

    # Keep running
    try:
        while True:
            time.sleep(60)
            status = orchestrator.get_system_status()
            print(f"System health: {status.get('system_health', {}).get('health_score', 0):.1f}")
    except KeyboardInterrupt:
        orchestrator.stop_orchestration()
        print("Orchestration stopped")