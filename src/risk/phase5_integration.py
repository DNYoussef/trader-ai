"""
Phase 5 Risk & Calibration Systems Integration for Super-Gary Trading Framework

This module integrates the sophisticated risk management and calibration components
(Brier Scorer, Convexity Manager, Enhanced Kelly) with the existing infrastructure
including dashboards, kill switches, and real-time monitoring systems.

Integration Points:
- Brier Score Calibration with position sizing
- Convexity optimization with regime detection
- Enhanced Kelly with survival constraints
- Real-time dashboard updates
- Kill switch triggers based on calibration failures
- Performance monitoring and alerting

Key Features:
- Unified risk management interface
- Real-time calibration tracking
- Automated risk adjustments
- Emergency stop mechanisms
- Performance attribution
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our Phase 5 components
from .brier_scorer import BrierScorer, CalibrationMetrics
from .convexity_manager import ConvexityManager, RegimeState, MarketRegime
from .kelly_enhanced import EnhancedKellyCriterion, MultiAssetKellyResult, SurvivalMode

# Import existing infrastructure
try:
    from ..safety.kill_switch_system import KillSwitchSystem
    from ..dashboard.risk_dashboard import RiskDashboard
    from ..monitoring.performance_monitor import PerformanceMonitor
except ImportError as e:
    logging.warning(f"Could not import existing systems: {e}")

class Phase5Mode(Enum):
    """Phase 5 operation modes"""
    FULL_OPERATIONAL = "full_operational"
    CALIBRATION_FOCUS = "calibration_focus"
    SURVIVAL_MODE = "survival_mode"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskSystemStatus:
    """Overall risk system status"""
    mode: Phase5Mode
    calibration_score: float
    convexity_adequacy: float
    survival_probability: float
    kill_switch_armed: bool
    regime_confidence: float
    portfolio_health: str
    last_update: datetime = field(default_factory=datetime.now)
    alerts: List[str] = field(default_factory=list)

@dataclass
class IntegratedRiskMetrics:
    """Integrated risk metrics from all Phase 5 systems"""
    # Calibration metrics
    overall_brier_score: float
    type_calibration_scores: Dict[str, float]
    position_sizing_multiplier: float

    # Convexity metrics
    regime_state: RegimeState
    convexity_score: float
    gamma_exposure: float
    event_horizon_risk: float

    # Kelly metrics
    portfolio_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    survival_probability: float
    factor_exposures: Dict[str, float]

    # Integration metrics
    system_coherence: float
    risk_budget_utilization: float
    emergency_cash_ratio: float

class Phase5Integration:
    """
    Integrated Phase 5 risk and calibration systems

    Coordinates Brier scoring, convexity optimization, and enhanced Kelly
    with existing infrastructure for comprehensive risk management.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Phase 5 systems
        self.brier_scorer: Optional[BrierScorer] = None
        self.convexity_manager: Optional[ConvexityManager] = None
        self.kelly_system: Optional[EnhancedKellyCriterion] = None

        # Existing infrastructure
        self.kill_switch: Optional[Any] = None
        self.dashboard: Optional[Any] = None
        self.performance_monitor: Optional[Any] = None

        # System state
        self.current_mode = Phase5Mode.FULL_OPERATIONAL
        self.system_status = self._create_initial_status()
        self.risk_metrics_history: List[IntegratedRiskMetrics] = []

        # Threading and monitoring
        self.monitoring_active = False
        self.update_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Data persistence
        self.data_path = Path(self.config.get('data_path', './data/phase5_integration'))
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._initialize_systems()

    def _default_config(self) -> Dict:
        """Default configuration for Phase 5 integration"""
        return {
            'update_frequency': 60,  # Seconds between system updates
            'calibration_threshold': 0.7,  # Minimum calibration score
            'convexity_threshold': 0.3,   # Minimum convexity adequacy
            'survival_threshold': 0.95,   # Minimum survival probability
            'emergency_cash_ratio': 0.2,  # Emergency cash requirement
            'kill_switch_triggers': {
                'calibration_failure': 0.4,
                'convexity_failure': 0.1,
                'survival_failure': 0.8,
                'system_incoherence': 0.3
            },
            'regime_transition_sensitivity': 0.3,
            'position_sizing_limits': {
                'max_single_position': 0.15,
                'max_sector_exposure': 0.35,
                'max_leverage': 2.0
            },
            'monitoring_enabled': True,
            'dashboard_updates': True,
            'real_time_alerts': True,
            'performance_attribution': True
        }

    def _initialize_systems(self):
        """Initialize all Phase 5 systems and integrations"""
        try:
            # Initialize Phase 5 components
            self.brier_scorer = BrierScorer(self.config.get('brier_config', {}))
            self.convexity_manager = ConvexityManager(self.config.get('convexity_config', {}))
            self.kelly_system = EnhancedKellyCriterion(self.config.get('kelly_config', {}))

            # Cross-link systems
            self.kelly_system.calibration_system = self.brier_scorer
            self.kelly_system.convexity_system = self.convexity_manager

            # Initialize existing infrastructure
            self._initialize_existing_systems()

            # Start monitoring if enabled
            if self.config.get('monitoring_enabled', True):
                self.start_monitoring()

            self.logger.info("Phase 5 integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Phase 5 systems: {e}")
            self.current_mode = Phase5Mode.EMERGENCY_STOP
            raise

    def _initialize_existing_systems(self):
        """Initialize connections to existing infrastructure"""
        try:
            # Try to initialize kill switch
            try:
                self.kill_switch = KillSwitchSystem()
                self.logger.info("Connected to kill switch system")
            except:
                self.logger.warning("Could not connect to kill switch system")

            # Try to initialize dashboard
            try:
                self.dashboard = RiskDashboard()
                self.logger.info("Connected to risk dashboard")
            except:
                self.logger.warning("Could not connect to risk dashboard")

            # Try to initialize performance monitor
            try:
                self.performance_monitor = PerformanceMonitor()
                self.logger.info("Connected to performance monitor")
            except:
                self.logger.warning("Could not connect to performance monitor")

        except Exception as e:
            self.logger.error(f"Error connecting to existing systems: {e}")

    def add_prediction_and_outcome(self,
                                 prediction_id: str,
                                 forecast: float,
                                 prediction_type: str,
                                 outcome: Optional[bool] = None,
                                 confidence: float = 1.0) -> bool:
        """
        Add prediction to calibration system

        Args:
            prediction_id: Unique prediction identifier
            forecast: Probability forecast [0,1]
            prediction_type: Type of prediction
            outcome: Actual outcome (if known)
            confidence: Model confidence

        Returns:
            Success status
        """
        try:
            if self.brier_scorer is None:
                self.logger.error("Brier scorer not initialized")
                return False

            # Add prediction
            self.brier_scorer.add_prediction(
                prediction_id, forecast, prediction_type, confidence
            )

            # Update outcome if provided
            if outcome is not None:
                self.brier_scorer.update_outcome(prediction_id, outcome)

            # Trigger system update
            self._update_integrated_metrics()

            return True

        except Exception as e:
            self.logger.error(f"Error adding prediction: {e}")
            return False

    def update_market_data(self,
                          price_data: pd.DataFrame,
                          volume_data: Optional[pd.DataFrame] = None,
                          volatility_data: Optional[pd.DataFrame] = None) -> RiskSystemStatus:
        """
        Update all systems with new market data

        Args:
            price_data: Market price data
            volume_data: Volume data
            volatility_data: Volatility metrics

        Returns:
            Updated system status
        """
        try:
            # Update convexity manager with market data
            if self.convexity_manager:
                regime_state = self.convexity_manager.update_market_data(
                    price_data, volume_data, volatility_data
                )

            # Update integrated metrics
            self._update_integrated_metrics()

            # Check for mode changes
            self._evaluate_system_mode()

            # Update dashboard if connected
            if self.dashboard and self.config.get('dashboard_updates', True):
                self._update_dashboard()

            return self.system_status

        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            self._trigger_emergency_mode(f"Market data update failed: {e}")
            return self.system_status

    def optimize_portfolio(self,
                          expected_returns: Dict[str, float],
                          assets_data: Dict[str, pd.Series]) -> Optional[MultiAssetKellyResult]:
        """
        Optimize portfolio using integrated systems

        Args:
            expected_returns: Expected returns by asset
            assets_data: Historical data by asset

        Returns:
            Optimized portfolio allocation
        """
        try:
            if self.kelly_system is None:
                self.logger.error("Kelly system not initialized")
                return None

            # Add asset profiles to Kelly system
            for asset, data in assets_data.items():
                market_data = self._get_market_data_for_asset(asset)
                self.kelly_system.add_asset_profile(asset, data, market_data)

            # Update correlation matrix
            returns_df = pd.DataFrame(assets_data)
            self.kelly_system.update_correlation_matrix(returns_df)

            # Get calibration-adjusted confidence scores
            confidence_scores = self._get_calibration_confidence_scores(list(expected_returns.keys()))

            # Optimize portfolio
            result = self.kelly_system.optimize_multi_asset_portfolio(
                expected_returns, confidence_scores
            )

            # Apply convexity constraints if in uncertain regime
            if self.convexity_manager and self.convexity_manager.regime_history:
                latest_regime = self.convexity_manager.regime_history[-1]
                if latest_regime.uncertainty > self.config['regime_transition_sensitivity']:
                    result = self._apply_convexity_adjustments(result, latest_regime)

            # Update system metrics
            self._update_integrated_metrics()

            # Log optimization result
            self.logger.info(f"Portfolio optimized: {len(result.optimal_weights)} assets, "
                           f"expected return: {result.expected_return:.3f}, "
                           f"volatility: {result.expected_volatility:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return None

    def get_position_sizing_recommendation(self,
                                         asset: str,
                                         base_kelly: float,
                                         market_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get comprehensive position sizing recommendation

        Args:
            asset: Asset identifier
            base_kelly: Base Kelly calculation
            market_conditions: Current market conditions

        Returns:
            Position sizing recommendation
        """
        try:
            recommendation = {
                'asset': asset,
                'timestamp': datetime.now(),
                'base_kelly': base_kelly,
                'recommendations': {},
                'risk_factors': {},
                'alerts': []
            }

            # Get calibration adjustment
            if self.brier_scorer:
                calibration_score = self.brier_scorer.get_calibration_score()
                calibration_multiplier = self.brier_scorer.get_position_size_multiplier(
                    prediction_type=None, base_kelly=base_kelly
                )
                recommendation['recommendations']['calibration_adjusted'] = base_kelly * calibration_multiplier
                recommendation['risk_factors']['calibration_score'] = calibration_score

            # Get convexity requirements
            if self.convexity_manager:
                if self.convexity_manager.regime_history:
                    current_regime = self.convexity_manager.regime_history[-1]
                    convexity_target = self.convexity_manager.get_convexity_requirements(
                        asset, base_kelly, current_regime
                    )

                    recommendation['recommendations']['convexity_adjusted'] = base_kelly * convexity_target.convexity_score
                    recommendation['risk_factors']['regime'] = current_regime.regime.value
                    recommendation['risk_factors']['regime_confidence'] = current_regime.confidence

            # Get Kelly survival adjustment
            if self.kelly_system and asset in self.kelly_system.asset_profiles:
                survival_kelly = self.kelly_system.calculate_survival_kelly(
                    asset, base_kelly, calibration_multiplier if 'calibration_multiplier' in locals() else 1.0
                )
                recommendation['recommendations']['survival_adjusted'] = survival_kelly

            # Final integrated recommendation
            adjustments = [
                recommendation['recommendations'].get('calibration_adjusted', base_kelly),
                recommendation['recommendations'].get('convexity_adjusted', base_kelly),
                recommendation['recommendations'].get('survival_adjusted', base_kelly)
            ]

            # Use most conservative adjustment
            final_recommendation = min(adjustments)
            recommendation['final_recommendation'] = final_recommendation
            recommendation['adjustment_factor'] = final_recommendation / base_kelly if base_kelly > 0 else 0

            # Add alerts if significant adjustments
            if final_recommendation < base_kelly * 0.5:
                recommendation['alerts'].append("Significant downward adjustment due to risk factors")

            return recommendation

        except Exception as e:
            self.logger.error(f"Error generating position sizing recommendation: {e}")
            return {
                'asset': asset,
                'error': str(e),
                'final_recommendation': base_kelly * 0.1,  # Very conservative fallback
                'alerts': ['Error in risk calculation - using conservative sizing']
            }

    def get_integrated_risk_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive risk dashboard data

        Returns:
            Dashboard data with all risk metrics
        """
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': {
                    'mode': self.current_mode.value,
                    'health': self.system_status.portfolio_health,
                    'alerts_count': len(self.system_status.alerts)
                },
                'calibration': {},
                'convexity': {},
                'kelly': {},
                'integration': {},
                'alerts': self.system_status.alerts
            }

            # Calibration metrics
            if self.brier_scorer:
                scoreboard = self.brier_scorer.get_performance_scoreboard()
                dashboard_data['calibration'] = {
                    'overall_score': scoreboard['overall_metrics'].get('calibration_score', 0),
                    'position_multiplier': scoreboard['overall_metrics'].get('position_multiplier', 1),
                    'total_predictions': scoreboard['overall_metrics'].get('total_predictions', 0),
                    'type_breakdown': scoreboard.get('type_breakdown', {}),
                    'recent_performance': scoreboard.get('recent_performance', {})
                }

            # Convexity metrics
            if self.convexity_manager and self.convexity_manager.regime_history:
                latest_regime = self.convexity_manager.regime_history[-1]
                dashboard_data['convexity'] = {
                    'current_regime': latest_regime.regime.value,
                    'regime_confidence': latest_regime.confidence,
                    'uncertainty': latest_regime.uncertainty,
                    'time_in_regime': latest_regime.time_in_regime,
                    'transition_probability': latest_regime.transition_probability,
                    'gamma_positions': len(self.convexity_manager.options_structures)
                }

            # Kelly metrics
            if self.kelly_system and self.kelly_system.sizing_history:
                latest_sizing = self.kelly_system.sizing_history[-1]
                dashboard_data['kelly'] = {
                    'portfolio_assets': len(latest_sizing['assets']),
                    'expected_return': latest_sizing['expected_return'],
                    'expected_volatility': latest_sizing['expected_volatility'],
                    'weights': latest_sizing['weights'],
                    'survival_probability': latest_sizing.get('survival_probability', 0.95)
                }

            # Integration metrics
            if self.risk_metrics_history:
                latest_metrics = self.risk_metrics_history[-1]
                dashboard_data['integration'] = {
                    'system_coherence': latest_metrics.system_coherence,
                    'risk_budget_utilization': latest_metrics.risk_budget_utilization,
                    'emergency_cash_ratio': latest_metrics.emergency_cash_ratio,
                    'factor_exposures': latest_metrics.factor_exposures
                }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_status': {'mode': 'error', 'health': 'unhealthy'}
            }

    def start_monitoring(self):
        """Start real-time monitoring of all systems"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("Started Phase 5 monitoring")

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Stopped Phase 5 monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update integrated metrics
                self._update_integrated_metrics()

                # Evaluate system mode
                self._evaluate_system_mode()

                # Check for kill switch triggers
                self._check_kill_switch_triggers()

                # Update dashboard
                if self.dashboard and self.config.get('dashboard_updates', True):
                    self._update_dashboard()

                # Sleep until next update
                time.sleep(self.config['update_frequency'])

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Brief pause before retry

    def _update_integrated_metrics(self):
        """Update integrated risk metrics from all systems"""
        try:
            # Gather metrics from all systems
            calibration_metrics = self._gather_calibration_metrics()
            convexity_metrics = self._gather_convexity_metrics()
            kelly_metrics = self._gather_kelly_metrics()

            # Calculate integration metrics
            integration_metrics = self._calculate_integration_metrics(
                calibration_metrics, convexity_metrics, kelly_metrics
            )

            # Create integrated metrics object
            integrated_metrics = IntegratedRiskMetrics(
                **calibration_metrics,
                **convexity_metrics,
                **kelly_metrics,
                **integration_metrics
            )

            # Add to history
            self.risk_metrics_history.append(integrated_metrics)

            # Trim history
            max_history = 1000
            self.risk_metrics_history = self.risk_metrics_history[-max_history:]

        except Exception as e:
            self.logger.error(f"Error updating integrated metrics: {e}")

    def _gather_calibration_metrics(self) -> Dict[str, Any]:
        """Gather metrics from calibration system"""
        if self.brier_scorer is None:
            return {
                'overall_brier_score': 0.5,
                'type_calibration_scores': {},
                'position_sizing_multiplier': 0.5
            }

        scoreboard = self.brier_scorer.get_performance_scoreboard()
        overall_metrics = scoreboard.get('overall_metrics', {})

        return {
            'overall_brier_score': 1 - overall_metrics.get('brier_score', 0.5),
            'type_calibration_scores': {
                ptype: metrics.get('calibration_score', 0.5)
                for ptype, metrics in scoreboard.get('type_breakdown', {}).items()
            },
            'position_sizing_multiplier': overall_metrics.get('position_multiplier', 0.5)
        }

    def _gather_convexity_metrics(self) -> Dict[str, Any]:
        """Gather metrics from convexity system"""
        if self.convexity_manager is None or not self.convexity_manager.regime_history:
            return {
                'regime_state': RegimeState(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    regime_probabilities={},
                    uncertainty=1.0,
                    time_in_regime=0,
                    transition_probability=1.0
                ),
                'convexity_score': 0.5,
                'gamma_exposure': 0.0,
                'event_horizon_risk': 0.5
            }

        latest_regime = self.convexity_manager.regime_history[-1]

        # Calculate average convexity score across positions
        convexity_scores = [target.convexity_score for target in self.convexity_manager.convexity_targets.values()]
        avg_convexity = np.mean(convexity_scores) if convexity_scores else 0.5

        # Calculate total gamma exposure
        total_gamma = sum(self.convexity_manager.gamma_positions.values())

        # Assess event horizon risk
        upcoming_events = self.convexity_manager._get_upcoming_events()
        event_risk = len([e for e in upcoming_events if (e['date'] - datetime.now()).days <= 7]) / 10

        return {
            'regime_state': latest_regime,
            'convexity_score': avg_convexity,
            'gamma_exposure': total_gamma,
            'event_horizon_risk': min(1.0, event_risk)
        }

    def _gather_kelly_metrics(self) -> Dict[str, Any]:
        """Gather metrics from Kelly system"""
        if self.kelly_system is None or not self.kelly_system.sizing_history:
            return {
                'portfolio_weights': {},
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'survival_probability': 0.95,
                'factor_exposures': {}
            }

        latest_sizing = self.kelly_system.sizing_history[-1]

        # Get factor exposures
        factor_exposures = self.kelly_system.get_factor_decomposition(latest_sizing['weights'])

        return {
            'portfolio_weights': latest_sizing['weights'],
            'expected_return': latest_sizing['expected_return'],
            'expected_volatility': latest_sizing['expected_volatility'],
            'survival_probability': latest_sizing.get('survival_probability', 0.95),
            'factor_exposures': factor_exposures
        }

    def _calculate_integration_metrics(self,
                                     calibration_metrics: Dict,
                                     convexity_metrics: Dict,
                                     kelly_metrics: Dict) -> Dict[str, Any]:
        """Calculate integration-specific metrics"""
        try:
            # System coherence: how well systems agree
            calibration_health = calibration_metrics['overall_brier_score']
            convexity_health = convexity_metrics['convexity_score']
            kelly_health = kelly_metrics['survival_probability']

            system_coherence = np.mean([calibration_health, convexity_health, kelly_health])

            # Risk budget utilization
            position_multiplier = calibration_metrics['position_sizing_multiplier']
            gamma_exposure = convexity_metrics['gamma_exposure']
            portfolio_concentration = self._calculate_portfolio_concentration(kelly_metrics['portfolio_weights'])

            risk_budget_utilization = np.mean([position_multiplier, abs(gamma_exposure), portfolio_concentration])

            # Emergency cash ratio
            cash_weight = kelly_metrics['portfolio_weights'].get('CASH', 0)
            required_cash = self.config['emergency_cash_ratio']
            emergency_cash_ratio = cash_weight / required_cash if required_cash > 0 else 1.0

            return {
                'system_coherence': system_coherence,
                'risk_budget_utilization': risk_budget_utilization,
                'emergency_cash_ratio': emergency_cash_ratio
            }

        except Exception as e:
            self.logger.error(f"Error calculating integration metrics: {e}")
            return {
                'system_coherence': 0.5,
                'risk_budget_utilization': 0.5,
                'emergency_cash_ratio': 1.0
            }

    def _calculate_portfolio_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration (1 - HHI)"""
        if not weights:
            return 0.0

        # Calculate Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights.values())
        concentration = 1 - hhi  # Higher value = less concentrated
        return concentration

    def _evaluate_system_mode(self):
        """Evaluate and potentially change system mode"""
        try:
            if not self.risk_metrics_history:
                return

            latest_metrics = self.risk_metrics_history[-1]

            # Check for emergency conditions
            if (latest_metrics.overall_brier_score < self.config['kill_switch_triggers']['calibration_failure'] or
                latest_metrics.convexity_score < self.config['kill_switch_triggers']['convexity_failure'] or
                latest_metrics.survival_probability < self.config['kill_switch_triggers']['survival_failure']):

                self._trigger_emergency_mode("Critical risk metrics threshold breached")
                return

            # Check for survival mode conditions
            if (latest_metrics.overall_brier_score < self.config['calibration_threshold'] or
                latest_metrics.survival_probability < self.config['survival_threshold']):

                if self.current_mode != Phase5Mode.SURVIVAL_MODE:
                    self.current_mode = Phase5Mode.SURVIVAL_MODE
                    self.logger.warning("Entering survival mode due to risk deterioration")

            # Check for calibration focus mode
            elif latest_metrics.overall_brier_score < self.config['calibration_threshold']:
                if self.current_mode != Phase5Mode.CALIBRATION_FOCUS:
                    self.current_mode = Phase5Mode.CALIBRATION_FOCUS
                    self.logger.info("Entering calibration focus mode")

            # Return to full operational
            elif (latest_metrics.overall_brier_score >= self.config['calibration_threshold'] and
                  latest_metrics.convexity_score >= self.config['convexity_threshold'] and
                  latest_metrics.survival_probability >= self.config['survival_threshold']):

                if self.current_mode != Phase5Mode.FULL_OPERATIONAL:
                    self.current_mode = Phase5Mode.FULL_OPERATIONAL
                    self.logger.info("Returning to full operational mode")

            # Update system status
            self._update_system_status()

        except Exception as e:
            self.logger.error(f"Error evaluating system mode: {e}")

    def _trigger_emergency_mode(self, reason: str):
        """Trigger emergency mode and safety measures"""
        self.current_mode = Phase5Mode.EMERGENCY_STOP
        self.logger.critical(f"EMERGENCY MODE TRIGGERED: {reason}")

        # Trigger kill switch if available
        if self.kill_switch:
            try:
                self.kill_switch.trigger_emergency_stop(reason)
            except Exception as e:
                self.logger.error(f"Failed to trigger kill switch: {e}")

        # Update system status
        self._update_system_status()
        self.system_status.alerts.append(f"EMERGENCY: {reason}")

    def _check_kill_switch_triggers(self):
        """Check for kill switch trigger conditions"""
        if not self.risk_metrics_history or not self.kill_switch:
            return

        try:
            latest_metrics = self.risk_metrics_history[-1]
            triggers = self.config['kill_switch_triggers']

            # Check each trigger condition
            if latest_metrics.overall_brier_score < triggers['calibration_failure']:
                self._trigger_kill_switch("Calibration failure")
            elif latest_metrics.convexity_score < triggers['convexity_failure']:
                self._trigger_kill_switch("Convexity failure")
            elif latest_metrics.survival_probability < triggers['survival_failure']:
                self._trigger_kill_switch("Survival probability failure")
            elif latest_metrics.system_coherence < triggers['system_incoherence']:
                self._trigger_kill_switch("System incoherence")

        except Exception as e:
            self.logger.error(f"Error checking kill switch triggers: {e}")

    def _trigger_kill_switch(self, reason: str):
        """Trigger kill switch with specific reason"""
        try:
            if self.kill_switch:
                self.kill_switch.trigger_emergency_stop(f"Phase 5 trigger: {reason}")
                self.logger.critical(f"Kill switch triggered: {reason}")
        except Exception as e:
            self.logger.error(f"Failed to trigger kill switch: {e}")

    def _update_dashboard(self):
        """Update risk dashboard with latest data"""
        try:
            if self.dashboard:
                dashboard_data = self.get_integrated_risk_dashboard()
                self.dashboard.update_risk_metrics(dashboard_data)
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")

    def _update_system_status(self):
        """Update overall system status"""
        if self.risk_metrics_history:
            latest_metrics = self.risk_metrics_history[-1]

            self.system_status = RiskSystemStatus(
                mode=self.current_mode,
                calibration_score=latest_metrics.overall_brier_score,
                convexity_adequacy=latest_metrics.convexity_score,
                survival_probability=latest_metrics.survival_probability,
                kill_switch_armed=self.kill_switch is not None,
                regime_confidence=latest_metrics.regime_state.confidence,
                portfolio_health=self._assess_portfolio_health(latest_metrics),
                alerts=self._generate_current_alerts(latest_metrics)
            )

    def _assess_portfolio_health(self, metrics: IntegratedRiskMetrics) -> str:
        """Assess overall portfolio health"""
        if self.current_mode == Phase5Mode.EMERGENCY_STOP:
            return "critical"
        elif self.current_mode == Phase5Mode.SURVIVAL_MODE:
            return "poor"
        elif (metrics.overall_brier_score >= 0.7 and
              metrics.convexity_score >= 0.5 and
              metrics.survival_probability >= 0.95):
            return "excellent"
        elif (metrics.overall_brier_score >= 0.6 and
              metrics.convexity_score >= 0.3 and
              metrics.survival_probability >= 0.9):
            return "good"
        else:
            return "fair"

    def _generate_current_alerts(self, metrics: IntegratedRiskMetrics) -> List[str]:
        """Generate current system alerts"""
        alerts = []

        # Calibration alerts
        if metrics.overall_brier_score < 0.5:
            alerts.append("Poor calibration performance detected")

        # Regime alerts
        if metrics.regime_state.uncertainty > 0.8:
            alerts.append("High regime uncertainty - consider defensive positioning")

        # Survival alerts
        if metrics.survival_probability < 0.9:
            alerts.append("Survival probability below threshold")

        # System coherence alerts
        if metrics.system_coherence < 0.4:
            alerts.append("System components showing poor coherence")

        return alerts

    def _create_initial_status(self) -> RiskSystemStatus:
        """Create initial system status"""
        return RiskSystemStatus(
            mode=Phase5Mode.FULL_OPERATIONAL,
            calibration_score=0.5,
            convexity_adequacy=0.5,
            survival_probability=0.95,
            kill_switch_armed=False,
            regime_confidence=0.5,
            portfolio_health="initializing",
            alerts=["System initializing"]
        )

    def _get_market_data_for_asset(self, asset: str) -> Dict[str, Any]:
        """Get market data for asset (placeholder for actual implementation)"""
        # This would connect to actual market data sources
        return {
            'liquidity_score': 0.8,
            'beta_equity': 1.0,
            'beta_duration': 0.0,
            'beta_inflation': 0.0,
            'crowding_score': 0.5
        }

    def _get_calibration_confidence_scores(self, assets: List[str]) -> Dict[str, float]:
        """Get calibration-based confidence scores for assets"""
        if self.brier_scorer is None:
            return {asset: 0.5 for asset in assets}

        # Use overall calibration score as proxy
        overall_score = self.brier_scorer.get_calibration_score()
        return {asset: overall_score for asset in assets}

    def _apply_convexity_adjustments(self,
                                   result: MultiAssetKellyResult,
                                   regime_state: RegimeState) -> MultiAssetKellyResult:
        """Apply convexity adjustments based on regime uncertainty"""
        # Reduce position sizes during uncertain regimes
        uncertainty_factor = 1 - (regime_state.uncertainty * 0.5)  # Max 50% reduction

        adjusted_weights = {
            asset: weight * uncertainty_factor
            for asset, weight in result.optimal_weights.items()
        }

        # Increase cash allocation
        cash_increase = sum(result.optimal_weights.values()) - sum(adjusted_weights.values())
        adjusted_weights['CASH'] = adjusted_weights.get('CASH', 0) + cash_increase

        # Create new result with adjusted weights
        result.optimal_weights = adjusted_weights
        result.expected_return *= uncertainty_factor
        result.expected_volatility *= uncertainty_factor

        return result

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create integrated Phase 5 system
    phase5 = Phase5Integration()

    # Simulate some predictions
    for i in range(10):
        pred_id = f"test_pred_{i}"
        forecast = np.random.uniform(0.3, 0.8)
        outcome = np.random.random() < forecast

        phase5.add_prediction_and_outcome(pred_id, forecast, "direction", outcome)

    # Simulate market data update
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    price_data = pd.DataFrame({
        'open': 100 + np.random.normal(0, 1, len(dates)).cumsum(),
        'high': 102 + np.random.normal(0, 1, len(dates)).cumsum(),
        'low': 98 + np.random.normal(0, 1, len(dates)).cumsum(),
        'close': 100 + np.random.normal(0, 1, len(dates)).cumsum()
    }, index=dates)

    status = phase5.update_market_data(price_data)
    print(f"System Status: {status.mode.value}")
    print(f"Portfolio Health: {status.portfolio_health}")
    print(f"Alerts: {len(status.alerts)}")

    # Get position sizing recommendation
    recommendation = phase5.get_position_sizing_recommendation('SPY', 0.15)
    print(f"Position Sizing for SPY: {recommendation['final_recommendation']:.3f}")
    print(f"Adjustment Factor: {recommendation['adjustment_factor']:.3f}")

    # Get integrated dashboard
    dashboard = phase5.get_integrated_risk_dashboard()
    print(f"Dashboard timestamp: {dashboard['timestamp']}")
    print(f"System mode: {dashboard['system_status']['mode']}")

    # Cleanup
    phase5.stop_monitoring()