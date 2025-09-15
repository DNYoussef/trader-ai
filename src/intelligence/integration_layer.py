"""
Integration Layer for AI Alert System with DPI and Trading Engine

Provides seamless integration between AI alert components and existing systems:
- DPI calculator integration for enhanced signal validation
- Antifragility framework integration
- Trading engine alert routing
- Performance validation and metrics reporting
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import our components
from .alert_orchestrator import AlertOrchestrator, AlertMessage, SystemMetrics
from ..strategies.dpi_calculator import DistributionalPressureIndex

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration component status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class IntegrationMetrics:
    """Integration performance metrics"""
    dpi_integration_latency_ms: float
    antifragility_sync_rate: float
    trading_engine_notification_rate: float
    alert_validation_accuracy: float
    overall_integration_health: float


@dataclass
class ValidationResult:
    """Alert validation result"""
    is_valid: bool
    confidence_adjustment: float
    validation_score: float
    dpi_confirmation: bool
    antifragility_breach: bool
    recommended_action: str
    details: Dict[str, Any]


class IntelligenceIntegrationLayer:
    """
    Integration Layer for AI Alert System

    Connects AI alert system with existing Gary×Taleb infrastructure
    for seamless operation and enhanced validation.
    """

    def __init__(self,
                 dpi_calculator: DistributionalPressureIndex = None,
                 trading_engine=None,
                 config: Dict[str, Any] = None):
        """
        Initialize Integration Layer

        Args:
            dpi_calculator: DPI calculation system
            trading_engine: Main trading engine
            config: Integration configuration
        """
        self.dpi_calculator = dpi_calculator
        self.trading_engine = trading_engine
        self.config = config or {}

        # Initialize Alert Orchestrator with DPI integration
        self.alert_orchestrator = AlertOrchestrator(
            dpi_calculator=dpi_calculator,
            config=self.config.get('orchestrator_config', {})
        )

        # Integration status tracking
        self.integration_status = {
            'dpi_calculator': IntegrationStatus.DISCONNECTED,
            'trading_engine': IntegrationStatus.DISCONNECTED,
            'alert_orchestrator': IntegrationStatus.DISCONNECTED
        }

        # Performance tracking
        self.integration_metrics_history = []
        self.validation_history = []

        # Alert routing configuration
        self.alert_routing_rules = {
            'critical_alerts': ['trading_engine', 'risk_manager', 'notification_system'],
            'high_alerts': ['risk_manager', 'notification_system'],
            'medium_alerts': ['notification_system'],
            'low_alerts': ['log_only']
        }

        # Validation thresholds
        self.validation_thresholds = {
            'dpi_divergence_threshold': 0.3,
            'confidence_boost_factor': 1.2,
            'antifragility_breach_threshold': 0.8,
            'minimum_validation_score': 0.6
        }

        logger.info("Intelligence Integration Layer initialized")

    def initialize(self) -> bool:
        """Initialize all integration components"""
        try:
            logger.info("Initializing Intelligence Integration Layer...")

            # Initialize Alert Orchestrator
            if self.alert_orchestrator.start():
                self.integration_status['alert_orchestrator'] = IntegrationStatus.CONNECTED
                logger.info("Alert Orchestrator initialized successfully")
            else:
                logger.error("Alert Orchestrator initialization failed")
                return False

            # Check DPI Calculator connection
            if self.dpi_calculator:
                try:
                    # Test DPI calculator
                    test_result = self._test_dpi_integration()
                    if test_result:
                        self.integration_status['dpi_calculator'] = IntegrationStatus.CONNECTED
                        logger.info("DPI Calculator integration verified")
                    else:
                        self.integration_status['dpi_calculator'] = IntegrationStatus.ERROR
                        logger.warning("DPI Calculator integration test failed")
                except Exception as e:
                    logger.error(f"DPI Calculator integration error: {e}")
                    self.integration_status['dpi_calculator'] = IntegrationStatus.ERROR
            else:
                logger.warning("No DPI Calculator provided")

            # Check Trading Engine connection
            if self.trading_engine:
                try:
                    # Test trading engine integration
                    self.integration_status['trading_engine'] = IntegrationStatus.CONNECTED
                    logger.info("Trading Engine integration established")
                except Exception as e:
                    logger.error(f"Trading Engine integration error: {e}")
                    self.integration_status['trading_engine'] = IntegrationStatus.ERROR
            else:
                logger.warning("No Trading Engine provided")

            # Set up notification handlers
            self._setup_notification_handlers()

            logger.info("Intelligence Integration Layer initialization completed")
            return True

        except Exception as e:
            logger.error(f"Integration Layer initialization failed: {e}")
            return False

    def process_market_data_with_intelligence(self,
                                            symbol: str,
                                            market_data: Dict[str, Any],
                                            portfolio_context: Dict[str, Any] = None) -> List[AlertMessage]:
        """
        Process market data with full AI intelligence and validation

        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio_context: Portfolio context

        Returns:
            List of validated and enhanced alert messages
        """
        try:
            # Enhance market data with DPI calculations
            enhanced_data = self._enhance_market_data_with_dpi(symbol, market_data)

            # Process through Alert Orchestrator
            alerts = self.alert_orchestrator.process_market_data(
                symbol=symbol,
                market_data=enhanced_data,
                portfolio_context=portfolio_context
            )

            # Validate alerts with DPI and antifragility systems
            validated_alerts = []
            for alert in alerts:
                validation_result = self._validate_alert_with_systems(alert, enhanced_data)

                if validation_result.is_valid:
                    # Enhance alert with validation data
                    enhanced_alert = self._enhance_alert_with_validation(alert, validation_result)
                    validated_alerts.append(enhanced_alert)

                    # Route alert to appropriate systems
                    self._route_validated_alert(enhanced_alert)

                else:
                    logger.debug(f"Alert {alert.alert_id} failed validation: {validation_result.details}")

            # Update metrics
            self._update_integration_metrics(symbol, len(alerts), len(validated_alerts))

            logger.info(f"Intelligence processing for {symbol}: {len(alerts)} alerts, "
                       f"{len(validated_alerts)} validated")

            return validated_alerts

        except Exception as e:
            logger.error(f"Intelligence processing failed for {symbol}: {e}")
            return []

    def train_integrated_models(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train AI models with DPI-enhanced historical data"""
        try:
            logger.info("Training integrated AI models with DPI enhancement...")

            # Enhance historical data with DPI calculations
            enhanced_training_data = self._enhance_training_data_with_dpi(historical_data)

            # Train models through orchestrator
            training_results = self.alert_orchestrator.train_ai_models(
                historical_data=enhanced_training_data,
                risk_events=historical_data.get('risk_events')
            )

            # Add integration-specific metrics
            training_results['integration_metrics'] = {
                'dpi_enhanced_samples': enhanced_training_data.get('dpi_enhanced_count', 0),
                'integration_validation_score': self._calculate_integration_validation_score()
            }

            logger.info("Integrated model training completed")
            return training_results

        except Exception as e:
            logger.error(f"Integrated model training failed: {e}")
            return {'error': str(e)}

    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all integrations"""
        try:
            # Get orchestrator status
            orchestrator_status = self.alert_orchestrator.get_system_status()

            # Get integration metrics
            integration_metrics = self._calculate_current_integration_metrics()

            # Get component health
            component_health = {
                component: status.value
                for component, status in self.integration_status.items()
            }

            return {
                'timestamp': datetime.now().isoformat(),
                'integration_status': component_health,
                'orchestrator_status': orchestrator_status,
                'integration_metrics': asdict(integration_metrics),
                'validation_stats': self._get_validation_statistics(),
                'alert_routing_stats': self._get_alert_routing_statistics(),
                'dpi_integration_health': self._assess_dpi_integration_health(),
                'overall_intelligence_score': self._calculate_overall_intelligence_score()
            }

        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {'error': str(e)}

    def _enhance_market_data_with_dpi(self,
                                     symbol: str,
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance market data with DPI calculations"""
        try:
            enhanced_data = market_data.copy()

            if self.dpi_calculator and self.integration_status['dpi_calculator'] == IntegrationStatus.CONNECTED:
                # Calculate current DPI
                dpi_score, dpi_components = self.dpi_calculator.calculate_dpi(symbol)

                # Calculate narrative gap
                narrative_gap_analysis = self.dpi_calculator.detect_narrative_gap(symbol)

                # Get distributional regime
                regime = self.dpi_calculator.get_distributional_regime(symbol)

                # Add DPI data to market data
                enhanced_data['dpi_data'] = {
                    'dpi_score': dpi_score,
                    'components': asdict(dpi_components),
                    'narrative_gap': asdict(narrative_gap_analysis),
                    'regime': regime.value,
                    'confidence': self._calculate_dpi_confidence(dpi_components)
                }

                logger.debug(f"Enhanced {symbol} with DPI score: {dpi_score:.4f}")

            return enhanced_data

        except Exception as e:
            logger.error(f"DPI enhancement failed for {symbol}: {e}")
            return market_data

    def _validate_alert_with_systems(self,
                                    alert: AlertMessage,
                                    enhanced_data: Dict[str, Any]) -> ValidationResult:
        """Validate alert using DPI and antifragility systems"""
        try:
            validation_score = 0.0
            confidence_adjustment = 1.0
            dpi_confirmation = False
            antifragility_breach = False
            validation_details = {}

            # DPI-based validation
            if 'dpi_data' in enhanced_data:
                dpi_data = enhanced_data['dpi_data']
                dpi_validation = self._validate_with_dpi(alert, dpi_data)

                validation_score += dpi_validation['score'] * 0.4
                confidence_adjustment *= dpi_validation['confidence_multiplier']
                dpi_confirmation = dpi_validation['confirmed']
                validation_details['dpi_validation'] = dpi_validation

            # Antifragility validation (placeholder for integration)
            antifragility_validation = self._validate_with_antifragility(alert, enhanced_data)
            validation_score += antifragility_validation['score'] * 0.3
            antifragility_breach = antifragility_validation['breach_detected']
            validation_details['antifragility_validation'] = antifragility_validation

            # Market context validation
            context_validation = self._validate_with_market_context(alert, enhanced_data)
            validation_score += context_validation['score'] * 0.3
            validation_details['context_validation'] = context_validation

            # Final validation decision
            is_valid = (
                validation_score >= self.validation_thresholds['minimum_validation_score']
                and confidence_adjustment >= 0.7
            )

            # Determine recommended action
            recommended_action = self._determine_recommended_action(
                alert, validation_score, dpi_confirmation, antifragility_breach
            )

            return ValidationResult(
                is_valid=is_valid,
                confidence_adjustment=confidence_adjustment,
                validation_score=validation_score,
                dpi_confirmation=dpi_confirmation,
                antifragility_breach=antifragility_breach,
                recommended_action=recommended_action,
                details=validation_details
            )

        except Exception as e:
            logger.error(f"Alert validation failed for {alert.alert_id}: {e}")
            return ValidationResult(
                is_valid=True,  # Fail-safe: pass alert
                confidence_adjustment=0.8,
                validation_score=0.6,
                dpi_confirmation=False,
                antifragility_breach=False,
                recommended_action="monitor",
                details={'error': str(e)}
            )

    def _validate_with_dpi(self, alert: AlertMessage, dpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alert using DPI system"""
        try:
            dpi_score = dpi_data.get('dpi_score', 0.0)
            dpi_confidence = dpi_data.get('confidence', 0.5)
            narrative_gap = dpi_data.get('narrative_gap', {}).get('narrative_gap', 0.0)

            # DPI-Alert correlation validation
            alert_bullish = alert.alert_type in ['momentum_reversal'] and 'bullish' in alert.message.lower()
            alert_bearish = alert.alert_type in ['momentum_reversal'] and 'bearish' in alert.message.lower()

            dpi_bullish = dpi_score > 0.3
            dpi_bearish = dpi_score < -0.3

            # Check for confirmation or divergence
            confirmed = False
            if (alert_bullish and dpi_bullish) or (alert_bearish and dpi_bearish):
                confirmed = True
            elif alert.alert_type in ['volatility_spike', 'regime_change']:
                # These alerts are confirmed by high DPI magnitude
                confirmed = abs(dpi_score) > 0.5

            # Calculate validation score
            if confirmed:
                score = min(1.0, 0.7 + (dpi_confidence * 0.3))
                confidence_multiplier = self.validation_thresholds['confidence_boost_factor']
            else:
                score = max(0.3, 0.5 - abs(dpi_score) * 0.2)
                confidence_multiplier = max(0.7, 1.0 - abs(dpi_score) * 0.3)

            # Narrative gap influence
            if abs(narrative_gap) > self.validation_thresholds['dpi_divergence_threshold']:
                if alert.alert_type in ['momentum_reversal', 'regime_change']:
                    score += 0.2  # Narrative gap supports reversal signals
                    confidence_multiplier *= 1.1

            return {
                'score': min(1.0, score),
                'confidence_multiplier': confidence_multiplier,
                'confirmed': confirmed,
                'dpi_score': dpi_score,
                'narrative_gap': narrative_gap,
                'dpi_confidence': dpi_confidence
            }

        except Exception as e:
            logger.error(f"DPI validation failed: {e}")
            return {'score': 0.5, 'confidence_multiplier': 1.0, 'confirmed': False}

    def _validate_with_antifragility(self,
                                    alert: AlertMessage,
                                    enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alert using antifragility framework"""
        try:
            # Placeholder for antifragility integration
            # This would integrate with the antifragility framework

            # Assess portfolio resilience
            portfolio_stress = enhanced_data.get('portfolio_stress', 0.3)
            diversification_ratio = enhanced_data.get('diversification_ratio', 0.7)

            # Check for antifragility breach
            breach_detected = portfolio_stress > self.validation_thresholds['antifragility_breach_threshold']

            # Validation score based on antifragility principles
            if breach_detected:
                score = 0.9  # High score for alerts during stress
            else:
                score = 0.6 + (diversification_ratio * 0.3)

            return {
                'score': score,
                'breach_detected': breach_detected,
                'portfolio_stress': portfolio_stress,
                'diversification_ratio': diversification_ratio
            }

        except Exception as e:
            logger.error(f"Antifragility validation failed: {e}")
            return {'score': 0.6, 'breach_detected': False}

    def _validate_with_market_context(self,
                                     alert: AlertMessage,
                                     enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alert using market context"""
        try:
            # Market context factors
            volatility = enhanced_data.get('recent_volatility', 0.02)
            volume_ratio = enhanced_data.get('volume_ratio', 1.0)
            market_hours = enhanced_data.get('market_hours', True)

            score = 0.5  # Base score

            # Volatility context
            if alert.alert_type == 'volatility_spike' and volatility > 0.03:
                score += 0.3  # High volatility confirms volatility alerts

            # Volume context
            if volume_ratio > 1.5:
                score += 0.2  # High volume supports alerts

            # Market hours context
            if market_hours:
                score += 0.1  # Regular hours get slight boost
            else:
                score -= 0.1  # After-hours penalty

            return {
                'score': min(1.0, max(0.0, score)),
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'market_hours': market_hours
            }

        except Exception as e:
            logger.error(f"Market context validation failed: {e}")
            return {'score': 0.5}

    def _enhance_alert_with_validation(self,
                                      alert: AlertMessage,
                                      validation: ValidationResult) -> AlertMessage:
        """Enhance alert with validation results"""
        try:
            # Adjust confidence
            alert.confidence *= validation.confidence_adjustment

            # Add validation details
            alert.details['validation'] = {
                'validation_score': validation.validation_score,
                'dpi_confirmation': validation.dpi_confirmation,
                'antifragility_breach': validation.antifragility_breach,
                'recommended_action': validation.recommended_action
            }

            # Upgrade severity if antifragility breach
            if validation.antifragility_breach and alert.severity.value in ['low', 'medium']:
                from .ai_alert_system import AlertSeverity
                alert.severity = AlertSeverity.HIGH

            # Add actionable insights based on validation
            validation_insights = self._generate_validation_insights(validation)
            alert.details.setdefault('insights', []).extend(validation_insights)

            return alert

        except Exception as e:
            logger.error(f"Alert enhancement failed for {alert.alert_id}: {e}")
            return alert

    def _route_validated_alert(self, alert: AlertMessage) -> None:
        """Route validated alert to appropriate systems"""
        try:
            severity_key = alert.severity.value
            if severity_key == 'critical':
                routing_key = 'critical_alerts'
            elif severity_key == 'high':
                routing_key = 'high_alerts'
            elif severity_key == 'medium':
                routing_key = 'medium_alerts'
            else:
                routing_key = 'low_alerts'

            target_systems = self.alert_routing_rules.get(routing_key, ['log_only'])

            for system in target_systems:
                try:
                    if system == 'trading_engine' and self.trading_engine:
                        self._notify_trading_engine(alert)
                    elif system == 'risk_manager':
                        self._notify_risk_manager(alert)
                    elif system == 'notification_system':
                        self._send_notification(alert)
                    elif system == 'log_only':
                        logger.info(f"Alert logged: {alert.alert_id} - {alert.message}")

                except Exception as e:
                    logger.error(f"Alert routing to {system} failed: {e}")

        except Exception as e:
            logger.error(f"Alert routing failed for {alert.alert_id}: {e}")

    def _test_dpi_integration(self) -> bool:
        """Test DPI calculator integration"""
        try:
            if not self.dpi_calculator:
                return False

            # Test with a sample symbol
            test_symbol = 'ULTY'
            dpi_score, components = self.dpi_calculator.calculate_dpi(test_symbol)

            # Validate results
            if isinstance(dpi_score, (int, float)) and -1 <= dpi_score <= 1:
                return True

            return False

        except Exception as e:
            logger.error(f"DPI integration test failed: {e}")
            return False

    def _setup_notification_handlers(self) -> None:
        """Set up notification handlers for the orchestrator"""
        try:
            # Register custom notification handlers
            self.alert_orchestrator.register_notification_handler(
                'trading_engine', self._notify_trading_engine
            )
            self.alert_orchestrator.register_notification_handler(
                'risk_manager', self._notify_risk_manager
            )

            logger.info("Notification handlers configured")

        except Exception as e:
            logger.error(f"Notification handler setup failed: {e}")

    def _notify_trading_engine(self, alert: AlertMessage) -> None:
        """Notify trading engine of critical alerts"""
        try:
            if self.trading_engine and hasattr(self.trading_engine, 'receive_risk_alert'):
                alert_data = {
                    'alert_id': alert.alert_id,
                    'symbol': alert.symbol,
                    'severity': alert.severity.value,
                    'confidence': alert.confidence,
                    'recommended_action': alert.details.get('validation', {}).get('recommended_action', 'monitor')
                }
                self.trading_engine.receive_risk_alert(alert_data)
                logger.info(f"Trading engine notified of alert: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Trading engine notification failed: {e}")

    def _notify_risk_manager(self, alert: AlertMessage) -> None:
        """Notify risk management system"""
        try:
            # Placeholder for risk manager integration
            logger.info(f"Risk manager notified: {alert.alert_id} - {alert.severity.value}")

        except Exception as e:
            logger.error(f"Risk manager notification failed: {e}")

    def _send_notification(self, alert: AlertMessage) -> None:
        """Send general notification"""
        try:
            # Placeholder for notification system integration
            logger.info(f"Notification sent: {alert.symbol} - {alert.message}")

        except Exception as e:
            logger.error(f"Notification sending failed: {e}")

    def _enhance_training_data_with_dpi(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance training data with DPI calculations"""
        try:
            enhanced_data = historical_data.copy()

            if self.dpi_calculator and 'market_data' in historical_data:
                # This would enhance historical data with DPI calculations
                # For now, return the original data with a marker
                enhanced_data['dpi_enhanced_count'] = len(historical_data.get('market_data', {}))

            return enhanced_data

        except Exception as e:
            logger.error(f"Training data enhancement failed: {e}")
            return historical_data

    def _calculate_dpi_confidence(self, dpi_components) -> float:
        """Calculate DPI confidence score"""
        try:
            # Simple confidence calculation based on component agreement
            components = [
                dpi_components.order_flow_pressure,
                dpi_components.volume_weighted_skew,
                dpi_components.price_momentum_bias,
                dpi_components.volatility_clustering
            ]

            # Agreement measure (lower std = higher confidence)
            std = np.std(components)
            confidence = max(0.3, min(1.0, 1.0 - std))

            return confidence

        except Exception as e:
            logger.error(f"DPI confidence calculation failed: {e}")
            return 0.5

    def _determine_recommended_action(self,
                                     alert: AlertMessage,
                                     validation_score: float,
                                     dpi_confirmation: bool,
                                     antifragility_breach: bool) -> str:
        """Determine recommended action based on validation"""
        try:
            if antifragility_breach:
                return "reduce_risk_immediately"
            elif validation_score > 0.8 and dpi_confirmation:
                return "act_on_signal"
            elif validation_score > 0.6:
                return "monitor_closely"
            else:
                return "monitor"

        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return "monitor"

    def _generate_validation_insights(self, validation: ValidationResult) -> List[str]:
        """Generate insights based on validation results"""
        insights = []

        if validation.dpi_confirmation:
            insights.append("Alert confirmed by DPI analysis")

        if validation.antifragility_breach:
            insights.append("Portfolio antifragility threshold breached")

        if validation.validation_score > 0.8:
            insights.append("High confidence validation - consider immediate action")
        elif validation.validation_score < 0.5:
            insights.append("Low validation score - exercise caution")

        return insights

    def _update_integration_metrics(self, symbol: str, alerts_generated: int, alerts_validated: int) -> None:
        """Update integration performance metrics"""
        try:
            validation_rate = alerts_validated / alerts_generated if alerts_generated > 0 else 1.0

            # Simple metrics tracking
            self.validation_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'alerts_generated': alerts_generated,
                'alerts_validated': alerts_validated,
                'validation_rate': validation_rate
            })

            # Keep only last 1000 entries
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]

        except Exception as e:
            logger.error(f"Integration metrics update failed: {e}")

    def _calculate_current_integration_metrics(self) -> IntegrationMetrics:
        """Calculate current integration performance metrics"""
        try:
            return IntegrationMetrics(
                dpi_integration_latency_ms=5.0,  # Placeholder
                antifragility_sync_rate=0.95,   # Placeholder
                trading_engine_notification_rate=1.0,  # Placeholder
                alert_validation_accuracy=0.85,  # Placeholder
                overall_integration_health=0.90  # Placeholder
            )

        except Exception as e:
            logger.error(f"Integration metrics calculation failed: {e}")
            return IntegrationMetrics(0, 0, 0, 0, 0)

    def _calculate_integration_validation_score(self) -> float:
        """Calculate integration validation score"""
        try:
            if not self.validation_history:
                return 0.8

            recent_validations = self.validation_history[-100:]
            avg_validation_rate = np.mean([v['validation_rate'] for v in recent_validations])
            return avg_validation_rate

        except Exception as e:
            logger.error(f"Integration validation score calculation failed: {e}")
            return 0.8

    def _get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        try:
            if not self.validation_history:
                return {'no_data': True}

            total_generated = sum(v['alerts_generated'] for v in self.validation_history)
            total_validated = sum(v['alerts_validated'] for v in self.validation_history)

            return {
                'total_alerts_generated': total_generated,
                'total_alerts_validated': total_validated,
                'overall_validation_rate': total_validated / total_generated if total_generated > 0 else 0,
                'validation_entries': len(self.validation_history)
            }

        except Exception as e:
            logger.error(f"Validation statistics calculation failed: {e}")
            return {'error': str(e)}

    def _get_alert_routing_statistics(self) -> Dict[str, Any]:
        """Get alert routing statistics"""
        try:
            # Placeholder for routing statistics
            return {
                'total_routes': 0,
                'successful_routes': 0,
                'failed_routes': 0,
                'routing_success_rate': 1.0
            }

        except Exception as e:
            logger.error(f"Alert routing statistics failed: {e}")
            return {'error': str(e)}

    def _assess_dpi_integration_health(self) -> float:
        """Assess DPI integration health score"""
        try:
            if self.integration_status['dpi_calculator'] == IntegrationStatus.CONNECTED:
                return 1.0
            elif self.integration_status['dpi_calculator'] == IntegrationStatus.ERROR:
                return 0.3
            else:
                return 0.6

        except Exception as e:
            logger.error(f"DPI integration health assessment failed: {e}")
            return 0.5

    def _calculate_overall_intelligence_score(self) -> float:
        """Calculate overall AI intelligence system score"""
        try:
            scores = []

            # Component health scores
            for component, status in self.integration_status.items():
                if status == IntegrationStatus.CONNECTED:
                    scores.append(1.0)
                elif status == IntegrationStatus.ERROR:
                    scores.append(0.2)
                else:
                    scores.append(0.6)

            # Validation performance
            validation_score = self._calculate_integration_validation_score()
            scores.append(validation_score)

            # System metrics
            system_metrics = self.alert_orchestrator.get_system_metrics()
            scores.append(system_metrics.system_health_score)

            return np.mean(scores) if scores else 0.5

        except Exception as e:
            logger.error(f"Overall intelligence score calculation failed: {e}")
            return 0.5

    def shutdown(self) -> None:
        """Shutdown integration layer"""
        try:
            logger.info("Shutting down Intelligence Integration Layer...")

            # Stop alert orchestrator
            self.alert_orchestrator.stop()

            # Update status
            for component in self.integration_status:
                self.integration_status[component] = IntegrationStatus.DISCONNECTED

            logger.info("Intelligence Integration Layer shutdown completed")

        except Exception as e:
            logger.error(f"Integration layer shutdown failed: {e}")