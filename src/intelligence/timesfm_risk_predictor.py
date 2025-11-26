"""
TimesFM-Based Risk Prediction System
Extends predictive warning system from 5-15min to 6-48hr horizon using TimesFM
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .timesfm_forecaster import TimesFMForecaster, VolatilityForecast

logger = logging.getLogger(__name__)


class RiskHorizon(Enum):
    """Risk forecast horizons"""
    IMMEDIATE = "immediate"  # 1-6 hours
    SHORT_TERM = "short_term"  # 6-24 hours
    MEDIUM_TERM = "medium_term"  # 24-48 hours
    LONG_TERM = "long_term"  # 48-168 hours (7 days)


class RiskEventType(Enum):
    """Extended risk event types for long-horizon prediction"""
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_COLLAPSE = "volatility_collapse"
    REGIME_SHIFT = "regime_shift"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MOMENTUM_REVERSAL = "momentum_reversal"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    BLACK_SWAN = "black_swan"


@dataclass
class MultiHorizonRiskForecast:
    """Risk forecast across multiple time horizons"""
    symbol: str
    forecast_time: datetime
    immediate_risk: Dict[str, float]  # 1-6hr
    short_term_risk: Dict[str, float]  # 6-24hr
    medium_term_risk: Dict[str, float]  # 24-48hr
    long_term_risk: Dict[str, float]  # 48hr-7d
    volatility_forecast: VolatilityForecast
    confidence: float
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """Enhanced risk alert with long-range prediction"""
    alert_id: str
    timestamp: datetime
    risk_type: RiskEventType
    probability: float
    horizon: RiskHorizon
    time_to_event_hours: float
    severity: float  # 0-1 scale
    description: str
    mitigating_actions: List[str]
    affected_strategies: List[str]
    confidence: float


class TimesFMRiskPredictor:
    """
    Multi-horizon risk prediction using TimesFM
    Extends warning horizon from 5-15min to 6-168 hours (7 days)
    """

    def __init__(self, forecaster: Optional[TimesFMForecaster] = None):
        """
        Initialize risk predictor

        Args:
            forecaster: TimesFM forecaster instance (creates new if None)
        """
        self.forecaster = forecaster or TimesFMForecaster(use_fallback=True)

        # Risk thresholds for different horizons
        self.risk_thresholds = {
            RiskHorizon.IMMEDIATE: {
                'volatility_spike': 0.30,  # 30% prob in 1-6hr
                'regime_shift': 0.40,
                'black_swan': 0.15
            },
            RiskHorizon.SHORT_TERM: {
                'volatility_spike': 0.25,  # 25% prob in 6-24hr
                'regime_shift': 0.35,
                'black_swan': 0.10
            },
            RiskHorizon.MEDIUM_TERM: {
                'volatility_spike': 0.20,
                'regime_shift': 0.30,
                'black_swan': 0.08
            },
            RiskHorizon.LONG_TERM: {
                'volatility_spike': 0.15,
                'regime_shift': 0.25,
                'black_swan': 0.05
            }
        }

        # Historical data buffers
        self.vix_history = []
        self.price_history = {}  # {symbol: history}
        self.alert_history = []

    def predict_multi_horizon_risk(self,
                                   symbol: str,
                                   vix_history: np.ndarray,
                                   price_history: np.ndarray,
                                   market_state: Dict[str, Any]) -> MultiHorizonRiskForecast:
        """
        Generate multi-horizon risk forecast

        Args:
            symbol: Trading symbol
            vix_history: Historical VIX data
            price_history: Historical price data
            market_state: Current market state dictionary

        Returns:
            MultiHorizonRiskForecast with risk across all horizons
        """
        # Forecast volatility for multiple horizons
        vol_forecasts = {
            'immediate': self.forecaster.forecast_volatility(vix_history, horizon_hours=6),
            'short_term': self.forecaster.forecast_volatility(vix_history, horizon_hours=24),
            'medium_term': self.forecaster.forecast_volatility(vix_history, horizon_hours=48),
            'long_term': self.forecaster.forecast_volatility(vix_history, horizon_hours=168)
        }

        # Calculate risk scores for each horizon
        immediate_risk = self._calculate_horizon_risk(
            vol_forecasts['immediate'], market_state, RiskHorizon.IMMEDIATE
        )

        short_term_risk = self._calculate_horizon_risk(
            vol_forecasts['short_term'], market_state, RiskHorizon.SHORT_TERM
        )

        medium_term_risk = self._calculate_horizon_risk(
            vol_forecasts['medium_term'], market_state, RiskHorizon.MEDIUM_TERM
        )

        long_term_risk = self._calculate_horizon_risk(
            vol_forecasts['long_term'], market_state, RiskHorizon.LONG_TERM
        )

        # Calculate overall confidence
        confidence = np.mean([
            vol_forecasts['immediate'].confidence,
            vol_forecasts['short_term'].confidence,
            vol_forecasts['medium_term'].confidence,
            vol_forecasts['long_term'].confidence
        ])

        # Generate recommendations
        recommendations = self._generate_recommendations(
            immediate_risk, short_term_risk, medium_term_risk, long_term_risk
        )

        return MultiHorizonRiskForecast(
            symbol=symbol,
            forecast_time=datetime.now(),
            immediate_risk=immediate_risk,
            short_term_risk=short_term_risk,
            medium_term_risk=medium_term_risk,
            long_term_risk=long_term_risk,
            volatility_forecast=vol_forecasts['medium_term'],  # Use 48hr as primary
            confidence=float(confidence),
            recommended_actions=recommendations,
            metadata={
                'market_state': market_state,
                'vix_current': float(vix_history[-1])
            }
        )

    def generate_risk_alerts(self,
                            risk_forecast: MultiHorizonRiskForecast,
                            min_probability: float = 0.20) -> List[RiskAlert]:
        """
        Generate risk alerts from forecast

        Args:
            risk_forecast: Multi-horizon risk forecast
            min_probability: Minimum probability threshold for alerts

        Returns:
            List of RiskAlert objects
        """
        alerts = []

        # Check all horizons for risks exceeding threshold
        horizon_map = {
            'immediate': (RiskHorizon.IMMEDIATE, 3),  # 3hr average
            'short_term': (RiskHorizon.SHORT_TERM, 15),  # 15hr average
            'medium_term': (RiskHorizon.MEDIUM_TERM, 36),  # 36hr average
            'long_term': (RiskHorizon.LONG_TERM, 84)  # 84hr (3.5 days) average
        }

        for horizon_key, (horizon, avg_hours) in horizon_map.items():
            risk_dict = getattr(risk_forecast, f"{horizon_key}_risk")

            for risk_type, probability in risk_dict.items():
                if probability >= min_probability:
                    # Create alert
                    alert = RiskAlert(
                        alert_id=f"{risk_forecast.symbol}_{horizon.value}_{risk_type}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        risk_type=RiskEventType(risk_type),
                        probability=float(probability),
                        horizon=horizon,
                        time_to_event_hours=float(avg_hours),
                        severity=self._calculate_severity(probability, risk_type),
                        description=self._generate_alert_description(risk_type, probability, avg_hours),
                        mitigating_actions=self._get_mitigating_actions(risk_type, horizon),
                        affected_strategies=self._get_affected_strategies(risk_type),
                        confidence=risk_forecast.confidence
                    )
                    alerts.append(alert)

        # Sort by severity (descending)
        alerts.sort(key=lambda a: a.severity * a.probability, reverse=True)

        self.alert_history.extend(alerts)
        return alerts

    def _calculate_horizon_risk(self,
                                vol_forecast: VolatilityForecast,
                                market_state: Dict[str, Any],
                                horizon: RiskHorizon) -> Dict[str, float]:
        """Calculate risk probabilities for a specific horizon"""
        risks = {}

        # Volatility spike risk
        risks['volatility_spike'] = vol_forecast.spike_probability

        # Volatility collapse risk (sudden calm)
        current_vix = market_state.get('vix_level', 20)
        if current_vix > 25:
            collapse_prob = np.mean(vol_forecast.vix_forecast < 15)
            risks['volatility_collapse'] = float(collapse_prob)
        else:
            risks['volatility_collapse'] = 0.0

        # Regime shift risk (change from current regime)
        current_regime = market_state.get('regime', 'normal')
        regime_changes = sum(1 for r in vol_forecast.regime_forecast if r != current_regime)
        risks['regime_shift'] = regime_changes / len(vol_forecast.regime_forecast)

        # Black swan risk (VIX > 40)
        risks['black_swan'] = vol_forecast.crisis_probability

        # Liquidity crisis risk (inferred from extreme volatility + regime)
        if vol_forecast.crisis_probability > 0.3 and risks['regime_shift'] > 0.5:
            risks['liquidity_crisis'] = (vol_forecast.crisis_probability + risks['regime_shift']) / 2
        else:
            risks['liquidity_crisis'] = 0.0

        # Momentum reversal risk (based on VIX trend)
        vix_trend = np.mean(np.diff(vol_forecast.vix_forecast[:10]))  # First 10 periods
        if abs(vix_trend) > 2.0:  # Strong trend
            risks['momentum_reversal'] = min(0.5, abs(vix_trend) / 10)
        else:
            risks['momentum_reversal'] = 0.0

        # Correlation breakdown (inferred from volatility spike + regime shift)
        if risks['volatility_spike'] > 0.3 and risks['regime_shift'] > 0.3:
            risks['correlation_breakdown'] = (risks['volatility_spike'] + risks['regime_shift']) / 2
        else:
            risks['correlation_breakdown'] = 0.0

        return risks

    def _calculate_severity(self, probability: float, risk_type: str) -> float:
        """Calculate risk severity (0-1)"""
        # Base severity on probability
        base_severity = probability

        # Adjust for risk type impact
        impact_multipliers = {
            'black_swan': 1.5,
            'liquidity_crisis': 1.4,
            'correlation_breakdown': 1.3,
            'regime_shift': 1.2,
            'volatility_spike': 1.1,
            'momentum_reversal': 1.0,
            'volatility_collapse': 0.8
        }

        multiplier = impact_multipliers.get(risk_type, 1.0)
        severity = min(1.0, base_severity * multiplier)

        return float(severity)

    def _generate_alert_description(self, risk_type: str, probability: float, hours: float) -> str:
        """Generate human-readable alert description"""
        descriptions = {
            'volatility_spike': f"VIX spike probability: {probability:.1%} within {hours:.0f} hours",
            'volatility_collapse': f"Volatility collapse probability: {probability:.1%} within {hours:.0f} hours",
            'regime_shift': f"Market regime shift probability: {probability:.1%} within {hours:.0f} hours",
            'black_swan': f"Black swan event probability: {probability:.1%} within {hours:.0f} hours",
            'liquidity_crisis': f"Liquidity crisis probability: {probability:.1%} within {hours:.0f} hours",
            'momentum_reversal': f"Momentum reversal probability: {probability:.1%} within {hours:.0f} hours",
            'correlation_breakdown': f"Correlation breakdown probability: {probability:.1%} within {hours:.0f} hours"
        }

        return descriptions.get(risk_type, f"Risk event '{risk_type}': {probability:.1%} in {hours:.0f}h")

    def _get_mitigating_actions(self, risk_type: str, horizon: RiskHorizon) -> List[str]:
        """Get recommended mitigating actions for risk type"""
        actions = {
            'volatility_spike': [
                "Reduce position sizes",
                "Activate tail hedge strategy",
                "Increase cash reserves",
                "Consider VIX calls"
            ],
            'black_swan': [
                "ACTIVATE KILL SWITCH if probability >50%",
                "Deploy crisis_alpha strategy",
                "Exit momentum positions",
                "Move to capital preservation mode"
            ],
            'regime_shift': [
                "Reassess all strategy allocations",
                "Prepare for strategy rotation",
                "Monitor correlation changes",
                "Adjust position sizes preemptively"
            ],
            'liquidity_crisis': [
                "Reduce position sizes significantly",
                "Avoid illiquid assets",
                "Increase cash to 80%+",
                "Prepare for gap fills"
            ],
            'momentum_reversal': [
                "Take profits on momentum trades",
                "Activate mean_reversion strategy",
                "Tighten stop losses",
                "Consider counter-trend positioning"
            ],
            'correlation_breakdown': [
                "Activate correlation_breakdown strategy",
                "Reassess hedges",
                "Monitor pair trades",
                "Prepare for diversification failure"
            ]
        }

        base_actions = actions.get(risk_type, ["Monitor situation closely"])

        # Add horizon-specific timing
        if horizon == RiskHorizon.IMMEDIATE:
            base_actions.insert(0, "IMMEDIATE ACTION REQUIRED")
        elif horizon == RiskHorizon.SHORT_TERM:
            base_actions.insert(0, "Prepare for action within 24h")
        elif horizon == RiskHorizon.MEDIUM_TERM:
            base_actions.insert(0, "Plan strategy adjustment for 24-48h window")
        else:
            base_actions.insert(0, "Monitor and prepare for next 3-7 days")

        return base_actions

    def _get_affected_strategies(self, risk_type: str) -> List[str]:
        """Get list of strategies affected by risk type"""
        strategy_impacts = {
            'volatility_spike': ['momentum_explosion', 'inequality_arbitrage'],
            'black_swan': ['momentum_explosion', 'mean_reversion', 'inequality_arbitrage'],
            'regime_shift': ['all'],
            'liquidity_crisis': ['all'],
            'momentum_reversal': ['momentum_explosion'],
            'correlation_breakdown': ['mean_reversion', 'inequality_arbitrage'],
            'volatility_collapse': ['tail_hedge', 'volatility_harvest', 'crisis_alpha']
        }

        return strategy_impacts.get(risk_type, [])

    def _generate_recommendations(self,
                                  immediate: Dict[str, float],
                                  short_term: Dict[str, float],
                                  medium_term: Dict[str, float],
                                  long_term: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations from multi-horizon risk"""
        recommendations = []

        # Check for immediate high-risk situations
        immediate_max = max(immediate.values())
        if immediate_max > 0.40:
            recommendations.append(f"⚠️ HIGH IMMEDIATE RISK ({immediate_max:.1%}) - Consider defensive positioning")

        # Check for building risk trends
        short_max = max(short_term.values())
        if short_max > 0.30:
            recommendations.append(f"📊 Elevated short-term risk ({short_max:.1%}) - Monitor closely")

        # Medium-term planning
        medium_max = max(medium_term.values())
        if medium_max > 0.25:
            recommendations.append(f"📅 Medium-term risk building ({medium_max:.1%}) - Plan strategy adjustments")

        # Long-term positioning
        long_max = max(long_term.values())
        if long_max > 0.20:
            recommendations.append(f"🔮 Long-term risk detected ({long_max:.1%}) - Consider portfolio rebalancing")

        # If no significant risks
        if not recommendations:
            recommendations.append("✅ No significant risks detected across all horizons")

        return recommendations


if __name__ == "__main__":
    # Test TimesFM risk predictor
    print("=== Testing TimesFM Risk Predictor ===")

    # Create synthetic data
    vix_history = np.random.normal(22, 6, 500)  # 500 hours of VIX
    vix_history = np.clip(vix_history, 10, 60)

    price_history = np.random.normal(400, 15, 500)  # SPY-like

    market_state = {
        'vix_level': float(vix_history[-1]),
        'regime': 'normal',
        'correlation': 0.65
    }

    # Initialize predictor
    predictor = TimesFMRiskPredictor()

    # Generate multi-horizon forecast
    print("\n1. Generating multi-horizon risk forecast...")
    forecast = predictor.predict_multi_horizon_risk(
        symbol='SPY',
        vix_history=vix_history,
        price_history=price_history,
        market_state=market_state
    )

    print("\nImmediate Risk (1-6hr):")
    for risk, prob in forecast.immediate_risk.items():
        if prob > 0.1:
            print(f"  {risk}: {prob:.1%}")

    print("\nShort-term Risk (6-24hr):")
    for risk, prob in forecast.short_term_risk.items():
        if prob > 0.1:
            print(f"  {risk}: {prob:.1%}")

    print("\nRecommendations:")
    for rec in forecast.recommended_actions:
        print(f"  {rec}")

    # Generate alerts
    print("\n2. Generating risk alerts...")
    alerts = predictor.generate_risk_alerts(forecast, min_probability=0.15)

    for alert in alerts[:5]:  # Top 5 alerts
        print(f"\n📢 ALERT: {alert.risk_type.value}")
        print(f"   Horizon: {alert.horizon.value} ({alert.time_to_event_hours:.0f}h)")
        print(f"   Probability: {alert.probability:.1%}")
        print(f"   Severity: {alert.severity:.2f}")
        print(f"   {alert.description}")

    print("\n=== TimesFM Risk Predictor Test Complete ===")