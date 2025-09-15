"""
AI Signal Generator - Gary-Style Mathematical Framework
Implements DPI, Narrative Gap, Repricing Potential with AI self-calibration
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging
from .ai_calibration_engine import ai_calibration_engine

logger = logging.getLogger(__name__)

@dataclass
class CohortData:
    """Wealth/income cohort for DPI calculation"""
    name: str
    income_percentile: Tuple[float, float]  # (min, max) percentiles
    population_weight: float
    net_cash_flow: float  # Current period net cash flow
    historical_flows: List[float] = field(default_factory=list)

@dataclass
class MarketExpectation:
    """Market-implied expectation for asset/path"""
    asset: str
    timeframe: str
    implied_probability: float
    implied_return: float
    confidence_interval: Tuple[float, float]
    source: str  # 'options', 'futures', 'rates', etc.

@dataclass
class AISignal:
    """Complete AI-generated signal with components"""
    asset: str
    timestamp: datetime
    dpi_component: float
    narrative_gap: float
    repricing_potential: float
    catalyst_timing: float
    carry_cost: float
    composite_signal: float
    ai_confidence: float
    ai_expected_return: float
    ai_risk_estimate: float
    signal_strength: str  # 'weak', 'medium', 'strong', 'gary_moment'

class AISignalGenerator:
    """
    AI-driven signal generation using mathematical framework:

    1. DPI_t = Sum(omega_g^AI * DeltaNetCashFlow_g) - learned cohort weights
    2. NG_t^(i) = E^AI[Path_i] - E^market[Path_i] - AI vs market expectations
    3. RP_t^(i) = |NG_t^(i)| × Conf_AI,t × φ(catalyst_t) - CarryCost_t^(i)
    4. S_i,AI = w1^AI × (ΔDPI) + w2^AI × NG_i + w3^AI × φ(catalyst) - w4^AI × carry_i
    """

    def __init__(self):
        self.ai_signal_weights = {
            'dpi': 0.4,      # w1^AI - weight for DPI component
            'narrative': 0.3, # w2^AI - weight for narrative gap
            'catalyst': 0.2,  # w3^AI - weight for catalyst timing
            'carry': 0.1      # w4^AI - weight for carry cost
        }

        # AI learns these cohort weights through performance feedback
        self.ai_cohort_weights = {
            'top_1_pct': 0.5,     # Ultra wealthy (top 1%)
            'top_10_pct': 0.3,    # High earners (1-10%)
            'middle_class': 0.15,  # Middle class (10-90%)
            'bottom_10_pct': 0.05  # Lower income (bottom 10%)
        }

        # Catalyst timing parameters (AI learns optimal λ and τ)
        self.catalyst_decay_rate = 0.1    # λ in exponential decay
        self.catalyst_half_life = 30      # τ in rational decay

        # Historical data for learning
        self.signal_history: List[AISignal] = []
        self.performance_feedback: List[Dict[str, Any]] = []

    def calculate_dpi(self, cohort_data: List[CohortData]) -> float:
        """
        Calculate Distributional Pressure Index with AI-learned weights

        DPI_t = Sum(omega_g^AI * DeltaNetCashFlow_g)
        where ω_g^AI are learned through gradient descent on prediction accuracy
        """
        if not cohort_data:
            return 0.0

        dpi = 0.0
        total_weight = 0.0

        for cohort in cohort_data:
            # Get AI weight for this cohort
            cohort_key = self._map_cohort_to_key(cohort)
            ai_weight = self.ai_cohort_weights.get(cohort_key, 0.1)

            # Calculate change in net cash flow
            if len(cohort.historical_flows) > 0:
                delta_flow = cohort.net_cash_flow - cohort.historical_flows[-1]
            else:
                delta_flow = cohort.net_cash_flow

            # Weight by AI-learned importance and population
            weighted_contribution = ai_weight * cohort.population_weight * delta_flow
            dpi += weighted_contribution
            total_weight += ai_weight * cohort.population_weight

        # Normalize by total weight
        if total_weight > 0:
            dpi /= total_weight

        return dpi

    def calculate_narrative_gap(self,
                              asset: str,
                              ai_model_expectation: float,
                              market_expectations: List[MarketExpectation]) -> float:
        """
        Calculate Narrative Gap: AI's view vs Market consensus

        NG_t^(i) = E^AI[Path_i] - E^market[Path_i]
        """
        if not market_expectations:
            return 0.0

        # Find market expectation for this asset
        market_exp = None
        for exp in market_expectations:
            if exp.asset.lower() == asset.lower():
                market_exp = exp
                break

        if not market_exp:
            return 0.0

        # Calculate gap between AI and market
        narrative_gap = ai_model_expectation - market_exp.implied_return

        # Make prediction for calibration
        prediction_id = ai_calibration_engine.make_prediction(
            prediction_value=min(1.0, max(0.0, (ai_model_expectation + 1) / 2)),  # Convert to 0-1
            confidence=0.7,  # Base confidence - will be adjusted
            context={
                'type': 'narrative_gap',
                'asset': asset,
                'ai_expectation': ai_model_expectation,
                'market_expectation': market_exp.implied_return,
                'gap': narrative_gap
            }
        )

        logger.info(f"Narrative gap for {asset}: {narrative_gap:.4f} (prediction: {prediction_id})")

        return narrative_gap

    def calculate_catalyst_timing_factor(self,
                                       catalyst_events: List[Dict[str, Any]]) -> float:
        """
        Calculate catalyst timing factor using AI-learned parameters

        φ(Δt) = e^(-λ*Δt) or φ(Δt) = 1/(1 + Δt/τ)
        """
        if not catalyst_events:
            return 0.1  # Low base catalyst factor

        max_factor = 0.0

        for event in catalyst_events:
            days_until = event.get('days_until_event', 999)
            importance = event.get('importance', 0.5)

            # Use exponential decay model (AI can learn to prefer rational decay)
            if self.catalyst_decay_rate > 0:
                decay_factor = np.exp(-self.catalyst_decay_rate * days_until / 30.0)
            else:
                # Rational decay model
                decay_factor = 1.0 / (1.0 + days_until / self.catalyst_half_life)

            # Weight by event importance
            event_factor = importance * decay_factor
            max_factor = max(max_factor, event_factor)

        return max_factor

    def calculate_repricing_potential(self,
                                    narrative_gap: float,
                                    ai_confidence: float,
                                    catalyst_factor: float,
                                    carry_cost: float) -> float:
        """
        Calculate Repricing Potential with AI confidence

        RP_t^(i) = |NG_t^(i)| × Conf_AI,t × φ(catalyst_t) - CarryCost_t^(i)
        """
        # Get AI's calibrated confidence
        calibrated_confidence = ai_calibration_engine.get_ai_decision_confidence(ai_confidence)

        repricing_potential = (
            abs(narrative_gap) *
            calibrated_confidence *
            catalyst_factor -
            carry_cost
        )

        return repricing_potential

    def generate_composite_signal(self,
                                asset: str,
                                cohort_data: List[CohortData],
                                ai_model_expectation: float,
                                market_expectations: List[MarketExpectation],
                                catalyst_events: List[Dict[str, Any]],
                                carry_cost: float = 0.0) -> AISignal:
        """
        Generate complete AI signal using mathematical framework

        S_i,AI = w1^AI × (ΔDPI) + w2^AI × NG_i + w3^AI × φ(catalyst) - w4^AI × carry_i
        """

        # Component calculations
        dpi = self.calculate_dpi(cohort_data)
        narrative_gap = self.calculate_narrative_gap(asset, ai_model_expectation, market_expectations)
        catalyst_factor = self.calculate_catalyst_timing_factor(catalyst_events)

        # AI confidence based on calibration history
        base_confidence = min(1.0, abs(narrative_gap) + catalyst_factor)
        ai_confidence = ai_calibration_engine.get_ai_decision_confidence(base_confidence)

        repricing_potential = self.calculate_repricing_potential(
            narrative_gap, ai_confidence, catalyst_factor, carry_cost
        )

        # Composite signal with AI-learned weights
        composite_signal = (
            self.ai_signal_weights['dpi'] * dpi +
            self.ai_signal_weights['narrative'] * narrative_gap +
            self.ai_signal_weights['catalyst'] * catalyst_factor -
            self.ai_signal_weights['carry'] * carry_cost
        )

        # AI risk estimate using calibrated parameters
        risk_estimate = self._calculate_ai_risk_estimate(narrative_gap, catalyst_factor)

        # Determine signal strength
        signal_strength = self._classify_signal_strength(
            composite_signal, ai_confidence, repricing_potential
        )

        signal = AISignal(
            asset=asset,
            timestamp=datetime.now(),
            dpi_component=dpi,
            narrative_gap=narrative_gap,
            repricing_potential=repricing_potential,
            catalyst_timing=catalyst_factor,
            carry_cost=carry_cost,
            composite_signal=composite_signal,
            ai_confidence=ai_confidence,
            ai_expected_return=ai_model_expectation,
            ai_risk_estimate=risk_estimate,
            signal_strength=signal_strength
        )

        # Store for learning
        self.signal_history.append(signal)

        # Make overall prediction for this signal
        signal_prediction_id = ai_calibration_engine.make_prediction(
            prediction_value=min(1.0, max(0.0, (composite_signal + 1) / 2)),
            confidence=ai_confidence,
            context={
                'type': 'composite_signal',
                'asset': asset,
                'signal_components': {
                    'dpi': dpi,
                    'narrative_gap': narrative_gap,
                    'catalyst_timing': catalyst_factor,
                    'repricing_potential': repricing_potential
                }
            }
        )

        logger.info(f"Generated AI signal for {asset}: {composite_signal:.4f} "
                   f"(strength: {signal_strength}, confidence: {ai_confidence:.3f})")

        return signal

    def update_signal_weights_from_performance(self,
                                             signal_id: str,
                                             actual_return: float,
                                             time_horizon_days: int):
        """
        Update AI signal weights based on actual performance
        Implements gradient descent on prediction accuracy
        """
        # Find the signal
        signal = None
        for s in self.signal_history:
            if s.asset == signal_id.split('_')[0]:  # Simple matching
                signal = s
                break

        if not signal:
            return

        # Calculate prediction error
        prediction_error = actual_return - signal.ai_expected_return

        # Update weights using gradient descent
        learning_rate = 0.01

        # Gradient with respect to each component
        if signal.dpi_component != 0:
            gradient_dpi = prediction_error * signal.dpi_component / abs(signal.composite_signal + 1e-8)
            self.ai_signal_weights['dpi'] += learning_rate * gradient_dpi

        if signal.narrative_gap != 0:
            gradient_narrative = prediction_error * signal.narrative_gap / abs(signal.composite_signal + 1e-8)
            self.ai_signal_weights['narrative'] += learning_rate * gradient_narrative

        if signal.catalyst_timing != 0:
            gradient_catalyst = prediction_error * signal.catalyst_timing / abs(signal.composite_signal + 1e-8)
            self.ai_signal_weights['catalyst'] += learning_rate * gradient_catalyst

        # Normalize weights to sum to 1.0 (excluding carry which is subtracted)
        total_positive_weight = (self.ai_signal_weights['dpi'] +
                               self.ai_signal_weights['narrative'] +
                               self.ai_signal_weights['catalyst'])

        if total_positive_weight > 0:
            self.ai_signal_weights['dpi'] /= total_positive_weight
            self.ai_signal_weights['narrative'] /= total_positive_weight
            self.ai_signal_weights['catalyst'] /= total_positive_weight

        # Ensure reasonable bounds
        for key in ['dpi', 'narrative', 'catalyst']:
            self.ai_signal_weights[key] = max(0.05, min(0.8, self.ai_signal_weights[key]))

        logger.info(f"Updated AI signal weights: {self.ai_signal_weights}")

    def _calculate_ai_risk_estimate(self, narrative_gap: float, catalyst_factor: float) -> float:
        """Calculate AI's risk estimate using calibrated parameters"""
        base_risk = abs(narrative_gap) * 0.5  # Base volatility from gap
        catalyst_risk = (1.0 - catalyst_factor) * 0.3  # Higher risk with distant catalysts

        # Use AI's learned risk aversion parameter
        ai_risk_aversion = ai_calibration_engine.utility_params.risk_aversion
        risk_adjustment = 1.0 + (ai_risk_aversion - 0.5) * 0.2

        return (base_risk + catalyst_risk) * risk_adjustment

    def _classify_signal_strength(self,
                                composite_signal: float,
                                ai_confidence: float,
                                repricing_potential: float) -> str:
        """Classify signal strength for UI display"""

        # Gary Moment: High conviction, high repricing potential
        if (abs(composite_signal) > 0.7 and
            ai_confidence > 0.8 and
            repricing_potential > 0.5):
            return 'gary_moment'

        # Strong signal
        elif (abs(composite_signal) > 0.5 and
              ai_confidence > 0.7):
            return 'strong'

        # Medium signal
        elif (abs(composite_signal) > 0.3 and
              ai_confidence > 0.6):
            return 'medium'

        # Weak signal
        else:
            return 'weak'

    def _map_cohort_to_key(self, cohort: CohortData) -> str:
        """Map cohort to AI weight key based on income percentile"""
        min_pct, max_pct = cohort.income_percentile

        if max_pct <= 1.0:
            return 'top_1_pct'
        elif max_pct <= 10.0:
            return 'top_10_pct'
        elif min_pct >= 90.0:
            return 'bottom_10_pct'
        else:
            return 'middle_class'

    def get_current_signal_weights(self) -> Dict[str, float]:
        """Get current AI-learned signal weights"""
        return self.ai_signal_weights.copy()

    def get_cohort_weights(self) -> Dict[str, float]:
        """Get current AI-learned cohort weights"""
        return self.ai_cohort_weights.copy()

    def export_signal_analysis(self, signal: AISignal) -> Dict[str, Any]:
        """Export signal analysis for UI display"""
        return {
            'asset': signal.asset,
            'timestamp': signal.timestamp.isoformat(),
            'signal_strength': signal.signal_strength,
            'composite_signal': signal.composite_signal,
            'ai_confidence': signal.ai_confidence,
            'components': {
                'dpi': signal.dpi_component,
                'narrative_gap': signal.narrative_gap,
                'repricing_potential': signal.repricing_potential,
                'catalyst_timing': signal.catalyst_timing,
                'carry_cost': signal.carry_cost
            },
            'ai_estimates': {
                'expected_return': signal.ai_expected_return,
                'risk_estimate': signal.ai_risk_estimate
            },
            'signal_weights': self.ai_signal_weights,
            'interpretation': self._generate_signal_interpretation(signal)
        }

    def _generate_signal_interpretation(self, signal: AISignal) -> str:
        """Generate human-readable interpretation of the signal"""
        interpretations = []

        if signal.dpi_component > 0.3:
            interpretations.append("Strong wealth concentration pressure supporting the trade")
        elif signal.dpi_component < -0.3:
            interpretations.append("Wealth flows opposing the trade thesis")

        if abs(signal.narrative_gap) > 0.4:
            interpretations.append(f"Significant gap between AI model and market consensus ({signal.narrative_gap:.2f})")

        if signal.catalyst_timing > 0.7:
            interpretations.append("High-impact catalyst approaching")
        elif signal.catalyst_timing < 0.3:
            interpretations.append("Distant or weak catalysts")

        if signal.signal_strength == 'gary_moment':
            interpretations.append("!!! GARY MOMENT: High conviction contrarian opportunity")

        return "; ".join(interpretations) if interpretations else "Mixed signals - proceed with caution"

# Global AI signal generator instance
ai_signal_generator = AISignalGenerator()