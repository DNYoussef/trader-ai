"""
Context-Aware Alert Filtering System

Advanced filtering system to reduce false positives through market context analysis:
- Market condition assessment and regime awareness
- Temporal filtering based on trading sessions and market hours
- Cross-asset correlation analysis for portfolio-wide context
- News sentiment and market microstructure integration
- Adaptive filtering based on historical performance
"""

import logging
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market trading sessions"""
    PRE_MARKET = "pre_market"
    REGULAR_HOURS = "regular_hours"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


class CorrelationRegime(Enum):
    """Market correlation regimes"""
    LOW_CORRELATION = "low_correlation"
    NORMAL_CORRELATION = "normal_correlation"
    HIGH_CORRELATION = "high_correlation"
    CRISIS_CORRELATION = "crisis_correlation"


@dataclass
class MarketContext:
    """Complete market context for filtering"""
    timestamp: datetime
    session: MarketSession
    volatility_regime: VolatilityRegime
    correlation_regime: CorrelationRegime
    market_stress_level: float  # 0-1 scale
    sector_rotation_intensity: float  # 0-1 scale
    liquidity_conditions: float  # 0-1 scale (1 = high liquidity)
    news_sentiment: float  # -1 to 1 scale
    macro_uncertainty: float  # 0-1 scale
    options_flow_sentiment: float  # -1 to 1 scale


@dataclass
class FilteringRules:
    """Filtering rules configuration"""
    session_filters: Dict[MarketSession, float]  # Threshold multipliers
    volatility_filters: Dict[VolatilityRegime, float]
    correlation_filters: Dict[CorrelationRegime, float]
    stress_threshold: float
    sentiment_threshold: float
    liquidity_threshold: float
    min_confidence_override: float  # Always pass if above this confidence


@dataclass
class FilterResult:
    """Result of context filtering"""
    passed: bool
    confidence_adjustment: float  # Multiplier for alert confidence
    suppression_reason: Optional[str]
    context_score: float  # Overall context favorability 0-1
    recommended_delay: int  # Suggested delay in minutes before re-evaluation


class ContextAwareFilter:
    """
    Context-Aware Alert Filtering System

    Reduces false positives by analyzing market context and applying
    intelligent filtering rules based on market conditions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Context-Aware Filter

        Args:
            config: Configuration dictionary for filter parameters
        """
        self.config = config or {}

        # Default filtering rules
        self.filtering_rules = FilteringRules(
            session_filters={
                MarketSession.REGULAR_HOURS: 1.0,
                MarketSession.PRE_MARKET: 0.8,
                MarketSession.AFTER_HOURS: 0.7,
                MarketSession.WEEKEND: 0.4,
                MarketSession.HOLIDAY: 0.3
            },
            volatility_filters={
                VolatilityRegime.LOW_VOL: 0.6,      # Harder to trigger in low vol
                VolatilityRegime.NORMAL_VOL: 1.0,   # Normal sensitivity
                VolatilityRegime.HIGH_VOL: 1.2,     # More sensitive in high vol
                VolatilityRegime.EXTREME_VOL: 1.4   # Very sensitive in extreme vol
            },
            correlation_filters={
                CorrelationRegime.LOW_CORRELATION: 1.0,
                CorrelationRegime.NORMAL_CORRELATION: 0.9,
                CorrelationRegime.HIGH_CORRELATION: 0.7,  # Suppress in high correlation
                CorrelationRegime.CRISIS_CORRELATION: 1.3  # More sensitive in crisis
            },
            stress_threshold=0.7,
            sentiment_threshold=0.8,
            liquidity_threshold=0.3,
            min_confidence_override=0.95
        )

        # Historical performance tracking
        self.filter_performance = {
            'total_alerts': 0,
            'passed_filters': 0,
            'false_positives': 0,
            'true_positives': 0,
            'suppression_accuracy': 0.0
        }

        # Market hours (US Eastern Time)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET

        logger.info("Context-Aware Filter initialized")

    def apply_filter(self,
                    alert_confidence: float,
                    alert_type: str,
                    symbol: str,
                    market_data: Dict[str, Any],
                    portfolio_context: Dict[str, Any] = None) -> FilterResult:
        """
        Apply context-aware filtering to an alert

        Args:
            alert_confidence: Original alert confidence (0-1)
            alert_type: Type of alert being filtered
            symbol: Trading symbol
            market_data: Current market data
            portfolio_context: Portfolio-wide context

        Returns:
            FilterResult with filtering decision and adjustments
        """
        try:
            # Build market context
            context = self._build_market_context(market_data, portfolio_context)

            # Calculate context score
            context_score = self._calculate_context_score(context)

            # Apply filtering rules
            filter_result = self._apply_filtering_rules(
                alert_confidence, alert_type, symbol, context
            )

            # Override for high-confidence alerts
            if alert_confidence >= self.filtering_rules.min_confidence_override:
                filter_result.passed = True
                filter_result.suppression_reason = None

            # Update performance tracking
            self._update_performance_tracking(filter_result)

            logger.debug(f"Filter result for {symbol}: passed={filter_result.passed}, "
                        f"context_score={context_score:.3f}")

            return filter_result

        except Exception as e:
            logger.error(f"Context filtering failed for {symbol}: {e}")
            # Fail-safe: pass the alert with reduced confidence
            return FilterResult(
                passed=True,
                confidence_adjustment=0.8,
                suppression_reason=None,
                context_score=0.5,
                recommended_delay=0
            )

    def _build_market_context(self,
                             market_data: Dict[str, Any],
                             portfolio_context: Dict[str, Any] = None) -> MarketContext:
        """Build comprehensive market context"""
        try:
            now = datetime.now()

            # Determine market session
            session = self._determine_market_session(now)

            # Assess volatility regime
            volatility_regime = self._assess_volatility_regime(market_data)

            # Assess correlation regime
            correlation_regime = self._assess_correlation_regime(market_data, portfolio_context)

            # Calculate market stress level
            stress_level = self._calculate_market_stress(market_data)

            # Calculate sector rotation intensity
            sector_rotation = self._calculate_sector_rotation(market_data, portfolio_context)

            # Assess liquidity conditions
            liquidity = self._assess_liquidity_conditions(market_data)

            # Get news sentiment (placeholder - would integrate with news API)
            news_sentiment = self._get_news_sentiment(market_data)

            # Calculate macro uncertainty
            macro_uncertainty = self._calculate_macro_uncertainty(market_data)

            # Get options flow sentiment (placeholder)
            options_sentiment = self._get_options_sentiment(market_data)

            return MarketContext(
                timestamp=now,
                session=session,
                volatility_regime=volatility_regime,
                correlation_regime=correlation_regime,
                market_stress_level=stress_level,
                sector_rotation_intensity=sector_rotation,
                liquidity_conditions=liquidity,
                news_sentiment=news_sentiment,
                macro_uncertainty=macro_uncertainty,
                options_flow_sentiment=options_sentiment
            )

        except Exception as e:
            logger.error(f"Market context building failed: {e}")
            return self._default_market_context()

    def _determine_market_session(self, timestamp: datetime) -> MarketSession:
        """Determine current market session"""
        try:
            # Check if weekend
            if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return MarketSession.WEEKEND

            current_time = timestamp.time()

            # Check if regular trading hours (9:30 AM - 4:00 PM ET)
            if self.market_open <= current_time <= self.market_close:
                return MarketSession.REGULAR_HOURS
            elif time(4, 0) < current_time <= time(20, 0):  # 4 PM - 8 PM ET
                return MarketSession.AFTER_HOURS
            elif time(4, 0) <= current_time < self.market_open:  # 4 AM - 9:30 AM ET
                return MarketSession.PRE_MARKET
            else:
                return MarketSession.AFTER_HOURS

        except Exception as e:
            logger.error(f"Session determination failed: {e}")
            return MarketSession.REGULAR_HOURS

    def _assess_volatility_regime(self, market_data: Dict[str, Any]) -> VolatilityRegime:
        """Assess current volatility regime"""
        try:
            # Use VIX if available, otherwise calculate from returns
            if 'vix' in market_data:
                vix = market_data['vix']
                if vix < 15:
                    return VolatilityRegime.LOW_VOL
                elif vix < 25:
                    return VolatilityRegime.NORMAL_VOL
                elif vix < 35:
                    return VolatilityRegime.HIGH_VOL
                else:
                    return VolatilityRegime.EXTREME_VOL
            else:
                # Calculate from recent returns
                volatility = market_data.get('recent_volatility', 0.02)
                if volatility < 0.01:
                    return VolatilityRegime.LOW_VOL
                elif volatility < 0.03:
                    return VolatilityRegime.NORMAL_VOL
                elif volatility < 0.05:
                    return VolatilityRegime.HIGH_VOL
                else:
                    return VolatilityRegime.EXTREME_VOL

        except Exception as e:
            logger.error(f"Volatility regime assessment failed: {e}")
            return VolatilityRegime.NORMAL_VOL

    def _assess_correlation_regime(self,
                                  market_data: Dict[str, Any],
                                  portfolio_context: Dict[str, Any] = None) -> CorrelationRegime:
        """Assess current correlation regime"""
        try:
            # Use portfolio correlation if available
            if portfolio_context and 'average_correlation' in portfolio_context:
                avg_corr = portfolio_context['average_correlation']
                if avg_corr < 0.3:
                    return CorrelationRegime.LOW_CORRELATION
                elif avg_corr < 0.6:
                    return CorrelationRegime.NORMAL_CORRELATION
                elif avg_corr < 0.8:
                    return CorrelationRegime.HIGH_CORRELATION
                else:
                    return CorrelationRegime.CRISIS_CORRELATION
            else:
                # Default based on volatility regime
                vol_regime = self._assess_volatility_regime(market_data)
                if vol_regime == VolatilityRegime.EXTREME_VOL:
                    return CorrelationRegime.CRISIS_CORRELATION
                elif vol_regime == VolatilityRegime.HIGH_VOL:
                    return CorrelationRegime.HIGH_CORRELATION
                else:
                    return CorrelationRegime.NORMAL_CORRELATION

        except Exception as e:
            logger.error(f"Correlation regime assessment failed: {e}")
            return CorrelationRegime.NORMAL_CORRELATION

    def _calculate_market_stress(self, market_data: Dict[str, Any]) -> float:
        """Calculate market stress level (0-1 scale)"""
        try:
            stress_factors = []

            # Volatility factor
            volatility = market_data.get('recent_volatility', 0.02)
            vol_stress = min(1.0, volatility / 0.05)  # 5% vol = max stress
            stress_factors.append(vol_stress)

            # Volume factor (high volume = potential stress)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            vol_stress = min(1.0, max(0.0, (volume_ratio - 1) / 2))  # 3x volume = max stress
            stress_factors.append(vol_stress)

            # Spread factor (wider spreads = higher stress)
            spread_ratio = market_data.get('bid_ask_spread_ratio', 1.0)
            spread_stress = min(1.0, max(0.0, (spread_ratio - 1) / 1))  # 2x spread = max stress
            stress_factors.append(spread_stress)

            # Cross-asset factors
            if 'sector_dispersion' in market_data:
                sector_stress = min(1.0, market_data['sector_dispersion'] / 0.1)
                stress_factors.append(sector_stress)

            return np.mean(stress_factors) if stress_factors else 0.3

        except Exception as e:
            logger.error(f"Market stress calculation failed: {e}")
            return 0.3

    def _calculate_sector_rotation(self,
                                  market_data: Dict[str, Any],
                                  portfolio_context: Dict[str, Any] = None) -> float:
        """Calculate sector rotation intensity"""
        try:
            if portfolio_context and 'sector_performance_dispersion' in portfolio_context:
                dispersion = portfolio_context['sector_performance_dispersion']
                return min(1.0, dispersion / 0.05)  # 5% dispersion = max rotation
            else:
                # Default based on volatility
                volatility = market_data.get('recent_volatility', 0.02)
                return min(1.0, volatility / 0.04)  # Proxy from volatility

        except Exception as e:
            logger.error(f"Sector rotation calculation failed: {e}")
            return 0.3

    def _assess_liquidity_conditions(self, market_data: Dict[str, Any]) -> float:
        """Assess liquidity conditions (0 = illiquid, 1 = highly liquid)"""
        try:
            liquidity_factors = []

            # Volume factor
            volume_ratio = market_data.get('volume_ratio', 1.0)
            vol_liquidity = min(1.0, volume_ratio / 2.0)  # 2x volume = max liquidity
            liquidity_factors.append(vol_liquidity)

            # Spread factor (tighter spreads = higher liquidity)
            spread_ratio = market_data.get('bid_ask_spread_ratio', 1.0)
            spread_liquidity = max(0.0, 1.0 - ((spread_ratio - 1) / 1.0))  # Inverse relationship
            liquidity_factors.append(spread_liquidity)

            # Market depth factor
            market_depth = market_data.get('market_depth_ratio', 1.0)
            depth_liquidity = min(1.0, market_depth)
            liquidity_factors.append(depth_liquidity)

            return np.mean(liquidity_factors) if liquidity_factors else 0.7

        except Exception as e:
            logger.error(f"Liquidity assessment failed: {e}")
            return 0.7

    def _get_news_sentiment(self, market_data: Dict[str, Any]) -> float:
        """Get news sentiment (-1 to 1 scale)"""
        try:
            # Placeholder for news sentiment integration
            # In production, this would integrate with news sentiment APIs
            return market_data.get('news_sentiment', 0.0)

        except Exception as e:
            logger.error(f"News sentiment retrieval failed: {e}")
            return 0.0

    def _calculate_macro_uncertainty(self, market_data: Dict[str, Any]) -> float:
        """Calculate macroeconomic uncertainty level"""
        try:
            # Proxy using term structure and volatility measures
            uncertainty_factors = []

            # Volatility uncertainty
            volatility = market_data.get('recent_volatility', 0.02)
            vol_uncertainty = min(1.0, volatility / 0.04)
            uncertainty_factors.append(vol_uncertainty)

            # Economic indicators uncertainty (placeholder)
            econ_uncertainty = market_data.get('economic_uncertainty', 0.3)
            uncertainty_factors.append(econ_uncertainty)

            return np.mean(uncertainty_factors) if uncertainty_factors else 0.3

        except Exception as e:
            logger.error(f"Macro uncertainty calculation failed: {e}")
            return 0.3

    def _get_options_sentiment(self, market_data: Dict[str, Any]) -> float:
        """Get options flow sentiment"""
        try:
            # Placeholder for options flow analysis
            # Would integrate with options data feeds
            return market_data.get('options_sentiment', 0.0)

        except Exception as e:
            logger.error(f"Options sentiment retrieval failed: {e}")
            return 0.0

    def _calculate_context_score(self, context: MarketContext) -> float:
        """Calculate overall context favorability score for alerts"""
        try:
            favorability_factors = []

            # Session favorability
            session_scores = {
                MarketSession.REGULAR_HOURS: 1.0,
                MarketSession.PRE_MARKET: 0.8,
                MarketSession.AFTER_HOURS: 0.6,
                MarketSession.WEEKEND: 0.2,
                MarketSession.HOLIDAY: 0.1
            }
            favorability_factors.append(session_scores[context.session])

            # Volatility favorability (moderate vol is best for alerts)
            vol_scores = {
                VolatilityRegime.LOW_VOL: 0.6,      # Fewer meaningful alerts
                VolatilityRegime.NORMAL_VOL: 1.0,   # Optimal
                VolatilityRegime.HIGH_VOL: 0.9,     # Good for alerts
                VolatilityRegime.EXTREME_VOL: 0.7   # Too noisy
            }
            favorability_factors.append(vol_scores[context.volatility_regime])

            # Correlation favorability
            corr_scores = {
                CorrelationRegime.LOW_CORRELATION: 1.0,      # Best for individual alerts
                CorrelationRegime.NORMAL_CORRELATION: 0.8,
                CorrelationRegime.HIGH_CORRELATION: 0.5,     # Less meaningful
                CorrelationRegime.CRISIS_CORRELATION: 0.7    # Important but noisy
            }
            favorability_factors.append(corr_scores[context.correlation_regime])

            # Liquidity favorability
            favorability_factors.append(context.liquidity_conditions)

            # Stress level (inverse - high stress reduces favorability)
            favorability_factors.append(1.0 - context.market_stress_level)

            return np.mean(favorability_factors)

        except Exception as e:
            logger.error(f"Context score calculation failed: {e}")
            return 0.5

    def _apply_filtering_rules(self,
                              alert_confidence: float,
                              alert_type: str,
                              symbol: str,
                              context: MarketContext) -> FilterResult:
        """Apply filtering rules based on context"""
        try:
            # Start with base confidence
            adjusted_confidence = alert_confidence

            # Apply session filter
            session_multiplier = self.filtering_rules.session_filters[context.session]
            adjusted_confidence *= session_multiplier

            # Apply volatility filter
            vol_multiplier = self.filtering_rules.volatility_filters[context.volatility_regime]
            adjusted_confidence *= vol_multiplier

            # Apply correlation filter
            corr_multiplier = self.filtering_rules.correlation_filters[context.correlation_regime]
            adjusted_confidence *= corr_multiplier

            # Check suppression conditions
            suppression_reason = None
            passed = True

            # Market stress suppression
            if context.market_stress_level > self.filtering_rules.stress_threshold:
                if adjusted_confidence < 0.8:  # Only pass high confidence in stress
                    passed = False
                    suppression_reason = f"High market stress ({context.market_stress_level:.2f})"

            # Sentiment suppression for certain alert types
            if abs(context.news_sentiment) > self.filtering_rules.sentiment_threshold:
                if alert_type in ['momentum_reversal', 'regime_change']:
                    passed = False
                    suppression_reason = f"Extreme news sentiment ({context.news_sentiment:.2f})"

            # Liquidity suppression
            if context.liquidity_conditions < self.filtering_rules.liquidity_threshold:
                if alert_type in ['volume_anomaly', 'liquidity_crisis']:
                    adjusted_confidence *= 0.7  # Reduce but don't suppress
                else:
                    passed = False
                    suppression_reason = f"Poor liquidity conditions ({context.liquidity_conditions:.2f})"

            # Weekend/Holiday suppression for non-critical alerts
            if context.session in [MarketSession.WEEKEND, MarketSession.HOLIDAY]:
                if adjusted_confidence < 0.7:
                    passed = False
                    suppression_reason = f"Non-trading hours: {context.session.value}"

            # Calculate recommended delay if suppressed
            recommended_delay = 0
            if not passed:
                if context.session in [MarketSession.WEEKEND, MarketSession.HOLIDAY]:
                    recommended_delay = 60  # Check again in 1 hour
                elif context.market_stress_level > 0.8:
                    recommended_delay = 30  # Check again in 30 minutes
                else:
                    recommended_delay = 15  # Default 15 minute delay

            # Context score
            context_score = self._calculate_context_score(context)

            # Final confidence adjustment
            confidence_adjustment = adjusted_confidence / alert_confidence if alert_confidence > 0 else 1.0

            return FilterResult(
                passed=passed,
                confidence_adjustment=confidence_adjustment,
                suppression_reason=suppression_reason,
                context_score=context_score,
                recommended_delay=recommended_delay
            )

        except Exception as e:
            logger.error(f"Filtering rules application failed: {e}")
            return FilterResult(
                passed=True,
                confidence_adjustment=0.8,
                suppression_reason=None,
                context_score=0.5,
                recommended_delay=0
            )

    def _update_performance_tracking(self, result: FilterResult) -> None:
        """Update filter performance metrics"""
        try:
            self.filter_performance['total_alerts'] += 1
            if result.passed:
                self.filter_performance['passed_filters'] += 1

            # Calculate suppression accuracy (would need feedback data)
            if self.filter_performance['total_alerts'] > 0:
                pass_rate = self.filter_performance['passed_filters'] / self.filter_performance['total_alerts']
                self.filter_performance['suppression_accuracy'] = 1.0 - pass_rate

        except Exception as e:
            logger.error(f"Performance tracking update failed: {e}")

    def _default_market_context(self) -> MarketContext:
        """Return default market context when building fails"""
        return MarketContext(
            timestamp=datetime.now(),
            session=MarketSession.REGULAR_HOURS,
            volatility_regime=VolatilityRegime.NORMAL_VOL,
            correlation_regime=CorrelationRegime.NORMAL_CORRELATION,
            market_stress_level=0.3,
            sector_rotation_intensity=0.3,
            liquidity_conditions=0.7,
            news_sentiment=0.0,
            macro_uncertainty=0.3,
            options_flow_sentiment=0.0
        )

    def update_filtering_rules(self,
                              performance_feedback: Dict[str, Any]) -> bool:
        """Update filtering rules based on performance feedback"""
        try:
            # Analyze false positive/negative rates
            fp_rate = performance_feedback.get('false_positive_rate', 0.05)
            fn_rate = performance_feedback.get('false_negative_rate', 0.05)

            # Adjust thresholds based on feedback
            if fp_rate > 0.1:  # Too many false positives
                # Make filters stricter
                for session in self.filtering_rules.session_filters:
                    self.filtering_rules.session_filters[session] *= 0.95

                self.filtering_rules.stress_threshold *= 0.95
                self.filtering_rules.min_confidence_override += 0.02

            elif fn_rate > 0.1:  # Too many false negatives
                # Make filters more permissive
                for session in self.filtering_rules.session_filters:
                    self.filtering_rules.session_filters[session] *= 1.05

                self.filtering_rules.stress_threshold *= 1.05
                self.filtering_rules.min_confidence_override -= 0.02

            logger.info(f"Filtering rules updated based on feedback: FP={fp_rate:.3f}, FN={fn_rate:.3f}")
            return True

        except Exception as e:
            logger.error(f"Filtering rules update failed: {e}")
            return False

    def get_filter_status(self) -> Dict[str, Any]:
        """Get current filter status and performance"""
        return {
            'total_alerts_processed': self.filter_performance['total_alerts'],
            'pass_rate': (self.filter_performance['passed_filters'] /
                         max(1, self.filter_performance['total_alerts'])),
            'current_rules': {
                'session_filters': {k.value: v for k, v in self.filtering_rules.session_filters.items()},
                'stress_threshold': self.filtering_rules.stress_threshold,
                'sentiment_threshold': self.filtering_rules.sentiment_threshold,
                'liquidity_threshold': self.filtering_rules.liquidity_threshold,
                'min_confidence_override': self.filtering_rules.min_confidence_override
            },
            'performance_metrics': self.filter_performance
        }

    def analyze_suppression_patterns(self,
                                   alert_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in alert suppression"""
        try:
            if not alert_history:
                return {'analysis': 'No alert history available'}

            # Count suppressions by reason
            suppression_reasons = {}
            session_suppressions = {}
            time_patterns = {}

            for alert in alert_history:
                if alert.get('suppressed', False):
                    reason = alert.get('suppression_reason', 'unknown')
                    suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1

                    session = alert.get('market_session', 'unknown')
                    session_suppressions[session] = session_suppressions.get(session, 0) + 1

                    hour = alert.get('timestamp', datetime.now()).hour
                    time_patterns[hour] = time_patterns.get(hour, 0) + 1

            return {
                'suppression_reasons': suppression_reasons,
                'session_patterns': session_suppressions,
                'hourly_patterns': time_patterns,
                'total_alerts_analyzed': len(alert_history),
                'suppression_rate': len([a for a in alert_history if a.get('suppressed', False)]) / len(alert_history)
            }

        except Exception as e:
            logger.error(f"Suppression pattern analysis failed: {e}")
            return {'analysis': f'Analysis failed: {str(e)}'}