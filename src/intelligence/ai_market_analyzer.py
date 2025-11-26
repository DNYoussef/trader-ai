"""
AI Market Analyzer - Integrated System
Combines InequalityHunter, BarbellStrategy, and new AI calibration system
for comprehensive market analysis and opportunity detection.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np

# Import existing systems
from .inequality_hunter import InequalityHunter, ContrarianOpportunity
from .ai_calibration_engine import ai_calibration_engine
from .ai_signal_generator import ai_signal_generator, CohortData, MarketExpectation, AISignal
from ..strategies.barbell_strategy import BarbellStrategy
from ..gates.gate_manager import GateManager

logger = logging.getLogger(__name__)

@dataclass
class IntegratedMarketSignal:
    """Combined signal from all AI systems"""
    timestamp: datetime
    asset: str

    # Gary-style mathematical signals
    ai_signal: AISignal

    # Inequality analysis
    inequality_metrics: Dict[str, float]
    contrarian_opportunity: Optional[ContrarianOpportunity]

    # Barbell strategy compatibility
    barbell_allocation: Dict[str, float]  # safe vs risky allocation
    kelly_position_size: float

    # AI calibration insights
    ai_confidence: float
    calibrated_confidence: float
    expected_utility: float

    # Risk metrics (EVT-based)
    var_95: float
    var_99: float
    expected_shortfall: float
    antifragility_score: float

    # Decision recommendation
    action: str  # 'buy', 'sell', 'hold', 'gary_moment'
    position_size_pct: float
    time_horizon: str
    reasoning: str

class AIMarketAnalyzer:
    """
    Comprehensive AI market analysis system integrating:
    1. Mathematical framework (DPI, NG, RP calculations)
    2. Inequality analysis (InequalityHunter)
    3. Barbell strategy constraints
    4. AI self-calibration
    5. EVT-based risk management
    6. Matt Freeman decision theory
    """

    def __init__(self, portfolio_manager=None):
        # Initialize existing systems
        self.inequality_hunter = InequalityHunter()
        self.gate_manager = GateManager()

        # Initialize barbell strategy if portfolio manager provided
        self.barbell_strategy = None
        if portfolio_manager:
            self.barbell_strategy = BarbellStrategy(
                portfolio_manager=portfolio_manager,
                inequality_hunter=self.inequality_hunter
            )

        # Market watching parameters
        self.watch_list = [
            'SPY', 'QQQ', 'IWM',  # Equity indices
            'TLT', 'IEF', 'SHY',  # Treasury ETFs
            'GLD', 'SLV',         # Precious metals
            'XLE', 'XLF',         # Sector ETFs
            'VIX',                # Volatility
            'DXY'                 # Dollar index
        ]

        # Analysis history
        self.analysis_history: List[IntegratedMarketSignal] = []

        # Real-time monitoring state
        self.is_monitoring = False
        self.last_analysis_time = None

    def analyze_market_opportunity(self,
                                 asset: str,
                                 current_price: float,
                                 market_data: Dict[str, Any] = None) -> IntegratedMarketSignal:
        """
        Comprehensive analysis combining all AI systems and mathematical framework
        """
        logger.info(f"Starting comprehensive analysis for {asset}")

        # 1. Generate AI signal using mathematical framework
        ai_signal = self._generate_ai_signal(asset, current_price, market_data)

        # 2. Inequality analysis using existing system
        inequality_metrics = self._analyze_inequality_factors(asset, market_data)
        contrarian_opportunity = self._find_contrarian_opportunity(asset, inequality_metrics)

        # 3. Barbell strategy analysis
        barbell_allocation, kelly_size = self._analyze_barbell_fit(asset, ai_signal, contrarian_opportunity)

        # 4. AI calibration and confidence
        ai_confidence = ai_signal.ai_confidence
        calibrated_confidence = ai_calibration_engine.get_ai_confidence_adjustment(ai_confidence)
        expected_utility = self._calculate_expected_utility(ai_signal, kelly_size)

        # 5. Risk analysis using EVT framework
        risk_metrics = self._calculate_evt_risk_metrics(ai_signal, kelly_size)

        # 6. Generate final recommendation
        action, position_size, time_horizon, reasoning = self._generate_recommendation(
            ai_signal, contrarian_opportunity, barbell_allocation, calibrated_confidence, risk_metrics
        )

        # Create integrated signal
        integrated_signal = IntegratedMarketSignal(
            timestamp=datetime.now(),
            asset=asset,
            ai_signal=ai_signal,
            inequality_metrics=inequality_metrics,
            contrarian_opportunity=contrarian_opportunity,
            barbell_allocation=barbell_allocation,
            kelly_position_size=kelly_size,
            ai_confidence=ai_confidence,
            calibrated_confidence=calibrated_confidence,
            expected_utility=expected_utility,
            var_95=risk_metrics['var_95'],
            var_99=risk_metrics['var_99'],
            expected_shortfall=risk_metrics['expected_shortfall'],
            antifragility_score=risk_metrics['antifragility_score'],
            action=action,
            position_size_pct=position_size,
            time_horizon=time_horizon,
            reasoning=reasoning
        )

        # Store in history
        self.analysis_history.append(integrated_signal)

        # Update AI calibration with this analysis
        self._record_ai_prediction(integrated_signal)

        logger.info(f"Analysis complete for {asset}: {action} with {position_size:.1f}% allocation")

        return integrated_signal

    def start_market_monitoring(self):
        """Start continuous market monitoring for watch list"""
        self.is_monitoring = True
        logger.info("Started AI market monitoring")

    def stop_market_monitoring(self):
        """Stop market monitoring"""
        self.is_monitoring = False
        logger.info("Stopped AI market monitoring")

    def get_current_opportunities(self) -> List[IntegratedMarketSignal]:
        """Get current top opportunities across all systems"""
        if not self.analysis_history:
            return []

        # Get recent analyses (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_analyses = [
            signal for signal in self.analysis_history
            if signal.timestamp >= recent_cutoff
        ]

        # Sort by expected utility and confidence
        opportunities = sorted(
            recent_analyses,
            key=lambda x: x.expected_utility * x.calibrated_confidence,
            reverse=True
        )

        # Return top 10 opportunities
        return opportunities[:10]

    def _generate_ai_signal(self,
                          asset: str,
                          current_price: float,
                          market_data: Dict[str, Any] = None) -> AISignal:
        """Generate AI signal using mathematical framework"""

        # Create mock cohort data (in production, this would come from real data)
        cohort_data = self._create_cohort_data_from_market(market_data)

        # AI model expectation (in production, this would be from ML models)
        ai_expectation = self._generate_ai_market_expectation(asset, current_price, market_data)

        # Market expectations (from options, futures, etc.)
        market_expectations = self._extract_market_expectations(asset, market_data)

        # Catalyst events
        catalyst_events = self._identify_catalyst_events(asset, market_data)

        # Carry cost
        carry_cost = self._calculate_carry_cost(asset, market_data)

        return ai_signal_generator.generate_composite_signal(
            asset=asset,
            cohort_data=cohort_data,
            ai_model_expectation=ai_expectation,
            market_expectations=market_expectations,
            catalyst_events=catalyst_events,
            carry_cost=carry_cost
        )

    def _analyze_inequality_factors(self,
                                  asset: str,
                                  market_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Use existing inequality hunter to analyze factors"""

        # Get inequality metrics from existing system
        inequality_analysis = self.inequality_hunter.analyze_market_conditions()

        return {
            'gini_coefficient': inequality_analysis.get('gini_coefficient', 0.5),
            'wealth_concentration': inequality_analysis.get('wealth_concentration', 0.7),
            'cash_flow_pressure': inequality_analysis.get('cash_flow_pressure', 0.0),
            'policy_pressure': inequality_analysis.get('policy_pressure', 0.0)
        }

    def _find_contrarian_opportunity(self,
                                   asset: str,
                                   inequality_metrics: Dict[str, float]) -> Optional[ContrarianOpportunity]:
        """Find contrarian opportunities using existing system"""

        # Use existing inequality hunter to find opportunities
        opportunities = self.inequality_hunter.find_consensus_blindspots()

        # Find opportunity for this asset
        for opp in opportunities:
            if asset.lower() in opp.affected_assets:
                return opp

        return None

    def _analyze_barbell_fit(self,
                           asset: str,
                           ai_signal: AISignal,
                           contrarian_opportunity: Optional[ContrarianOpportunity]) -> Tuple[Dict[str, float], float]:
        """Analyze how asset fits into barbell strategy"""

        if not self.barbell_strategy:
            # Default allocation if no barbell strategy
            return {'safe': 0.8, 'risky': 0.2}, 0.05

        # Determine if asset is safe or risky
        is_safe_asset = self._classify_asset_safety(asset, ai_signal)

        # Get current barbell allocation
        allocation = self.barbell_strategy.get_current_allocation()

        # Calculate Kelly position size using AI's calibrated parameters
        kelly_size = ai_calibration_engine.calculate_ai_kelly_fraction(
            expected_return=ai_signal.ai_expected_return,
            variance=ai_signal.ai_risk_estimate ** 2
        )

        # Adjust for barbell constraints
        if is_safe_asset:
            max_allocation = allocation.get('safe_available', 0.8)
        else:
            max_allocation = allocation.get('risky_available', 0.2)

        kelly_size = min(kelly_size, max_allocation)

        return allocation, kelly_size

    def _calculate_expected_utility(self,
                                  ai_signal: AISignal,
                                  position_size: float) -> float:
        """Calculate expected utility using AI's calibrated utility function"""

        # Expected return from position
        expected_return = ai_signal.ai_expected_return * position_size

        # Calculate utility
        utility = ai_calibration_engine.calculate_ai_utility(expected_return)

        return utility

    def _calculate_evt_risk_metrics(self,
                                  ai_signal: AISignal,
                                  position_size: float) -> Dict[str, float]:
        """Calculate EVT-based risk metrics using mathematical framework"""

        # Mock EVT parameters (in production, these would be estimated from data)
        # POT/GPD parameters for tail estimation
        threshold_u = 0.02  # 2% threshold
        shape_xi = 0.1      # Shape parameter
        scale_beta = 0.01   # Scale parameter
        exceedance_prob = 0.05  # 5% exceedance probability

        # VaR calculations using formula:
        # VaR_q ≈ u + (β/ξ) * [((1-q)/p̂_u)^(-ξ) - 1]

        def calculate_var(confidence_level: float) -> float:
            q = confidence_level
            var_q = threshold_u + (scale_beta / shape_xi) * (
                ((1 - q) / exceedance_prob) ** (-shape_xi) - 1
            )
            return var_q * position_size

        var_95 = calculate_var(0.95)
        var_99 = calculate_var(0.99)

        # Expected Shortfall: ES_q ≈ VaR_q/(1-ξ) + (β-ξu)/(1-ξ)
        expected_shortfall = (var_99 / (1 - shape_xi) +
                            (scale_beta - shape_xi * threshold_u) / (1 - shape_xi))

        # Antifragility score: A = E[ΔP&L|tail shock] - λE[|θ|] - ρDrawdown
        antifragility_score = (
            max(0, ai_signal.ai_expected_return) * 2.0 -  # Tail upside
            0.1 * position_size -                          # Position cost (λ)
            0.2 * var_99                                   # Drawdown penalty (ρ)
        )

        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'antifragility_score': antifragility_score
        }

    def _generate_recommendation(self,
                               ai_signal: AISignal,
                               contrarian_opportunity: Optional[ContrarianOpportunity],
                               barbell_allocation: Dict[str, float],
                               calibrated_confidence: float,
                               risk_metrics: Dict[str, float]) -> Tuple[str, float, str, str]:
        """Generate final trading recommendation"""

        # Decision logic combining all factors
        action = 'hold'
        position_size = 0.0
        time_horizon = 'medium'
        reasoning_parts = []

        # High conviction Gary moment
        if (ai_signal.signal_strength == 'gary_moment' and
            calibrated_confidence > 0.8 and
            contrarian_opportunity and
            contrarian_opportunity.conviction_score > 0.8):

            action = 'gary_moment'
            position_size = min(0.15, ai_signal.ai_confidence * 0.2)  # Max 15% for Gary moments
            time_horizon = 'long'
            reasoning_parts.append("!!! GARY MOMENT: High conviction contrarian opportunity detected")

        # Strong buy signal
        elif (ai_signal.composite_signal > 0.5 and
              calibrated_confidence > 0.7 and
              risk_metrics['antifragility_score'] > 0):

            action = 'buy'
            position_size = min(0.1, ai_signal.kelly_position_size)
            time_horizon = 'medium'
            reasoning_parts.append("Strong AI signal with positive antifragility")

        # Moderate buy
        elif (ai_signal.composite_signal > 0.3 and
              calibrated_confidence > 0.6):

            action = 'buy'
            position_size = min(0.05, ai_signal.kelly_position_size * 0.5)
            time_horizon = 'short'
            reasoning_parts.append("Moderate AI signal with acceptable confidence")

        # Sell signal
        elif (ai_signal.composite_signal < -0.3 and
              calibrated_confidence > 0.6):

            action = 'sell'
            position_size = min(0.05, abs(ai_signal.kelly_position_size) * 0.5)
            time_horizon = 'short'
            reasoning_parts.append("Negative AI signal suggests downside")

        # Add risk considerations
        if risk_metrics['var_99'] > 0.1:
            reasoning_parts.append("High tail risk - reduced position size")
            position_size *= 0.5

        if contrarian_opportunity:
            reasoning_parts.append(f"Contrarian opportunity: {contrarian_opportunity.opportunity_type}")

        # Add AI calibration insights
        if calibrated_confidence < ai_signal.ai_confidence * 0.8:
            reasoning_parts.append("AI overconfidence detected - adjusted down")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No clear signal"

        return action, position_size, time_horizon, reasoning

    def _record_ai_prediction(self, signal: IntegratedMarketSignal):
        """Record AI prediction for later calibration"""

        # Convert action to probability
        if signal.action in ['buy', 'gary_moment']:
            prediction_prob = 0.7 + (signal.position_size_pct / 0.2) * 0.2  # 0.7-0.9 range
        elif signal.action == 'sell':
            prediction_prob = 0.3 - (signal.position_size_pct / 0.2) * 0.2  # 0.1-0.3 range
        else:
            prediction_prob = 0.5  # Neutral

        ai_calibration_engine.make_prediction(
            prediction_value=prediction_prob,
            confidence=signal.calibrated_confidence,
            context={
                'type': 'integrated_market_analysis',
                'asset': signal.asset,
                'action': signal.action,
                'reasoning': signal.reasoning
            }
        )

    # Helper methods for data generation (mock implementations)
    def _create_cohort_data_from_market(self, market_data: Dict[str, Any] = None) -> List[CohortData]:
        """Create cohort data from market conditions (mock implementation)"""
        return [
            CohortData(
                name="Ultra Wealthy",
                income_percentile=(99.0, 100.0),
                population_weight=0.01,
                net_cash_flow=1000000,  # Mock data
                historical_flows=[900000, 950000, 980000]
            ),
            CohortData(
                name="High Earners",
                income_percentile=(90.0, 99.0),
                population_weight=0.09,
                net_cash_flow=50000,
                historical_flows=[45000, 47000, 48000]
            ),
            CohortData(
                name="Middle Class",
                income_percentile=(20.0, 90.0),
                population_weight=0.70,
                net_cash_flow=-5000,  # Negative cash flow
                historical_flows=[-3000, -4000, -4500]
            ),
            CohortData(
                name="Lower Income",
                income_percentile=(0.0, 20.0),
                population_weight=0.20,
                net_cash_flow=-15000,
                historical_flows=[-12000, -13000, -14000]
            )
        ]

    def _generate_ai_market_expectation(self,
                                      asset: str,
                                      current_price: float,
                                      market_data: Dict[str, Any] = None) -> float:
        """Generate AI's market expectation (mock implementation)"""
        # In production, this would use ML models, fundamental analysis, etc.
        base_return = 0.08  # Base market return

        # Adjust based on asset type
        if asset in ['TLT', 'IEF', 'SHY']:
            return 0.03  # Lower returns for bonds
        elif asset in ['GLD', 'SLV']:
            return 0.05  # Moderate returns for precious metals
        elif asset == 'VIX':
            return -0.05  # VIX tends to decay
        else:
            return base_return + (hash(asset) % 100 - 50) / 1000  # Add some variation

    def _extract_market_expectations(self,
                                   asset: str,
                                   market_data: Dict[str, Any] = None) -> List[MarketExpectation]:
        """Extract market expectations from options, futures, etc. (mock implementation)"""
        return [
            MarketExpectation(
                asset=asset,
                timeframe="1Y",
                implied_probability=0.6,
                implied_return=0.06,  # Market expects 6% return
                confidence_interval=(0.02, 0.10),
                source="options"
            )
        ]

    def _identify_catalyst_events(self,
                                asset: str,
                                market_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Identify upcoming catalyst events (mock implementation)"""
        return [
            {
                'name': 'Fed Meeting',
                'days_until_event': 14,
                'importance': 0.8,
                'expected_impact': 'high_volatility'
            },
            {
                'name': 'Earnings Season',
                'days_until_event': 30,
                'importance': 0.6,
                'expected_impact': 'sector_rotation'
            }
        ]

    def _calculate_carry_cost(self,
                            asset: str,
                            market_data: Dict[str, Any] = None) -> float:
        """Calculate carry cost for holding the asset (mock implementation)"""
        # Simple carry cost model
        if asset in ['GLD', 'SLV']:
            return 0.02  # Storage costs for precious metals
        elif asset.startswith('TLT'):
            return -0.01  # Positive carry for bonds
        else:
            return 0.005  # Small carry cost for most assets

    def _classify_asset_safety(self, asset: str, ai_signal: AISignal) -> bool:
        """Classify if asset is safe or risky for barbell strategy"""
        safe_assets = ['TLT', 'IEF', 'SHY', 'CASH']
        return asset in safe_assets or ai_signal.ai_risk_estimate < 0.1

    def export_analysis_summary(self) -> Dict[str, Any]:
        """Export comprehensive analysis summary for UI"""
        current_opportunities = self.get_current_opportunities()

        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': self.is_monitoring,
            'ai_calibration': ai_calibration_engine.export_calibration_report(),
            'signal_weights': ai_signal_generator.get_current_signal_weights(),
            'top_opportunities': [
                {
                    'asset': opp.asset,
                    'action': opp.action,
                    'position_size': opp.position_size_pct,
                    'confidence': opp.calibrated_confidence,
                    'expected_utility': opp.expected_utility,
                    'reasoning': opp.reasoning,
                    'gary_moment': opp.action == 'gary_moment'
                }
                for opp in current_opportunities[:5]
            ],
            'risk_summary': {
                'total_var_99': sum(opp.var_99 for opp in current_opportunities),
                'avg_antifragility': np.mean([opp.antifragility_score for opp in current_opportunities]) if current_opportunities else 0,
                'portfolio_exposure': sum(opp.position_size_pct for opp in current_opportunities if opp.action in ['buy', 'gary_moment'])
            }
        }

# Global AI market analyzer instance
ai_market_analyzer = AIMarketAnalyzer()