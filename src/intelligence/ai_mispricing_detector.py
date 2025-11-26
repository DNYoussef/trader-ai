"""
AI Mispricing Detector - The Heart of Gary's Strategy
Finds assets where consensus is wrong about inequality effects using mathematical framework.
Manages dynamic 80/20 barbell allocation based on safety evolution.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np

# Import core systems
from .ai_calibration_engine import ai_calibration_engine
from .ai_signal_generator import ai_signal_generator
from .ai_data_stream_integration import ai_data_stream_integrator

logger = logging.getLogger(__name__)

@dataclass
class MispricingOpportunity:
    """Single mispricing where consensus is wrong about inequality"""
    asset: str
    mispricing_type: str  # 'inequality_blind', 'wealth_flow_miss', 'consensus_wrong'

    # Core Gary framework
    consensus_belief: str
    inequality_reality: str
    conviction_score: float  # How wrong is consensus (0-1)

    # Mathematical signals
    dpi_signal: float        # Distributional Pressure Index
    narrative_gap: float     # AI model vs market expectation
    repricing_potential: float  # RP calculation
    composite_signal: float     # Full mathematical framework

    # AI enhancement
    ai_confidence: float
    calibrated_confidence: float
    expected_utility: float
    ai_expected_return: float

    # Position details
    current_price: float
    target_price: float
    stop_loss: float
    position_size_pct: float
    kelly_fraction: float

    # Risk management
    var_95: float
    var_99: float
    antifragility_score: float
    time_horizon_days: int

    # Safety classification for barbell
    safety_score: float      # 0-1, where 1 = perfectly safe
    allocation_bucket: str   # 'risky_20', 'safe_80', 'transition'

    # Tracking fields
    discovered_at: datetime
    last_updated: datetime
    historical_accuracy: float

@dataclass
class BarbellAllocation:
    """Dynamic barbell allocation based on opportunity safety"""
    safe_allocation: float = 0.8
    risky_allocation: float = 0.2
    transition_allocation: float = 0.0

    safe_assets: List[str] = field(default_factory=list)
    risky_assets: List[str] = field(default_factory=list)
    transition_assets: List[str] = field(default_factory=list)

    last_rebalance: Optional[datetime] = None
    rebalance_reason: str = ""

class AIMispricingDetector:
    """
    AI system for finding consensus blind spots about inequality effects:

    1. Scans for assets where consensus ignores inequality
    2. Uses mathematical framework to validate mispricings
    3. Manages dynamic 80/20 barbell allocation
    4. Promotes assets from risky to safe as they become validated
    5. Continuous learning and calibration
    """

    def __init__(self):
        self.active_mispricings: Dict[str, MispricingOpportunity] = {}
        self.historical_mispricings: List[MispricingOpportunity] = []
        self.barbell_allocation = BarbellAllocation()

        # Mispricing detection parameters
        self.min_conviction_threshold = 0.6
        self.min_ai_confidence = 0.7
        self.safety_promotion_threshold = 0.85

        # Asset universe for scanning
        self.scan_universe = [
            # Equity indices
            'SPY', 'QQQ', 'IWM', 'VTI', 'VTEB',
            # Treasury ETFs
            'TLT', 'IEF', 'SHY', 'TIPS',
            # Precious metals
            'GLD', 'SLV', 'PDBC',
            # Sector ETFs
            'XLE', 'XLF', 'XLU', 'XLK', 'XLV',
            # International
            'EFA', 'EEM', 'VEA',
            # Real Estate
            'VNQ', 'REIT',
            # Volatility
            'VIX', 'UVXY'
        ]

        # Initialize safe assets (known low-risk)
        self.barbell_allocation.safe_assets = ['SHY', 'TLT', 'IEF', 'TIPS']

    async def scan_for_mispricings(self) -> List[MispricingOpportunity]:
        """
        Main function: Scan for assets where consensus is wrong about inequality
        """
        logger.info("Starting AI mispricing scan across asset universe")

        new_mispricings = []

        for asset in self.scan_universe:
            try:
                # Get current market data and AI analysis
                mispricing = await self._analyze_asset_mispricing(asset)

                if mispricing and self._validate_mispricing(mispricing):
                    new_mispricings.append(mispricing)

                    # Add to active tracking
                    self.active_mispricings[asset] = mispricing

                    logger.info(f"Detected mispricing in {asset}: {mispricing.mispricing_type} "
                              f"(conviction: {mispricing.conviction_score:.2f})")

            except Exception as e:
                logger.error(f"Error analyzing {asset}: {e}")

        # Update barbell allocation based on new mispricings
        await self._update_barbell_allocation()

        # Record AI predictions for calibration
        await self._record_mispricing_predictions(new_mispricings)

        return new_mispricings

    async def _analyze_asset_mispricing(self, asset: str) -> Optional[MispricingOpportunity]:
        """
        Analyze single asset for inequality-based mispricing using full mathematical framework
        """

        # Get AI-enhanced market data
        market_data = await self._get_enhanced_market_data(asset)
        if not market_data:
            return None

        # Generate AI signal using mathematical framework
        ai_signal = await self._generate_comprehensive_signal(asset, market_data)

        # Identify specific mispricing type
        mispricing_type, consensus_belief, reality = self._identify_mispricing_type(asset, ai_signal, market_data)

        if not mispricing_type:
            return None  # No significant mispricing detected

        # Calculate conviction score (how wrong is consensus)
        conviction_score = self._calculate_conviction_score(ai_signal, market_data)

        if conviction_score < self.min_conviction_threshold:
            return None  # Conviction too low

        # Get AI confidence with calibration
        ai_confidence = ai_signal.ai_confidence
        calibrated_confidence = ai_calibration_engine.get_ai_confidence_adjustment(ai_confidence)

        if calibrated_confidence < self.min_ai_confidence:
            return None  # AI not confident enough

        # Calculate position sizing using Kelly with AI parameters
        kelly_fraction = ai_calibration_engine.calculate_ai_kelly_fraction(
            expected_return=ai_signal.ai_expected_return,
            variance=ai_signal.ai_risk_estimate ** 2
        )

        # Calculate expected utility
        expected_utility = ai_calibration_engine.calculate_ai_utility(
            outcome=ai_signal.ai_expected_return * kelly_fraction
        )

        # Risk metrics using EVT framework
        risk_metrics = self._calculate_risk_metrics(ai_signal, kelly_fraction)

        # Safety score for barbell classification
        safety_score = self._calculate_safety_score(asset, ai_signal, risk_metrics)
        allocation_bucket = self._classify_allocation_bucket(safety_score)

        # Position sizing based on bucket
        position_size_pct = self._calculate_position_size(kelly_fraction, allocation_bucket, safety_score)

        # Price targets
        current_price = market_data.get('current_price', 100.0)
        target_price, stop_loss = self._calculate_price_targets(
            current_price, ai_signal.ai_expected_return, risk_metrics
        )

        mispricing = MispricingOpportunity(
            asset=asset,
            mispricing_type=mispricing_type,
            consensus_belief=consensus_belief,
            inequality_reality=reality,
            conviction_score=conviction_score,
            dpi_signal=ai_signal.dpi_component,
            narrative_gap=ai_signal.narrative_gap,
            repricing_potential=ai_signal.repricing_potential,
            composite_signal=ai_signal.composite_signal,
            ai_confidence=ai_confidence,
            calibrated_confidence=calibrated_confidence,
            expected_utility=expected_utility,
            ai_expected_return=ai_signal.ai_expected_return,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size_pct=position_size_pct,
            kelly_fraction=kelly_fraction,
            var_95=risk_metrics['var_95'],
            var_99=risk_metrics['var_99'],
            antifragility_score=risk_metrics['antifragility_score'],
            time_horizon_days=int(30 + conviction_score * 150),  # 30-180 days based on conviction
            safety_score=safety_score,
            allocation_bucket=allocation_bucket,
            discovered_at=datetime.now(),
            last_updated=datetime.now(),
            historical_accuracy=0.0  # Will be updated as we track outcomes
        )

        return mispricing

    def _identify_mispricing_type(self, asset: str, ai_signal, market_data: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
        """
        Identify specific type of inequality-based mispricing
        """

        # Treasury bond mispricing - wealth concentration effects
        if asset in ['TLT', 'IEF', 'TIPS']:
            if ai_signal.narrative_gap > 0.3:  # AI more bullish than market
                return (
                    'wealth_flow_miss',
                    'Rising rates will hurt bond prices due to Fed policy',
                    'Wealth concentration creates massive demand for safe assets regardless of rates'
                )

        # Equity index mispricing - inequality support
        elif asset in ['SPY', 'QQQ', 'IWM']:
            if ai_signal.dpi_component > 0.4:  # Strong wealth concentration pressure
                return (
                    'inequality_blind',
                    'Market weakness due to economic concerns',
                    'Concentrated wealth continues flowing into scarce assets (stocks)'
                )

        # Precious metals - inequality hedge
        elif asset in ['GLD', 'SLV']:
            if ai_signal.composite_signal > 0.5:
                return (
                    'consensus_wrong',
                    'Gold is outdated, crypto/stocks are better inflation hedges',
                    'Wealth inequality drives real asset demand - gold benefits from concentration'
                )

        # Volatility mispricing
        elif asset in ['VIX', 'UVXY']:
            if ai_signal.repricing_potential > 0.3:
                return (
                    'inequality_blind',
                    'Volatility will remain low due to central bank support',
                    'Inequality creates systemic instability that markets underestimate'
                )

        # Sector rotation mispricing
        elif asset == 'XLF':  # Financials
            if ai_signal.narrative_gap < -0.3:  # AI more bearish
                return (
                    'wealth_flow_miss',
                    'Banks benefit from rising rates',
                    'Wealth concentration reduces lending demand and increases defaults'
                )

        elif asset == 'XLE':  # Energy
            if ai_signal.composite_signal > 0.4:
                return (
                    'inequality_blind',
                    'ESG transition will hurt energy companies',
                    'Wealthy can afford transition costs; poor still need cheap energy'
                )

        # Real estate mispricing
        elif asset in ['VNQ', 'REIT']:
            if ai_signal.dpi_component > 0.5:
                return (
                    'consensus_wrong',
                    'Higher rates will crash real estate',
                    'Cash-rich investors (inequality beneficiaries) continue buying regardless of rates'
                )

        # International mispricing
        elif asset in ['EFA', 'EEM']:
            if ai_signal.narrative_gap > 0.2:
                return (
                    'inequality_blind',
                    'US inequality is unique problem',
                    'Wealth concentration is global phenomenon affecting all markets similarly'
                )

        return None, "", ""

    def _calculate_conviction_score(self, ai_signal, market_data: Dict[str, Any]) -> float:
        """
        Calculate how wrong consensus is (Gary's key insight)
        """

        # Base conviction from composite signal strength
        base_conviction = min(1.0, abs(ai_signal.composite_signal))

        # Boost conviction if multiple signals align
        signal_alignment = 0.0
        total_signals = 0

        if abs(ai_signal.dpi_component) > 0.2:
            signal_alignment += 1.0 if ai_signal.dpi_component * ai_signal.composite_signal > 0 else -0.5
            total_signals += 1

        if abs(ai_signal.narrative_gap) > 0.2:
            signal_alignment += 1.0 if ai_signal.narrative_gap * ai_signal.composite_signal > 0 else -0.5
            total_signals += 1

        if ai_signal.repricing_potential > 0.1:
            signal_alignment += 1.0
            total_signals += 1

        # Normalize alignment
        if total_signals > 0:
            alignment_boost = (signal_alignment / total_signals) * 0.2
        else:
            alignment_boost = 0.0

        # Time-based catalyst boost
        catalyst_boost = ai_signal.catalyst_timing * 0.1

        # Final conviction score
        conviction = min(1.0, base_conviction + alignment_boost + catalyst_boost)

        return conviction

    def _calculate_safety_score(self, asset: str, ai_signal, risk_metrics: Dict[str, Any]) -> float:
        """
        Calculate safety score for barbell allocation (0=risky, 1=safe)
        """

        # Base safety by asset class
        if asset in ['SHY', 'TIPS']:
            base_safety = 0.9  # Very safe
        elif asset in ['TLT', 'IEF']:
            base_safety = 0.8  # Mostly safe
        elif asset in ['SPY', 'VTI']:
            base_safety = 0.6  # Moderate
        elif asset in ['QQQ', 'XLK']:
            base_safety = 0.5  # Growth = riskier
        elif asset in ['GLD', 'SLV']:
            base_safety = 0.7  # Real assets = moderate safety
        else:
            base_safety = 0.4  # Default risky

        # Adjust for AI confidence (higher confidence = safer)
        confidence_adjustment = ai_signal.ai_confidence * 0.2

        # Adjust for risk metrics
        risk_adjustment = 0.0
        if risk_metrics['var_99'] < 0.05:  # Low tail risk
            risk_adjustment += 0.1
        if risk_metrics['antifragility_score'] > 0.1:  # Positive antifragility
            risk_adjustment += 0.1

        # Historical accuracy boost (assets that work become safer)
        # This would be updated as we track performance

        final_safety = min(1.0, base_safety + confidence_adjustment + risk_adjustment)

        return final_safety

    def _classify_allocation_bucket(self, safety_score: float) -> str:
        """
        Classify asset into barbell bucket based on safety
        """
        if safety_score >= self.safety_promotion_threshold:
            return 'safe_80'
        elif safety_score >= 0.6:
            return 'transition'  # Moving toward safe
        else:
            return 'risky_20'

    def _calculate_position_size(self, kelly_fraction: float, allocation_bucket: str, safety_score: float) -> float:
        """
        Calculate position size based on Kelly and barbell constraints
        """

        # Base Kelly position
        base_position = kelly_fraction

        # Apply barbell constraints
        if allocation_bucket == 'safe_80':
            # Can use up to 80% of portfolio, but limit individual positions
            max_position = 0.20  # Max 20% in any single safe asset
            multiplier = 1.0
        elif allocation_bucket == 'transition':
            # Moderate sizing
            max_position = 0.10  # Max 10% in transition assets
            multiplier = 0.8
        else:  # risky_20
            # Limited to 20% total, be more conservative per position
            max_position = 0.05  # Max 5% in any risky asset
            multiplier = 0.6

        # Apply safety score weighting
        safety_multiplier = 0.5 + (safety_score * 0.5)  # 0.5x to 1.0x based on safety

        position_size = min(max_position, base_position * multiplier * safety_multiplier)

        return max(0.01, position_size)  # Minimum 1% position

    async def _update_barbell_allocation(self):
        """
        Update barbell allocation based on current mispricings and safety evolution
        """

        # Categorize current mispricings by bucket
        safe_assets = []
        risky_assets = []
        transition_assets = []

        for asset, mispricing in self.active_mispricings.items():
            if mispricing.allocation_bucket == 'safe_80':
                safe_assets.append(asset)
            elif mispricing.allocation_bucket == 'risky_20':
                risky_assets.append(asset)
            else:
                transition_assets.append(asset)

        # Check if any assets should be promoted to safe
        promotions = []
        for asset in transition_assets + risky_assets:
            mispricing = self.active_mispricings[asset]

            # Promote if safety score improved significantly
            if mispricing.safety_score >= self.safety_promotion_threshold:
                promotions.append(asset)
                safe_assets.append(asset)

                # Update the mispricing record
                mispricing.allocation_bucket = 'safe_80'
                mispricing.last_updated = datetime.now()

        # Update barbell allocation
        total_safe_weight = sum(
            self.active_mispricings[asset].position_size_pct
            for asset in safe_assets
        )
        total_risky_weight = sum(
            self.active_mispricings[asset].position_size_pct
            for asset in risky_assets
        )

        # Ensure barbell constraints
        if total_safe_weight > 0.8:
            # Scale down safe positions to fit 80% limit
            scale_factor = 0.8 / total_safe_weight
            for asset in safe_assets:
                self.active_mispricings[asset].position_size_pct *= scale_factor

        if total_risky_weight > 0.2:
            # Scale down risky positions to fit 20% limit
            scale_factor = 0.2 / total_risky_weight
            for asset in risky_assets:
                self.active_mispricings[asset].position_size_pct *= scale_factor

        # Update allocation record
        self.barbell_allocation.safe_assets = safe_assets
        self.barbell_allocation.risky_assets = risky_assets
        self.barbell_allocation.transition_assets = transition_assets
        self.barbell_allocation.last_rebalance = datetime.now()

        if promotions:
            self.barbell_allocation.rebalance_reason = f"Promoted {promotions} to safe allocation"
            logger.info(f"Promoted assets to safe allocation: {promotions}")

    def _validate_mispricing(self, mispricing: MispricingOpportunity) -> bool:
        """
        Validate that mispricing meets all criteria
        """

        # Minimum thresholds
        if mispricing.conviction_score < self.min_conviction_threshold:
            return False

        if mispricing.calibrated_confidence < self.min_ai_confidence:
            return False

        # Risk limits
        if mispricing.var_99 > 0.15:  # Max 15% tail risk
            return False

        # Position size limits
        if mispricing.position_size_pct < 0.01 or mispricing.position_size_pct > 0.25:
            return False

        # Expected utility must be positive
        if mispricing.expected_utility <= 0:
            return False

        return True

    async def _get_enhanced_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Get AI-enhanced market data for analysis
        """

        # Mock market data (in production, this would fetch real data)
        base_price = 100.0 + hash(asset) % 50

        market_data = {
            'asset': asset,
            'current_price': base_price + np.random.normal(0, 2),
            'volume': 1000000 * (1 + np.random.random()),
            'bid_ask_spread': 0.01 + np.random.random() * 0.01,
            'implied_volatility': 0.15 + np.random.random() * 0.1,
            'option_skew': np.random.normal(0, 0.02),
            'market_cap': 1e9 * (1 + np.random.random() * 10),
            'analyst_consensus': np.random.choice(['buy', 'hold', 'sell']),
            'news_sentiment': np.random.normal(0, 0.3),
        }

        return market_data

    async def _generate_comprehensive_signal(self, asset: str, market_data: Dict[str, Any]):
        """
        Generate comprehensive AI signal using the mathematical framework
        """

        # Use existing AI signal generator
        from .ai_signal_generator import MarketExpectation

        # Create cohort data based on asset type
        cohort_data = self._create_asset_specific_cohorts(asset)

        # AI expectation based on inequality analysis
        ai_expectation = self._generate_ai_expectation(asset, market_data)

        # Market expectations
        market_expectations = [
            MarketExpectation(
                asset=asset,
                timeframe="1Y",
                implied_probability=0.6,
                implied_return=0.05 + np.random.normal(0, 0.02),
                confidence_interval=(0.02, 0.08),
                source="consensus"
            )
        ]

        # Catalyst events
        catalyst_events = [
            {
                'name': 'Fed Meeting',
                'days_until_event': 14 + np.random.randint(-7, 7),
                'importance': 0.8,
                'expected_impact': 'inequality_relevant'
            }
        ]

        # Generate signal
        signal = ai_signal_generator.generate_composite_signal(
            asset=asset,
            cohort_data=cohort_data,
            ai_model_expectation=ai_expectation,
            market_expectations=market_expectations,
            catalyst_events=catalyst_events,
            carry_cost=0.02
        )

        return signal

    def _create_asset_specific_cohorts(self, asset: str) -> List:
        """
        Create cohort data specific to asset type
        """
        from .ai_signal_generator import CohortData

        # Base cohorts with asset-specific flows
        if asset in ['TLT', 'IEF', 'SHY']:  # Treasuries - safe haven flows
            multiplier = 2.0  # Wealthy buy more bonds
        elif asset in ['SPY', 'QQQ']:  # Equities - asset inflation
            multiplier = 1.5
        elif asset in ['GLD', 'SLV']:  # Precious metals - hedge flows
            multiplier = 1.2
        else:
            multiplier = 1.0

        return [
            CohortData(
                name="Ultra Wealthy",
                income_percentile=(99.0, 100.0),
                population_weight=0.01,
                net_cash_flow=1000000 * multiplier,
                historical_flows=[900000, 950000, 980000]
            ),
            CohortData(
                name="High Earners",
                income_percentile=(90.0, 99.0),
                population_weight=0.09,
                net_cash_flow=50000 * multiplier,
                historical_flows=[45000, 47000, 48000]
            ),
            CohortData(
                name="Middle Class",
                income_percentile=(20.0, 90.0),
                population_weight=0.70,
                net_cash_flow=-5000,  # Negative regardless of asset
                historical_flows=[-3000, -4000, -4500]
            ),
            CohortData(
                name="Lower Income",
                income_percentile=(0.0, 20.0),
                population_weight=0.20,
                net_cash_flow=-15000,  # Negative regardless of asset
                historical_flows=[-12000, -13000, -14000]
            )
        ]

    def _generate_ai_expectation(self, asset: str, market_data: Dict[str, Any]) -> float:
        """
        Generate AI's expectation based on inequality analysis
        """

        # Get current inequality metrics
        inequality_metrics = ai_data_stream_integrator.get_ai_enhanced_inequality_metrics()

        gini = inequality_metrics.get('giniCoefficient', 0.48)
        wealth_concentration = inequality_metrics.get('top1PercentWealth', 32.0)

        # Asset-specific inequality impact
        if asset in ['TLT', 'IEF', 'SHY']:
            # Bonds benefit from wealth concentration (safe haven demand)
            return 0.03 + (gini - 0.4) * 0.2 + (wealth_concentration - 30) * 0.01

        elif asset in ['SPY', 'QQQ', 'VTI']:
            # Stocks benefit from asset inflation due to inequality
            return 0.08 + (gini - 0.4) * 0.3 + (wealth_concentration - 30) * 0.015

        elif asset in ['GLD', 'SLV']:
            # Precious metals as inequality hedge
            return 0.05 + (gini - 0.4) * 0.25 + (wealth_concentration - 30) * 0.012

        elif asset == 'VIX':
            # Volatility increases with inequality instability
            return -0.02 + (gini - 0.4) * 0.1  # Usually decays, but inequality adds risk

        else:
            # Default expectation
            return 0.06 + (gini - 0.4) * 0.15

    def _calculate_risk_metrics(self, ai_signal, kelly_fraction: float) -> Dict[str, float]:
        """
        Calculate EVT-based risk metrics
        """

        # Mock EVT parameters (in production, estimate from data)
        threshold_u = 0.02
        shape_xi = 0.1
        scale_beta = 0.01
        exceedance_prob = 0.05

        # VaR calculations
        var_95 = threshold_u + (scale_beta / shape_xi) * (
            ((1 - 0.95) / exceedance_prob) ** (-shape_xi) - 1
        ) * kelly_fraction

        var_99 = threshold_u + (scale_beta / shape_xi) * (
            ((1 - 0.99) / exceedance_prob) ** (-shape_xi) - 1
        ) * kelly_fraction

        # Expected shortfall
        expected_shortfall = (var_99 / (1 - shape_xi) +
                            (scale_beta - shape_xi * threshold_u) / (1 - shape_xi))

        # Antifragility score
        antifragility_score = (
            max(0, ai_signal.ai_expected_return) * 2.0 -  # Tail upside
            0.1 * kelly_fraction -                         # Position cost
            0.2 * var_99                                   # Drawdown penalty
        )

        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'antifragility_score': antifragility_score
        }

    def _calculate_price_targets(self, current_price: float, expected_return: float, risk_metrics: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate target price and stop loss
        """

        target_price = current_price * (1 + expected_return)

        # Stop loss based on VaR
        stop_loss = current_price * (1 - risk_metrics['var_95'] * 1.5)  # 1.5x VaR buffer

        return target_price, stop_loss

    async def _record_mispricing_predictions(self, mispricings: List[MispricingOpportunity]):
        """
        Record AI predictions for each mispricing for later calibration
        """

        for mispricing in mispricings:
            # Convert expected return to probability for calibration
            return_prob = min(1.0, max(0.0, (mispricing.ai_expected_return + 0.2) / 0.4))

            ai_calibration_engine.make_prediction(
                prediction_value=return_prob,
                confidence=mispricing.calibrated_confidence,
                context={
                    'type': 'mispricing_detection',
                    'asset': mispricing.asset,
                    'mispricing_type': mispricing.mispricing_type,
                    'conviction_score': mispricing.conviction_score,
                    'allocation_bucket': mispricing.allocation_bucket
                }
            )

    def get_current_mispricings_for_ui(self) -> List[Dict[str, Any]]:
        """
        Get current mispricings formatted for UI display
        """

        ui_mispricings = []

        for asset, mispricing in self.active_mispricings.items():
            ui_mispricing = {
                'id': f"mispricing_{asset}_{mispricing.discovered_at.strftime('%Y%m%d_%H%M%S')}",
                'symbol': asset,
                'thesis': f"{mispricing.mispricing_type}: {mispricing.inequality_reality}",
                'consensusView': mispricing.consensus_belief,
                'contrarianView': mispricing.inequality_reality,
                'inequalityCorrelation': mispricing.dpi_signal,
                'convictionScore': mispricing.conviction_score,
                'expectedPayoff': 1.0 + mispricing.ai_expected_return,
                'timeframeDays': mispricing.time_horizon_days,
                'entryPrice': mispricing.current_price,
                'targetPrice': mispricing.target_price,
                'stopLoss': mispricing.stop_loss,
                'currentPrice': mispricing.current_price,
                'historicalAccuracy': mispricing.historical_accuracy,
                'garyMomentScore': mispricing.conviction_score,
                'allocationBucket': mispricing.allocation_bucket,
                'safetyScore': mispricing.safety_score,
                'positionSize': mispricing.position_size_pct,
                'supportingData': [
                    {'metric': 'DPI Signal', 'value': mispricing.dpi_signal * 100, 'trend': 'up' if mispricing.dpi_signal > 0 else 'down'},
                    {'metric': 'Narrative Gap', 'value': abs(mispricing.narrative_gap) * 100, 'trend': 'up'},
                    {'metric': 'AI Confidence', 'value': mispricing.calibrated_confidence * 100, 'trend': 'up'},
                    {'metric': 'Repricing Potential', 'value': mispricing.repricing_potential * 100, 'trend': 'up'}
                ]
            }
            ui_mispricings.append(ui_mispricing)

        # Sort by conviction score
        ui_mispricings.sort(key=lambda x: x['convictionScore'], reverse=True)

        return ui_mispricings

    def get_barbell_allocation_status(self) -> Dict[str, Any]:
        """
        Get current barbell allocation status
        """

        return {
            'allocation_summary': {
                'safe_allocation': self.barbell_allocation.safe_allocation,
                'risky_allocation': self.barbell_allocation.risky_allocation,
                'transition_allocation': len(self.barbell_allocation.transition_assets) * 0.05
            },
            'safe_assets': self.barbell_allocation.safe_assets,
            'risky_assets': self.barbell_allocation.risky_assets,
            'transition_assets': self.barbell_allocation.transition_assets,
            'last_rebalance': self.barbell_allocation.last_rebalance.isoformat() if self.barbell_allocation.last_rebalance else None,
            'rebalance_reason': self.barbell_allocation.rebalance_reason,
            'total_mispricings': len(self.active_mispricings),
            'safety_promotions_available': len([
                m for m in self.active_mispricings.values()
                if m.safety_score >= self.safety_promotion_threshold and m.allocation_bucket != 'safe_80'
            ])
        }

# Global AI mispricing detector instance
ai_mispricing_detector = AIMispricingDetector()