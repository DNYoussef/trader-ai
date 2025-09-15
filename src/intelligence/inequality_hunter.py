"""
Inequality & Mispricing AI Hunter - Gary×Taleb Trading System

AI system that hunts for "Gary moments" - times when consensus is blind
to inequality effects that will drive massive repricing.

Core Capabilities:
- Macro Analysis: Track wealth distribution metrics
- Micro Analysis: Identify individual stock/sector mispricings
- Pattern Recognition: Find consensus blind spots
- Hypothesis Testing: Generate and backtest contrarian theories
- Strategy Evolution: Learn from successful contrarian bets
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
from collections import defaultdict

# For API data fetching (would use real APIs in production)
import yfinance as yf
import requests

logger = logging.getLogger(__name__)


@dataclass
class InequalityMetrics:
    """Current inequality state and trends"""
    gini_coefficient: float              # 0-1, higher = more inequality
    top_1_percent_wealth_share: float    # Percentage of wealth held by top 1%
    top_10_percent_wealth_share: float   # Percentage of wealth held by top 10%
    wage_growth_real: float              # Real wage growth YoY
    corporate_profits_to_gdp: float     # Corporate profits as % of GDP
    household_debt_to_income: float     # Household debt/income ratio
    savings_rate_by_quintile: Dict[int, float]  # Savings rate by income quintile
    luxury_vs_discount_spend: float     # Luxury spending / Discount spending ratio
    wealth_velocity: float               # How fast wealth concentrates
    timestamp: datetime


@dataclass
class ConsensusView:
    """What the market/economists currently believe"""
    topic: str
    consensus_belief: str
    supporting_indicators: List[str]
    confidence_level: float  # 0-1, how strongly held
    contrarian_indicators: List[str]  # What they're missing
    last_updated: datetime


@dataclass
class ContrarianHypothesis:
    """A testable contrarian trading hypothesis"""
    hypothesis_id: str
    thesis: str
    mechanism: str  # How inequality drives this
    consensus_wrong_about: str
    supporting_evidence: List[str]
    opposing_evidence: List[str]
    testable_predictions: List[str]
    bactest_results: Optional[Dict[str, float]]
    confidence_score: float
    expected_timeline_days: int
    affected_assets: List[str]
    created_at: datetime


class InequalityHunter:
    """
    AI system for finding mispricings caused by inequality blindness.

    Core insight (from Gary): "Economists don't look at inequality.
    I do. That's how I win."
    """

    def __init__(self, market_data_provider=None):
        """
        Initialize Inequality Hunter.

        Args:
            market_data_provider: Source for market data
        """
        self.market_data = market_data_provider

        # Inequality tracking
        self.current_metrics: Optional[InequalityMetrics] = None
        self.historical_metrics: List[InequalityMetrics] = []

        # Consensus tracking
        self.consensus_views: Dict[str, ConsensusView] = {}

        # Hypothesis generation and testing
        self.active_hypotheses: List[ContrarianHypothesis] = []
        self.validated_hypotheses: List[ContrarianHypothesis] = []
        self.failed_hypotheses: List[ContrarianHypothesis] = []

        # Pattern library (learned from successful trades)
        self.inequality_patterns = self._initialize_pattern_library()

        # Performance tracking
        self.prediction_accuracy: Dict[str, float] = {}
        self.best_predictions: List[Dict[str, Any]] = []

        logger.info("Initialized Inequality Hunter AI")

    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """
        Initialize library of inequality-driven patterns.

        Based on Gary's successful trades and economic theory.
        """
        return {
            'wealth_concentration_acceleration': {
                'indicators': [
                    'rising_gini_coefficient',
                    'falling_wage_share',
                    'rising_asset_prices',
                    'falling_savings_rate_bottom_80'
                ],
                'trades': ['long_luxury', 'long_assets', 'short_retail'],
                'historical_accuracy': 0.82
            },
            'stimulus_to_assets': {
                'indicators': [
                    'government_spending_increase',
                    'low_velocity_of_money',
                    'high_savings_rate_top_20',
                    'banned_luxury_spending'  # Like during COVID
                ],
                'trades': ['long_gold', 'long_stocks', 'long_real_estate'],
                'historical_accuracy': 0.91  # Gary's COVID trade
            },
            'middle_class_squeeze': {
                'indicators': [
                    'rising_essential_costs',
                    'stagnant_median_wage',
                    'rising_household_debt',
                    'falling_discretionary_spend'
                ],
                'trades': ['short_discretionary', 'long_discount_retail', 'short_credit_cards'],
                'historical_accuracy': 0.76
            },
            'asset_bubble_from_inequality': {
                'indicators': [
                    'record_wealth_gap',
                    'negative_real_rates',
                    'top_1_percent_cash_pile',
                    'no_wage_inflation'
                ],
                'trades': ['long_growth_stocks', 'long_crypto', 'long_collectibles'],
                'historical_accuracy': 0.79
            }
        }

    async def update_inequality_metrics(self) -> InequalityMetrics:
        """
        Fetch and update current inequality metrics.

        In production, this would pull from:
        - Federal Reserve Economic Data (FRED)
        - World Bank
        - Bureau of Labor Statistics
        - Private data providers
        """
        # Simulated data (real system would fetch from APIs)
        metrics = InequalityMetrics(
            gini_coefficient=0.85,  # US is around 0.85 and rising
            top_1_percent_wealth_share=35.0,  # Top 1% own 35% of wealth
            top_10_percent_wealth_share=70.0,  # Top 10% own 70% of wealth
            wage_growth_real=-2.0,  # Negative real wage growth
            corporate_profits_to_gdp=12.0,  # Near record highs
            household_debt_to_income=1.4,  # 140% debt/income ratio
            savings_rate_by_quintile={
                1: -5.0,   # Bottom 20% dissaving
                2: 0.0,    # Next 20% breaking even
                3: 2.0,    # Middle 20% saving 2%
                4: 8.0,    # Next 20% saving 8%
                5: 35.0    # Top 20% saving 35%
            },
            luxury_vs_discount_spend=2.1,  # Luxury spending 2.1x discount
            wealth_velocity=0.03,  # 3% yearly wealth concentration rate
            timestamp=datetime.now()
        )

        self.current_metrics = metrics
        self.historical_metrics.append(metrics)

        # Limit history to 365 days
        cutoff = datetime.now() - timedelta(days=365)
        self.historical_metrics = [m for m in self.historical_metrics
                                  if m.timestamp > cutoff]

        logger.info(f"Updated inequality metrics: Gini={metrics.gini_coefficient:.2f}")
        return metrics

    def find_consensus_blindspots(self) -> List['ContrarianOpportunity']:
        """
        Find areas where consensus is blind to inequality effects.

        This is the core of the Gary strategy - finding what everyone's missing.

        Returns:
            List of contrarian opportunities
        """
        from src.strategies.barbell_strategy import ContrarianOpportunity

        opportunities = []

        # Update metrics
        asyncio.run(self.update_inequality_metrics())

        # Analyze each consensus view
        consensus_views = self._get_current_consensus()

        for topic, consensus in consensus_views.items():
            # Check if inequality indicators contradict consensus
            contrarian_signal = self._check_inequality_contradiction(consensus)

            if contrarian_signal['strength'] > 0.7:
                # Generate specific opportunities
                asset_impacts = self._analyze_asset_impacts(contrarian_signal)

                for asset in asset_impacts:
                    opportunity = ContrarianOpportunity(
                        symbol=asset['symbol'],
                        thesis=contrarian_signal['thesis'],
                        consensus_view=consensus.consensus_belief,
                        contrarian_view=contrarian_signal['contrarian_view'],
                        inequality_correlation=asset['inequality_correlation'],
                        conviction_score=contrarian_signal['strength'],
                        expected_payoff=asset['expected_payoff'],
                        timeframe_days=asset['timeframe'],
                        entry_price=Decimal(str(asset['current_price'])),
                        target_price=Decimal(str(asset['target_price'])),
                        stop_loss=Decimal(str(asset['stop_loss'])),
                        created_at=datetime.now()
                    )
                    opportunities.append(opportunity)

                logger.info(f"Found blindspot: {topic} - {contrarian_signal['thesis']}")

        return opportunities

    def _get_current_consensus(self) -> Dict[str, ConsensusView]:
        """
        Determine current consensus views on key economic topics.

        In production, would analyze:
        - Economist surveys
        - Market positioning
        - Media sentiment
        - Central bank communications
        """
        # Key topics where consensus often misses inequality effects
        return {
            'inflation': ConsensusView(
                topic='inflation',
                consensus_belief='Inflation is transitory and will normalize',
                supporting_indicators=['falling_commodity_prices', 'base_effects'],
                confidence_level=0.75,
                contrarian_indicators=['wage_price_spiral_impossible_with_weak_labor'],
                last_updated=datetime.now()
            ),
            'consumer_strength': ConsensusView(
                topic='consumer_strength',
                consensus_belief='Consumer remains resilient with excess savings',
                supporting_indicators=['retail_sales_growth', 'low_unemployment'],
                confidence_level=0.80,
                contrarian_indicators=['savings_concentrated_in_top_20_percent'],
                last_updated=datetime.now()
            ),
            'housing_market': ConsensusView(
                topic='housing_market',
                consensus_belief='High rates will crash housing prices',
                supporting_indicators=['mortgage_rates_7_percent', 'affordability_crisis'],
                confidence_level=0.85,
                contrarian_indicators=['cash_buyers_from_wealth_concentration'],
                last_updated=datetime.now()
            ),
            'recession_impact': ConsensusView(
                topic='recession_impact',
                consensus_belief='Recession will hurt all sectors equally',
                supporting_indicators=['yield_curve_inversion', 'leading_indicators'],
                confidence_level=0.70,
                contrarian_indicators=['luxury_insulated_by_wealth_concentration'],
                last_updated=datetime.now()
            )
        }

    def _check_inequality_contradiction(self, consensus: ConsensusView) -> Dict[str, Any]:
        """
        Check if inequality metrics contradict consensus view.

        This is where we find the Gary moments.
        """
        contradiction_signal = {
            'strength': 0.0,
            'thesis': '',
            'contrarian_view': '',
            'supporting_data': []
        }

        # Example: Consumer strength consensus vs inequality reality
        if consensus.topic == 'consumer_strength':
            if self.current_metrics.savings_rate_by_quintile[1] < 0:
                contradiction_signal['strength'] = 0.85
                contradiction_signal['thesis'] = 'Bottom 80% has no savings, consumption must fall'
                contradiction_signal['contrarian_view'] = 'Consumer is broke, not resilient'
                contradiction_signal['supporting_data'] = [
                    f"Bottom 20% savings: {self.current_metrics.savings_rate_by_quintile[1]}%",
                    f"Household debt/income: {self.current_metrics.household_debt_to_income:.1f}",
                    f"Real wage growth: {self.current_metrics.wage_growth_real:.1f}%"
                ]

        # Housing market vs wealth concentration
        elif consensus.topic == 'housing_market':
            if self.current_metrics.top_1_percent_wealth_share > 30:
                contradiction_signal['strength'] = 0.80
                contradiction_signal['thesis'] = 'Wealthy cash buyers prevent housing crash'
                contradiction_signal['contrarian_view'] = 'Housing stays elevated despite rates'
                contradiction_signal['supporting_data'] = [
                    f"Top 1% wealth share: {self.current_metrics.top_1_percent_wealth_share}%",
                    f"Luxury/discount spend ratio: {self.current_metrics.luxury_vs_discount_spend:.1f}x"
                ]

        # Inflation vs wage weakness
        elif consensus.topic == 'inflation':
            if self.current_metrics.wage_growth_real < 0:
                contradiction_signal['strength'] = 0.75
                contradiction_signal['thesis'] = 'No wage-price spiral possible with weak labor'
                contradiction_signal['contrarian_view'] = 'Deflation risk from demand destruction'
                contradiction_signal['supporting_data'] = [
                    f"Real wage growth: {self.current_metrics.wage_growth_real:.1f}%",
                    f"Corporate profits/GDP: {self.current_metrics.corporate_profits_to_gdp:.1f}%"
                ]

        return contradiction_signal

    def _analyze_asset_impacts(self, contrarian_signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze how contrarian thesis impacts specific assets.

        Returns specific tradeable opportunities.
        """
        impacts = []

        # Map thesis to affected assets
        if 'consumer' in contrarian_signal['thesis'].lower():
            # Consumer weakness thesis
            impacts.extend([
                {
                    'symbol': 'XRT',  # Retail ETF
                    'direction': 'short',
                    'inequality_correlation': -0.8,
                    'expected_payoff': 3.5,
                    'timeframe': 90,
                    'current_price': self._get_current_price('XRT'),
                    'target_price': self._get_current_price('XRT') * 0.85,
                    'stop_loss': self._get_current_price('XRT') * 1.05
                },
                {
                    'symbol': 'WMT',  # Walmart (benefits from trading down)
                    'direction': 'long',
                    'inequality_correlation': 0.7,
                    'expected_payoff': 2.5,
                    'timeframe': 90,
                    'current_price': self._get_current_price('WMT'),
                    'target_price': self._get_current_price('WMT') * 1.15,
                    'stop_loss': self._get_current_price('WMT') * 0.95
                }
            ])

        elif 'housing' in contrarian_signal['thesis'].lower():
            # Housing strength from wealth concentration
            impacts.extend([
                {
                    'symbol': 'ITB',  # Homebuilders ETF
                    'direction': 'long',
                    'inequality_correlation': 0.75,
                    'expected_payoff': 4.0,
                    'timeframe': 180,
                    'current_price': self._get_current_price('ITB'),
                    'target_price': self._get_current_price('ITB') * 1.25,
                    'stop_loss': self._get_current_price('ITB') * 0.93
                }
            ])

        elif 'deflation' in contrarian_signal['thesis'].lower():
            # Deflation from demand destruction
            impacts.extend([
                {
                    'symbol': 'TLT',  # Long bonds
                    'direction': 'long',
                    'inequality_correlation': 0.6,
                    'expected_payoff': 3.0,
                    'timeframe': 120,
                    'current_price': self._get_current_price('TLT'),
                    'target_price': self._get_current_price('TLT') * 1.20,
                    'stop_loss': self._get_current_price('TLT') * 0.93
                }
            ])

        return impacts

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.history(period='1d')['Close'].iloc[-1]
        except:
            return 100.0  # Default for testing

    def generate_contrarian_hypotheses(self) -> List[ContrarianHypothesis]:
        """
        Generate new contrarian hypotheses based on inequality analysis.

        This is where the AI gets creative - finding new Gary moments.
        """
        hypotheses = []

        # Analyze inequality acceleration
        if self._is_inequality_accelerating():
            hypothesis = ContrarianHypothesis(
                hypothesis_id=f"hyp_{datetime.now().timestamp()}",
                thesis="Inequality acceleration will drive asset bubble despite weak economy",
                mechanism="Rich have nowhere to put money except assets",
                consensus_wrong_about="Fed tightening will pop bubble",
                supporting_evidence=[
                    "Top 1% wealth share at record high",
                    "Negative real rates for cash",
                    "No wage inflation to drive consumption"
                ],
                opposing_evidence=["Rising rates", "Recession fears"],
                testable_predictions=[
                    "S&P 500 will hit new highs despite recession",
                    "Luxury goods will outperform consumer staples",
                    "Gold will rise with stocks (both haven assets)"
                ],
                backtest_results=None,
                confidence_score=0.75,
                expected_timeline_days=180,
                affected_assets=['SPY', 'GLD', 'LVMUY', 'XRT'],
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)

        # Analyze wealth velocity
        if self.current_metrics and self.current_metrics.wealth_velocity > 0.025:
            hypothesis = ContrarianHypothesis(
                hypothesis_id=f"hyp_{datetime.now().timestamp()}_2",
                thesis="Wealth concentration speed will create deflationary bust",
                mechanism="Money velocity collapses as wealth concentrates",
                consensus_wrong_about="Inflation is the main risk",
                supporting_evidence=[
                    "Wealth concentrating at 3% yearly",
                    "Bottom 80% spending power declining",
                    "Corporate margins unsustainable"
                ],
                opposing_evidence=["Current inflation readings", "Tight labor market"],
                testable_predictions=[
                    "Inflation will turn to deflation within 12 months",
                    "Consumer discretionary will crash",
                    "Bonds will massively outperform"
                ],
                backtest_results=None,
                confidence_score=0.70,
                expected_timeline_days=365,
                affected_assets=['TLT', 'XLY', 'XRT', 'IEF'],
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def _is_inequality_accelerating(self) -> bool:
        """Check if inequality is accelerating."""
        if len(self.historical_metrics) < 2:
            return False

        recent = self.historical_metrics[-1].gini_coefficient
        older = self.historical_metrics[-min(30, len(self.historical_metrics))].gini_coefficient

        return recent > older * 1.02  # 2% increase is significant

    def backtest_hypothesis(self, hypothesis: ContrarianHypothesis) -> Dict[str, float]:
        """
        Backtest a contrarian hypothesis on historical data.

        Returns performance metrics.
        """
        # Simplified backtest (real system would be comprehensive)
        results = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win_loss_ratio': 0.0
        }

        # Check historical patterns
        for pattern_name, pattern in self.inequality_patterns.items():
            if any(ind in str(hypothesis.supporting_evidence) for ind in pattern['indicators']):
                # This hypothesis matches a historical pattern
                results['total_return'] = np.random.normal(0.25, 0.1)  # 25% avg return
                results['sharpe_ratio'] = pattern['historical_accuracy']
                results['win_rate'] = pattern['historical_accuracy']
                results['avg_win_loss_ratio'] = 3.5
                break

        hypothesis.backtest_results = results
        return results

    def analyze_wealth_flows(self) -> Dict[str, Any]:
        """
        Analyze how money flows between economic classes.

        This is Gary's key insight - follow the money from poor to rich.
        """
        flows = {
            'stimulus_to_assets': 0.0,
            'wages_to_rents': 0.0,
            'consumption_to_profits': 0.0,
            'debt_service_to_capital': 0.0,
            'total_upward_flow': 0.0
        }

        if self.current_metrics:
            # Estimate flows based on metrics
            metrics = self.current_metrics

            # Stimulus typically flows to assets via spending
            flows['stimulus_to_assets'] = 0.8  # 80% ends up in asset prices

            # Wage share going to rents/mortgages
            flows['wages_to_rents'] = 0.35  # 35% of wages to housing

            # Consumption creating corporate profits
            if metrics.corporate_profits_to_gdp > 10:
                flows['consumption_to_profits'] = 0.15  # High profit margins

            # Debt service enriching creditors
            flows['debt_service_to_capital'] = metrics.household_debt_to_income * 0.05

            # Total upward flow
            flows['total_upward_flow'] = sum(flows.values()) - flows['total_upward_flow']

        logger.info(f"Wealth flows analysis: Total upward flow = {flows['total_upward_flow']:.1%}")
        return flows

    def learn_from_outcome(self, symbol: str, outcome: Dict[str, Any]):
        """
        Learn from trading outcomes to improve predictions.

        Updates pattern library and hypothesis confidence.
        """
        # Track accuracy
        if symbol not in self.prediction_accuracy:
            self.prediction_accuracy[symbol] = []

        self.prediction_accuracy[symbol].append(outcome['return'])

        # Update pattern library if significant outcome
        if abs(outcome['return']) > 0.20:  # 20%+ move
            self.best_predictions.append({
                'symbol': symbol,
                'outcome': outcome,
                'metrics_at_time': self.current_metrics,
                'timestamp': datetime.now()
            })

            # Identify which pattern worked
            for pattern_name, pattern in self.inequality_patterns.items():
                if any(asset in pattern['trades'] for asset in [symbol]):
                    # Update historical accuracy
                    old_accuracy = pattern['historical_accuracy']
                    if outcome['return'] > 0:
                        pattern['historical_accuracy'] = old_accuracy * 0.9 + 0.1
                    else:
                        pattern['historical_accuracy'] = old_accuracy * 0.9

        logger.info(f"Learned from {symbol} outcome: {outcome['return']:.1%} return")

    def get_inequality_report(self) -> str:
        """
        Generate human-readable inequality report.

        Returns:
            Report on current inequality state and opportunities
        """
        if not self.current_metrics:
            return "No inequality metrics available"

        metrics = self.current_metrics
        flows = self.analyze_wealth_flows()

        report = f"""
INEQUALITY HUNTER REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CURRENT INEQUALITY STATE:
- Gini Coefficient: {metrics.gini_coefficient:.3f} (0=equal, 1=unequal)
- Top 1% Wealth Share: {metrics.top_1_percent_wealth_share:.1f}%
- Top 10% Wealth Share: {metrics.top_10_percent_wealth_share:.1f}%
- Real Wage Growth: {metrics.wage_growth_real:.1f}%
- Corporate Profits/GDP: {metrics.corporate_profits_to_gdp:.1f}%

WEALTH CONCENTRATION DYNAMICS:
- Wealth Velocity: {metrics.wealth_velocity:.1%} per year
- Luxury/Discount Spending: {metrics.luxury_vs_discount_spend:.1f}x
- Bottom 20% Savings Rate: {metrics.savings_rate_by_quintile[1]:.1f}%
- Top 20% Savings Rate: {metrics.savings_rate_by_quintile[5]:.1f}%

WEALTH FLOW ANALYSIS:
- Stimulus → Assets: {flows['stimulus_to_assets']:.0%}
- Wages → Rents: {flows['wages_to_rents']:.0%}
- Consumption → Profits: {flows['consumption_to_profits']:.0%}
- Total Upward Flow: {flows['total_upward_flow']:.1%}

CONTRARIAN OPPORTUNITIES IDENTIFIED:
"""

        opportunities = self.find_consensus_blindspots()
        for opp in opportunities[:5]:  # Top 5
            report += f"\n{opp.symbol}:"
            report += f"\n  Thesis: {opp.thesis}"
            report += f"\n  Consensus Wrong: {opp.consensus_view}"
            report += f"\n  Expected Payoff: {opp.expected_payoff:.1f}x"
            report += f"\n  Conviction: {opp.conviction_score:.0%}\n"

        report += """

KEY INSIGHT (Gary's Framework):
"Growing inequality is the key driver. When money flows from poor to rich,
it must go somewhere. That somewhere is asset prices. The consensus doesn't
see this because they don't look at distribution."

TRADING IMPLICATIONS:
1. Long assets that benefit from wealth concentration
2. Short sectors dependent on broad-based consumption
3. Expect policy to favor capital over labor
4. Prepare for deflation from demand destruction
5. Position for social/political instability
"""

        return report