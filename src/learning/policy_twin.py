"""
Policy Twin - Ethical Trading Framework and Advisory System

This module implements the Policy Twin component that provides advisory output
for ethical trading decisions. It calculates trades that would "erase alpha but
improve society" and tracks social impact metrics while maintaining profitable
trading operations.

Key Features:
- Ethical trading framework and guidelines
- Social impact metrics calculation
- Policy recommendations generation
- Transparency and accountability reporting
- Balance between profit and social responsibility
- Identification of exploitative vs constructive alpha
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AlphaType(Enum):
    CONSTRUCTIVE = "constructive"  # Creates value for society
    EXPLOITATIVE = "exploitative"  # Extracts value from market inefficiencies
    NEUTRAL = "neutral"  # No significant social impact

class SocialImpactCategory(Enum):
    MARKET_EFFICIENCY = "market_efficiency"
    PRICE_DISCOVERY = "price_discovery"
    LIQUIDITY_PROVISION = "liquidity_provision"
    INFORMATION_DIFFUSION = "information_diffusion"
    CAPITAL_ALLOCATION = "capital_allocation"
    MARKET_STABILITY = "market_stability"
    RETAIL_PROTECTION = "retail_protection"

@dataclass
class EthicalTrade:
    """Represents a trade with ethical considerations"""
    original_trade_id: str
    symbol: str
    action: str  # buy/sell
    quantity: float
    price: float
    strategy_id: str
    alpha_type: AlphaType
    social_impact_score: float  # -1 to 1 scale
    ethical_considerations: List[str] = field(default_factory=list)
    alternative_actions: List[str] = field(default_factory=list)
    transparency_level: float = 0.0  # 0 to 1 scale
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SocialImpactMetric:
    """Represents a social impact measurement"""
    category: SocialImpactCategory
    metric_name: str
    value: float
    description: str
    measurement_date: datetime
    confidence_level: float = 0.0

@dataclass
class PolicyRecommendation:
    """Represents a policy recommendation"""
    recommendation_id: str
    title: str
    description: str
    rationale: str
    implementation_difficulty: float  # 0 to 1 scale
    expected_impact: float  # -1 to 1 scale
    trade_ids_affected: List[str] = field(default_factory=list)
    timeline: str = ""
    stakeholders: List[str] = field(default_factory=list)

class EthicalFramework(ABC):
    """Abstract base class for ethical trading frameworks"""

    @abstractmethod
    def evaluate_trade_ethics(self, trade_data: Dict[str, Any]) -> Tuple[AlphaType, float]:
        """Evaluate the ethical implications of a trade"""
        pass

    @abstractmethod
    def suggest_alternatives(self, trade_data: Dict[str, Any]) -> List[str]:
        """Suggest alternative actions that improve social impact"""
        pass

class StakeholderWelfareFramework(EthicalFramework):
    """Framework based on stakeholder welfare optimization"""

    def __init__(self):
        self.stakeholder_weights = {
            "shareholders": 0.3,
            "customers": 0.25,
            "employees": 0.2,
            "communities": 0.15,
            "environment": 0.1
        }

    def evaluate_trade_ethics(self, trade_data: Dict[str, Any]) -> Tuple[AlphaType, float]:
        """Evaluate trade based on stakeholder welfare impact"""
        try:
            symbol = trade_data.get("symbol", "")
            strategy = trade_data.get("strategy_id", "")
            quantity = abs(trade_data.get("quantity", 0))

            # Analyze trade characteristics
            impact_scores = {}

            # Market efficiency impact
            if "arbitrage" in strategy.lower():
                impact_scores["market_efficiency"] = 0.8  # Arbitrage improves efficiency
            elif "momentum" in strategy.lower():
                impact_scores["market_efficiency"] = -0.2  # May increase volatility
            else:
                impact_scores["market_efficiency"] = 0.1

            # Liquidity provision impact
            if quantity > 10000:  # Large trade
                impact_scores["liquidity"] = -0.3  # May reduce liquidity
            else:
                impact_scores["liquidity"] = 0.2  # Provides liquidity

            # Information diffusion impact
            if "earnings" in strategy.lower() or "news" in strategy.lower():
                impact_scores["information"] = 0.6  # Helps price discovery
            else:
                impact_scores["information"] = 0.0

            # Market stability impact
            leverage = trade_data.get("leverage", 1.0)
            if leverage > 3.0:
                impact_scores["stability"] = -0.5  # High leverage destabilizes
            else:
                impact_scores["stability"] = 0.1

            # Calculate overall social impact
            overall_impact = np.mean(list(impact_scores.values()))

            # Determine alpha type
            if overall_impact > 0.3:
                alpha_type = AlphaType.CONSTRUCTIVE
            elif overall_impact < -0.3:
                alpha_type = AlphaType.EXPLOITATIVE
            else:
                alpha_type = AlphaType.NEUTRAL

            return alpha_type, overall_impact

        except Exception as e:
            logger.error(f"Error evaluating trade ethics: {e}")
            return AlphaType.NEUTRAL, 0.0

    def suggest_alternatives(self, trade_data: Dict[str, Any]) -> List[str]:
        """Suggest more ethical alternatives"""
        alternatives = []

        quantity = abs(trade_data.get("quantity", 0))
        strategy = trade_data.get("strategy_id", "")

        # Size-based alternatives
        if quantity > 10000:
            alternatives.append("Split large order into smaller blocks to reduce market impact")

        # Strategy-based alternatives
        if "momentum" in strategy.lower():
            alternatives.append("Use mean-reversion strategy to provide counter-cyclical liquidity")

        if "news" in strategy.lower():
            alternatives.append("Delay execution to allow information to diffuse naturally")

        # Timing alternatives
        alternatives.append("Execute during high-volume periods to minimize price impact")
        alternatives.append("Use TWAP/VWAP execution to reduce market disruption")

        # Disclosure alternatives
        alternatives.append("Increase position transparency through voluntary disclosure")

        return alternatives

class MarketEfficiencyFramework(EthicalFramework):
    """Framework focused on improving market efficiency"""

    def evaluate_trade_ethics(self, trade_data: Dict[str, Any]) -> Tuple[AlphaType, float]:
        """Evaluate based on contribution to market efficiency"""
        try:
            strategy = trade_data.get("strategy_id", "")
            price_impact = trade_data.get("expected_price_impact", 0.0)

            efficiency_score = 0.0

            # Arbitrage strategies improve efficiency
            if "arbitrage" in strategy.lower():
                efficiency_score += 0.8

            # Mean reversion provides stabilizing force
            if "mean_reversion" in strategy.lower():
                efficiency_score += 0.4

            # High-frequency strategies may harm efficiency
            if "hft" in strategy.lower() or "latency" in strategy.lower():
                efficiency_score -= 0.6

            # Large price impact reduces efficiency
            efficiency_score -= abs(price_impact) * 10

            # Determine alpha type
            if efficiency_score > 0.3:
                alpha_type = AlphaType.CONSTRUCTIVE
            elif efficiency_score < -0.3:
                alpha_type = AlphaType.EXPLOITATIVE
            else:
                alpha_type = AlphaType.NEUTRAL

            return alpha_type, efficiency_score

        except Exception as e:
            logger.error(f"Error evaluating market efficiency: {e}")
            return AlphaType.NEUTRAL, 0.0

    def suggest_alternatives(self, trade_data: Dict[str, Any]) -> List[str]:
        """Suggest efficiency-improving alternatives"""
        alternatives = [
            "Implement randomized execution timing to reduce predictability",
            "Use limit orders instead of market orders to provide liquidity",
            "Share non-material information to improve price discovery",
            "Adopt longer holding periods to reduce turnover costs"
        ]

        return alternatives

class PolicyTwin:
    """Main Policy Twin engine for ethical trading analysis"""

    def __init__(self):
        self.ethical_frameworks = [
            StakeholderWelfareFramework(),
            MarketEfficiencyFramework()
        ]
        self.social_impact_metrics: List[SocialImpactMetric] = []
        self.policy_recommendations: List[PolicyRecommendation] = []
        self.ethical_trades: List[EthicalTrade] = []

        # Configuration parameters
        self.transparency_threshold = 0.7
        self.social_impact_threshold = 0.5
        self.policy_update_frequency = timedelta(days=7)

    def analyze_trade_ethics(self, trade_data: Dict[str, Any]) -> EthicalTrade:
        """Analyze the ethical implications of a trade"""
        try:
            # Evaluate using all frameworks
            framework_results = []
            all_alternatives = []

            for framework in self.ethical_frameworks:
                alpha_type, impact_score = framework.evaluate_trade_ethics(trade_data)
                alternatives = framework.suggest_alternatives(trade_data)

                framework_results.append((alpha_type, impact_score))
                all_alternatives.extend(alternatives)

            # Aggregate results
            impact_scores = [result[1] for result in framework_results]
            avg_impact = np.mean(impact_scores)

            # Determine consensus alpha type
            alpha_types = [result[0] for result in framework_results]
            type_counts = {alpha_type: alpha_types.count(alpha_type) for alpha_type in AlphaType}
            consensus_type = max(type_counts, key=type_counts.get)

            # Generate ethical considerations
            ethical_considerations = self._generate_ethical_considerations(trade_data, avg_impact)

            # Calculate transparency level
            transparency_level = self._calculate_transparency_level(trade_data)

            ethical_trade = EthicalTrade(
                original_trade_id=trade_data.get("trade_id", ""),
                symbol=trade_data.get("symbol", ""),
                action=trade_data.get("action", ""),
                quantity=trade_data.get("quantity", 0),
                price=trade_data.get("price", 0),
                strategy_id=trade_data.get("strategy_id", ""),
                alpha_type=consensus_type,
                social_impact_score=avg_impact,
                ethical_considerations=ethical_considerations,
                alternative_actions=list(set(all_alternatives)),  # Remove duplicates
                transparency_level=transparency_level
            )

            self.ethical_trades.append(ethical_trade)
            logger.info(f"Analyzed trade ethics for {ethical_trade.original_trade_id}")

            return ethical_trade

        except Exception as e:
            logger.error(f"Error analyzing trade ethics: {e}")
            return self._create_neutral_ethical_trade(trade_data)

    def calculate_social_impact_metrics(self, time_period: timedelta = timedelta(days=30)) -> List[SocialImpactMetric]:
        """Calculate comprehensive social impact metrics"""
        try:
            cutoff_date = datetime.now() - time_period
            recent_trades = [t for t in self.ethical_trades if t.timestamp >= cutoff_date]

            if not recent_trades:
                return []

            metrics = []

            # Market Efficiency Metric
            efficiency_contributions = [
                t.social_impact_score for t in recent_trades
                if t.alpha_type in [AlphaType.CONSTRUCTIVE, AlphaType.EXPLOITATIVE]
            ]

            if efficiency_contributions:
                avg_efficiency_impact = np.mean(efficiency_contributions)
                metrics.append(SocialImpactMetric(
                    category=SocialImpactCategory.MARKET_EFFICIENCY,
                    metric_name="Average Market Efficiency Contribution",
                    value=avg_efficiency_impact,
                    description=f"Average contribution to market efficiency over {time_period.days} days",
                    measurement_date=datetime.now(),
                    confidence_level=min(1.0, len(efficiency_contributions) / 50.0)
                ))

            # Price Discovery Metric
            price_discovery_trades = [
                t for t in recent_trades
                if "arbitrage" in t.strategy_id.lower() or "earnings" in t.strategy_id.lower()
            ]

            if price_discovery_trades:
                price_discovery_impact = len(price_discovery_trades) / len(recent_trades)
                metrics.append(SocialImpactMetric(
                    category=SocialImpactCategory.PRICE_DISCOVERY,
                    metric_name="Price Discovery Participation Rate",
                    value=price_discovery_impact,
                    description="Percentage of trades contributing to price discovery",
                    measurement_date=datetime.now(),
                    confidence_level=0.8
                ))

            # Liquidity Provision Metric
            total_volume = sum(abs(t.quantity * t.price) for t in recent_trades)
            avg_trade_size = total_volume / len(recent_trades) if recent_trades else 0

            liquidity_score = min(1.0, total_volume / 10_000_000)  # Normalize to $10M
            metrics.append(SocialImpactMetric(
                category=SocialImpactCategory.LIQUIDITY_PROVISION,
                metric_name="Liquidity Provision Score",
                value=liquidity_score,
                description="Overall contribution to market liquidity",
                measurement_date=datetime.now(),
                confidence_level=0.9
            ))

            # Market Stability Metric
            constructive_trades = [t for t in recent_trades if t.alpha_type == AlphaType.CONSTRUCTIVE]
            exploitative_trades = [t for t in recent_trades if t.alpha_type == AlphaType.EXPLOITATIVE]

            stability_ratio = len(constructive_trades) / max(len(exploitative_trades), 1)
            stability_score = min(1.0, stability_ratio / 2.0)  # Normalize

            metrics.append(SocialImpactMetric(
                category=SocialImpactCategory.MARKET_STABILITY,
                metric_name="Market Stability Ratio",
                value=stability_score,
                description="Ratio of constructive to exploitative trades",
                measurement_date=datetime.now(),
                confidence_level=0.8
            ))

            # Transparency Metric
            avg_transparency = np.mean([t.transparency_level for t in recent_trades])
            metrics.append(SocialImpactMetric(
                category=SocialImpactCategory.RETAIL_PROTECTION,
                metric_name="Average Transparency Level",
                value=avg_transparency,
                description="Average level of trade transparency",
                measurement_date=datetime.now(),
                confidence_level=1.0
            ))

            self.social_impact_metrics.extend(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Error calculating social impact metrics: {e}")
            return []

    def generate_policy_recommendations(self) -> List[PolicyRecommendation]:
        """Generate policy recommendations based on ethical analysis"""
        try:
            recommendations = []

            # Analyze recent ethical trades
            recent_trades = [t for t in self.ethical_trades if
                           t.timestamp >= datetime.now() - timedelta(days=30)]

            if not recent_trades:
                return recommendations

            # Recommendation 1: Increase transparency for large trades
            large_trades = [t for t in recent_trades if abs(t.quantity * t.price) > 1_000_000]
            low_transparency_large_trades = [t for t in large_trades if t.transparency_level < 0.5]

            if len(low_transparency_large_trades) > len(large_trades) * 0.3:
                recommendations.append(PolicyRecommendation(
                    recommendation_id="TRANSPARENCY_001",
                    title="Increase Transparency for Large Trades",
                    description="Implement mandatory disclosure for trades exceeding $1M notional",
                    rationale=f"{len(low_transparency_large_trades)} out of {len(large_trades)} large trades had low transparency",
                    implementation_difficulty=0.4,
                    expected_impact=0.6,
                    trade_ids_affected=[t.original_trade_id for t in low_transparency_large_trades],
                    timeline="2-4 weeks",
                    stakeholders=["compliance", "trading", "legal"]
                ))

            # Recommendation 2: Reduce exploitative alpha strategies
            exploitative_trades = [t for t in recent_trades if t.alpha_type == AlphaType.EXPLOITATIVE]
            if len(exploitative_trades) > len(recent_trades) * 0.4:
                recommendations.append(PolicyRecommendation(
                    recommendation_id="STRATEGY_001",
                    title="Rebalance Towards Constructive Alpha",
                    description="Reduce allocation to exploitative strategies and increase constructive alpha generation",
                    rationale=f"{len(exploitative_trades)} exploitative trades out of {len(recent_trades)} total",
                    implementation_difficulty=0.7,
                    expected_impact=0.8,
                    trade_ids_affected=[t.original_trade_id for t in exploitative_trades],
                    timeline="1-3 months",
                    stakeholders=["portfolio_management", "strategy", "risk"]
                ))

            # Recommendation 3: Implement ethical scoring in position sizing
            low_social_impact_trades = [t for t in recent_trades if t.social_impact_score < -0.3]
            if len(low_social_impact_trades) > 5:
                recommendations.append(PolicyRecommendation(
                    recommendation_id="SIZING_001",
                    title="Integrate Social Impact in Position Sizing",
                    description="Reduce position sizes for trades with negative social impact scores",
                    rationale=f"{len(low_social_impact_trades)} trades with significant negative social impact",
                    implementation_difficulty=0.5,
                    expected_impact=0.5,
                    trade_ids_affected=[t.original_trade_id for t in low_social_impact_trades],
                    timeline="2-6 weeks",
                    stakeholders=["risk_management", "trading", "quantitative_research"]
                ))

            # Recommendation 4: Establish ethical trade monitoring
            if not hasattr(self, '_monitoring_established'):
                recommendations.append(PolicyRecommendation(
                    recommendation_id="MONITORING_001",
                    title="Establish Real-time Ethical Trade Monitoring",
                    description="Implement automated monitoring system for ethical trade evaluation",
                    rationale="Need systematic monitoring of ethical implications across all trades",
                    implementation_difficulty=0.8,
                    expected_impact=0.9,
                    timeline="2-4 months",
                    stakeholders=["technology", "compliance", "trading"]
                ))

            self.policy_recommendations.extend(recommendations)
            return recommendations

        except Exception as e:
            logger.error(f"Error generating policy recommendations: {e}")
            return []

    def create_transparency_report(self) -> Dict[str, Any]:
        """Create comprehensive transparency and accountability report"""
        try:
            recent_trades = [t for t in self.ethical_trades if
                           t.timestamp >= datetime.now() - timedelta(days=30)]

            # Overall statistics
            total_trades = len(recent_trades)
            constructive_trades = len([t for t in recent_trades if t.alpha_type == AlphaType.CONSTRUCTIVE])
            exploitative_trades = len([t for t in recent_trades if t.alpha_type == AlphaType.EXPLOITATIVE])
            neutral_trades = len([t for t in recent_trades if t.alpha_type == AlphaType.NEUTRAL])

            # Calculate average metrics
            avg_social_impact = np.mean([t.social_impact_score for t in recent_trades]) if recent_trades else 0
            avg_transparency = np.mean([t.transparency_level for t in recent_trades]) if recent_trades else 0

            # Volume analysis
            total_volume = sum(abs(t.quantity * t.price) for t in recent_trades)
            constructive_volume = sum(abs(t.quantity * t.price) for t in recent_trades
                                    if t.alpha_type == AlphaType.CONSTRUCTIVE)

            report = {
                "report_period": {
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "total_days": 30
                },
                "trade_summary": {
                    "total_trades": total_trades,
                    "constructive_trades": constructive_trades,
                    "exploitative_trades": exploitative_trades,
                    "neutral_trades": neutral_trades,
                    "constructive_percentage": (constructive_trades / total_trades * 100) if total_trades > 0 else 0
                },
                "impact_metrics": {
                    "average_social_impact_score": avg_social_impact,
                    "average_transparency_level": avg_transparency,
                    "total_volume_traded": total_volume,
                    "constructive_volume_percentage": (constructive_volume / total_volume * 100) if total_volume > 0 else 0
                },
                "social_impact_breakdown": self._create_impact_breakdown(recent_trades),
                "recent_recommendations": [
                    {
                        "id": rec.recommendation_id,
                        "title": rec.title,
                        "expected_impact": rec.expected_impact,
                        "implementation_status": "pending"  # Would track actual implementation
                    }
                    for rec in self.policy_recommendations[-5:]  # Last 5 recommendations
                ],
                "transparency_initiatives": [
                    "Voluntary position disclosure program",
                    "Regular social impact reporting",
                    "Ethical framework integration",
                    "Stakeholder engagement initiatives"
                ],
                "commitment_statement": (
                    "We are committed to generating alpha through constructive market participation "
                    "that enhances market efficiency, improves price discovery, and creates positive "
                    "social value while maintaining competitive returns."
                )
            }

            return report

        except Exception as e:
            logger.error(f"Error creating transparency report: {e}")
            return {"error": "Failed to generate transparency report"}

    def _generate_ethical_considerations(self, trade_data: Dict[str, Any], impact_score: float) -> List[str]:
        """Generate specific ethical considerations for a trade"""
        considerations = []

        # Impact-based considerations
        if impact_score < -0.5:
            considerations.append("Trade has significant negative social impact")
            considerations.append("Consider alternative execution methods")

        if impact_score > 0.5:
            considerations.append("Trade contributes positively to market efficiency")

        # Size-based considerations
        quantity = abs(trade_data.get("quantity", 0))
        if quantity > 10000:
            considerations.append("Large trade size may impact market liquidity")
            considerations.append("Consider using algorithmic execution to minimize impact")

        # Strategy-based considerations
        strategy = trade_data.get("strategy_id", "").lower()
        if "hft" in strategy:
            considerations.append("High-frequency strategy may not benefit long-term investors")

        if "arbitrage" in strategy:
            considerations.append("Arbitrage activity helps improve market efficiency")

        return considerations

    def _calculate_transparency_level(self, trade_data: Dict[str, Any]) -> float:
        """Calculate transparency level for a trade"""
        transparency = 0.5  # Base level

        # Increase transparency for disclosed positions
        if trade_data.get("disclosed", False):
            transparency += 0.3

        # Increase transparency for public strategies
        if trade_data.get("public_strategy", False):
            transparency += 0.2

        # Decrease transparency for proprietary algorithms
        if "proprietary" in trade_data.get("strategy_id", "").lower():
            transparency -= 0.1

        return np.clip(transparency, 0.0, 1.0)

    def _create_neutral_ethical_trade(self, trade_data: Dict[str, Any]) -> EthicalTrade:
        """Create a neutral ethical trade when analysis fails"""
        return EthicalTrade(
            original_trade_id=trade_data.get("trade_id", "unknown"),
            symbol=trade_data.get("symbol", ""),
            action=trade_data.get("action", ""),
            quantity=trade_data.get("quantity", 0),
            price=trade_data.get("price", 0),
            strategy_id=trade_data.get("strategy_id", ""),
            alpha_type=AlphaType.NEUTRAL,
            social_impact_score=0.0,
            ethical_considerations=["Unable to fully analyze ethical implications"],
            alternative_actions=["Conduct deeper ethical analysis"],
            transparency_level=0.5
        )

    def _create_impact_breakdown(self, trades: List[EthicalTrade]) -> Dict[str, Any]:
        """Create detailed breakdown of social impact"""
        if not trades:
            return {}

        impact_ranges = {
            "highly_positive": len([t for t in trades if t.social_impact_score > 0.5]),
            "positive": len([t for t in trades if 0 < t.social_impact_score <= 0.5]),
            "neutral": len([t for t in trades if t.social_impact_score == 0]),
            "negative": len([t for t in trades if -0.5 <= t.social_impact_score < 0]),
            "highly_negative": len([t for t in trades if t.social_impact_score < -0.5])
        }

        strategy_impact = {}
        for trade in trades:
            strategy = trade.strategy_id
            if strategy not in strategy_impact:
                strategy_impact[strategy] = []
            strategy_impact[strategy].append(trade.social_impact_score)

        strategy_averages = {
            strategy: np.mean(scores)
            for strategy, scores in strategy_impact.items()
        }

        return {
            "impact_distribution": impact_ranges,
            "strategy_impact_averages": strategy_averages,
            "most_positive_strategy": max(strategy_averages, key=strategy_averages.get) if strategy_averages else None,
            "least_positive_strategy": min(strategy_averages, key=strategy_averages.get) if strategy_averages else None
        }

# Example usage and testing
def test_policy_twin():
    """Test the Policy Twin system"""
    policy_twin = PolicyTwin()

    # Sample trade data
    sample_trades = [
        {
            "trade_id": "TRADE_001",
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 5000,
            "price": 150.0,
            "strategy_id": "momentum_strategy",
            "disclosed": False,
            "expected_price_impact": 0.001
        },
        {
            "trade_id": "TRADE_002",
            "symbol": "MSFT",
            "action": "sell",
            "quantity": 15000,
            "price": 300.0,
            "strategy_id": "arbitrage_strategy",
            "disclosed": True,
            "expected_price_impact": 0.002
        }
    ]

    # Analyze trades
    for trade_data in sample_trades:
        ethical_trade = policy_twin.analyze_trade_ethics(trade_data)
        print(f"Trade {ethical_trade.original_trade_id}: {ethical_trade.alpha_type.value} "
              f"(Impact: {ethical_trade.social_impact_score:.3f})")

    # Calculate social impact metrics
    metrics = policy_twin.calculate_social_impact_metrics()
    print(f"\nCalculated {len(metrics)} social impact metrics")

    # Generate policy recommendations
    recommendations = policy_twin.generate_policy_recommendations()
    print(f"Generated {len(recommendations)} policy recommendations")

    # Create transparency report
    report = policy_twin.create_transparency_report()
    print(f"\nTransparency Report Summary:")
    print(f"Total trades analyzed: {report['trade_summary']['total_trades']}")
    print(f"Constructive trades: {report['trade_summary']['constructive_percentage']:.1f}%")
    print(f"Average social impact: {report['impact_metrics']['average_social_impact_score']:.3f}")

    return policy_twin

if __name__ == "__main__":
    test_policy_twin()