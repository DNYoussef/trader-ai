"""
Black Swan Hunting System Demonstration
Shows: Market Conditions (A) -> LLM Strategy Selection (B) -> Convex Rewards

The system demonstrates how the AI selects strategies from its toolbox
based on market conditions, optimizing for:
1. Black swan capture (exponential rewards)
2. Risk resilience (small losses acceptable)
3. Antifragility (benefits from volatility)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from src.strategies.black_swan_strategies import (
    BlackSwanStrategyToolbox,
    MarketState
)
from src.strategies.convex_reward_function import (
    ConvexRewardFunction,
    TradeOutcome
)
from src.intelligence.local_llm_orchestrator import (
    LocalLLMOrchestrator,
    MarketContext
)

class BlackSwanSystemDemonstration:
    """
    Demonstrates the complete flow:
    Market Conditions -> LLM Selection -> Strategy Execution -> Convex Rewards
    """

    def __init__(self):
        self.strategy_toolbox = BlackSwanStrategyToolbox()
        self.convex_reward = ConvexRewardFunction()
        self.llm_orchestrator = LocalLLMOrchestrator()

        # Strategy descriptions for clarity
        self.strategy_descriptions = {
            'tail_hedge': "Buy deep OTM puts for extreme downside protection",
            'volatility_harvest': "Trade VIX futures during backwardation/contango",
            'crisis_alpha': "Execute pre-planned crisis response protocol",
            'momentum_explosion': "Ride parabolic moves with trailing stops",
            'mean_reversion': "Trade extreme oversold/overbought conditions",
            'correlation_breakdown': "Pairs trading when correlations break",
            'inequality_arbitrage': "Trade wealth concentration effects",
            'event_catalyst': "Position before known high-impact events"
        }

        logger.info("=" * 80)
        logger.info("BLACK SWAN HUNTING SYSTEM DEMONSTRATION")
        logger.info("Market Conditions (A) -> LLM Choice (B) -> Convex Rewards")
        logger.info("=" * 80)

    def create_market_scenarios(self) -> List[Dict]:
        """Create different market scenarios to demonstrate strategy selection"""

        scenarios = [
            {
                'name': "CALM BEFORE THE STORM",
                'description': "Low VIX, tight correlations, but building pressure",
                'market_state': MarketState(
                    timestamp=datetime.now(),
                    vix_level=12.5,
                    vix_percentile=0.15,
                    spy_returns_5d=0.02,
                    spy_returns_20d=0.05,
                    put_call_ratio=0.7,
                    market_breadth=0.65,
                    correlation=0.85,
                    volume_ratio=0.8,
                    regime='calm'
                ),
                'expected_strategy': 'tail_hedge',
                'rationale': "Cheap insurance when no one expects problems"
            },
            {
                'name': "VOLATILITY SPIKE",
                'description': "VIX jumping, correlations rising, fear emerging",
                'market_state': MarketState(
                    timestamp=datetime.now(),
                    vix_level=35.0,
                    vix_percentile=0.92,
                    spy_returns_5d=-0.08,
                    spy_returns_20d=-0.05,
                    put_call_ratio=1.8,
                    market_breadth=0.25,
                    correlation=0.95,
                    volume_ratio=2.5,
                    regime='crisis'
                ),
                'expected_strategy': 'crisis_alpha',
                'rationale': "Execute crisis playbook during panic"
            },
            {
                'name': "CORRELATION BREAKDOWN",
                'description': "Normal relationships breaking, pairs diverging",
                'market_state': MarketState(
                    timestamp=datetime.now(),
                    vix_level=22.0,
                    vix_percentile=0.65,
                    spy_returns_5d=0.01,
                    spy_returns_20d=-0.02,
                    put_call_ratio=1.1,
                    market_breadth=0.45,
                    correlation=0.15,  # Very low correlation
                    volume_ratio=1.3,
                    regime='volatile'
                ),
                'expected_strategy': 'correlation_breakdown',
                'rationale': "Profit from relationship divergences"
            },
            {
                'name': "MELT-UP CONDITIONS",
                'description': "Strong momentum, FOMO building, parabolic moves",
                'market_state': MarketState(
                    timestamp=datetime.now(),
                    vix_level=15.0,
                    vix_percentile=0.25,
                    spy_returns_5d=0.06,
                    spy_returns_20d=0.15,  # Strong momentum
                    put_call_ratio=0.5,  # Bullish sentiment
                    market_breadth=0.75,
                    correlation=0.6,
                    volume_ratio=1.8,
                    regime='bullish'
                ),
                'expected_strategy': 'momentum_explosion',
                'rationale': "Ride the bubble with protection"
            },
            {
                'name': "EXTREME OVERSOLD",
                'description': "Massive selloff, capitulation, potential reversal",
                'market_state': MarketState(
                    timestamp=datetime.now(),
                    vix_level=45.0,
                    vix_percentile=0.98,
                    spy_returns_5d=-0.15,  # Extreme selloff
                    spy_returns_20d=-0.25,
                    put_call_ratio=2.5,  # Extreme fear
                    market_breadth=0.10,
                    correlation=0.98,
                    volume_ratio=3.5,
                    regime='crash'
                ),
                'expected_strategy': 'mean_reversion',
                'rationale': "Buy when there's blood in the streets"
            }
        ]

        return scenarios

    def demonstrate_strategy_selection(self, scenario: Dict):
        """Show how LLM selects strategy based on market conditions"""

        logger.info(f"\n{'=' * 80}")
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info(f"Description: {scenario['description']}")
        logger.info("=" * 80)

        market_state = scenario['market_state']

        # Display market conditions (INPUT A)
        logger.info("\n📊 MARKET CONDITIONS (Input A):")
        logger.info(f"  VIX Level: {market_state.vix_level:.1f} (Percentile: {market_state.vix_percentile:.1%})")
        logger.info(f"  SPY Returns: 5-day: {market_state.spy_returns_5d:.2%}, 20-day: {market_state.spy_returns_20d:.2%}")
        logger.info(f"  Put/Call Ratio: {market_state.put_call_ratio:.2f}")
        logger.info(f"  Market Breadth: {market_state.market_breadth:.1%}")
        logger.info(f"  Correlation: {market_state.correlation:.2f}")
        logger.info(f"  Volume Ratio: {market_state.volume_ratio:.1f}")
        logger.info(f"  Regime: {market_state.regime.upper()}")

        # LLM Strategy Selection (CHOICE B)
        logger.info("\n🤖 LLM STRATEGY SELECTION (Choice B):")

        # Create market context for LLM
        market_context = MarketContext(
            timestamp=market_state.timestamp,
            vix_level=market_state.vix_level,
            vix_percentile=market_state.vix_percentile,
            spy_returns_5d=market_state.spy_returns_5d,
            spy_returns_20d=market_state.spy_returns_20d,
            put_call_ratio=market_state.put_call_ratio,
            market_breadth=market_state.market_breadth,
            correlation=market_state.correlation,
            volume_ratio=market_state.volume_ratio,
            recent_events=["Scenario simulation"],
            sector_performance={},
            black_swan_indicators={
                'tail_risk': market_state.vix_level / 20,  # Normalized
                'correlation_risk': abs(market_state.correlation - 0.5) * 2,
                'momentum_risk': abs(market_state.spy_returns_20d) / 0.1
            }
        )

        # Get LLM recommendation
        recommendation = self.llm_orchestrator.get_strategy_recommendation(
            market_context,
            available_capital=100000  # $100k for example
        )

        if recommendation:
            logger.info(f"  Selected Strategy: {recommendation.strategy_name}")
            logger.info(f"  Confidence: {recommendation.confidence:.1%}")
            logger.info(f"  Reasoning: {recommendation.reasoning}")
            logger.info(f"  Expected Convexity: {recommendation.expected_convexity:.1f}x")
            logger.info(f"  Position Size: {recommendation.position_size:.1%} of capital")
        else:
            # Fallback selection
            logger.info(f"  Selected Strategy: {scenario['expected_strategy']} (fallback)")
            logger.info(f"  Reasoning: {scenario['rationale']}")

        selected_strategy = recommendation.strategy_name if recommendation else scenario['expected_strategy']

        # Show strategy toolbox
        logger.info("\n🧰 STRATEGY TOOLBOX:")
        for strategy_name, description in self.strategy_descriptions.items():
            if strategy_name == selected_strategy:
                logger.info(f"  ✅ {strategy_name}: {description}")
            else:
                logger.info(f"  ⚪ {strategy_name}: {description}")

        return selected_strategy

    def simulate_trade_outcomes(self, strategy: str, market_scenario: Dict) -> List[TradeOutcome]:
        """Simulate different possible outcomes for the selected strategy"""

        logger.info(f"\n📈 SIMULATING TRADE OUTCOMES FOR: {strategy}")

        outcomes = []

        # Simulate 3 scenarios: Black Swan Capture, Small Loss, Normal Win

        # Scenario 1: Black Swan Capture (rare but huge payoff)
        if market_scenario['market_state'].vix_level > 30:
            black_swan_outcome = TradeOutcome(
                strategy_name=strategy,
                entry_date=datetime.now(),
                exit_date=datetime.now() + timedelta(days=5),
                symbol=f"{strategy.upper()}_TRADE",
                returns=2.5,  # 250% return
                max_drawdown=-0.05,
                holding_period_days=5,
                volatility_during_trade=0.15,
                is_black_swan_period=True,
                black_swan_captured=True,
                convexity_achieved=50.0  # 50x convexity
            )
            outcomes.append(('BLACK SWAN CAPTURE', black_swan_outcome))

        # Scenario 2: Small Loss (acceptable, happens often)
        small_loss = TradeOutcome(
            strategy_name=strategy,
            entry_date=datetime.now(),
            exit_date=datetime.now() + timedelta(days=3),
            symbol=f"{strategy.upper()}_HEDGE",
            returns=-0.02,  # 2% loss
            max_drawdown=-0.02,
            holding_period_days=3,
            volatility_during_trade=0.05,
            is_black_swan_period=False,
            black_swan_captured=False,
            convexity_achieved=0.0
        )
        outcomes.append(('SMALL LOSS', small_loss))

        # Scenario 3: Normal Win
        normal_win = TradeOutcome(
            strategy_name=strategy,
            entry_date=datetime.now(),
            exit_date=datetime.now() + timedelta(days=10),
            symbol=f"{strategy.upper()}_WIN",
            returns=0.15,  # 15% return
            max_drawdown=-0.03,
            holding_period_days=10,
            volatility_during_trade=0.08,
            is_black_swan_period=False,
            black_swan_captured=False,
            convexity_achieved=5.0
        )
        outcomes.append(('NORMAL WIN', normal_win))

        return outcomes

    def calculate_convex_rewards(self, outcomes: List[tuple]):
        """Calculate rewards using our convex function"""

        logger.info("\n💎 CONVEX REWARD CALCULATIONS:")
        logger.info("(Optimizing for Antifragility, not just profits)")

        total_reward = 0

        for scenario_name, outcome in outcomes:
            reward_metrics = self.convex_reward.calculate_reward(outcome)

            logger.info(f"\n  {scenario_name}:")
            logger.info(f"    Returns: {outcome.returns:.2%}")
            logger.info(f"    Base Reward: {reward_metrics.base_reward:.2f}")

            if outcome.black_swan_captured:
                logger.info(f"    🦢 Black Swan Multiplier: {reward_metrics.black_swan_multiplier:.1f}x")

            logger.info(f"    Convexity Bonus: {reward_metrics.convexity_bonus:.2f}")
            logger.info(f"    Final Reward: {reward_metrics.final_reward:.2f}")

            total_reward += reward_metrics.final_reward

        logger.info(f"\n  📊 TOTAL REWARD SCORE: {total_reward:.2f}")

        # Explain the reward philosophy
        logger.info("\n  🎯 REWARD PHILOSOPHY:")
        logger.info("    • Small losses: Minimal penalty (cost of insurance)")
        logger.info("    • Black swan capture: Exponential rewards (10x+ multiplier)")
        logger.info("    • Convexity: Bonus for asymmetric risk/reward")
        logger.info("    • Antifragility: System gets stronger from volatility")

        return total_reward

    def run_demonstration(self):
        """Run complete demonstration of the system"""

        scenarios = self.create_market_scenarios()

        logger.info("\n" + "=" * 80)
        logger.info("DEMONSTRATING COMPLETE SYSTEM FLOW")
        logger.info("=" * 80)

        all_rewards = []

        for scenario in scenarios:
            # Step 1: LLM selects strategy based on market conditions
            selected_strategy = self.demonstrate_strategy_selection(scenario)

            # Step 2: Simulate possible outcomes
            outcomes = self.simulate_trade_outcomes(selected_strategy, scenario)

            # Step 3: Calculate convex rewards
            total_reward = self.calculate_convex_rewards(outcomes)

            all_rewards.append({
                'scenario': scenario['name'],
                'strategy': selected_strategy,
                'reward': total_reward
            })

            logger.info("\n" + "-" * 80)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SYSTEM PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        for result in all_rewards:
            logger.info(f"  {result['scenario']}")
            logger.info(f"    Strategy: {result['strategy']}")
            logger.info(f"    Reward Score: {result['reward']:.2f}")

        avg_reward = sum(r['reward'] for r in all_rewards) / len(all_rewards)
        logger.info(f"\n  Average Reward Score: {avg_reward:.2f}")

        # Key insights
        logger.info("\n🔑 KEY INSIGHTS:")
        logger.info("  1. The system doesn't optimize for maximum profit")
        logger.info("  2. It optimizes for surviving and thriving in black swan events")
        logger.info("  3. Small losses are acceptable (cost of being ready)")
        logger.info("  4. Massive rewards for correctly identifying tail events")
        logger.info("  5. Each strategy is a tool, selected based on market conditions")
        logger.info("  6. The LLM learns which tools work best in which situations")
        logger.info("  7. Antifragility: The system benefits from market chaos")

    def explain_system_philosophy(self):
        """Explain the core philosophy of the system"""

        logger.info("\n" + "=" * 80)
        logger.info("NASSIM TALEB'S BARBELL STRATEGY IMPLEMENTATION")
        logger.info("=" * 80)

        logger.info("\n📚 CORE PHILOSOPHY:")

        logger.info("\n1️⃣ BARBELL ALLOCATION (80/20 Rule):")
        logger.info("   • 80% in boring, safe investments (cash, treasuries)")
        logger.info("   • 20% in aggressive black swan hunting strategies")
        logger.info("   • Never in the middle (no medium risk)")

        logger.info("\n2️⃣ CONVEX PAYOFF STRUCTURE:")
        logger.info("   • Limited downside (can only lose 20% max)")
        logger.info("   • Unlimited upside (black swans can return 10x-100x)")
        logger.info("   • Small frequent losses, rare massive wins")

        logger.info("\n3️⃣ ANTIFRAGILITY PRINCIPLES:")
        logger.info("   • System gets stronger from volatility")
        logger.info("   • Each crisis teaches the AI better patterns")
        logger.info("   • Errors have small costs, successes have huge rewards")

        logger.info("\n4️⃣ LLM AS STRATEGY SELECTOR:")
        logger.info("   • Input: Market conditions (fear, greed, volatility, etc.)")
        logger.info("   • Process: Pattern matching against historical black swans")
        logger.info("   • Output: Optimal strategy from toolbox")
        logger.info("   • Learning: Reinforcement from convex reward function")

        logger.info("\n5️⃣ REWARD FUNCTION DESIGN:")
        logger.info("   • NOT optimizing for Sharpe ratio or consistent returns")
        logger.info("   • Optimizing for survival + black swan capture")
        logger.info("   • Exponential rewards for tail event profits")
        logger.info("   • Minimal penalties for small losses")
        logger.info("   • Extreme penalties for blow-ups")

        logger.info("\n🎯 THE GOAL:")
        logger.info("   Find the gap between consensus and reality")
        logger.info("   Position for black swans BEFORE they're obvious")
        logger.info("   Survive all environments, thrive in chaos")
        logger.info("   Build wealth through rare, massive wins")

def main():
    """Main demonstration execution"""

    demo = BlackSwanSystemDemonstration()

    # Explain the philosophy first
    demo.explain_system_philosophy()

    # Run the demonstration
    demo.run_demonstration()

    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\n✅ System ready to hunt black swans in real markets!")
    logger.info("🦢 Remember: We're not trying to predict the future,")
    logger.info("   we're positioning to benefit from unpredictability.")

if __name__ == "__main__":
    main()