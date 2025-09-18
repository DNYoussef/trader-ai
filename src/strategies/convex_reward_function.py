"""
Convex Reward Function - Taleb-Inspired Reward System
Implements exponential rewards for black swan captures and minimal penalties for small losses

This is the core of our antifragile AI training - teaching the model to:
1. Aggressively pursue massive upside (exponential rewards)
2. Accept small losses as cost of optionality (mild penalties)
3. Absolutely avoid blowups (extreme penalties)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradeOutcome:
    """Represents the outcome of a trade for reward calculation"""
    strategy_name: str
    entry_date: datetime
    exit_date: datetime
    symbol: str
    returns: float  # Percentage return
    max_drawdown: float  # Maximum drawdown during trade
    holding_period_days: int
    volatility_during_trade: float
    is_black_swan_period: bool
    black_swan_captured: bool
    convexity_achieved: float  # Actual upside/downside ratio

@dataclass
class RewardMetrics:
    """Detailed reward calculation metrics"""
    base_reward: float
    convexity_bonus: float
    black_swan_multiplier: float
    antifragility_bonus: float
    blowup_penalty: float
    final_reward: float
    reward_components: Dict[str, float]

class ConvexRewardFunction:
    """
    Implements Taleb-inspired convex reward system

    Core Philosophy:
    - Massive rewards for tail event captures (3x-10x multiplier)
    - Small penalties for controlled losses (0.5x penalty)
    - Catastrophic penalties for blowups (10x-100x penalty)
    - Bonus for antifragile behavior (profits during volatility)
    """

    def __init__(self,
                 black_swan_threshold: float = 0.50,  # 50%+ gain for black swan
                 small_loss_threshold: float = -0.10,  # -10% for acceptable loss
                 blowup_threshold: float = -0.50,      # -50% for catastrophic loss
                 max_reward_multiplier: float = 10.0): # Maximum reward multiplier
        """
        Initialize convex reward function

        Args:
            black_swan_threshold: Return threshold for black swan classification
            small_loss_threshold: Threshold for acceptable small losses
            blowup_threshold: Threshold for catastrophic losses
            max_reward_multiplier: Maximum multiplier for exceptional trades
        """
        self.black_swan_threshold = black_swan_threshold
        self.small_loss_threshold = small_loss_threshold
        self.blowup_threshold = blowup_threshold
        self.max_reward_multiplier = max_reward_multiplier

        # Reward calculation parameters
        self.convexity_target = 10.0  # Target 10:1 upside/downside
        self.antifragility_threshold = 0.20  # 20% return with <5% drawdown

        # Track reward history for analysis
        self.reward_history: List[RewardMetrics] = []

        # Database for storing reward calculations
        self.db_path = Path("data/black_swan_training.db")

        logger.info(f"ConvexRewardFunction initialized with thresholds: "
                   f"black_swan={black_swan_threshold}, blowup={blowup_threshold}")

    def calculate_reward(self,
                        outcome: TradeOutcome,
                        market_conditions: Optional[Dict] = None) -> RewardMetrics:
        """
        Calculate convex reward for a trade outcome

        Args:
            outcome: Trade outcome details
            market_conditions: Optional market conditions during trade

        Returns:
            RewardMetrics with detailed reward calculation
        """
        returns = outcome.returns
        max_drawdown = outcome.max_drawdown

        # Initialize reward components
        reward_components = {}

        # 1. BASE REWARD (linear component)
        base_reward = returns
        reward_components['base'] = base_reward

        # 2. CONVEXITY CALCULATION
        if max_drawdown != 0:
            actual_convexity = abs(returns / max_drawdown)
        else:
            actual_convexity = float('inf') if returns > 0 else 0

        # 3. BLACK SWAN CAPTURE (exponential rewards)
        black_swan_multiplier = 1.0
        if returns >= self.black_swan_threshold:
            # Exponential scaling for massive gains
            excess_return = returns - self.black_swan_threshold
            black_swan_multiplier = 1 + (excess_return ** 2) * 10  # Quadratic scaling

            # Extra bonus if it was during an actual black swan period
            if outcome.is_black_swan_period and outcome.black_swan_captured:
                black_swan_multiplier *= 2.0

            # Cap at maximum multiplier
            black_swan_multiplier = min(black_swan_multiplier, self.max_reward_multiplier)

            reward_components['black_swan'] = base_reward * (black_swan_multiplier - 1)

        # 4. SMALL LOSS PENALTY (mild)
        if self.small_loss_threshold <= returns < 0:
            # Only 50% penalty for small losses - these are acceptable
            base_reward *= 0.5
            reward_components['small_loss_adjustment'] = base_reward * 0.5

        # 5. CATASTROPHIC LOSS PENALTY (extreme)
        blowup_penalty = 0
        if returns <= self.blowup_threshold:
            # Severe exponential penalty for blowups
            blowup_magnitude = abs(returns - self.blowup_threshold)
            blowup_penalty = (blowup_magnitude ** 2) * 100  # Quadratic penalty
            reward_components['blowup'] = -blowup_penalty

        # 6. CONVEXITY BONUS
        convexity_bonus = 0
        if actual_convexity > self.convexity_target:
            # Reward achieving target convexity
            convexity_excess = actual_convexity - self.convexity_target
            convexity_bonus = min(convexity_excess * 0.1, 1.0) * abs(base_reward)
            reward_components['convexity'] = convexity_bonus

        # 7. ANTIFRAGILITY BONUS (profits with low drawdown)
        antifragility_bonus = 0
        if returns >= self.antifragility_threshold and abs(max_drawdown) < 0.05:
            # Achieved good returns with minimal drawdown - true antifragility
            antifragility_bonus = returns * 2.0  # Double reward
            reward_components['antifragility'] = antifragility_bonus

        # 8. VOLATILITY HARVESTING BONUS
        if market_conditions:
            vix_level = market_conditions.get('vix_level', 20)
            if vix_level > 30 and returns > 0:
                # Bonus for profits during high volatility
                vol_bonus = returns * (vix_level / 30 - 1)
                reward_components['volatility_harvest'] = vol_bonus
                base_reward += vol_bonus

        # 9. TIME EFFICIENCY ADJUSTMENT
        if outcome.holding_period_days > 0:
            # Prefer faster captures
            time_factor = min(1.0, 10 / outcome.holding_period_days)
            base_reward *= time_factor
            reward_components['time_efficiency'] = base_reward * (time_factor - 1)

        # 10. CALCULATE FINAL REWARD
        final_reward = (
            base_reward * black_swan_multiplier +
            convexity_bonus +
            antifragility_bonus -
            blowup_penalty
        )

        # Create metrics object
        metrics = RewardMetrics(
            base_reward=returns,  # Original return
            convexity_bonus=convexity_bonus,
            black_swan_multiplier=black_swan_multiplier,
            antifragility_bonus=antifragility_bonus,
            blowup_penalty=blowup_penalty,
            final_reward=final_reward,
            reward_components=reward_components
        )

        # Store in history
        self.reward_history.append(metrics)
        self._save_reward_to_db(outcome, metrics)

        logger.debug(f"Reward calculated: returns={returns:.2%} -> reward={final_reward:.4f} "
                    f"(multiplier={black_swan_multiplier:.1f}x)")

        return metrics

    def calculate_batch_rewards(self, outcomes: List[TradeOutcome]) -> List[RewardMetrics]:
        """
        Calculate rewards for multiple trade outcomes

        Args:
            outcomes: List of trade outcomes

        Returns:
            List of reward metrics
        """
        rewards = []
        for outcome in outcomes:
            reward = self.calculate_reward(outcome)
            rewards.append(reward)

        # Log statistics
        if rewards:
            avg_reward = np.mean([r.final_reward for r in rewards])
            max_reward = max(r.final_reward for r in rewards)
            min_reward = min(r.final_reward for r in rewards)

            logger.info(f"Batch rewards calculated: count={len(rewards)}, "
                       f"avg={avg_reward:.4f}, max={max_reward:.4f}, min={min_reward:.4f}")

        return rewards

    def optimize_for_strategy(self, strategy_name: str, historical_outcomes: List[TradeOutcome]) -> Dict[str, float]:
        """
        Optimize reward parameters for a specific strategy based on historical performance

        Args:
            strategy_name: Name of the strategy
            historical_outcomes: Historical trade outcomes for the strategy

        Returns:
            Optimized parameters for the strategy
        """
        strategy_outcomes = [o for o in historical_outcomes if o.strategy_name == strategy_name]

        if not strategy_outcomes:
            return {}

        # Calculate strategy characteristics
        returns = [o.returns for o in strategy_outcomes]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        max_return = max(returns)
        min_return = min(returns)

        # Calculate black swan capture rate
        black_swans_captured = sum(1 for o in strategy_outcomes if o.black_swan_captured)
        capture_rate = black_swans_captured / len(strategy_outcomes) if strategy_outcomes else 0

        # Optimize thresholds based on strategy profile
        optimized_params = {
            'black_swan_threshold': np.percentile([r for r in returns if r > 0], 90) if any(r > 0 for r in returns) else 0.5,
            'small_loss_threshold': np.percentile([r for r in returns if r < 0], 25) if any(r < 0 for r in returns) else -0.1,
            'blowup_threshold': min_return * 0.8,  # 80% of worst loss
            'optimal_multiplier': 1 + (capture_rate * 9),  # 1-10x based on capture rate
            'strategy_weight': min(1.0, capture_rate * 2)  # Weight based on success rate
        }

        logger.info(f"Optimized parameters for {strategy_name}: {optimized_params}")

        return optimized_params

    def _save_reward_to_db(self, outcome: TradeOutcome, metrics: RewardMetrics):
        """Save reward calculation to database for training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Save to training examples
                cursor.execute('''
                    INSERT INTO training_examples
                    (date, market_state, selected_strategy, outcome, reward, convexity_achieved, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    outcome.entry_date.isoformat(),
                    json.dumps({'volatility': outcome.volatility_during_trade}),
                    outcome.strategy_name,
                    outcome.returns,
                    metrics.final_reward,
                    outcome.convexity_achieved,
                    json.dumps(metrics.reward_components)
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error saving reward to database: {e}")

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward distribution"""
        if not self.reward_history:
            return {}

        rewards = [r.final_reward for r in self.reward_history]
        black_swan_count = sum(1 for r in self.reward_history if r.black_swan_multiplier > 1)
        blowup_count = sum(1 for r in self.reward_history if r.blowup_penalty > 0)

        return {
            'total_calculations': len(self.reward_history),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'black_swan_captures': black_swan_count,
            'blowups': blowup_count,
            'avg_convexity_bonus': np.mean([r.convexity_bonus for r in self.reward_history]),
            'avg_antifragility_bonus': np.mean([r.antifragility_bonus for r in self.reward_history]),
            'reward_skewness': float(pd.Series(rewards).skew()),
            'reward_kurtosis': float(pd.Series(rewards).kurtosis())
        }

    def plot_reward_curve(self, return_range: Tuple[float, float] = (-1.0, 2.0), steps: int = 100) -> Dict[str, List[float]]:
        """
        Generate data for plotting the reward curve

        Args:
            return_range: Range of returns to plot
            steps: Number of steps in the range

        Returns:
            Dictionary with returns and corresponding rewards
        """
        returns = np.linspace(return_range[0], return_range[1], steps)
        rewards = []

        for ret in returns:
            outcome = TradeOutcome(
                strategy_name='test',
                entry_date=datetime.now(),
                exit_date=datetime.now(),
                symbol='TEST',
                returns=ret,
                max_drawdown=-abs(ret) * 0.3,  # Assume 30% of return as drawdown
                holding_period_days=10,
                volatility_during_trade=0.02,
                is_black_swan_period=False,
                black_swan_captured=ret > self.black_swan_threshold,
                convexity_achieved=abs(ret / (abs(ret) * 0.3)) if ret != 0 else 0
            )

            metrics = self.calculate_reward(outcome)
            rewards.append(metrics.final_reward)

        return {
            'returns': returns.tolist(),
            'rewards': rewards
        }


class AdaptiveRewardSystem:
    """
    Adaptive reward system that learns optimal reward functions for each strategy
    """

    def __init__(self):
        self.strategy_reward_functions: Dict[str, ConvexRewardFunction] = {}
        self.global_reward_function = ConvexRewardFunction()
        self.performance_history: Dict[str, List[float]] = {}

    def get_reward_function(self, strategy_name: str) -> ConvexRewardFunction:
        """Get or create reward function for a strategy"""
        if strategy_name not in self.strategy_reward_functions:
            self.strategy_reward_functions[strategy_name] = ConvexRewardFunction()
        return self.strategy_reward_functions[strategy_name]

    def update_from_performance(self, strategy_name: str, outcomes: List[TradeOutcome]):
        """Update reward function based on strategy performance"""
        reward_func = self.get_reward_function(strategy_name)

        # Optimize parameters based on outcomes
        optimized_params = reward_func.optimize_for_strategy(strategy_name, outcomes)

        # Update reward function parameters
        if 'black_swan_threshold' in optimized_params:
            reward_func.black_swan_threshold = optimized_params['black_swan_threshold']
        if 'small_loss_threshold' in optimized_params:
            reward_func.small_loss_threshold = optimized_params['small_loss_threshold']
        if 'blowup_threshold' in optimized_params:
            reward_func.blowup_threshold = optimized_params['blowup_threshold']

        # Track performance
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []

        strategy_returns = [o.returns for o in outcomes if o.strategy_name == strategy_name]
        if strategy_returns:
            self.performance_history[strategy_name].extend(strategy_returns)

    def calculate_portfolio_reward(self, strategy_outcomes: Dict[str, TradeOutcome]) -> float:
        """Calculate combined reward for portfolio of strategies"""
        total_reward = 0
        strategy_weights = {}

        # Calculate individual rewards
        for strategy_name, outcome in strategy_outcomes.items():
            reward_func = self.get_reward_function(strategy_name)
            metrics = reward_func.calculate_reward(outcome)

            # Get strategy weight based on historical performance
            if strategy_name in self.performance_history and self.performance_history[strategy_name]:
                avg_return = np.mean(self.performance_history[strategy_name])
                weight = max(0.1, min(1.0, 0.5 + avg_return))  # Weight between 0.1 and 1.0
            else:
                weight = 0.5  # Default weight

            strategy_weights[strategy_name] = weight
            total_reward += metrics.final_reward * weight

        # Normalize by total weight
        total_weight = sum(strategy_weights.values())
        if total_weight > 0:
            total_reward /= total_weight

        return total_reward


if __name__ == "__main__":
    # Test the reward function
    reward_func = ConvexRewardFunction()

    # Test various outcomes
    test_outcomes = [
        # Black swan capture
        TradeOutcome(
            strategy_name='tail_hedge',
            entry_date=datetime.now(),
            exit_date=datetime.now(),
            symbol='SPY_PUT',
            returns=2.5,  # 250% return
            max_drawdown=-0.02,
            holding_period_days=5,
            volatility_during_trade=0.05,
            is_black_swan_period=True,
            black_swan_captured=True,
            convexity_achieved=125.0
        ),
        # Small acceptable loss
        TradeOutcome(
            strategy_name='mean_reversion',
            entry_date=datetime.now(),
            exit_date=datetime.now(),
            symbol='AAPL',
            returns=-0.05,  # -5% loss
            max_drawdown=-0.05,
            holding_period_days=3,
            volatility_during_trade=0.02,
            is_black_swan_period=False,
            black_swan_captured=False,
            convexity_achieved=1.0
        ),
        # Catastrophic loss (blowup)
        TradeOutcome(
            strategy_name='momentum',
            entry_date=datetime.now(),
            exit_date=datetime.now(),
            symbol='TSLA',
            returns=-0.60,  # -60% loss
            max_drawdown=-0.70,
            holding_period_days=10,
            volatility_during_trade=0.10,
            is_black_swan_period=False,
            black_swan_captured=False,
            convexity_achieved=0.86
        ),
        # Antifragile trade
        TradeOutcome(
            strategy_name='crisis_alpha',
            entry_date=datetime.now(),
            exit_date=datetime.now(),
            symbol='GLD',
            returns=0.25,  # 25% return
            max_drawdown=-0.02,  # Only 2% drawdown
            holding_period_days=15,
            volatility_during_trade=0.03,
            is_black_swan_period=False,
            black_swan_captured=False,
            convexity_achieved=12.5
        )
    ]

    print("Testing Convex Reward Function:")
    print("=" * 60)

    for outcome in test_outcomes:
        metrics = reward_func.calculate_reward(outcome)
        print(f"\nStrategy: {outcome.strategy_name}")
        print(f"  Returns: {outcome.returns:.1%}")
        print(f"  Max Drawdown: {outcome.max_drawdown:.1%}")
        print(f"  Convexity Achieved: {outcome.convexity_achieved:.1f}x")
        print(f"  Base Reward: {metrics.base_reward:.4f}")
        print(f"  Black Swan Multiplier: {metrics.black_swan_multiplier:.1f}x")
        print(f"  Final Reward: {metrics.final_reward:.4f}")

    # Get statistics
    stats = reward_func.get_reward_statistics()
    print(f"\nReward Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")