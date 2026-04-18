"""
Strategy Simulator for Reward Calculation

Phase 2 of the training pipeline:
- Takes a strategy selection from the model
- Simulates how that strategy would perform over a horizon
- Returns actual PnL to use as reward signal

The 8 strategies have different allocations:
0: ultra_defensive  - 20% SPY, 50% TLT, 30% Cash
1: defensive        - 40% SPY, 30% TLT, 30% Cash
2: balanced_safe    - 60% SPY, 20% TLT, 20% Cash
3: balanced_growth  - 70% SPY, 20% TLT, 10% Cash
4: growth           - 80% SPY, 15% TLT, 5% Cash
5: aggressive_growth- 90% SPY, 10% TLT, 0% Cash
6: contrarian_long  - 85% SPY, 15% TLT, 0% Cash (buy the dip)
7: tactical_opportunity - 75% SPY, 25% TLT, 0% Cash

We simulate forward and calculate:
- Total return
- Max drawdown
- Risk-adjusted return (Sharpe-like)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Strategy allocations: (SPY%, TLT%, Cash%)
STRATEGY_ALLOCATIONS = {
    0: (0.20, 0.50, 0.30),  # ultra_defensive
    1: (0.40, 0.30, 0.30),  # defensive
    2: (0.60, 0.20, 0.20),  # balanced_safe
    3: (0.70, 0.20, 0.10),  # balanced_growth
    4: (0.80, 0.15, 0.05),  # growth
    5: (0.90, 0.10, 0.00),  # aggressive_growth
    6: (0.85, 0.15, 0.00),  # contrarian_long
    7: (0.75, 0.25, 0.00),  # tactical_opportunity
}

STRATEGY_NAMES = [
    'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
    'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
]


@dataclass
class SimulationResult:
    """Result of strategy simulation."""
    total_return: float        # Cumulative return over horizon
    max_drawdown: float        # Maximum peak-to-trough decline
    volatility: float          # Annualized volatility
    sharpe_ratio: float        # Risk-adjusted return
    final_nav: float           # Final NAV (starting from 1.0)
    daily_returns: np.ndarray  # Daily return series


class StrategySimulator:
    """
    Simulates strategy performance using historical returns.

    Used to calculate reward signal for training:
    - Positive reward for good strategy choices
    - Negative reward (punishment) for bad choices
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        """
        Args:
            spy_returns: Array of daily SPY returns
            tlt_returns: Array of daily TLT returns
            dates: Optional date array for indexing
            risk_free_rate: Annual risk-free rate for Sharpe
        """
        self.spy_returns = np.asarray(spy_returns)
        self.tlt_returns = np.asarray(tlt_returns)
        self.dates = dates
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def simulate_strategy(
        self,
        strategy_idx: int,
        start_idx: int,
        horizon_days: int = 5,
        initial_nav: float = 1.0
    ) -> SimulationResult:
        """
        Simulate a strategy from start_idx for horizon_days.

        Args:
            strategy_idx: Strategy index (0-7)
            start_idx: Starting index in return arrays
            horizon_days: Number of days to simulate
            initial_nav: Starting NAV

        Returns:
            SimulationResult with performance metrics
        """
        if strategy_idx not in STRATEGY_ALLOCATIONS:
            raise ValueError(f"Invalid strategy: {strategy_idx}")

        spy_alloc, tlt_alloc, cash_alloc = STRATEGY_ALLOCATIONS[strategy_idx]

        # Get returns for simulation period
        end_idx = min(start_idx + horizon_days, len(self.spy_returns))
        actual_days = end_idx - start_idx

        if actual_days <= 0:
            return SimulationResult(
                total_return=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                final_nav=initial_nav,
                daily_returns=np.array([0.0])
            )

        spy_ret = self.spy_returns[start_idx:end_idx]
        tlt_ret = self.tlt_returns[start_idx:end_idx]

        # Calculate portfolio returns
        portfolio_returns = (
            spy_alloc * spy_ret +
            tlt_alloc * tlt_ret +
            cash_alloc * self.daily_rf
        )

        # Calculate NAV series
        nav = initial_nav * np.cumprod(1 + portfolio_returns)
        nav = np.insert(nav, 0, initial_nav)  # Add initial NAV

        # Total return
        total_return = (nav[-1] / nav[0]) - 1

        # Max drawdown
        running_max = np.maximum.accumulate(nav)
        drawdowns = (nav - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        # Volatility (annualized)
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.0

        # Sharpe ratio
        excess_returns = portfolio_returns - self.daily_rf
        sharpe = 0.0
        if volatility > 0:
            sharpe = (np.mean(excess_returns) * 252) / volatility

        return SimulationResult(
            total_return=float(total_return),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            final_nav=float(nav[-1]),
            daily_returns=portfolio_returns
        )

    def compare_strategies(
        self,
        start_idx: int,
        horizon_days: int = 5
    ) -> Dict[int, SimulationResult]:
        """
        Simulate all 8 strategies from the same starting point.

        Returns dict mapping strategy_idx -> SimulationResult
        """
        results = {}
        for strategy_idx in range(8):
            results[strategy_idx] = self.simulate_strategy(
                strategy_idx, start_idx, horizon_days
            )
        return results

    def calculate_reward(
        self,
        chosen_strategy: int,
        start_idx: int,
        horizon_days: int = 5,
        reward_type: str = 'relative'
    ) -> Tuple[float, Dict]:
        """
        Calculate reward for a strategy choice.

        Args:
            chosen_strategy: Strategy index model chose
            start_idx: Starting index
            horizon_days: Simulation horizon
            reward_type: 'absolute' (raw return) or 'relative' (vs best strategy)

        Returns:
            (reward, info_dict)
        """
        # Simulate chosen strategy
        result = self.simulate_strategy(chosen_strategy, start_idx, horizon_days)

        info = {
            'chosen_strategy': chosen_strategy,
            'total_return': result.total_return,
            'max_drawdown': result.max_drawdown,
            'sharpe': result.sharpe_ratio,
        }

        if reward_type == 'absolute':
            # Raw return as reward, penalize drawdown
            reward = result.total_return - 0.5 * result.max_drawdown
        else:
            # Compare to all strategies
            all_results = self.compare_strategies(start_idx, horizon_days)
            returns = [r.total_return for r in all_results.values()]

            best_return = max(returns)
            worst_return = min(returns)
            mean_return = np.mean(returns)

            # Best strategy
            best_strategy = max(all_results.keys(), key=lambda k: all_results[k].total_return)
            info['best_strategy'] = best_strategy
            info['best_return'] = best_return

            # Reward: how close to optimal?
            if best_return > worst_return:
                # Normalize to [-1, 1]
                reward = 2 * (result.total_return - worst_return) / (best_return - worst_return + 1e-8) - 1
            else:
                reward = 0.0

            # Bonus for picking the best
            if chosen_strategy == best_strategy:
                reward += 0.5

            # Penalty for high drawdown
            reward -= 0.3 * result.max_drawdown

        info['reward'] = reward
        return float(reward), info


class BatchStrategySimulator:
    """
    Batch simulator for training with GPU tensors.

    Pre-computes strategy returns for efficient batch reward calculation.
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        horizon_days: int = 5,
        device: str = 'cpu'
    ):
        self.horizon_days = horizon_days
        self.device = device
        self.n_samples = len(spy_returns) - horizon_days

        # Pre-compute portfolio returns for each strategy at each time
        self.strategy_returns = {}

        for strategy_idx, (spy_alloc, tlt_alloc, cash_alloc) in STRATEGY_ALLOCATIONS.items():
            returns = np.zeros(self.n_samples)

            for i in range(self.n_samples):
                spy_ret = spy_returns[i:i+horizon_days]
                tlt_ret = tlt_returns[i:i+horizon_days]

                # Cumulative return over horizon
                portfolio_daily = spy_alloc * spy_ret + tlt_alloc * tlt_ret
                cumulative = np.prod(1 + portfolio_daily) - 1
                returns[i] = cumulative

            self.strategy_returns[strategy_idx] = torch.tensor(
                returns, dtype=torch.float32, device=device
            )

        # Stack into tensor (n_samples, 8)
        self.all_returns = torch.stack(
            [self.strategy_returns[i] for i in range(8)], dim=1
        )

        logger.info(f"Pre-computed returns: {self.all_returns.shape}")

    def get_reward(
        self,
        strategy_indices: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get rewards for batch of strategy choices.

        Args:
            strategy_indices: (batch,) chosen strategies
            sample_indices: (batch,) time indices

        Returns:
            rewards: (batch,) reward values
        """
        batch_size = strategy_indices.shape[0]

        # Clamp indices to valid range
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        # Get return of chosen strategy
        chosen_returns = self.all_returns[sample_indices, strategy_indices]

        # Get best possible return at each time
        best_returns, _ = self.all_returns[sample_indices].max(dim=1)
        worst_returns, _ = self.all_returns[sample_indices].min(dim=1)

        # Normalized reward: how close to best?
        range_returns = best_returns - worst_returns + 1e-8
        reward = 2 * (chosen_returns - worst_returns) / range_returns - 1

        # Bonus for picking best
        best_strategies = self.all_returns[sample_indices].argmax(dim=1)
        is_best = (strategy_indices == best_strategies).float()
        reward = reward + 0.5 * is_best

        return reward

    def get_optimal_strategy(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """Get the optimal strategy at each time index."""
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)
        return self.all_returns[sample_indices].argmax(dim=1)
