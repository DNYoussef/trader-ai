"""
Continuous Strategy Simulator for Phase 3 Training

Evaluates arbitrary allocations (not just 8 discrete strategies):
- Blended strategies: weighted combination of 8 base strategies
- Direct allocations: raw SPY/TLT/Cash percentages

Provides rewards for RL training with continuous action spaces.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .strategy_simulator import STRATEGY_ALLOCATIONS, STRATEGY_NAMES

logger = logging.getLogger(__name__)


class ContinuousStrategySimulator:
    """
    Simulator for continuous allocation evaluation.

    Supports:
    1. Discrete strategies (backward compatible)
    2. Blended strategies (weighted combination)
    3. Direct allocations (any SPY/TLT/Cash mix)

    Computes returns, drawdowns, and reward signals for RL.
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        horizon_days: int = 5,
        device: str = 'cpu',
        risk_free_rate: float = 0.02
    ):
        self.horizon_days = horizon_days
        self.device = device
        self.n_samples = len(spy_returns) - horizon_days
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Store returns as tensors
        self.spy_returns = torch.tensor(
            spy_returns, dtype=torch.float32, device=device
        )
        self.tlt_returns = torch.tensor(
            tlt_returns, dtype=torch.float32, device=device
        )

        # Pre-compute cumulative returns for each base strategy
        self._precompute_strategy_returns()

        # Pre-compute SPY/TLT cumulative returns for direct allocation
        self._precompute_asset_returns()

        logger.info(f"ContinuousSimulator: {self.n_samples} samples, "
                   f"{horizon_days}-day horizon")

    def _precompute_strategy_returns(self):
        """Pre-compute returns for discrete strategies."""
        self.strategy_returns = {}

        for idx, (spy_alloc, tlt_alloc, cash_alloc) in STRATEGY_ALLOCATIONS.items():
            returns = torch.zeros(self.n_samples, device=self.device)

            for i in range(self.n_samples):
                spy_ret = self.spy_returns[i:i+self.horizon_days]
                tlt_ret = self.tlt_returns[i:i+self.horizon_days]

                portfolio_daily = spy_alloc * spy_ret + tlt_alloc * tlt_ret
                cumulative = torch.prod(1 + portfolio_daily) - 1
                returns[i] = cumulative

            self.strategy_returns[idx] = returns

        # Stack: (n_samples, 8)
        self.all_strategy_returns = torch.stack(
            [self.strategy_returns[i] for i in range(8)], dim=1
        )

    def _precompute_asset_returns(self):
        """Pre-compute cumulative returns for SPY and TLT."""
        self.spy_cumulative = torch.zeros(self.n_samples, device=self.device)
        self.tlt_cumulative = torch.zeros(self.n_samples, device=self.device)

        for i in range(self.n_samples):
            spy_ret = self.spy_returns[i:i+self.horizon_days]
            tlt_ret = self.tlt_returns[i:i+self.horizon_days]

            self.spy_cumulative[i] = torch.prod(1 + spy_ret) - 1
            self.tlt_cumulative[i] = torch.prod(1 + tlt_ret) - 1

    def get_discrete_reward(
        self,
        strategy_indices: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """Get rewards for discrete strategy selection (Phase 2 compatible)."""
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        chosen_returns = self.all_strategy_returns[sample_indices, strategy_indices]
        best_returns, _ = self.all_strategy_returns[sample_indices].max(dim=1)
        worst_returns, _ = self.all_strategy_returns[sample_indices].min(dim=1)

        range_returns = best_returns - worst_returns + 1e-8
        reward = 2 * (chosen_returns - worst_returns) / range_returns - 1

        best_strategies = self.all_strategy_returns[sample_indices].argmax(dim=1)
        is_best = (strategy_indices == best_strategies).float()
        reward = reward + 0.5 * is_best

        return reward

    def get_blended_return(
        self,
        strategy_weights: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate return for blended strategy allocation.

        Args:
            strategy_weights: (batch, 8) weights summing to 1
            sample_indices: (batch,) time indices

        Returns:
            returns: (batch,) cumulative returns
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        # Get returns for each strategy at each sample: (batch, 8)
        strategy_rets = self.all_strategy_returns[sample_indices]

        # Weighted combination: (batch, 8) * (batch, 8) -> (batch,)
        blended_returns = (strategy_weights * strategy_rets).sum(dim=-1)

        return blended_returns

    def get_direct_return(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate return for direct allocation (SPY%, TLT%, Cash%).

        Args:
            allocations: (batch, 3) SPY/TLT/Cash percentages (sum to 1)
            sample_indices: (batch,) time indices

        Returns:
            returns: (batch,) cumulative returns
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        spy_alloc = allocations[:, 0]
        tlt_alloc = allocations[:, 1]
        # cash_alloc = allocations[:, 2]  # Cash contributes ~0 return

        spy_rets = self.spy_cumulative[sample_indices]
        tlt_rets = self.tlt_cumulative[sample_indices]

        # Weighted return
        portfolio_returns = spy_alloc * spy_rets + tlt_alloc * tlt_rets

        return portfolio_returns

    def get_blended_reward(
        self,
        strategy_weights: torch.Tensor,
        sample_indices: torch.Tensor,
        reward_type: str = 'relative'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get reward for blended strategy allocation.

        Args:
            strategy_weights: (batch, 8) blend weights
            sample_indices: (batch,) time indices
            reward_type: 'relative' (vs best discrete) or 'absolute' (raw return)

        Returns:
            rewards: (batch,) reward values
            info: Dict with additional metrics
        """
        blended_returns = self.get_blended_return(strategy_weights, sample_indices)
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        info = {'returns': blended_returns}

        if reward_type == 'absolute':
            # Raw return scaled to reasonable range
            reward = blended_returns * 10  # Scale 1% return to 0.1 reward

        else:  # relative
            # Compare to discrete strategies
            best_discrete, _ = self.all_strategy_returns[sample_indices].max(dim=1)
            worst_discrete, _ = self.all_strategy_returns[sample_indices].min(dim=1)
            mean_discrete = self.all_strategy_returns[sample_indices].mean(dim=1)

            range_returns = best_discrete - worst_discrete + 1e-8

            # Normalized: how close to best discrete?
            reward = 2 * (blended_returns - worst_discrete) / range_returns - 1

            # Bonus for beating best discrete strategy
            beat_best = (blended_returns > best_discrete).float()
            reward = reward + 0.5 * beat_best

            # Additional bonus for beating mean by large margin
            above_mean = (blended_returns > mean_discrete + 0.01).float()
            reward = reward + 0.1 * above_mean

            info['best_discrete_return'] = best_discrete
            info['beat_best'] = beat_best

        return reward, info

    def get_direct_reward(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor,
        reward_type: str = 'relative'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get reward for direct allocation.

        Args:
            allocations: (batch, 3) SPY/TLT/Cash percentages
            sample_indices: (batch,) time indices
            reward_type: 'relative' or 'absolute'

        Returns:
            rewards: (batch,) reward values
            info: Dict with additional metrics
        """
        direct_returns = self.get_direct_return(allocations, sample_indices)
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        info = {'returns': direct_returns, 'allocations': allocations}

        if reward_type == 'absolute':
            reward = direct_returns * 10

        else:  # relative
            best_discrete, _ = self.all_strategy_returns[sample_indices].max(dim=1)
            worst_discrete, _ = self.all_strategy_returns[sample_indices].min(dim=1)

            range_returns = best_discrete - worst_discrete + 1e-8
            reward = 2 * (direct_returns - worst_discrete) / range_returns - 1

            beat_best = (direct_returns > best_discrete).float()
            reward = reward + 0.5 * beat_best

            info['best_discrete_return'] = best_discrete
            info['beat_best'] = beat_best

        return reward, info

    def get_optimal_allocation(
        self,
        sample_indices: torch.Tensor,
        mode: str = 'discrete'
    ) -> torch.Tensor:
        """
        Get optimal allocation at each time index.

        Args:
            sample_indices: (batch,) time indices
            mode: 'discrete' (best strategy idx), 'direct' (optimal SPY/TLT/Cash)

        Returns:
            For 'discrete': (batch,) best strategy indices
            For 'direct': (batch, 3) optimal allocations
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        if mode == 'discrete':
            return self.all_strategy_returns[sample_indices].argmax(dim=1)

        elif mode == 'direct':
            # Find optimal allocation by comparing SPY vs TLT
            spy_rets = self.spy_cumulative[sample_indices]
            tlt_rets = self.tlt_cumulative[sample_indices]

            batch_size = len(sample_indices)
            allocations = torch.zeros(batch_size, 3, device=self.device)

            # Simple heuristic: go 100% to best performing asset
            spy_better = spy_rets > tlt_rets
            allocations[spy_better, 0] = 1.0  # 100% SPY
            allocations[~spy_better, 1] = 1.0  # 100% TLT

            return allocations

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_metrics(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive metrics for allocations.

        Args:
            allocations: (batch, 3) SPY/TLT/Cash
            sample_indices: (batch,) time indices

        Returns:
            Dict with returns, vs_best, vs_mean, allocation stats
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        direct_returns = self.get_direct_return(allocations, sample_indices)
        best_discrete, best_idx = self.all_strategy_returns[sample_indices].max(dim=1)
        worst_discrete, _ = self.all_strategy_returns[sample_indices].min(dim=1)
        mean_discrete = self.all_strategy_returns[sample_indices].mean(dim=1)

        return {
            'returns': direct_returns,
            'best_discrete_return': best_discrete,
            'worst_discrete_return': worst_discrete,
            'mean_discrete_return': mean_discrete,
            'best_strategy_idx': best_idx,
            'vs_best': direct_returns - best_discrete,
            'vs_mean': direct_returns - mean_discrete,
            'beat_best_rate': (direct_returns > best_discrete).float().mean(),
            'beat_mean_rate': (direct_returns > mean_discrete).float().mean(),
            'spy_allocation': allocations[:, 0],
            'tlt_allocation': allocations[:, 1],
            'cash_allocation': allocations[:, 2],
        }


class BatchContinuousSimulator(ContinuousStrategySimulator):
    """
    Alias for ContinuousStrategySimulator with batch-optimized methods.

    Provides drop-in replacement for BatchStrategySimulator with
    additional continuous allocation support.
    """

    def get_reward(
        self,
        action: torch.Tensor,
        sample_indices: torch.Tensor,
        mode: str = 'discrete'
    ) -> torch.Tensor:
        """
        Unified reward interface for any action type.

        Args:
            action: Strategy indices (batch,), weights (batch,8), or allocs (batch,3)
            sample_indices: (batch,) time indices
            mode: 'discrete', 'blended', or 'direct'

        Returns:
            rewards: (batch,)
        """
        if mode == 'discrete':
            return self.get_discrete_reward(action, sample_indices)

        elif mode == 'blended':
            reward, _ = self.get_blended_reward(action, sample_indices)
            return reward

        elif mode == 'direct':
            reward, _ = self.get_direct_reward(action, sample_indices)
            return reward

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_optimal_strategy(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """Backward compatible: get best discrete strategy."""
        return self.get_optimal_allocation(sample_indices, mode='discrete')


def create_continuous_simulator(
    spy_returns: np.ndarray,
    tlt_returns: np.ndarray,
    horizon_days: int = 5,
    device: str = 'cpu'
) -> BatchContinuousSimulator:
    """Factory function for continuous simulator."""
    return BatchContinuousSimulator(
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        horizon_days=horizon_days,
        device=device
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing ContinuousStrategySimulator...")

    # Generate fake returns
    np.random.seed(42)
    n_days = 1000
    spy_returns = np.random.randn(n_days) * 0.01 + 0.0003
    tlt_returns = np.random.randn(n_days) * 0.005 + 0.0001

    sim = create_continuous_simulator(
        spy_returns, tlt_returns, horizon_days=5, device='cpu'
    )

    print(f"Samples: {sim.n_samples}")

    # Test discrete
    batch = 32
    indices = torch.randint(0, sim.n_samples, (batch,))
    strategies = torch.randint(0, 8, (batch,))

    reward = sim.get_reward(strategies, indices, mode='discrete')
    print(f"Discrete reward: mean={reward.mean():.3f}, std={reward.std():.3f}")

    # Test blended
    weights = F.softmax(torch.randn(batch, 8), dim=-1)
    reward = sim.get_reward(weights, indices, mode='blended')
    print(f"Blended reward: mean={reward.mean():.3f}, std={reward.std():.3f}")

    # Test direct
    allocations = F.softmax(torch.randn(batch, 3), dim=-1)
    reward = sim.get_reward(allocations, indices, mode='direct')
    print(f"Direct reward: mean={reward.mean():.3f}, std={reward.std():.3f}")

    # Test metrics
    metrics = sim.compute_metrics(allocations, indices)
    print(f"\nMetrics:")
    print(f"  Beat best rate: {metrics['beat_best_rate'].item()*100:.1f}%")
    print(f"  Beat mean rate: {metrics['beat_mean_rate'].item()*100:.1f}%")
    print(f"  Avg SPY alloc: {metrics['spy_allocation'].mean().item()*100:.1f}%")

    print("\n[OK] ContinuousSimulator test passed!")
