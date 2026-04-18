"""
Loss-Averse Strategy Simulator

For barbell strategy where AI controls 20% "risky" portion:
- Asymmetric rewards: losses hurt 2x more than gains help
- Anti-paralysis: penalty for sitting in too much cash
- Drawdown penalty: extra punishment for large losses

This creates a model that:
1. Avoids big losses (loss averse)
2. Still takes positions (not paralyzed)
3. Captures upside when confident
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch

from .strategy_simulator import STRATEGY_ALLOCATIONS, STRATEGY_NAMES

logger = logging.getLogger(__name__)


class LossAverseSimulator:
    """
    Simulator with loss-averse reward shaping.

    Reward structure:
    - Gains: reward = return * 1.0
    - Losses: reward = return * 2.0 (hurts twice as much)
    - Cash penalty: -0.1 for each 10% in cash above 20%
    - Drawdown penalty: extra -0.5 for losses > 2%

    This encourages the model to:
    - Avoid losses (loss aversion)
    - Stay invested (anti-paralysis)
    - Be defensive but not frozen
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        horizon_days: int = 5,
        device: str = 'cpu',
        loss_aversion: float = 2.0,      # Losses hurt 2x more
        cash_penalty_rate: float = 0.01,  # Penalty per 10% excess cash
        max_cash_free: float = 0.20,      # Cash up to 20% is free
        drawdown_threshold: float = 0.02, # Extra penalty for >2% loss
        drawdown_penalty: float = 0.5,    # How much extra penalty
    ):
        self.horizon_days = horizon_days
        self.device = device
        self.n_samples = len(spy_returns) - horizon_days

        self.loss_aversion = loss_aversion
        self.cash_penalty_rate = cash_penalty_rate
        self.max_cash_free = max_cash_free
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty

        # Store returns
        self.spy_returns = torch.tensor(spy_returns, dtype=torch.float32, device=device)
        self.tlt_returns = torch.tensor(tlt_returns, dtype=torch.float32, device=device)

        # Pre-compute cumulative returns
        self._precompute_returns()

        logger.info(f"LossAverseSimulator: {self.n_samples} samples, "
                   f"loss_aversion={loss_aversion}, cash_penalty={cash_penalty_rate}")

    def _precompute_returns(self):
        """Pre-compute cumulative returns for SPY and TLT."""
        self.spy_cumulative = torch.zeros(self.n_samples, device=self.device)
        self.tlt_cumulative = torch.zeros(self.n_samples, device=self.device)

        for i in range(self.n_samples):
            spy_ret = self.spy_returns[i:i+self.horizon_days]
            tlt_ret = self.tlt_returns[i:i+self.horizon_days]

            self.spy_cumulative[i] = torch.prod(1 + spy_ret) - 1
            self.tlt_cumulative[i] = torch.prod(1 + tlt_ret) - 1

    def get_portfolio_return(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate portfolio return for given allocations.

        Args:
            allocations: (batch, 3) SPY/TLT/Cash percentages
            sample_indices: (batch,) time indices

        Returns:
            returns: (batch,) portfolio returns
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        spy_alloc = allocations[:, 0]
        tlt_alloc = allocations[:, 1]
        # cash_alloc = allocations[:, 2]  # Cash earns ~0

        spy_rets = self.spy_cumulative[sample_indices]
        tlt_rets = self.tlt_cumulative[sample_indices]

        portfolio_returns = spy_alloc * spy_rets + tlt_alloc * tlt_rets

        return portfolio_returns

    def get_reward(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss-averse reward.

        Args:
            allocations: (batch, 3) SPY/TLT/Cash percentages
            sample_indices: (batch,) time indices

        Returns:
            rewards: (batch,) reward values
            info: Dict with breakdown
        """
        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        # Get raw portfolio return
        portfolio_return = self.get_portfolio_return(allocations, sample_indices)

        # === ASYMMETRIC REWARD (Loss Aversion) ===
        # Gains: multiply by 1.0
        # Losses: multiply by loss_aversion (2.0)
        is_loss = portfolio_return < 0
        asymmetric_multiplier = torch.where(
            is_loss,
            torch.tensor(self.loss_aversion, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        base_reward = portfolio_return * asymmetric_multiplier * 10  # Scale to reasonable range

        # === CASH PENALTY (Anti-Paralysis) ===
        cash_alloc = allocations[:, 2]
        excess_cash = (cash_alloc - self.max_cash_free).clamp(min=0)
        cash_penalty = excess_cash * self.cash_penalty_rate * 10  # Penalty per excess cash

        # === DRAWDOWN PENALTY (Extra Loss Aversion for Big Losses) ===
        big_loss = portfolio_return < -self.drawdown_threshold
        drawdown_penalty = torch.where(
            big_loss,
            torch.tensor(self.drawdown_penalty, device=self.device),
            torch.tensor(0.0, device=self.device)
        )

        # === TOTAL REWARD ===
        reward = base_reward - cash_penalty - drawdown_penalty

        info = {
            'portfolio_return': portfolio_return,
            'base_reward': base_reward,
            'cash_penalty': cash_penalty,
            'drawdown_penalty': drawdown_penalty,
            'is_loss': is_loss.float(),
            'cash_allocation': cash_alloc,
        }

        return reward, info

    def get_blended_reward(
        self,
        strategy_weights: torch.Tensor,
        sample_indices: torch.Tensor,
        strategy_allocations: Dict[int, Tuple[float, float, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get reward for blended strategy weights.

        Args:
            strategy_weights: (batch, 8) blend weights
            sample_indices: (batch,) time indices

        Returns:
            rewards, info
        """
        if strategy_allocations is None:
            strategy_allocations = STRATEGY_ALLOCATIONS

        device = strategy_weights.device

        # Build allocation matrix
        alloc_matrix = torch.tensor([
            strategy_allocations[i] for i in range(8)
        ], device=device, dtype=torch.float32)

        # Weighted combination: (batch, 8) @ (8, 3) -> (batch, 3)
        allocations = torch.matmul(strategy_weights, alloc_matrix)

        return self.get_reward(allocations, sample_indices)


class BarbellAISimulator(LossAverseSimulator):
    """
    Specialized simulator for 80/20 barbell strategy.

    The AI controls the 20% "risky" portion:
    - Should be more aggressive than pure loss-averse
    - But still avoid catastrophic losses
    - Minimum equity exposure to stay in the game

    Constraints:
    - Min SPY allocation: 30% (of the 20% AI portion)
    - Max Cash: 40% (can't be fully paralyzed)
    - Loss aversion still applies but slightly reduced
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        horizon_days: int = 5,
        device: str = 'cpu',
        min_spy: float = 0.30,   # Minimum 30% SPY
        max_cash: float = 0.40,  # Maximum 40% cash
    ):
        super().__init__(
            spy_returns=spy_returns,
            tlt_returns=tlt_returns,
            horizon_days=horizon_days,
            device=device,
            loss_aversion=1.8,       # Slightly less loss averse (risky portion)
            cash_penalty_rate=0.02,  # Higher penalty for excess cash
            max_cash_free=0.15,      # Only 15% cash is "free"
            drawdown_threshold=0.03, # 3% threshold for big loss
            drawdown_penalty=0.4,
        )

        self.min_spy = min_spy
        self.max_cash = max_cash

        logger.info(f"BarbellAISimulator: min_spy={min_spy}, max_cash={max_cash}")

    def apply_constraints(
        self,
        allocations: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply barbell constraints to allocations.

        Ensures:
        - SPY >= min_spy
        - Cash <= max_cash
        - Normalizes to sum to 1
        """
        spy = allocations[:, 0]
        tlt = allocations[:, 1]
        cash = allocations[:, 2]

        # Enforce minimum SPY
        spy = spy.clamp(min=self.min_spy)

        # Enforce maximum cash
        cash = cash.clamp(max=self.max_cash)

        # Renormalize
        total = spy + tlt + cash
        spy = spy / total
        tlt = tlt / total
        cash = cash / total

        return torch.stack([spy, tlt, cash], dim=-1)

    def get_reward_with_constraints(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get reward after applying constraints."""
        constrained = self.apply_constraints(allocations)
        reward, info = self.get_reward(constrained, sample_indices)
        info['constrained_allocation'] = constrained
        info['original_allocation'] = allocations
        return reward, info


def create_barbell_simulator(
    spy_returns: np.ndarray,
    tlt_returns: np.ndarray,
    horizon_days: int = 5,
    device: str = 'cpu'
) -> BarbellAISimulator:
    """Factory function for barbell AI simulator."""
    return BarbellAISimulator(
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        horizon_days=horizon_days,
        device=device
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing LossAverseSimulator...")

    # Generate fake returns
    np.random.seed(42)
    n_days = 1000
    spy_returns = np.random.randn(n_days) * 0.01 + 0.0003
    tlt_returns = np.random.randn(n_days) * 0.005 + 0.0001

    sim = create_barbell_simulator(spy_returns, tlt_returns, device='cpu')

    # Test allocations
    batch = 4
    allocations = torch.tensor([
        [0.7, 0.2, 0.1],   # Growth
        [0.3, 0.3, 0.4],   # Defensive (high cash - will be penalized)
        [0.5, 0.3, 0.2],   # Balanced
        [0.1, 0.1, 0.8],   # Cash heavy (will be constrained)
    ])

    indices = torch.randint(0, sim.n_samples, (batch,))

    # Without constraints
    reward, info = sim.get_reward(allocations, indices)
    print(f"\nWithout constraints:")
    print(f"  Rewards: {reward.tolist()}")
    print(f"  Cash penalties: {info['cash_penalty'].tolist()}")

    # With constraints
    reward_c, info_c = sim.get_reward_with_constraints(allocations, indices)
    print(f"\nWith constraints:")
    print(f"  Rewards: {reward_c.tolist()}")
    print(f"  Original alloc[3]: {allocations[3].tolist()}")
    print(f"  Constrained[3]:    {info_c['constrained_allocation'][3].tolist()}")

    print("\n[OK] LossAverseSimulator test passed!")
