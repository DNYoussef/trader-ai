"""
Graduated Risk Simulator

Risk scaling based on portfolio size:
- SMALL portfolios ($200-500): Aggressive - grow fast, accept volatility
- MEDIUM portfolios ($500-2000): Balanced - moderate risk
- LARGE portfolios ($2000+): Conservative - protect gains

The idea: You can afford to lose when small, can't afford to lose when big.
"""

import logging
from typing import Dict, Tuple
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tiers based on portfolio value."""
    AGGRESSIVE = "aggressive"   # $200-500: Grow fast
    MODERATE = "moderate"       # $500-1000: Balanced
    CONSERVATIVE = "conservative"  # $1000-2000: Careful
    PRESERVATION = "preservation"  # $2000+: Protect gains


# Risk parameters by tier
TIER_CONFIGS = {
    RiskTier.AGGRESSIVE: {
        'capital_range': (200, 500),
        'target_spy': 0.80,      # 80% stocks
        'target_tlt': 0.15,      # 15% bonds
        'target_cash': 0.05,     # 5% cash
        'max_spy': 0.90,
        'min_spy': 0.60,
        'loss_aversion': 1.2,    # Low loss aversion
        'description': 'Aggressive growth - maximize upside'
    },
    RiskTier.MODERATE: {
        'capital_range': (500, 1000),
        'target_spy': 0.65,      # 65% stocks
        'target_tlt': 0.25,      # 25% bonds
        'target_cash': 0.10,     # 10% cash
        'max_spy': 0.75,
        'min_spy': 0.50,
        'loss_aversion': 1.5,    # Moderate loss aversion
        'description': 'Balanced growth - controlled risk'
    },
    RiskTier.CONSERVATIVE: {
        'capital_range': (1000, 2000),
        'target_spy': 0.50,      # 50% stocks
        'target_tlt': 0.30,      # 30% bonds
        'target_cash': 0.20,     # 20% cash
        'max_spy': 0.60,
        'min_spy': 0.40,
        'loss_aversion': 2.0,    # High loss aversion
        'description': 'Conservative - protect gains'
    },
    RiskTier.PRESERVATION: {
        'capital_range': (2000, float('inf')),
        'target_spy': 0.40,      # 40% stocks
        'target_tlt': 0.35,      # 35% bonds
        'target_cash': 0.25,     # 25% cash
        'max_spy': 0.50,
        'min_spy': 0.30,
        'loss_aversion': 2.5,    # Very high loss aversion
        'description': 'Wealth preservation - minimize downside'
    },
}


def get_risk_tier(capital: float) -> RiskTier:
    """Determine risk tier based on capital."""
    if capital < 500:
        return RiskTier.AGGRESSIVE
    elif capital < 1000:
        return RiskTier.MODERATE
    elif capital < 2000:
        return RiskTier.CONSERVATIVE
    else:
        return RiskTier.PRESERVATION


def get_tier_config(capital: float) -> dict:
    """Get configuration for capital level."""
    tier = get_risk_tier(capital)
    return TIER_CONFIGS[tier]


def interpolate_risk_params(capital: float) -> dict:
    """
    Smoothly interpolate risk parameters based on capital.

    Returns continuous values instead of discrete tiers.
    """
    # Define breakpoints
    breakpoints = [200, 500, 1000, 2000, 5000]

    # SPY allocation decreases as capital grows
    spy_targets = [0.80, 0.65, 0.50, 0.40, 0.35]

    # TLT allocation increases
    tlt_targets = [0.15, 0.25, 0.30, 0.35, 0.35]

    # Cash increases
    cash_targets = [0.05, 0.10, 0.20, 0.25, 0.30]

    # Loss aversion increases
    loss_aversion = [1.2, 1.5, 2.0, 2.5, 3.0]

    # Interpolate
    target_spy = np.interp(capital, breakpoints, spy_targets)
    target_tlt = np.interp(capital, breakpoints, tlt_targets)
    target_cash = np.interp(capital, breakpoints, cash_targets)
    loss_av = np.interp(capital, breakpoints, loss_aversion)

    # Normalize to sum to 1
    total = target_spy + target_tlt + target_cash

    return {
        'target_spy': target_spy / total,
        'target_tlt': target_tlt / total,
        'target_cash': target_cash / total,
        'loss_aversion': loss_av,
        'capital': capital,
        'tier': get_risk_tier(capital).value,
    }


class GraduatedRiskSimulator:
    """
    Simulator with portfolio-size-based risk graduation.

    The key insight: When you have $200, losing 50% means losing $100.
    When you have $5000, losing 50% means losing $2500.

    So we should be:
    - Aggressive when small (grow fast, losses are small in absolute terms)
    - Conservative when large (protect gains, losses hurt more)
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        horizon_days: int = 5,
        device: str = 'cpu',
        initial_capital: float = 200.0,
    ):
        self.horizon_days = horizon_days
        self.device = device
        self.n_samples = len(spy_returns) - horizon_days
        self.current_capital = initial_capital

        # Store returns
        self.spy_returns = torch.tensor(spy_returns, dtype=torch.float32, device=device)
        self.tlt_returns = torch.tensor(tlt_returns, dtype=torch.float32, device=device)

        # Pre-compute cumulative returns
        self._precompute_returns()

        logger.info(f"GraduatedRiskSimulator: {self.n_samples} samples, "
                   f"initial_capital=${initial_capital}")

    def _precompute_returns(self):
        """Pre-compute cumulative returns."""
        self.spy_cumulative = torch.zeros(self.n_samples, device=self.device)
        self.tlt_cumulative = torch.zeros(self.n_samples, device=self.device)

        for i in range(self.n_samples):
            spy_ret = self.spy_returns[i:i+self.horizon_days]
            tlt_ret = self.tlt_returns[i:i+self.horizon_days]

            self.spy_cumulative[i] = torch.prod(1 + spy_ret) - 1
            self.tlt_cumulative[i] = torch.prod(1 + tlt_ret) - 1

    def update_capital(self, new_capital: float):
        """Update current capital level."""
        old_tier = get_risk_tier(self.current_capital)
        new_tier = get_risk_tier(new_capital)

        self.current_capital = new_capital

        if old_tier != new_tier:
            logger.info(f"Risk tier changed: {old_tier.value} -> {new_tier.value} "
                       f"(capital: ${new_capital:.2f})")

    def get_target_allocation(self, capital: float = None) -> torch.Tensor:
        """Get target allocation for current/given capital level."""
        cap = capital if capital is not None else self.current_capital
        params = interpolate_risk_params(cap)

        return torch.tensor([
            params['target_spy'],
            params['target_tlt'],
            params['target_cash']
        ], device=self.device, dtype=torch.float32)

    def get_risk_params(self, capital: float = None) -> dict:
        """Get all risk parameters for capital level."""
        cap = capital if capital is not None else self.current_capital
        return interpolate_risk_params(cap)

    def apply_risk_constraints(
        self,
        allocations: torch.Tensor,
        capital: float = None
    ) -> torch.Tensor:
        """
        Apply risk constraints based on capital level.

        Pushes allocation toward target for the capital tier.
        """
        cap = capital if capital is not None else self.current_capital
        params = interpolate_risk_params(cap)

        spy = allocations[:, 0]
        tlt = allocations[:, 1]
        cash = allocations[:, 2]

        # Get tier constraints
        tier_config = get_tier_config(cap)

        # Clamp SPY to tier range
        spy = spy.clamp(min=tier_config['min_spy'], max=tier_config['max_spy'])

        # Renormalize
        total = spy + tlt + cash
        spy = spy / total
        tlt = tlt / total
        cash = cash / total

        return torch.stack([spy, tlt, cash], dim=-1)

    def get_reward(
        self,
        allocations: torch.Tensor,
        sample_indices: torch.Tensor,
        capital: float = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate reward with graduated risk penalties.

        Key features:
        - Loss aversion scales with capital (higher capital = more loss averse)
        - Bonus for staying near target allocation
        - Extra penalty for big drawdowns at high capital
        """
        cap = capital if capital is not None else self.current_capital
        params = interpolate_risk_params(cap)

        sample_indices = torch.clamp(sample_indices, 0, self.n_samples - 1)

        # Get returns
        spy_ret = self.spy_cumulative[sample_indices]
        tlt_ret = self.tlt_cumulative[sample_indices]

        # Portfolio return
        portfolio_return = (
            allocations[:, 0] * spy_ret +
            allocations[:, 1] * tlt_ret
        )

        # === GRADUATED LOSS AVERSION ===
        is_loss = portfolio_return < 0
        loss_multiplier = torch.where(
            is_loss,
            torch.tensor(params['loss_aversion'], device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        base_reward = portfolio_return * loss_multiplier * 10

        # === ALLOCATION ALIGNMENT BONUS ===
        # Reward for being close to target allocation
        target = self.get_target_allocation(cap)
        allocation_diff = torch.abs(allocations - target.unsqueeze(0)).sum(dim=-1)
        alignment_bonus = (1 - allocation_diff) * 0.1

        # === BIG LOSS PENALTY (scales with capital) ===
        # At high capital, big losses are especially bad
        drawdown_threshold = 0.03 - (cap / 10000) * 0.01  # Lower threshold at higher capital
        drawdown_threshold = max(0.01, drawdown_threshold)  # Min 1%

        big_loss = portfolio_return < -drawdown_threshold
        drawdown_penalty = torch.where(
            big_loss,
            torch.tensor(0.5 + cap / 2000, device=self.device),  # Scales with capital
            torch.tensor(0.0, device=self.device)
        )

        # Total reward
        reward = base_reward + alignment_bonus - drawdown_penalty

        info = {
            'portfolio_return': portfolio_return,
            'base_reward': base_reward,
            'alignment_bonus': alignment_bonus,
            'drawdown_penalty': drawdown_penalty,
            'is_loss': is_loss.float(),
            'loss_aversion': params['loss_aversion'],
            'tier': params['tier'],
            'target_allocation': target,
        }

        return reward, info

    def simulate_graduated_portfolio(
        self,
        allocations_fn,
        initial_capital: float = 200.0,
        n_days: int = 252,
        rebalance_freq: int = 5,
    ) -> dict:
        """
        Simulate portfolio with graduated risk over time.

        As capital grows, risk automatically reduces.

        Args:
            allocations_fn: Function(features, capital) -> allocations
            initial_capital: Starting capital
            n_days: Number of days to simulate
            rebalance_freq: Days between rebalances

        Returns:
            Dict with simulation results
        """
        capital = initial_capital
        history = []

        for day in range(n_days):
            # Get current risk tier
            params = interpolate_risk_params(capital)
            target = self.get_target_allocation(capital)

            # Get day's returns
            idx = min(day, self.n_samples - 1)
            spy_ret = self.spy_cumulative[idx].item() / self.horizon_days  # Daily approx
            tlt_ret = self.tlt_cumulative[idx].item() / self.horizon_days

            # Current allocation (start at target, then drift)
            if day == 0 or day % rebalance_freq == 0:
                allocation = target.numpy()

            # Apply returns
            daily_return = allocation[0] * spy_ret + allocation[1] * tlt_ret
            capital *= (1 + daily_return)

            history.append({
                'day': day,
                'capital': capital,
                'return': daily_return * 100,
                'tier': params['tier'],
                'spy_pct': allocation[0] * 100,
                'tlt_pct': allocation[1] * 100,
                'cash_pct': allocation[2] * 100,
                'loss_aversion': params['loss_aversion'],
            })

        return {
            'history': history,
            'final_capital': capital,
            'total_return': (capital / initial_capital - 1) * 100,
        }


def create_graduated_simulator(
    spy_returns: np.ndarray,
    tlt_returns: np.ndarray,
    horizon_days: int = 5,
    device: str = 'cpu',
    initial_capital: float = 200.0,
) -> GraduatedRiskSimulator:
    """Factory function."""
    return GraduatedRiskSimulator(
        spy_returns=spy_returns,
        tlt_returns=tlt_returns,
        horizon_days=horizon_days,
        device=device,
        initial_capital=initial_capital,
    )


def print_risk_schedule():
    """Print the graduated risk schedule."""
    print("\n" + "=" * 70)
    print("GRADUATED RISK SCHEDULE")
    print("=" * 70)
    print(f"{'Capital':>12} {'Tier':<15} {'SPY%':>8} {'TLT%':>8} {'Cash%':>8} {'Loss Av':>10}")
    print("-" * 70)

    for cap in [200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]:
        params = interpolate_risk_params(cap)
        print(f"${cap:>10,.0f}  {params['tier']:<15} "
              f"{params['target_spy']*100:>7.1f}% "
              f"{params['target_tlt']*100:>7.1f}% "
              f"{params['target_cash']*100:>7.1f}% "
              f"{params['loss_aversion']:>9.2f}x")

    print("=" * 70)
    print("\nPHILOSOPHY:")
    print("  - Small capital: Take risks to grow (losses are small in $ terms)")
    print("  - Large capital: Protect gains (losses hurt more in $ terms)")
    print("=" * 70)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print_risk_schedule()
