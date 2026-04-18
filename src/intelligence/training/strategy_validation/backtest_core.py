"""
Backtest Core Computations

Provides fundamental backtesting calculations with realistic assumptions
including slippage, fees, and execution delays.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestAssumptions:
    """
    Realistic backtesting assumptions.

    All values in basis points (bps) unless otherwise noted.
    1 bps = 0.01% = 0.0001
    """
    slippage_bps: float = 5.0       # Market impact slippage
    fee_bps: float = 10.0           # Trading fees (commission + exchange)
    delay_bars: int = 1             # Execution delay in bars
    spread_bps: float = 2.0         # Bid-ask spread
    funding_rate_daily: float = 0.0 # Daily funding rate for leveraged positions

    @property
    def total_cost_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return self.slippage_bps + self.fee_bps + self.spread_bps

    def to_dict(self) -> dict:
        return {
            'slippage_bps': self.slippage_bps,
            'fee_bps': self.fee_bps,
            'delay_bars': self.delay_bars,
            'spread_bps': self.spread_bps,
            'funding_rate_daily': self.funding_rate_daily,
            'total_cost_bps': self.total_cost_bps,
        }


def compute_bar_returns(
    positions: np.ndarray,
    close: np.ndarray,
    assumptions: Optional[BacktestAssumptions] = None,
) -> np.ndarray:
    """
    Compute bar-level returns with realistic cost assumptions.

    Args:
        positions: Position array with values in {-1, 0, +1}
        close: Array of close prices
        assumptions: Backtest cost assumptions

    Returns:
        bar_returns: Array of per-bar returns after costs
    """
    if assumptions is None:
        assumptions = BacktestAssumptions()

    n = len(close)
    if len(positions) != n:
        raise ValueError(f"positions length {len(positions)} != close length {n}")

    # Apply execution delay
    if assumptions.delay_bars > 0:
        delayed_positions = np.zeros_like(positions)
        delayed_positions[assumptions.delay_bars:] = positions[:-assumptions.delay_bars]
        positions = delayed_positions

    # Log returns
    log_returns = np.zeros(n)
    log_returns[1:] = np.diff(np.log(close))

    # Strategy returns (position at bar i affects return at bar i+1)
    strat_returns = np.zeros(n - 1)
    strat_returns = positions[:-1] * log_returns[1:]

    # Transaction costs on position changes
    position_changes = np.abs(np.diff(positions))
    cost_per_change = assumptions.total_cost_bps / 10000.0

    # Apply costs (shift to align with returns)
    costs = np.zeros(n - 1)
    costs[:-1] = position_changes[:-1] * cost_per_change

    # Apply funding cost for held positions
    if assumptions.funding_rate_daily != 0:
        funding_cost = np.abs(positions[:-1]) * (assumptions.funding_rate_daily / 10000.0)
        costs += funding_cost

    return strat_returns - costs


def objective_profit_factor(bar_returns: np.ndarray) -> float:
    """
    Profit factor = sum(gains) / sum(losses).

    Higher is better. Values > 1.5 indicate tradeable edge.
    """
    gains = bar_returns[bar_returns > 0].sum()
    losses = abs(bar_returns[bar_returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def objective_sharpe(
    bar_returns: np.ndarray,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe ratio.

    Higher is better. Values > 1.0 are considered good.
    """
    if len(bar_returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor
    excess_returns = bar_returns - daily_rf

    mean_excess = np.mean(excess_returns)
    std_returns = np.std(bar_returns, ddof=1)

    if std_returns == 0:
        return 0.0

    return np.sqrt(annualization_factor) * mean_excess / std_returns


def objective_sortino(
    bar_returns: np.ndarray,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sortino ratio (downside risk only).

    Better than Sharpe for asymmetric return distributions.
    """
    if len(bar_returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor
    excess_returns = bar_returns - daily_rf

    mean_excess = np.mean(excess_returns)

    downside_returns = bar_returns[bar_returns < 0]
    if len(downside_returns) < 2:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return float('inf') if mean_excess > 0 else 0.0

    return np.sqrt(annualization_factor) * mean_excess / downside_std


def objective_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown (peak-to-trough decline).

    Lower is better. Values < 20% are considered acceptable for most strategies.
    Returns positive value representing the drawdown magnitude.
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return abs(np.min(drawdowns))


def objective_calmar(
    bar_returns: np.ndarray,
    annualization_factor: float = 252.0,
) -> float:
    """
    Calmar ratio = annualized return / max drawdown.

    Higher is better. Captures return efficiency relative to worst loss.
    """
    if len(bar_returns) < 2:
        return 0.0

    # Build equity curve
    cumulative = np.exp(np.cumsum(bar_returns))
    equity = np.concatenate([[1.0], cumulative])

    # Annualized return
    total_return = np.sum(bar_returns)
    n_years = len(bar_returns) / annualization_factor
    ann_return = total_return / n_years if n_years > 0 else 0.0

    # Max drawdown
    max_dd = objective_max_drawdown(equity)

    if max_dd == 0:
        return float('inf') if ann_return > 0 else 0.0

    return ann_return / max_dd


def compute_equity_curve(
    bar_returns: np.ndarray,
    initial_capital: float = 1.0,
) -> np.ndarray:
    """
    Compute cumulative equity curve from bar returns.

    Args:
        bar_returns: Array of per-bar returns
        initial_capital: Starting capital (default: 1.0 for normalized)

    Returns:
        equity: Cumulative equity curve
    """
    cumulative = np.exp(np.cumsum(bar_returns))
    return np.concatenate([[initial_capital], initial_capital * cumulative])


def compute_trade_statistics(
    bar_returns: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """
    Compute detailed trade statistics.

    Args:
        bar_returns: Array of per-bar returns
        positions: Array of positions

    Returns:
        Dict with trade statistics
    """
    # Filter to active bars (non-zero returns when in position)
    active_mask = positions[:-1] != 0
    active_returns = bar_returns[active_mask] if len(bar_returns) == len(positions) - 1 else bar_returns

    if len(active_returns) == 0:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'expectancy': 0.0,
        }

    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]

    n_trades = len(active_returns)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    largest_win = np.max(wins) if len(wins) > 0 else 0.0
    largest_loss = np.min(losses) if len(losses) > 0 else 0.0

    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss),
        'expectancy': float(expectancy),
    }


if __name__ == "__main__":
    # Test backtest core
    print("=== Backtest Core Test ===")

    np.random.seed(42)

    # Generate synthetic data
    n = 252
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    positions = np.random.choice([-1, 0, 1], n, p=[0.1, 0.7, 0.2])

    # Test with different assumptions
    assumptions = BacktestAssumptions(
        slippage_bps=5,
        fee_bps=10,
        delay_bars=1,
        spread_bps=2,
    )

    print(f"Assumptions: {assumptions.to_dict()}")

    returns = compute_bar_returns(positions, close, assumptions)
    print(f"\nBar returns shape: {returns.shape}")
    print(f"Total return: {np.sum(returns):.4f}")

    print(f"\nObjectives:")
    print(f"  Profit Factor: {objective_profit_factor(returns):.3f}")
    print(f"  Sharpe Ratio: {objective_sharpe(returns):.3f}")
    print(f"  Sortino Ratio: {objective_sortino(returns):.3f}")

    equity = compute_equity_curve(returns)
    print(f"  Max Drawdown: {objective_max_drawdown(equity):.1%}")
    print(f"  Calmar Ratio: {objective_calmar(returns):.3f}")

    stats = compute_trade_statistics(returns, positions)
    print(f"\nTrade Statistics: {stats}")

    print("\n=== Test Complete ===")
