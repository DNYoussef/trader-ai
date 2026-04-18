"""
Objective Functions for Strategy Validation

Provides objective functions used by MCPT to evaluate strategy performance.
These functions take strategy returns and produce a single scalar score.

Key functions:
- profit_factor: Sum of gains / sum of losses
- sharpe_ratio: Risk-adjusted return (annualized)
- sortino_ratio: Downside-risk adjusted return
- calmar_ratio: Return / max drawdown
- max_drawdown: Maximum peak-to-trough decline

Performance: Uses Numba JIT compilation for 5-15x speedup on numerical operations.
"""

import numpy as np
from typing import Optional, Dict, Callable

# Import JIT-compiled numerical kernels for performance
from .objectives_numba import (
    profit_factor_core,
    sharpe_ratio_core,
    sortino_ratio_core,
    max_drawdown_core,
    max_drawdown_from_returns_core,
    ulcer_index_core,
    win_rate_core,
    expectancy_core,
    calmar_ratio_core,
)


def profit_factor(strategy_returns: np.ndarray) -> float:
    """
    Calculate profit factor: sum of gains / sum of losses.

    Profit factor is the primary metric for MCPT validation because:
    1. It's robust to outliers (unlike mean return)
    2. It captures the gain/loss asymmetry
    3. Values > 1.5 indicate a potentially tradeable strategy
    4. Values > 2.0 indicate strong edge

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Profit factor (1e6 if no losses, 0 if no gains)
    """
    return profit_factor_core(np.asarray(strategy_returns, dtype=np.float64))


def sharpe_ratio(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = sqrt(252) * (mean_excess_return) / std(excess_returns)

    Args:
        strategy_returns: Array of per-bar returns
        risk_free_rate: Annual risk-free rate (default: 0)
        annualization_factor: Bars per year (default: 252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    return sharpe_ratio_core(
        np.asarray(strategy_returns, dtype=np.float64),
        float(risk_free_rate),
        float(annualization_factor),
    )


def sortino_ratio(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Calculate Sortino ratio (downside-only volatility).

    Unlike Sharpe, Sortino only penalizes downside volatility,
    which better captures asymmetric return profiles.

    Args:
        strategy_returns: Array of per-bar returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Bars per year

    Returns:
        Annualized Sortino ratio
    """
    return sortino_ratio_core(
        np.asarray(strategy_returns, dtype=np.float64),
        float(risk_free_rate),
        float(annualization_factor),
    )


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown (peak-to-trough decline).

    Args:
        equity_curve: Cumulative equity curve

    Returns:
        Maximum drawdown as positive percentage (e.g., 0.20 for 20%)
    """
    return max_drawdown_core(np.asarray(equity_curve, dtype=np.float64))


def max_drawdown_from_returns(strategy_returns: np.ndarray) -> float:
    """
    Calculate max drawdown from returns array.

    Convenience function that builds equity curve first.

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Maximum drawdown as positive percentage
    """
    return max_drawdown_from_returns_core(np.asarray(strategy_returns, dtype=np.float64))


def calmar_ratio(
    strategy_returns: np.ndarray,
    annualization_factor: float = 252.0,
) -> float:
    """
    Calculate Calmar ratio: annualized return (CAGR) / max drawdown.

    Calmar is useful for strategies with infrequent large drawdowns.

    Args:
        strategy_returns: Array of per-bar returns (log returns)
        annualization_factor: Bars per year

    Returns:
        Calmar ratio
    """
    return calmar_ratio_core(
        np.asarray(strategy_returns, dtype=np.float64),
        float(annualization_factor),
    )


def win_rate(strategy_returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Win rate as percentage (0-1)
    """
    return win_rate_core(np.asarray(strategy_returns, dtype=np.float64))


def expectancy(strategy_returns: np.ndarray) -> float:
    """
    Calculate expectancy (average return per trade).

    Expectancy = (Win_Rate * Avg_Win) - (Loss_Rate * Avg_Loss)

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Expectancy per trade
    """
    return expectancy_core(np.asarray(strategy_returns, dtype=np.float64))


def recovery_factor(strategy_returns: np.ndarray) -> float:
    """
    Calculate recovery factor: total return / max drawdown.

    Measures how well the strategy recovers from drawdowns.

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Recovery factor
    """
    if len(strategy_returns) < 2:
        return 0.0

    total_return = np.sum(strategy_returns)
    mdd = max_drawdown_from_returns(strategy_returns)

    if mdd == 0:
        return float('inf') if total_return > 0 else 0.0

    return total_return / mdd


def ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (measures magnitude and duration of drawdowns).

    Lower is better. Ulcer Index < 5% is considered good.

    Args:
        equity_curve: Cumulative equity curve

    Returns:
        Ulcer Index as percentage
    """
    return ulcer_index_core(np.asarray(equity_curve, dtype=np.float64))


def get_objective_function(name: str) -> Callable[[np.ndarray], float]:
    """
    Get objective function by name.

    Args:
        name: One of 'profit_factor', 'sharpe', 'sortino', 'calmar', 'win_rate'

    Returns:
        Objective function that takes strategy_returns and returns float
    """
    objectives = {
        'profit_factor': profit_factor,
        'sharpe': sharpe_ratio,
        'sharpe_ratio': sharpe_ratio,
        'sortino': sortino_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar': calmar_ratio,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
    }

    if name not in objectives:
        raise ValueError(f"Unknown objective: {name}. Available: {list(objectives.keys())}")

    return objectives[name]


def compute_all_metrics(strategy_returns: np.ndarray) -> Dict[str, float]:
    """
    Compute all performance metrics for a strategy.

    Args:
        strategy_returns: Array of per-bar returns

    Returns:
        Dict with all metrics
    """
    # Build equity curve
    cumulative = np.exp(np.cumsum(strategy_returns))
    equity = np.concatenate([[1.0], cumulative])

    return {
        'total_return': float(np.sum(strategy_returns)),
        'profit_factor': float(profit_factor(strategy_returns)),
        'sharpe_ratio': float(sharpe_ratio(strategy_returns)),
        'sortino_ratio': float(sortino_ratio(strategy_returns)),
        'calmar_ratio': float(calmar_ratio(strategy_returns)),
        'max_drawdown': float(max_drawdown(equity)),
        'win_rate': float(win_rate(strategy_returns)),
        'expectancy': float(expectancy(strategy_returns)),
        'recovery_factor': float(recovery_factor(strategy_returns)),
        'ulcer_index': float(ulcer_index(equity)),
        'n_trades': int(np.sum(strategy_returns != 0)),
    }


if __name__ == "__main__":
    # Test objective functions
    print("=== Objective Functions Test ===")

    np.random.seed(42)

    # Simulate strategy returns
    n_bars = 252  # 1 year of daily data

    # Strategy with edge (slight positive bias)
    returns_with_edge = np.random.randn(n_bars) * 0.01 + 0.0002  # 0.02% daily edge

    # Random strategy (no edge)
    returns_random = np.random.randn(n_bars) * 0.01

    print("\nStrategy with edge:")
    metrics = compute_all_metrics(returns_with_edge)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nRandom strategy:")
    metrics = compute_all_metrics(returns_random)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Test Complete ===")
