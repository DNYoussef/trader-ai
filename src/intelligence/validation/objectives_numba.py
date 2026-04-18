"""
Numba-compiled numerical kernels for objective functions.

This module provides JIT-compiled implementations of computationally
intensive numerical operations used in strategy validation.

Expected speedup: 5-15x on numerical loops and array operations.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def profit_factor_core(strategy_returns: np.ndarray) -> float:
    """
    Numba-compiled profit factor calculation.

    Args:
        strategy_returns: Array of per-bar returns (float64)

    Returns:
        Profit factor (gains / losses)
    """
    gains = 0.0
    losses = 0.0

    for ret in strategy_returns:
        if ret > 0.0:
            gains += ret
        elif ret < 0.0:
            losses += abs(ret)

    if losses < 1e-10:
        return 1e6 if gains > 0.0 else 0.0

    return gains / losses


@njit(cache=True)
def sharpe_ratio_core(
    strategy_returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: float,
) -> float:
    """
    Numba-compiled Sharpe ratio calculation.

    Args:
        strategy_returns: Array of per-bar returns (float64)
        risk_free_rate: Annual risk-free rate
        annualization_factor: Bars per year (e.g., 252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    n = len(strategy_returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor

    # Calculate mean and variance of excess returns in one pass
    mean_sum = 0.0
    for i in range(n):
        mean_sum += strategy_returns[i] - daily_rf

    mean_excess = mean_sum / n

    # Calculate variance (using Welch-Satterthwaite correction, ddof=1)
    var_sum = 0.0
    for i in range(n):
        diff = (strategy_returns[i] - daily_rf) - mean_excess
        var_sum += diff * diff

    if n <= 1:
        return 0.0

    variance = var_sum / (n - 1)
    std_excess = np.sqrt(variance)

    if std_excess < 1e-10:
        return 0.0

    return np.sqrt(annualization_factor) * mean_excess / std_excess


@njit(cache=True)
def sortino_ratio_core(
    strategy_returns: np.ndarray,
    risk_free_rate: float,
    annualization_factor: float,
) -> float:
    """
    Numba-compiled Sortino ratio calculation.

    Args:
        strategy_returns: Array of per-bar returns (float64)
        risk_free_rate: Annual risk-free rate
        annualization_factor: Bars per year

    Returns:
        Annualized Sortino ratio
    """
    n = len(strategy_returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor

    # Calculate mean excess return
    mean_sum = 0.0
    for i in range(n):
        mean_sum += strategy_returns[i] - daily_rf
    mean_excess = mean_sum / n

    # Calculate downside deviation (only negative returns)
    downside_count = 0
    downside_sum = 0.0

    for ret in strategy_returns:
        if ret < 0.0:
            downside_sum += ret
            downside_count += 1

    if downside_count < 2:
        return np.inf if mean_excess > 0.0 else 0.0

    downside_mean = downside_sum / downside_count

    # Downside variance
    downside_var = 0.0
    for ret in strategy_returns:
        if ret < 0.0:
            diff = ret - downside_mean
            downside_var += diff * diff

    downside_std = np.sqrt(downside_var / (downside_count - 1))

    if downside_std < 1e-10:
        return np.inf if mean_excess > 0.0 else 0.0

    return np.sqrt(annualization_factor) * mean_excess / downside_std


@njit(cache=True)
def max_drawdown_core(equity_curve: np.ndarray) -> float:
    """
    Numba-compiled maximum drawdown calculation.

    Args:
        equity_curve: Cumulative equity curve (float64)

    Returns:
        Maximum drawdown as positive percentage
    """
    n = len(equity_curve)
    if n < 2:
        return 0.0

    running_max = equity_curve[0]
    max_dd = 0.0

    for i in range(n):
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]

        if running_max > 0.0:
            drawdown = (equity_curve[i] - running_max) / running_max
            if drawdown < max_dd:
                max_dd = drawdown

    return abs(max_dd)


@njit(cache=True)
def max_drawdown_from_returns_core(strategy_returns: np.ndarray) -> float:
    """
    Numba-compiled max drawdown from returns.

    Args:
        strategy_returns: Array of per-bar log returns (float64)

    Returns:
        Maximum drawdown as positive percentage
    """
    n = len(strategy_returns)
    if n < 2:
        return 0.0

    # Build equity curve from log returns
    equity = np.empty(n + 1, dtype=np.float64)
    equity[0] = 1.0

    cumsum = 0.0
    for i in range(n):
        cumsum += strategy_returns[i]
        equity[i + 1] = np.exp(cumsum)

    return max_drawdown_core(equity)


@njit(cache=True)
def ulcer_index_core(equity_curve: np.ndarray) -> float:
    """
    Numba-compiled Ulcer Index calculation.

    Args:
        equity_curve: Cumulative equity curve (float64)

    Returns:
        Ulcer Index as percentage
    """
    n = len(equity_curve)
    if n < 2:
        return 0.0

    running_max = equity_curve[0]
    dd_squared_sum = 0.0

    for i in range(n):
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]

        if running_max > 0.0:
            drawdown_pct = 100.0 * (equity_curve[i] - running_max) / running_max
            dd_squared_sum += drawdown_pct * drawdown_pct

    return np.sqrt(dd_squared_sum / n)


@njit(cache=True)
def win_rate_core(strategy_returns: np.ndarray) -> float:
    """
    Numba-compiled win rate calculation.

    Args:
        strategy_returns: Array of per-bar returns (float64)

    Returns:
        Win rate as percentage (0-1)
    """
    total_trades = 0
    wins = 0

    for ret in strategy_returns:
        if abs(ret) > 1e-10:  # Non-zero return (active trade)
            total_trades += 1
            if ret > 0.0:
                wins += 1

    if total_trades == 0:
        return 0.0

    return float(wins) / float(total_trades)


@njit(cache=True)
def expectancy_core(strategy_returns: np.ndarray) -> float:
    """
    Numba-compiled expectancy calculation.

    Args:
        strategy_returns: Array of per-bar returns (float64)

    Returns:
        Expectancy per trade
    """
    total_trades = 0
    wins = 0
    win_sum = 0.0
    loss_sum = 0.0

    for ret in strategy_returns:
        if abs(ret) > 1e-10:  # Non-zero return (active trade)
            total_trades += 1
            if ret > 0.0:
                wins += 1
                win_sum += ret
            else:
                loss_sum += ret

    if total_trades == 0:
        return 0.0

    avg_win = win_sum / wins if wins > 0 else 0.0
    losses = total_trades - wins
    avg_loss = loss_sum / losses if losses > 0 else 0.0

    win_rate = float(wins) / float(total_trades)
    loss_rate = 1.0 - win_rate

    return win_rate * avg_win + loss_rate * avg_loss


@njit(cache=True)
def calmar_ratio_core(
    strategy_returns: np.ndarray,
    annualization_factor: float,
) -> float:
    """
    Numba-compiled Calmar ratio calculation.

    Args:
        strategy_returns: Array of per-bar log returns (float64)
        annualization_factor: Bars per year

    Returns:
        Calmar ratio (CAGR / max drawdown)
    """
    n = len(strategy_returns)
    if n < 2:
        return 0.0

    # Calculate total compound return
    total_log_return = 0.0
    for ret in strategy_returns:
        total_log_return += ret

    total_compound = np.exp(total_log_return) - 1.0

    # Calculate annualized return (CAGR)
    n_years = n / annualization_factor
    if n_years <= 0.0:
        return 0.0

    ann_return = (total_compound + 1.0) ** (1.0 / n_years) - 1.0

    # Calculate max drawdown
    mdd = max_drawdown_from_returns_core(strategy_returns)

    if mdd < 1e-10:
        return np.inf if ann_return > 0.0 else 0.0

    return ann_return / mdd
