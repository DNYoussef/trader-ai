"""
Numba-compiled numerical kernels for signal interface operations.

This module provides JIT-compiled implementations of computationally
intensive numerical operations used in strategy signal processing.

Expected speedup: 5-15x on array operations and loops.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def strategy_returns_core(
    log_returns: np.ndarray,
    positions: np.ndarray,
    fees_bps: float,
) -> np.ndarray:
    """
    Numba-compiled strategy returns calculation.

    The key formula: returns[i] = positions[i] * log_return[i+1]

    This means the position taken at bar i affects the return from bar i to bar i+1.
    This is the correct way to avoid lookahead bias.

    Args:
        log_returns: Array of close-to-close log returns
        positions: Array of positions {-1, 0, +1} (length = len(log_returns) + 1)
        fees_bps: Transaction costs in basis points (applied on position changes)

    Returns:
        strategy_returns: Array of per-bar returns (length = len(log_returns))
    """
    n = len(log_returns)
    strat_returns = np.empty(n, dtype=np.float64)

    # Shift positions (position at bar i affects return at bar i+1)
    for i in range(n):
        strat_returns[i] = positions[i] * log_returns[i]

    # Apply transaction costs on position changes
    if fees_bps > 0.0:
        fee_rate = fees_bps / 10000.0

        for i in range(1, len(positions)):
            if abs(positions[i] - positions[i - 1]) > 0.001:  # Position changed
                # Apply fee to the corresponding return
                if i - 1 < n:
                    strat_returns[i - 1] -= fee_rate * abs(positions[i] - positions[i - 1])

    return strat_returns


@njit(cache=True)
def equity_curve_core(
    log_returns: np.ndarray,
    positions: np.ndarray,
    initial_capital: float,
    fees_bps: float,
) -> np.ndarray:
    """
    Numba-compiled equity curve calculation.

    Args:
        log_returns: Array of close-to-close log returns
        positions: Array of positions {-1, 0, +1}
        initial_capital: Starting capital (default: 1.0 for normalized)
        fees_bps: Transaction costs in basis points

    Returns:
        equity: Cumulative equity curve (length = len(log_returns) + 1)
    """
    n = len(log_returns)
    returns = strategy_returns_core(log_returns, positions, fees_bps)

    # Cumulative returns (log -> multiply)
    equity = np.empty(n + 1, dtype=np.float64)
    equity[0] = initial_capital

    cumsum = 0.0
    for i in range(n):
        cumsum += returns[i]
        equity[i + 1] = initial_capital * np.exp(cumsum)

    return equity


@njit(cache=True)
def compare_signals_core(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> tuple:
    """
    Numba-compiled signal comparison.

    Args:
        signal_a: First position signal array
        signal_b: Second position signal array

    Returns:
        Tuple of (agreement_count, disagreement_count, signal_a_active, signal_b_active)
    """
    n = len(signal_a)
    agreement = 0
    disagreement = 0
    both_nonzero_count = 0
    signal_a_active = 0
    signal_b_active = 0

    for i in range(n):
        # Count active signals
        if abs(signal_a[i]) > 0.001:
            signal_a_active += 1
        if abs(signal_b[i]) > 0.001:
            signal_b_active += 1

        # Check agreement/disagreement when both non-zero
        if abs(signal_a[i]) > 0.001 and abs(signal_b[i]) > 0.001:
            both_nonzero_count += 1
            # Same sign = agreement
            if signal_a[i] * signal_b[i] > 0.0:
                agreement += 1
            else:
                disagreement += 1

    return (agreement, disagreement, signal_a_active, signal_b_active)
