"""
Numba-compiled numerical kernels for bar permutation operations.

This module provides JIT-compiled implementations of computationally
intensive numerical operations used in MCPT bar permutation.

Expected speedup: 5-15x on reconstruction loops.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def reconstruct_ohlc_core(
    perm_bars: np.ndarray,
    relative_open: np.ndarray,
    relative_high: np.ndarray,
    relative_low: np.ndarray,
    relative_close: np.ndarray,
    recon_start: int,
    n_bars: int,
    initial_close: float,
) -> None:
    """
    Numba-compiled OHLC reconstruction from relative movements.

    Modifies perm_bars in-place for maximum performance.

    Args:
        perm_bars: Output array (n_bars, 4) to fill with reconstructed OHLC
        relative_open/high/low/close: Relative movement arrays
        recon_start: Index to start reconstruction
        n_bars: Total number of bars
        initial_close: Close price of the bar before recon_start
    """
    n_recon = n_bars - recon_start

    if n_recon <= 0:
        return

    # Build cumulative gaps for open prices
    gaps = np.empty(n_recon, dtype=np.float64)
    gaps[0] = relative_open[0]

    for i in range(1, n_recon):
        gaps[i] = relative_open[i] + relative_close[i - 1]

    # Cumulative sum for open prices
    open_prices = initial_close + np.cumsum(gaps)

    # Fill in OHLC (vectorized indexing works in Numba)
    for i in range(n_recon):
        idx = recon_start + i
        perm_bars[idx, 0] = open_prices[i]  # Open
        perm_bars[idx, 1] = open_prices[i] + relative_high[i]  # High
        perm_bars[idx, 2] = open_prices[i] + relative_low[i]   # Low
        perm_bars[idx, 3] = open_prices[i] + relative_close[i] # Close
