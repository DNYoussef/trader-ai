"""
Bar Permutation for Monte Carlo Permutation Testing (MCPT)

Vendored from: https://github.com/neurotrader888/mcpt
License: MIT
Author: neurotrader888

This module provides bar-level permutation for OHLC data while preserving
the statistical properties of individual bars (relative O/H/L/C relationships).

Key insight: By shuffling the relative movements rather than raw prices,
we preserve the "shape" of bars while randomizing their sequence.
"""

import numpy as np
import pandas as pd
from typing import List, Union

# Import JIT-compiled reconstruction kernel for performance
from .bar_permute_numba import reconstruct_ohlc_core


def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: int = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate a permuted version of OHLC data.

    The permutation preserves:
    - Statistical properties of individual bars
    - Relative O/H/L/C relationships within each bar
    - Data before start_index (for walk-forward testing)

    The permutation randomizes:
    - The sequence of bar movements
    - Temporal dependencies (destroys autocorrelation)

    Args:
        ohlc: DataFrame with columns ['open', 'high', 'low', 'close']
              OR list of DataFrames for multi-market permutation
        start_index: Index from which to start permuting (default: 0)
                    Data before this index is preserved unchanged.
                    Use this for walk-forward MCPT to keep training data intact.
        seed: Random seed for reproducibility (default: None)

    Returns:
        Permuted DataFrame(s) with same structure as input

    Example:
        # In-sample MCPT (permute all data)
        perm_df = get_permutation(train_df)

        # Walk-forward MCPT (keep first 252 bars unchanged)
        perm_df = get_permutation(full_df, start_index=252)
    """
    assert start_index >= 0, "start_index must be non-negative"

    if seed is not None:
        np.random.seed(seed)

    # Handle single vs multiple markets
    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match across markets"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    # Indices for permutation
    # Start permuting AT start_index (not after it)
    # Example: start_index=252 means bars [0:252) are preserved, bar 252 onward are permuted
    perm_index = start_index  # Correct: Start permuting AT start_index
    # When start_index=0, we skip bar 0 in permutation (used as anchor)
    # This reduces perm_n by 1 to avoid including the NaN gap at position 0
    if start_index == 0:
        perm_n = n_bars - perm_index - 1
    else:
        perm_n = n_bars - perm_index

    if perm_n <= 0:
        # Nothing to permute
        return ohlc if n_markets > 1 else ohlc[0]

    # Arrays to hold relative movements
    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    # Calculate relative movements for each market
    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        # Store the anchor bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Calculate relative movements:
        # - Open relative to previous close (gap)
        # - High/Low/Close relative to open (intrabar movement)
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        # When start_index=0, skip the NaN gap value at index 0
        extract_start = perm_index + 1 if start_index == 0 else perm_index
        relative_open[mkt_i] = r_o[extract_start:]
        relative_high[mkt_i] = r_h[extract_start:]
        relative_low[mkt_i] = r_l[extract_start:]
        relative_close[mkt_i] = r_c[extract_start:]

    # Generate permutation indices
    idx = np.arange(perm_n)

    # Permute intrabar movements together (H/L/C stay correlated)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Permute gaps separately (preserves gap distribution)
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Reconstruct permuted OHLC
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy unchanged bars
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]

        # Set anchor bar
        # (provides starting point for reconstruction chain)
        perm_bars[start_index] = start_bar[mkt_i]

        # Reconstruct from permuted relative movements using JIT-compiled core
        # When start_index > 0: Can permute bar at start_index using previous close
        # When start_index = 0: Keep bar 0 as anchor, start reconstruction from bar 1
        recon_start = perm_index if start_index > 0 else perm_index + 1
        n_recon = n_bars - recon_start

        if n_recon > 0:
            # JIT-COMPILED RECONSTRUCTION using Numba (5-15x faster than NumPy)
            # Extract relative movements for this market
            r_open = relative_open[mkt_i][:n_recon]
            r_high = relative_high[mkt_i][:n_recon]
            r_low = relative_low[mkt_i][:n_recon]
            r_close = relative_close[mkt_i][:n_recon]

            # Get initial close price (anchor point for reconstruction)
            initial_close = perm_bars[recon_start - 1, 3]

            # Call JIT-compiled reconstruction (modifies perm_bars in-place)
            reconstruct_ohlc_core(
                perm_bars, r_open, r_high, r_low, r_close,
                recon_start, n_bars, initial_close
            )

        # Convert back from log prices
        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(
            perm_bars,
            index=time_index,
            columns=['open', 'high', 'low', 'close']
        )

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]


if __name__ == "__main__":
    # Simple test
    print("=== Bar Permutation Test ===")

    np.random.seed(42)

    # Create synthetic OHLC data
    n_bars = 100
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = close + np.random.randn(n_bars) * 0.2

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })

    print(f"Original data shape: {df.shape}")
    print(f"Original close range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # Permute entire dataset
    perm_df = get_permutation(df, seed=123)
    print(f"\nPermuted data shape: {perm_df.shape}")
    print(f"Permuted close range: {perm_df['close'].min():.2f} - {perm_df['close'].max():.2f}")

    # Permute with start_index (for walk-forward)
    perm_wf = get_permutation(df, start_index=50, seed=456)
    print(f"\nWalk-forward permutation (start_index=50):")
    print(f"First 50 bars unchanged: {np.allclose(df['close'].iloc[:50], perm_wf['close'].iloc[:50])}")

    print("\n=== Test Complete ===")
