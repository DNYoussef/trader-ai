"""
Unit tests for bar permutation vectorization.

Tests numerical equivalence between original sequential loop
and vectorized NumPy implementation.
"""

import numpy as np
import pandas as pd
import time
from bar_permute import get_permutation


def get_permutation_sequential(
    ohlc,
    start_index=0,
    seed=None
):
    """
    Original sequential implementation for comparison.
    This is a copy of the original code before vectorization.
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

    perm_index = start_index
    if start_index == 0:
        perm_n = n_bars - perm_index - 1
    else:
        perm_n = n_bars - perm_index

    if perm_n <= 0:
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

        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        extract_start = perm_index + 1 if start_index == 0 else perm_index
        relative_open[mkt_i] = r_o[extract_start:]
        relative_high[mkt_i] = r_h[extract_start:]
        relative_low[mkt_i] = r_l[extract_start:]
        relative_close[mkt_i] = r_c[extract_start:]

    # Generate permutation indices
    idx = np.arange(perm_n)

    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Reconstruct permuted OHLC
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]

        perm_bars[start_index] = start_bar[mkt_i]

        # ORIGINAL SEQUENTIAL LOOP
        recon_start = perm_index if start_index > 0 else perm_index + 1
        for i in range(recon_start, n_bars):
            extract_offset = 1 if start_index == 0 else 0
            k = i - perm_index - extract_offset
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

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


def create_test_data(n_bars=1000, seed=42):
    """Create synthetic OHLC data for testing."""
    np.random.seed(seed)

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

    return df


def test_numerical_equivalence_start_index_0():
    """Test vectorized vs sequential with start_index=0."""
    print("\n=== Test: Numerical Equivalence (start_index=0) ===")

    df = create_test_data(n_bars=1000)

    # Run both versions with same seed
    np.random.seed(123)
    result_sequential = get_permutation_sequential(df, start_index=0, seed=123)

    np.random.seed(123)
    result_vectorized = get_permutation(df, start_index=0, seed=123)

    # Compare results
    for col in ['open', 'high', 'low', 'close']:
        diff = np.abs(result_sequential[col] - result_vectorized[col])
        max_diff = diff.max()
        rel_diff = (diff / result_sequential[col]).max()

        print(f"  {col:5s} - max_abs_diff: {max_diff:.2e}, max_rel_diff: {rel_diff:.2e}")

        assert np.allclose(result_sequential[col], result_vectorized[col], rtol=1e-10), \
            f"Mismatch in {col} column"

    print("  PASS: Results are numerically equivalent")


def test_numerical_equivalence_start_index_nonzero():
    """Test vectorized vs sequential with start_index>0."""
    print("\n=== Test: Numerical Equivalence (start_index=252) ===")

    df = create_test_data(n_bars=1000)

    # Run both versions with same seed
    np.random.seed(456)
    result_sequential = get_permutation_sequential(df, start_index=252, seed=456)

    np.random.seed(456)
    result_vectorized = get_permutation(df, start_index=252, seed=456)

    # Compare results
    for col in ['open', 'high', 'low', 'close']:
        diff = np.abs(result_sequential[col] - result_vectorized[col])
        max_diff = diff.max()
        rel_diff = (diff / result_sequential[col]).max()

        print(f"  {col:5s} - max_abs_diff: {max_diff:.2e}, max_rel_diff: {rel_diff:.2e}")

        assert np.allclose(result_sequential[col], result_vectorized[col], rtol=1e-10), \
            f"Mismatch in {col} column"

    print("  PASS: Results are numerically equivalent")


def test_numerical_equivalence_multi_market():
    """Test vectorized vs sequential with multiple markets."""
    print("\n=== Test: Numerical Equivalence (Multi-Market) ===")

    df1 = create_test_data(n_bars=1000, seed=1)
    df2 = create_test_data(n_bars=1000, seed=2)
    df3 = create_test_data(n_bars=1000, seed=3)

    # Run both versions with same seed
    np.random.seed(789)
    result_sequential = get_permutation_sequential([df1, df2, df3], start_index=0, seed=789)

    np.random.seed(789)
    result_vectorized = get_permutation([df1, df2, df3], start_index=0, seed=789)

    # Compare results for each market
    for mkt_i in range(3):
        print(f"  Market {mkt_i}:")
        for col in ['open', 'high', 'low', 'close']:
            diff = np.abs(result_sequential[mkt_i][col] - result_vectorized[mkt_i][col])
            max_diff = diff.max()
            rel_diff = (diff / result_sequential[mkt_i][col]).max()

            print(f"    {col:5s} - max_abs_diff: {max_diff:.2e}, max_rel_diff: {rel_diff:.2e}")

            assert np.allclose(
                result_sequential[mkt_i][col],
                result_vectorized[mkt_i][col],
                rtol=1e-10
            ), f"Mismatch in market {mkt_i}, {col} column"

    print("  PASS: Results are numerically equivalent")


def test_performance_comparison():
    """Benchmark vectorized vs sequential implementation."""
    print("\n=== Performance Comparison ===")

    sizes = [100, 500, 1000, 5000, 10000]

    for n_bars in sizes:
        df = create_test_data(n_bars=n_bars)

        # Benchmark sequential
        start = time.time()
        for _ in range(10):
            _ = get_permutation_sequential(df, start_index=0, seed=999)
        time_sequential = (time.time() - start) / 10

        # Benchmark vectorized
        start = time.time()
        for _ in range(10):
            _ = get_permutation(df, start_index=0, seed=999)
        time_vectorized = (time.time() - start) / 10

        speedup = time_sequential / time_vectorized

        print(f"  n_bars={n_bars:5d}: sequential={time_sequential*1000:6.2f}ms, "
              f"vectorized={time_vectorized*1000:6.2f}ms, speedup={speedup:5.1f}x")


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test: Edge Cases ===")

    # Very small dataset
    df_small = create_test_data(n_bars=10)
    np.random.seed(111)
    result_seq = get_permutation_sequential(df_small, start_index=0, seed=111)
    np.random.seed(111)
    result_vec = get_permutation(df_small, start_index=0, seed=111)

    assert np.allclose(result_seq.values, result_vec.values, rtol=1e-10), \
        "Mismatch for small dataset"
    print("  PASS: Small dataset (n=10)")

    # start_index at end (no permutation)
    df = create_test_data(n_bars=100)
    result_seq = get_permutation_sequential(df, start_index=99)
    result_vec = get_permutation(df, start_index=99)

    assert np.allclose(result_seq.values, result_vec.values, rtol=1e-10), \
        "Mismatch for start_index at end"
    print("  PASS: start_index at end (no permutation)")

    # start_index near end
    np.random.seed(222)
    result_seq = get_permutation_sequential(df, start_index=95, seed=222)
    np.random.seed(222)
    result_vec = get_permutation(df, start_index=95, seed=222)

    assert np.allclose(result_seq.values, result_vec.values, rtol=1e-10), \
        "Mismatch for start_index near end"
    print("  PASS: start_index near end (n_recon=5)")


if __name__ == "__main__":
    print("="*60)
    print("Bar Permutation Vectorization Test Suite")
    print("="*60)

    # Numerical equivalence tests
    test_numerical_equivalence_start_index_0()
    test_numerical_equivalence_start_index_nonzero()
    test_numerical_equivalence_multi_market()

    # Edge case tests
    test_edge_cases()

    # Performance comparison
    test_performance_comparison()

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
