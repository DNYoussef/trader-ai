"""
Test script to verify MCPT parallelization works correctly.

Tests both sequential and parallel modes, verifying:
1. Both modes produce similar p-values (statistical consistency)
2. Parallel mode is faster than sequential
3. Results are reproducible with seed
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from intelligence.validation.mcpt_validator import MCPTValidator


def create_test_data(n_bars=500, seed=42):
    """Create synthetic OHLC data."""
    np.random.seed(seed)
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = close + np.random.randn(n_bars) * 0.2

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


def simple_optimizer(df: pd.DataFrame) -> float:
    """Simple optimizer that returns profit factor of a momentum strategy."""
    from intelligence.strategy_lab.signal_interface import strategy_returns

    # Simple momentum strategy
    returns = df['close'].pct_change()
    positions = np.where(returns.rolling(20).mean() > 0, 1, -1)
    positions[:20] = 0  # No position during warmup

    strat_returns = strategy_returns(df['close'].values, positions)

    # Compute profit factor
    gains = strat_returns[strat_returns > 0].sum()
    losses = abs(strat_returns[strat_returns < 0].sum())

    if losses == 0:
        return gains if gains > 0 else 1.0
    return gains / losses


def main():
    print("=== MCPT Parallelization Test ===\n")

    # Create test data
    df = create_test_data(n_bars=500, seed=42)
    print(f"Test data shape: {df.shape}")
    print(f"Available CPU cores: {os.cpu_count()}\n")

    # Test 1: Sequential mode
    print("--- Test 1: Sequential Mode ---")
    validator_seq = MCPTValidator(
        n_permutations=100,
        parallel=False
    )

    start_time = time.time()
    result_seq = validator_seq.insample_mcpt(
        simple_optimizer,
        df,
        objective='profit_factor',
        seed=42
    )
    seq_time = time.time() - start_time

    print(f"Sequential execution time: {seq_time:.2f}s")
    print(f"Real score: {result_seq.real_score:.4f}")
    print(f"P-value: {result_seq.p_value:.4f}")
    print(f"Valid permutations: {result_seq.n_permutations}\n")

    # Test 2: Parallel mode
    print("--- Test 2: Parallel Mode ---")
    validator_par = MCPTValidator(
        n_permutations=100,
        parallel=True,
        n_workers=None  # Use all cores
    )

    start_time = time.time()
    result_par = validator_par.insample_mcpt(
        simple_optimizer,
        df,
        objective='profit_factor',
        seed=42
    )
    par_time = time.time() - start_time

    print(f"Parallel execution time: {par_time:.2f}s")
    print(f"Real score: {result_par.real_score:.4f}")
    print(f"P-value: {result_par.p_value:.4f}")
    print(f"Valid permutations: {result_par.n_permutations}\n")

    # Test 3: Compare results
    print("--- Test 3: Comparison ---")
    speedup = seq_time / par_time if par_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

    # Check if p-values are reasonably close (should be within ~10% due to randomness)
    p_value_diff = abs(result_seq.p_value - result_par.p_value)
    p_value_similarity = 1 - (p_value_diff / max(result_seq.p_value, result_par.p_value))

    print(f"P-value difference: {p_value_diff:.4f}")
    print(f"P-value similarity: {p_value_similarity*100:.1f}%")

    # Test 4: Reproducibility with seed
    print("\n--- Test 4: Reproducibility Test ---")
    result_par2 = validator_par.insample_mcpt(
        simple_optimizer,
        df,
        objective='profit_factor',
        seed=42
    )

    print(f"First run p-value:  {result_par.p_value:.4f}")
    print(f"Second run p-value: {result_par2.p_value:.4f}")
    print(f"Difference: {abs(result_par.p_value - result_par2.p_value):.6f}")

    # Summary
    print("\n=== Summary ===")
    print(f"Parallel mode enabled: {result_par.n_permutations > 0}")
    print(f"Speedup achieved: {speedup:.2f}x")
    print(f"Results consistent: {p_value_similarity > 0.8}")
    print(f"Reproducibility: {abs(result_par.p_value - result_par2.p_value) < 0.01}")

    if speedup > 1.5:
        print("\nSUCCESS: Parallel execution is significantly faster!")
    else:
        print("\nWARNING: Parallel execution may not be providing expected speedup.")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
