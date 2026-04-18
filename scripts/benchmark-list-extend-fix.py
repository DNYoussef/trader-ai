"""
Benchmark script to demonstrate performance improvement from list.extend() fix.

Compares old (list.extend) vs new (pre-allocated array) approaches.
"""

import numpy as np
import time
from typing import List


def old_approach(data_chunks: List[np.ndarray]) -> np.ndarray:
    """Original approach using list.extend() - O(n^2)"""
    result = []
    for chunk in data_chunks:
        result.extend(chunk.tolist())
    return np.array(result)


def new_approach(data_chunks: List[np.ndarray]) -> np.ndarray:
    """Optimized approach using pre-allocated array - O(n)"""
    # Calculate total size
    total_size = sum(len(chunk) for chunk in data_chunks)

    # Pre-allocate
    result = np.empty(total_size, dtype=np.float64)
    current_idx = 0

    # Fill using slicing
    for chunk in data_chunks:
        chunk_size = len(chunk)
        result[current_idx:current_idx + chunk_size] = chunk
        current_idx += chunk_size

    return result


def benchmark():
    """Run benchmark comparison."""
    print("=" * 60)
    print("list.extend() Performance Fix Benchmark")
    print("=" * 60)

    # Generate test data similar to walk-forward validation
    np.random.seed(42)
    n_chunks = 16  # Number of walk-forward folds
    chunk_size = 63  # Step size in bars

    data_chunks = [np.random.randn(chunk_size) * 0.01 for _ in range(n_chunks)]

    print(f"\nTest Configuration:")
    print(f"  Number of chunks: {n_chunks}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Total elements: {n_chunks * chunk_size}")

    # Warm up
    _ = old_approach(data_chunks)
    _ = new_approach(data_chunks)

    # Benchmark old approach
    print("\nBenchmarking OLD approach (list.extend)...")
    start = time.perf_counter()
    for _ in range(100):
        result_old = old_approach(data_chunks)
    old_time = time.perf_counter() - start

    # Benchmark new approach
    print("Benchmarking NEW approach (pre-allocated array)...")
    start = time.perf_counter()
    for _ in range(100):
        result_new = new_approach(data_chunks)
    new_time = time.perf_counter() - start

    # Verify equivalence
    np.testing.assert_array_almost_equal(result_old, result_new)
    print("Output equivalence verified!")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Old approach: {old_time*1000:.3f} ms (100 iterations)")
    print(f"New approach: {new_time*1000:.3f} ms (100 iterations)")
    print(f"Speedup: {old_time/new_time:.2f}x")
    print(f"Time saved per iteration: {(old_time-new_time)*10:.3f} ms")

    # Scale to realistic validation run
    print("\n" + "=" * 60)
    print("EXTRAPOLATION TO FULL VALIDATION")
    print("=" * 60)
    print("Typical validation battery:")
    print("  - Walk-forward: 1 call")
    print("  - Block bootstrap MC: 500 paths x 13 blocks = 6,500 calls")
    print("  - Parameter grid: ~10 parameter combinations")
    print(f"\nEstimated time saved per validation:")
    validation_calls = 1 + (500 * 13)  # Walk-forward + MC bootstrap
    time_saved = (old_time - new_time) * validation_calls
    print(f"  {time_saved:.3f} seconds")
    print(f"  = {time_saved/60:.2f} minutes")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
