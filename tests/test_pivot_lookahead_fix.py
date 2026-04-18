"""
Test script to verify pivot lookahead bias fix

Tests that:
1. Pivots are marked at their actual index (not i+window)
2. get_confirmed_pivot_levels only returns confirmed pivots
3. No future data is used
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:/Projects/trader-ai/src')

from intelligence.breakout_sweep_signal import mark_pivots, get_confirmed_pivot_levels


def test_pivot_marking():
    """Test that pivots are marked at correct index"""
    # Create simple test data with clear pivot at index 10
    df = pd.DataFrame({
        'high': [10, 11, 12, 13, 14, 15, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
        'low': [9, 10, 11, 12, 13, 14, 19, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    })

    window = 3
    pivots = mark_pivots(df, window=window)

    print("Test Data:")
    print(df.head(15))
    print("\nPivots Series:")
    print(pivots[pivots != 0])

    # OLD BUG: Pivot at i=6 would be marked at i+window=9
    # NEW FIX: Pivot at i=6 should be marked at i=6

    pivot_indices = pivots[pivots != 0].index.tolist()
    print(f"\nPivot found at indices: {pivot_indices}")

    # The clear pivot high is at index 6 (high=20)
    # It should be marked at index 6, not 9
    if 6 in pivot_indices:
        print("PASS: Pivot marked at actual index 6")
    else:
        print(f"FAIL: Pivot not at index 6. Found at: {pivot_indices}")

    return pivots


def test_confirmation_lag():
    """Test that get_confirmed_pivot_levels enforces confirmation lag"""
    # Create test data
    df = pd.DataFrame({
        'high': list(range(1, 51)),
        'low': list(range(0, 50))
    })

    # Insert clear pivot at index 10
    df.loc[10, 'high'] = 100
    df.loc[10, 'low'] = 99

    window = 7

    # At bar 10, the pivot should NOT be visible (needs 7 more bars)
    pivot_highs, pivot_lows = get_confirmed_pivot_levels(df, current_idx=10, window=window)
    print(f"\nAt current_idx=10, confirmed pivots: {len(pivot_highs)} highs, {len(pivot_lows)} lows")

    # At bar 17, the pivot at index 10 should be visible
    pivot_highs, pivot_lows = get_confirmed_pivot_levels(df, current_idx=17, window=window)
    print(f"At current_idx=17, confirmed pivots: {len(pivot_highs)} highs, {len(pivot_lows)} lows")

    if len(pivot_highs) > 0:
        print(f"Confirmed pivot high at index: {pivot_highs[0][0]}, price: {pivot_highs[0][1]}")
        if pivot_highs[0][0] == 10:
            print("PASS: Pivot at index 10 confirmed at bar 17")
        else:
            print(f"FAIL: Wrong pivot index")
    else:
        print("FAIL: No pivot found")

    # At bar 16, the pivot should NOT be visible yet
    pivot_highs16, pivot_lows16 = get_confirmed_pivot_levels(df, current_idx=16, window=window)
    if len(pivot_highs16) == 0:
        print("PASS: Pivot at index 10 not visible at bar 16 (confirmation lag enforced)")
    else:
        print("FAIL: Pivot visible too early (lookahead bias)")


if __name__ == '__main__':
    print("=" * 60)
    print("PIVOT LOOKAHEAD FIX VERIFICATION")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("TEST 1: Pivot marking at correct index")
    print("=" * 60)
    test_pivot_marking()

    print("\n" + "=" * 60)
    print("TEST 2: Confirmation lag enforcement")
    print("=" * 60)
    test_confirmation_lag()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
