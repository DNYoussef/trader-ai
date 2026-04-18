"""
Script to fix pivot lookahead bias in breakout_sweep_signal.py
"""

import re

filepath = "D:/Projects/trader-ai/src/intelligence/breakout_sweep_signal.py"

# Read the file
with open(filepath, 'r') as f:
    content = f.read()

# Fix 1: Update mark_pivots docstring
old_docstring = '''    """
    Mark pivot highs and lows with confirmation delay.

    CRITICAL: Pivots are only marked AFTER the confirmation window passes.
    A pivot at index i is only confirmed at index i + window.

    Args:
        df: OHLCV DataFrame
        window: Confirmation window (default: 7)

    Returns:
        Series: +1 = pivot high, -1 = pivot low, 0 = no pivot
    """'''

new_docstring = '''    """
    Mark pivot highs and lows with confirmation delay.

    CRITICAL FIX: Pivots are marked at their actual index i, NOT i+window.

    TEMPORAL LOGIC:
    - At bar i, we check if its a pivot using window bars on EACH SIDE
    - Left window: bars [i-window, i-1] are historical (valid)
    - Right window: bars [i+1, i+window] are historical when processing bar i+window
    - Therefore: pivot at i is only KNOWN at bar i+window (natural lag)
    - We mark it at index i to preserve the actual pivot location
    - Downstream code must account for this window-bar confirmation lag

    ANTI-LOOKAHEAD:
    - The loop starts at i=window and ends at len(df)-window
    - This ensures we never access future bars beyond whats needed for confirmation
    - When processing bar current_idx, only use pivots where current_idx >= i+window

    Args:
        df: OHLCV DataFrame
        window: Confirmation window (default: 7)

    Returns:
        Series: +1 = pivot high, -1 = pivot low, 0 = no pivot
        Pivots marked at their actual index, but only visible after window-bar delay
    """'''

content = content.replace(old_docstring, new_docstring)

# Fix 2: Update pivot marking code in mark_pivots
old_marking = '''        # Mark pivot at position i + window (after confirmation)
        confirm_idx = i + window
        if confirm_idx < len(df):
            if is_pivot_high:
                pivots.iloc[confirm_idx] = 1
            elif is_pivot_low:
                pivots.iloc[confirm_idx] = -1'''

new_marking = '''        # FIX: Mark pivot at actual index i (not i+window)
        # This preserves the true pivot location
        # Confirmation lag is enforced in get_confirmed_pivot_levels()
        if is_pivot_high:
            pivots.iloc[i] = 1
        elif is_pivot_low:
            pivots.iloc[i] = -1'''

content = content.replace(old_marking, new_marking)

# Fix 3: Update get_confirmed_pivot_levels docstring
old_get_docstring = '''    """
    Get confirmed pivot levels visible at current index.

    Only returns pivots that were confirmed BEFORE current_idx.

    Args:
        df: OHLCV DataFrame
        current_idx: Current bar index
        window: Pivot confirmation window
        backcandles: How far back to look for pivots

    Returns:
        Tuple of (pivot_highs, pivot_lows)
        Each is list of (original_idx, price)
    """'''

new_get_docstring = '''    """
    Get confirmed pivot levels visible at current index.

    CRITICAL FIX: Only returns pivots that are CONFIRMED at current_idx.
    A pivot at index i requires i+window bars to be confirmed.
    Therefore, at current_idx, we can only see pivots where i+window <= current_idx.

    TEMPORAL LOGIC:
    - Pivot at index i needs bars [i-window, i+window] to be confirmed
    - At current_idx, the right side bars [i+1, i+window] must be historical
    - This means i+window <= current_idx, or i <= current_idx - window
    - We only return pivots in range [start_idx, current_idx - window]

    Args:
        df: OHLCV DataFrame
        current_idx: Current bar index
        window: Pivot confirmation window
        backcandles: How far back to look for pivots

    Returns:
        Tuple of (pivot_highs, pivot_lows)
        Each is list of (original_idx, price)
    """'''

content = content.replace(old_get_docstring, new_get_docstring)

# Fix 4: Add anti-lookahead check in get_confirmed_pivot_levels
old_pivot_check = '''        if is_pivot_high:
            pivot_highs.append((i, current_high))
        if is_pivot_low:
            pivot_lows.append((i, current_low))

    return pivot_highs, pivot_lows'''

new_pivot_check = '''        # ANTI-LOOKAHEAD CHECK: Verify we have confirmation at current_idx
        # At current_idx, we can only see pivots where i+window <= current_idx
        if i + window > current_idx:
            continue  # This pivot is not yet confirmed

        if is_pivot_high:
            pivot_highs.append((i, current_high))
        if is_pivot_low:
            pivot_lows.append((i, current_low))

    return pivot_highs, pivot_lows'''

content = content.replace(old_pivot_check, new_pivot_check)

# Write the fixed file
with open(filepath, 'w') as f:
    f.write(content)

print("Pivot lookahead fixes applied successfully!")
print("\nChanges made:")
print("1. Updated mark_pivots() docstring with temporal logic explanation")
print("2. Changed pivot marking from i+window to i")
print("3. Updated get_confirmed_pivot_levels() docstring")
print("4. Added anti-lookahead check: if i + window > current_idx: continue")
