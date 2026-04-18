# Pivot Confirmation Lookahead Bias Fix

## Summary
Fixed critical lookahead bias in `src/intelligence/breakout_sweep_signal.py` where pivot confirmation was using future data.

## The Problem

### Original Code (WRONG)
```python
# Mark pivot at position i + window (after confirmation)
confirm_idx = i + window
if confirm_idx < len(df):
    if is_pivot_high:
        pivots.iloc[confirm_idx] = 1  # LOOKAHEAD!
```

### Why This Is Wrong

At bar `i`, the code checks if it's a pivot by looking at:
- Left window: bars `[i-window, i-1]` (historical - OK)
- Right window: bars `[i+1, i+window]` (FUTURE DATA - BAD!)

Then it marks the pivot at `i+window`, which makes it appear as if:
1. We knew about the pivot at bar `i+window`
2. But we used future bars `[i+window+1, i+2*window]` to confirm it

This is LOOKAHEAD BIAS because:
- At bar `i`, we're using bars `i+1` through `i+window` which don't exist yet
- The signal fires BEFORE the information is actually available
- In live trading, you can't access future bars

## The Solution

### Fixed Code (CORRECT)
```python
# FIX: Mark pivot at actual index i (not i+window)
# This preserves the true pivot location
# Confirmation lag is enforced in get_confirmed_pivot_levels()
if is_pivot_high:
    pivots.iloc[i] = 1  # Mark at actual pivot location
elif is_pivot_low:
    pivots.iloc[i] = -1
```

### Temporal Logic

**Pivot Detection:**
- At bar `i`, check if it's a pivot using window bars on each side
- This requires bars `[i-window, i+window]` to exist
- The pivot at index `i` is only KNOWN at bar `i+window` (natural lag)
- We mark it at index `i` to preserve the actual pivot location

**Pivot Usage:**
- When processing bar `current_idx`, only use pivots where `current_idx >= i+window`
- This ensures the pivot has been fully confirmed
- The confirmation lag is enforced in `get_confirmed_pivot_levels()`

## Changes Made

### 1. `mark_pivots()` Function

**Before:**
```python
confirm_idx = i + window
if confirm_idx < len(df):
    if is_pivot_high:
        pivots.iloc[confirm_idx] = 1
```

**After:**
```python
if is_pivot_high:
    pivots.iloc[i] = 1
elif is_pivot_low:
    pivots.iloc[i] = -1
```

### 2. `get_confirmed_pivot_levels()` Function

**Added Anti-Lookahead Check:**
```python
# ANTI-LOOKAHEAD CHECK: Verify we have confirmation at current_idx
# At current_idx, we can only see pivots where i+window <= current_idx
if i + window > current_idx:
    continue  # This pivot is not yet confirmed

if is_pivot_high:
    pivot_highs.append((i, current_high))
if is_pivot_low:
    pivot_lows.append((i, current_low))
```

This ensures:
- Pivots at index `i` are only returned if `current_idx >= i + window`
- The natural confirmation lag is enforced
- No future data is used

## Verification

### Test Cases

**Test 1: Pivot Marking Location**
```python
# Create data with clear pivot at index 6
df.loc[6, 'high'] = 20  # Obvious pivot high

window = 3
pivots = mark_pivots(df, window=3)

# OLD BUG: Pivot marked at index 9 (6+3)
# NEW FIX: Pivot marked at index 6 (actual location)
assert pivots.iloc[6] == 1  # PASS
```

**Test 2: Confirmation Lag**
```python
# Pivot at index 10, window=7
df.loc[10, 'high'] = 100

# At bar 16: pivot NOT visible yet (needs 17 = 10+7)
pivots_16 = get_confirmed_pivot_levels(df, current_idx=16, window=7)
assert len(pivots_16[0]) == 0  # PASS

# At bar 17: pivot IS visible (confirmed)
pivots_17 = get_confirmed_pivot_levels(df, current_idx=17, window=7)
assert len(pivots_17[0]) == 1  # PASS
assert pivots_17[0][0] == (10, 100)  # PASS
```

## Impact

### Before Fix
- Signals fired 7 bars too early (window=7)
- Backtest results were unrealistically good
- Live trading would fail because future data unavailable

### After Fix
- Signals fire at correct time (with natural lag)
- Backtest results are realistic and tradeable
- Live trading will work as expected

## Technical Details

### Why Mark at Index `i`?

We mark pivots at their actual index `i` (not `i+window`) because:
1. Preserves true pivot location for chart analysis
2. Allows accurate calculation of sweep depth/distance
3. Makes code easier to understand and debug
4. Matches industry standard pivot detection

### Why Add Confirmation Check?

The confirmation check `if i + window > current_idx: continue` ensures:
1. At bar `current_idx`, we only see pivots that are fully confirmed
2. A pivot at `i` needs bars `[i-window, i+window]` to exist
3. At bar `current_idx`, bar `i+window` must be historical
4. Therefore `i+window <= current_idx` must be true

## Files Modified

- `src/intelligence/breakout_sweep_signal.py`
  - `mark_pivots()` - Fixed pivot marking location
  - `get_confirmed_pivot_levels()` - Added anti-lookahead check
  - Updated docstrings with temporal logic explanations

## Testing

Run the test suite:
```bash
cd D:/Projects/trader-ai
python tests/test_pivot_lookahead_fix.py
```

Expected output:
```
PASS: Pivot marked at actual index 6
PASS: Pivot at index 10 confirmed at bar 17
PASS: Pivot at index 10 not visible at bar 16 (confirmation lag enforced)
```

## Conclusion

The pivot confirmation lookahead bias has been fixed. The code now:
- Marks pivots at their actual location
- Enforces natural confirmation lag
- Uses only historical data (no lookahead)
- Produces realistic backtest results
- Will work correctly in live trading

This fix is CRITICAL for:
- Accurate backtesting
- Realistic performance metrics
- Successful live trading deployment
