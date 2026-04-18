"""
Breakout + Liquidity Sweep Signal Module for Trader-AI

Detects breakout patterns with liquidity sweeps (stop hunts).
Pivots are only confirmed after N bars pass to avoid lookahead.

Based on: https://github.com/lj-valencia/BreakOutLiquiditySweep
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def mark_pivots(
    df: pd.DataFrame,
    window: int = 7
) -> pd.Series:
    """
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
    """
    pivots = pd.Series(0, index=df.index)

    for i in range(window, len(df) - window):
        # Check for pivot high at index i
        # Must be higher than all bars in window on both sides
        is_pivot_high = True
        is_pivot_low = True

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]

        for j in range(1, window + 1):
            if df['high'].iloc[i - j] >= current_high or df['high'].iloc[i + j] >= current_high:
                is_pivot_high = False
            if df['low'].iloc[i - j] <= current_low or df['low'].iloc[i + j] <= current_low:
                is_pivot_low = False

        # FIX: Mark pivot at actual index i (not i+window)
        # This preserves the true pivot location
        # Confirmation lag is enforced in get_confirmed_pivot_levels()
        if is_pivot_high:
            pivots.iloc[i] = 1
        elif is_pivot_low:
            pivots.iloc[i] = -1

    return pivots


def get_confirmed_pivot_levels(
    df: pd.DataFrame,
    current_idx: int,
    window: int = 7,
    backcandles: int = 40
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
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
    """
    pivot_highs = []
    pivot_lows = []

    # Look back from current position
    start_idx = max(0, current_idx - backcandles - window)

    for i in range(start_idx, current_idx - window):
        if i < window or i >= len(df) - window:
            continue

        # Check for pivot high
        is_pivot_high = True
        is_pivot_low = True

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]

        for j in range(1, window + 1):
            if i - j < 0 or i + j >= len(df):
                is_pivot_high = False
                is_pivot_low = False
                break
            if df['high'].iloc[i - j] >= current_high or df['high'].iloc[i + j] >= current_high:
                is_pivot_high = False
            if df['low'].iloc[i - j] <= current_low or df['low'].iloc[i + j] <= current_low:
                is_pivot_low = False

        # ANTI-LOOKAHEAD CHECK: Verify we have confirmation at current_idx
        # At current_idx, we can only see pivots where i+window <= current_idx
        if i + window > current_idx:
            continue  # This pivot is not yet confirmed

        if is_pivot_high:
            pivot_highs.append((i, current_high))
        if is_pivot_low:
            pivot_lows.append((i, current_low))

    return pivot_highs, pivot_lows


def breakout_sweep_signal(
    df: pd.DataFrame,
    idx: int,
    backcandles: int = 40,
    window: int = 7
) -> Tuple[int, Optional[int]]:
    """
    Detect breakout with liquidity sweep at current index.

    Bullish breakout (+2): Price sweeps below recent pivot low then closes above
    Bearish breakout (-2): Price sweeps above recent pivot high then closes below

    CRITICAL: Only fires on FIRST candle crossing the level.

    Args:
        df: OHLCV DataFrame
        idx: Current bar index
        backcandles: Lookback for pivot levels
        window: Pivot confirmation window

    Returns:
        Tuple of (signal, level_idx)
        signal: +2 bullish sweep, -2 bearish sweep, 0 no signal
        level_idx: Index of the broken pivot level (or None)
    """
    if idx < backcandles + window:
        return 0, None

    # Get confirmed pivot levels
    pivot_highs, pivot_lows = get_confirmed_pivot_levels(
        df, idx, window, backcandles
    )

    current_high = df['high'].iloc[idx]
    current_low = df['low'].iloc[idx]
    current_close = df['close'].iloc[idx]
    prev_close = df['close'].iloc[idx - 1]

    # Check for bullish sweep (sweep below pivot low, close above)
    for pivot_idx, pivot_price in reversed(pivot_lows):
        # Sweep condition: current low goes below pivot
        if current_low < pivot_price:
            # Recovery: closes above pivot
            if current_close > pivot_price:
                # First crossing check: previous close was above pivot
                if prev_close >= pivot_price:
                    return 2, pivot_idx

    # Check for bearish sweep (sweep above pivot high, close below)
    for pivot_idx, pivot_price in reversed(pivot_highs):
        # Sweep condition: current high goes above pivot
        if current_high > pivot_price:
            # Recovery: closes below pivot
            if current_close < pivot_price:
                # First crossing check: previous close was below pivot
                if prev_close <= pivot_price:
                    return -2, pivot_idx

    return 0, None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr.fillna(method='bfill')


def breakout_sweep_depth_atr(
    df: pd.DataFrame,
    idx: int,
    level_idx: int
) -> float:
    """
    Calculate sweep depth normalized by ATR.

    Measures how far price swept past the level before reversing.
    """
    if level_idx is None or idx >= len(df) or level_idx >= len(df):
        return 0.0

    atr = calculate_atr(df)
    current_atr = atr.iloc[idx] if idx < len(atr) else atr.iloc[-1]

    if current_atr == 0:
        return 0.0

    pivot_price = df['high'].iloc[level_idx] if df['close'].iloc[idx] < df['open'].iloc[idx] else df['low'].iloc[level_idx]
    current_extreme = df['low'].iloc[idx] if df['close'].iloc[idx] > df['open'].iloc[idx] else df['high'].iloc[idx]

    depth = abs(current_extreme - pivot_price)

    return depth / current_atr


def breakout_break_distance_atr(
    df: pd.DataFrame,
    idx: int,
    level_idx: int
) -> float:
    """
    Calculate distance beyond broken level, normalized by ATR.

    Measures how far price closed beyond the level after the sweep.
    """
    if level_idx is None or idx >= len(df) or level_idx >= len(df):
        return 0.0

    atr = calculate_atr(df)
    current_atr = atr.iloc[idx] if idx < len(atr) else atr.iloc[-1]

    if current_atr == 0:
        return 0.0

    current_close = df['close'].iloc[idx]

    # Determine if bullish or bearish based on close vs open
    if current_close > df['open'].iloc[idx]:
        # Bullish - measure distance above pivot low
        pivot_price = df['low'].iloc[level_idx]
        distance = current_close - pivot_price
    else:
        # Bearish - measure distance below pivot high
        pivot_price = df['high'].iloc[level_idx]
        distance = pivot_price - current_close

    return max(0, distance / current_atr)


def ma_trend_filter(
    df: pd.DataFrame,
    ema_len: int = 200,
    backcandles: int = 15
) -> pd.Series:
    """
    Moving average trend filter.

    Returns +1 if price consistently above EMA (bullish trend).
    Returns -1 if price consistently below EMA (bearish trend).
    Returns 0 if mixed/neutral.
    """
    ema = df['close'].ewm(span=ema_len, adjust=False).mean()

    signals = pd.Series(0, index=df.index)

    for i in range(backcandles, len(df)):
        above_count = 0
        below_count = 0

        for j in range(backcandles):
            if df['close'].iloc[i - j] > ema.iloc[i - j]:
                above_count += 1
            else:
                below_count += 1

        if above_count >= backcandles * 0.8:
            signals.iloc[i] = 1
        elif below_count >= backcandles * 0.8:
            signals.iloc[i] = -1

    return signals


def generate_breakout_sweep_features(
    df: pd.DataFrame,
    idx: int,
    backcandles: int = 40,
    window: int = 7
) -> Dict[str, float]:
    """
    Generate all breakout/sweep features for the 38D feature vector.

    Args:
        df: OHLCV DataFrame
        idx: Current bar index

    Returns:
        Dict with feature values
    """
    if idx < backcandles + window:
        return {
            'bos_sweep_signal': 0.0,
            'bos_sweep_depth_atr': 0.0,
            'bos_break_distance_atr': 0.0
        }

    # Get breakout signal
    signal, level_idx = breakout_sweep_signal(df, idx, backcandles, window)

    # Normalize signal from +/-2 to +/-1
    normalized_signal = signal / 2.0 if signal != 0 else 0.0

    # Calculate ATR-normalized metrics
    if level_idx is not None:
        sweep_depth = breakout_sweep_depth_atr(df, idx, level_idx)
        break_distance = breakout_break_distance_atr(df, idx, level_idx)
    else:
        sweep_depth = 0.0
        break_distance = 0.0

    return {
        'bos_sweep_signal': normalized_signal,
        'bos_sweep_depth_atr': sweep_depth,
        'bos_break_distance_atr': break_distance
    }


def breakout_sweep_signal_for_backtest(
    df: pd.DataFrame,
    idx: int
) -> Tuple[int, Dict]:
    """
    Generate breakout/sweep signal for backtesting at specific index.

    CRITICAL: Only uses data up to and including idx (no lookahead).

    Args:
        df: Full OHLCV DataFrame
        idx: Current bar index

    Returns:
        Tuple of (signal, metadata)
        signal: +1 long, -1 short, 0 no signal
        metadata: dict with strength metrics
    """
    features = generate_breakout_sweep_features(df, idx)

    # Signal is the normalized sweep signal
    raw_signal = features['bos_sweep_signal']
    signal = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)

    # Get trend filter for additional context
    trend = ma_trend_filter(df.iloc[:idx + 1])
    trend_dir = trend.iloc[-1] if len(trend) > 0 else 0

    metadata = {
        'sweep_depth_atr': features['bos_sweep_depth_atr'],
        'break_distance_atr': features['bos_break_distance_atr'],
        'trend_filter': trend_dir,
        'signal_type': 'BREAKOUT_SWEEP'
    }

    # Only take signals aligned with trend
    if signal != 0 and trend_dir != 0 and signal != trend_dir:
        signal = 0  # Reject counter-trend signals

    return signal, metadata
