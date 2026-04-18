"""
Ichimoku Signal Module for Trader-AI

Manual Ichimoku calculation to avoid lookahead bias.
Provides trend detection, cloud strength, and retracement signals.

Based on: https://github.com/lj-valencia/Ichimoku_Trend
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def ichimoku_manual(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52
) -> pd.DataFrame:
    """
    Calculate Ichimoku components manually - NO LOOKAHEAD.

    CRITICAL: This implementation does NOT shift senkou spans forward,
    which would cause lookahead bias in backtesting.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns
        tenkan: Tenkan-sen period (default: 9)
        kijun: Kijun-sen period (default: 26)
        senkou: Senkou Span B period (default: 52)

    Returns:
        DataFrame with Ichimoku components added
    """
    result = df.copy()

    # Tenkan-sen: (highest high + lowest low) / 2 for tenkan periods
    high_tenkan = result['high'].rolling(window=tenkan).max()
    low_tenkan = result['low'].rolling(window=tenkan).min()
    result['tenkan_sen'] = (high_tenkan + low_tenkan) / 2

    # Kijun-sen: (highest high + lowest low) / 2 for kijun periods
    high_kijun = result['high'].rolling(window=kijun).max()
    low_kijun = result['low'].rolling(window=kijun).min()
    result['kijun_sen'] = (high_kijun + low_kijun) / 2

    # Senkou Span A: (Tenkan + Kijun) / 2 - NO forward shift
    result['senkou_span_a'] = (result['tenkan_sen'] + result['kijun_sen']) / 2

    # Senkou Span B - NO forward shift
    high_senkou = result['high'].rolling(window=senkou).max()
    low_senkou = result['low'].rolling(window=senkou).min()
    result['senkou_span_b'] = (high_senkou + low_senkou) / 2

    # Cloud boundaries
    result['cloud_top'] = result[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    result['cloud_bottom'] = result[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    result['cloud_thickness'] = result['cloud_top'] - result['cloud_bottom']

    # Price position relative to cloud
    result['above_cloud'] = (result['close'] > result['cloud_top']).astype(int)
    result['below_cloud'] = (result['close'] < result['cloud_bottom']).astype(int)
    result['in_cloud'] = ((result['close'] >= result['cloud_bottom']) &
                          (result['close'] <= result['cloud_top'])).astype(int)

    return result


def ema_trend_signal(
    df: pd.DataFrame,
    ema_len: int = 100,
    backcandles: int = 5
) -> pd.Series:
    """
    EMA trend confirmation signal.

    Returns +1 if last N candles are fully above EMA.
    Returns -1 if last N candles are fully below EMA.
    Returns 0 otherwise.
    """
    ema = df['close'].ewm(span=ema_len, adjust=False).mean()

    signals = pd.Series(0, index=df.index)

    for i in range(backcandles, len(df)):
        all_above = True
        all_below = True

        for j in range(backcandles):
            idx = i - j
            if df['open'].iloc[idx] <= ema.iloc[idx] or df['close'].iloc[idx] <= ema.iloc[idx]:
                all_above = False
            if df['open'].iloc[idx] >= ema.iloc[idx] or df['close'].iloc[idx] >= ema.iloc[idx]:
                all_below = False

        if all_above:
            signals.iloc[i] = 1
        elif all_below:
            signals.iloc[i] = -1

    return signals


def ichimoku_cloud_strength_ratio(
    df: pd.DataFrame,
    cloud_lookback: int = 10
) -> pd.Series:
    """
    Calculate cloud strength ratio - proportion of bars above cloud.
    """
    if 'cloud_top' not in df.columns:
        df = ichimoku_manual(df)

    fully_above = ((df['open'] > df['cloud_top']) &
                   (df['close'] > df['cloud_top'])).astype(float)

    ratio = fully_above.rolling(window=cloud_lookback).mean()

    return ratio.fillna(0)


def ichimoku_retracement_signal(
    df: pd.DataFrame,
    cloud_lookback: int = 10,
    min_above: int = 7
) -> pd.Series:
    """
    Ichimoku retracement (bounce) signal.

    Bullish (+1): Uptrend + opens in cloud + closes above cloud
    Bearish (-1): Downtrend + opens in cloud + closes below cloud
    """
    if 'cloud_top' not in df.columns:
        df = ichimoku_manual(df)

    signals = pd.Series(0, index=df.index)

    fully_above = ((df['open'] > df['cloud_top']) &
                   (df['close'] > df['cloud_top']))
    fully_below = ((df['open'] < df['cloud_bottom']) &
                   (df['close'] < df['cloud_bottom']))

    for i in range(cloud_lookback, len(df)):
        lookback_start = i - cloud_lookback
        lookback_end = i

        above_count = fully_above.iloc[lookback_start:lookback_end].sum()
        below_count = fully_below.iloc[lookback_start:lookback_end].sum()

        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        cloud_top = df['cloud_top'].iloc[i]
        cloud_bottom = df['cloud_bottom'].iloc[i]

        opens_in_cloud = (open_price >= cloud_bottom) and (open_price <= cloud_top)
        closes_above = close_price > cloud_top
        closes_below = close_price < cloud_bottom

        if above_count >= min_above and opens_in_cloud and closes_above:
            signals.iloc[i] = 1
        elif below_count >= min_above and opens_in_cloud and closes_below:
            signals.iloc[i] = -1

    return signals


def generate_ichimoku_features(
    df: pd.DataFrame,
    ema_len: int = 100,
    backcandles: int = 5,
    cloud_lookback: int = 10,
    min_above: int = 7
) -> Dict[str, float]:
    """
    Generate all Ichimoku-based features for the 38D feature vector.
    """
    if len(df) < max(ema_len, cloud_lookback, 52):
        return {
            'ichi_trend_ema100': 0.0,
            'ichi_cloud_strength': 0.0,
            'ichi_retracement': 0.0
        }

    ichi_df = ichimoku_manual(df)
    trend_signal = ema_trend_signal(ichi_df, ema_len, backcandles)
    cloud_strength = ichimoku_cloud_strength_ratio(ichi_df, cloud_lookback)
    retracement = ichimoku_retracement_signal(ichi_df, cloud_lookback, min_above)

    return {
        'ichi_trend_ema100': float(trend_signal.iloc[-1]),
        'ichi_cloud_strength': float(cloud_strength.iloc[-1]),
        'ichi_retracement': float(retracement.iloc[-1])
    }


def ichimoku_signal_for_backtest(
    df: pd.DataFrame,
    idx: int
) -> Tuple[int, Dict]:
    """
    Generate Ichimoku signal for backtesting at specific index.

    CRITICAL: Only uses data up to and including idx (no lookahead).
    """
    historical = df.iloc[:idx + 1].copy()

    if len(historical) < 52:
        return 0, {}

    features = generate_ichimoku_features(historical)
    signal = int(features['ichi_retracement'])

    metadata = {
        'trend_signal': features['ichi_trend_ema100'],
        'cloud_strength': features['ichi_cloud_strength'],
        'signal_type': 'ICHIMOKU_RETRACEMENT'
    }

    return signal, metadata
