"""
Signal Interface Module

Converts strategy signals to position vectors and calculates returns.

Key Concepts:
- Position Signal: Vector with values in {-1, 0, +1} per bar
- Strategy Returns: positions[i] * log_returns[i+1] (shifted forward)

This is the bridge between the strategy's analyze() method and
the MCPT validation system.

Performance: Uses Numba JIT compilation for 5-15x speedup on numerical operations.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Callable, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.strategies.black_swan_strategies import BaseStrategy, MarketState

# Import JIT-compiled numerical kernels for performance
from .signal_interface_numba import (
    strategy_returns_core,
    equity_curve_core,
    compare_signals_core,
)

logger = logging.getLogger(__name__)


def build_position_signal(
    df_ohlc: pd.DataFrame,
    strategy: 'BaseStrategy',
    market_state_builder: Optional[Callable] = None,
) -> np.ndarray:
    """
    Convert strategy to position vector per bar.

    For each bar, calls strategy.analyze() and converts the signal to:
    - +1 for 'buy' signal
    - -1 for 'sell' signal
    - 0 for 'hold' or no signal

    CRITICAL: This function only uses data up to the current bar (no lookahead).

    Args:
        df_ohlc: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        strategy: Strategy instance with analyze() method
        market_state_builder: Optional function to build MarketState from historical data
                            If None, uses default_market_state_builder

    Returns:
        positions: np.ndarray of shape (n_bars,) with values in {-1, 0, +1}
    """
    n_bars = len(df_ohlc)
    positions = np.zeros(n_bars)

    if market_state_builder is None:
        market_state_builder = default_market_state_builder

    for i in range(n_bars):
        # Only use data up to and including bar i
        historical = df_ohlc.iloc[:i + 1]

        if len(historical) < 20:  # Need minimum history
            positions[i] = 0
            continue

        try:
            # Build market state from historical data
            market_state = market_state_builder(historical)

            # Get strategy signal
            signal = strategy.analyze(market_state, historical)

            if signal is None:
                positions[i] = 0
            elif signal.action == 'buy':
                positions[i] = 1
            elif signal.action == 'sell':
                positions[i] = -1
            else:  # 'hold'
                positions[i] = 0

        except Exception as e:
            logger.debug(f"Strategy analysis failed at bar {i}: {e}")
            positions[i] = 0

    return positions


def strategy_returns(
    close: np.ndarray,
    positions: np.ndarray,
    fees_bps: float = 0.0,
) -> np.ndarray:
    """
    Compute strategy returns per bar.

    The key formula: returns[i] = positions[i] * log_return[i+1]

    This means the position taken at bar i affects the return from bar i to bar i+1.
    This is the correct way to avoid lookahead bias.

    Args:
        close: Array of close prices
        positions: Array of positions {-1, 0, +1}
        fees_bps: Transaction costs in basis points (applied on position changes)

    Returns:
        strategy_returns: Array of per-bar returns (length = len(close) - 1)
    """
    # Close-to-close log returns
    log_returns = np.diff(np.log(close))

    # Use JIT-compiled core for performance (5-15x faster)
    return strategy_returns_core(
        np.asarray(log_returns, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        float(fees_bps),
    )


def equity_curve(
    close: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = 1.0,
    fees_bps: float = 0.0,
) -> np.ndarray:
    """
    Compute cumulative equity curve from positions.

    Args:
        close: Array of close prices
        positions: Array of positions {-1, 0, +1}
        initial_capital: Starting capital (default: 1.0 for normalized)
        fees_bps: Transaction costs in basis points

    Returns:
        equity: Cumulative equity curve (length = len(close))
    """
    # Close-to-close log returns
    log_returns = np.diff(np.log(close))

    # Use JIT-compiled core for performance (5-15x faster)
    return equity_curve_core(
        np.asarray(log_returns, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        float(initial_capital),
        float(fees_bps),
    )


def position_from_signal_function(
    df_ohlc: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, int], int],
) -> np.ndarray:
    """
    Build positions using a simple signal function.

    This is an alternative to build_position_signal() for simpler strategies
    that don't need the full strategy.analyze() interface.

    Args:
        df_ohlc: OHLCV DataFrame
        signal_fn: Function(df, idx) -> {-1, 0, +1}

    Returns:
        positions: np.ndarray with values in {-1, 0, +1}
    """
    n_bars = len(df_ohlc)
    positions = np.zeros(n_bars)

    for i in range(n_bars):
        try:
            positions[i] = signal_fn(df_ohlc, i)
        except Exception as e:
            logger.debug(f"Signal function failed at bar {i}: {e}")
            positions[i] = 0

    return positions


def default_market_state_builder(historical: pd.DataFrame) -> 'MarketState':
    """
    Build a MarketState from historical data for strategy.analyze().

    This is a simplified version - the actual implementation should
    compute all the required fields.
    """
    # Import here to avoid circular imports
    from src.strategies.black_swan_strategies import MarketState
    from datetime import datetime

    close = historical['close'].values
    n = len(close)

    # Calculate basic metrics
    returns = np.diff(np.log(close)) if n > 1 else np.array([0.0])
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2

    # VIX approximation from volatility
    vix_level = volatility * 100  # Rough approximation

    # Calculate returns
    spy_returns_5d = (close[-1] / close[-6] - 1) if n > 5 else 0.0
    spy_returns_20d = (close[-1] / close[-21] - 1) if n > 20 else 0.0

    # Volume ratio
    volume = historical['volume'].values if 'volume' in historical.columns else np.ones(n)
    vol_20d_avg = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
    volume_ratio = volume[-1] / vol_20d_avg if vol_20d_avg > 0 else 1.0

    # Determine regime
    if vix_level > 30:
        regime = 'crisis'
    elif vix_level > 20:
        regime = 'volatile'
    elif vix_level > 15:
        regime = 'normal'
    else:
        regime = 'calm'

    return MarketState(
        timestamp=datetime.now(),
        vix_level=vix_level,
        vix_percentile=0.5,  # Would need VIX history for percentile
        spy_returns_5d=spy_returns_5d,
        spy_returns_20d=spy_returns_20d,
        put_call_ratio=1.0,  # Would need options data
        market_breadth=0.5,  # Would need breadth data
        correlation=0.5,  # Would need correlation calculation
        volume_ratio=volume_ratio,
        regime=regime,
        indicators={},
    )


def compare_signals(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> dict:
    """
    Compare two signal arrays for agreement/disagreement.

    Useful for ensemble validation and consensus checking.

    Args:
        signal_a: First position signal array
        signal_b: Second position signal array

    Returns:
        Dict with agreement metrics
    """
    assert len(signal_a) == len(signal_b), "Signals must have same length"

    # Use JIT-compiled core for performance
    agreement, disagreement, signal_a_active, signal_b_active = compare_signals_core(
        np.asarray(signal_a, dtype=np.float64),
        np.asarray(signal_b, dtype=np.float64),
    )

    # Correlation (not JIT-compiled due to numpy complexity)
    correlation = np.corrcoef(signal_a, signal_b)[0, 1] if len(signal_a) > 1 else 0.0

    total_nonzero = agreement + disagreement

    return {
        'agreement_count': int(agreement),
        'disagreement_count': int(disagreement),
        'agreement_rate': agreement / total_nonzero if total_nonzero > 0 else 0.0,
        'correlation': float(correlation),
        'signal_a_active': int(signal_a_active),
        'signal_b_active': int(signal_b_active),
    }


if __name__ == "__main__":
    # Simple test
    print("=== Signal Interface Test ===")

    # Create synthetic data
    np.random.seed(42)
    n_bars = 100

    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    positions = np.random.choice([-1, 0, 1], n_bars, p=[0.1, 0.8, 0.1])

    print(f"Close prices: {close[:5]}...")
    print(f"Positions: {positions[:20]}...")

    # Calculate returns
    returns = strategy_returns(close, positions, fees_bps=10)
    print(f"Strategy returns: {returns[:10]}...")
    print(f"Total return: {np.sum(returns):.4f}")

    # Calculate equity curve
    equity = equity_curve(close, positions, initial_capital=10000, fees_bps=10)
    print(f"Equity curve: {equity[:5]}... -> {equity[-1]:.2f}")

    print("\n=== Test Complete ===")
