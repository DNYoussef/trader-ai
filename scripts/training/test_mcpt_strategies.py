#!/usr/bin/env python3
"""
Test MCPT Validation on Existing 8 Strategies

This script validates the 8 black swan strategies using Monte Carlo
Permutation Testing to verify they have statistical significance.

Usage:
    python scripts/training/test_mcpt_strategies.py [--quick] [--strategy NAME]
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party'))

from src.intelligence.validation import (
    MCPTValidator,
    MCPTResult,
    passes_gate,
    profit_factor,
    compute_all_metrics,
)
from src.intelligence.strategy_lab import (
    StrategySpec,
    build_position_signal,
    strategy_returns,
)
from src.intelligence.strategy_lab.strategy_spec import EXISTING_STRATEGIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    In production, this would load real market data.
    """
    np.random.seed(seed)

    # Generate trending price series with volatility clustering
    returns = np.random.randn(n_bars) * 0.01

    # Add some momentum
    for i in range(1, n_bars):
        returns[i] += 0.1 * returns[i-1]

    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = np.roll(close, 1) + np.random.randn(n_bars) * 0.1
    open_price[0] = close[0]

    # Volume
    volume = np.random.uniform(1e6, 5e6, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'symbol': 'SPY',
        'date': pd.date_range(start='2020-01-01', periods=n_bars, freq='D'),
    })

    return df


def create_simple_signal_function(strategy_id: int):
    """
    Create a simple signal function based on strategy ID.

    These are simplified versions of the actual strategies for testing.
    In production, we would use the full strategy.analyze() method.
    """
    if strategy_id == 0:  # TailHedge
        def signal_fn(df, idx):
            if idx < 20:
                return 0
            # Buy protection when volatility is low
            returns = np.diff(np.log(df['close'].iloc[:idx+1].values[-20:]))
            vol = np.std(returns) * np.sqrt(252) * 100
            if vol < 15:  # Low vol = buy protection
                return 1
            return 0
        return signal_fn

    elif strategy_id == 1:  # VolatilityHarvest
        def signal_fn(df, idx):
            if idx < 30:
                return 0
            # Long volatility when recent vol spike
            returns = np.diff(np.log(df['close'].iloc[:idx+1].values[-20:]))
            vol = np.std(returns) * np.sqrt(252) * 100
            if vol > 25:  # High vol = long vol
                return 1
            return 0
        return signal_fn

    elif strategy_id == 2:  # CrisisAlpha
        def signal_fn(df, idx):
            if idx < 20:
                return 0
            # Safe haven when drawdown
            close = df['close'].iloc[:idx+1].values
            max_close = np.max(close)
            drawdown = (close[-1] - max_close) / max_close
            if drawdown < -0.05:  # 5% drawdown = safe haven
                return 1
            return 0
        return signal_fn

    elif strategy_id == 3:  # MomentumExplosion
        def signal_fn(df, idx):
            if idx < 20:
                return 0
            # Momentum breakout
            returns_20d = df['close'].iloc[idx] / df['close'].iloc[idx-20] - 1
            if returns_20d > 0.03:
                return 1
            elif returns_20d < -0.03:
                return -1
            return 0
        return signal_fn

    elif strategy_id == 4:  # MeanReversion
        def signal_fn(df, idx):
            if idx < 50:
                return 0
            # Mean reversion on extreme moves
            close = df['close'].iloc[:idx+1].values
            ma_50 = np.mean(close[-50:])
            deviation = (close[-1] - ma_50) / ma_50
            if deviation < -0.05:  # Oversold
                return 1
            elif deviation > 0.05:  # Overbought
                return -1
            return 0
        return signal_fn

    elif strategy_id == 5:  # CorrelationBreakdown
        def signal_fn(df, idx):
            if idx < 30:
                return 0
            # Simplified: trend following on regime change
            returns_5d = df['close'].iloc[idx] / df['close'].iloc[idx-5] - 1
            returns_20d = df['close'].iloc[idx] / df['close'].iloc[idx-20] - 1
            # Divergence = regime change
            if abs(returns_5d - returns_20d) > 0.02:
                return 1 if returns_5d > 0 else -1
            return 0
        return signal_fn

    elif strategy_id == 6:  # InequalityArbitrage
        def signal_fn(df, idx):
            if idx < 10:
                return 0
            # Simplified: contrarian on extreme moves
            returns_5d = df['close'].iloc[idx] / df['close'].iloc[idx-5] - 1
            if returns_5d < -0.03:
                return 1  # Buy dip
            elif returns_5d > 0.03:
                return -1  # Sell rip
            return 0
        return signal_fn

    elif strategy_id == 7:  # EventCatalyst
        def signal_fn(df, idx):
            if idx < 10:
                return 0
            # Simplified: momentum on volume spike
            if 'volume' not in df.columns:
                return 0
            vol_ratio = df['volume'].iloc[idx] / df['volume'].iloc[idx-10:idx].mean()
            if vol_ratio > 1.5:
                returns_1d = df['close'].iloc[idx] / df['close'].iloc[idx-1] - 1
                return 1 if returns_1d > 0 else -1
            return 0
        return signal_fn

    else:
        def signal_fn(df, idx):
            return 0
        return signal_fn


def test_strategy_mcpt(
    strategy_spec: StrategySpec,
    data: pd.DataFrame,
    n_permutations: int = 100,
) -> dict:
    """
    Test a single strategy with MCPT.

    Args:
        strategy_spec: Strategy specification
        data: OHLCV DataFrame
        n_permutations: Number of permutations for testing

    Returns:
        Dict with validation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {strategy_spec.name} (ID: {strategy_spec.strategy_id})")
    logger.info(f"{'='*60}")

    # Create signal function
    signal_fn = create_simple_signal_function(strategy_spec.strategy_id)

    # Build positions
    positions = np.array([signal_fn(data, i) for i in range(len(data))])

    # Calculate returns
    returns = strategy_returns(data['close'].values, positions)

    # Compute baseline metrics
    metrics = compute_all_metrics(returns)
    logger.info(f"Baseline metrics:")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    logger.info(f"  N Trades: {metrics['n_trades']}")

    # Run MCPT validation
    validator = MCPTValidator(n_permutations=n_permutations)

    # Quick p-value estimate
    p_value = validator.quick_validate(positions, data['close'].values, n_permutations)

    logger.info(f"\nMCPT Results:")
    logger.info(f"  p-value: {p_value:.4f}")
    logger.info(f"  Passes G0-G2: {p_value < 0.05}")
    logger.info(f"  Passes G3-G5: {p_value < 0.02}")
    logger.info(f"  Passes G6-G8: {p_value < 0.01}")

    return {
        'strategy_name': strategy_spec.name,
        'strategy_id': strategy_spec.strategy_id,
        'profit_factor': metrics['profit_factor'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'win_rate': metrics['win_rate'],
        'max_drawdown': metrics['max_drawdown'],
        'n_trades': metrics['n_trades'],
        'p_value': p_value,
        'passes_g2': p_value < 0.05,
        'passes_g5': p_value < 0.02,
        'passes_g8': p_value < 0.01,
    }


def main():
    parser = argparse.ArgumentParser(description='Test MCPT on existing strategies')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer permutations')
    parser.add_argument('--strategy', type=str, help='Test specific strategy by name')
    parser.add_argument('--n-bars', type=int, default=500, help='Number of bars to generate')
    parser.add_argument('--n-permutations', type=int, default=100, help='Number of permutations')
    args = parser.parse_args()

    n_permutations = 20 if args.quick else args.n_permutations

    logger.info("="*60)
    logger.info("MCPT Strategy Validation Test")
    logger.info("="*60)
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"N bars: {args.n_bars}")
    logger.info(f"N permutations: {n_permutations}")

    # Generate synthetic data
    logger.info("\nGenerating synthetic market data...")
    data = generate_synthetic_data(n_bars=args.n_bars)
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")

    # Test strategies
    results = []

    if args.strategy:
        # Test specific strategy
        strategies = [s for s in EXISTING_STRATEGIES if s.name == args.strategy]
        if not strategies:
            logger.error(f"Strategy not found: {args.strategy}")
            logger.info(f"Available: {[s.name for s in EXISTING_STRATEGIES]}")
            return
    else:
        strategies = EXISTING_STRATEGIES

    for spec in strategies:
        result = test_strategy_mcpt(spec, data, n_permutations)
        results.append(result)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    passed_g2 = sum(1 for r in results if r['passes_g2'])
    passed_g5 = sum(1 for r in results if r['passes_g5'])
    passed_g8 = sum(1 for r in results if r['passes_g8'])

    logger.info(f"Strategies tested: {len(results)}")
    logger.info(f"Passed G0-G2 (p < 0.05): {passed_g2}/{len(results)}")
    logger.info(f"Passed G3-G5 (p < 0.02): {passed_g5}/{len(results)}")
    logger.info(f"Passed G6-G8 (p < 0.01): {passed_g8}/{len(results)}")

    logger.info("\nDetailed Results:")
    for r in results:
        status = "PASS" if r['passes_g2'] else "FAIL"
        logger.info(f"  {r['strategy_name']:25} p={r['p_value']:.3f} PF={r['profit_factor']:.2f} [{status}]")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = project_root / 'outputs' / 'mcpt_validation_results.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
