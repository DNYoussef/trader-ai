#!/usr/bin/env python3
"""
Strategy Lab Training CLI

Unified script for:
1. Training strategy selector (Phase A)
2. Validating strategies with MCPT + robustness battery
3. Generating and validating new strategy candidates (Phase B)

Usage:
    # Train strategy selector
    python scripts/training/train_strategy_lab.py --mode select --data-path data/ohlcv.parquet

    # Validate existing strategies
    python scripts/training/train_strategy_lab.py --mode validate --strategy momentum

    # Generate and validate new strategies
    python scripts/training/train_strategy_lab.py --mode generate --template donchian
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'third_party'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load OHLCV data from various formats."""
    path = Path(data_path)

    if not path.exists():
        # Generate synthetic data for testing
        logger.warning(f"Data file not found: {data_path}. Generating synthetic data.")
        return generate_synthetic_data()

    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_synthetic_data(n_bars: int = 252 * 5, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Generate realistic price series
    returns = np.random.randn(n_bars) * 0.01
    for i in range(1, n_bars):
        returns[i] += 0.1 * returns[i-1]  # Momentum

    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = np.roll(close, 1) + np.random.randn(n_bars) * 0.1
    open_price[0] = close[0]
    volume = np.random.uniform(1e6, 5e6, n_bars)

    return pd.DataFrame({
        'date': pd.date_range(start='2019-01-01', periods=n_bars, freq='D'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'symbol': 'SPY',
    })


def mode_select(args):
    """Train strategy selector model."""
    from src.intelligence.strategy_lab.strategy_selector import StrategySelector
    from src.intelligence.training.selector_trainer import (
        StrategySelectorTrainer, TrainingConfig, create_default_signal_functions
    )
    from src.data.strategy_labeler import (
        generate_black_swan_labels, create_default_black_swan_signals
    )

    logger.info("="*60)
    logger.info("STRATEGY SELECTOR TRAINING")
    logger.info("="*60)

    # Load data
    data = load_data(args.data_path)
    logger.info(f"Loaded data: {len(data)} bars")

    # Generate labels
    signal_fns = create_default_black_swan_signals()

    logger.info("Generating strategy labels...")
    labels = generate_black_swan_labels(
        data, signal_fns,
        forward_horizon=args.forward_horizon,
        objective=args.objective,
    )

    # Generate features (simplified - use actual feature generator in production)
    logger.info("Generating features...")
    n_samples = len(labels)
    X = np.random.randn(n_samples, 38)  # Placeholder - use actual features

    # Training config
    config = TrainingConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        n_splits=args.n_splits,
        forward_horizon=args.forward_horizon,
        label_metric=args.objective,
    )

    # Train
    trainer = StrategySelectorTrainer(config)
    result = trainer.train_walk_forward(X, labels, save_path=args.output_path)

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Train accuracy: {result.train_accuracy:.3f}")
    logger.info(f"Test accuracy: {result.test_accuracy:.3f}")
    logger.info(f"Model saved to: {result.model_path}")

    return result


def mode_validate(args):
    """Validate strategies with full robustness battery."""
    from src.intelligence.training.strategy_validation import (
        ValidationBattery, ValidationConfig, run_full_validation
    )

    logger.info("="*60)
    logger.info("STRATEGY VALIDATION")
    logger.info("="*60)

    # Load data
    data = load_data(args.data_path)
    logger.info(f"Loaded data: {len(data)} bars")

    # Get strategy function
    strategy_fn, param_grid = get_strategy_by_name(args.strategy)

    if strategy_fn is None:
        logger.error(f"Unknown strategy: {args.strategy}")
        return None

    # Validation config
    config = ValidationConfig(
        train_window_bars=args.train_window,
        step_bars=args.step_bars,
        mcpt_is_perms=args.mcpt_perms,
        mcpt_wf_perms=args.wf_mcpt_perms,
        mc_paths=args.mc_paths,
    )

    # Run validation
    result = run_full_validation(
        strategy_id=f"{args.strategy}_{datetime.now().strftime('%Y%m%d')}",
        template_name=args.strategy,
        strategy_fn=strategy_fn,
        data=data,
        param_grid=param_grid,
        config=config,
        mlflow_log=not args.no_mlflow,
    )

    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULT")
    logger.info("="*60)
    logger.info(f"Passed: {result.passed}")
    logger.info(f"IS Profit Factor: {result.is_profit_factor:.3f}")
    logger.info(f"WF Profit Factor: {result.wf_profit_factor:.3f}")
    logger.info(f"MCPT IS p-value: {result.mcpt_is_pvalue:.4f}")
    logger.info(f"MCPT WF p-value: {result.mcpt_wf_pvalue:.4f}")
    logger.info(f"Cliff Score: {result.cliff_score:.3f}")
    logger.info(f"MC Ruin Prob: {result.mc_ruin_prob:.1%}")

    if not result.passed:
        logger.warning(f"Failure reasons: {result.failure_reasons}")

    # Save result
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Result saved to: {output_path}")

    return result


def mode_generate(args):
    """Generate and validate new strategy candidates."""
    from src.intelligence.training.strategy_validation import (
        ValidationBattery, ValidationConfig
    )

    logger.info("="*60)
    logger.info("STRATEGY GENERATION")
    logger.info("="*60)

    # Load data
    data = load_data(args.data_path)
    logger.info(f"Loaded data: {len(data)} bars")

    # Get template
    template = get_template_by_name(args.template)
    if template is None:
        logger.error(f"Unknown template: {args.template}")
        return None

    strategy_fn = template['fn']
    param_ranges = template['params']

    # Generate candidates
    logger.info(f"Generating {args.n_candidates} candidates...")
    candidates = generate_candidates(param_ranges, args.n_candidates, args.seed)

    # Validation config
    config = ValidationConfig(
        train_window_bars=args.train_window,
        step_bars=args.step_bars,
        mcpt_is_perms=min(args.mcpt_perms, 100),  # Reduce for screening
        mcpt_wf_perms=min(args.wf_mcpt_perms, 50),
        mc_paths=min(args.mc_paths, 200),
    )

    battery = ValidationBattery(config)

    # Validate each candidate
    survivors = []
    for i, params in enumerate(candidates):
        logger.info(f"\nCandidate {i+1}/{len(candidates)}: {params}")

        try:
            result = battery.run(
                strategy_id=f"{args.template}_{i}",
                template_name=args.template,
                strategy_fn=strategy_fn,
                data=data,
                best_params=params,
            )

            if result.passed:
                survivors.append({
                    'params': params,
                    'result': result.to_dict(),
                })
                logger.info(f"  PASSED (PF={result.is_profit_factor:.2f})")
            else:
                logger.info(f"  FAILED: {result.failure_reasons[:2]}")

        except Exception as e:
            logger.warning(f"  ERROR: {e}")

    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Candidates tested: {len(candidates)}")
    logger.info(f"Survivors: {len(survivors)}")
    logger.info(f"Survival rate: {len(survivors)/len(candidates):.1%}")

    # Save survivors
    if args.output_path and survivors:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(survivors, f, indent=2, default=str)
        logger.info(f"Survivors saved to: {output_path}")

    return survivors


def get_strategy_by_name(name: str):
    """Get strategy function and param grid by name."""
    STRATEGIES = {
        'momentum': {
            'fn': momentum_strategy,
            'grid': {
                'lookback': [10, 15, 20, 25, 30],
                'threshold': [0.01, 0.02, 0.03, 0.04],
            }
        },
        'mean_reversion': {
            'fn': mean_reversion_strategy,
            'grid': {
                'lookback': [20, 30, 50, 75],
                'z_threshold': [1.5, 2.0, 2.5, 3.0],
            }
        },
        'donchian': {
            'fn': donchian_strategy,
            'grid': {
                'lookback': [10, 20, 30, 50, 75],
            }
        },
    }

    if name not in STRATEGIES:
        return None, None

    return STRATEGIES[name]['fn'], STRATEGIES[name]['grid']


def get_template_by_name(name: str):
    """Get strategy template by name."""
    TEMPLATES = {
        'momentum': {
            'fn': momentum_strategy,
            'params': {
                'lookback': {'min': 5, 'max': 50, 'type': int},
                'threshold': {'min': 0.005, 'max': 0.10, 'type': float},
            }
        },
        'donchian': {
            'fn': donchian_strategy,
            'params': {
                'lookback': {'min': 5, 'max': 100, 'type': int},
            }
        },
        'mean_reversion': {
            'fn': mean_reversion_strategy,
            'params': {
                'lookback': {'min': 10, 'max': 100, 'type': int},
                'z_threshold': {'min': 1.0, 'max': 4.0, 'type': float},
            }
        },
    }

    return TEMPLATES.get(name)


def generate_candidates(param_ranges: dict, n: int, seed: int) -> list:
    """Generate random parameter combinations."""
    np.random.seed(seed)
    candidates = []

    for _ in range(n):
        params = {}
        for name, spec in param_ranges.items():
            if spec['type'] == int:
                params[name] = np.random.randint(spec['min'], spec['max'] + 1)
            else:
                params[name] = np.random.uniform(spec['min'], spec['max'])
        candidates.append(params)

    return candidates


# Strategy implementations
def momentum_strategy(data, lookback=20, threshold=0.02):
    close = data['close'].values
    n = len(close)
    positions = np.zeros(n)
    for i in range(lookback, n):
        ret = close[i] / close[i - lookback] - 1
        if ret > threshold:
            positions[i] = 1
        elif ret < -threshold:
            positions[i] = -1
    return positions


def donchian_strategy(data, lookback=20):
    close = data['close'].values
    high = data['high'].values if 'high' in data.columns else close
    low = data['low'].values if 'low' in data.columns else close
    n = len(close)
    positions = np.zeros(n)

    for i in range(lookback, n):
        upper = np.max(high[i-lookback:i])
        lower = np.min(low[i-lookback:i])

        if close[i] > upper:
            positions[i] = 1
        elif close[i] < lower:
            positions[i] = -1
        else:
            positions[i] = positions[i-1] if i > 0 else 0

    return positions


def mean_reversion_strategy(data, lookback=50, z_threshold=2.0):
    close = data['close'].values
    n = len(close)
    positions = np.zeros(n)

    for i in range(lookback, n):
        window = close[i-lookback:i]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            continue

        z = (close[i] - mean) / std

        if z < -z_threshold:
            positions[i] = 1  # Oversold - buy
        elif z > z_threshold:
            positions[i] = -1  # Overbought - sell
        else:
            positions[i] = 0

    return positions


def main():
    parser = argparse.ArgumentParser(description='Strategy Lab Training CLI')

    parser.add_argument('--mode', required=True,
                       choices=['select', 'validate', 'generate'],
                       help='Operating mode')

    # Data
    parser.add_argument('--data-path', default='data/ohlcv.parquet',
                       help='Path to OHLCV data')
    parser.add_argument('--output-path', default=None,
                       help='Path to save output')

    # Strategy selection
    parser.add_argument('--strategy', default='momentum',
                       help='Strategy name for validation')
    parser.add_argument('--template', default='donchian',
                       help='Template name for generation')

    # Training params
    parser.add_argument('--train-window', type=int, default=252*4,
                       help='Training window in bars')
    parser.add_argument('--test-window', type=int, default=63,
                       help='Test window in bars')
    parser.add_argument('--step-bars', type=int, default=63,
                       help='Walk-forward step size')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of walk-forward splits')
    parser.add_argument('--forward-horizon', type=int, default=20,
                       help='Forward horizon for labeling')
    parser.add_argument('--objective', default='sharpe',
                       choices=['sharpe', 'return', 'profit_factor'],
                       help='Objective for optimization')

    # MCPT
    parser.add_argument('--mcpt-perms', type=int, default=1000,
                       help='Number of in-sample MCPT permutations')
    parser.add_argument('--wf-mcpt-perms', type=int, default=200,
                       help='Number of walk-forward MCPT permutations')

    # Monte Carlo
    parser.add_argument('--mc-paths', type=int, default=500,
                       help='Number of Monte Carlo paths')

    # Generation
    parser.add_argument('--n-candidates', type=int, default=100,
                       help='Number of candidates to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # MLflow
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow logging')

    args = parser.parse_args()

    # Route to mode
    if args.mode == 'select':
        mode_select(args)
    elif args.mode == 'validate':
        mode_validate(args)
    elif args.mode == 'generate':
        mode_generate(args)


if __name__ == "__main__":
    main()
