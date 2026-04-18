"""
Parameter Sensitivity Analysis

Tests strategy robustness across parameter variations.
Computes cliff score to detect overfitting to specific parameters.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import itertools
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridSearchResult:
    """Result of parameter grid search."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    cliff_score: float
    heatmap_data: Optional[np.ndarray] = None
    param_names: Optional[List[str]] = None


def run_parameter_grid(
    strategy_fn: Callable,
    data: Any,
    param_grid: Dict[str, List],
    objective_fn: Callable,
    verbose: bool = False,
) -> GridSearchResult:
    """
    Run parameter grid search for a strategy.

    Args:
        strategy_fn: Function(data, **params) -> positions array
        data: Market data (DataFrame or dict)
        param_grid: Dict mapping param names to lists of values
        objective_fn: Function(bar_returns) -> float score
        verbose: Whether to log progress

    Returns:
        GridSearchResult with best params, scores, and cliff score
    """
    from .backtest_core import compute_bar_returns, BacktestAssumptions

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    n_combinations = len(all_combinations)

    if verbose:
        logger.info(f"Running grid search with {n_combinations} combinations")

    results = []
    scores = []

    for i, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))

        try:
            # Run strategy with these params
            positions = strategy_fn(data, **params)

            # Compute returns
            close = data['close'].values if hasattr(data, 'values') else data['close']
            bar_returns = compute_bar_returns(positions, close, BacktestAssumptions())

            # Compute objective
            score = objective_fn(bar_returns)

            results.append({
                'params': params,
                'score': score,
                'n_trades': int(np.sum(np.abs(np.diff(positions)) > 0)),
            })
            scores.append(score)

        except Exception as e:
            logger.debug(f"Params {params} failed: {e}")
            results.append({
                'params': params,
                'score': float('-inf'),
                'error': str(e),
            })
            scores.append(float('-inf'))

        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Completed {i + 1}/{n_combinations} combinations")

    # Find best
    scores_array = np.array(scores)
    valid_mask = np.isfinite(scores_array)

    if not np.any(valid_mask):
        logger.warning("No valid parameter combinations found")
        return GridSearchResult(
            best_params={},
            best_score=float('-inf'),
            all_results=results,
            cliff_score=0.0,
        )

    best_idx = np.argmax(np.where(valid_mask, scores_array, -np.inf))
    best_params = results[best_idx]['params']
    best_score = results[best_idx]['score']

    # Compute cliff score
    cliff_score = compute_cliff_score_from_results(results, best_score)

    # Build heatmap data (if 2D grid)
    heatmap_data = None
    if len(param_names) == 2:
        heatmap_data = _build_heatmap_2d(results, param_grid, param_names)

    return GridSearchResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results,
        cliff_score=cliff_score,
        heatmap_data=heatmap_data,
        param_names=param_names,
    )


def compute_cliff_score(
    grid_results: List[Dict],
    best_score: float,
    threshold_pct: float = 0.80,
) -> float:
    """
    Compute cliff score: % of neighboring cells within threshold of best.

    A high cliff score (> 0.3) indicates robust parameters.
    A low cliff score (< 0.1) indicates overfitting to specific params.

    Args:
        grid_results: List of dicts with 'params' and 'score'
        best_score: Best achieved score
        threshold_pct: What % of best score counts as "close" (default 80%)

    Returns:
        Cliff score between 0 and 1
    """
    return compute_cliff_score_from_results(grid_results, best_score, threshold_pct)


def compute_cliff_score_from_results(
    results: List[Dict],
    best_score: float,
    threshold_pct: float = 0.80,
) -> float:
    """
    Internal cliff score computation.

    Measures what fraction of parameter combinations achieve
    at least threshold_pct of the best score.
    """
    if best_score <= 0 or not np.isfinite(best_score):
        return 0.0

    threshold = best_score * threshold_pct
    valid_scores = [r['score'] for r in results if np.isfinite(r['score'])]

    if len(valid_scores) == 0:
        return 0.0

    above_threshold = sum(1 for s in valid_scores if s >= threshold)
    cliff_score = above_threshold / len(valid_scores)

    return cliff_score


def _build_heatmap_2d(
    results: List[Dict],
    param_grid: Dict[str, List],
    param_names: List[str],
) -> np.ndarray:
    """Build 2D heatmap from grid search results."""
    p1_vals = param_grid[param_names[0]]
    p2_vals = param_grid[param_names[1]]

    heatmap = np.full((len(p1_vals), len(p2_vals)), np.nan)

    for r in results:
        if not np.isfinite(r['score']):
            continue
        try:
            i = p1_vals.index(r['params'][param_names[0]])
            j = p2_vals.index(r['params'][param_names[1]])
            heatmap[i, j] = r['score']
        except (ValueError, KeyError):
            continue

    return heatmap


def export_heatmap_artifact(
    grid_result: GridSearchResult,
    out_path: str,
    title: str = "Parameter Sensitivity Heatmap",
):
    """
    Export parameter sensitivity heatmap as image artifact.

    Args:
        grid_result: GridSearchResult from run_parameter_grid
        out_path: Output file path (e.g., "artifacts/param_heatmap.png")
        title: Plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap export")
        return

    if grid_result.heatmap_data is None:
        logger.warning("No 2D heatmap data available")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(grid_result.heatmap_data, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Objective Score')

    ax.set_title(f"{title}\nCliff Score: {grid_result.cliff_score:.2f}")

    if grid_result.param_names:
        ax.set_xlabel(grid_result.param_names[1])
        ax.set_ylabel(grid_result.param_names[0])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Heatmap saved to {out_path}")


def sensitivity_1d(
    strategy_fn: Callable,
    data: Any,
    param_name: str,
    param_values: List,
    objective_fn: Callable,
    base_params: Optional[Dict] = None,
) -> Dict:
    """
    Compute 1D sensitivity for a single parameter.

    Args:
        strategy_fn: Strategy function
        data: Market data
        param_name: Name of parameter to vary
        param_values: List of values to test
        objective_fn: Objective function
        base_params: Base parameters (param_name will be overridden)

    Returns:
        Dict with values, scores, and sensitivity metrics
    """
    from .backtest_core import compute_bar_returns, BacktestAssumptions

    base_params = base_params or {}
    scores = []

    for val in param_values:
        params = {**base_params, param_name: val}

        try:
            positions = strategy_fn(data, **params)
            close = data['close'].values if hasattr(data, 'values') else data['close']
            bar_returns = compute_bar_returns(positions, close, BacktestAssumptions())
            score = objective_fn(bar_returns)
            scores.append(score)
        except Exception:
            scores.append(float('nan'))

    scores_array = np.array(scores)
    valid_mask = np.isfinite(scores_array)

    if np.sum(valid_mask) < 2:
        return {
            'param_name': param_name,
            'values': param_values,
            'scores': scores,
            'sensitivity': 0.0,
            'stable_range': [],
        }

    # Sensitivity = coefficient of variation
    valid_scores = scores_array[valid_mask]
    sensitivity = np.std(valid_scores) / (np.abs(np.mean(valid_scores)) + 1e-8)

    # Find stable range (values within 80% of max)
    max_score = np.max(valid_scores)
    threshold = max_score * 0.8
    stable_indices = np.where((scores_array >= threshold) & valid_mask)[0]
    stable_range = [param_values[i] for i in stable_indices]

    return {
        'param_name': param_name,
        'values': param_values,
        'scores': scores,
        'sensitivity': float(sensitivity),
        'stable_range': stable_range,
        'best_value': param_values[np.argmax(np.where(valid_mask, scores_array, -np.inf))],
        'best_score': float(max_score),
    }


if __name__ == "__main__":
    # Test parameter sensitivity
    print("=== Parameter Sensitivity Test ===")

    import pandas as pd

    np.random.seed(42)

    # Generate synthetic data
    n = 500
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    data = pd.DataFrame({'close': close})

    # Simple momentum strategy
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

    # Run grid search
    param_grid = {
        'lookback': [10, 15, 20, 25, 30],
        'threshold': [0.01, 0.02, 0.03, 0.04, 0.05],
    }

    from .backtest_core import objective_profit_factor

    result = run_parameter_grid(
        momentum_strategy,
        data,
        param_grid,
        objective_profit_factor,
        verbose=True,
    )

    print(f"\nBest params: {result.best_params}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Cliff score: {result.cliff_score:.3f}")

    print("\n=== Test Complete ===")
