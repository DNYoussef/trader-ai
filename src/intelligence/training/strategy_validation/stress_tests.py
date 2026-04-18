"""
Stress Tests for Strategy Robustness

Tests strategy performance under adverse conditions:
- Slippage sensitivity
- Execution delay sensitivity
- Fee sensitivity
- Combined worst-case scenarios
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .backtest_core import (
    BacktestAssumptions,
    compute_bar_returns,
    objective_profit_factor,
    objective_sharpe,
)

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Result of a stress test."""
    test_type: str
    param_values: List
    scores: List[float]
    baseline_score: float
    degradation_at_max: float  # % degradation at max stress
    break_even_point: Optional[float]  # Where score crosses 1.0 (for PF) or 0 (for Sharpe)
    is_robust: bool  # True if strategy survives all stress levels


def stress_slippage(
    strategy_fn: Callable,
    data: Any,
    strategy_params: Dict,
    slippage_bps_list: List[float] = [1, 2, 5, 10, 20, 50],
    objective_fn: Callable = objective_profit_factor,
) -> StressTestResult:
    """
    Test strategy sensitivity to slippage.

    Args:
        strategy_fn: Function(data, **params) -> positions
        data: Market data
        strategy_params: Strategy parameters
        slippage_bps_list: List of slippage values to test (in bps)
        objective_fn: Objective function

    Returns:
        StressTestResult with degradation analysis
    """
    close = data['close'].values if hasattr(data, 'values') else data['close']
    positions = strategy_fn(data, **strategy_params)

    scores = []
    baseline_score = None

    for slippage in slippage_bps_list:
        assumptions = BacktestAssumptions(
            slippage_bps=slippage,
            fee_bps=10.0,  # Fixed
            delay_bars=1,
            spread_bps=2.0,
        )

        bar_returns = compute_bar_returns(positions, close, assumptions)
        score = objective_fn(bar_returns)
        scores.append(score)

        if slippage == slippage_bps_list[0]:
            baseline_score = score

    # Compute degradation
    baseline_score = scores[0] if baseline_score is None else baseline_score
    degradation = (baseline_score - scores[-1]) / baseline_score if baseline_score > 0 else 0

    # Find break-even point
    break_even = _find_break_even(slippage_bps_list, scores, objective_fn)

    # Is robust? (maintains positive edge at 2x base slippage)
    is_robust = len(scores) >= 2 and scores[min(2, len(scores)-1)] > 1.0

    return StressTestResult(
        test_type='slippage',
        param_values=slippage_bps_list,
        scores=scores,
        baseline_score=baseline_score,
        degradation_at_max=degradation,
        break_even_point=break_even,
        is_robust=is_robust,
    )


def stress_delay(
    strategy_fn: Callable,
    data: Any,
    strategy_params: Dict,
    delay_bars_list: List[int] = [0, 1, 2, 3, 5],
    objective_fn: Callable = objective_profit_factor,
) -> StressTestResult:
    """
    Test strategy sensitivity to execution delay.

    Args:
        strategy_fn: Function(data, **params) -> positions
        data: Market data
        strategy_params: Strategy parameters
        delay_bars_list: List of delay values to test
        objective_fn: Objective function

    Returns:
        StressTestResult with degradation analysis
    """
    close = data['close'].values if hasattr(data, 'values') else data['close']
    positions = strategy_fn(data, **strategy_params)

    scores = []

    for delay in delay_bars_list:
        assumptions = BacktestAssumptions(
            slippage_bps=5.0,
            fee_bps=10.0,
            delay_bars=delay,
            spread_bps=2.0,
        )

        bar_returns = compute_bar_returns(positions, close, assumptions)
        score = objective_fn(bar_returns)
        scores.append(score)

    baseline_score = scores[0]
    degradation = (baseline_score - scores[-1]) / baseline_score if baseline_score > 0 else 0

    break_even = _find_break_even(delay_bars_list, scores, objective_fn)
    is_robust = len(scores) >= 2 and scores[1] > 1.0  # Survives 1-bar delay

    return StressTestResult(
        test_type='delay',
        param_values=delay_bars_list,
        scores=scores,
        baseline_score=baseline_score,
        degradation_at_max=degradation,
        break_even_point=break_even,
        is_robust=is_robust,
    )


def stress_fee(
    strategy_fn: Callable,
    data: Any,
    strategy_params: Dict,
    fee_bps_list: List[float] = [0, 5, 10, 20, 50, 100],
    objective_fn: Callable = objective_profit_factor,
) -> StressTestResult:
    """
    Test strategy sensitivity to trading fees.

    Args:
        strategy_fn: Function(data, **params) -> positions
        data: Market data
        strategy_params: Strategy parameters
        fee_bps_list: List of fee values to test (in bps)
        objective_fn: Objective function

    Returns:
        StressTestResult with degradation analysis
    """
    close = data['close'].values if hasattr(data, 'values') else data['close']
    positions = strategy_fn(data, **strategy_params)

    scores = []

    for fee in fee_bps_list:
        assumptions = BacktestAssumptions(
            slippage_bps=5.0,
            fee_bps=fee,
            delay_bars=1,
            spread_bps=2.0,
        )

        bar_returns = compute_bar_returns(positions, close, assumptions)
        score = objective_fn(bar_returns)
        scores.append(score)

    baseline_score = scores[1] if len(scores) > 1 else scores[0]  # Use 5bps as baseline
    degradation = (baseline_score - scores[-1]) / baseline_score if baseline_score > 0 else 0

    break_even = _find_break_even(fee_bps_list, scores, objective_fn)
    is_robust = len(scores) >= 3 and scores[2] > 1.0  # Survives 10bps fees

    return StressTestResult(
        test_type='fee',
        param_values=fee_bps_list,
        scores=scores,
        baseline_score=baseline_score,
        degradation_at_max=degradation,
        break_even_point=break_even,
        is_robust=is_robust,
    )


def stress_combined(
    strategy_fn: Callable,
    data: Any,
    strategy_params: Dict,
    stress_levels: List[str] = ['low', 'medium', 'high', 'extreme'],
    objective_fn: Callable = objective_profit_factor,
) -> StressTestResult:
    """
    Combined stress test with simultaneous adverse conditions.

    Stress levels:
    - low: 2bps slip, 5bps fee, 0 delay
    - medium: 5bps slip, 10bps fee, 1 delay
    - high: 10bps slip, 20bps fee, 2 delay
    - extreme: 20bps slip, 50bps fee, 3 delay

    Args:
        strategy_fn: Function(data, **params) -> positions
        data: Market data
        strategy_params: Strategy parameters
        stress_levels: List of stress levels to test
        objective_fn: Objective function

    Returns:
        StressTestResult with degradation analysis
    """
    STRESS_CONFIGS = {
        'low': BacktestAssumptions(slippage_bps=2, fee_bps=5, delay_bars=0, spread_bps=1),
        'medium': BacktestAssumptions(slippage_bps=5, fee_bps=10, delay_bars=1, spread_bps=2),
        'high': BacktestAssumptions(slippage_bps=10, fee_bps=20, delay_bars=2, spread_bps=5),
        'extreme': BacktestAssumptions(slippage_bps=20, fee_bps=50, delay_bars=3, spread_bps=10),
    }

    close = data['close'].values if hasattr(data, 'values') else data['close']
    positions = strategy_fn(data, **strategy_params)

    scores = []

    for level in stress_levels:
        assumptions = STRESS_CONFIGS.get(level, STRESS_CONFIGS['medium'])
        bar_returns = compute_bar_returns(positions, close, assumptions)
        score = objective_fn(bar_returns)
        scores.append(score)

    baseline_score = scores[0]
    degradation = (baseline_score - scores[-1]) / baseline_score if baseline_score > 0 else 0

    # Strategy is robust if it survives "high" stress
    high_idx = stress_levels.index('high') if 'high' in stress_levels else min(2, len(scores)-1)
    is_robust = scores[high_idx] > 1.0

    return StressTestResult(
        test_type='combined',
        param_values=stress_levels,
        scores=scores,
        baseline_score=baseline_score,
        degradation_at_max=degradation,
        break_even_point=None,  # Not applicable for categorical levels
        is_robust=is_robust,
    )


def _find_break_even(
    param_values: List,
    scores: List[float],
    objective_fn: Callable,
) -> Optional[float]:
    """
    Find the parameter value where objective crosses break-even.

    For profit_factor: break-even is 1.0
    For sharpe: break-even is 0.0
    """
    # Determine threshold based on objective function
    if objective_fn == objective_profit_factor:
        threshold = 1.0
    elif objective_fn == objective_sharpe:
        threshold = 0.0
    else:
        threshold = 1.0  # Default

    # Find crossing point
    for i in range(len(scores) - 1):
        if scores[i] >= threshold and scores[i + 1] < threshold:
            # Linear interpolation
            if scores[i] != scores[i + 1]:
                frac = (scores[i] - threshold) / (scores[i] - scores[i + 1])
                break_even = param_values[i] + frac * (param_values[i + 1] - param_values[i])
                return float(break_even)

    return None


def run_all_stress_tests(
    strategy_fn: Callable,
    data: Any,
    strategy_params: Dict,
    objective_fn: Callable = objective_profit_factor,
) -> Dict[str, StressTestResult]:
    """
    Run all stress tests for a strategy.

    Args:
        strategy_fn: Strategy function
        data: Market data
        strategy_params: Strategy parameters
        objective_fn: Objective function

    Returns:
        Dict mapping test type to StressTestResult
    """
    results = {}

    results['slippage'] = stress_slippage(
        strategy_fn, data, strategy_params, objective_fn=objective_fn
    )
    results['delay'] = stress_delay(
        strategy_fn, data, strategy_params, objective_fn=objective_fn
    )
    results['fee'] = stress_fee(
        strategy_fn, data, strategy_params, objective_fn=objective_fn
    )
    results['combined'] = stress_combined(
        strategy_fn, data, strategy_params, objective_fn=objective_fn
    )

    return results


def stress_test_summary(results: Dict[str, StressTestResult]) -> Dict:
    """
    Generate summary of all stress test results.

    Args:
        results: Dict from run_all_stress_tests

    Returns:
        Summary dict with pass/fail and key metrics
    """
    all_robust = all(r.is_robust for r in results.values())

    min_score = min(r.scores[-1] for r in results.values())
    max_degradation = max(r.degradation_at_max for r in results.values())

    return {
        'all_robust': all_robust,
        'min_score_at_max_stress': min_score,
        'max_degradation_pct': max_degradation,
        'slippage_robust': results['slippage'].is_robust,
        'delay_robust': results['delay'].is_robust,
        'fee_robust': results['fee'].is_robust,
        'combined_robust': results['combined'].is_robust,
        'slippage_break_even_bps': results['slippage'].break_even_point,
        'delay_break_even_bars': results['delay'].break_even_point,
        'fee_break_even_bps': results['fee'].break_even_point,
    }


if __name__ == "__main__":
    # Test stress tests
    print("=== Stress Tests Demo ===")

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

    # Run all stress tests
    params = {'lookback': 20, 'threshold': 0.02}
    results = run_all_stress_tests(momentum_strategy, data, params)

    print("\nStress Test Results:")
    for test_type, result in results.items():
        print(f"\n{test_type.upper()}:")
        print(f"  Baseline: {result.baseline_score:.3f}")
        print(f"  Scores: {[f'{s:.3f}' for s in result.scores]}")
        print(f"  Degradation: {result.degradation_at_max:.1%}")
        print(f"  Break-even: {result.break_even_point}")
        print(f"  Robust: {result.is_robust}")

    summary = stress_test_summary(results)
    print(f"\nSummary: {summary}")

    print("\n=== Test Complete ===")
