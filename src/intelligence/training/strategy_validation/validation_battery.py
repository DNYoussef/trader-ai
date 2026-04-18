"""
Validation Battery Orchestrator

Runs the complete validation suite for strategy robustness:
1. In-sample optimization
2. Parameter sensitivity (cliff score)
3. Walk-forward validation
4. MCPT (in-sample + walk-forward)
5. Stress tests
6. Monte Carlo simulations

All results are logged to MLflow for tracking and gate validation.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging

from .backtest_core import (
    BacktestAssumptions,
    compute_bar_returns,
    objective_profit_factor,
    objective_sharpe,
    objective_max_drawdown,
    objective_calmar,
    compute_equity_curve,
)
from .parameter_sensitivity import run_parameter_grid, compute_cliff_score
from .stress_tests import run_all_stress_tests, stress_test_summary
from .monte_carlo import shuffle_bar_returns_mc, block_bootstrap_mc

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Complete validation result for a strategy."""
    strategy_id: str
    template_name: str
    timestamp: str

    # In-sample metrics
    is_profit_factor: float
    is_sharpe: float
    is_max_drawdown: float
    is_calmar: float

    # Walk-forward metrics
    wf_profit_factor: float
    wf_sharpe: float
    wf_n_folds: int

    # MCPT
    mcpt_is_pvalue: float
    mcpt_wf_pvalue: float
    mcpt_passed: bool

    # Parameter sensitivity
    cliff_score: float
    best_params: Dict

    # Stress tests
    stress_all_robust: bool
    stress_min_score: float
    stress_max_degradation: float
    stress_slippage_break_even: Optional[float]

    # Monte Carlo
    mc_cagr_p5: float
    mc_maxdd_p95: float
    mc_ruin_prob: float

    # Overall
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'template_name': self.template_name,
            'timestamp': self.timestamp,
            'is_profit_factor': self.is_profit_factor,
            'is_sharpe': self.is_sharpe,
            'is_max_drawdown': self.is_max_drawdown,
            'is_calmar': self.is_calmar,
            'wf_profit_factor': self.wf_profit_factor,
            'wf_sharpe': self.wf_sharpe,
            'wf_n_folds': self.wf_n_folds,
            'mcpt_is_pvalue': self.mcpt_is_pvalue,
            'mcpt_wf_pvalue': self.mcpt_wf_pvalue,
            'mcpt_passed': self.mcpt_passed,
            'cliff_score': self.cliff_score,
            'best_params': self.best_params,
            'stress_all_robust': self.stress_all_robust,
            'stress_min_score': self.stress_min_score,
            'stress_max_degradation': self.stress_max_degradation,
            'stress_slippage_break_even': self.stress_slippage_break_even,
            'mc_cagr_p5': self.mc_cagr_p5,
            'mc_maxdd_p95': self.mc_maxdd_p95,
            'mc_ruin_prob': self.mc_ruin_prob,
            'passed': self.passed,
            'failure_reasons': self.failure_reasons,
        }


@dataclass
class ValidationConfig:
    """Configuration for validation battery."""
    # Walk-forward
    train_window_bars: int = 252 * 4  # 4 years
    step_bars: int = 63              # 3 months
    n_folds: int = 4

    # MCPT
    mcpt_is_perms: int = 1000
    mcpt_wf_perms: int = 200
    mcpt_is_threshold: float = 0.01
    mcpt_wf_threshold: float = 0.05

    # Parameter sensitivity
    min_cliff_score: float = 0.20

    # Stress tests
    require_stress_robust: bool = True

    # Monte Carlo
    mc_paths: int = 500
    mc_block_len: int = 20
    max_ruin_prob: float = 0.10

    # Thresholds
    min_profit_factor: float = 1.2
    min_sharpe: float = 0.5
    max_drawdown: float = 0.30


class ValidationBattery:
    """
    Complete validation battery for trading strategies.

    Orchestrates all validation tests and produces a unified result.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def run(
        self,
        strategy_id: str,
        template_name: str,
        strategy_fn: Callable,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List]] = None,
        best_params: Optional[Dict] = None,
    ) -> ValidationResult:
        """
        Run complete validation battery.

        Args:
            strategy_id: Unique strategy identifier
            template_name: Strategy template name
            strategy_fn: Function(data, **params) -> positions
            data: OHLCV DataFrame
            param_grid: Parameter grid for sensitivity analysis
            best_params: Pre-optimized parameters (if no grid search needed)

        Returns:
            ValidationResult with all metrics and pass/fail status
        """
        logger.info(f"Running validation battery for {strategy_id}")
        timestamp = datetime.now().isoformat()

        failure_reasons = []
        close = data['close'].values

        # 1. Parameter optimization (if grid provided)
        if param_grid and not best_params:
            logger.info("Running parameter grid search...")
            grid_result = run_parameter_grid(
                strategy_fn, data, param_grid,
                objective_profit_factor, verbose=True
            )
            best_params = grid_result.best_params
            cliff_score = grid_result.cliff_score
        else:
            best_params = best_params or {}
            cliff_score = 1.0  # No grid = assume stable

        if cliff_score < self.config.min_cliff_score:
            failure_reasons.append(f"cliff_score {cliff_score:.2f} < {self.config.min_cliff_score}")

        # 2. In-sample metrics
        logger.info("Computing in-sample metrics...")
        positions = strategy_fn(data, **best_params)
        bar_returns = compute_bar_returns(positions, close, BacktestAssumptions())
        equity = compute_equity_curve(bar_returns)

        is_pf = objective_profit_factor(bar_returns)
        is_sharpe = objective_sharpe(bar_returns)
        is_maxdd = objective_max_drawdown(equity)
        is_calmar = objective_calmar(bar_returns)

        if is_pf < self.config.min_profit_factor:
            failure_reasons.append(f"is_pf {is_pf:.2f} < {self.config.min_profit_factor}")
        if is_sharpe < self.config.min_sharpe:
            failure_reasons.append(f"is_sharpe {is_sharpe:.2f} < {self.config.min_sharpe}")
        if is_maxdd > self.config.max_drawdown:
            failure_reasons.append(f"is_maxdd {is_maxdd:.1%} > {self.config.max_drawdown:.1%}")

        # 3. Walk-forward validation
        logger.info("Running walk-forward validation...")
        wf_pf, wf_sharpe, wf_n_folds = self._run_walk_forward(
            strategy_fn, data, best_params
        )

        if wf_pf < 1.0:
            failure_reasons.append(f"wf_pf {wf_pf:.2f} < 1.0")

        # 4. MCPT validation
        logger.info("Running MCPT validation...")
        mcpt_is_p, mcpt_wf_p = self._run_mcpt(
            strategy_fn, data, best_params
        )

        mcpt_passed = (
            mcpt_is_p < self.config.mcpt_is_threshold and
            mcpt_wf_p < self.config.mcpt_wf_threshold
        )

        if not mcpt_passed:
            failure_reasons.append(f"mcpt failed: is_p={mcpt_is_p:.3f}, wf_p={mcpt_wf_p:.3f}")

        # 5. Stress tests
        logger.info("Running stress tests...")
        stress_results = run_all_stress_tests(
            strategy_fn, data, best_params
        )
        stress_summary = stress_test_summary(stress_results)

        if self.config.require_stress_robust and not stress_summary['all_robust']:
            failure_reasons.append("stress tests failed")

        # 6. Monte Carlo
        logger.info("Running Monte Carlo simulations...")
        shuffle_mc = shuffle_bar_returns_mc(
            bar_returns, n_paths=self.config.mc_paths
        )
        block_mc = block_bootstrap_mc(
            bar_returns, block_len=self.config.mc_block_len,
            n_paths=self.config.mc_paths
        )

        mc_cagr_p5 = block_mc['cagr'].p5
        mc_maxdd_p95 = block_mc['max_drawdown'].p95
        mc_ruin_prob = block_mc['max_drawdown'].ruin_probability

        if mc_ruin_prob > self.config.max_ruin_prob:
            failure_reasons.append(f"mc_ruin_prob {mc_ruin_prob:.1%} > {self.config.max_ruin_prob:.1%}")

        # Build result
        passed = len(failure_reasons) == 0

        result = ValidationResult(
            strategy_id=strategy_id,
            template_name=template_name,
            timestamp=timestamp,
            is_profit_factor=is_pf,
            is_sharpe=is_sharpe,
            is_max_drawdown=is_maxdd,
            is_calmar=is_calmar,
            wf_profit_factor=wf_pf,
            wf_sharpe=wf_sharpe,
            wf_n_folds=wf_n_folds,
            mcpt_is_pvalue=mcpt_is_p,
            mcpt_wf_pvalue=mcpt_wf_p,
            mcpt_passed=mcpt_passed,
            cliff_score=cliff_score,
            best_params=best_params,
            stress_all_robust=stress_summary['all_robust'],
            stress_min_score=stress_summary['min_score_at_max_stress'],
            stress_max_degradation=stress_summary['max_degradation_pct'],
            stress_slippage_break_even=stress_summary['slippage_break_even_bps'],
            mc_cagr_p5=mc_cagr_p5,
            mc_maxdd_p95=mc_maxdd_p95,
            mc_ruin_prob=mc_ruin_prob,
            passed=passed,
            failure_reasons=failure_reasons,
        )

        logger.info(f"Validation {'PASSED' if passed else 'FAILED'}: {failure_reasons}")

        return result

    def _run_walk_forward(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        params: Dict,
    ) -> Tuple[float, float, int]:
        """Run walk-forward validation."""
        close = data['close'].values
        n = len(close)

        train_window = self.config.train_window_bars
        step = self.config.step_bars

        # Pre-calculate total size to avoid O(n^2) memory operations
        total_test_bars = 0
        n_folds = 0
        start = 0
        while start + train_window + step <= n:
            test_start = start + train_window
            test_end = min(test_start + step, n)
            total_test_bars += (test_end - test_start)
            n_folds += 1
            start += step

        if total_test_bars == 0:
            return 0.0, 0.0, 0

        # Pre-allocate array for O(n) memory operations
        oos_returns = np.empty(total_test_bars, dtype=np.float64)
        current_idx = 0
        n_folds = 0

        start = 0
        while start + train_window + step <= n:
            # Train window
            train_end = start + train_window

            # Test window
            test_start = train_end
            test_end = min(test_start + step, n)

            # Get test positions
            test_data = data.iloc[test_start:test_end]
            positions = strategy_fn(test_data, **params)

            # Compute test returns
            bar_returns = compute_bar_returns(
                positions, test_data['close'].values, BacktestAssumptions()
            )

            # Use array slicing instead of list.extend()
            batch_size = len(bar_returns)
            oos_returns[current_idx:current_idx + batch_size] = bar_returns
            current_idx += batch_size

            n_folds += 1
            start += step

        wf_pf = objective_profit_factor(oos_returns)
        wf_sharpe = objective_sharpe(oos_returns)

        return wf_pf, wf_sharpe, n_folds

    def _run_mcpt(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        params: Dict,
    ) -> Tuple[float, float]:
        """Run MCPT in-sample and walk-forward tests."""
        try:
            from src.intelligence.validation.mcpt_validator import MCPTValidator
        except ImportError:
            from ...validation.mcpt_validator import MCPTValidator

        validator = MCPTValidator(
            n_permutations=self.config.mcpt_is_perms,
            n_walkforward_permutations=self.config.mcpt_wf_perms,
        )

        close = data['close'].values

        # Build optimizer function
        def optimizer_fn(perm_data):
            positions = strategy_fn(perm_data, **params)
            bar_returns = compute_bar_returns(
                positions, perm_data['close'].values, BacktestAssumptions()
            )
            return objective_profit_factor(bar_returns)

        # In-sample MCPT
        is_result = validator.insample_mcpt(optimizer_fn, data)
        is_pvalue = is_result.p_value

        # Walk-forward MCPT (only if in-sample passes)
        if is_pvalue < self.config.mcpt_is_threshold:
            wf_result = validator.walkforward_mcpt(
                optimizer_fn, data,
                train_window=self.config.train_window_bars
            )
            wf_pvalue = wf_result.p_value
        else:
            wf_pvalue = 1.0  # Skip if in-sample failed

        return is_pvalue, wf_pvalue


def run_full_validation(
    strategy_id: str,
    template_name: str,
    strategy_fn: Callable,
    data: pd.DataFrame,
    param_grid: Optional[Dict] = None,
    config: Optional[ValidationConfig] = None,
    mlflow_log: bool = True,
) -> ValidationResult:
    """
    Convenience function to run full validation with optional MLflow logging.

    Args:
        strategy_id: Strategy identifier
        template_name: Template name
        strategy_fn: Strategy function
        data: Market data
        param_grid: Parameter grid
        config: Validation config
        mlflow_log: Whether to log to MLflow

    Returns:
        ValidationResult
    """
    battery = ValidationBattery(config)
    result = battery.run(strategy_id, template_name, strategy_fn, data, param_grid)

    if mlflow_log:
        try:
            import mlflow
            mlflow.set_experiment("trader-ai-strategy-lab")

            with mlflow.start_run(run_name=f"validate_{strategy_id}"):
                mlflow.set_tags({
                    'run_kind': 'strategy_validation',
                    'strategy_id': strategy_id,
                    'template_name': template_name,
                })

                mlflow.log_metrics({
                    'is_pf': result.is_profit_factor,
                    'is_sharpe': result.is_sharpe,
                    'is_maxdd': result.is_max_drawdown,
                    'wf_pf': result.wf_profit_factor,
                    'wf_sharpe': result.wf_sharpe,
                    'mcpt_is_p': result.mcpt_is_pvalue,
                    'mcpt_wf_p': result.mcpt_wf_pvalue,
                    'cliff_score': result.cliff_score,
                    'mc_cagr_p5': result.mc_cagr_p5,
                    'mc_maxdd_p95': result.mc_maxdd_p95,
                    'mc_ruin_prob': result.mc_ruin_prob,
                    'passed': 1.0 if result.passed else 0.0,
                })

                mlflow.log_dict(result.to_dict(), 'validation_result.json')

        except ImportError:
            logger.warning("MLflow not available, skipping logging")

    return result


if __name__ == "__main__":
    # Test validation battery
    print("=== Validation Battery Test ===")

    np.random.seed(42)

    # Generate synthetic data
    n = 252 * 5  # 5 years
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    data = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.uniform(1e6, 5e6, n),
    })

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

    # Run validation with reduced settings for testing
    config = ValidationConfig(
        train_window_bars=252,
        step_bars=63,
        mcpt_is_perms=50,
        mcpt_wf_perms=20,
        mc_paths=100,
    )

    param_grid = {
        'lookback': [15, 20, 25],
        'threshold': [0.015, 0.02, 0.025],
    }

    result = run_full_validation(
        strategy_id="test_momentum",
        template_name="momentum",
        strategy_fn=momentum_strategy,
        data=data,
        param_grid=param_grid,
        config=config,
        mlflow_log=False,
    )

    print(f"\nValidation Result:")
    print(f"  Passed: {result.passed}")
    print(f"  IS PF: {result.is_profit_factor:.3f}")
    print(f"  IS Sharpe: {result.is_sharpe:.3f}")
    print(f"  WF PF: {result.wf_profit_factor:.3f}")
    print(f"  MCPT IS p: {result.mcpt_is_pvalue:.3f}")
    print(f"  MCPT WF p: {result.mcpt_wf_pvalue:.3f}")
    print(f"  Cliff Score: {result.cliff_score:.3f}")
    print(f"  MC Ruin Prob: {result.mc_ruin_prob:.1%}")

    if not result.passed:
        print(f"  Failure reasons: {result.failure_reasons}")

    print("\n=== Test Complete ===")
