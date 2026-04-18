"""
MCPT Validator - Monte Carlo Permutation Test Wrapper

Provides a clean interface for validating trading strategies using MCPT.

Key concepts:
- In-sample MCPT: Tests if optimized strategy beats random permutations
- Walk-forward MCPT: Tests if out-of-sample performance is significant
- p-value: Probability that result is from noise (lower = better)

Usage:
    validator = MCPTValidator(n_permutations=1000)

    # In-sample test
    result = validator.insample_mcpt(optimizer_fn, data, objective='profit_factor')

    # Walk-forward test
    result = validator.walkforward_mcpt(walkforward_fn, data, train_window=252)
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import sys
import os

# Add third_party to path for vendored MCPT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party'))

from mcpt import get_permutation
from .objectives import profit_factor, sharpe_ratio, get_objective_function

logger = logging.getLogger(__name__)


# Module-level worker functions for pickling compatibility
def _run_single_permutation(args: Tuple[int, pd.DataFrame, Any, int]) -> Optional[float]:
    """
    Worker function for parallel MCPT insample permutations.

    Must be at module level for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (seed, data, optimizer_fn, permutation_index)

    Returns:
        Permutation score or None if failed
    """
    seed, data, optimizer_fn, perm_idx = args
    np.random.seed(seed)

    try:
        # Generate permuted data
        perm_data = get_permutation(data, start_index=0)

        # Get score on permuted data
        perm_score = optimizer_fn(perm_data)
        return perm_score
    except Exception as e:
        logger.debug(f"Permutation {perm_idx} failed: {e}")
        return None


def _run_single_walkforward_permutation(args: Tuple[int, pd.DataFrame, Any, int, int]) -> Optional[float]:
    """
    Worker function for parallel MCPT walk-forward permutations.

    Must be at module level for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (seed, data, walkforward_fn, train_window, permutation_index)

    Returns:
        Permutation score or None if failed
    """
    seed, data, walkforward_fn, train_window, perm_idx = args
    np.random.seed(seed)

    try:
        # Generate permuted data (preserve first train_window bars)
        perm_data = get_permutation(data, start_index=train_window)

        # Get score on permuted data
        perm_score = walkforward_fn(perm_data)
        return perm_score
    except Exception as e:
        logger.debug(f"Permutation {perm_idx} failed: {e}")
        return None


@dataclass
class MCPTResult:
    """Result of an MCPT validation test."""
    real_score: float
    p_value: float
    perm_scores: List[float]
    n_permutations: int
    objective: str
    test_type: str  # 'insample' or 'walkforward'
    passed: bool

    def __post_init__(self):
        # Determine if test passed based on type
        if self.test_type == 'insample':
            self.passed = self.p_value < 0.01  # 1% threshold
        else:  # walkforward
            self.passed = self.p_value < 0.05  # 5% threshold

    @property
    def percentile(self) -> float:
        """Percentile rank of real score among permutations."""
        if not self.perm_scores:
            return 100.0
        return 100 * (1 - self.p_value)

    def to_dict(self) -> Dict:
        return {
            'real_score': self.real_score,
            'p_value': self.p_value,
            'n_permutations': self.n_permutations,
            'objective': self.objective,
            'test_type': self.test_type,
            'passed': self.passed,
            'percentile': self.percentile,
            'perm_mean': np.mean(self.perm_scores) if self.perm_scores else 0.0,
            'perm_std': np.std(self.perm_scores) if self.perm_scores else 0.0,
        }


class MCPTValidator:
    """
    Monte Carlo Permutation Test validator for trading strategies.

    MCPT answers the question: "Is this strategy's performance statistically
    significant, or could it have been achieved by random chance?"

    The test works by:
    1. Running the strategy on real data to get a score
    2. Running the strategy on N permuted versions of the data
    3. Computing p-value = (count where perm >= real) / N

    A low p-value means the strategy is unlikely to have achieved its
    score by random chance.
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        n_walkforward_permutations: int = 200,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        """
        Initialize MCPT validator.

        Args:
            n_permutations: Number of permutations for in-sample test
            n_walkforward_permutations: Number of permutations for walk-forward
            parallel: Whether to use parallel processing (default: True)
            n_workers: Number of parallel workers (default: os.cpu_count())
        """
        self.n_permutations = n_permutations
        self.n_walkforward_permutations = n_walkforward_permutations
        self.parallel = parallel
        self.n_workers = n_workers or os.cpu_count()

    def insample_mcpt(
        self,
        optimizer_fn: Callable[[pd.DataFrame], float],
        data: pd.DataFrame,
        objective: str = 'profit_factor',
        seed: Optional[int] = None,
    ) -> MCPTResult:
        """
        Run in-sample Monte Carlo Permutation Test.

        The optimizer_fn should:
        1. Take OHLC DataFrame as input
        2. Optimize strategy parameters on that data
        3. Return the objective score

        Args:
            optimizer_fn: Function that optimizes strategy and returns score
            data: OHLC DataFrame with columns ['open', 'high', 'low', 'close']
            objective: Name of objective function used
            seed: Random seed for reproducibility

        Returns:
            MCPTResult with real_score, p_value, and permutation scores
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Running in-sample MCPT with {self.n_permutations} permutations")

        # 1. Get score on real data
        real_score = optimizer_fn(data)
        logger.info(f"Real data score: {real_score:.4f}")

        # 2. Get scores on permuted data
        perm_scores = []
        better_count = 0

        if self.parallel:
            # Parallel execution with ProcessPoolExecutor
            base_seed = seed if seed is not None else np.random.randint(0, 2**31)
            seeds = [base_seed + i for i in range(self.n_permutations)]
            args_list = [(seeds[i], data, optimizer_fn, i) for i in range(self.n_permutations)]

            logger.info(f"Running parallel MCPT with {self.n_workers} workers")

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                completed = 0
                for result in executor.map(_run_single_permutation, args_list):
                    if result is not None:
                        perm_scores.append(result)
                        if result >= real_score:
                            better_count += 1

                    completed += 1
                    if completed % 100 == 0:
                        logger.info(f"Completed {completed}/{self.n_permutations} permutations")
        else:
            # Sequential execution (backward compatibility)
            for i in range(self.n_permutations):
                # Generate permuted data
                perm_data = get_permutation(data, start_index=0)

                # Get score on permuted data
                try:
                    perm_score = optimizer_fn(perm_data)
                    perm_scores.append(perm_score)

                    if perm_score >= real_score:
                        better_count += 1

                except Exception as e:
                    logger.debug(f"Permutation {i} failed: {e}")
                    continue

                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{self.n_permutations} permutations")

        # 3. Compute p-value
        n_valid = len(perm_scores)
        if n_valid == 0:
            logger.warning("No valid permutations completed")
            p_value = 1.0
        else:
            p_value = better_count / n_valid

        logger.info(f"In-sample MCPT: p-value = {p_value:.4f} ({better_count}/{n_valid} better)")

        return MCPTResult(
            real_score=real_score,
            p_value=p_value,
            perm_scores=perm_scores,
            n_permutations=n_valid,
            objective=objective,
            test_type='insample',
            passed=p_value < 0.01,
        )

    def walkforward_mcpt(
        self,
        walkforward_fn: Callable[[pd.DataFrame], float],
        data: pd.DataFrame,
        train_window: int = 252,
        objective: str = 'profit_factor',
        seed: Optional[int] = None,
    ) -> MCPTResult:
        """
        Run walk-forward Monte Carlo Permutation Test.

        CRITICAL: This test permutes only the data AFTER the first training window.
        This tests whether out-of-sample performance is significant.

        The walkforward_fn should:
        1. Take full OHLC DataFrame as input
        2. Run walk-forward optimization with train/test splits
        3. Return the aggregate out-of-sample score

        Args:
            walkforward_fn: Function that runs walk-forward and returns OOS score
            data: Full OHLC DataFrame
            train_window: Number of bars in first training window
            objective: Name of objective function used
            seed: Random seed for reproducibility

        Returns:
            MCPTResult with real_score, p_value, and permutation scores
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Running walk-forward MCPT with {self.n_walkforward_permutations} permutations")
        logger.info(f"Train window: {train_window} bars (preserved)")

        # 1. Get score on real data
        real_score = walkforward_fn(data)
        logger.info(f"Real walk-forward score: {real_score:.4f}")

        # 2. Get scores on permuted data (permute ONLY after train_window)
        perm_scores = []
        better_count = 0

        if self.parallel:
            # Parallel execution with ProcessPoolExecutor
            base_seed = seed if seed is not None else np.random.randint(0, 2**31)
            seeds = [base_seed + i for i in range(self.n_walkforward_permutations)]
            args_list = [(seeds[i], data, walkforward_fn, train_window, i)
                        for i in range(self.n_walkforward_permutations)]

            logger.info(f"Running parallel walk-forward MCPT with {self.n_workers} workers")

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                completed = 0
                for result in executor.map(_run_single_walkforward_permutation, args_list):
                    if result is not None:
                        perm_scores.append(result)
                        if result >= real_score:
                            better_count += 1

                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"Completed {completed}/{self.n_walkforward_permutations} permutations")
        else:
            # Sequential execution (backward compatibility)
            for i in range(self.n_walkforward_permutations):
                # Generate permuted data (preserve first train_window bars)
                perm_data = get_permutation(data, start_index=train_window)

                # Get score on permuted data
                try:
                    perm_score = walkforward_fn(perm_data)
                    perm_scores.append(perm_score)

                    if perm_score >= real_score:
                        better_count += 1

                except Exception as e:
                    logger.debug(f"Permutation {i} failed: {e}")
                    continue

                if (i + 1) % 50 == 0:
                    logger.info(f"Completed {i + 1}/{self.n_walkforward_permutations} permutations")

        # 3. Compute p-value
        n_valid = len(perm_scores)
        if n_valid == 0:
            logger.warning("No valid permutations completed")
            p_value = 1.0
        else:
            p_value = better_count / n_valid

        logger.info(f"Walk-forward MCPT: p-value = {p_value:.4f} ({better_count}/{n_valid} better)")

        return MCPTResult(
            real_score=real_score,
            p_value=p_value,
            perm_scores=perm_scores,
            n_permutations=n_valid,
            objective=objective,
            test_type='walkforward',
            passed=p_value < 0.05,
        )

    def validate_strategy(
        self,
        signal_fn: Callable[[pd.DataFrame, int], int],
        data: pd.DataFrame,
        objective: str = 'profit_factor',
        train_window: int = 252,
        run_walkforward: bool = True,
    ) -> Dict[str, MCPTResult]:
        """
        Full validation of a strategy signal function.

        Args:
            signal_fn: Function(df, idx) -> {-1, 0, +1}
            data: OHLC DataFrame
            objective: Objective function name
            train_window: Walk-forward training window
            run_walkforward: Whether to run walk-forward test

        Returns:
            Dict with 'insample' and optionally 'walkforward' MCPTResult
        """
        from ..strategy_lab.signal_interface import (
            position_from_signal_function,
            strategy_returns,
        )

        objective_fn = get_objective_function(objective)

        def optimizer_fn(df: pd.DataFrame) -> float:
            """Optimizer that computes objective on signal."""
            positions = position_from_signal_function(df, signal_fn)
            returns = strategy_returns(df['close'].values, positions)
            return objective_fn(returns)

        results = {}

        # In-sample test
        results['insample'] = self.insample_mcpt(optimizer_fn, data, objective)

        # Walk-forward test (only if in-sample passed)
        if run_walkforward and results['insample'].passed:
            results['walkforward'] = self.walkforward_mcpt(
                optimizer_fn, data, train_window, objective
            )
        elif run_walkforward:
            logger.info("Skipping walk-forward test (in-sample failed)")

        return results

    def quick_validate(
        self,
        positions: np.ndarray,
        close_prices: np.ndarray,
        n_permutations: int = 100,
    ) -> float:
        """
        Quick validation with reduced permutations.

        Useful for screening many strategies quickly.

        Args:
            positions: Position array {-1, 0, +1}
            close_prices: Array of close prices
            n_permutations: Number of permutations (default: 100)

        Returns:
            Estimated p-value
        """
        from ..strategy_lab.signal_interface import strategy_returns

        # Real score
        real_returns = strategy_returns(close_prices, positions)
        real_score = profit_factor(real_returns)

        # Create synthetic OHLC for permutation
        df = pd.DataFrame({
            'open': close_prices,
            'high': close_prices * 1.001,
            'low': close_prices * 0.999,
            'close': close_prices,
        })

        better_count = 0
        for _ in range(n_permutations):
            perm_df = get_permutation(df)
            perm_returns = strategy_returns(perm_df['close'].values, positions)
            perm_score = profit_factor(perm_returns)

            if perm_score >= real_score:
                better_count += 1

        return better_count / n_permutations


@lru_cache(maxsize=128)
def passes_gate(insample_pvalue: float, walkforward_pvalue: float, gate_level: int) -> bool:
    """
    Check if p-values pass the specified gate level.

    Gate thresholds (from plan):
    - G0-G2: insample < 0.05, walkforward < 0.10 (learning)
    - G3-G5: insample < 0.02, walkforward < 0.05 (real money)
    - G6-G8: insample < 0.01, walkforward < 0.02 (significant capital)
    - G9-G12: insample < 0.005, walkforward < 0.01 (institutional)

    Args:
        insample_pvalue: p-value from in-sample MCPT
        walkforward_pvalue: p-value from walk-forward MCPT
        gate_level: Current gate level (0-12)

    Returns:
        True if both p-values pass the gate thresholds
    """
    if gate_level <= 2:
        return insample_pvalue < 0.05 and walkforward_pvalue < 0.10
    elif gate_level <= 5:
        return insample_pvalue < 0.02 and walkforward_pvalue < 0.05
    elif gate_level <= 8:
        return insample_pvalue < 0.01 and walkforward_pvalue < 0.02
    else:
        return insample_pvalue < 0.005 and walkforward_pvalue < 0.01


if __name__ == "__main__":
    # Test MCPT validator
    print("=== MCPT Validator Test ===")

    np.random.seed(42)

    # Create synthetic OHLC data
    n_bars = 500
    close = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    open_price = close + np.random.randn(n_bars) * 0.2

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })

    print(f"Data shape: {df.shape}")

    # Create a simple momentum signal function
    def momentum_signal(df: pd.DataFrame, idx: int) -> int:
        if idx < 20:
            return 0
        returns_20d = df['close'].iloc[idx] / df['close'].iloc[idx - 20] - 1
        if returns_20d > 0.02:
            return 1
        elif returns_20d < -0.02:
            return -1
        return 0

    # Quick validation
    validator = MCPTValidator(n_permutations=50)  # Small for testing

    print("\nRunning quick validation...")
    positions = np.array([momentum_signal(df, i) for i in range(len(df))])
    p_value = validator.quick_validate(positions, df['close'].values, n_permutations=50)
    print(f"Quick validation p-value: {p_value:.4f}")

    print("\n=== Test Complete ===")
