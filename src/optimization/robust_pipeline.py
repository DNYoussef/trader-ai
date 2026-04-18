"""
Robust MOO Pipeline with fallback and caching.

Implements the WOVEN strategy:
1. GlobalMOO (cloud, best-in-class) finds OUTER BOUNDS of Pareto frontier
2. Pymoo (local, fast) searches WITHIN those bounds
3. Fallback to Pymoo-only if GlobalMOO unavailable

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 2
PATTERN: meta_calculus/moo_integration.py compare_optimizers()
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import time
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.optimization.trading_oracle import TradingOracle, OracleResult
from src.optimization.pymoo_adapter import (
    PymooAdapter, PymooConfig, OptimizationResult, PYMOO_AVAILABLE
)
from src.optimization.globalmoo_client import (
    GlobalMOOClient, GlobalMOOConfig, MockGlobalMOOClient, get_globalmoo_client
)
from src.utils.nnc_feature_flags import get_flag

logger = logging.getLogger(__name__)


@dataclass
class WovenConfig:
    """Configuration for woven GlobalMOO + Pymoo strategy."""
    # GlobalMOO settings
    globalmoo_enabled: bool = True
    globalmoo_iterations: int = 30

    # Pymoo settings
    pymoo_enabled: bool = True
    pymoo_pop_size: int = 40
    pymoo_generations: int = 50

    # Woven strategy settings
    bounds_margin: float = 0.10  # 10% margin on GlobalMOO bounds
    use_consensus: bool = True  # Find solutions in both Pareto fronts

    # Fallback settings
    fallback_to_pymoo: bool = True  # Use Pymoo if GlobalMOO fails

    # Caching settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_dir: str = "./data/moo_cache"


@dataclass
class CacheEntry:
    """Cache entry for MOO results."""
    result: OptimizationResult
    timestamp: datetime
    problem_hash: str


class ResultCache:
    """Simple file-based cache for MOO results."""

    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
        self._memory_cache: Dict[str, CacheEntry] = {}

    def _compute_hash(self, oracle: TradingOracle) -> str:
        """Compute hash of problem specification."""
        lb, ub = oracle.get_bounds()
        spec = {
            'bounds': [lb.tolist(), ub.tolist()],
            'n_obj': oracle.n_obj,
            'n_constr': oracle.n_constr,
            'objectives': [o.name for o in oracle.get_objectives()],
        }
        spec_str = json.dumps(spec, sort_keys=True)
        return hashlib.md5(spec_str.encode()).hexdigest()[:12]

    def get(self, oracle: TradingOracle) -> Optional[OptimizationResult]:
        """
        Get cached result if available and not expired.

        Args:
            oracle: TradingOracle for cache key

        Returns:
            OptimizationResult or None if not cached/expired
        """
        problem_hash = self._compute_hash(oracle)

        # Check memory cache first
        if problem_hash in self._memory_cache:
            entry = self._memory_cache[problem_hash]
            if datetime.now() - entry.timestamp < self.ttl:
                logger.debug(f"Cache hit (memory): {problem_hash}")
                return entry.result
            else:
                del self._memory_cache[problem_hash]

        # Check file cache
        cache_file = self.cache_dir / f"{problem_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                timestamp = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - timestamp < self.ttl:
                    result = OptimizationResult(
                        pareto_front=np.array(data['pareto_front']),
                        pareto_set=np.array(data['pareto_set']),
                        n_solutions=data['n_solutions'],
                        n_evaluations=data['n_evaluations'],
                        runtime_seconds=data['runtime_seconds'],
                        converged=data['converged'],
                        metadata=data['metadata'],
                    )
                    logger.debug(f"Cache hit (file): {problem_hash}")
                    return result
                else:
                    cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        return None

    def set(self, oracle: TradingOracle, result: OptimizationResult) -> None:
        """
        Cache optimization result.

        Args:
            oracle: TradingOracle for cache key
            result: Result to cache
        """
        problem_hash = self._compute_hash(oracle)
        timestamp = datetime.now()

        # Memory cache
        self._memory_cache[problem_hash] = CacheEntry(
            result=result,
            timestamp=timestamp,
            problem_hash=problem_hash,
        )

        # File cache
        try:
            cache_file = self.cache_dir / f"{problem_hash}.json"
            data = {
                'pareto_front': result.pareto_front.tolist() if len(result.pareto_front) > 0 else [],
                'pareto_set': result.pareto_set.tolist() if len(result.pareto_set) > 0 else [],
                'n_solutions': result.n_solutions,
                'n_evaluations': result.n_evaluations,
                'runtime_seconds': result.runtime_seconds,
                'converged': result.converged,
                'metadata': result.metadata,
                'timestamp': timestamp.isoformat(),
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)

            logger.debug(f"Cached result: {problem_hash}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


class WovenMOOPipeline:
    """
    Woven GlobalMOO + Pymoo optimization pipeline.

    Strategy:
    1. Phase 1: GlobalMOO finds outer edges of Pareto frontier
    2. Phase 2: Pymoo searches within GlobalMOO-defined bounds
    3. Combined result: Union of both fronts, optionally filtered for consensus

    If GlobalMOO is unavailable, falls back to Pymoo-only with wider search.
    """

    def __init__(self, config: Optional[WovenConfig] = None):
        """
        Initialize woven pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or WovenConfig()

        # Initialize components
        if self.config.pymoo_enabled:
            pymoo_config = PymooConfig(
                population_size=self.config.pymoo_pop_size,
                n_generations=self.config.pymoo_generations,
            )
            self.pymoo = PymooAdapter(pymoo_config)
        else:
            self.pymoo = None

        if self.config.globalmoo_enabled:
            self.globalmoo = get_globalmoo_client(use_mock=not get_flag('use_globalmoo', False))
        else:
            self.globalmoo = None

        # Initialize cache
        if self.config.cache_enabled:
            self.cache = ResultCache(
                self.config.cache_dir,
                self.config.cache_ttl_seconds
            )
        else:
            self.cache = None

    def optimize(
        self,
        oracle: TradingOracle,
        use_cache: bool = True,
    ) -> OptimizationResult:
        """
        Run woven optimization.

        Args:
            oracle: TradingOracle to optimize
            use_cache: Whether to use cached results

        Returns:
            OptimizationResult with combined Pareto front
        """
        start_time = time.time()

        # Check cache
        if use_cache and self.cache is not None:
            cached = self.cache.get(oracle)
            if cached is not None:
                logger.info("Using cached optimization result")
                return cached

        # Phase 1: GlobalMOO (if available)
        globalmoo_result = None
        globalmoo_bounds = None

        if self.globalmoo is not None and self.globalmoo.is_available:
            try:
                logger.info("Phase 1: Running GlobalMOO for outer edges...")
                globalmoo_result = self.globalmoo.optimize(
                    oracle,
                    n_iterations=self.config.globalmoo_iterations,
                )

                if globalmoo_result.n_solutions > 0:
                    # Extract bounds for Phase 2
                    globalmoo_bounds = self.globalmoo.extract_bounds_from_pareto(
                        globalmoo_result.pareto_set,
                        margin=self.config.bounds_margin,
                    )
                    logger.info(f"GlobalMOO found {globalmoo_result.n_solutions} Pareto solutions")

            except Exception as e:
                logger.warning(f"GlobalMOO failed: {e}")
                if not self.config.fallback_to_pymoo:
                    raise

        # Phase 2: Pymoo (local refinement)
        pymoo_result = None

        if self.pymoo is not None:
            logger.info("Phase 2: Running Pymoo for local refinement...")

            if globalmoo_bounds is not None:
                # Search within GlobalMOO-defined bounds
                pymoo_result = self.pymoo.optimize_bounded(
                    oracle,
                    bounds=globalmoo_bounds,
                )
            else:
                # Full search (no GlobalMOO bounds)
                pymoo_result = self.pymoo.optimize(oracle)

            logger.info(f"Pymoo found {pymoo_result.n_solutions} Pareto solutions")

        # Combine results
        result = self._combine_results(globalmoo_result, pymoo_result, oracle)

        # Update runtime
        result.runtime_seconds = time.time() - start_time

        # Cache result
        if self.cache is not None:
            self.cache.set(oracle, result)

        return result

    def _combine_results(
        self,
        globalmoo_result: Optional[OptimizationResult],
        pymoo_result: Optional[OptimizationResult],
        oracle: TradingOracle,
    ) -> OptimizationResult:
        """
        Combine GlobalMOO and Pymoo results.

        Args:
            globalmoo_result: Result from GlobalMOO (may be None)
            pymoo_result: Result from Pymoo (may be None)
            oracle: Original oracle for metadata

        Returns:
            Combined OptimizationResult
        """
        pareto_sets = []
        pareto_fronts = []
        n_evaluations = 0
        converged = False
        sources = []

        if globalmoo_result is not None and globalmoo_result.n_solutions > 0:
            pareto_sets.append(globalmoo_result.pareto_set)
            pareto_fronts.append(globalmoo_result.pareto_front)
            n_evaluations += globalmoo_result.n_evaluations
            converged = converged or globalmoo_result.converged
            sources.append('GlobalMOO')

        if pymoo_result is not None and pymoo_result.n_solutions > 0:
            pareto_sets.append(pymoo_result.pareto_set)
            pareto_fronts.append(pymoo_result.pareto_front)
            n_evaluations += pymoo_result.n_evaluations
            converged = converged or pymoo_result.converged
            sources.append('Pymoo')

        if not pareto_sets:
            return OptimizationResult(
                pareto_front=np.array([]),
                pareto_set=np.array([]),
                n_solutions=0,
                n_evaluations=n_evaluations,
                runtime_seconds=0.0,
                converged=False,
                metadata={'error': 'No solutions found', 'sources': sources}
            )

        # Concatenate all solutions
        combined_set = np.vstack(pareto_sets)
        combined_front = np.vstack(pareto_fronts)

        # Remove duplicates and re-filter for Pareto optimality
        combined_set, combined_front = self._filter_pareto(combined_set, combined_front)

        return OptimizationResult(
            pareto_front=combined_front,
            pareto_set=combined_set,
            n_solutions=len(combined_set),
            n_evaluations=n_evaluations,
            runtime_seconds=0.0,  # Will be updated by caller
            converged=converged,
            metadata={
                'sources': sources,
                'strategy': 'woven' if len(sources) > 1 else sources[0] if sources else 'none',
                'objective_names': [o.name for o in oracle.get_objectives()],
            }
        )

    def _filter_pareto(
        self,
        X: np.ndarray,
        F: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter combined solutions for Pareto optimality.

        Args:
            X: Decision variables (n x n_var)
            F: Objective values (n x n_obj)

        Returns:
            Filtered (X, F) containing only non-dominated solutions
        """
        n = len(X)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue

            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue

                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_pareto[i] = False
                    break

        return X[is_pareto], F[is_pareto]

    def get_best_solution(
        self,
        result: OptimizationResult,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Get best solution from Pareto front.

        Delegates to Pymoo adapter for weighted selection.

        Args:
            result: OptimizationResult
            weights: Objective weights

        Returns:
            Tuple of (decision_variables, objective_values)
        """
        if self.pymoo is not None:
            return self.pymoo.get_best_solution(result, weights)

        # Fallback: return first solution
        if result.n_solutions == 0:
            raise ValueError("No solutions available")

        obj_names = result.metadata.get('objective_names', [])
        best_f = {
            name: float(result.pareto_front[0, i])
            for i, name in enumerate(obj_names)
        }
        return result.pareto_set[0], best_f


class RobustMOOPipeline(WovenMOOPipeline):
    """
    Alias for WovenMOOPipeline for backward compatibility.

    The "robust" pipeline IS the woven pipeline with:
    - GlobalMOO for outer edges
    - Pymoo for local refinement
    - Caching for performance
    - Fallback to Pymoo-only
    """
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_woven_optimization(
    oracle: TradingOracle,
    globalmoo_iterations: int = 30,
    pymoo_generations: int = 50,
    use_cache: bool = True,
) -> OptimizationResult:
    """
    Convenience function for woven optimization.

    Args:
        oracle: TradingOracle to optimize
        globalmoo_iterations: GlobalMOO iterations
        pymoo_generations: Pymoo generations
        use_cache: Whether to use caching

    Returns:
        OptimizationResult
    """
    config = WovenConfig(
        globalmoo_iterations=globalmoo_iterations,
        pymoo_generations=pymoo_generations,
        cache_enabled=use_cache,
    )
    pipeline = WovenMOOPipeline(config)
    return pipeline.optimize(oracle, use_cache=use_cache)


def run_pymoo_only(
    oracle: TradingOracle,
    pop_size: int = 40,
    n_gen: int = 50,
) -> OptimizationResult:
    """
    Convenience function for Pymoo-only optimization.

    Args:
        oracle: TradingOracle to optimize
        pop_size: Population size
        n_gen: Number of generations

    Returns:
        OptimizationResult
    """
    config = WovenConfig(
        globalmoo_enabled=False,
        pymoo_pop_size=pop_size,
        pymoo_generations=n_gen,
        cache_enabled=False,
    )
    pipeline = WovenMOOPipeline(config)
    return pipeline.optimize(oracle, use_cache=False)
