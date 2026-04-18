"""
Pymoo adapter for local NSGA-II optimization.

Wraps pymoo library to work with TradingOracle interface.
This is the LOCAL, FAST optimizer used as primary or fallback.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 2
PATTERN: meta_calculus/moo_integration.py PymooAdapter
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import time

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    Problem = object  # Dummy for type hints

from src.optimization.trading_oracle import TradingOracle, OracleResult

logger = logging.getLogger(__name__)


@dataclass
class PymooConfig:
    """Configuration for Pymoo optimization."""
    algorithm: str = "NSGA2"
    population_size: int = 40
    n_generations: int = 50
    seed: int = 42
    verbose: bool = False
    timeout_seconds: float = 60.0


@dataclass
class OptimizationResult:
    """Result from MOO optimization."""
    pareto_front: np.ndarray  # Objective values (n_solutions x n_objectives)
    pareto_set: np.ndarray    # Decision variables (n_solutions x n_variables)
    n_solutions: int
    n_evaluations: int
    runtime_seconds: float
    converged: bool
    metadata: Dict[str, Any]


class TradingProblem(Problem):
    """
    Pymoo Problem wrapper for TradingOracle.

    Translates between pymoo's Problem interface and our oracle interface.
    """

    def __init__(self, oracle: TradingOracle):
        """
        Initialize trading problem from oracle.

        Args:
            oracle: TradingOracle instance
        """
        self.oracle = oracle

        lb, ub = oracle.get_bounds()
        n_var = len(lb)
        n_obj = oracle.n_obj
        n_constr = oracle.n_constr

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_constr,  # Treat all as inequality for pymoo
            xl=lb,
            xu=ub
        )

        self.evaluation_count = 0

    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate population of solutions.

        Args:
            x: Population matrix (n_pop x n_var)
            out: Output dictionary for pymoo
        """
        n_pop = x.shape[0]
        n_obj = self.oracle.n_obj
        n_constr = self.oracle.n_constr

        F = np.zeros((n_pop, n_obj))
        G = np.zeros((n_pop, n_constr)) if n_constr > 0 else None

        objective_names = [obj.name for obj in self.oracle.get_objectives()]
        constraint_names = [c.name for c in self.oracle.get_constraints()]

        for i in range(n_pop):
            result = self.oracle.evaluate(x[i])
            self.evaluation_count += 1

            # Extract objectives in order
            for j, name in enumerate(objective_names):
                F[i, j] = result.objectives.get(name, 0.0)

            # Extract constraints (convert to pymoo format: g(x) <= 0)
            if G is not None:
                for j, name in enumerate(constraint_names):
                    # For equality constraints, use |g(x)| - epsilon
                    G[i, j] = result.constraints.get(name, 0.0) - 0.01

        out["F"] = F
        if G is not None:
            out["G"] = G


class PymooAdapter:
    """
    Adapter for pymoo multi-objective optimization.

    Provides a clean interface for running NSGA-II on TradingOracle problems.
    """

    def __init__(self, config: Optional[PymooConfig] = None):
        """
        Initialize pymoo adapter.

        Args:
            config: Optimization configuration
        """
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo is not installed. Install with: pip install pymoo"
            )

        self.config = config or PymooConfig()

    def optimize(
        self,
        oracle: TradingOracle,
        config_override: Optional[PymooConfig] = None,
        initial_population: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        Args:
            oracle: TradingOracle to optimize
            config_override: Optional config to override defaults
            initial_population: Optional initial population (n_pop x n_var)

        Returns:
            OptimizationResult with Pareto front and set
        """
        config = config_override or self.config
        start_time = time.time()

        # Create pymoo problem
        problem = TradingProblem(oracle)

        # Create algorithm
        if config.algorithm == "NSGA2":
            algorithm = NSGA2(pop_size=config.population_size)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Set termination criteria
        termination = get_termination("n_gen", config.n_generations)

        # Run optimization
        try:
            result = minimize(
                problem,
                algorithm,
                termination,
                seed=config.seed,
                verbose=config.verbose,
            )

            runtime = time.time() - start_time

            # Check for timeout
            if runtime > config.timeout_seconds:
                logger.warning(
                    f"Optimization exceeded timeout: {runtime:.1f}s > {config.timeout_seconds}s"
                )

            # Extract results
            if result.X is not None and result.F is not None:
                pareto_set = result.X
                pareto_front = result.F

                # Handle single solution case
                if pareto_set.ndim == 1:
                    pareto_set = pareto_set.reshape(1, -1)
                    pareto_front = pareto_front.reshape(1, -1)

                n_solutions = pareto_set.shape[0]
                converged = True
            else:
                pareto_set = np.array([])
                pareto_front = np.array([])
                n_solutions = 0
                converged = False

            return OptimizationResult(
                pareto_front=pareto_front,
                pareto_set=pareto_set,
                n_solutions=n_solutions,
                n_evaluations=problem.evaluation_count,
                runtime_seconds=runtime,
                converged=converged,
                metadata={
                    'algorithm': config.algorithm,
                    'population_size': config.population_size,
                    'n_generations': config.n_generations,
                    'n_objectives': oracle.n_obj,
                    'n_variables': oracle.n_var,
                    'objective_names': [o.name for o in oracle.get_objectives()],
                }
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            runtime = time.time() - start_time

            return OptimizationResult(
                pareto_front=np.array([]),
                pareto_set=np.array([]),
                n_solutions=0,
                n_evaluations=problem.evaluation_count,
                runtime_seconds=runtime,
                converged=False,
                metadata={'error': str(e)}
            )

    def optimize_bounded(
        self,
        oracle: TradingOracle,
        bounds: Tuple[np.ndarray, np.ndarray],
        config_override: Optional[PymooConfig] = None,
    ) -> OptimizationResult:
        """
        Run optimization within specified bounds.

        Used by WovenMOOPipeline to search within GlobalMOO-defined edges.

        Args:
            oracle: TradingOracle to optimize
            bounds: (lower_bounds, upper_bounds) to constrain search
            config_override: Optional config override

        Returns:
            OptimizationResult
        """
        # Create bounded version of oracle
        class BoundedOracle(TradingOracle):
            def __init__(self, inner_oracle, new_bounds):
                self._inner = inner_oracle
                self._bounds = new_bounds

            def evaluate(self, x):
                return self._inner.evaluate(x)

            def get_bounds(self):
                return self._bounds

            def get_objectives(self):
                return self._inner.get_objectives()

            def get_constraints(self):
                return self._inner.get_constraints()

        bounded_oracle = BoundedOracle(oracle, bounds)
        return self.optimize(bounded_oracle, config_override)

    def get_best_solution(
        self,
        result: OptimizationResult,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Select best solution from Pareto front using weighted sum.

        Args:
            result: OptimizationResult from optimize()
            weights: Objective weights (default: equal weights)

        Returns:
            Tuple of (decision_variables, objective_values)
        """
        if result.n_solutions == 0:
            raise ValueError("No solutions in result")

        n_obj = result.pareto_front.shape[1]

        # Default to equal weights
        if weights is None:
            weight_values = np.ones(n_obj) / n_obj
        else:
            obj_names = result.metadata.get('objective_names', [])
            weight_values = np.array([
                weights.get(name, 1.0 / n_obj) for name in obj_names
            ])
            weight_values /= weight_values.sum()

        # Normalize objectives to [0, 1] range
        F = result.pareto_front
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range < 1e-10] = 1.0  # Avoid division by zero

        F_normalized = (F - F_min) / F_range

        # Weighted sum
        scores = np.dot(F_normalized, weight_values)

        # Best solution has lowest score (all objectives minimized)
        best_idx = np.argmin(scores)

        best_x = result.pareto_set[best_idx]
        best_f = {
            name: float(result.pareto_front[best_idx, i])
            for i, name in enumerate(result.metadata.get('objective_names', []))
        }

        return best_x, best_f

    @staticmethod
    def is_available() -> bool:
        """Check if pymoo is available."""
        return PYMOO_AVAILABLE


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_nsga2(
    oracle: TradingOracle,
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 42,
) -> OptimizationResult:
    """
    Convenience function to run NSGA-II.

    Args:
        oracle: TradingOracle to optimize
        pop_size: Population size
        n_gen: Number of generations
        seed: Random seed

    Returns:
        OptimizationResult
    """
    config = PymooConfig(
        algorithm="NSGA2",
        population_size=pop_size,
        n_generations=n_gen,
        seed=seed,
    )
    adapter = PymooAdapter(config)
    return adapter.optimize(oracle)
