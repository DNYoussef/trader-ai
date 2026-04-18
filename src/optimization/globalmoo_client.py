"""
GlobalMOO cloud API client.

Client for the GlobalMOO cloud optimization service (https://app.globalmoo.com).
Used to find outer edges of Pareto frontier before local refinement.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 2
PATTERN: meta_calculus/moo_integration.py GlobalMOOClient

GlobalMOO is BEST-IN-CLASS for finding global Pareto edges.
Use GlobalMOO to define edges, then Pymoo to search within.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import time
import os
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.optimization.trading_oracle import TradingOracle, OracleResult
from src.optimization.pymoo_adapter import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class GlobalMOOConfig:
    """Configuration for GlobalMOO API."""
    api_url: str = "https://app.globalmoo.com/api/"
    api_key_env_var: str = "GLOBALMOO_API_KEY"
    project_name: str = "TraderAI_Strategy_Optimization"
    default_iterations: int = 30
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0


class GlobalMOOClient:
    """
    Client for GlobalMOO cloud optimization API.

    GlobalMOO provides best-in-class global Pareto frontier discovery.
    Use for finding outer edges, then refine with local Pymoo.
    """

    def __init__(self, config: Optional[GlobalMOOConfig] = None):
        """
        Initialize GlobalMOO client.

        Args:
            config: API configuration
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is not installed. Install with: pip install requests"
            )

        self.config = config or GlobalMOOConfig()
        self._api_key = os.environ.get(self.config.api_key_env_var)
        self._project_id = None

    @property
    def is_available(self) -> bool:
        """Check if GlobalMOO API is available (key configured)."""
        return self._api_key is not None and len(self._api_key) > 0

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        Make API request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            timeout: Request timeout

        Returns:
            Response JSON dict
        """
        if not self.is_available:
            raise ValueError("GlobalMOO API key not configured")

        url = f"{self.config.api_url.rstrip('/')}/{endpoint}"
        timeout = timeout or self.config.timeout_seconds

        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(
                    f"GlobalMOO API request failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        raise ConnectionError(f"GlobalMOO API failed after {self.config.retry_attempts} attempts: {last_error}")

    def create_project(
        self,
        oracle: TradingOracle,
        project_name: Optional[str] = None,
    ) -> str:
        """
        Create a GlobalMOO project for the optimization problem.

        Args:
            oracle: TradingOracle defining the problem
            project_name: Optional custom project name

        Returns:
            Project ID
        """
        lb, ub = oracle.get_bounds()
        objectives = oracle.get_objectives()
        constraints = oracle.get_constraints()

        # Build input variables specification
        input_vars = []
        for i in range(oracle.n_var):
            input_vars.append({
                "name": f"x_{i}",
                "type": "float",
                "min": float(lb[i]),
                "max": float(ub[i]),
            })

        # Build objectives specification
        obj_specs = []
        for obj in objectives:
            obj_specs.append({
                "name": obj.name,
                "direction": obj.direction,
                "bounds": list(obj.bounds) if obj.bounds != (-np.inf, np.inf) else None,
            })

        # Build constraints specification
        constr_specs = []
        for constr in constraints:
            constr_specs.append({
                "name": constr.name,
                "type": constr.type,
                "expression": constr.expression,
            })

        project_data = {
            "name": project_name or self.config.project_name,
            "input_variables": input_vars,
            "objectives": obj_specs,
            "constraints": constr_specs,
        }

        response = self._make_request("POST", "projects", data=project_data)
        self._project_id = response.get("project_id")

        logger.info(f"Created GlobalMOO project: {self._project_id}")
        return self._project_id

    def submit_evaluation(
        self,
        project_id: str,
        x: np.ndarray,
        result: OracleResult,
    ) -> Dict:
        """
        Submit an evaluation result to GlobalMOO.

        Args:
            project_id: Project ID
            x: Decision variables evaluated
            result: OracleResult from oracle.evaluate()

        Returns:
            API response
        """
        eval_data = {
            "project_id": project_id,
            "input_values": x.tolist(),
            "objective_values": result.objectives,
            "constraint_values": result.constraints,
            "feasible": result.feasible,
        }

        return self._make_request("POST", f"projects/{project_id}/evaluations", data=eval_data)

    def get_suggestions(
        self,
        project_id: str,
        n_suggestions: int = 5,
    ) -> List[np.ndarray]:
        """
        Get suggested points to evaluate from GlobalMOO.

        Args:
            project_id: Project ID
            n_suggestions: Number of suggestions to request

        Returns:
            List of decision variable arrays to evaluate
        """
        response = self._make_request(
            "GET",
            f"projects/{project_id}/suggestions?n={n_suggestions}"
        )

        suggestions = response.get("suggestions", [])
        return [np.array(s["input_values"]) for s in suggestions]

    def get_pareto_front(
        self,
        project_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current Pareto front from GlobalMOO.

        Args:
            project_id: Project ID

        Returns:
            Tuple of (pareto_set, pareto_front) arrays
        """
        response = self._make_request("GET", f"projects/{project_id}/pareto")

        pareto_data = response.get("pareto_solutions", [])

        if not pareto_data:
            return np.array([]), np.array([])

        pareto_set = np.array([s["input_values"] for s in pareto_data])
        pareto_front = np.array([list(s["objective_values"].values()) for s in pareto_data])

        return pareto_set, pareto_front

    def optimize(
        self,
        oracle: TradingOracle,
        n_iterations: Optional[int] = None,
        project_name: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Run full GlobalMOO optimization loop.

        Args:
            oracle: TradingOracle to optimize
            n_iterations: Number of optimization iterations
            project_name: Optional project name

        Returns:
            OptimizationResult
        """
        if not self.is_available:
            logger.warning("GlobalMOO API not available, returning empty result")
            return OptimizationResult(
                pareto_front=np.array([]),
                pareto_set=np.array([]),
                n_solutions=0,
                n_evaluations=0,
                runtime_seconds=0.0,
                converged=False,
                metadata={'error': 'API key not configured'}
            )

        n_iterations = n_iterations or self.config.default_iterations
        start_time = time.time()
        n_evaluations = 0

        try:
            # Create project
            project_id = self.create_project(oracle, project_name)

            # Optimization loop
            for iteration in range(n_iterations):
                # Get suggestions
                suggestions = self.get_suggestions(project_id, n_suggestions=5)

                # Evaluate suggestions
                for x in suggestions:
                    result = oracle.evaluate(x)
                    self.submit_evaluation(project_id, x, result)
                    n_evaluations += 1

                # Log progress
                if iteration % 10 == 0:
                    logger.info(f"GlobalMOO iteration {iteration}/{n_iterations}")

            # Get final Pareto front
            pareto_set, pareto_front = self.get_pareto_front(project_id)
            runtime = time.time() - start_time

            return OptimizationResult(
                pareto_front=pareto_front,
                pareto_set=pareto_set,
                n_solutions=len(pareto_set),
                n_evaluations=n_evaluations,
                runtime_seconds=runtime,
                converged=True,
                metadata={
                    'project_id': project_id,
                    'n_iterations': n_iterations,
                    'source': 'GlobalMOO',
                }
            )

        except Exception as e:
            logger.error(f"GlobalMOO optimization failed: {e}")
            runtime = time.time() - start_time

            return OptimizationResult(
                pareto_front=np.array([]),
                pareto_set=np.array([]),
                n_solutions=0,
                n_evaluations=n_evaluations,
                runtime_seconds=runtime,
                converged=False,
                metadata={'error': str(e)}
            )

    def extract_bounds_from_pareto(
        self,
        pareto_set: np.ndarray,
        margin: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract bounds from Pareto set for local refinement.

        Used by WovenMOOPipeline to define search space for Pymoo.

        Args:
            pareto_set: Decision variables of Pareto solutions
            margin: Margin to add to bounds (0.10 = 10%)

        Returns:
            (lower_bounds, upper_bounds) with margin
        """
        if len(pareto_set) == 0:
            raise ValueError("Empty Pareto set")

        # Get min/max for each variable
        lb = pareto_set.min(axis=0)
        ub = pareto_set.max(axis=0)

        # Add margin
        range_val = ub - lb
        range_val[range_val < 1e-6] = 1e-6  # Avoid zero range

        lb_with_margin = lb - margin * range_val
        ub_with_margin = ub + margin * range_val

        return lb_with_margin, ub_with_margin


class MockGlobalMOOClient(GlobalMOOClient):
    """
    Mock GlobalMOO client for testing and offline use.

    Uses Latin Hypercube Sampling to simulate GlobalMOO suggestions.
    """

    def __init__(self, config: Optional[GlobalMOOConfig] = None):
        """Initialize mock client."""
        self.config = config or GlobalMOOConfig()
        self._project_id = "mock_project"
        self._evaluations = []

    @property
    def is_available(self) -> bool:
        """Mock is always available."""
        return True

    def create_project(
        self,
        oracle: TradingOracle,
        project_name: Optional[str] = None,
    ) -> str:
        """Create mock project."""
        self._oracle = oracle
        self._evaluations = []
        return self._project_id

    def submit_evaluation(
        self,
        project_id: str,
        x: np.ndarray,
        result: OracleResult,
    ) -> Dict:
        """Store mock evaluation."""
        self._evaluations.append({
            'x': x,
            'objectives': result.objectives,
            'feasible': result.feasible,
        })
        return {'status': 'ok'}

    def get_suggestions(
        self,
        project_id: str,
        n_suggestions: int = 5,
    ) -> List[np.ndarray]:
        """Generate Latin Hypercube samples as suggestions."""
        lb, ub = self._oracle.get_bounds()
        n_var = len(lb)

        # Latin Hypercube Sampling
        samples = np.random.rand(n_suggestions, n_var)
        samples = lb + samples * (ub - lb)

        return [s for s in samples]

    def get_pareto_front(
        self,
        project_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Pareto front from stored evaluations."""
        if not self._evaluations:
            return np.array([]), np.array([])

        # Filter feasible solutions
        feasible = [e for e in self._evaluations if e['feasible']]
        if not feasible:
            feasible = self._evaluations

        # Simple non-dominated sort
        pareto = []
        for e in feasible:
            dominated = False
            for other in feasible:
                if e == other:
                    continue
                # Check if other dominates e (all objectives <=, at least one <)
                obj_e = list(e['objectives'].values())
                obj_other = list(other['objectives'].values())
                if all(o <= e for o, e in zip(obj_other, obj_e)) and \
                   any(o < e for o, e in zip(obj_other, obj_e)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(e)

        if not pareto:
            pareto = feasible[:5]  # Fallback

        pareto_set = np.array([e['x'] for e in pareto])
        pareto_front = np.array([list(e['objectives'].values()) for e in pareto])

        return pareto_set, pareto_front


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_globalmoo_client(use_mock: bool = False) -> GlobalMOOClient:
    """
    Get GlobalMOO client instance.

    Args:
        use_mock: If True, return mock client for testing

    Returns:
        GlobalMOOClient or MockGlobalMOOClient
    """
    if use_mock:
        return MockGlobalMOOClient()

    client = GlobalMOOClient()
    if not client.is_available:
        logger.warning("GlobalMOO API not available, falling back to mock client")
        return MockGlobalMOOClient()

    return client
