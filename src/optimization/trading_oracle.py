"""
Black-box evaluator for trading objectives.
Single interface for all MOO problems.

This module provides the oracle interface that MOO adapters consume.
Oracles evaluate decision variables against multiple objectives.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 2
PATTERN: meta_calculus/moo_integration.py PhysicsOracle
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging

# NNC imports
from src.utils.multiplicative import (
    GeometricOperations,
    MultiplicativeRisk,
    KEvolution,
    NUMERICAL_EPSILON,
)
from src.utils.nnc_feature_flags import use_nnc_returns, get_flag

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveSpec:
    """Specification for a single objective."""
    name: str
    direction: str = "minimize"  # "minimize" or "maximize"
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    weight: float = 1.0
    description: str = ""


@dataclass
class ConstraintSpec:
    """Specification for a single constraint."""
    name: str
    type: str  # "eq" (equality), "ineq_le" (<=), "ineq_ge" (>=)
    expression: str = ""
    description: str = ""


@dataclass
class OracleResult:
    """Result from oracle evaluation."""
    objectives: Dict[str, float]
    constraints: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feasible: bool = True


class TradingOracle(ABC):
    """
    Abstract oracle that MOO adapters consume.

    All objectives are MINIMIZED by convention.
    To maximize something (e.g., return), negate it.
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> OracleResult:
        """
        Evaluate decision variables against objectives.

        Args:
            x: Decision variable vector (e.g., allocation weights)

        Returns:
            OracleResult with objective values (minimize all by convention)
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lower_bounds, upper_bounds) for decision variables.

        Returns:
            Tuple of (lb, ub) numpy arrays
        """
        pass

    @abstractmethod
    def get_objectives(self) -> List[ObjectiveSpec]:
        """Return list of objective specifications."""
        pass

    @abstractmethod
    def get_constraints(self) -> List[ConstraintSpec]:
        """Return list of constraint specifications."""
        pass

    @property
    def n_var(self) -> int:
        """Number of decision variables."""
        lb, _ = self.get_bounds()
        return len(lb)

    @property
    def n_obj(self) -> int:
        """Number of objectives."""
        return len(self.get_objectives())

    @property
    def n_constr(self) -> int:
        """Number of constraints."""
        return len(self.get_constraints())


class PortfolioOracle(TradingOracle):
    """
    Oracle for portfolio allocation problems.

    Decision variables: Asset weights [w1, w2, ..., wn]
    Objectives: neg_return (minimize), volatility (minimize), concentration (minimize)
    Constraint: Weights sum to 1
    """

    def __init__(
        self,
        returns_data: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None,
        use_geometric_returns: Optional[bool] = None,
        risk_free_rate: float = 0.04 / 252,  # Daily rate
    ):
        """
        Initialize portfolio oracle.

        Args:
            returns_data: Expected returns for each asset (n_assets,)
            cov_matrix: Covariance matrix (n_assets x n_assets)
            asset_names: Optional names for assets
            use_geometric_returns: Override NNC flag for returns calculation
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns = np.array(returns_data)
        self.cov = np.array(cov_matrix)
        self.n_assets = len(returns_data)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
        self.risk_free_rate = risk_free_rate

        # Use flag if not explicitly specified
        self._use_geometric = use_geometric_returns if use_geometric_returns is not None else use_nnc_returns()

        # Validate inputs
        if self.cov.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Covariance matrix shape mismatch: expected ({self.n_assets}, {self.n_assets})")

    def evaluate(self, weights: np.ndarray) -> OracleResult:
        """
        Evaluate portfolio allocation.

        Args:
            weights: Asset allocation weights (should sum to 1)

        Returns:
            OracleResult with neg_return, volatility, concentration, neg_sharpe
        """
        weights = np.array(weights).flatten()

        if len(weights) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} weights, got {len(weights)}")

        # Negate return (we minimize, so negate to maximize)
        expected_return = np.dot(weights, self.returns)
        neg_return = -expected_return

        # Volatility (minimize)
        portfolio_var = np.dot(weights.T, np.dot(self.cov, weights))
        volatility = np.sqrt(max(portfolio_var, 0))

        # Max weight (minimize concentration)
        concentration = np.max(weights)

        # Negative Sharpe ratio (minimize)
        if volatility > NUMERICAL_EPSILON:
            sharpe = (expected_return - self.risk_free_rate) / volatility
        else:
            sharpe = 0.0
        neg_sharpe = -sharpe

        # Check constraint: weights sum to 1
        weight_sum = np.sum(weights)
        constraint_violation = abs(weight_sum - 1.0)

        return OracleResult(
            objectives={
                'neg_return': float(neg_return),
                'volatility': float(volatility),
                'concentration': float(concentration),
                'neg_sharpe': float(neg_sharpe),
            },
            constraints={
                'weight_sum_eq_1': float(constraint_violation),
            },
            metadata={
                'expected_return': float(expected_return),
                'sharpe_ratio': float(sharpe),
                'weight_sum': float(weight_sum),
                'use_geometric': self._use_geometric,
            },
            feasible=constraint_violation < 0.01
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Asset weights bounded in [0, 1]."""
        lb = np.zeros(self.n_assets)
        ub = np.ones(self.n_assets)
        return (lb, ub)

    def get_objectives(self) -> List[ObjectiveSpec]:
        """Four objectives: neg_return, volatility, concentration, neg_sharpe."""
        return [
            ObjectiveSpec(
                name='neg_return',
                direction='minimize',
                bounds=(-1, 1),
                description='Negative expected return (minimizing maximizes return)'
            ),
            ObjectiveSpec(
                name='volatility',
                direction='minimize',
                bounds=(0, 1),
                description='Portfolio volatility (standard deviation)'
            ),
            ObjectiveSpec(
                name='concentration',
                direction='minimize',
                bounds=(0, 1),
                description='Maximum weight (lower = more diversified)'
            ),
            ObjectiveSpec(
                name='neg_sharpe',
                direction='minimize',
                bounds=(-10, 10),
                description='Negative Sharpe ratio'
            ),
        ]

    def get_constraints(self) -> List[ConstraintSpec]:
        """Single constraint: weights sum to 1."""
        return [
            ConstraintSpec(
                name='weight_sum_eq_1',
                type='eq',
                expression='sum(weights) == 1',
                description='Portfolio weights must sum to 1'
            )
        ]


class StrategySelectionOracle(TradingOracle):
    """
    Oracle for strategy selection optimization.

    Decision variables: Strategy weights or allocation parameters
    Objectives: neg_return, max_drawdown, neg_win_rate, neg_consensus
    """

    def __init__(
        self,
        historical_returns: np.ndarray,
        strategy_names: Optional[List[str]] = None,
        n_strategies: int = 8,
    ):
        """
        Initialize strategy selection oracle.

        Args:
            historical_returns: Matrix of historical returns (n_periods x n_strategies)
            strategy_names: Names of strategies
            n_strategies: Number of strategies
        """
        self.historical_returns = np.array(historical_returns)

        if len(self.historical_returns.shape) == 1:
            # Single strategy, reshape
            self.historical_returns = self.historical_returns.reshape(-1, 1)

        self.n_periods = self.historical_returns.shape[0]
        self.n_strategies = self.historical_returns.shape[1] if len(self.historical_returns.shape) > 1 else n_strategies

        self.strategy_names = strategy_names or [
            'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
            'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
        ][:self.n_strategies]

    def evaluate(self, strategy_weights: np.ndarray) -> OracleResult:
        """
        Evaluate strategy allocation.

        Args:
            strategy_weights: Weight for each strategy (should sum to 1)

        Returns:
            OracleResult with multiple objectives
        """
        weights = np.array(strategy_weights).flatten()

        if len(weights) != self.n_strategies:
            raise ValueError(f"Expected {self.n_strategies} weights, got {len(weights)}")

        # Portfolio returns as weighted combination
        portfolio_returns = np.dot(self.historical_returns, weights)

        # Objective 1: Negative geometric return (minimize to maximize)
        geo_return = GeometricOperations.geometric_mean_return(list(portfolio_returns))
        neg_return = -geo_return

        # Objective 2: Maximum drawdown (minimize)
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = float(np.max(drawdowns))

        # Objective 3: Negative win rate (minimize to maximize)
        win_rate = np.mean(portfolio_returns > 0)
        neg_win_rate = -win_rate

        # Objective 4: Strategy concentration (minimize)
        concentration = float(np.max(weights))

        # Constraint: weights sum to 1
        weight_sum = np.sum(weights)
        constraint_violation = abs(weight_sum - 1.0)

        return OracleResult(
            objectives={
                'neg_return': float(neg_return),
                'max_drawdown': float(max_drawdown),
                'neg_win_rate': float(neg_win_rate),
                'concentration': float(concentration),
            },
            constraints={
                'weight_sum_eq_1': float(constraint_violation),
            },
            metadata={
                'geometric_return': float(geo_return),
                'win_rate': float(win_rate),
                'n_periods': self.n_periods,
            },
            feasible=constraint_violation < 0.01
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy weights bounded in [0, 1]."""
        lb = np.zeros(self.n_strategies)
        ub = np.ones(self.n_strategies)
        return (lb, ub)

    def get_objectives(self) -> List[ObjectiveSpec]:
        """Four objectives for strategy selection."""
        return [
            ObjectiveSpec(name='neg_return', direction='minimize',
                         description='Negative geometric return'),
            ObjectiveSpec(name='max_drawdown', direction='minimize',
                         description='Maximum drawdown'),
            ObjectiveSpec(name='neg_win_rate', direction='minimize',
                         description='Negative win rate'),
            ObjectiveSpec(name='concentration', direction='minimize',
                         description='Strategy concentration'),
        ]

    def get_constraints(self) -> List[ConstraintSpec]:
        """Weights sum to 1."""
        return [
            ConstraintSpec(name='weight_sum_eq_1', type='eq',
                          expression='sum(weights) == 1')
        ]


class GateProgressionOracle(TradingOracle):
    """
    Oracle for optimizing gate progression parameters.

    Decision variables: [k_weight, volatility_regime_w, risk_tolerance]
    Objectives: neg_growth_rate, time_to_next_gate, risk_adjusted_progression
    """

    def __init__(
        self,
        current_nav: float,
        target_nav: float,
        historical_returns: np.ndarray,
        current_gate_capital: float = 200.0,
    ):
        """
        Initialize gate progression oracle.

        Args:
            current_nav: Current portfolio NAV
            target_nav: Target NAV for next gate
            historical_returns: Historical daily returns
            current_gate_capital: Capital at current gate level
        """
        self.current_nav = current_nav
        self.target_nav = target_nav
        self.historical_returns = np.array(historical_returns)
        self.current_gate_capital = current_gate_capital

        # Get k from current capital
        self.base_k = KEvolution.k_for_gate(current_gate_capital)

    def evaluate(self, params: np.ndarray) -> OracleResult:
        """
        Evaluate gate progression parameters.

        Args:
            params: [k_adjustment, w_regime, risk_tolerance]

        Returns:
            OracleResult with progression objectives
        """
        params = np.array(params).flatten()
        k_adjustment, w_regime, risk_tolerance = params[0], params[1], params[2]

        # Adjusted k
        k = np.clip(self.base_k + k_adjustment, 0, 1)

        # Calculate growth metrics using NNC formulas
        # n = (2/3) * (1 - k) / (1 + w)
        from src.utils.multiplicative import MetaFriedmannFormulas
        n = MetaFriedmannFormulas.expansion_exponent(k, w_regime)

        # Simulate growth with geometric returns
        geo_return = GeometricOperations.geometric_mean_return(list(self.historical_returns))

        # Objective 1: Negative growth rate (minimize to maximize)
        if geo_return > 0:
            # Adjust growth by expansion exponent
            adjusted_growth = geo_return * (1 + n)
            neg_growth = -adjusted_growth
        else:
            neg_growth = 0.0

        # Objective 2: Time to next gate (minimize)
        if geo_return > NUMERICAL_EPSILON:
            growth_needed = self.target_nav / self.current_nav
            time_to_gate = np.log(growth_needed) / np.log(1 + geo_return)
        else:
            time_to_gate = float('inf')

        # Objective 3: Risk-adjusted metric (minimize)
        volatility = np.std(self.historical_returns)
        if volatility > NUMERICAL_EPSILON:
            risk_adjusted = volatility / (1 + risk_tolerance)
        else:
            risk_adjusted = 0.0

        return OracleResult(
            objectives={
                'neg_growth_rate': float(neg_growth),
                'time_to_gate': float(min(time_to_gate, 1000)),  # Cap at 1000 days
                'risk_adjusted': float(risk_adjusted),
            },
            metadata={
                'k_value': float(k),
                'expansion_n': float(n),
                'geo_return': float(geo_return),
            },
            feasible=True
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parameter bounds."""
        lb = np.array([-0.1, -0.5, 0.0])  # k_adj, w_regime, risk_tol
        ub = np.array([0.1, 0.5, 1.0])
        return (lb, ub)

    def get_objectives(self) -> List[ObjectiveSpec]:
        """Three objectives for gate progression."""
        return [
            ObjectiveSpec(name='neg_growth_rate', direction='minimize'),
            ObjectiveSpec(name='time_to_gate', direction='minimize'),
            ObjectiveSpec(name='risk_adjusted', direction='minimize'),
        ]

    def get_constraints(self) -> List[ConstraintSpec]:
        """No explicit constraints."""
        return []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_oracle(
    oracle_type: str,
    **kwargs
) -> TradingOracle:
    """
    Factory function to create oracles.

    Args:
        oracle_type: One of 'portfolio', 'strategy', 'gate'
        **kwargs: Arguments passed to oracle constructor

    Returns:
        Appropriate TradingOracle subclass
    """
    oracle_map = {
        'portfolio': PortfolioOracle,
        'strategy': StrategySelectionOracle,
        'gate': GateProgressionOracle,
    }

    if oracle_type not in oracle_map:
        raise ValueError(f"Unknown oracle type: {oracle_type}. Available: {list(oracle_map.keys())}")

    return oracle_map[oracle_type](**kwargs)
