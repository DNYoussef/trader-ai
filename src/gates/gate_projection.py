"""
Gate Projection using Star-Euler method.

Projects gate progression timing with 95% confidence intervals.
Uses NNC formulas for exact exponential projection.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 3
Formulas: F4 (n = 2/3*(1-k)/(1+w)), F5 (m = 2-2k), F6 (k(L) evolution)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy import stats

# NNC imports
from src.utils.multiplicative import (
    GeometricOperations,
    GeometricDerivative,
    KEvolution,
    MetaFriedmannFormulas,
    BoundedNNC,
    NUMERICAL_EPSILON,
)
from src.utils.nnc_feature_flags import (
    use_star_euler_projection,
    use_k_evolution,
    get_flag,
)

logger = logging.getLogger(__name__)


# Gate thresholds (G0 to G12)
GATE_THRESHOLDS = [
    200,      # G0
    500,      # G1
    1_000,    # G2
    2_500,    # G3
    5_000,    # G4
    10_000,   # G5
    25_000,   # G6
    50_000,   # G7
    100_000,  # G8
    250_000,  # G9
    500_000,  # G10
    1_000_000,  # G11
    10_000_000,  # G12
]


@dataclass
class GateProjection:
    """Projection result for a gate."""
    gate_number: int
    threshold: float
    current_capital: float
    days_to_gate: float
    target_date: datetime
    confidence_interval: Tuple[float, float]  # (lower_days, upper_days)
    confidence_level: float
    probability_of_reaching: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionMetrics:
    """Comprehensive projection metrics."""
    current_gate: int
    current_capital: float
    growth_rate: float
    star_derivative: float
    k_value: float
    expansion_exponent: float
    projections: List[GateProjection]
    timestamp: datetime = field(default_factory=datetime.now)


class StarEulerProjection:
    """
    Gate projection using Star-Euler method from NNC.

    Star-Euler: NAV(t) = NAV(0) * g^t
    where g is the star-derivative (instantaneous growth rate)

    This is EXACT for exponential growth, unlike classical Euler
    which approximates.

    Key Features:
    - Exact exponential projection (no approximation error)
    - k-evolution adjustment for gate-dependent uncertainty
    - 95% confidence intervals using historical volatility
    - Meta-Friedmann expansion exponent for growth modeling
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        lookback_periods: int = 60,
    ):
        """
        Initialize Star-Euler projector.

        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            lookback_periods: Periods for growth estimation
        """
        self._confidence = confidence_level
        self._lookback = lookback_periods
        self._bounded_nnc = BoundedNNC()

    def project_gates(
        self,
        current_capital: float,
        returns_history: List[float],
        current_date: Optional[datetime] = None,
    ) -> ProjectionMetrics:
        """
        Project all future gates from current position.

        Args:
            current_capital: Current portfolio value
            returns_history: Historical returns for estimation
            current_date: Reference date (defaults to now)

        Returns:
            ProjectionMetrics with all gate projections
        """
        current_date = current_date or datetime.now()

        # Find current gate
        current_gate = self._find_current_gate(current_capital)

        # Estimate growth parameters
        growth_rate, volatility = self._estimate_growth(returns_history)

        # Calculate star-derivative
        star_deriv = self._calculate_star_derivative(returns_history)

        # Get k value for current capital
        k = KEvolution.k_for_gate(current_capital) if use_k_evolution() else 0.0

        # Meta-Friedmann expansion exponent (from cosmology analogy)
        # n = (2/3)*(1-k)/(1+w) where w is "equation of state" (set w=0 for matter-dominated)
        # Uses static method: MetaFriedmannFormulas.expansion_exponent(k, w)
        expansion_exp = MetaFriedmannFormulas.expansion_exponent(k, w=0.0)

        # Project each future gate
        projections = []
        for gate_num in range(current_gate + 1, len(GATE_THRESHOLDS)):
            projection = self._project_single_gate(
                gate_number=gate_num,
                current_capital=current_capital,
                growth_rate=growth_rate,
                volatility=volatility,
                star_derivative=star_deriv,
                k_value=k,
                current_date=current_date,
            )
            projections.append(projection)

        return ProjectionMetrics(
            current_gate=current_gate,
            current_capital=current_capital,
            growth_rate=growth_rate,
            star_derivative=star_deriv,
            k_value=k,
            expansion_exponent=expansion_exp,
            projections=projections,
        )

    def project_single_target(
        self,
        current_capital: float,
        target_capital: float,
        returns_history: List[float],
        current_date: Optional[datetime] = None,
    ) -> GateProjection:
        """
        Project time to reach a specific capital target.

        Args:
            current_capital: Current portfolio value
            target_capital: Target value to reach
            returns_history: Historical returns
            current_date: Reference date

        Returns:
            GateProjection for the target
        """
        current_date = current_date or datetime.now()

        growth_rate, volatility = self._estimate_growth(returns_history)
        star_deriv = self._calculate_star_derivative(returns_history)
        k = KEvolution.k_for_gate(current_capital) if use_k_evolution() else 0.0

        # Find which gate this corresponds to (or create custom)
        gate_num = -1
        for i, threshold in enumerate(GATE_THRESHOLDS):
            if abs(threshold - target_capital) / target_capital < 0.01:
                gate_num = i
                break

        return self._project_single_gate(
            gate_number=gate_num,
            current_capital=current_capital,
            target_override=target_capital,
            growth_rate=growth_rate,
            volatility=volatility,
            star_derivative=star_deriv,
            k_value=k,
            current_date=current_date,
        )

    def _project_single_gate(
        self,
        gate_number: int,
        current_capital: float,
        growth_rate: float,
        volatility: float,
        star_derivative: float,
        k_value: float,
        current_date: datetime,
        target_override: Optional[float] = None,
    ) -> GateProjection:
        """Project time to reach a single gate."""

        target = target_override or GATE_THRESHOLDS[gate_number]

        if current_capital >= target:
            # Already at or past this gate
            return GateProjection(
                gate_number=gate_number,
                threshold=target,
                current_capital=current_capital,
                days_to_gate=0,
                target_date=current_date,
                confidence_interval=(0, 0),
                confidence_level=self._confidence,
                probability_of_reaching=1.0,
                method='already_reached',
            )

        if growth_rate <= 0:
            # Cannot reach with zero/negative growth
            return GateProjection(
                gate_number=gate_number,
                threshold=target,
                current_capital=current_capital,
                days_to_gate=float('inf'),
                target_date=current_date + timedelta(days=36500),  # 100 years
                confidence_interval=(float('inf'), float('inf')),
                confidence_level=self._confidence,
                probability_of_reaching=0.0,
                method='negative_growth',
            )

        # Star-Euler projection: t = log(Target/Current) / log(g)
        # where g is star-derivative (instantaneous growth factor)
        ratio = target / current_capital

        if star_derivative > 1.0 and use_star_euler_projection():
            # Use Star-Euler (exact for exponential)
            log_ratio = np.log(ratio)
            log_star = np.log(star_derivative)
            days_to_gate = log_ratio / log_star if log_star > NUMERICAL_EPSILON else float('inf')
            method = 'star_euler'
        else:
            # Fallback to classical exponential
            # t = log(Target/Current) / r
            days_to_gate = np.log(ratio) / growth_rate if growth_rate > NUMERICAL_EPSILON else float('inf')
            method = 'classical'

        # Apply k-adjustment (higher k = more conservative/longer projection)
        # As k increases, uncertainty increases, so we stretch the estimate
        k_adjustment = 1.0 + k_value * 0.5  # 50% stretch at k=1
        days_adjusted = days_to_gate * k_adjustment

        # Calculate confidence interval using volatility
        # Standard error of time estimate
        if volatility > 0 and days_adjusted > 0:
            # Using delta method approximation
            se_days = days_adjusted * volatility / growth_rate if growth_rate > NUMERICAL_EPSILON else days_adjusted * 0.5
            z = stats.norm.ppf((1 + self._confidence) / 2)
            ci_lower = max(1, days_adjusted - z * se_days)
            ci_upper = days_adjusted + z * se_days
        else:
            ci_lower = days_adjusted * 0.5
            ci_upper = days_adjusted * 2.0

        # Probability of reaching (based on Sharpe-like metric)
        # Higher growth/vol ratio = higher probability
        if volatility > 0:
            sharpe_proxy = growth_rate / volatility * np.sqrt(252)
            # Convert to probability using normal CDF
            prob = float(stats.norm.cdf(sharpe_proxy * np.sqrt(days_adjusted / 252)))
        else:
            prob = 0.9 if growth_rate > 0 else 0.1

        target_date = current_date + timedelta(days=int(days_adjusted))

        return GateProjection(
            gate_number=gate_number,
            threshold=target,
            current_capital=current_capital,
            days_to_gate=float(days_adjusted),
            target_date=target_date,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            confidence_level=self._confidence,
            probability_of_reaching=float(np.clip(prob, 0.01, 0.99)),
            method=method,
            metadata={
                'k_value': k_value,
                'k_adjustment': k_adjustment,
                'star_derivative': star_derivative,
                'growth_rate': growth_rate,
                'volatility': volatility,
                'raw_days': float(days_to_gate),
            }
        )

    def _estimate_growth(
        self,
        returns: List[float],
    ) -> Tuple[float, float]:
        """
        Estimate growth rate and volatility from returns.

        Returns:
            Tuple of (daily_growth_rate, daily_volatility)
        """
        if len(returns) < 2:
            return (0.0, 0.1)

        returns_arr = np.array(returns[-self._lookback:])

        # Use geometric mean for growth (more accurate for compound returns)
        geo_mean = GeometricOperations.geometric_mean_return(list(returns_arr))

        # Volatility of log returns
        log_returns = np.log(1 + np.clip(returns_arr, -0.99, np.inf))
        volatility = np.std(log_returns, ddof=1)

        return (float(geo_mean), float(volatility))

    def _calculate_star_derivative(
        self,
        returns: List[float],
    ) -> float:
        """
        Calculate star-derivative (instantaneous growth factor).

        Star-derivative: D*[f] = exp(f'/f)
        For NAV: represents multiplicative growth factor per period
        """
        if len(returns) < 5:
            return 1.0

        # Build NAV series from returns
        nav = [1.0]
        for r in returns[-self._lookback:]:
            nav.append(nav[-1] * (1 + r))

        nav_arr = np.array(nav)

        # Compute geometric derivative
        # GeometricDerivative expects (f_values, x_values) where x is the independent variable
        geo_deriv = GeometricDerivative()
        x_values = np.arange(len(nav_arr), dtype=float)
        star_values = geo_deriv(nav_arr, x_values)

        if len(star_values) > 0:
            # Return geometric mean of recent star-derivatives
            recent_star = star_values[-min(20, len(star_values)):]
            return float(np.clip(GeometricOperations.geometric_mean(list(recent_star)), 0.9, 1.2))

        return 1.0

    def _find_current_gate(self, capital: float) -> int:
        """Find the current gate based on capital."""
        for i, threshold in enumerate(GATE_THRESHOLDS):
            if capital < threshold:
                return max(0, i - 1)
        return len(GATE_THRESHOLDS) - 1


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def project_all_gates(
    current_capital: float,
    returns_history: List[float],
) -> ProjectionMetrics:
    """
    Convenience function to project all gates.

    Args:
        current_capital: Current portfolio value
        returns_history: Historical returns

    Returns:
        ProjectionMetrics with all projections
    """
    projector = StarEulerProjection()
    return projector.project_gates(current_capital, returns_history)


def get_next_gate_projection(
    current_capital: float,
    returns_history: List[float],
) -> Optional[GateProjection]:
    """
    Get projection for next gate only.

    Args:
        current_capital: Current portfolio value
        returns_history: Historical returns

    Returns:
        GateProjection for next gate, or None if at max gate
    """
    metrics = project_all_gates(current_capital, returns_history)

    if metrics.projections:
        return metrics.projections[0]
    return None


def format_projection_summary(metrics: ProjectionMetrics) -> str:
    """
    Format projection metrics as readable summary.

    Args:
        metrics: ProjectionMetrics to format

    Returns:
        Formatted string summary
    """
    lines = [
        f"Gate Projection Summary",
        f"=======================",
        f"Current Gate: G{metrics.current_gate} (${metrics.current_capital:,.0f})",
        f"Growth Rate: {metrics.growth_rate:.2%} daily",
        f"Star-Derivative: {metrics.star_derivative:.4f}",
        f"k-Value: {metrics.k_value:.3f}",
        f"Expansion Exponent: {metrics.expansion_exponent:.3f}",
        f"",
        f"Future Gate Projections:",
        f"------------------------",
    ]

    for proj in metrics.projections[:5]:  # Show next 5 gates
        ci_low = proj.confidence_interval[0]
        ci_high = proj.confidence_interval[1]
        lines.append(
            f"  G{proj.gate_number} (${proj.threshold:,.0f}): "
            f"{proj.days_to_gate:.0f} days "
            f"[{ci_low:.0f}-{ci_high:.0f}] "
            f"({proj.probability_of_reaching:.0%} prob)"
        )

    return "\n".join(lines)
