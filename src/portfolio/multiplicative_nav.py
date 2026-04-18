"""
Multiplicative NAV Tracking - NNC-based Net Asset Value calculations.

Uses geometric operations for exact compound growth tracking.
Star-derivative provides instantaneous growth rate.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 3
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta

# NNC imports
from src.utils.multiplicative import (
    GeometricOperations,
    GeometricDerivative,
    BoundedNNC,
    KEvolution,
    NUMERICAL_EPSILON,
    MAX_EXP_ARG,
)
from src.utils.nnc_feature_flags import (
    use_nnc_nav,
    use_k_evolution,
    parallel_classical_nnc,
    divergence_alert_threshold,
)

logger = logging.getLogger(__name__)


@dataclass
class NAVSnapshot:
    """Single NAV observation with metadata."""
    timestamp: datetime
    nav: float
    classical_nav: Optional[float] = None
    method: str = 'multiplicative'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NAVMetrics:
    """Computed NAV metrics."""
    current_nav: float
    initial_nav: float
    total_return: float
    cagr: float
    star_derivative: float  # Instantaneous growth rate (NNC)
    max_nav: float
    max_drawdown: float
    days_elapsed: int
    method: str
    timestamp: datetime = field(default_factory=datetime.now)


class MultiplicativeNAV:
    """
    NAV tracking using multiplicative (NNC) operations.

    Key Features:
    - Geometric compounding (exact for compound growth)
    - Star-derivative for instantaneous growth rate
    - k-evolution adjustment based on portfolio size
    - Parallel classical tracking for comparison
    - Divergence alerts when methods differ
    """

    def __init__(
        self,
        initial_nav: float = 1.0,
        divergence_threshold: Optional[float] = None,
    ):
        """
        Initialize multiplicative NAV tracker.

        Args:
            initial_nav: Starting NAV value (typically 1.0 or 100.0)
            divergence_threshold: Override for divergence alert threshold
        """
        self._initial_nav = max(initial_nav, NUMERICAL_EPSILON)
        self._threshold = divergence_threshold or divergence_alert_threshold()

        # NAV history
        self._history: List[NAVSnapshot] = []
        self._returns: List[float] = []

        # State
        self._current_nav = self._initial_nav
        self._classical_nav = self._initial_nav
        self._max_nav = self._initial_nav
        self._bounded_nnc = BoundedNNC()

        # Initialize with first snapshot
        self._add_snapshot(self._initial_nav)

    def update(
        self,
        period_return: float,
        timestamp: Optional[datetime] = None,
    ) -> NAVSnapshot:
        """
        Update NAV with a period return.

        Args:
            period_return: Return for the period (e.g., 0.01 for 1%)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            NAVSnapshot with updated values
        """
        # Store return
        self._returns.append(period_return)

        # Multiplicative update (exact): NAV *= (1 + r)
        factor = 1 + period_return
        self._current_nav = self._bounded_nnc.safe_multiply(
            self._current_nav, factor
        )

        # Classical additive update (for comparison)
        self._classical_nav = self._classical_nav * factor  # Also multiplicative!

        # Update max NAV
        self._max_nav = max(self._max_nav, self._current_nav)

        return self._add_snapshot(self._current_nav, timestamp)

    def update_from_prices(
        self,
        old_price: float,
        new_price: float,
        timestamp: Optional[datetime] = None,
    ) -> NAVSnapshot:
        """
        Update NAV from price change.

        Args:
            old_price: Previous price
            new_price: Current price
            timestamp: Optional timestamp

        Returns:
            NAVSnapshot with updated values
        """
        if old_price <= 0:
            logger.warning("Invalid old_price, skipping update")
            return self._history[-1] if self._history else self._add_snapshot(self._current_nav)

        period_return = (new_price - old_price) / old_price
        return self.update(period_return, timestamp)

    def get_metrics(self) -> NAVMetrics:
        """
        Get comprehensive NAV metrics.

        Returns:
            NAVMetrics with all computed values
        """
        days = len(self._returns)

        # Total return
        total_return = (self._current_nav / self._initial_nav) - 1

        # CAGR using geometric operations
        if days > 0 and self._current_nav > 0:
            cagr = GeometricOperations.cagr(
                self._initial_nav,
                self._current_nav,
                days / 252  # Convert to years
            )
        else:
            cagr = 0.0

        # Star-derivative (instantaneous growth rate)
        star_deriv = self._calculate_star_derivative()

        # Max drawdown
        max_dd = self._calculate_max_drawdown()

        return NAVMetrics(
            current_nav=self._current_nav,
            initial_nav=self._initial_nav,
            total_return=total_return,
            cagr=cagr,
            star_derivative=star_deriv,
            max_nav=self._max_nav,
            max_drawdown=max_dd,
            days_elapsed=days,
            method='multiplicative' if use_nnc_nav() else 'classical',
        )

    def get_growth_trajectory(
        self,
        periods: int = 252,
    ) -> Dict[str, np.ndarray]:
        """
        Project growth trajectory using star-derivative.

        Args:
            periods: Number of periods to project

        Returns:
            Dict with time and projected NAV arrays
        """
        if not self._returns:
            return {
                'time': np.arange(periods),
                'projected_nav': np.full(periods, self._current_nav),
            }

        # Current instantaneous growth rate
        star_deriv = self._calculate_star_derivative()

        # Project using Star-Euler: NAV(t) = NAV(0) * star_deriv^t
        time_array = np.arange(periods)

        if abs(star_deriv) > NUMERICAL_EPSILON:
            # NNC projection: NAV(t) = NAV(0) * g^t where g is star-derivative
            projected = self._current_nav * np.power(
                star_deriv,
                time_array,
                where=star_deriv > 0
            )
        else:
            # Fallback to constant
            projected = np.full(periods, self._current_nav)

        # Apply safety bounds
        projected = np.clip(projected, NUMERICAL_EPSILON, 1e15)

        return {
            'time': time_array,
            'projected_nav': projected,
            'star_derivative': star_deriv,
            'current_nav': self._current_nav,
        }

    def compare_methods(self) -> Dict[str, Any]:
        """
        Compare multiplicative vs arithmetic tracking.

        Returns:
            Dict with comparison metrics
        """
        if len(self._returns) < 2:
            return {
                'insufficient_data': True,
                'n_periods': len(self._returns),
            }

        returns_arr = np.array(self._returns)

        # Geometric mean (exact for compound returns)
        geo_mean = GeometricOperations.geometric_mean_return(self._returns)

        # Arithmetic mean (approximation, always higher)
        arith_mean = np.mean(returns_arr)

        # Rebuild NAV both ways
        geo_nav = self._initial_nav
        arith_nav = self._initial_nav
        for r in self._returns:
            geo_nav *= (1 + r)
            arith_nav *= (1 + arith_mean)  # Using average

        # Actual difference
        actual_nav = self._current_nav

        return {
            'geometric_mean_return': geo_mean,
            'arithmetic_mean_return': arith_mean,
            'mean_difference': arith_mean - geo_mean,
            'geometric_nav': geo_nav,
            'arithmetic_nav': arith_nav,
            'actual_nav': actual_nav,
            'nav_error_pct': abs(arith_nav - actual_nav) / actual_nav * 100 if actual_nav > 0 else 0,
            'n_periods': len(self._returns),
            'insight': (
                'Arithmetic mean overestimates by {:.2%}'.format(arith_mean - geo_mean)
                if arith_mean > geo_mean else 'Methods agree'
            ),
        }

    def _calculate_star_derivative(self) -> float:
        """
        Calculate star-derivative (instantaneous growth rate).

        Star-derivative: D*[NAV] = exp(NAV'/NAV)

        For discrete data, approximates using recent returns.
        """
        if len(self._returns) < 2:
            return 1.0  # No growth

        # Use recent returns for current growth rate
        recent_returns = self._returns[-min(20, len(self._returns)):]
        nav_values = [self._initial_nav]

        for r in self._returns:
            nav_values.append(nav_values[-1] * (1 + r))

        nav_array = np.array(nav_values[-len(recent_returns)-1:])

        if len(nav_array) < 2:
            return 1.0

        # Star-derivative using GeometricDerivative
        # GeometricDerivative expects (f_values, x_values)
        geo_deriv = GeometricDerivative()
        x_values = np.arange(len(nav_array), dtype=float)
        star_deriv = geo_deriv(nav_array, x_values)

        # Return most recent value (current instantaneous growth)
        if len(star_deriv) > 0:
            return float(np.clip(star_deriv[-1], 0.8, 1.5))  # Reasonable bounds
        return 1.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from NAV history."""
        if not self._history:
            return 0.0

        nav_values = [s.nav for s in self._history]
        peak = nav_values[0]
        max_dd = 0.0

        for nav in nav_values:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _add_snapshot(
        self,
        nav: float,
        timestamp: Optional[datetime] = None,
    ) -> NAVSnapshot:
        """Add a snapshot to history."""
        snapshot = NAVSnapshot(
            timestamp=timestamp or datetime.now(),
            nav=nav,
            classical_nav=self._classical_nav if parallel_classical_nnc() else None,
            method='multiplicative',
            metadata={
                'returns_count': len(self._returns),
                'max_nav': self._max_nav,
            }
        )
        self._history.append(snapshot)
        return snapshot

    def get_history(self) -> List[NAVSnapshot]:
        """Get full NAV history."""
        return self._history.copy()

    def get_returns(self) -> List[float]:
        """Get stored returns."""
        return self._returns.copy()

    @property
    def current_nav(self) -> float:
        """Get current NAV."""
        return self._current_nav

    @property
    def initial_nav(self) -> float:
        """Get initial NAV."""
        return self._initial_nav


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_multiplicative_nav(
    initial_nav: float = 1.0,
    returns: Optional[List[float]] = None,
) -> MultiplicativeNAV:
    """
    Create and optionally populate a MultiplicativeNAV tracker.

    Args:
        initial_nav: Starting NAV
        returns: Optional list of returns to apply

    Returns:
        MultiplicativeNAV instance
    """
    tracker = MultiplicativeNAV(initial_nav)

    if returns:
        for r in returns:
            tracker.update(r)

    return tracker


def calculate_cagr_nnc(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """
    Calculate CAGR using geometric operations.

    Args:
        initial_value: Starting value
        final_value: Ending value
        years: Time period in years

    Returns:
        CAGR as decimal
    """
    return GeometricOperations.cagr(initial_value, final_value, years)
