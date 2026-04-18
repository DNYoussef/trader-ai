"""
Hybrid Risk Model - Runs BOTH additive and multiplicative risk calculations.

Alerts when divergence exceeds threshold, providing safety during NNC rollout.
This is the key integration point for NNC risk calculations.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 3
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime

# NNC imports
from src.utils.multiplicative import (
    GeometricOperations,
    MultiplicativeRisk,
    KEvolution,
    PrelecWeighting,
    MetaFriedmannFormulas,
    NUMERICAL_EPSILON,
)
from src.utils.nnc_feature_flags import (
    use_nnc_risk,
    use_prelec_weighting,
    use_k_evolution,
    parallel_classical_nnc,
    divergence_alert_threshold,
    get_flag,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk calculation results."""
    # Core metrics
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown: float
    volatility: float
    annualized_volatility: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Method used
    method: str  # 'additive', 'multiplicative', or 'hybrid'

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceAlert:
    """Alert when classical and NNC risk diverge significantly."""
    metric_name: str
    classical_value: float
    nnc_value: float
    divergence_pct: float
    threshold_pct: float
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "warning"  # "warning", "critical"

    def __str__(self):
        return (
            f"[{self.severity.upper()}] Risk divergence in {self.metric_name}: "
            f"classical={self.classical_value:.4f}, NNC={self.nnc_value:.4f}, "
            f"divergence={self.divergence_pct:.1%} (threshold={self.threshold_pct:.1%})"
        )


class HybridRiskModel:
    """
    Hybrid risk model running BOTH additive and multiplicative calculations.

    Key Features:
    - Parallel classical + NNC calculations
    - Divergence alerts when results differ significantly
    - Configurable via feature flags
    - k-evolution based on portfolio size
    - Prelec probability weighting for tail risks
    """

    def __init__(
        self,
        divergence_threshold: Optional[float] = None,
        annualization_factor: float = np.sqrt(252),
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize hybrid risk model.

        Args:
            divergence_threshold: Override for divergence alert threshold
            annualization_factor: Factor for annualizing metrics (sqrt(252) for daily)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self._threshold = divergence_threshold or divergence_alert_threshold()
        self._ann_factor = annualization_factor
        self._risk_free_rate = risk_free_rate
        self._alerts: List[DivergenceAlert] = []

    def calculate_classical_risk(
        self,
        returns: np.ndarray,
        portfolio_value: Optional[float] = None,
    ) -> RiskMetrics:
        """
        Calculate risk metrics using classical (additive) methods.

        Args:
            returns: Array of period returns
            portfolio_value: Optional current portfolio value

        Returns:
            RiskMetrics using classical calculations
        """
        returns = np.array(returns)

        if len(returns) < 2:
            return self._empty_metrics('additive')

        # Volatility (standard deviation)
        volatility = np.std(returns, ddof=1)
        ann_volatility = volatility * self._ann_factor

        # VaR (parametric, assumes normal)
        mean_return = np.mean(returns)
        var_95 = mean_return - 1.645 * volatility
        var_99 = mean_return - 2.326 * volatility

        # CVaR (Expected Shortfall)
        sorted_returns = np.sort(returns)
        cutoff_95 = int(len(returns) * 0.05)
        cutoff_95 = max(1, cutoff_95)
        cvar_95 = np.mean(sorted_returns[:cutoff_95])

        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)

        # Sharpe Ratio (annualized)
        daily_rf = self._risk_free_rate / 252
        excess_return = mean_return - daily_rf
        sharpe = (excess_return / volatility * self._ann_factor) if volatility > NUMERICAL_EPSILON else 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else volatility
        sortino = (excess_return / downside_vol * self._ann_factor) if downside_vol > NUMERICAL_EPSILON else 0.0

        # Calmar Ratio
        annualized_return = mean_return * 252
        calmar = annualized_return / max_drawdown if max_drawdown > NUMERICAL_EPSILON else 0.0

        return RiskMetrics(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility),
            annualized_volatility=float(ann_volatility),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            method='additive',
            metadata={'n_periods': len(returns), 'mean_return': float(mean_return)}
        )

    def calculate_multiplicative_risk(
        self,
        returns: np.ndarray,
        portfolio_value: Optional[float] = None,
    ) -> RiskMetrics:
        """
        Calculate risk metrics using multiplicative (NNC) methods.

        Key differences from classical:
        - Geometric mean instead of arithmetic mean
        - Multiplicative risk compounding
        - Prelec weighting for tail probabilities
        - k-evolution based on portfolio size

        Args:
            returns: Array of period returns
            portfolio_value: Optional current portfolio value

        Returns:
            RiskMetrics using NNC calculations
        """
        returns = np.array(returns)

        if len(returns) < 2:
            return self._empty_metrics('multiplicative')

        # Get k value based on portfolio size
        k = 0.0
        if portfolio_value and portfolio_value > 0 and use_k_evolution():
            k = KEvolution.k_for_gate(portfolio_value)

        # Geometric mean return (exact for compound growth)
        geo_return = GeometricOperations.geometric_mean_return(list(returns))

        # Log-returns for volatility (more appropriate for multiplicative)
        log_returns = np.log(1 + np.clip(returns, -0.99, np.inf))
        log_volatility = np.std(log_returns, ddof=1)
        volatility = np.exp(log_volatility) - 1  # Convert back
        ann_volatility = volatility * self._ann_factor

        # VaR using log-normal assumption
        log_mean = np.mean(log_returns)
        log_std = np.std(log_returns, ddof=1)
        var_95_log = log_mean - 1.645 * log_std
        var_99_log = log_mean - 2.326 * log_std
        var_95 = np.exp(var_95_log) - 1
        var_99 = np.exp(var_99_log) - 1

        # CVaR with Prelec weighting (overweight tail risks)
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)

        if use_prelec_weighting():
            # Apply Prelec weights to tail
            weights = np.array([
                PrelecWeighting.weight((i + 1) / n) - PrelecWeighting.weight(i / n)
                for i in range(n)
            ])
            weights = weights / weights.sum()
            cutoff = int(n * 0.05)
            cutoff = max(1, cutoff)
            cvar_95 = np.average(sorted_returns[:cutoff], weights=weights[:cutoff])
        else:
            cutoff = int(n * 0.05)
            cutoff = max(1, cutoff)
            cvar_95 = np.mean(sorted_returns[:cutoff])

        # Max Drawdown (same as classical, already multiplicative)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)

        # NNC Sharpe using geometric return
        daily_rf = self._risk_free_rate / 252
        excess_geo_return = geo_return - daily_rf
        sharpe = (excess_geo_return / volatility * self._ann_factor) if volatility > NUMERICAL_EPSILON else 0.0

        # Adjust Sharpe by k (higher k = more conservative estimate)
        # k=0: full Sharpe, k=1: zero Sharpe (maximum uncertainty)
        sharpe_adjusted = sharpe * (1 - k)

        # Sortino with geometric downside
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_log = np.log(1 + np.clip(downside_returns, -0.99, np.inf))
            downside_vol = np.exp(np.std(downside_log, ddof=1)) - 1
        else:
            downside_vol = volatility
        sortino = (excess_geo_return / downside_vol * self._ann_factor) if downside_vol > NUMERICAL_EPSILON else 0.0

        # Calmar with geometric return
        annualized_geo = (1 + geo_return) ** 252 - 1
        calmar = annualized_geo / max_drawdown if max_drawdown > NUMERICAL_EPSILON else 0.0

        return RiskMetrics(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility),
            annualized_volatility=float(ann_volatility),
            sharpe_ratio=float(sharpe_adjusted),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            method='multiplicative',
            metadata={
                'n_periods': len(returns),
                'geo_return': float(geo_return),
                'k_value': float(k),
                'prelec_weighted': use_prelec_weighting(),
            }
        )

    def calculate_risk(
        self,
        returns: np.ndarray,
        portfolio_value: Optional[float] = None,
    ) -> Tuple[RiskMetrics, List[DivergenceAlert]]:
        """
        Calculate risk using configured method with divergence checking.

        If parallel_classical_nnc is enabled, calculates both and alerts on divergence.

        Args:
            returns: Array of period returns
            portfolio_value: Optional current portfolio value

        Returns:
            Tuple of (primary RiskMetrics, list of DivergenceAlerts)
        """
        self._alerts = []

        # Calculate classical
        classical = self.calculate_classical_risk(returns, portfolio_value)

        # Calculate multiplicative if NNC enabled or parallel mode
        if use_nnc_risk() or parallel_classical_nnc():
            multiplicative = self.calculate_multiplicative_risk(returns, portfolio_value)

            if parallel_classical_nnc():
                # Check for divergence
                self._check_divergence('VaR_95', classical.var_95, multiplicative.var_95)
                self._check_divergence('CVaR_95', classical.cvar_95, multiplicative.cvar_95)
                self._check_divergence('volatility', classical.volatility, multiplicative.volatility)
                self._check_divergence('sharpe_ratio', classical.sharpe_ratio, multiplicative.sharpe_ratio)

            # Return appropriate primary result
            if use_nnc_risk():
                primary = multiplicative
                primary.metadata['classical_sharpe'] = classical.sharpe_ratio
                primary.metadata['classical_var_95'] = classical.var_95
            else:
                primary = classical
                primary.metadata['nnc_sharpe'] = multiplicative.sharpe_ratio
                primary.metadata['nnc_var_95'] = multiplicative.var_95
        else:
            primary = classical

        return primary, self._alerts

    def _check_divergence(
        self,
        metric_name: str,
        classical_value: float,
        nnc_value: float,
    ) -> None:
        """Check and record divergence between classical and NNC values."""
        if abs(classical_value) < NUMERICAL_EPSILON and abs(nnc_value) < NUMERICAL_EPSILON:
            return

        reference = max(abs(classical_value), abs(nnc_value), NUMERICAL_EPSILON)
        divergence = abs(classical_value - nnc_value) / reference

        if divergence > self._threshold:
            severity = "critical" if divergence > self._threshold * 2 else "warning"
            alert = DivergenceAlert(
                metric_name=metric_name,
                classical_value=classical_value,
                nnc_value=nnc_value,
                divergence_pct=divergence,
                threshold_pct=self._threshold,
                severity=severity,
            )
            self._alerts.append(alert)
            logger.warning(str(alert))

    def _empty_metrics(self, method: str) -> RiskMetrics:
        """Return empty metrics for insufficient data."""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            method=method,
            metadata={'error': 'Insufficient data'}
        )

    def calculate_survival_probability(
        self,
        daily_risk: float,
        days: int,
        method: str = 'auto',
    ) -> Dict[str, float]:
        """
        Calculate survival probability over N days.

        Args:
            daily_risk: Daily risk/loss probability
            days: Number of days
            method: 'additive', 'multiplicative', or 'auto'

        Returns:
            Dict with survival probabilities by method
        """
        if method == 'auto':
            method = 'multiplicative' if use_nnc_risk() else 'additive'

        # Multiplicative (correct): P(survive) = (1-r)^n
        mult_survival = MultiplicativeRisk.amplification_factor(daily_risk, days)

        # Additive (incorrect but common): P(survive) = 1 - n*r
        add_survival = max(0, 1 - days * daily_risk)

        result = {
            'multiplicative_survival': mult_survival,
            'additive_survival': add_survival,
            'divergence': abs(mult_survival - add_survival),
            'primary': mult_survival if method == 'multiplicative' else add_survival,
            'method': method,
            'warning': None,
        }

        # Add warning if additive significantly underestimates risk
        if add_survival > 0 and mult_survival / add_survival < 0.8:
            result['warning'] = (
                f"Additive model underestimates risk: "
                f"mult={mult_survival:.3f} vs add={add_survival:.3f}"
            )

        return result

    def get_alerts(self) -> List[DivergenceAlert]:
        """Get list of divergence alerts from last calculation."""
        return self._alerts.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get current model configuration status."""
        return {
            'divergence_threshold': self._threshold,
            'use_nnc_risk': use_nnc_risk(),
            'use_prelec_weighting': use_prelec_weighting(),
            'use_k_evolution': use_k_evolution(),
            'parallel_mode': parallel_classical_nnc(),
            'annualization_factor': self._ann_factor,
            'risk_free_rate': self._risk_free_rate,
            'n_alerts': len(self._alerts),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_hybrid_risk(
    returns: List[float],
    portfolio_value: Optional[float] = None,
) -> Tuple[RiskMetrics, List[DivergenceAlert]]:
    """
    Convenience function for hybrid risk calculation.

    Args:
        returns: List of period returns
        portfolio_value: Optional portfolio value for k-evolution

    Returns:
        Tuple of (RiskMetrics, alerts)
    """
    model = HybridRiskModel()
    return model.calculate_risk(np.array(returns), portfolio_value)


def compare_risk_methods(
    returns: List[float],
    portfolio_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare classical vs NNC risk calculations.

    Args:
        returns: List of period returns
        portfolio_value: Optional portfolio value

    Returns:
        Dict with comparison of both methods
    """
    model = HybridRiskModel()
    returns_arr = np.array(returns)

    classical = model.calculate_classical_risk(returns_arr, portfolio_value)
    nnc = model.calculate_multiplicative_risk(returns_arr, portfolio_value)

    return {
        'classical': {
            'sharpe': classical.sharpe_ratio,
            'var_95': classical.var_95,
            'cvar_95': classical.cvar_95,
            'volatility': classical.volatility,
        },
        'nnc': {
            'sharpe': nnc.sharpe_ratio,
            'var_95': nnc.var_95,
            'cvar_95': nnc.cvar_95,
            'volatility': nnc.volatility,
            'k_value': nnc.metadata.get('k_value', 0),
        },
        'divergence': {
            'sharpe': abs(classical.sharpe_ratio - nnc.sharpe_ratio),
            'var_95': abs(classical.var_95 - nnc.var_95),
            'cvar_95': abs(classical.cvar_95 - nnc.cvar_95),
        }
    }
