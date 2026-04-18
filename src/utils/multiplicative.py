"""
Single source of truth for all Non-Newtonian Calculus (NNC) operations.
DO NOT duplicate this logic elsewhere.

FORMULAS COPIED EXACTLY FROM:
- meta_calculus/core/derivatives.py
- meta_calculus/scalar_friedmann.py
- meta_calculus/k_evolution.py

Author: trader-ai NNC Integration
Version: 1.0.0
Source: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1
"""
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

# =============================================================================
# CONSTANTS FROM META-CALCULUS-TOOLKIT
# =============================================================================

# Numerical stability constants
# SOURCE: meta_calculus/core/derivatives.py line 54
NUMERICAL_EPSILON = 1e-12

# Maximum exponent to prevent overflow
# SOURCE: meta_calculus/core/derivatives.py line 62
BETA_PRIME_MAX = 1e100

# Maximum exponential argument (prevents overflow)
MAX_EXP_ARG = 700  # np.exp(709) ~ overflow


# =============================================================================
# F1: GEOMETRIC DERIVATIVE
# =============================================================================

class GeometricDerivative:
    """
    Compute geometric derivatives (alpha=I, beta=exp).

    FORMULA (F1): D_G[f](a) = exp(f'(a) / f(a))

    SOURCE: meta_calculus/core/derivatives.py:76-77, 123

    Key Property: Exponential functions have CONSTANT geometric derivative
    D_G[e^(kx)] = e^k (independent of x)

    Use Case for Trader-AI: Compound growth rate of NAV, position returns
    """

    def __init__(self, epsilon: float = NUMERICAL_EPSILON):
        """
        Initialize geometric derivative calculator.

        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon

    def __call__(self, f_values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """
        Compute geometric derivative from discrete values.

        EXACT IMPLEMENTATION FROM: derivatives.py:122-124

        Args:
            f_values: Function values f(x)
            x_values: Independent variable values

        Returns:
            Geometric derivative D_G[f](x) at each point
        """
        # Compute classical derivative numerically (central difference via gradient)
        df_dx = np.gradient(f_values, x_values)

        # Handle potential division by zero
        # SOURCE: derivatives.py:120
        fx_safe = np.where(np.abs(f_values) < self.epsilon, self.epsilon, f_values)

        # Compute ratio and clip to prevent overflow
        ratio = df_dx / fx_safe
        ratio_clipped = np.clip(ratio, -MAX_EXP_ARG, MAX_EXP_ARG)

        # FORMULA: D_G[f](x) = exp(f'(x) / f(x))
        # SOURCE: derivatives.py:123
        geo_deriv = np.exp(ratio_clipped)

        return geo_deriv

    def at_point(self, f_prime: float, f_val: float) -> float:
        """
        Compute geometric derivative at a single point.

        Args:
            f_prime: Classical derivative f'(a)
            f_val: Function value f(a)

        Returns:
            D_G[f](a)
        """
        if abs(f_val) < self.epsilon:
            f_val = self.epsilon if f_val >= 0 else -self.epsilon

        ratio = f_prime / f_val
        ratio_clipped = max(min(ratio, MAX_EXP_ARG), -MAX_EXP_ARG)

        return np.exp(ratio_clipped)


# =============================================================================
# F2: BIGEOMETRIC DERIVATIVE (SCALE-INVARIANT)
# =============================================================================

class BigeometricDerivative:
    """
    Compute bigeometric derivatives (alpha=exp, beta=exp).

    FORMULA (F2): D_BG[f](a) = exp(a * f'(a) / f(a)) = exp(elasticity)

    SOURCE: meta_calculus/core/derivatives.py:168-169, 230-231

    CRITICAL PROPERTY - Power Law Theorem:
    D_BG[x^n] = e^n  (CONSTANT for all x > 0!)

    This is the ELASTICITY formula - measures scale-invariant rates.

    Use Case for Trader-AI: Gate progression (power-law growth), position sizing
    """

    def __init__(self, epsilon: float = NUMERICAL_EPSILON):
        """
        Initialize bigeometric derivative calculator.

        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon

    def __call__(self, f_values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """
        Compute bigeometric derivative from discrete values.

        EXACT IMPLEMENTATION FROM: derivatives.py:228-232

        Args:
            f_values: Function values f(x)
            x_values: Independent variable values (MUST be positive)

        Returns:
            Bigeometric derivative D_BG[f](x) at each point
        """
        # Compute classical derivative numerically
        df_dx = np.gradient(f_values, x_values)

        # Handle potential division by zero
        fx_safe = np.where(np.abs(f_values) < self.epsilon, self.epsilon, f_values)

        # FORMULA: elasticity = x * f'(x) / f(x)
        # SOURCE: derivatives.py:230
        elasticity = x_values * df_dx / fx_safe

        # Clip to prevent overflow
        elasticity_clipped = np.clip(elasticity, -MAX_EXP_ARG, MAX_EXP_ARG)

        # FORMULA: D_BG[f](x) = exp(elasticity)
        # SOURCE: derivatives.py:231
        bigeo_deriv = np.exp(elasticity_clipped)

        return bigeo_deriv

    def elasticity(self, f_values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
        """
        FORMULA (F3): elasticity = a * f'(a) / f(a)

        Fractional change in output per fractional change in input.
        If elasticity = 2, a 1% increase in input causes a 2% increase in output.

        SOURCE: meta_calculus/core/derivatives.py:170-171
                meta_calculus/bigeometric_operators.py:75

        Args:
            f_values: Function values f(x)
            x_values: Independent variable values

        Returns:
            Elasticity at each point
        """
        df_dx = np.gradient(f_values, x_values)
        fx_safe = np.where(np.abs(f_values) < self.epsilon, self.epsilon, f_values)
        return x_values * df_dx / fx_safe

    def at_point(self, x: float, f_prime: float, f_val: float) -> float:
        """
        Compute bigeometric derivative at a single point.

        Args:
            x: Independent variable value (MUST be positive)
            f_prime: Classical derivative f'(x)
            f_val: Function value f(x)

        Returns:
            D_BG[f](x)
        """
        if abs(f_val) < self.epsilon:
            f_val = self.epsilon if f_val >= 0 else -self.epsilon

        elasticity = x * f_prime / f_val
        elasticity_clipped = max(min(elasticity, MAX_EXP_ARG), -MAX_EXP_ARG)

        return np.exp(elasticity_clipped)


# =============================================================================
# F4, F5: META-FRIEDMANN FORMULAS
# =============================================================================

class MetaFriedmannFormulas:
    """
    Meta-Friedmann cosmology formulas adapted for trader-ai.

    These formulas describe how expansion and density exponents
    depend on the meta-weight parameter k.

    SOURCE: meta_calculus/scalar_friedmann.py:240-298

    For Trader-AI:
    - k = risk adaptation parameter (varies by gate/capital)
    - w = volatility regime indicator
    - n = growth exponent for portfolio value
    - m = risk density decay rate
    """

    @staticmethod
    def expansion_exponent(k: float, w: float) -> float:
        """
        FORMULA (F4): n = (2/3) * (1 - k) / (1 + w)

        SOURCE: meta_calculus/scalar_friedmann.py:240-244
        EXACT CODE: return (2.0 / 3.0) * (1.0 - self.k) / (1.0 + self.w)

        Variables:
        - k: Meta-weight parameter (0 = classical, 1 = constant)
        - w: Equation of state (-1 to 1)
        - n: Expansion exponent (a(t) ~ t^n)

        Special Cases:
        - k = 0 (classical):  n = 2 / (3 * (1 + w))
        - k = 1 (constant):   n = 0 (no growth)

        For Trader-AI: w = volatility regime indicator
        - w < 0: Low volatility (faster growth allowed)
        - w > 0: High volatility (slower growth, more caution)

        Args:
            k: Meta-weight parameter (0 <= k <= 1)
            w: Equation of state (-1 < w <= 1)

        Returns:
            Expansion exponent n
        """
        if w <= -1:
            return float('inf')  # Singularity at w = -1

        return (2.0 / 3.0) * (1.0 - k) / (1.0 + w)

    @staticmethod
    def density_exponent(k: float) -> float:
        """
        FORMULA (F5): m = 2 - 2k

        SOURCE: meta_calculus/scalar_friedmann.py:246-248
        EXACT CODE: return 2.0 - 2.0 * self.k

        Variables:
        - k: Meta-weight parameter
        - m: Density exponent (rho ~ t^(-m))

        Interpretation:
        - k = 0: m = 2 (classical singularity at t=0)
        - k = 1: m = 0 (constant density, NO singularity!)
        - k > 1: m < 0 (density vanishes at t=0)

        For Trader-AI: Risk density decay rate across gates

        Args:
            k: Meta-weight parameter

        Returns:
            Density exponent m
        """
        return 2.0 - 2.0 * k

    @staticmethod
    def meta_hubble(n: float, k: float, t: float) -> float:
        """
        Meta-Hubble parameter: H_meta = n * t^(k-1)

        SOURCE: meta_calculus/scalar_friedmann.py:278-287

        For Trader-AI: Instantaneous "growth rate" at time t

        Args:
            n: Expansion exponent
            k: Meta-weight parameter
            t: Time parameter (must be > 0)

        Returns:
            Meta-Hubble parameter
        """
        if t <= 0:
            if k > 1:
                return 0.0
            elif k < 1:
                return float('inf')
            else:
                return n
        return n * t ** (k - 1)


# =============================================================================
# F6: k(L) SPATIAL PATTERN (MOO-VERIFIED)
# =============================================================================

class KEvolution:
    """
    Compute k(L) spatial pattern - VERIFIED BY MOO.

    FORMULA (F6): k(L) = -0.0137 * log10(L) + 0.1593

    Statistics:
    - R^2 = 0.71 (explains 71% of variance)
    - p-value = 0.008 (statistically significant)

    SOURCE: meta_calculus/k_evolution.py:9-11, 74-88

    For Trader-AI:
    - G0 ($200):    L ~ small  -> k ~ 0.15 (more classical)
    - G12 ($10M+):  L ~ large  -> k ~ 0.05 (more meta)
    """

    # Verified parameters from MOO optimization
    # SOURCE: k_evolution.py:41-42
    SPATIAL_SLOPE = -0.0137
    SPATIAL_INTERCEPT = 0.1593

    # Valid range for k
    K_MIN = 0.0
    K_MAX = 1.0

    # Default scaling factor for capital -> length scale conversion
    CAPITAL_SCALE_FACTOR = 1e6

    @classmethod
    def k_spatial(cls, L: float) -> float:
        """
        Compute k from spatial pattern.

        EXACT IMPLEMENTATION FROM: k_evolution.py:70-88
        k = self.params.spatial_slope * log_L + self.params.spatial_intercept

        Args:
            L: Length scale (for trader-ai: portfolio size proxy)

        Returns:
            k value from spatial pattern, clipped to [K_MIN, K_MAX]
        """
        if L <= 0:
            return cls.K_MAX  # Maximum caution for invalid input

        log_L = np.log10(L)
        k = cls.SPATIAL_SLOPE * log_L + cls.SPATIAL_INTERCEPT
        return float(np.clip(k, cls.K_MIN, cls.K_MAX))

    @classmethod
    def k_for_gate(cls, gate_capital: float) -> float:
        """
        Map gate capital to k value.

        For Trader-AI gates:
        - G0 ($200): k ~ 0.15 (more classical, conservative)
        - G12 ($10M+): k ~ 0.05 (more meta, aggressive)

        Args:
            gate_capital: Capital at gate level (USD)

        Returns:
            k value appropriate for the gate
        """
        if gate_capital <= 0:
            return cls.K_MAX

        # Use capital as proxy for L (scaling factor for finance)
        L = gate_capital * cls.CAPITAL_SCALE_FACTOR
        return cls.k_spatial(L)

    @classmethod
    def k_for_portfolio_size(cls, nav: float) -> float:
        """
        Convenience method to get k from NAV.

        Args:
            nav: Net Asset Value

        Returns:
            k value
        """
        return cls.k_for_gate(nav)


# =============================================================================
# F7: PRELEC PROBABILITY WEIGHTING (BETA-ARITHMETIC)
# =============================================================================

class PrelecWeighting:
    """
    Prelec probability weighting (beta-arithmetic).

    FORMULA (F7): w(p) = exp(-(-ln(p))^alpha)

    SOURCE: Meginniss NNC paper (Paper 1 in NNC-INTEGRATION-OPPORTUNITIES.md)

    Behavior:
    - alpha < 1: Overweight small probabilities (Kahneman/Tversky finding)
    - alpha = 1: No weighting (w(p) = p)
    - alpha > 1: Underweight small probabilities

    Use Case for Trader-AI: P(ruin) adjustment, Kelly criterion modification
    """

    # Default alpha from Kahneman/Tversky experiments
    DEFAULT_ALPHA = 0.65

    @staticmethod
    def weight(p: float, alpha: float = DEFAULT_ALPHA) -> float:
        """
        Prelec probability weighting function.

        Args:
            p: Objective probability (0 < p < 1)
            alpha: Weighting parameter (default 0.65)

        Returns:
            Weighted probability w(p)
        """
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0

        # FORMULA: w(p) = exp(-(-ln(p))^alpha)
        neg_ln_p = -np.log(p)
        return float(np.exp(-np.power(neg_ln_p, alpha)))

    @staticmethod
    def inverse(w: float, alpha: float = DEFAULT_ALPHA) -> float:
        """
        Inverse Prelec transform: p = w^(-1)(w_p).

        Args:
            w: Weighted probability
            alpha: Weighting parameter

        Returns:
            Original probability p
        """
        if w <= 0:
            return 0.0
        if w >= 1:
            return 1.0

        # Inverse: p = exp(-(-ln(w))^(1/alpha))
        neg_ln_w = -np.log(w)
        return float(np.exp(-np.power(neg_ln_w, 1.0 / alpha)))

    @classmethod
    def adjust_probability(cls, p: float, alpha: float = DEFAULT_ALPHA) -> float:
        """
        Adjust a probability using Prelec weighting.
        Alias for weight() with clearer semantics.

        Args:
            p: Raw probability
            alpha: Weighting parameter

        Returns:
            Subjectively adjusted probability
        """
        return cls.weight(p, alpha)


# =============================================================================
# GEOMETRIC OPERATIONS
# =============================================================================

class GeometricOperations:
    """
    Geometric mean and related operations.

    For finance: Geometric mean is the correct average for returns
    because returns compound multiplicatively.

    SOURCE: Ozyapici NNC paper - "Geometric mean BEST for stocks"
    """

    @staticmethod
    def geometric_mean(values: List[float]) -> float:
        """
        Geometric mean with edge case handling.

        For returns: pass multipliers (1+r), not raw returns.

        Args:
            values: List of positive values (e.g., [1.05, 0.98, 1.03])

        Returns:
            Geometric mean of the values
        """
        if not values:
            return 1.0

        # Handle edge cases: replace non-positive values
        positive_values = [max(v, NUMERICAL_EPSILON) for v in values]

        # Compute in log space for numerical stability
        log_sum = sum(np.log(v) for v in positive_values)
        return float(np.exp(log_sum / len(values)))

    @staticmethod
    def geometric_mean_return(returns: List[float]) -> float:
        """
        Geometric mean return from percentage returns.

        Input: [0.05, -0.02, 0.03] (5%, -2%, 3%)
        Output: Geometric average return

        Args:
            returns: List of period returns (as decimals)

        Returns:
            Geometric mean return (as decimal)
        """
        if not returns:
            return 0.0

        multipliers = [1 + r for r in returns]
        geo_mean = GeometricOperations.geometric_mean(multipliers)
        return geo_mean - 1

    @staticmethod
    def cagr(start_value: float, end_value: float, years: float) -> float:
        """
        Compound Annual Growth Rate.

        CAGR = (end/start)^(1/years) - 1

        Args:
            start_value: Initial value
            end_value: Final value
            years: Number of years

        Returns:
            CAGR as decimal
        """
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0

        return float(np.power(end_value / start_value, 1.0 / years) - 1)


# =============================================================================
# MULTIPLICATIVE RISK
# =============================================================================

class MultiplicativeRisk:
    """
    Risk compounding using multiplicative model (delta*n NOT delta+n).

    From Filip NNC paper: Risk factors MULTIPLY, not add.
    This is critical for correct compound risk calculation.

    SOURCE: Filip, "A non-newtonian examination of the theory of exogenous
            economic growth" - amplification modeling
    """

    @staticmethod
    def compound_survival(factors: List[float]) -> float:
        """
        Compound survival probabilities multiplicatively.

        factors: [0.99, 0.95, 0.98] -> 0.99 * 0.95 * 0.98 = 0.921

        NOT: 1 - (0.01 + 0.05 + 0.02) = 0.92 (additive - WRONG for compound)

        Args:
            factors: List of survival probabilities (each in [0, 1])

        Returns:
            Compound survival probability
        """
        if not factors:
            return 1.0

        return float(np.prod(factors))

    @staticmethod
    def amplification_factor(base_risk: float, periods: int) -> float:
        """
        Risk amplifies multiplicatively: (1-base)^periods

        Example: 1% daily risk over 100 days
        Multiplicative: 0.99^100 = 0.366 (36.6% survival)
        Additive (WRONG): 1 - 100*0.01 = 0.00 (0% survival)

        Args:
            base_risk: Risk per period (as decimal, e.g., 0.01 for 1%)
            periods: Number of periods

        Returns:
            Survival probability after all periods
        """
        if base_risk < 0 or base_risk > 1:
            raise ValueError(f"base_risk must be in [0, 1], got {base_risk}")
        if periods < 0:
            raise ValueError(f"periods must be non-negative, got {periods}")

        return float((1 - base_risk) ** periods)

    @staticmethod
    def effective_daily_risk(target_survival: float, days: int) -> float:
        """
        Calculate daily risk needed to achieve target survival over N days.

        If we want 50% survival after 100 days:
        0.50 = (1 - r)^100
        r = 1 - 0.50^(1/100) = 0.0069 (0.69% daily risk)

        Args:
            target_survival: Desired survival probability (e.g., 0.5 for 50%)
            days: Number of days

        Returns:
            Required daily risk (as decimal)
        """
        if target_survival <= 0 or target_survival > 1:
            raise ValueError(f"target_survival must be in (0, 1], got {target_survival}")
        if days <= 0:
            raise ValueError(f"days must be positive, got {days}")

        return float(1 - np.power(target_survival, 1.0 / days))


# =============================================================================
# BOUNDED NNC (SAFE WRAPPER)
# =============================================================================

@dataclass
class BoundedNNCConfig:
    """Configuration for bounded NNC operations."""
    epsilon: float = NUMERICAL_EPSILON
    max_exp_arg: float = MAX_EXP_ARG
    min_value: float = 1e-300
    max_value: float = 1e300


class BoundedNNC:
    """
    Safe wrapper for NNC operations with bounds checking.

    Prevents overflow/underflow and handles edge cases gracefully.
    This class should be used for production calculations.
    """

    def __init__(self, config: Optional[BoundedNNCConfig] = None):
        """
        Initialize bounded NNC calculator.

        Args:
            config: Configuration for bounds. Defaults to safe values.
        """
        self.config = config or BoundedNNCConfig()
        self.geo_deriv = GeometricDerivative(epsilon=self.config.epsilon)
        self.bigeo_deriv = BigeometricDerivative(epsilon=self.config.epsilon)

    def safe_exp(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safe exponential with overflow protection.

        Args:
            x: Input value(s)

        Returns:
            exp(x) bounded to prevent overflow
        """
        x_clipped = np.clip(x, -self.config.max_exp_arg, self.config.max_exp_arg)
        return np.exp(x_clipped)

    def safe_log(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safe logarithm with underflow protection.

        Args:
            x: Input value(s) (will be clipped to [epsilon, inf))

        Returns:
            log(x) with protected input
        """
        x_safe = np.maximum(x, self.config.epsilon)
        return np.log(x_safe)

    def geometric_derivative(self, f_values: np.ndarray,
                             x_values: np.ndarray) -> np.ndarray:
        """
        Safe geometric derivative computation.

        Args:
            f_values: Function values
            x_values: Independent variable values

        Returns:
            Geometric derivative with bounds protection
        """
        result = self.geo_deriv(f_values, x_values)
        return np.clip(result, self.config.min_value, self.config.max_value)

    def bigeometric_derivative(self, f_values: np.ndarray,
                                x_values: np.ndarray) -> np.ndarray:
        """
        Safe bigeometric derivative computation.

        Args:
            f_values: Function values
            x_values: Independent variable values (must be positive)

        Returns:
            Bigeometric derivative with bounds protection
        """
        result = self.bigeo_deriv(f_values, x_values)
        return np.clip(result, self.config.min_value, self.config.max_value)

    def star_derivative(self, f_values: np.ndarray, x_values: np.ndarray,
                        method: str = 'geometric') -> np.ndarray:
        """
        Generic star-derivative (f* notation from NNC).

        Args:
            f_values: Function values
            x_values: Independent variable values
            method: 'geometric' or 'bigeometric'

        Returns:
            Star-derivative values
        """
        if method == 'geometric':
            return self.geometric_derivative(f_values, x_values)
        elif method == 'bigeometric':
            return self.bigeometric_derivative(f_values, x_values)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'geometric' or 'bigeometric'")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def geometric_mean(values: List[float]) -> float:
    """Shorthand for GeometricOperations.geometric_mean."""
    return GeometricOperations.geometric_mean(values)


def geometric_return(returns: List[float]) -> float:
    """Shorthand for GeometricOperations.geometric_mean_return."""
    return GeometricOperations.geometric_mean_return(returns)


def prelec_weight(p: float, alpha: float = 0.65) -> float:
    """Shorthand for PrelecWeighting.weight."""
    return PrelecWeighting.weight(p, alpha)


def k_for_capital(capital: float) -> float:
    """Shorthand for KEvolution.k_for_gate."""
    return KEvolution.k_for_gate(capital)


def compound_survival(factors: List[float]) -> float:
    """Shorthand for MultiplicativeRisk.compound_survival."""
    return MultiplicativeRisk.compound_survival(factors)


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__source__ = "NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1"
__formulas__ = [
    "F1: D_G[f](a) = exp(f'(a) / f(a))",
    "F2: D_BG[f](a) = exp(a * f'(a) / f(a))",
    "F3: elasticity = a * f'(a) / f(a)",
    "F4: n = (2/3) * (1 - k) / (1 + w)",
    "F5: m = 2 - 2k",
    "F6: k(L) = -0.0137 * log10(L) + 0.1593",
    "F7: w(p) = exp(-(-ln(p))^alpha)",
]
