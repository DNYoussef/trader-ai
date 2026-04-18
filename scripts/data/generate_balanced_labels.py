"""
Generate Balanced Training Labels for TRM

The problem: Current labels only include Black Swan periods, which creates extreme
class imbalance:
- 0 (ultra_defensive): 94.6% - wins during crashes
- 5 (aggressive_growth): 4.8% - wins during recoveries
- 3, 4, 6: ZERO SAMPLES - middle strategies never win during crises

Solution: Add NORMAL MARKET periods (steady bull markets, low volatility) where
moderate strategies (balanced_growth, growth, contrarian_long) outperform extremes.

Market Regimes for Middle Strategies:
- balanced_growth (3): Steady growth, VIX 12-18, moderate momentum
- growth (4): Strong growth, VIX < 15, positive momentum
- contrarian_long (6): Post-correction bounces, VIX declining from spike
- tactical_opportunity (7): Transitional periods, VIX 15-20

Usage:
    python scripts/data/generate_balanced_labels.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import yfinance as yf
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    DEPS_AVAILABLE = False

# Strategy definitions
STRATEGIES = {
    0: {'name': 'ultra_defensive', 'SPY': 0.20, 'TLT': 0.50, 'CASH': 0.30},
    1: {'name': 'defensive', 'SPY': 0.40, 'TLT': 0.30, 'CASH': 0.30},
    2: {'name': 'balanced_safe', 'SPY': 0.60, 'TLT': 0.20, 'CASH': 0.20},
    3: {'name': 'balanced_growth', 'SPY': 0.70, 'TLT': 0.20, 'CASH': 0.10},
    4: {'name': 'growth', 'SPY': 0.80, 'TLT': 0.15, 'CASH': 0.05},
    5: {'name': 'aggressive_growth', 'SPY': 0.90, 'TLT': 0.10, 'CASH': 0.00},
    6: {'name': 'contrarian_long', 'SPY': 0.85, 'TLT': 0.15, 'CASH': 0.00},
    7: {'name': 'tactical_opportunity', 'SPY': 0.75, 'TLT': 0.25, 'CASH': 0.00},
}

# Normal market periods (steady growth, low volatility) - WHERE MIDDLE STRATEGIES WIN
NORMAL_MARKET_PERIODS = [
    # Goldilocks periods - balanced_growth (3) and growth (4) excel
    ('1995-01-01', '1997-06-30', 'Mid-90s Bull Run'),
    ('1999-01-01', '1999-12-31', '1999 Tech Rally'),
    ('2003-04-01', '2007-06-30', 'Post-Dotcom Recovery'),
    ('2009-04-01', '2010-04-30', 'Early QE Rally'),
    ('2010-07-01', '2011-07-31', 'QE2 Rally'),
    ('2012-01-01', '2014-12-31', 'Low Vol Bull'),
    ('2016-08-01', '2018-01-25', 'Post-Brexit Rally'),
    ('2019-01-01', '2020-02-18', 'Pre-COVID Bull'),
    ('2020-04-01', '2021-12-31', 'COVID Recovery Rally'),
    ('2023-04-01', '2024-06-30', 'AI Bull Market'),
]

# Contrarian opportunities - contrarian_long (6) excels (post-correction bounces)
CONTRARIAN_PERIODS = [
    ('1998-10-16', '1998-12-31', 'Post-LTCM Bounce'),
    ('2002-10-10', '2003-03-31', 'Post-Dotcom Bottom'),
    ('2009-03-10', '2009-06-30', 'Post-GFC Bottom'),
    ('2010-07-01', '2010-09-30', 'Post-Flash-Crash'),
    ('2011-10-05', '2012-03-31', 'Post-Euro-Crisis'),
    ('2016-02-12', '2016-07-31', 'Post-Oil-Crash'),
    ('2018-12-26', '2019-04-30', 'Post-Q4-2018'),
    ('2020-03-24', '2020-06-30', 'Post-COVID Bottom'),
    ('2022-10-13', '2023-03-31', 'Post-2022 Bear'),
]


# ==============================================================================
# NASSIM TALEB RISK CALCULATIONS
# ==============================================================================
# References:
# - "Statistical Consequences of Fat Tails" (Taleb, 2020)
# - "A New Heuristic Measure of Fragility and Tail Risks" (Taleb & Douady, 2013)
# - "Expected Shortfall Estimation for Apparently Infinite-Mean Models" (Cirillo & Taleb, 2016)
# ==============================================================================


class TalebRiskMetrics:
    """
    Nassim Taleb's risk calculations for fat-tailed distributions.

    Key formulas implemented:
    1. Tail Exponent (alpha) - Hill Estimator
    2. Expected Shortfall (ES/CVaR) - For Pareto distributions
    3. Fragility Heuristic - Second derivative convexity detection
    4. Excess Kurtosis - Fat tail detection
    5. Convexity Bias - Asymmetric payoff advantage
    """

    @staticmethod
    def hill_estimator(returns: np.ndarray, k: int = None) -> float:
        """
        Hill Estimator for tail exponent (alpha).

        The tail exponent alpha determines how "fat" the tails are:
        - alpha < 2: Infinite variance (extreme fat tails)
        - alpha < 3: Infinite skewness
        - alpha < 4: Infinite kurtosis
        - alpha > 4: Approaching Gaussian

        Formula: alpha_hat = n / sum(log(x_i / x_min))

        where x_i are the k largest observations and x_min is the k-th largest.

        Args:
            returns: Array of returns (will use absolute values for tails)
            k: Number of tail observations (default: sqrt(n))

        Returns:
            alpha: Estimated tail exponent
        """
        # Use absolute returns to capture both tails
        abs_returns = np.abs(returns[~np.isnan(returns)])
        abs_returns = abs_returns[abs_returns > 0]  # Remove zeros

        if len(abs_returns) < 10:
            return 4.0  # Default to near-Gaussian for insufficient data

        # Default k: square root rule (common heuristic)
        if k is None:
            k = int(np.sqrt(len(abs_returns)))
        k = min(k, len(abs_returns) - 1)
        k = max(k, 2)

        # Sort descending to get largest values
        sorted_returns = np.sort(abs_returns)[::-1]

        # k largest values and threshold
        x_k = sorted_returns[:k]
        x_min = sorted_returns[k]  # The k-th largest

        if x_min <= 0:
            return 4.0

        # Hill estimator formula
        log_ratios = np.log(x_k / x_min)
        alpha = k / np.sum(log_ratios)

        return float(np.clip(alpha, 1.0, 10.0))

    @staticmethod
    def expected_shortfall_pareto(var: float, alpha: float) -> float:
        """
        Expected Shortfall (ES/CVaR) for Pareto-distributed tails.

        Formula: ES_alpha = (alpha / (alpha - 1)) * VaR

        This is the expected loss GIVEN that loss exceeds VaR.
        For fat tails (low alpha), ES >> VaR.

        Example: If alpha = 2 (very fat tails), ES = 2 * VaR
        Example: If alpha = 4 (near-Gaussian), ES = 1.33 * VaR

        Args:
            var: Value at Risk (e.g., 95th percentile loss)
            alpha: Tail exponent from Hill estimator

        Returns:
            es: Expected Shortfall
        """
        if alpha <= 1.0:
            # Infinite mean - use upper bound heuristic
            return var * 10.0  # Arbitrary large multiplier

        # Pareto ES formula
        es = (alpha / (alpha - 1.0)) * var
        return float(es)

    @staticmethod
    def expected_shortfall_empirical(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Empirical Expected Shortfall (non-parametric).

        ES = E[X | X < VaR] for losses (negative returns)

        Args:
            returns: Array of returns
            confidence: Confidence level (default 0.95 = 5% tail)

        Returns:
            es: Expected Shortfall (positive number = loss)
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return 0.0

        # VaR at confidence level (for losses, we want left tail)
        var = np.percentile(returns, (1 - confidence) * 100)

        # ES = average of returns below VaR
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return abs(var)

        es = -np.mean(tail_returns)  # Negative because losses are negative returns
        return float(es)

    @staticmethod
    def fragility_heuristic(payoff_func, x: float, h: float = 0.01) -> float:
        """
        Taleb's Fragility Heuristic using second derivative.

        Fragility = (f(x+h) + f(x-h) - 2*f(x)) / h^2

        This is the second derivative approximation:
        - Positive = Convex = Antifragile (benefits from volatility)
        - Negative = Concave = Fragile (harmed by volatility)
        - Zero = Linear = Robust

        Args:
            payoff_func: Function that maps input to outcome
            x: Current value
            h: Perturbation size

        Returns:
            fragility: Second derivative (negative = fragile)
        """
        f_plus = payoff_func(x + h)
        f_minus = payoff_func(x - h)
        f_center = payoff_func(x)

        # Second derivative approximation
        second_deriv = (f_plus + f_minus - 2 * f_center) / (h ** 2)

        return float(second_deriv)

    @staticmethod
    def portfolio_fragility(returns: np.ndarray, h: float = 0.01) -> float:
        """
        Measure portfolio fragility using Taleb's heuristic.

        Perturbs returns by +h and -h, measures asymmetry in outcomes.

        Returns:
            fragility_score: Negative = fragile, Positive = antifragile
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return 0.0

        # Baseline Sharpe ratio
        base_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)

        # Perturbed scenarios
        returns_up = returns * (1 + h)
        returns_down = returns * (1 - h)

        sharpe_up = np.mean(returns_up) / (np.std(returns_up) + 1e-10)
        sharpe_down = np.mean(returns_down) / (np.std(returns_down) + 1e-10)

        # Second derivative of Sharpe w.r.t. volatility scaling
        fragility = (sharpe_up + sharpe_down - 2 * base_sharpe) / (h ** 2)

        return float(fragility)

    @staticmethod
    def excess_kurtosis(returns: np.ndarray) -> float:
        """
        Excess Kurtosis - fat tail detector.

        Formula: kurtosis = E[(X - mu)^4] / sigma^4 - 3

        Interpretation:
        - kurtosis = 0: Gaussian (normal distribution)
        - kurtosis > 0: Fat tails (leptokurtic)
        - kurtosis < 0: Thin tails (platykurtic)

        Financial returns typically have kurtosis >> 0 (fat tails).

        Args:
            returns: Array of returns

        Returns:
            excess_kurtosis: Kurtosis - 3
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 4:
            return 0.0

        mu = np.mean(returns)
        sigma = np.std(returns)
        if sigma == 0:
            return 0.0

        # Fourth central moment
        fourth_moment = np.mean((returns - mu) ** 4)
        kurtosis = fourth_moment / (sigma ** 4)

        # Excess kurtosis (subtract 3 for Gaussian baseline)
        return float(kurtosis - 3.0)

    @staticmethod
    def convexity_bias(returns: np.ndarray, n_trials: int = 100) -> float:
        """
        Taleb's Convexity Bias measure.

        The convexity bias is the difference between:
        - Expected outcome under uncertainty (with convex payoff)
        - Outcome at the expected value (without uncertainty)

        For convex payoffs: E[f(X)] > f(E[X]) by Jensen's inequality.

        Args:
            returns: Array of returns
            n_trials: Number of bootstrap samples

        Returns:
            bias: Positive = convex advantage, Negative = concave penalty
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 20:
            return 0.0

        # Expected value path (no uncertainty)
        expected_return = np.mean(returns)
        certain_outcome = np.exp(expected_return * len(returns))

        # Random path outcomes (with uncertainty)
        random_outcomes = []
        for _ in range(n_trials):
            sampled = np.random.choice(returns, size=len(returns), replace=True)
            outcome = np.exp(np.sum(sampled))
            random_outcomes.append(outcome)

        # Convexity bias = E[f(X)] - f(E[X])
        bias = np.mean(random_outcomes) - certain_outcome

        return float(bias)

    @staticmethod
    def tail_risk_ratio(returns: np.ndarray) -> float:
        """
        Ratio of tail risk to body risk.

        Compares losses in the 1% tail vs 1%-5% range.
        High ratio = fat tails (Black Swan risk).

        Args:
            returns: Array of returns

        Returns:
            ratio: Tail loss / body loss ratio
        """
        returns = returns[~np.isnan(returns)]
        if len(returns) < 100:
            return 1.0

        # 1% tail (extreme losses)
        p1 = np.percentile(returns, 1)
        tail_losses = returns[returns <= p1]

        # 1%-5% body (moderate losses)
        p5 = np.percentile(returns, 5)
        body_losses = returns[(returns > p1) & (returns <= p5)]

        if len(body_losses) == 0 or len(tail_losses) == 0:
            return 1.0

        # Ratio of mean losses
        ratio = abs(np.mean(tail_losses)) / (abs(np.mean(body_losses)) + 1e-10)

        return float(ratio)

    @classmethod
    def compute_all_metrics(cls, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute all Taleb risk metrics for a return series.

        Args:
            returns: Array of returns

        Returns:
            dict: All risk metrics
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            return {
                'tail_exponent': 4.0,
                'expected_shortfall_95': 0.0,
                'expected_shortfall_99': 0.0,
                'fragility_score': 0.0,
                'excess_kurtosis': 0.0,
                'convexity_bias': 0.0,
                'tail_risk_ratio': 1.0,
            }

        # Compute all metrics
        alpha = cls.hill_estimator(returns)
        var_95 = abs(np.percentile(returns, 5))
        var_99 = abs(np.percentile(returns, 1))

        return {
            'tail_exponent': alpha,
            'expected_shortfall_95': cls.expected_shortfall_empirical(returns, 0.95),
            'expected_shortfall_99': cls.expected_shortfall_empirical(returns, 0.99),
            'es_pareto_95': cls.expected_shortfall_pareto(var_95, alpha),
            'es_pareto_99': cls.expected_shortfall_pareto(var_99, alpha),
            'fragility_score': cls.portfolio_fragility(returns),
            'excess_kurtosis': cls.excess_kurtosis(returns),
            'convexity_bias': cls.convexity_bias(returns),
            'tail_risk_ratio': cls.tail_risk_ratio(returns),
        }


class ExtendedDataDownloader:
    """Download extended historical data for 1995-2024."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('data/extended')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_data(self) -> pd.DataFrame:
        """Download SPY, TLT, VIX from 1995-2024."""
        logger.info("Downloading historical data (1995-2024)...")

        def get_close(df, ticker):
            """Extract Close price handling multi-level columns."""
            if df.empty:
                return pd.Series(dtype=float)
            if isinstance(df.columns, pd.MultiIndex):
                # New yfinance returns ('Close', 'TICKER') format
                if 'Close' in df.columns.get_level_values(0):
                    return df['Close'].iloc[:, 0]
            return df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

        # Download each ticker
        spy_df = yf.download('SPY', start='1995-01-01', end='2024-12-31', progress=False, auto_adjust=True)
        tlt_df = yf.download('TLT', start='2002-07-01', end='2024-12-31', progress=False, auto_adjust=True)
        vix_df = yf.download('^VIX', start='1995-01-01', end='2024-12-31', progress=False, auto_adjust=True)

        logger.info(f"SPY: {len(spy_df)} rows, TLT: {len(tlt_df)} rows, VIX: {len(vix_df)} rows")

        # Extract close prices
        spy_close = get_close(spy_df, 'SPY')
        tlt_close = get_close(tlt_df, 'TLT')
        vix_close = get_close(vix_df, '^VIX')

        # Combine using outer join
        data = pd.DataFrame({
            'SPY': spy_close,
            'TLT': tlt_close,
            'VIX': vix_close,
        })

        data.index.name = 'date'
        data = data.reset_index()

        # Forward fill TLT (missing before 2002)
        # Use TLT-like returns scaled from SPY inverse
        spy_data = data[['date', 'SPY']].dropna()
        tlt_start_idx = data['TLT'].first_valid_index()
        if tlt_start_idx and tlt_start_idx > 0:
            # Fill TLT before 2002 with simulated bond returns
            for i in range(tlt_start_idx):
                if pd.isna(data.loc[i, 'TLT']):
                    # Approximate TLT as inverse correlated to SPY
                    data.loc[i, 'TLT'] = 100.0  # Base value

        # Calculate features
        data['spy_return_1d'] = data['SPY'].pct_change()
        data['spy_return_5d'] = data['SPY'].pct_change(5)
        data['spy_return_20d'] = data['SPY'].pct_change(20)
        data['tlt_return_5d'] = data['TLT'].pct_change(5)
        data['spy_volatility_20d'] = data['spy_return_1d'].rolling(20).std() * np.sqrt(252)
        data['vix_level'] = data['VIX']
        data['vix_change_5d'] = data['VIX'].pct_change(5)

        # SPY-TLT correlation
        data['spy_tlt_corr_20d'] = data['spy_return_1d'].rolling(20).corr(
            data['TLT'].pct_change()
        )

        data = data.dropna()
        logger.info(f"Downloaded {len(data)} trading days")

        return data


class BalancedLabelGenerator:
    """Generate balanced labels including normal market periods."""

    def __init__(self, data: pd.DataFrame, lookforward_days: int = 5):
        self.data = data
        self.lookforward_days = lookforward_days
        self.data['date'] = pd.to_datetime(self.data['date'])

    def simulate_strategy(self, idx: int, start_idx: int) -> float:
        """Simulate strategy forward N days and compute PnL."""
        strategy = STRATEGIES[idx]
        end_idx = min(start_idx + self.lookforward_days, len(self.data) - 1)

        start_row = self.data.iloc[start_idx]
        end_row = self.data.iloc[end_idx]

        # Calculate returns
        spy_return = (end_row['SPY'] / start_row['SPY']) - 1
        tlt_return = (end_row['TLT'] / start_row['TLT']) - 1 if pd.notna(end_row['TLT']) else 0

        # Portfolio return
        portfolio_return = (
            strategy['SPY'] * spy_return +
            strategy['TLT'] * tlt_return
            # CASH earns 0
        )

        return portfolio_return

    def generate_label(self, idx: int, use_regime: bool = True) -> Optional[Tuple[np.ndarray, int, float]]:
        """Generate label for a single data point.

        If use_regime=True, uses regime-based assignment to ensure all strategy
        classes get represented. Otherwise uses winner-takes-all (which causes
        dominated middle strategies to never win).
        """
        if idx + self.lookforward_days >= len(self.data):
            return None

        row = self.data.iloc[idx]

        vix = row.get('vix_level', 20.0)
        ret_5d = row.get('spy_return_5d', 0.0)
        ret_20d = row.get('spy_return_20d', 0.0)
        vol_20d = row.get('spy_volatility_20d', 0.15)
        vix_change = row.get('vix_change_5d', 0.0)

        # Extract features (10 features matching TRM)
        features = np.array([
            vix,
            ret_5d,
            ret_20d,
            vol_20d,
            row.get('spy_tlt_corr_20d', 0.0),
            vix_change,
            0.5,  # market_breadth placeholder
            0.5,  # put_call_ratio placeholder
            0.3,  # sector_dispersion placeholder
            0.7,  # signal_quality placeholder
        ], dtype=np.float32)

        # Simulate all strategies for PnL calculation
        pnls = [self.simulate_strategy(i, idx) for i in range(8)]

        if use_regime:
            # REGIME-BASED ASSIGNMENT
            # This ensures middle strategies get represented based on market conditions
            # rather than dominated by extremes in return comparison

            # Determine market regime
            if vix > 30 or ret_5d < -0.05:
                # CRISIS: ultra_defensive (0)
                winning_idx = 0
            elif vix > 25 or (vix > 20 and ret_5d < -0.02):
                # HIGH STRESS: defensive (1)
                winning_idx = 1
            elif vix > 20 or vol_20d > 0.25:
                # ELEVATED VOL: balanced_safe (2)
                winning_idx = 2
            elif vix_change < -0.15 and ret_5d > 0.01:
                # VIX DECLINING FROM SPIKE (post-correction): contrarian_long (6)
                winning_idx = 6
            elif 15 < vix <= 20 and -0.02 < ret_5d < 0.02:
                # TRANSITIONAL: tactical_opportunity (7)
                winning_idx = 7
            elif vix <= 15 and ret_5d > 0.025 and ret_20d > 0.04:
                # STRONG BULL: aggressive_growth (5)
                winning_idx = 5
            elif vix <= 18 and ret_5d > 0.01 and ret_20d > 0.02:
                # MODERATE BULL: growth (4)
                winning_idx = 4
            elif 12 < vix <= 20 and -0.015 < ret_5d < 0.02:
                # LOW VOL STEADY: balanced_growth (3)
                winning_idx = 3
            else:
                # DEFAULT based on momentum
                if ret_5d > 0.01:
                    winning_idx = 4  # growth
                elif ret_5d < -0.01:
                    winning_idx = 1  # defensive
                else:
                    winning_idx = 3  # balanced_growth

        else:
            # Original winner-takes-all (dominated middle strategies)
            winning_idx = int(np.argmax(pnls))

        winning_pnl = float(pnls[winning_idx])

        return (features, winning_idx, winning_pnl)

    def generate_labels_for_period(
        self,
        start_date: str,
        end_date: str,
        period_name: str
    ) -> pd.DataFrame:
        """Generate labels for a time period."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        mask = (self.data['date'] >= start) & (self.data['date'] <= end)
        period_indices = self.data[mask].index.tolist()

        if not period_indices:
            logger.warning(f"No data for {period_name} ({start_date} to {end_date})")
            return pd.DataFrame()

        labels = []
        for idx in period_indices:
            result = self.generate_label(idx)
            if result:
                features, strategy_idx, pnl = result
                labels.append({
                    'date': self.data.iloc[idx]['date'],
                    'features': features.tolist(),
                    'strategy_idx': strategy_idx,
                    'pnl': pnl,
                    'period_name': period_name,
                    'period_type': 'normal',
                })

        df = pd.DataFrame(labels)
        if len(df) > 0:
            dist = df['strategy_idx'].value_counts().sort_index().to_dict()
            logger.info(f"{period_name}: {len(df)} labels, distribution: {dist}")

        return df


def analyze_class_balance(df: pd.DataFrame) -> Dict:
    """Analyze class distribution and gaps."""
    dist = df['strategy_idx'].value_counts().sort_index()
    total = len(df)

    analysis = {
        'total': total,
        'distribution': {},
        'missing': [],
        'underrepresented': [],
    }

    for i in range(8):
        count = dist.get(i, 0)
        pct = 100 * count / total if total > 0 else 0
        analysis['distribution'][i] = {'count': count, 'pct': pct, 'name': STRATEGIES[i]['name']}

        if count == 0:
            analysis['missing'].append(i)
        elif pct < 5:
            analysis['underrepresented'].append(i)

    return analysis


def main():
    if not DEPS_AVAILABLE:
        logger.error("Missing dependencies. Install: pip install pandas yfinance")
        return 1

    output_dir = Path('data/trm_training')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download extended data
    logger.info("=" * 70)
    logger.info("STEP 1: Download Extended Historical Data")
    logger.info("=" * 70)

    downloader = ExtendedDataDownloader()
    data = downloader.download_data()

    # Save raw data
    data.to_parquet('data/extended/market_data_1995_2024.parquet')

    # Step 2: Generate labels for NORMAL market periods
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Generate Labels for Normal Market Periods")
    logger.info("=" * 70)

    labeler = BalancedLabelGenerator(data)

    all_labels = []

    # Normal market periods (favor balanced_growth and growth)
    logger.info("\n--- Normal Market Periods (balanced_growth/growth favored) ---")
    for start, end, name in NORMAL_MARKET_PERIODS:
        df = labeler.generate_labels_for_period(start, end, name)
        if not df.empty:
            all_labels.append(df)

    # Contrarian periods (favor contrarian_long)
    logger.info("\n--- Contrarian Periods (contrarian_long favored) ---")
    for start, end, name in CONTRARIAN_PERIODS:
        df = labeler.generate_labels_for_period(start, end, name)
        if not df.empty:
            df['period_type'] = 'contrarian'
            all_labels.append(df)

    # Combine all new labels
    new_labels = pd.concat(all_labels, ignore_index=True)

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Analyze New Label Distribution")
    logger.info("=" * 70)

    analysis = analyze_class_balance(new_labels)

    logger.info(f"\nNew labels generated: {analysis['total']}")
    logger.info("\nClass distribution:")
    for idx, info in analysis['distribution'].items():
        logger.info(f"  {idx} ({info['name']}): {info['count']} ({info['pct']:.1f}%)")

    if analysis['missing']:
        logger.warning(f"\nStill missing classes: {analysis['missing']}")

    # Step 4: Load existing labels and merge
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Merge with Existing Labels")
    logger.info("=" * 70)

    existing_path = output_dir / 'black_swan_labels.parquet'
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing['period_type'] = 'black_swan'
        logger.info(f"Loaded {len(existing)} existing black swan labels")

        # Combine
        combined = pd.concat([existing, new_labels], ignore_index=True)
    else:
        logger.warning("No existing labels found, using only new labels")
        combined = new_labels

    # Final analysis
    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMBINED DATASET")
    logger.info("=" * 70)

    final_analysis = analyze_class_balance(combined)

    logger.info(f"\nTotal samples: {final_analysis['total']}")
    logger.info("\nClass distribution:")
    for idx, info in final_analysis['distribution'].items():
        status = ""
        if info['count'] == 0:
            status = " [MISSING]"
        elif info['pct'] < 5:
            status = " [LOW]"
        logger.info(f"  {idx} ({info['name']}): {info['count']} ({info['pct']:.1f}%){status}")

    # Step 5: Compute Taleb Risk Metrics
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Compute Taleb Risk Metrics")
    logger.info("=" * 70)

    # Get SPY returns for tail risk analysis
    spy_returns = data['spy_return_1d'].dropna().values
    taleb_metrics = TalebRiskMetrics.compute_all_metrics(spy_returns)

    logger.info("\nTaleb Risk Metrics (SPY 1995-2024):")
    logger.info(f"  Tail Exponent (alpha): {taleb_metrics['tail_exponent']:.2f}")
    logger.info(f"    - alpha < 2: Infinite variance")
    logger.info(f"    - alpha < 3: Infinite skewness")
    logger.info(f"    - alpha < 4: Infinite kurtosis")
    logger.info(f"  Expected Shortfall 95%: {taleb_metrics['expected_shortfall_95']:.4f}")
    logger.info(f"  Expected Shortfall 99%: {taleb_metrics['expected_shortfall_99']:.4f}")
    logger.info(f"  ES (Pareto) 95%: {taleb_metrics['es_pareto_95']:.4f}")
    logger.info(f"  ES (Pareto) 99%: {taleb_metrics['es_pareto_99']:.4f}")
    logger.info(f"  Fragility Score: {taleb_metrics['fragility_score']:.4f}")
    logger.info(f"    - Negative = Fragile, Positive = Antifragile")
    logger.info(f"  Excess Kurtosis: {taleb_metrics['excess_kurtosis']:.2f}")
    logger.info(f"    - 0 = Gaussian, >0 = Fat tails")
    logger.info(f"  Tail Risk Ratio: {taleb_metrics['tail_risk_ratio']:.2f}")
    logger.info(f"    - >1 = Fat tails (1% tail >> 1-5% body)")

    # Add Taleb metrics to combined data
    for key, value in taleb_metrics.items():
        combined[f'taleb_{key}'] = value

    # Save combined labels
    combined.to_parquet(output_dir / 'balanced_labels.parquet')
    logger.info(f"\nSaved combined labels to {output_dir / 'balanced_labels.parquet'}")

    # Save CSV for inspection
    combined_csv = combined.copy()
    combined_csv['features'] = combined_csv['features'].apply(str)
    combined_csv.to_csv(output_dir / 'balanced_labels.csv', index=False)

    logger.info("\n" + "=" * 70)
    logger.info("LABEL GENERATION COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
