"""
Daily (1-Day) Simulator

Based on Crayfer's quant contact insight:
"Firms use AI on shorter timeframes because the further you zoom out,
the more psychology affects price."

Changes from 5-day simulator:
- Horizon: 1 day (not 5)
- Rebalancing: Daily (not weekly)
- Features: Intraday patterns that resolve within a day
- Less psychology influence = more predictable patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DailySimulationResult:
    """Result of daily simulation step."""
    date: str
    spy_return: float
    tlt_return: float
    portfolio_return: float
    spy_allocation: float
    tlt_allocation: float
    cash_allocation: float
    cumulative_return: float
    drawdown: float


class DailySimulator:
    """
    1-day horizon simulator for strategy training.

    Key differences from 5-day:
    - Predicts NEXT DAY return (not week)
    - Features focus on short-term patterns
    - More samples per unit time (252/year vs ~50/year)
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            spy_returns: Array of daily SPY returns
            tlt_returns: Array of daily TLT returns
            dates: Optional array of dates
            risk_free_rate: Annual risk-free rate
        """
        self.spy_returns = np.asarray(spy_returns)
        self.tlt_returns = np.asarray(tlt_returns)
        self.dates = dates
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Track simulation state
        self.cumulative_return = 0.0
        self.peak_nav = 1.0
        self.current_nav = 1.0
        self.history: List[DailySimulationResult] = []

    def get_next_day_return(self, idx: int) -> Tuple[float, float]:
        """
        Get next day returns for SPY and TLT.

        Args:
            idx: Current time index

        Returns:
            (spy_return, tlt_return) for day idx+1
        """
        if idx + 1 >= len(self.spy_returns):
            return 0.0, 0.0

        return float(self.spy_returns[idx + 1]), float(self.tlt_returns[idx + 1])

    def simulate_day(
        self,
        idx: int,
        allocations: np.ndarray,  # (spy, tlt, cash)
    ) -> DailySimulationResult:
        """
        Simulate one day with given allocation.

        Args:
            idx: Current day index (predict idx+1)
            allocations: (3,) array of [SPY%, TLT%, Cash%]

        Returns:
            DailySimulationResult
        """
        spy_alloc = allocations[0]
        tlt_alloc = allocations[1]
        cash_alloc = allocations[2]

        # Get next day returns
        spy_ret, tlt_ret = self.get_next_day_return(idx)

        # Portfolio return
        portfolio_ret = (
            spy_alloc * spy_ret +
            tlt_alloc * tlt_ret +
            cash_alloc * self.daily_rf
        )

        # Update state
        self.current_nav *= (1 + portfolio_ret)
        self.cumulative_return = self.current_nav - 1.0
        self.peak_nav = max(self.peak_nav, self.current_nav)
        drawdown = (self.peak_nav - self.current_nav) / self.peak_nav

        date_str = str(self.dates[idx]) if self.dates is not None else f"Day_{idx}"

        result = DailySimulationResult(
            date=date_str,
            spy_return=spy_ret,
            tlt_return=tlt_ret,
            portfolio_return=portfolio_ret,
            spy_allocation=spy_alloc,
            tlt_allocation=tlt_alloc,
            cash_allocation=cash_alloc,
            cumulative_return=self.cumulative_return,
            drawdown=drawdown,
        )

        self.history.append(result)
        return result

    def compute_reward(
        self,
        portfolio_return: float,
        spy_return: float,
        drawdown: float,
        capital_norm: float = 0.0,  # 0 = low capital, 1 = high capital
    ) -> float:
        """
        Compute reward for training.

        Incorporates:
        - Return (positive reward for gains)
        - Risk-adjustment (penalize relative to SPY when underperforming)
        - Drawdown penalty (especially at high capital)
        - Loss aversion (asymmetric penalty for losses)
        """
        # Base reward from return
        reward = portfolio_return * 10  # Scale up

        # Alpha: did we beat SPY?
        alpha = portfolio_return - spy_return
        if alpha > 0:
            reward += alpha * 5  # Bonus for outperformance
        else:
            reward += alpha * 2  # Smaller penalty for underperformance

        # Drawdown penalty (stronger at high capital)
        if drawdown > 0.02:  # More than 2% drawdown
            dd_penalty = -drawdown * (1 + capital_norm * 2)  # 1x to 3x penalty
            reward += dd_penalty

        # Loss aversion (asymmetric)
        if portfolio_return < 0:
            loss_multiplier = 1.5 + capital_norm * 1.0  # 1.5x to 2.5x
            reward *= loss_multiplier

        return reward

    def reset(self):
        """Reset simulator state."""
        self.cumulative_return = 0.0
        self.peak_nav = 1.0
        self.current_nav = 1.0
        self.history = []

    def get_performance_stats(self) -> Dict:
        """Get performance statistics from simulation history."""
        if not self.history:
            return {}

        returns = [r.portfolio_return for r in self.history]
        spy_returns = [r.spy_return for r in self.history]

        total_return = self.cumulative_return
        annualized = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = (annualized - self.risk_free_rate) / volatility if volatility > 0 else 0
        max_dd = max(r.drawdown for r in self.history)

        # Compare to buy-and-hold SPY
        spy_total = np.prod([1 + r for r in spy_returns]) - 1
        alpha = total_return - spy_total

        return {
            'total_return': total_return,
            'annualized_return': annualized,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_days': len(self.history),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            'spy_total': spy_total,
            'alpha': alpha,
        }


class BatchDailySimulator:
    """
    Batch simulator for efficient training.

    Simulates multiple trajectories in parallel using vectorized operations.
    """

    def __init__(
        self,
        spy_returns: np.ndarray,
        tlt_returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ):
        self.spy_returns = np.asarray(spy_returns)
        self.tlt_returns = np.asarray(tlt_returns)
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.n_days = len(spy_returns)

    def simulate_batch(
        self,
        time_indices: np.ndarray,  # (batch_size,) indices into return arrays
        allocations: np.ndarray,   # (batch_size, 3) allocations
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate batch of single-day outcomes.

        Args:
            time_indices: Starting day indices
            allocations: [SPY, TLT, Cash] allocations per sample

        Returns:
            portfolio_returns: (batch_size,)
            spy_returns: (batch_size,)
            tlt_returns: (batch_size,)
        """
        batch_size = len(time_indices)

        # Get next day returns (vectorized)
        next_indices = np.clip(time_indices + 1, 0, self.n_days - 1)
        spy_ret = self.spy_returns[next_indices]
        tlt_ret = self.tlt_returns[next_indices]

        # Portfolio returns
        portfolio_ret = (
            allocations[:, 0] * spy_ret +
            allocations[:, 1] * tlt_ret +
            allocations[:, 2] * self.daily_rf
        )

        return portfolio_ret, spy_ret, tlt_ret

    def compute_batch_reward(
        self,
        portfolio_returns: np.ndarray,
        spy_returns: np.ndarray,
        capital_norms: np.ndarray,  # (batch_size,) capital normalization 0-1
        drawdowns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute rewards for batch.

        Args:
            portfolio_returns: (batch_size,) portfolio returns
            spy_returns: (batch_size,) SPY returns
            capital_norms: (batch_size,) capital level (0=low, 1=high)
            drawdowns: Optional (batch_size,) current drawdowns

        Returns:
            rewards: (batch_size,)
        """
        # Base reward from return
        rewards = portfolio_returns * 10

        # Alpha bonus/penalty
        alpha = portfolio_returns - spy_returns
        alpha_reward = np.where(alpha > 0, alpha * 5, alpha * 2)
        rewards += alpha_reward

        # Drawdown penalty (if provided)
        if drawdowns is not None:
            dd_mask = drawdowns > 0.02
            dd_penalty = -drawdowns * (1 + capital_norms * 2)
            rewards = np.where(dd_mask, rewards + dd_penalty, rewards)

        # Loss aversion
        loss_mask = portfolio_returns < 0
        loss_multiplier = 1.5 + capital_norms * 1.0
        rewards = np.where(loss_mask, rewards * loss_multiplier, rewards)

        return rewards


class DailyDataLoader:
    """
    Load and prepare daily data for training.

    Focuses on 1-day features that are most predictive for next-day returns.
    """

    def __init__(
        self,
        spy_df: pd.DataFrame,
        tlt_df: pd.DataFrame,
        vix_df: pd.DataFrame,
    ):
        """
        Args:
            spy_df: DataFrame with OHLCV for SPY
            tlt_df: DataFrame with OHLCV for TLT
            vix_df: DataFrame with VIX data
        """
        self.spy = spy_df
        self.tlt = tlt_df
        self.vix = vix_df

        # Compute returns
        self.spy_returns = self.spy['Close'].pct_change().values
        self.tlt_returns = self.tlt['Close'].pct_change().values

    def compute_daily_features(self, idx: int) -> np.ndarray:
        """
        Compute features for predicting next day.

        Focuses on short-term patterns that resolve within a day.

        Args:
            idx: Current day index

        Returns:
            Feature array
        """
        features = []

        # === Price Action Features ===
        spy_close = self.spy['Close'].iloc[idx]
        spy_open = self.spy['Open'].iloc[idx]
        spy_high = self.spy['High'].iloc[idx]
        spy_low = self.spy['Low'].iloc[idx]

        # Intraday range and position
        features.append(spy_close / spy_open - 1)  # Daily return (open to close)
        features.append((spy_high - spy_low) / spy_open)  # Daily range
        features.append((spy_close - spy_low) / (spy_high - spy_low + 1e-8))  # Close position in range

        # === Short-term Momentum (1-5 days) ===
        for period in [1, 2, 3, 5]:
            if idx >= period:
                ret = self.spy_returns[idx - period + 1:idx + 1].sum()
            else:
                ret = 0.0
            features.append(ret)

        # === Volume Features ===
        vol_ratio = self.spy['Volume'].iloc[idx] / self.spy['Volume'].iloc[max(0, idx-5):idx].mean()
        features.append(vol_ratio if not np.isnan(vol_ratio) else 1.0)

        # === Short-term Volatility ===
        if idx >= 5:
            vol_5d = np.std(self.spy_returns[idx-4:idx+1]) * np.sqrt(252)
        else:
            vol_5d = 0.15
        features.append(vol_5d)

        # === VIX Features ===
        vix_level = self.vix['Close'].iloc[idx] / 100
        features.append(vix_level)

        if idx >= 5:
            vix_change = self.vix['Close'].iloc[idx] / self.vix['Close'].iloc[idx-5] - 1
        else:
            vix_change = 0.0
        features.append(vix_change)

        # === TLT Features ===
        features.append(self.tlt_returns[idx] if idx > 0 else 0)  # TLT daily return

        if idx >= 5:
            tlt_5d = self.tlt_returns[idx-4:idx+1].sum()
        else:
            tlt_5d = 0.0
        features.append(tlt_5d)

        # === Correlation (recent) ===
        if idx >= 10:
            spy_10 = self.spy_returns[idx-9:idx+1]
            tlt_10 = self.tlt_returns[idx-9:idx+1]
            corr = np.corrcoef(spy_10, tlt_10)[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0.0)

        # === Day of Week (0-4) ===
        if hasattr(self.spy.index, 'dayofweek'):
            dow = self.spy.index[idx].dayofweek / 4
        else:
            dow = 0.5
        features.append(dow)

        # === Gap (overnight) ===
        if idx > 0:
            gap = self.spy['Open'].iloc[idx] / self.spy['Close'].iloc[idx-1] - 1
        else:
            gap = 0.0
        features.append(gap)

        return np.array(features, dtype=np.float32)

    def get_batch_features(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get features for batch of indices."""
        return np.stack([self.compute_daily_features(i) for i in indices])

    def get_batch_targets(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get next-day returns for batch of indices."""
        next_indices = np.clip(indices + 1, 0, len(self.spy_returns) - 1)
        return self.spy_returns[next_indices]


def demo_daily_simulator():
    """Demo the daily simulator."""
    print("Daily Simulator Demo")
    print("=" * 50)

    # Generate synthetic data
    n_days = 252
    np.random.seed(42)
    spy_returns = np.random.normal(0.0004, 0.01, n_days)  # ~10% annual, ~16% vol
    tlt_returns = np.random.normal(0.0001, 0.008, n_days)  # ~2.5% annual, ~13% vol

    sim = DailySimulator(spy_returns, tlt_returns)

    # Simulate with simple strategy
    print("\nSimulating 60/40 portfolio...")
    allocations = np.array([0.60, 0.40, 0.00])

    for i in range(n_days - 1):
        sim.simulate_day(i, allocations)

    stats = sim.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total Return: {stats['total_return']:.2%}")
    print(f"  Annualized: {stats['annualized_return']:.2%}")
    print(f"  Volatility: {stats['volatility']:.2%}")
    print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Alpha vs SPY: {stats['alpha']:.2%}")

    # Test batch simulator
    print("\n\nBatch Simulator Test...")
    batch_sim = BatchDailySimulator(spy_returns, tlt_returns)

    batch_indices = np.array([10, 50, 100, 150, 200])
    batch_allocs = np.array([
        [0.90, 0.10, 0.00],  # Aggressive
        [0.70, 0.20, 0.10],  # Balanced
        [0.60, 0.40, 0.00],  # 60/40
        [0.40, 0.30, 0.30],  # Defensive
        [0.80, 0.15, 0.05],  # Growth
    ])
    capital_norms = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    port_ret, spy_ret, tlt_ret = batch_sim.simulate_batch(batch_indices, batch_allocs)
    rewards = batch_sim.compute_batch_reward(port_ret, spy_ret, capital_norms)

    print(f"\nBatch Results:")
    for i, (alloc, p_ret, reward) in enumerate(zip(batch_allocs, port_ret, rewards)):
        print(f"  Alloc {alloc} -> Return: {p_ret:.4f}, Reward: {reward:.4f}")


if __name__ == '__main__':
    demo_daily_simulator()
