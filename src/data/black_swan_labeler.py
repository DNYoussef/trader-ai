"""
Black Swan Labeler - Identifies and labels tail events in historical data
Part of the Black Swan Hunting AI System

This module labels historical market data with black swan characteristics,
enabling the AI to learn patterns that precede extreme market events.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TailEvent:
    """Represents a detected tail event"""
    date: datetime
    symbol: str
    event_type: str  # 'crash', 'melt_up', 'volatility_spike', 'correlation_breakdown'
    magnitude: float  # Size of the move in standard deviations
    duration_days: int
    recovery_days: Optional[int]
    preceding_regime: str
    convexity_score: float
    pre_event_signals: Dict[str, Any]

class BlackSwanLabeler:
    """
    Labels historical data with black swan characteristics
    Implements multiple detection methods for robustness
    """

    def __init__(self,
                 crash_threshold: float = -3.0,  # -3 sigma for crashes
                 melt_up_threshold: float = 3.0,  # +3 sigma for melt-ups
                 vol_spike_threshold: float = 2.0,  # 2x normal volatility
                 correlation_breakdown_threshold: float = 0.5):  # 50% correlation drop
        """
        Initialize the Black Swan Labeler

        Args:
            crash_threshold: Sigma threshold for crash detection
            melt_up_threshold: Sigma threshold for melt-up detection
            vol_spike_threshold: Multiplier for volatility spike detection
            correlation_breakdown_threshold: Threshold for correlation breakdown
        """
        self.crash_threshold = crash_threshold
        self.melt_up_threshold = melt_up_threshold
        self.vol_spike_threshold = vol_spike_threshold
        self.correlation_breakdown_threshold = correlation_breakdown_threshold

        # Pre-event signal windows
        self.signal_windows = {
            'short': 5,   # 5 days before
            'medium': 20,  # 20 days before
            'long': 60     # 60 days before
        }

        logger.info("BlackSwanLabeler initialized with thresholds: "
                   f"crash={crash_threshold}, melt_up={melt_up_threshold}")

    def label_tail_events(self,
                         returns_df: pd.DataFrame,
                         price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Labels historical data with black swan characteristics

        Args:
            returns_df: DataFrame with returns data (columns: date, symbol, returns)
            price_df: Optional DataFrame with price data for additional metrics

        Returns:
            DataFrame with added columns:
            - is_black_swan: bool (True if any type of black swan)
            - black_swan_type: str (type of event)
            - sigma_move: float (move in standard deviations)
            - convexity_score: float (upside/downside ratio)
            - recovery_time: int (days to recover)
            - pre_crash_divergence: float (divergence signals before event)
            - volatility_regime: str (volatility environment)
            - correlation_regime: float (market correlation level)
        """
        df = returns_df.copy()

        # Calculate rolling statistics
        df = self._calculate_rolling_stats(df)

        # Detect different types of tail events
        df = self._detect_sigma_events(df)
        df = self._detect_volatility_spikes(df)
        df = self._detect_correlation_breakdowns(df)

        # Calculate convexity scores
        df = self._calculate_convexity_scores(df)

        # Label volatility and correlation regimes
        df = self._label_market_regimes(df)

        # Detect pre-event signals
        df = self._detect_pre_event_signals(df)

        # Calculate recovery times
        if price_df is not None:
            df = self._calculate_recovery_times(df, price_df)

        # Combine all black swan indicators
        df['is_black_swan'] = (
            df['is_crash'] |
            df['is_melt_up'] |
            df['is_vol_spike'] |
            df['is_correlation_breakdown']
        )

        # Assign black swan type
        df['black_swan_type'] = 'normal'
        df.loc[df['is_crash'], 'black_swan_type'] = 'crash'
        df.loc[df['is_melt_up'], 'black_swan_type'] = 'melt_up'
        df.loc[df['is_vol_spike'], 'black_swan_type'] = 'volatility_spike'
        df.loc[df['is_correlation_breakdown'], 'black_swan_type'] = 'correlation_breakdown'

        logger.info(f"Labeled {df['is_black_swan'].sum()} black swan events "
                   f"out of {len(df)} observations")

        return df

    def _calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling statistics for event detection"""
        # Group by symbol for individual stock statistics
        if 'symbol' in df.columns:
            grouped = df.groupby('symbol')

            # Rolling volatility
            df['volatility_20d'] = grouped['returns'].transform(
                lambda x: x.rolling(20, min_periods=10).std()
            )
            df['volatility_60d'] = grouped['returns'].transform(
                lambda x: x.rolling(60, min_periods=30).std()
            )

            # Rolling mean and std for z-score
            df['returns_mean_60d'] = grouped['returns'].transform(
                lambda x: x.rolling(60, min_periods=30).mean()
            )
            df['returns_std_60d'] = grouped['returns'].transform(
                lambda x: x.rolling(60, min_periods=30).std()
            )

            # Z-score (sigma move)
            df['sigma_move'] = (
                (df['returns'] - df['returns_mean_60d']) /
                df['returns_std_60d'].replace(0, np.nan)
            )

            # Skewness and kurtosis
            df['skew_20d'] = grouped['returns'].transform(
                lambda x: x.rolling(20, min_periods=10).skew()
            )
            df['kurtosis_20d'] = grouped['returns'].transform(
                lambda x: x.rolling(20, min_periods=10).apply(lambda y: stats.kurtosis(y))
            )

        else:
            # Single series analysis
            df['volatility_20d'] = df['returns'].rolling(20, min_periods=10).std()
            df['volatility_60d'] = df['returns'].rolling(60, min_periods=30).std()
            df['returns_mean_60d'] = df['returns'].rolling(60, min_periods=30).mean()
            df['returns_std_60d'] = df['returns'].rolling(60, min_periods=30).std()
            df['sigma_move'] = (
                (df['returns'] - df['returns_mean_60d']) /
                df['returns_std_60d'].replace(0, np.nan)
            )
            df['skew_20d'] = df['returns'].rolling(20, min_periods=10).skew()
            df['kurtosis_20d'] = df['returns'].rolling(20, min_periods=10).apply(
                lambda x: stats.kurtosis(x)
            )

        return df

    def _detect_sigma_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect extreme sigma moves (crashes and melt-ups)"""
        df['is_crash'] = df['sigma_move'] <= self.crash_threshold
        df['is_melt_up'] = df['sigma_move'] >= self.melt_up_threshold

        # Multi-day cumulative moves
        if 'symbol' in df.columns:
            grouped = df.groupby('symbol')
            df['returns_5d'] = grouped['returns'].transform(
                lambda x: x.rolling(5, min_periods=3).sum()
            )
            df['sigma_move_5d'] = (
                (df['returns_5d'] - df['returns_mean_60d'] * 5) /
                (df['returns_std_60d'] * np.sqrt(5)).replace(0, np.nan)
            )
        else:
            df['returns_5d'] = df['returns'].rolling(5, min_periods=3).sum()
            df['sigma_move_5d'] = (
                (df['returns_5d'] - df['returns_mean_60d'] * 5) /
                (df['returns_std_60d'] * np.sqrt(5)).replace(0, np.nan)
            )

        # Mark multi-day crashes/melt-ups
        df['is_crash'] |= df['sigma_move_5d'] <= self.crash_threshold
        df['is_melt_up'] |= df['sigma_move_5d'] >= self.melt_up_threshold

        return df

    def _detect_volatility_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect sudden volatility regime changes"""
        # Volatility ratio
        df['vol_ratio'] = df['volatility_20d'] / df['volatility_60d'].replace(0, np.nan)

        # Volatility spike detection
        df['is_vol_spike'] = df['vol_ratio'] >= self.vol_spike_threshold

        # Also detect using percentile ranks
        if 'symbol' in df.columns:
            df['vol_percentile'] = df.groupby('symbol')['volatility_20d'].transform(
                lambda x: x.rolling(252, min_periods=100).rank(pct=True).iloc[-1]
            )
        else:
            df['vol_percentile'] = df['volatility_20d'].rolling(
                252, min_periods=100
            ).rank(pct=True).iloc[-1]

        # Mark extreme volatility percentiles
        df['is_vol_spike'] |= df['vol_percentile'] >= 0.95

        return df

    def _detect_correlation_breakdowns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect when normal correlations break down"""
        if 'symbol' not in df.columns or df['symbol'].nunique() < 2:
            # Need multiple symbols for correlation analysis
            df['is_correlation_breakdown'] = False
            return df

        # Calculate rolling correlations
        # This is simplified - in production, calculate pairwise correlations
        pivot_df = df.pivot(index='date', columns='symbol', values='returns')

        # Average pairwise correlation
        rolling_corr = pivot_df.rolling(60, min_periods=30).corr()

        # Calculate mean correlation for each date
        mean_correlations = []
        for date in pivot_df.index:
            if date in rolling_corr.index.get_level_values(0):
                corr_matrix = rolling_corr.loc[date]
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu(corr_matrix.values, k=1)
                mean_corr = upper_tri[upper_tri != 0].mean()
                mean_correlations.append(mean_corr)
            else:
                mean_correlations.append(np.nan)

        mean_corr_series = pd.Series(mean_correlations, index=pivot_df.index)

        # Detect correlation breakdowns
        corr_change = mean_corr_series.diff()
        breakdown_threshold = -self.correlation_breakdown_threshold

        # Map back to original dataframe
        df = df.merge(
            pd.DataFrame({
                'date': pivot_df.index,
                'mean_correlation': mean_corr_series.values,
                'correlation_change': corr_change.values
            }),
            on='date',
            how='left'
        )

        df['is_correlation_breakdown'] = df['correlation_change'] <= breakdown_threshold

        return df

    def _calculate_convexity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate convexity scores (upside/downside ratio)
        Higher scores indicate better risk/reward
        """
        if 'symbol' in df.columns:
            grouped = df.groupby('symbol')

            # Separate positive and negative returns
            df['positive_returns'] = df['returns'].where(df['returns'] > 0, 0)
            df['negative_returns'] = df['returns'].where(df['returns'] < 0, 0).abs()

            # Rolling sums
            df['upside_sum_20d'] = grouped['positive_returns'].transform(
                lambda x: x.rolling(20, min_periods=10).sum()
            )
            df['downside_sum_20d'] = grouped['negative_returns'].transform(
                lambda x: x.rolling(20, min_periods=10).sum()
            )

            # Convexity score
            df['convexity_score'] = (
                df['upside_sum_20d'] /
                df['downside_sum_20d'].replace(0, np.nan)
            ).fillna(0)

            # Cap extreme values
            df['convexity_score'] = df['convexity_score'].clip(0, 100)

        else:
            df['positive_returns'] = df['returns'].where(df['returns'] > 0, 0)
            df['negative_returns'] = df['returns'].where(df['returns'] < 0, 0).abs()
            df['upside_sum_20d'] = df['positive_returns'].rolling(20, min_periods=10).sum()
            df['downside_sum_20d'] = df['negative_returns'].rolling(20, min_periods=10).sum()
            df['convexity_score'] = (
                df['upside_sum_20d'] /
                df['downside_sum_20d'].replace(0, np.nan)
            ).fillna(0).clip(0, 100)

        return df

    def _label_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label volatility and market regimes"""
        # Volatility regime
        conditions = [
            df['volatility_20d'] < df['volatility_60d'].quantile(0.25),
            df['volatility_20d'] < df['volatility_60d'].quantile(0.50),
            df['volatility_20d'] < df['volatility_60d'].quantile(0.75),
            df['volatility_20d'] >= df['volatility_60d'].quantile(0.75)
        ]
        choices = ['low_vol', 'normal_vol', 'elevated_vol', 'high_vol']
        df['volatility_regime'] = np.select(conditions, choices, default='normal_vol')

        # Trend regime (if we have enough data)
        if 'symbol' in df.columns:
            df['returns_20d'] = df.groupby('symbol')['returns'].transform(
                lambda x: x.rolling(20, min_periods=10).sum()
            )
        else:
            df['returns_20d'] = df['returns'].rolling(20, min_periods=10).sum()

        df['trend_regime'] = pd.cut(
            df['returns_20d'],
            bins=[-np.inf, -0.10, -0.02, 0.02, 0.10, np.inf],
            labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
        )

        return df

    def _detect_pre_event_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect signals that precede black swan events"""
        # Divergence indicators
        if 'symbol' in df.columns:
            # Price vs volatility divergence
            df['price_vol_divergence'] = (
                df.groupby('symbol')['returns'].transform(
                    lambda x: x.rolling(20).sum()
                ) -
                df.groupby('symbol')['volatility_20d'].transform(
                    lambda x: x.pct_change(20)
                )
            )

            # Unusual option activity proxy (using volume)
            if 'volume' in df.columns:
                df['volume_spike'] = (
                    df.groupby('symbol')['volume'].transform(
                        lambda x: x / x.rolling(20, min_periods=10).mean()
                    )
                )
        else:
            df['price_vol_divergence'] = (
                df['returns'].rolling(20).sum() -
                df['volatility_20d'].pct_change(20)
            )

            if 'volume' in df.columns:
                df['volume_spike'] = (
                    df['volume'] / df['volume'].rolling(20, min_periods=10).mean()
                )

        # Extreme skewness as warning signal
        df['skew_extreme'] = df['skew_20d'].abs() > 2

        # Kurtosis spike (fat tail warning)
        df['kurtosis_extreme'] = df['kurtosis_20d'] > 5

        # Combine pre-event signals
        df['pre_event_warning'] = (
            df['skew_extreme'] |
            df['kurtosis_extreme'] |
            (df.get('volume_spike', 0) > 3)
        )

        return df

    def _calculate_recovery_times(self,
                                 df: pd.DataFrame,
                                 price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time to recover from drawdowns"""
        df['recovery_time'] = 0

        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_df = price_df[price_df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('date')

                # Calculate drawdowns
                symbol_df['cummax'] = symbol_df['close'].cummax()
                symbol_df['drawdown'] = (
                    symbol_df['close'] - symbol_df['cummax']
                ) / symbol_df['cummax']

                # Find drawdown periods
                in_drawdown = symbol_df['drawdown'] < -0.05  # 5% drawdown threshold

                # Calculate recovery times
                recovery_times = []
                for idx, row in symbol_df.iterrows():
                    if in_drawdown.loc[idx]:
                        # Find when price recovers
                        future_df = symbol_df.loc[idx:]
                        recovery_idx = future_df[
                            future_df['close'] >= symbol_df.loc[idx, 'cummax']
                        ].index

                        if len(recovery_idx) > 0:
                            recovery_date = recovery_idx[0]
                            days_to_recover = (recovery_date - idx).days
                            recovery_times.append(days_to_recover)
                        else:
                            recovery_times.append(np.nan)
                    else:
                        recovery_times.append(0)

                # Map back to main dataframe
                df.loc[df['symbol'] == symbol, 'recovery_time'] = recovery_times

        return df

    def get_black_swan_statistics(self, labeled_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics about detected black swan events

        Args:
            labeled_df: DataFrame with labeled black swan events

        Returns:
            Dictionary with statistics
        """
        total_events = labeled_df['is_black_swan'].sum()
        total_observations = len(labeled_df)

        stats = {
            'total_black_swans': total_events,
            'black_swan_frequency': total_events / total_observations if total_observations > 0 else 0,
            'crash_events': labeled_df['is_crash'].sum(),
            'melt_up_events': labeled_df['is_melt_up'].sum(),
            'volatility_spikes': labeled_df['is_vol_spike'].sum(),
            'correlation_breakdowns': labeled_df.get('is_correlation_breakdown', pd.Series([False])).sum(),
            'avg_sigma_move': labeled_df[labeled_df['is_black_swan']]['sigma_move'].abs().mean(),
            'max_sigma_move': labeled_df['sigma_move'].abs().max(),
            'avg_recovery_time': labeled_df[labeled_df['is_black_swan']]['recovery_time'].mean(),
            'events_with_warnings': labeled_df[
                labeled_df['is_black_swan'] & labeled_df['pre_event_warning']
            ].shape[0]
        }

        # Calculate by regime
        if 'volatility_regime' in labeled_df.columns:
            regime_stats = labeled_df.groupby('volatility_regime')['is_black_swan'].agg([
                'sum', 'mean'
            ]).to_dict('index')
            stats['by_volatility_regime'] = regime_stats

        if 'trend_regime' in labeled_df.columns:
            trend_stats = labeled_df.groupby('trend_regime')['is_black_swan'].agg([
                'sum', 'mean'
            ]).to_dict('index')
            stats['by_trend_regime'] = trend_stats

        return stats


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('.')

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

    # Simulate returns with some extreme events
    returns = np.random.normal(0.0005, 0.02, len(dates))

    # Add some black swan events
    black_swan_indices = [100, 300, 500, 700, 900]
    for idx in black_swan_indices:
        if idx < len(returns):
            # Add extreme negative returns (crashes)
            returns[idx] = np.random.uniform(-0.10, -0.05)

    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'returns': returns
    })

    # Initialize labeler
    labeler = BlackSwanLabeler()

    # Label the data
    labeled_df = labeler.label_tail_events(df)

    # Get statistics
    stats = labeler.get_black_swan_statistics(labeled_df)

    print("Black Swan Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nTotal black swans detected: {stats['total_black_swans']}")
    print(f"Black swan frequency: {stats['black_swan_frequency']:.2%}")