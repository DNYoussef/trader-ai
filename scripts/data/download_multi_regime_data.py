"""
Download Multi-Regime Market Data for TRM Training (2010-2024)

Downloads diverse market conditions: bull/bear/sideways/crisis regimes
to ensure all 8 strategies get adequate training samples.

Creates ~4,500 labeled samples across all regimes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Symbols for market regime detection
CORE_SYMBOLS = ['SPY', 'TLT', 'GLD', 'VIX', 'HYG']

# Extended symbols for feature extraction
EXTENDED_SYMBOLS = [
    'SPY', 'QQQ', 'IWM',  # Equities
    'TLT', 'IEF', 'SHY',  # Bonds
    'GLD', 'SLV',         # Commodities
    'VIX', 'VIXY',        # Volatility (VIX needs special handling)
    'HYG', 'LQD',         # Credit
    'UUP', 'FXE',         # Currency
    'EEM', 'EFA',         # International
]

# Known regime periods for validation
KNOWN_REGIMES = {
    'bull_markets': [
        ('2013-01-01', '2015-06-30'),  # Post-2012 recovery
        ('2016-07-01', '2018-01-31'),  # Trump rally
        ('2019-01-01', '2020-01-31'),  # 2019 bull run
        ('2020-04-01', '2021-12-31'),  # COVID recovery
        ('2023-01-01', '2024-06-30'),  # 2023-24 rally
    ],
    'bear_markets': [
        ('2011-07-01', '2011-10-31'),  # Debt ceiling crisis
        ('2015-08-01', '2016-02-29'),  # China devaluation
        ('2018-10-01', '2018-12-31'),  # Q4 2018 selloff
        ('2020-02-15', '2020-03-23'),  # COVID crash
        ('2022-01-01', '2022-10-31'),  # 2022 bear market
    ],
    'sideways_markets': [
        ('2010-06-01', '2011-06-30'),  # Post-GFC recovery
        ('2015-01-01', '2015-07-31'),  # 2015 chop
        ('2021-01-01', '2021-03-31'),  # Meme stock volatility
    ],
    'crisis_periods': [
        ('2010-05-01', '2010-05-31'),  # Flash Crash
        ('2011-08-01', '2011-08-31'),  # US Downgrade
        ('2015-08-20', '2015-08-26'),  # China Black Monday
        ('2018-02-01', '2018-02-15'),  # Volmageddon
        ('2020-03-01', '2020-03-23'),  # COVID crash
        ('2023-03-08', '2023-03-15'),  # SVB collapse
    ]
}


class MultiRegimeDataDownloader:
    """Downloads market data with regime diversity for TRM training."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).resolve().parents[2] / 'data' / 'regimes'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_yfinance_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download data using yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        logger.info(f"Downloading {len(symbols)} symbols from {start_date} to {end_date}")

        all_data = []
        for symbol in symbols:
            try:
                # Handle VIX specially (it's ^VIX in Yahoo)
                yahoo_symbol = '^VIX' if symbol == 'VIX' else symbol

                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start_date, end=end_date)

                if not df.empty:
                    df = df.reset_index()
                    df['symbol'] = symbol
                    df = df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    # Ensure date is datetime without timezone
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    all_data.append(df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
                    logger.debug(f"Downloaded {len(df)} rows for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
                continue

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total rows downloaded: {len(combined)}")
            return combined

        return pd.DataFrame()

    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection."""
        # Pivot to get SPY, TLT, VIX columns
        pivot_df = df.pivot_table(
            index='date',
            columns='symbol',
            values='close',
            aggfunc='first'
        ).reset_index()

        features = pd.DataFrame()
        features['date'] = pivot_df['date']

        # SPY returns
        if 'SPY' in pivot_df.columns:
            features['spy_return_5d'] = pivot_df['SPY'].pct_change(5)
            features['spy_return_20d'] = pivot_df['SPY'].pct_change(20)
            features['spy_volatility_20d'] = pivot_df['SPY'].pct_change().rolling(20).std() * np.sqrt(252)

        # VIX level
        if 'VIX' in pivot_df.columns:
            features['vix_level'] = pivot_df['VIX']
            features['vix_change_5d'] = pivot_df['VIX'].pct_change(5)

        # TLT correlation (flight to quality)
        if 'SPY' in pivot_df.columns and 'TLT' in pivot_df.columns:
            spy_ret = pivot_df['SPY'].pct_change()
            tlt_ret = pivot_df['TLT'].pct_change()
            features['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)

        # Credit spread proxy (HYG vs SPY correlation)
        if 'HYG' in pivot_df.columns and 'SPY' in pivot_df.columns:
            hyg_ret = pivot_df['HYG'].pct_change()
            spy_ret = pivot_df['SPY'].pct_change()
            features['credit_stress'] = -hyg_ret.rolling(20).corr(spy_ret)

        return features.dropna()

    def classify_regime(self, features: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime based on features."""
        features = features.copy()

        # Initialize regime column
        features['regime'] = 'normal'

        # Crisis: VIX > 30 or large negative returns
        crisis_mask = (
            (features['vix_level'] > 30) |
            (features['spy_return_5d'] < -0.05)
        )
        features.loc[crisis_mask, 'regime'] = 'crisis'

        # Bear: Negative 20d returns, elevated VIX
        bear_mask = (
            (features['spy_return_20d'] < -0.03) &
            (features['vix_level'] > 18) &
            (features['regime'] != 'crisis')
        )
        features.loc[bear_mask, 'regime'] = 'bear'

        # Bull: Positive returns, low VIX
        bull_mask = (
            (features['spy_return_20d'] > 0.02) &
            (features['vix_level'] < 20) &
            (features['regime'] != 'crisis')
        )
        features.loc[bull_mask, 'regime'] = 'bull'

        # High volatility: VIX 20-30 but not in crisis
        high_vol_mask = (
            (features['vix_level'].between(20, 30)) &
            (features['regime'] == 'normal')
        )
        features.loc[high_vol_mask, 'regime'] = 'high_volatility'

        return features

    def generate_strategy_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy labels based on regime and features."""
        features = features.copy()

        # Strategy mapping based on regime and features
        # 0: ultra_defensive, 1: defensive, 2: balanced_safe, 3: balanced_growth
        # 4: growth, 5: aggressive_growth, 6: contrarian_long, 7: tactical_opportunity

        features['optimal_strategy'] = 3  # Default: balanced_growth

        # Crisis -> ultra_defensive (0)
        features.loc[features['regime'] == 'crisis', 'optimal_strategy'] = 0

        # Bear -> defensive (1) or contrarian_long (6) if oversold
        bear_mask = features['regime'] == 'bear'
        # Contrarian: oversold bounce candidate (5d return < -5% in bear)
        oversold = features['spy_return_5d'] < -0.05
        features.loc[bear_mask & ~oversold, 'optimal_strategy'] = 1
        features.loc[bear_mask & oversold, 'optimal_strategy'] = 6

        # High volatility -> balanced_safe (2) or tactical (7)
        high_vol_mask = features['regime'] == 'high_volatility'
        positive_momentum = features['spy_return_5d'] > 0
        features.loc[high_vol_mask & ~positive_momentum, 'optimal_strategy'] = 2
        features.loc[high_vol_mask & positive_momentum, 'optimal_strategy'] = 7

        # Bull -> growth (4) or aggressive_growth (5)
        bull_mask = features['regime'] == 'bull'
        strong_bull = features['spy_return_20d'] > 0.05
        features.loc[bull_mask & ~strong_bull, 'optimal_strategy'] = 4
        features.loc[bull_mask & strong_bull, 'optimal_strategy'] = 5

        # Normal market with slight weakness -> contrarian_long (6)
        # This ensures strategy 6 gets some samples outside bear markets
        normal_mask = features['regime'] == 'normal'
        slight_dip = (features['spy_return_5d'] < -0.02) & (features['spy_return_5d'] > -0.05)
        features.loc[normal_mask & slight_dip, 'optimal_strategy'] = 6

        # Normal -> balanced_growth (3)
        # Already set as default for remaining normal samples

        return features

    def download_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download 2010-2024 data and create labeled dataset."""
        logger.info("=" * 60)
        logger.info("Downloading Multi-Regime Dataset (2010-2024)")
        logger.info("=" * 60)

        # Download raw data
        raw_data = self.download_yfinance_data(
            symbols=EXTENDED_SYMBOLS,
            start_date='2010-01-01',
            end_date='2024-12-31'
        )

        if raw_data.empty:
            logger.error("No data downloaded")
            return pd.DataFrame(), pd.DataFrame()

        # Save raw data
        raw_path = self.data_dir / 'raw_market_data_2010_2024.parquet'
        raw_data.to_parquet(raw_path)
        logger.info(f"Saved raw data to {raw_path}")

        # Calculate features and classify regimes
        features = self.calculate_regime_features(raw_data)
        labeled = self.classify_regime(features)
        labeled = self.generate_strategy_labels(labeled)

        # Log regime distribution
        regime_counts = labeled['regime'].value_counts()
        logger.info("\nRegime Distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count} samples ({100*count/len(labeled):.1f}%)")

        # Log strategy distribution
        strategy_counts = labeled['optimal_strategy'].value_counts().sort_index()
        logger.info("\nStrategy Distribution:")
        strategy_names = [
            'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
            'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
        ]
        for idx in range(8):
            count = strategy_counts.get(idx, 0)
            logger.info(f"  {idx} ({strategy_names[idx]}): {count} samples ({100*count/len(labeled):.1f}%)")

        # Save labeled data
        labeled_path = self.data_dir / 'multi_regime_labeled_2010_2024.parquet'
        labeled.to_parquet(labeled_path)
        logger.info(f"\nSaved labeled data to {labeled_path}")
        logger.info(f"Total samples: {len(labeled)}")

        return raw_data, labeled


def main():
    """Main entry point."""
    downloader = MultiRegimeDataDownloader()
    raw_data, labeled_data = downloader.download_full_dataset()

    if not labeled_data.empty:
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Raw data rows: {len(raw_data)}")
        logger.info(f"Labeled samples: {len(labeled_data)}")

        # Check if all 8 strategies have samples
        strategy_counts = labeled_data['optimal_strategy'].value_counts()
        min_samples = strategy_counts.min()

        if min_samples < 50:
            logger.warning(f"WARNING: Some strategies have < 50 samples (min: {min_samples})")
            logger.warning("Consider adjusting regime classification thresholds")
        else:
            logger.info(f"SUCCESS: All strategies have >= {min_samples} samples")


if __name__ == '__main__':
    main()
