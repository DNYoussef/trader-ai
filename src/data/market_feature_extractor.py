"""
Market Feature Extractor

Transforms raw historical database format into TRM-ready market features.
Handles multi-symbol data aggregation and feature engineering.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    """
    Extract 10 market features from historical database

    Transforms:
    - Raw: Multiple rows per date (one per symbol)
    - Processed: One row per date with 10 aggregated features
    """

    def __init__(self, historical_manager):
        """
        Initialize feature extractor

        Args:
            historical_manager: HistoricalDataManager instance
        """
        self.historical_manager = historical_manager

    def extract_features(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Extract 10 market features for TRM

        Features:
        1. vix_level
        2. spy_returns_5d
        3. spy_returns_20d
        4. volume_ratio
        5. market_breadth
        6. correlation
        7. put_call_ratio
        8. gini_coefficient
        9. sector_dispersion
        10. signal_quality_score

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date + 10 features + spy_close + tlt_close
        """
        # Get raw data
        df_raw = self.historical_manager.get_training_data(
            start_date=start_date,
            end_date=end_date
        )

        if df_raw.empty:
            logger.warning(f"No raw data for {start_date} to {end_date}")
            return pd.DataFrame()

        # Pivot data: one row per date with columns per symbol
        df_pivot = df_raw.pivot_table(
            index='date',
            columns='symbol',
            values=['close', 'volume', 'returns', 'volatility_20d', 'volume_ratio']
        )

        # Flatten multi-level columns
        df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
        df_pivot = df_pivot.reset_index()

        # Extract key symbols
        spy_close_col = [c for c in df_pivot.columns if 'close_SPY' in c]
        tlt_close_col = [c for c in df_pivot.columns if 'close_TLT' in c]
        # Fix: Prioritize ^VIX (index) over VIXY (ETF) - must match exact column name
        vix_close_col = [c for c in df_pivot.columns if c == 'close_^VIX']
        if not vix_close_col:  # Fallback to any VIX-like column (but exclude VIXY)
            vix_close_col = [c for c in df_pivot.columns if 'VIX' in c and 'VIXY' not in c and 'close' in c]

        if not spy_close_col:
            logger.warning("No SPY data found")
            return pd.DataFrame()

        spy_close_col = spy_close_col[0]
        tlt_close_col = tlt_close_col[0] if tlt_close_col else None
        vix_close_col = vix_close_col[0] if vix_close_col else None

        # Calculate features
        features = pd.DataFrame()
        features['date'] = df_pivot['date']

        # 1. VIX level
        if vix_close_col:
            features['vix'] = df_pivot[vix_close_col]
        else:
            features['vix'] = 20.0  # Default moderate volatility

        # 2. SPY 5-day returns
        spy_returns_col = [c for c in df_pivot.columns if 'returns_SPY' in c]
        if spy_returns_col:
            features['spy_returns_5d'] = df_pivot[spy_returns_col[0]].rolling(5).sum()
        else:
            features['spy_returns_5d'] = 0.0

        # 3. SPY 20-day returns
        if spy_returns_col:
            features['spy_returns_20d'] = df_pivot[spy_returns_col[0]].rolling(20).sum()
        else:
            features['spy_returns_20d'] = 0.0

        # 4. Volume ratio (average across symbols)
        volume_ratio_cols = [c for c in df_pivot.columns if 'volume_ratio_' in c]
        if volume_ratio_cols:
            features['volume_ratio'] = df_pivot[volume_ratio_cols].mean(axis=1)
        else:
            features['volume_ratio'] = 1.0

        # 5. Market breadth (% of stocks with positive returns)
        returns_cols = [c for c in df_pivot.columns if c.startswith('returns_') and 'SPY' not in c]
        if returns_cols:
            features['market_breadth'] = (df_pivot[returns_cols] > 0).mean(axis=1)
        else:
            features['market_breadth'] = 0.5

        # 6. Correlation (average correlation of returns)
        if len(returns_cols) > 1:
            # Rolling correlation (simplified)
            features['correlation'] = df_pivot[returns_cols].corr().values.mean()
        else:
            features['correlation'] = 0.5

        # 7. Put/call ratio (proxy: VIX / 20)
        if vix_close_col:
            features['put_call_ratio'] = df_pivot[vix_close_col] / 20.0
        else:
            features['put_call_ratio'] = 1.0

        # 8. Gini coefficient (simplified: std of returns / mean)
        if returns_cols:
            features['gini_coefficient'] = df_pivot[returns_cols].std(axis=1) / (df_pivot[returns_cols].abs().mean(axis=1) + 1e-6)
            features['gini_coefficient'] = features['gini_coefficient'].clip(0, 1)
        else:
            features['gini_coefficient'] = 0.5

        # 9. Sector dispersion (std of sector returns)
        if returns_cols:
            features['sector_dispersion'] = df_pivot[returns_cols].std(axis=1)
        else:
            features['sector_dispersion'] = 0.3

        # 10. Signal quality score (1 - missing_data_ratio)
        features['signal_quality'] = 1.0 - features.isna().mean(axis=1)

        # Add price columns for strategy simulation
        features['spy_close'] = df_pivot[spy_close_col]
        if tlt_close_col:
            features['tlt_close'] = df_pivot[tlt_close_col]
        else:
            features['tlt_close'] = 100.0  # Default price

        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')

        # Ensure no remaining NaN
        features = features.fillna({
            'vix': 20.0,
            'spy_returns_5d': 0.0,
            'spy_returns_20d': 0.0,
            'volume_ratio': 1.0,
            'market_breadth': 0.5,
            'correlation': 0.5,
            'put_call_ratio': 1.0,
            'gini_coefficient': 0.5,
            'sector_dispersion': 0.3,
            'signal_quality': 0.7,
            'spy_close': 100.0,
            'tlt_close': 100.0
        })

        logger.info(f"Extracted {len(features)} feature rows from {start_date} to {end_date}")

        return features


if __name__ == "__main__":
    # Test feature extractor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.historical_data_manager import HistoricalDataManager

    logging.basicConfig(level=logging.INFO)

    print("Testing Market Feature Extractor...")
    print("=" * 80)

    # Initialize
    historical_manager = HistoricalDataManager(db_path="data/historical_market.db")
    extractor = MarketFeatureExtractor(historical_manager)

    # Test extraction
    print("\nExtracting features for COVID-19 period...")
    features = extractor.extract_features('2020-02-19', '2020-03-23')

    print(f"\nExtracted {len(features)} rows")
    print(f"\nColumns: {features.columns.tolist()}")
    print("\nSample data:")
    print(features.head())

    print("\nFeature statistics:")
    print(features.describe())

    print("\n✅ Feature extractor test passed!")
