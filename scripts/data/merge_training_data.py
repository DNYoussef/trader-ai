"""
Merge Multi-Regime Data with Black Swan Labels

Combines:
1. Multi-regime labeled data (2010-2024 with strategy labels)
2. Black Swan labeled data (crisis periods with tail event labels)

Creates a unified training dataset with:
- Regime-based strategy labels
- Black Swan event markers
- Crisis-enhanced samples for rare strategies

Usage:
    python scripts/data/merge_training_data.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Black Swan events from the project's historical database
BLACK_SWAN_EVENTS = {
    'asian_crisis_1997': ('1997-07-01', '1997-12-31'),
    'ltcm_1998': ('1998-08-01', '1998-10-31'),
    'dotcom_crash_2000': ('2000-03-01', '2002-10-31'),
    'sept_11_2001': ('2001-09-01', '2001-10-31'),
    'gfc_2008': ('2007-12-01', '2009-03-31'),
    'flash_crash_2010': ('2010-05-01', '2010-05-31'),
    'us_downgrade_2011': ('2011-08-01', '2011-08-31'),
    'china_deval_2015': ('2015-08-01', '2016-02-29'),
    'volmageddon_2018': ('2018-02-01', '2018-02-15'),
    'q4_selloff_2018': ('2018-10-01', '2018-12-31'),
    'covid_crash_2020': ('2020-02-15', '2020-03-23'),
    'svb_collapse_2023': ('2023-03-08', '2023-03-15'),
}


class TrainingDataMerger:
    """Merges regime data with Black Swan labels."""

    def __init__(
        self,
        regime_data_path: Optional[Path] = None,
        black_swan_data_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        self.regime_data_path = regime_data_path or Path('data/regimes/multi_regime_labeled_2010_2024.parquet')
        self.black_swan_data_path = black_swan_data_path or Path('data/trm_training/black_swan_labels.parquet')
        self.output_dir = output_dir or Path('data/training')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_regime_data(self) -> pd.DataFrame:
        """Load regime-labeled data."""
        if not self.regime_data_path.exists():
            logger.error(f"Regime data not found at {self.regime_data_path}")
            logger.info("Run: python scripts/data/download_multi_regime_data.py first")
            return pd.DataFrame()

        df = pd.read_parquet(self.regime_data_path)
        logger.info(f"Loaded {len(df)} regime-labeled samples")
        return df

    def load_black_swan_data(self) -> pd.DataFrame:
        """Load Black Swan labeled data."""
        if not self.black_swan_data_path.exists():
            logger.warning(f"Black Swan data not found at {self.black_swan_data_path}")
            logger.info("Will use event markers from predefined list")
            return pd.DataFrame()

        df = pd.read_parquet(self.black_swan_data_path)
        logger.info(f"Loaded {len(df)} Black Swan samples")
        return df

    def add_black_swan_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Black Swan event markers to regime data."""
        df = df.copy()
        df['is_black_swan'] = False
        df['black_swan_event'] = ''

        for event_name, (start, end) in BLACK_SWAN_EVENTS.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)

            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            n_matches = mask.sum()

            if n_matches > 0:
                df.loc[mask, 'is_black_swan'] = True
                df.loc[mask, 'black_swan_event'] = event_name
                logger.info(f"Marked {n_matches} samples as {event_name}")

        total_bs = df['is_black_swan'].sum()
        logger.info(f"Total Black Swan samples: {total_bs} ({100*total_bs/len(df):.1f}%)")

        return df

    def enhance_crisis_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance crisis samples for better rare strategy representation.

        During Black Swan events, adjust strategy labels to favor:
        - ultra_defensive (0) during peak crisis
        - contrarian_long (6) during recovery phase
        """
        df = df.copy()

        # For Black Swan events, ensure ultra_defensive is used during crisis
        crisis_mask = df['is_black_swan'] & (df['regime'] == 'crisis')
        df.loc[crisis_mask, 'optimal_strategy'] = 0

        # During bear phase of Black Swan, favor contrarian at oversold levels
        bear_bs_mask = df['is_black_swan'] & (df['regime'] == 'bear')
        if 'spy_return_5d' in df.columns:
            oversold = df['spy_return_5d'] < -0.05
            df.loc[bear_bs_mask & oversold, 'optimal_strategy'] = 6

        # Log strategy distribution after enhancement
        strategy_counts = df['optimal_strategy'].value_counts().sort_index()
        logger.info("\nStrategy distribution after enhancement:")
        strategy_names = [
            'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
            'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
        ]
        for idx in range(8):
            count = strategy_counts.get(idx, 0)
            logger.info(f"  {idx} ({strategy_names[idx]}): {count} samples ({100*count/len(df):.1f}%)")

        return df

    def compute_sample_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-sample weights for training.

        Weights based on:
        1. Inverse class frequency (rare strategies weighted higher)
        2. Black Swan bonus (crisis samples more important)
        3. Recency (newer samples slightly higher weight)
        """
        df = df.copy()

        # 1. Class weights (sqrt inverse frequency, capped at 5x)
        strategy_counts = df['optimal_strategy'].value_counts()
        total = len(df)
        class_weights = {}
        for strategy in range(8):
            count = strategy_counts.get(strategy, 1)
            weight = np.sqrt(total / count)
            class_weights[strategy] = min(weight, 5.0)

        df['class_weight'] = df['optimal_strategy'].map(class_weights)

        # 2. Black Swan bonus (1.5x for crisis events)
        df['bs_weight'] = 1.0
        df.loc[df['is_black_swan'], 'bs_weight'] = 1.5

        # 3. Recency weight (linear decay, oldest=0.8, newest=1.2)
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
            n = len(df_sorted)
            recency = np.linspace(0.8, 1.2, n)
            df.loc[df_sorted.index, 'recency_weight'] = recency
        else:
            df['recency_weight'] = 1.0

        # Combined weight
        df['sample_weight'] = df['class_weight'] * df['bs_weight'] * df['recency_weight']

        # Normalize to mean=1
        df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()

        logger.info(f"\nSample weight stats:")
        logger.info(f"  Min: {df['sample_weight'].min():.3f}")
        logger.info(f"  Max: {df['sample_weight'].max():.3f}")
        logger.info(f"  Mean: {df['sample_weight'].mean():.3f}")
        logger.info(f"  Std: {df['sample_weight'].std():.3f}")

        return df

    def merge_with_black_swan_features(
        self,
        regime_df: pd.DataFrame,
        black_swan_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge regime data with Black Swan feature data if available.

        Black Swan data may have additional features like:
        - tail_risk_score
        - crisis_intensity
        - recovery_signal
        """
        if black_swan_df.empty:
            return regime_df

        # Check for common date column
        if 'date' not in regime_df.columns or 'date' not in black_swan_df.columns:
            logger.warning("Cannot merge: missing date column")
            return regime_df

        # Ensure both dates are datetime
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        black_swan_df['date'] = pd.to_datetime(black_swan_df['date'])

        # Identify additional features in Black Swan data
        # Exclude 'features' column as it contains numpy arrays
        regime_cols = set(regime_df.columns)
        bs_cols = set(black_swan_df.columns)
        additional_cols = bs_cols - regime_cols - {'date', 'features'}

        if not additional_cols:
            logger.info("No additional features to merge from Black Swan data")
            return regime_df

        logger.info(f"Merging additional features: {additional_cols}")

        # Left join to preserve all regime data
        merged = pd.merge(
            regime_df,
            black_swan_df[['date'] + list(additional_cols)],
            on='date',
            how='left'
        )

        # Fill NaN for non-Black Swan periods
        for col in additional_cols:
            if merged[col].dtype in ['float64', 'int64']:
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = merged[col].fillna('')

        logger.info(f"Merged dataset: {len(merged)} samples with {len(merged.columns)} columns")

        return merged

    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Create time-based train/val/test splits.

        Uses chronological split to avoid data leakage:
        - Train: earliest 70%
        - Val: middle 15%
        - Test: latest 15%
        """
        df = df.sort_values('date').reset_index(drop=True)
        n = len(df)

        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        splits = {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'test': df.iloc[val_end:].copy(),
        }

        for name, split_df in splits.items():
            logger.info(f"{name}: {len(split_df)} samples ({100*len(split_df)/n:.1f}%)")
            logger.info(f"  Date range: {split_df['date'].min()} to {split_df['date'].max()}")

        return splits

    def _clean_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe for parquet serialization."""
        df_clean = df.copy()
        cols_to_drop = []

        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                # Check if contains numpy arrays or lists
                non_null = df_clean[col].dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    if isinstance(sample, (np.ndarray, list)):
                        # Drop columns with arrays (they're redundant with extracted features)
                        cols_to_drop.append(col)
                        logger.info(f"Dropping column {col} (contains arrays)")

        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        return df_clean

    def save_merged_data(
        self,
        df: pd.DataFrame,
        splits: Optional[Dict[str, pd.DataFrame]] = None
    ) -> None:
        """Save merged dataset and splits."""
        # Clean for parquet serialization
        df_clean = self._clean_for_parquet(df)

        # Save full dataset
        full_path = self.output_dir / 'merged_regime_blackswan.parquet'
        df_clean.to_parquet(full_path)
        logger.info(f"Saved full dataset to {full_path}")

        # Save splits (apply same cleaning)
        if splits:
            for name, split_df in splits.items():
                split_clean = self._clean_for_parquet(split_df)
                split_path = self.output_dir / f'{name}_data.parquet'
                split_clean.to_parquet(split_path)
                logger.info(f"Saved {name} split to {split_path}")

        # Save metadata
        metadata = {
            'total_samples': len(df),
            'n_black_swan': int(df['is_black_swan'].sum()),
            'strategy_distribution': df['optimal_strategy'].value_counts().to_dict(),
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'date_range': [str(df['date'].min()), str(df['date'].max())],
            'features': list(df.columns),
            'created': datetime.now().isoformat(),
        }

        import json
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")

    def merge(self) -> pd.DataFrame:
        """Run full merge pipeline."""
        logger.info("=" * 60)
        logger.info("Merging Training Data")
        logger.info("=" * 60)

        # Load data
        regime_df = self.load_regime_data()
        if regime_df.empty:
            return pd.DataFrame()

        black_swan_df = self.load_black_swan_data()

        # Add Black Swan markers
        merged_df = self.add_black_swan_markers(regime_df)

        # Enhance crisis samples
        merged_df = self.enhance_crisis_samples(merged_df)

        # Merge with Black Swan features if available
        merged_df = self.merge_with_black_swan_features(merged_df, black_swan_df)

        # Compute sample weights
        merged_df = self.compute_sample_weights(merged_df)

        # Create splits
        splits = self.create_train_val_test_split(merged_df)

        # Save
        self.save_merged_data(merged_df, splits)

        logger.info("\n" + "=" * 60)
        logger.info("MERGE COMPLETE")
        logger.info("=" * 60)

        return merged_df


def main():
    merger = TrainingDataMerger()
    merged_df = merger.merge()

    if not merged_df.empty:
        logger.info(f"\nFinal dataset: {len(merged_df)} samples")
        logger.info(f"Black Swan events: {merged_df['is_black_swan'].sum()}")
        logger.info(f"Unique regimes: {merged_df['regime'].nunique()}")
        logger.info(f"All 8 strategies represented: {merged_df['optimal_strategy'].nunique() == 8}")


if __name__ == '__main__':
    main()
