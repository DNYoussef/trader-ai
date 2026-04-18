"""
Expand Crisis Dataset: 1997-2024 + Synthetic Augmentation

This script:
1. Downloads extended historical data (1997-2024) to capture more Black Swans
2. Creates synthetic crisis patterns based on real crisis signatures
3. Oversamples crisis periods for balanced training

Major Black Swan Events to Capture:
- 1997: Asian Financial Crisis
- 1998: LTCM Collapse, Russian Default
- 2000-2002: Dot-com Crash
- 2001: September 11 Attacks
- 2007-2009: Global Financial Crisis
- 2010: Flash Crash
- 2011: US Debt Downgrade, European Debt Crisis
- 2015: China Devaluation
- 2018: Volmageddon, Q4 Selloff
- 2020: COVID Crash
- 2022: Crypto/Tech Crash
- 2023: SVB Collapse
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Extended Black Swan events (1997-2024)
BLACK_SWAN_EVENTS = {
    # 1997-1999
    'asian_crisis_1997': {
        'start': '1997-07-02',
        'peak': '1997-10-27',
        'end': '1997-12-31',
        'severity': 0.8,
        'description': 'Asian Financial Crisis - Thai baht collapse spreads'
    },
    'ltcm_1998': {
        'start': '1998-08-17',
        'peak': '1998-09-23',
        'end': '1998-10-15',
        'severity': 0.85,
        'description': 'LTCM collapse and Russian default'
    },

    # 2000-2002 Dot-com
    'dotcom_crash_2000': {
        'start': '2000-03-10',
        'peak': '2000-04-14',
        'end': '2000-05-24',
        'severity': 0.9,
        'description': 'Dot-com bubble burst begins'
    },
    'dotcom_bottom_2002': {
        'start': '2002-07-01',
        'peak': '2002-10-09',
        'end': '2002-10-31',
        'severity': 0.85,
        'description': 'Dot-com crash bottom'
    },

    # 2001
    'sept_11_2001': {
        'start': '2001-09-10',
        'peak': '2001-09-17',
        'end': '2001-09-28',
        'severity': 0.95,
        'description': 'September 11 terrorist attacks'
    },

    # 2007-2009 GFC
    'gfc_start_2007': {
        'start': '2007-07-16',
        'peak': '2007-08-16',
        'end': '2007-09-18',
        'severity': 0.7,
        'description': 'GFC begins - subprime concerns'
    },
    'bear_stearns_2008': {
        'start': '2008-03-10',
        'peak': '2008-03-17',
        'end': '2008-03-31',
        'severity': 0.85,
        'description': 'Bear Stearns collapse'
    },
    'lehman_2008': {
        'start': '2008-09-12',
        'peak': '2008-10-10',
        'end': '2008-11-20',
        'severity': 1.0,
        'description': 'Lehman Brothers collapse - peak GFC'
    },
    'gfc_bottom_2009': {
        'start': '2009-02-23',
        'peak': '2009-03-09',
        'end': '2009-03-23',
        'severity': 0.9,
        'description': 'GFC market bottom'
    },

    # 2010-2015
    'flash_crash_2010': {
        'start': '2010-05-06',
        'peak': '2010-05-06',
        'end': '2010-05-20',
        'severity': 0.9,
        'description': 'Flash Crash - 1000 point drop in minutes'
    },
    'us_downgrade_2011': {
        'start': '2011-08-04',
        'peak': '2011-08-08',
        'end': '2011-08-22',
        'severity': 0.75,
        'description': 'US debt downgrade'
    },
    'euro_crisis_2011': {
        'start': '2011-09-12',
        'peak': '2011-10-04',
        'end': '2011-10-27',
        'severity': 0.7,
        'description': 'European sovereign debt crisis'
    },
    'china_deval_2015': {
        'start': '2015-08-11',
        'peak': '2015-08-24',
        'end': '2015-09-29',
        'severity': 0.8,
        'description': 'China devaluation Black Monday'
    },

    # 2018
    'volmageddon_2018': {
        'start': '2018-02-02',
        'peak': '2018-02-05',
        'end': '2018-02-12',
        'severity': 0.85,
        'description': 'Volmageddon - XIV collapse'
    },
    'q4_selloff_2018': {
        'start': '2018-10-03',
        'peak': '2018-12-24',
        'end': '2018-12-31',
        'severity': 0.75,
        'description': 'Q4 2018 selloff'
    },

    # 2020s
    'covid_crash_2020': {
        'start': '2020-02-19',
        'peak': '2020-03-16',
        'end': '2020-03-23',
        'severity': 1.0,
        'description': 'COVID-19 crash - fastest bear market ever'
    },
    'crypto_crash_2022': {
        'start': '2022-01-03',
        'peak': '2022-06-13',
        'end': '2022-06-18',
        'severity': 0.7,
        'description': 'Crypto/Tech crash'
    },
    'svb_collapse_2023': {
        'start': '2023-03-08',
        'peak': '2023-03-13',
        'end': '2023-03-20',
        'severity': 0.65,
        'description': 'SVB bank collapse'
    },
}


class ExtendedDataDownloader:
    """Downloads extended historical data from 1997-2024."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('data/extended')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_extended_data(self) -> pd.DataFrame:
        """Download SPY, TLT equivalents, and VIX from 1997-2024."""
        logger.info("Downloading extended historical data (1997-2024)...")

        # Symbols to download
        # Note: TLT started in 2002, use IEF or proxy before that
        symbols = {
            'SPY': 'SPY',      # S&P 500 ETF (1993+)
            'TLT': 'TLT',      # 20+ Year Treasury (2002+)
            'IEF': 'IEF',      # 7-10 Year Treasury (2002+)
            'VIX': '^VIX',     # Volatility Index
            'GLD': 'GLD',      # Gold (2004+)
        }

        all_data = {}
        for name, ticker in symbols.items():
            try:
                logger.info(f"Downloading {name} ({ticker})...")
                df = yf.download(ticker, start='1997-01-01', end='2024-12-31', progress=False)
                if not df.empty:
                    all_data[name] = df['Close']
                    logger.info(f"  {name}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

        # Combine into single DataFrame
        combined = pd.DataFrame(all_data)
        combined.index.name = 'date'
        combined = combined.reset_index()

        # Calculate features
        if 'SPY' in combined.columns:
            combined['spy_return_1d'] = combined['SPY'].pct_change()
            combined['spy_return_5d'] = combined['SPY'].pct_change(5)
            combined['spy_return_20d'] = combined['SPY'].pct_change(20)
            combined['spy_volatility_20d'] = combined['spy_return_1d'].rolling(20).std() * np.sqrt(252)

        if 'VIX' in combined.columns:
            combined['vix_level'] = combined['VIX']
            combined['vix_change_5d'] = combined['VIX'].pct_change(5)

        if 'SPY' in combined.columns and 'TLT' in combined.columns:
            spy_ret = combined['SPY'].pct_change()
            tlt_ret = combined['TLT'].pct_change()
            combined['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)

        # Drop rows with missing data
        combined = combined.dropna()

        logger.info(f"Combined dataset: {len(combined)} rows")

        # Save raw data
        combined.to_parquet(self.output_dir / 'extended_market_data.parquet')

        return combined


class CrisisPatternAugmenter:
    """Creates synthetic crisis patterns for data augmentation."""

    def __init__(self, real_crisis_data: pd.DataFrame):
        """
        Args:
            real_crisis_data: DataFrame with actual crisis period data
        """
        self.real_crisis_data = real_crisis_data
        self.crisis_signatures = self._extract_crisis_signatures()

    def _extract_crisis_signatures(self) -> Dict[str, Dict]:
        """Extract common patterns from real crises."""
        signatures = {}

        for event_name, event_info in BLACK_SWAN_EVENTS.items():
            start = pd.to_datetime(event_info['start'])
            peak = pd.to_datetime(event_info['peak'])
            end = pd.to_datetime(event_info['end'])

            # Get data for this crisis
            mask = (
                (self.real_crisis_data['date'] >= start - timedelta(days=30)) &
                (self.real_crisis_data['date'] <= end + timedelta(days=10))
            )
            crisis_data = self.real_crisis_data[mask].copy()

            if len(crisis_data) < 10:
                continue

            # Extract signature
            signatures[event_name] = {
                'pre_crisis_vix': crisis_data['vix_level'].iloc[:10].mean() if 'vix_level' in crisis_data else 20,
                'peak_vix': crisis_data['vix_level'].max() if 'vix_level' in crisis_data else 40,
                'pre_crisis_return': crisis_data['spy_return_5d'].iloc[:5].mean() if 'spy_return_5d' in crisis_data else -0.01,
                'peak_drawdown': crisis_data['spy_return_5d'].min() if 'spy_return_5d' in crisis_data else -0.10,
                'duration_days': (end - start).days,
                'severity': event_info['severity'],
            }

        logger.info(f"Extracted {len(signatures)} crisis signatures")
        return signatures

    def generate_synthetic_crisis(
        self,
        base_data: pd.DataFrame,
        insert_idx: int,
        severity: float = 0.8
    ) -> pd.DataFrame:
        """
        Generate a synthetic crisis by modifying base data.

        Args:
            base_data: DataFrame to modify
            insert_idx: Index where to insert crisis
            severity: Crisis intensity (0.5-1.0)

        Returns:
            Modified DataFrame with synthetic crisis
        """
        augmented = base_data.copy()

        # Crisis parameters based on severity
        crisis_duration = int(15 + severity * 20)  # 15-35 days
        vix_spike = 20 + severity * 50  # VIX to 20-70
        max_drawdown = -0.05 - severity * 0.15  # -5% to -20%

        # Modify features around insert_idx
        start_idx = max(0, insert_idx - 20)
        end_idx = min(len(augmented), insert_idx + crisis_duration)

        for i in range(start_idx, end_idx):
            if i >= len(augmented):
                break

            # Days relative to crisis start
            days_to_crisis = insert_idx - i
            days_into_crisis = i - insert_idx

            if days_to_crisis > 0:
                # Pre-crisis: gradual VIX rise
                if 'vix_level' in augmented.columns:
                    vix_increase = (20 - days_to_crisis) / 20 * (vix_spike - 20) * 0.3
                    augmented.loc[augmented.index[i], 'vix_level'] += max(0, vix_increase)

            elif days_into_crisis >= 0 and days_into_crisis < crisis_duration:
                # During crisis
                crisis_progress = days_into_crisis / crisis_duration

                if 'vix_level' in augmented.columns:
                    # VIX spikes then decays
                    if crisis_progress < 0.3:
                        vix_mult = 1 + (vix_spike / 20 - 1) * (crisis_progress / 0.3)
                    else:
                        vix_mult = 1 + (vix_spike / 20 - 1) * (1 - (crisis_progress - 0.3) / 0.7)
                    augmented.loc[augmented.index[i], 'vix_level'] = 20 * vix_mult

                if 'spy_return_5d' in augmented.columns:
                    # Returns drop sharply then recover
                    if crisis_progress < 0.3:
                        ret_mult = max_drawdown * (crisis_progress / 0.3)
                    else:
                        ret_mult = max_drawdown * (1 - (crisis_progress - 0.3) / 0.7)
                    augmented.loc[augmented.index[i], 'spy_return_5d'] = ret_mult

        return augmented

    def augment_dataset(
        self,
        base_data: pd.DataFrame,
        n_synthetic: int = 20,
        min_spacing: int = 60
    ) -> pd.DataFrame:
        """
        Augment dataset with synthetic crises.

        Args:
            base_data: Original dataset
            n_synthetic: Number of synthetic crises to add
            min_spacing: Minimum days between synthetic crises

        Returns:
            Augmented dataset
        """
        logger.info(f"Generating {n_synthetic} synthetic crisis patterns...")

        augmented = base_data.copy()

        # Find valid insertion points (not near real crises)
        valid_indices = []
        for i in range(100, len(base_data) - 100):
            # Check not near existing crisis
            is_near_crisis = False
            for event_info in BLACK_SWAN_EVENTS.values():
                crisis_date = pd.to_datetime(event_info['start'])
                if 'date' in base_data.columns:
                    row_date = base_data.iloc[i]['date']
                    if abs((row_date - crisis_date).days) < 60:
                        is_near_crisis = True
                        break

            if not is_near_crisis:
                valid_indices.append(i)

        # Select insertion points with minimum spacing
        np.random.seed(42)
        selected_indices = []
        for _ in range(n_synthetic):
            if not valid_indices:
                break

            idx = np.random.choice(valid_indices)
            selected_indices.append(idx)

            # Remove nearby indices
            valid_indices = [v for v in valid_indices if abs(v - idx) > min_spacing]

        # Generate synthetic crises
        for idx in selected_indices:
            severity = np.random.uniform(0.5, 1.0)
            augmented = self.generate_synthetic_crisis(augmented, idx, severity)

        logger.info(f"Added {len(selected_indices)} synthetic crises")

        return augmented


class CrisisLabelGenerator:
    """Generates crisis labels for training."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generate_labels(
        self,
        horizons: List[int] = [30, 7, 1]
    ) -> pd.DataFrame:
        """Generate crisis labels for each horizon."""
        labeled = self.data.copy()

        # Initialize label columns
        for h in horizons:
            labeled[f'crisis_in_{h}d'] = 0

        dates = pd.to_datetime(labeled['date'])

        # Label each Black Swan event
        for event_name, event_info in BLACK_SWAN_EVENTS.items():
            crisis_start = pd.to_datetime(event_info['start'])

            for horizon in horizons:
                warning_start = crisis_start - timedelta(days=horizon)

                mask = (dates >= warning_start) & (dates < crisis_start)
                n_labeled = mask.sum()

                labeled.loc[mask, f'crisis_in_{h}d'] = 1

                if n_labeled > 0:
                    logger.debug(f"{event_name}: {n_labeled} samples for {horizon}d horizon")

        # Log label distribution
        logger.info("\nCrisis label distribution:")
        for h in horizons:
            col = f'crisis_in_{h}d'
            n_crisis = labeled[col].sum()
            pct = 100 * n_crisis / len(labeled)
            logger.info(f"  {h}d horizon: {n_crisis} samples ({pct:.2f}%)")

        return labeled


def main():
    if not DEPS_AVAILABLE:
        logger.error("Missing dependencies. Install: pip install pandas yfinance")
        return 1

    output_dir = Path('data/extended')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download extended data
    logger.info("=" * 60)
    logger.info("STEP 1: Download Extended Historical Data")
    logger.info("=" * 60)

    downloader = ExtendedDataDownloader(output_dir)
    extended_data = downloader.download_extended_data()

    # Step 2: Augment with synthetic crises
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Augment with Synthetic Crisis Patterns")
    logger.info("=" * 60)

    augmenter = CrisisPatternAugmenter(extended_data)
    augmented_data = augmenter.augment_dataset(extended_data, n_synthetic=30)

    # Step 3: Generate crisis labels
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Generate Crisis Labels")
    logger.info("=" * 60)

    labeler = CrisisLabelGenerator(augmented_data)
    labeled_data = labeler.generate_labels(horizons=[30, 7, 1])

    # Save final dataset
    labeled_data.to_parquet(output_dir / 'crisis_training_data.parquet')
    logger.info(f"\nSaved {len(labeled_data)} samples to {output_dir / 'crisis_training_data.parquet'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET EXPANSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(labeled_data)}")
    logger.info(f"Date range: {labeled_data['date'].min()} to {labeled_data['date'].max()}")
    logger.info(f"Real Black Swan events: {len(BLACK_SWAN_EVENTS)}")
    logger.info(f"Synthetic crises added: 30")

    return 0


if __name__ == '__main__':
    sys.exit(main())
