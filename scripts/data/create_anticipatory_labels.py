"""
Create Anticipatory Labels for Black Swan Detection

The key insight: We want the model to see signals BEFORE a crisis,
not just react AFTER it happens.

This script creates:
1. Lookahead labels - "In N days, what regime will we be in?"
2. Early warning signals - "Is a crisis coming in the next 5/10/20 days?"
3. Strategy transition labels - "Should we be switching strategies now?"
4. Reward signals that value EARLY correct positioning

Example: If COVID crash happens on March 15, 2020:
- March 5 (10 days before): Label as "crisis_imminent", strategy=defensive
- March 1 (14 days before): Label as "warning", strategy=balanced_safe
- Feb 20 (23 days before): Label as "normal" but early_warning_10d=True

The model learns to recognize the PRECURSORS, not just the event.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Black Swan events with EXACT crisis start dates
BLACK_SWAN_EVENTS = {
    'flash_crash_2010': {
        'crisis_start': '2010-05-06',
        'crisis_end': '2010-05-20',
        'severity': 0.9,  # Flash crash severity
    },
    'us_downgrade_2011': {
        'crisis_start': '2011-08-05',
        'crisis_end': '2011-08-22',
        'severity': 0.7,
    },
    'china_deval_2015': {
        'crisis_start': '2015-08-24',  # "Black Monday"
        'crisis_end': '2015-09-30',
        'severity': 0.8,
    },
    'volmageddon_2018': {
        'crisis_start': '2018-02-05',
        'crisis_end': '2018-02-12',
        'severity': 0.85,
    },
    'q4_selloff_2018': {
        'crisis_start': '2018-10-10',
        'crisis_end': '2018-12-24',
        'severity': 0.75,
    },
    'covid_crash_2020': {
        'crisis_start': '2020-02-24',
        'crisis_end': '2020-03-23',
        'severity': 1.0,  # Maximum severity
    },
    'svb_collapse_2023': {
        'crisis_start': '2023-03-10',
        'crisis_end': '2023-03-15',
        'severity': 0.6,
    },
}

# Lookahead windows for early warning
LOOKAHEAD_WINDOWS = [5, 10, 20]  # Days to look ahead

# Strategy mapping for anticipatory positioning
ANTICIPATORY_STRATEGY_MAP = {
    'crisis_imminent': 0,      # ultra_defensive (5 days or less before)
    'crisis_warning': 1,       # defensive (6-10 days before)
    'elevated_risk': 2,        # balanced_safe (11-20 days before)
    'recovery_early': 6,       # contrarian_long (first 5 days after crisis)
    'recovery_mid': 7,         # tactical_opportunity (6-15 days after)
}


class AnticipatoryLabeler:
    """Creates training labels that reward early crisis detection."""

    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        self.input_path = input_path or Path('data/training/merged_regime_blackswan.parquet')
        self.output_dir = output_dir or Path('data/training/anticipatory')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load merged regime data."""
        if not self.input_path.exists():
            logger.error(f"Input data not found: {self.input_path}")
            return pd.DataFrame()

        df = pd.read_parquet(self.input_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} samples")
        return df

    def add_lookahead_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lookahead labels for each Black Swan event.

        For each day, we look FORWARD to see if a crisis is coming.
        """
        df = df.copy()

        # Initialize lookahead columns
        for window in LOOKAHEAD_WINDOWS:
            df[f'crisis_in_{window}d'] = False
            df[f'days_to_crisis_{window}d'] = -1  # -1 means no crisis in window

        df['nearest_crisis_days'] = -1
        df['nearest_crisis_name'] = ''
        df['crisis_severity'] = 0.0

        for event_name, event_info in BLACK_SWAN_EVENTS.items():
            crisis_start = pd.to_datetime(event_info['crisis_start'])
            crisis_end = pd.to_datetime(event_info['crisis_end'])
            severity = event_info['severity']

            # For each lookahead window
            for window in LOOKAHEAD_WINDOWS:
                # Mark days where crisis is within window
                warning_start = crisis_start - timedelta(days=window)

                mask = (df['date'] >= warning_start) & (df['date'] < crisis_start)
                df.loc[mask, f'crisis_in_{window}d'] = True

                # Calculate exact days to crisis
                for idx in df[mask].index:
                    days_to = (crisis_start - df.loc[idx, 'date']).days
                    current_val = df.loc[idx, f'days_to_crisis_{window}d']
                    if current_val == -1 or days_to < current_val:
                        df.loc[idx, f'days_to_crisis_{window}d'] = days_to

            # Track nearest crisis for all pre-crisis days
            max_lookback = 30  # Look up to 30 days before
            warning_start = crisis_start - timedelta(days=max_lookback)

            pre_crisis_mask = (df['date'] >= warning_start) & (df['date'] < crisis_start)
            for idx in df[pre_crisis_mask].index:
                days_to = (crisis_start - df.loc[idx, 'date']).days
                current_nearest = df.loc[idx, 'nearest_crisis_days']

                if current_nearest == -1 or days_to < current_nearest:
                    df.loc[idx, 'nearest_crisis_days'] = days_to
                    df.loc[idx, 'nearest_crisis_name'] = event_name
                    df.loc[idx, 'crisis_severity'] = severity

            # Also mark the crisis period itself
            crisis_mask = (df['date'] >= crisis_start) & (df['date'] <= crisis_end)
            df.loc[crisis_mask, 'nearest_crisis_days'] = 0
            df.loc[crisis_mask, 'nearest_crisis_name'] = event_name
            df.loc[crisis_mask, 'crisis_severity'] = severity

        # Log statistics
        for window in LOOKAHEAD_WINDOWS:
            n_warnings = df[f'crisis_in_{window}d'].sum()
            logger.info(f"Samples with crisis in {window}d: {n_warnings}")

        return df

    def create_anticipatory_strategy_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create strategy labels that ANTICIPATE crises.

        Instead of labeling based on current regime, label based on
        what strategy SHOULD have been used given future knowledge.
        """
        df = df.copy()

        # Initialize anticipatory strategy (default to current optimal)
        df['anticipatory_strategy'] = df['optimal_strategy']
        df['anticipatory_reason'] = 'normal'

        # Crisis imminent (0-5 days before) -> ultra_defensive
        imminent_mask = (df['nearest_crisis_days'] > 0) & (df['nearest_crisis_days'] <= 5)
        df.loc[imminent_mask, 'anticipatory_strategy'] = ANTICIPATORY_STRATEGY_MAP['crisis_imminent']
        df.loc[imminent_mask, 'anticipatory_reason'] = 'crisis_imminent'
        logger.info(f"Crisis imminent (0-5d): {imminent_mask.sum()} samples -> ultra_defensive")

        # Crisis warning (6-10 days before) -> defensive
        warning_mask = (df['nearest_crisis_days'] > 5) & (df['nearest_crisis_days'] <= 10)
        df.loc[warning_mask, 'anticipatory_strategy'] = ANTICIPATORY_STRATEGY_MAP['crisis_warning']
        df.loc[warning_mask, 'anticipatory_reason'] = 'crisis_warning'
        logger.info(f"Crisis warning (6-10d): {warning_mask.sum()} samples -> defensive")

        # Elevated risk (11-20 days before) -> balanced_safe
        elevated_mask = (df['nearest_crisis_days'] > 10) & (df['nearest_crisis_days'] <= 20)
        df.loc[elevated_mask, 'anticipatory_strategy'] = ANTICIPATORY_STRATEGY_MAP['elevated_risk']
        df.loc[elevated_mask, 'anticipatory_reason'] = 'elevated_risk'
        logger.info(f"Elevated risk (11-20d): {elevated_mask.sum()} samples -> balanced_safe")

        # During crisis (days_to = 0) -> ultra_defensive
        crisis_mask = df['nearest_crisis_days'] == 0
        df.loc[crisis_mask, 'anticipatory_strategy'] = 0
        df.loc[crisis_mask, 'anticipatory_reason'] = 'in_crisis'
        logger.info(f"In crisis: {crisis_mask.sum()} samples -> ultra_defensive")

        return df

    def create_transition_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for when strategy transitions should happen.

        We want the model to learn:
        - WHEN to switch from growth -> defensive
        - WHEN to switch from defensive -> growth (recovery)
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Detect strategy transitions
        df['prev_strategy'] = df['anticipatory_strategy'].shift(1)
        df['strategy_changed'] = df['anticipatory_strategy'] != df['prev_strategy']
        df['strategy_changed'] = df['strategy_changed'].fillna(False)

        # Categorize transitions
        df['transition_type'] = 'none'

        # Defensive transition (moving to lower-risk strategy)
        defensive_transition = (
            df['strategy_changed'] &
            (df['anticipatory_strategy'] < df['prev_strategy'])
        )
        df.loc[defensive_transition, 'transition_type'] = 'go_defensive'

        # Aggressive transition (moving to higher-risk strategy)
        aggressive_transition = (
            df['strategy_changed'] &
            (df['anticipatory_strategy'] > df['prev_strategy'])
        )
        df.loc[aggressive_transition, 'transition_type'] = 'go_aggressive'

        # Log transition statistics
        transition_counts = df['transition_type'].value_counts()
        logger.info(f"\nTransition types:\n{transition_counts}")

        return df

    def create_reward_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a reward signal that values EARLY correct positioning.

        Reward structure:
        - Correct defensive position before crisis: HIGH reward
        - Earlier positioning: HIGHER reward (bonus for anticipation)
        - Wrong position during crisis: PENALTY
        - Correct growth position in bull market: MODERATE reward
        """
        df = df.copy()

        # Base reward from correct strategy
        df['base_reward'] = 0.0

        # Anticipation bonus: reward for being defensive BEFORE crisis
        # Scale by days before crisis (more days = more bonus)
        pre_crisis_mask = (df['nearest_crisis_days'] > 0) & (df['nearest_crisis_days'] <= 20)
        is_defensive = df['anticipatory_strategy'].isin([0, 1, 2])

        # Correct early positioning gets bonus scaled by how early
        early_correct = pre_crisis_mask & is_defensive
        df.loc[early_correct, 'base_reward'] = (
            df.loc[early_correct, 'nearest_crisis_days'] / 5.0 *
            df.loc[early_correct, 'crisis_severity']
        )  # Up to 4x base reward for 20 days early on severe crisis

        # During crisis: defensive = reward, aggressive = penalty
        in_crisis = df['nearest_crisis_days'] == 0
        df.loc[in_crisis & is_defensive, 'base_reward'] = 2.0 * df.loc[in_crisis & is_defensive, 'crisis_severity']
        df.loc[in_crisis & ~is_defensive, 'base_reward'] = -3.0 * df.loc[in_crisis & ~is_defensive, 'crisis_severity']

        # Normal times: growth strategies get moderate reward
        normal_times = df['nearest_crisis_days'] == -1
        is_growth = df['anticipatory_strategy'].isin([4, 5])
        df.loc[normal_times & is_growth, 'base_reward'] = 0.5

        # Transition bonus: correct strategy CHANGE gets extra reward
        correct_defensive_transition = (
            (df['transition_type'] == 'go_defensive') &
            (df['nearest_crisis_days'] > 0) &
            (df['nearest_crisis_days'] <= 20)
        )
        df.loc[correct_defensive_transition, 'base_reward'] += 1.0

        # Normalize rewards to [-1, 1] range for training stability
        max_reward = df['base_reward'].abs().max()
        if max_reward > 0:
            df['normalized_reward'] = df['base_reward'] / max_reward
        else:
            df['normalized_reward'] = 0.0

        # Log reward statistics
        logger.info(f"\nReward statistics:")
        logger.info(f"  Min: {df['base_reward'].min():.3f}")
        logger.info(f"  Max: {df['base_reward'].max():.3f}")
        logger.info(f"  Mean: {df['base_reward'].mean():.3f}")
        logger.info(f"  Positive rewards: {(df['base_reward'] > 0).sum()}")
        logger.info(f"  Negative rewards: {(df['base_reward'] < 0).sum()}")

        return df

    def create_multi_horizon_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels for multiple prediction horizons.

        The model should predict:
        - Optimal strategy for TODAY
        - Optimal strategy for 5 DAYS from now
        - Optimal strategy for 10 DAYS from now

        This teaches temporal awareness.
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Create future strategy labels
        for horizon in [5, 10, 20]:
            col_name = f'strategy_{horizon}d_ahead'
            df[col_name] = df['anticipatory_strategy'].shift(-horizon)
            df[col_name] = df[col_name].fillna(df['anticipatory_strategy'])
            df[col_name] = df[col_name].astype(int)

        return df

    def save_anticipatory_data(self, df: pd.DataFrame) -> None:
        """Save anticipatory-labeled dataset."""
        # Save full dataset
        full_path = self.output_dir / 'anticipatory_labeled.parquet'
        df.to_parquet(full_path)
        logger.info(f"Saved full dataset to {full_path}")

        # Create time-based splits
        df_sorted = df.sort_values('date').reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        train_df.to_parquet(self.output_dir / 'train_anticipatory.parquet')
        val_df.to_parquet(self.output_dir / 'val_anticipatory.parquet')
        test_df.to_parquet(self.output_dir / 'test_anticipatory.parquet')

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save metadata
        import json
        metadata = {
            'total_samples': len(df),
            'lookahead_windows': LOOKAHEAD_WINDOWS,
            'black_swan_events': list(BLACK_SWAN_EVENTS.keys()),
            'strategy_distribution': df['anticipatory_strategy'].value_counts().to_dict(),
            'transition_distribution': df['transition_type'].value_counts().to_dict(),
            'reward_stats': {
                'min': float(df['base_reward'].min()),
                'max': float(df['base_reward'].max()),
                'mean': float(df['base_reward'].mean()),
            },
            'created': datetime.now().isoformat(),
        }

        with open(self.output_dir / 'anticipatory_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def run(self) -> pd.DataFrame:
        """Run full anticipatory labeling pipeline."""
        logger.info("=" * 60)
        logger.info("Creating Anticipatory Training Labels")
        logger.info("=" * 60)

        # Load data
        df = self.load_data()
        if df.empty:
            return df

        # Add lookahead labels
        logger.info("\n[1/5] Adding lookahead labels...")
        df = self.add_lookahead_labels(df)

        # Create anticipatory strategy labels
        logger.info("\n[2/5] Creating anticipatory strategy labels...")
        df = self.create_anticipatory_strategy_labels(df)

        # Create transition labels
        logger.info("\n[3/5] Creating transition labels...")
        df = self.create_transition_labels(df)

        # Create reward signal
        logger.info("\n[4/5] Creating reward signal...")
        df = self.create_reward_signal(df)

        # Create multi-horizon labels
        logger.info("\n[5/5] Creating multi-horizon labels...")
        df = self.create_multi_horizon_labels(df)

        # Save
        self.save_anticipatory_data(df)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("ANTICIPATORY LABELING COMPLETE")
        logger.info("=" * 60)

        strategy_names = [
            'ultra_defensive', 'defensive', 'balanced_safe', 'balanced_growth',
            'growth', 'aggressive_growth', 'contrarian_long', 'tactical_opportunity'
        ]
        logger.info("\nAnticipatory Strategy Distribution:")
        for idx in range(8):
            count = (df['anticipatory_strategy'] == idx).sum()
            logger.info(f"  {idx} ({strategy_names[idx]}): {count} ({100*count/len(df):.1f}%)")

        return df


def main():
    labeler = AnticipatoryLabeler()
    df = labeler.run()

    if not df.empty:
        logger.info(f"\nTotal samples: {len(df)}")
        logger.info(f"Samples with early warning (5d): {df['crisis_in_5d'].sum()}")
        logger.info(f"Samples with early warning (10d): {df['crisis_in_10d'].sum()}")
        logger.info(f"Samples with early warning (20d): {df['crisis_in_20d'].sum()}")


if __name__ == '__main__':
    main()
