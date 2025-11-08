"""
TRM Data Loader for Black Swan Strategy Classification

Implements PyTorch Dataset and DataModule for loading, normalizing, and splitting
the TRM training data with stratified splitting by crisis period.

Features:
- Z-score normalization computed on training split only
- Stratified splitting by period_name (70% train, 15% val, 15% test)
- Handles small dataset gracefully with adaptive batch sizing
- Exports normalization parameters for inference
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TRMDataset(Dataset):
    """
    PyTorch Dataset for TRM training data.

    Loads parquet file with black swan labels and applies z-score normalization
    to 10 market features. Normalization parameters are computed only on the
    training split to prevent data leakage.

    Data Format:
        - date: timestamp
        - features: list of 10 floats (vix, spy_returns_5d, spy_returns_20d,
                    volume_ratio, market_breadth, correlation, put_call_ratio,
                    gini_coefficient, sector_dispersion, signal_quality)
        - strategy_idx: int (0-7) - target strategy label
        - pnl: float - realized profit/loss
        - period_name: str - crisis period identifier for stratification

    Args:
        data_path: Path to parquet file with training data
        split: One of 'train', 'val', 'test'
        indices: Indices for this split (from stratified splitting)
        normalization_params: Optional dict with 'mean' and 'std' arrays
                            If None, computes from this dataset (training split)
    """

    def __init__(
        self,
        data_path: Path,
        split: str,
        indices: np.ndarray,
        normalization_params: Optional[Dict[str, np.ndarray]] = None
    ):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"

        self.split = split
        self.data_path = Path(data_path)

        # Load full dataset
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)

        # Subset to this split's indices
        df = df.iloc[indices].reset_index(drop=True)

        # Extract features array from list column
        # features is a list of 10 floats per row
        features_list = df['features'].tolist()
        self.features = np.array(features_list, dtype=np.float32)  # Shape: (n_samples, 10)

        # Extract labels and metadata
        self.strategy_labels = df['strategy_idx'].values.astype(np.int64)
        self.pnl_values = df['pnl'].values.astype(np.float32)
        self.dates = df['date'].values
        self.periods = df['period_name'].values

        # Compute or apply normalization
        if normalization_params is None:
            # Training split: compute normalization parameters
            logger.info(f"Computing normalization parameters on {split} split")
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)

            # Prevent division by zero
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
        else:
            # Val/test splits: use training split's parameters
            logger.info(f"Applying normalization parameters to {split} split")
            self.mean = normalization_params['mean']
            self.std = normalization_params['std']

        # Apply z-score normalization: (x - mean) / std
        self.features = (self.features - self.mean) / self.std

        logger.info(
            f"Initialized {split} dataset: {len(self)} samples, "
            f"{self.features.shape[1]} features"
        )
        logger.info(f"Strategy distribution: {np.bincount(self.strategy_labels)}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Returns:
            features_tensor: (10,) float tensor with normalized features
            strategy_label: int (0-7) target strategy
            pnl_value: float realized profit/loss
        """
        features_tensor = torch.from_numpy(self.features[idx])
        strategy_label = int(self.strategy_labels[idx])
        pnl_value = float(self.pnl_values[idx])

        return features_tensor, strategy_label, pnl_value

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """Return normalization parameters for this dataset."""
        return {
            'mean': self.mean,
            'std': self.std
        }


class TRMDataModule:
    """
    Data module for managing TRM dataset splits and dataloaders.

    Handles:
    - Stratified splitting by crisis period (70/15/15 train/val/test)
    - Normalization parameter computation on training split
    - DataLoader creation with adaptive batch sizing
    - Normalization parameter export for inference

    Args:
        data_path: Path to parquet file with training data
        random_seed: Random seed for reproducible splitting
    """

    def __init__(
        self,
        data_path: Path,
        random_seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.random_seed = random_seed

        # Initialize datasets as None (lazy loading)
        self.train_dataset: Optional[TRMDataset] = None
        self.val_dataset: Optional[TRMDataset] = None
        self.test_dataset: Optional[TRMDataset] = None

        # Perform stratified splitting
        self._create_stratified_splits()

    def _create_stratified_splits(self):
        """
        Create stratified train/val/test splits by period_name.

        Uses StratifiedShuffleSplit to ensure all crisis periods are
        represented proportionally in each split while maintaining
        temporal order within periods. For periods with very few samples
        (< 6), all samples are placed in the training set to avoid
        stratification issues.

        Split ratios: 70% train, 15% val, 15% test
        """
        logger.info("Creating stratified splits by crisis period")

        # Load dataset to get period labels
        df = pd.read_parquet(self.data_path)
        n_samples = len(df)

        # Use period_name as stratification key
        periods = df['period_name'].values

        # Identify periods with too few samples for stratification
        period_counts = pd.Series(periods).value_counts()
        small_periods = period_counts[period_counts < 6].index.tolist()

        if small_periods:
            logger.warning(
                f"Found {len(small_periods)} periods with < 6 samples: {small_periods}. "
                "These will be assigned to training set only."
            )

        # Separate indices for small and large periods
        small_mask = np.isin(periods, small_periods)
        small_indices = np.where(small_mask)[0]
        large_indices = np.where(~small_mask)[0]

        # For large periods, do stratified splitting
        if len(large_indices) > 0:
            large_periods = periods[large_indices]

            # First split: 70% train, 30% temp (val+test)
            splitter_train = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.30,
                random_state=self.random_seed
            )

            train_idx_large, temp_idx_large = next(splitter_train.split(
                np.zeros(len(large_indices)),  # Dummy X
                large_periods
            ))

            # Second split: split temp into 50/50 (15% val, 15% test of total)
            temp_periods = large_periods[temp_idx_large]
            splitter_temp = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.50,
                random_state=self.random_seed
            )

            val_idx_temp, test_idx_temp = next(splitter_temp.split(
                np.zeros(len(temp_idx_large)),
                temp_periods
            ))

            # Map back to original indices
            train_idx = large_indices[train_idx_large]
            val_idx = large_indices[temp_idx_large[val_idx_temp]]
            test_idx = large_indices[temp_idx_large[test_idx_temp]]
        else:
            # No large periods, use empty arrays
            train_idx = np.array([], dtype=int)
            val_idx = np.array([], dtype=int)
            test_idx = np.array([], dtype=int)

        # Add small period indices to training set
        self.train_indices = np.concatenate([train_idx, small_indices])
        self.val_indices = val_idx
        self.test_indices = test_idx

        logger.info(
            f"Split sizes: train={len(self.train_indices)} "
            f"({len(self.train_indices)/n_samples:.1%}), "
            f"val={len(self.val_indices)} "
            f"({len(self.val_indices)/n_samples:.1%}), "
            f"test={len(self.test_indices)} "
            f"({len(self.test_indices)/n_samples:.1%})"
        )

        # Verify stratification by period
        train_periods = pd.Series(periods[self.train_indices]).value_counts()
        val_periods = pd.Series(periods[self.val_indices]).value_counts()
        test_periods = pd.Series(periods[self.test_indices]).value_counts()

        logger.info(f"Train period distribution:\n{train_periods}")
        logger.info(f"Val period distribution:\n{val_periods}")
        logger.info(f"Test period distribution:\n{test_periods}")

    def setup_datasets(self):
        """
        Initialize train/val/test datasets with proper normalization.

        Training dataset computes normalization parameters, val/test
        datasets apply the training parameters.
        """
        if self.train_dataset is not None:
            logger.info("Datasets already initialized")
            return

        logger.info("Setting up datasets with normalization")

        # Create training dataset (computes normalization)
        self.train_dataset = TRMDataset(
            data_path=self.data_path,
            split='train',
            indices=self.train_indices,
            normalization_params=None  # Compute from training data
        )

        # Get normalization parameters from training dataset
        norm_params = self.train_dataset.get_normalization_params()

        # Create val/test datasets (apply training normalization)
        self.val_dataset = TRMDataset(
            data_path=self.data_path,
            split='val',
            indices=self.val_indices,
            normalization_params=norm_params
        )

        self.test_dataset = TRMDataset(
            data_path=self.data_path,
            split='test',
            indices=self.test_indices,
            normalization_params=norm_params
        )

        logger.info("Datasets initialized successfully")

    def create_dataloaders(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders with adaptive batch sizing.

        Args:
            batch_size: Desired batch size (will be reduced if dataset is small)
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes for data loading

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_dataset is None:
            self.setup_datasets()

        # Adaptive batch size: reduce if dataset is smaller than batch_size
        actual_batch_size = min(batch_size, len(self.train_dataset))

        if actual_batch_size < batch_size:
            logger.warning(
                f"Reducing batch size from {batch_size} to {actual_batch_size} "
                f"due to small dataset size ({len(self.train_dataset)} samples)"
            )

        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=actual_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=actual_batch_size,
            shuffle=False,  # Never shuffle val/test
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        logger.info(
            f"Created dataloaders: batch_size={actual_batch_size}, "
            f"num_workers={num_workers}"
        )

        return train_loader, val_loader, test_loader

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """
        Get normalization parameters from training dataset.

        Returns:
            Dict with 'mean' and 'std' arrays (shape: (10,))
        """
        if self.train_dataset is None:
            self.setup_datasets()

        return self.train_dataset.get_normalization_params()

    def save_normalization_params(self, filepath: Path):
        """
        Save normalization parameters to JSON file for inference.

        Args:
            filepath: Output path for JSON file
        """
        norm_params = self.get_normalization_params()

        # Convert numpy arrays to lists for JSON serialization
        params_json = {
            'mean': norm_params['mean'].tolist(),
            'std': norm_params['std'].tolist(),
            'feature_names': [
                'vix',
                'spy_returns_5d',
                'spy_returns_20d',
                'volume_ratio',
                'market_breadth',
                'correlation',
                'put_call_ratio',
                'gini_coefficient',
                'sector_dispersion',
                'signal_quality'
            ]
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(params_json, f, indent=2)

        logger.info(f"Saved normalization parameters to {filepath}")

    @staticmethod
    def load_normalization_params(filepath: Path) -> Dict[str, np.ndarray]:
        """
        Load normalization parameters from JSON file.

        Args:
            filepath: Path to JSON file with normalization parameters

        Returns:
            Dict with 'mean' and 'std' numpy arrays
        """
        with open(filepath, 'r') as f:
            params_json = json.load(f)

        return {
            'mean': np.array(params_json['mean'], dtype=np.float32),
            'std': np.array(params_json['std'], dtype=np.float32)
        }


def main():
    """
    Example usage and validation of TRM data loader.
    """
    import argparse

    parser = argparse.ArgumentParser(description='TRM Data Loader')
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/trm_training/black_swan_labels.parquet',
        help='Path to training data parquet file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for dataloaders'
    )
    parser.add_argument(
        '--save_norm_params',
        type=str,
        default='models/trm_normalization_params.json',
        help='Path to save normalization parameters'
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create data module
    logger.info("Creating TRM DataModule")
    data_module = TRMDataModule(
        data_path=args.data_path,
        random_seed=42
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = data_module.create_dataloaders(
        batch_size=args.batch_size,
        shuffle=True
    )

    # Print dataset statistics
    logger.info("\n" + "="*80)
    logger.info("Dataset Statistics")
    logger.info("="*80)
    logger.info(f"Training samples: {len(data_module.train_dataset)}")
    logger.info(f"Validation samples: {len(data_module.val_dataset)}")
    logger.info(f"Test samples: {len(data_module.test_dataset)}")
    logger.info(f"Total batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # Sample batch
    logger.info("\n" + "="*80)
    logger.info("Sample Batch")
    logger.info("="*80)
    features, labels, pnls = next(iter(train_loader))
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"PNLs shape: {pnls.shape}")
    logger.info(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
    logger.info(f"Labels: {labels.unique()}")

    # Save normalization parameters
    data_module.save_normalization_params(args.save_norm_params)

    # Verify loading
    loaded_params = TRMDataModule.load_normalization_params(args.save_norm_params)
    logger.info("\n" + "="*80)
    logger.info("Normalization Parameters")
    logger.info("="*80)
    logger.info(f"Mean: {loaded_params['mean']}")
    logger.info(f"Std: {loaded_params['std']}")

    logger.info("\n✅ TRM Data Loader validation complete!")


if __name__ == '__main__':
    main()
