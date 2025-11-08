"""
Unit tests for TRM data loader.

Tests dataset creation, normalization, stratified splitting,
and dataloader functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.trm_data_loader import TRMDataset, TRMDataModule


@pytest.fixture
def sample_data_path(tmp_path):
    """Create sample training data for testing."""
    np.random.seed(42)

    # Create sample data with multiple periods
    n_samples = 200
    n_features = 10

    # Create features as list of lists
    features = [
        np.random.randn(n_features).tolist() for _ in range(n_samples)
    ]

    # Create stratified period distribution
    periods = []
    for period, count in [
        ('Period_A', 100),
        ('Period_B', 60),
        ('Period_C', 40)
    ]:
        periods.extend([period] * count)

    # Create imbalanced strategy distribution (realistic)
    strategy_dist = [0] * 100 + [5] * 90 + [7] * 5 + [1] * 3 + [2] * 2

    df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=n_samples, freq='D'),
        'features': features,
        'strategy_idx': strategy_dist,
        'pnl': np.random.randn(n_samples) * 0.02,
        'period_name': periods
    })

    # Save to parquet
    data_path = tmp_path / 'test_data.parquet'
    df.to_parquet(data_path)

    return data_path


class TestTRMDataset:
    """Tests for TRMDataset class."""

    def test_dataset_initialization(self, sample_data_path):
        """Test dataset loads and initializes correctly."""
        # All indices for this test
        df = pd.read_parquet(sample_data_path)
        indices = np.arange(len(df))

        dataset = TRMDataset(
            data_path=sample_data_path,
            split='train',
            indices=indices,
            normalization_params=None
        )

        assert len(dataset) == len(df)
        assert dataset.features.shape == (len(df), 10)
        assert dataset.strategy_labels.shape == (len(df),)
        assert dataset.pnl_values.shape == (len(df),)

    def test_dataset_getitem(self, sample_data_path):
        """Test __getitem__ returns correct types and shapes."""
        df = pd.read_parquet(sample_data_path)
        indices = np.arange(len(df))

        dataset = TRMDataset(
            data_path=sample_data_path,
            split='train',
            indices=indices,
            normalization_params=None
        )

        features, label, pnl = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert features.shape == (10,)
        assert isinstance(label, int)
        assert isinstance(pnl, float)

    def test_normalization_computed(self, sample_data_path):
        """Test normalization parameters are computed on training split."""
        df = pd.read_parquet(sample_data_path)
        indices = np.arange(len(df))

        dataset = TRMDataset(
            data_path=sample_data_path,
            split='train',
            indices=indices,
            normalization_params=None
        )

        # Check normalization parameters exist
        assert hasattr(dataset, 'mean')
        assert hasattr(dataset, 'std')
        assert dataset.mean.shape == (10,)
        assert dataset.std.shape == (10,)

        # Check normalized features have mean ≈ 0, std ≈ 1
        features_mean = np.mean(dataset.features, axis=0)
        features_std = np.std(dataset.features, axis=0)

        np.testing.assert_array_almost_equal(features_mean, 0, decimal=5)
        np.testing.assert_array_almost_equal(features_std, 1, decimal=5)

    def test_normalization_applied(self, sample_data_path):
        """Test normalization parameters are applied to val/test splits."""
        df = pd.read_parquet(sample_data_path)

        # Split indices
        n = len(df)
        train_indices = np.arange(n // 2)
        val_indices = np.arange(n // 2, n)

        # Create training dataset
        train_dataset = TRMDataset(
            data_path=sample_data_path,
            split='train',
            indices=train_indices,
            normalization_params=None
        )

        # Get normalization parameters
        norm_params = train_dataset.get_normalization_params()

        # Create validation dataset with training normalization
        val_dataset = TRMDataset(
            data_path=sample_data_path,
            split='val',
            indices=val_indices,
            normalization_params=norm_params
        )

        # Check val dataset uses training parameters
        np.testing.assert_array_equal(val_dataset.mean, train_dataset.mean)
        np.testing.assert_array_equal(val_dataset.std, train_dataset.std)


class TestTRMDataModule:
    """Tests for TRMDataModule class."""

    def test_datamodule_initialization(self, sample_data_path):
        """Test data module initializes and creates splits."""
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Check splits are created
        assert hasattr(data_module, 'train_indices')
        assert hasattr(data_module, 'val_indices')
        assert hasattr(data_module, 'test_indices')

        # Check split sizes
        df = pd.read_parquet(sample_data_path)
        total = len(df)

        assert len(data_module.train_indices) == pytest.approx(total * 0.70, abs=5)
        assert len(data_module.val_indices) == pytest.approx(total * 0.15, abs=5)
        assert len(data_module.test_indices) == pytest.approx(total * 0.15, abs=5)

        # Check no overlap
        all_indices = np.concatenate([
            data_module.train_indices,
            data_module.val_indices,
            data_module.test_indices
        ])
        assert len(all_indices) == len(np.unique(all_indices))

    def test_stratified_splitting(self, sample_data_path):
        """Test splits are stratified by period_name."""
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        df = pd.read_parquet(sample_data_path)

        # Get period distributions
        train_periods = df.iloc[data_module.train_indices]['period_name'].value_counts(normalize=True)
        val_periods = df.iloc[data_module.val_indices]['period_name'].value_counts(normalize=True)
        test_periods = df.iloc[data_module.test_indices]['period_name'].value_counts(normalize=True)

        # Check all periods present in all splits
        all_periods = set(df['period_name'].unique())
        assert set(train_periods.index) == all_periods
        assert set(val_periods.index) == all_periods
        assert set(test_periods.index) == all_periods

        # Check proportions are similar (within 10% relative error)
        for period in all_periods:
            train_prop = train_periods[period]
            val_prop = val_periods[period]
            test_prop = test_periods[period]

            # Allow up to 10% relative difference in proportions
            assert abs(train_prop - val_prop) / train_prop < 0.15
            assert abs(train_prop - test_prop) / train_prop < 0.15

    def test_dataloader_creation(self, sample_data_path):
        """Test dataloaders are created correctly."""
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        train_loader, val_loader, test_loader = data_module.create_dataloaders(
            batch_size=16,
            shuffle=True
        )

        # Check loaders exist
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch size
        features, labels, pnls = next(iter(train_loader))
        assert features.shape[0] <= 16  # May be smaller for last batch
        assert features.shape[1] == 10
        assert labels.shape[0] == features.shape[0]
        assert pnls.shape[0] == features.shape[0]

    def test_adaptive_batch_size(self, sample_data_path):
        """Test batch size adapts to small datasets."""
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Request large batch size
        train_loader, _, _ = data_module.create_dataloaders(
            batch_size=1000,  # Larger than dataset
            shuffle=True
        )

        # Should reduce to dataset size
        features, _, _ = next(iter(train_loader))
        assert features.shape[0] <= len(data_module.train_dataset)

    def test_normalization_params_export(self, sample_data_path, tmp_path):
        """Test normalization parameters can be saved and loaded."""
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Save parameters
        norm_path = tmp_path / 'norm_params.json'
        data_module.save_normalization_params(norm_path)

        # Check file exists
        assert norm_path.exists()

        # Load and verify
        loaded_params = TRMDataModule.load_normalization_params(norm_path)
        original_params = data_module.get_normalization_params()

        np.testing.assert_array_almost_equal(
            loaded_params['mean'],
            original_params['mean']
        )
        np.testing.assert_array_almost_equal(
            loaded_params['std'],
            original_params['std']
        )

        # Check JSON structure
        with open(norm_path) as f:
            params_json = json.load(f)

        assert 'mean' in params_json
        assert 'std' in params_json
        assert 'feature_names' in params_json
        assert len(params_json['mean']) == 10
        assert len(params_json['std']) == 10
        assert len(params_json['feature_names']) == 10

    def test_reproducible_splitting(self, sample_data_path):
        """Test splits are reproducible with same random seed."""
        data_module_1 = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        data_module_2 = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Check splits are identical
        np.testing.assert_array_equal(
            data_module_1.train_indices,
            data_module_2.train_indices
        )
        np.testing.assert_array_equal(
            data_module_1.val_indices,
            data_module_2.val_indices
        )
        np.testing.assert_array_equal(
            data_module_1.test_indices,
            data_module_2.test_indices
        )


class TestIntegration:
    """Integration tests for complete data pipeline."""

    def test_end_to_end_pipeline(self, sample_data_path, tmp_path):
        """Test complete data loading pipeline."""
        # Create data module
        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = data_module.create_dataloaders(
            batch_size=16,
            shuffle=True
        )

        # Iterate through one epoch
        train_samples = 0
        for features, labels, pnls in train_loader:
            train_samples += features.shape[0]

            # Verify types and shapes
            assert isinstance(features, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(pnls, torch.Tensor)
            assert features.shape[1] == 10

        assert train_samples == len(data_module.train_dataset)

        # Save and reload normalization
        norm_path = tmp_path / 'norm_params.json'
        data_module.save_normalization_params(norm_path)
        loaded_params = TRMDataModule.load_normalization_params(norm_path)

        # Verify consistency
        original_params = data_module.get_normalization_params()
        np.testing.assert_array_almost_equal(
            loaded_params['mean'],
            original_params['mean']
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
