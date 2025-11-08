"""
Standalone tests for TRM data loader (without conftest dependencies).

Run with: python tests/test_trm_data_loader_standalone.py
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.training.trm_data_loader import TRMDataset, TRMDataModule


def create_sample_data(tmp_path, n_samples=200):
    """Create sample training data for testing."""
    np.random.seed(42)
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


def test_dataset_initialization():
    """Test dataset loads and initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

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
        print("✅ Dataset initialization test passed")


def test_dataset_getitem():
    """Test __getitem__ returns correct types and shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

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
        print("✅ Dataset __getitem__ test passed")


def test_normalization():
    """Test normalization parameters are computed and applied correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

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
        print("✅ Normalization test passed")


def test_datamodule_splits():
    """Test data module creates proper splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

        data_module = TRMDataModule(
            data_path=sample_data_path,
            random_seed=42
        )

        # Check splits are created
        assert hasattr(data_module, 'train_indices')
        assert hasattr(data_module, 'val_indices')
        assert hasattr(data_module, 'test_indices')

        # Check split sizes are reasonable
        df = pd.read_parquet(sample_data_path)
        total = len(df)

        assert len(data_module.train_indices) > total * 0.6
        assert len(data_module.val_indices) > total * 0.1
        assert len(data_module.test_indices) > total * 0.1

        # Check no overlap
        all_indices = np.concatenate([
            data_module.train_indices,
            data_module.val_indices,
            data_module.test_indices
        ])
        assert len(all_indices) == len(np.unique(all_indices))
        print("✅ DataModule splits test passed")


def test_dataloaders():
    """Test dataloaders work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

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

        # Check batch
        features, labels, pnls = next(iter(train_loader))
        assert features.shape[0] <= 16
        assert features.shape[1] == 10
        assert labels.shape[0] == features.shape[0]
        assert pnls.shape[0] == features.shape[0]
        print("✅ DataLoaders test passed")


def test_normalization_export():
    """Test normalization parameters can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sample_data_path = create_sample_data(tmp_path)

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
        print("✅ Normalization export test passed")


def test_real_data():
    """Test with real black swan data."""
    data_path = Path('data/trm_training/black_swan_labels.parquet')

    if not data_path.exists():
        print("⚠️  Real data not found, skipping real data test")
        return

    data_module = TRMDataModule(
        data_path=data_path,
        random_seed=42
    )

    train_loader, val_loader, test_loader = data_module.create_dataloaders(
        batch_size=32,
        shuffle=True
    )

    # Check datasets created
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None

    # Check split sizes
    assert len(data_module.train_dataset) > 500
    assert len(data_module.val_dataset) > 100
    assert len(data_module.test_dataset) > 100

    # Check batch
    features, labels, pnls = next(iter(train_loader))
    assert features.shape[1] == 10
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64

    print(f"✅ Real data test passed: {len(data_module.train_dataset)} train, "
          f"{len(data_module.val_dataset)} val, {len(data_module.test_dataset)} test")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Running TRM Data Loader Tests")
    print("="*80 + "\n")

    test_dataset_initialization()
    test_dataset_getitem()
    test_normalization()
    test_datamodule_splits()
    test_dataloaders()
    test_normalization_export()
    test_real_data()

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
