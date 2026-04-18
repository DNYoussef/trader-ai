"""
TRM Regime Classification Training

Trains TRM to classify market regimes (Bull/Sideways/Bear) instead of
predicting optimal future allocations (which was impossible).

Key differences from strategy training:
- 3 classes instead of 8
- Labels based on CURRENT state (observable) not future returns (unpredictable)
- Strong feature-label signal (r=0.68 for VIX/put_call_ratio)
- Expected to achieve high accuracy and potentially grok

Usage:
    python scripts/train_regime_classifier.py --max_epochs 10000
    python scripts/train_regime_classifier.py --paper --max_epochs 50000
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Optional
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.models.trm_model import TinyRecursiveModel
from src.training.meta_grokfast import (
    MetaGrokFast, TRM_CONFIG, TRM_PAPER_CONFIG, TRM_ENHANCED_CONFIG
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RegimeDataset(Dataset):
    """Dataset for regime classification."""

    def __init__(
        self,
        parquet_path: str,
        split: str = 'train',
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        normalize: bool = True,
        norm_params: Optional[dict] = None
    ):
        """
        Load regime dataset.

        Args:
            parquet_path: Path to regime_labels.parquet
            split: 'train', 'val', or 'test'
            train_frac: Fraction for training
            val_frac: Fraction for validation (rest is test)
            seed: Random seed for reproducible splits
            normalize: Whether to normalize features
            norm_params: Pre-computed normalization parameters
        """
        df = pd.read_parquet(parquet_path)

        # Extract features and labels
        X = np.array(df['features'].tolist(), dtype=np.float32)
        y = df['regime'].values.astype(np.int64)

        # Sort by date for temporal split
        df['date'] = pd.to_datetime(df['date'])
        sort_idx = df['date'].argsort()
        X = X[sort_idx]
        y = y[sort_idx]

        # Split indices
        n = len(X)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == 'train':
            X = X[:n_train]
            y = y[:n_train]
        elif split == 'val':
            X = X[n_train:n_train + n_val]
            y = y[n_train:n_train + n_val]
        else:  # test
            X = X[n_train + n_val:]
            y = y[n_train + n_val:]

        # Normalize features
        if normalize:
            if norm_params is None:
                # Compute from this split (should only be train)
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0) + 1e-8
            else:
                self.mean = np.array(norm_params['mean'])
                self.std = np.array(norm_params['std'])
            X = (X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        self.X = torch.from_numpy(X).float()  # Ensure float32
        self.y = torch.from_numpy(y).long()   # Ensure int64

        logger.info(f"Loaded {split} split: {len(self.X)} samples")
        logger.info(f"  Class distribution: {np.bincount(y)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_norm_params(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        logits = output['strategy_logits']  # Shape: (batch, 3)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        output = model(x)
        logits = output['strategy_logits']
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_acc = {}
    for c in range(3):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc[c] = 100.0 * (all_preds[mask] == c).mean()

    return total_loss / total, 100.0 * correct / total, class_acc


def main():
    parser = argparse.ArgumentParser(description='Train TRM for Regime Classification')
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--paper', action='store_true', help='Use TRM paper config')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced config')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load datasets
    data_path = project_root / 'data' / 'trm_training' / 'regime_labels.parquet'

    train_dataset = RegimeDataset(str(data_path), split='train')
    norm_params = train_dataset.get_norm_params()

    val_dataset = RegimeDataset(str(data_path), split='val', norm_params=norm_params)
    test_dataset = RegimeDataset(str(data_path), split='test', norm_params=norm_params)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model - 3 classes instead of 8!
    model = TinyRecursiveModel(
        input_dim=10,
        hidden_dim=256,  # Smaller for simpler task
        output_dim=3,    # 3 regimes: Bull, Sideways, Bear
        num_latent_steps=4,
        num_recursion_cycles=2,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Select config
    if args.paper:
        config = TRM_PAPER_CONFIG
        config_name = "TRM_PAPER"
    elif args.enhanced:
        config = TRM_ENHANCED_CONFIG
        config_name = "ENHANCED"
    else:
        config = TRM_CONFIG
        config_name = "CONSERVATIVE"

    # Override if specified
    if args.lr is not None:
        config.lr = args.lr
        config.muon_lr = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay

    optimizer = MetaGrokFast(model.parameters(), config=config)
    logger.info(f"Optimizer: {config_name}")
    logger.info(f"  lr={config.lr}, weight_decay={config.weight_decay}, lambda={config.grokfast_lambda}")

    # Class-weighted loss (handle imbalance)
    class_counts = np.bincount(train_dataset.y.numpy())
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    history = []

    print("\n" + "="*80)
    print("REGIME CLASSIFICATION TRAINING")
    print("="*80)
    print("Classes: 0=BULL, 1=SIDEWAYS, 2=BEAR")
    print("Expected: Should learn quickly (strong signal in features)")
    print("="*80 + "\n")

    checkpoint_dir = project_root / 'models' / 'regime_classifier'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, class_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Determine phase
        if val_acc > 90:
            phase = "EXCELLENT"
        elif val_acc > 70:
            phase = "GOOD"
        elif val_acc > 50:
            phase = "LEARNING"
        else:
            phase = "STARTING"

        # Log progress
        if epoch <= 20 or epoch % 100 == 0 or val_acc > best_val_acc:
            gap = train_acc - val_acc
            print(f"Epoch {epoch:5d} [{phase:10s}] | "
                  f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                  f"Gap: {gap:+.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config_name': config_name
            }, checkpoint_dir / 'best_model.pt')

        # Early stopping if excellent performance
        if val_acc > 95 and train_acc > 95:
            print(f"\nExcellent performance achieved at epoch {epoch}!")
            break

        # Save checkpoint every 500 epochs
        if epoch % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    # Final evaluation on test set
    test_loss, test_acc, test_class_acc = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Per-class test accuracy:")
    regime_names = ['BULL', 'SIDEWAYS', 'BEAR']
    for c, acc in test_class_acc.items():
        print(f"  {regime_names[c]}: {acc:.1f}%")
    print("="*80)

    # Save results
    results = {
        'config': config_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_class_acc': test_class_acc,
        'total_epochs': len(history),
        'history': history[-100:]
    }

    with open(checkpoint_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save normalization params
    with open(checkpoint_dir / 'norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    logger.info(f"Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
