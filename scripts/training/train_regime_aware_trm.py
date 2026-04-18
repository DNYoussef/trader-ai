"""
Production Training Script for Regime-Aware TRM

Full training pipeline:
1. Load multi-regime labeled data
2. Initialize RegimeAwareTRM with pretrained base TRM (if available)
3. Train with focal loss and early stopping
4. Save best checkpoint

Usage:
    python scripts/training/train_regime_aware_trm.py --epochs 100 --batch-size 256

Prerequisites:
    1. Run download_multi_regime_data.py to create labeled dataset
    2. (Optional) Pretrained TRM checkpoint for warm start
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing components
try:
    from src.models.regime_aware_trm import RegimeAwareTRM, RegimeAwareTRMConfig
    MODEL_AVAILABLE = True
except ImportError:
    logger.error("RegimeAwareTRM not available")
    MODEL_AVAILABLE = False

try:
    from src.training.focal_loss import TRMFocalLoss, compute_class_weights_from_labels
    FOCAL_LOSS_AVAILABLE = True
except ImportError:
    logger.error("TRMFocalLoss not available")
    FOCAL_LOSS_AVAILABLE = False

try:
    from src.training.meta_grokfast import MetaGrokFast
    METAGROKFAST_AVAILABLE = True
except ImportError:
    logger.warning("MetaGrokFast not available, using AdamW")
    METAGROKFAST_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not available, skipping experiment tracking")
    MLFLOW_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.error("pandas not available")
    PANDAS_AVAILABLE = False


class RegimeAwareDataset(Dataset):
    """Dataset for regime-aware TRM training."""

    def __init__(
        self,
        data_path: Path,
        sequence_length: int = 20,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Args:
            data_path: Path to parquet file with labeled data
            sequence_length: Number of timesteps for regime features
            feature_cols: List of feature column names
        """
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas required for dataset loading")

        self.sequence_length = sequence_length

        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)

        # Default feature columns
        if feature_cols is None:
            feature_cols = [
                'vix_level', 'spy_return_5d', 'spy_return_20d',
                'spy_volatility_20d', 'spy_tlt_corr_20d'
            ]

        # Extract features (will pad to 10 for TRM)
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < 3:
            raise ValueError(f"Not enough feature columns. Available: {df.columns.tolist()}")

        self.features = df[available_cols].values.astype(np.float32)
        self.targets = df['optimal_strategy'].values.astype(np.int64)
        self.regimes = df['regime'].astype('category').cat.codes.values.astype(np.int64)

        # Pad features to 10 dimensions
        if self.features.shape[1] < 10:
            padding = np.zeros((len(self.features), 10 - self.features.shape[1]), dtype=np.float32)
            self.features = np.concatenate([self.features, padding], axis=1)

        # Create regime feature sequences
        self.regime_features = self._create_sequences(self.features[:, :5])

        # Generate synthetic PnL for training (will be replaced with real backtest data)
        np.random.seed(42)
        self.pnl = np.random.normal(0, 0.02, len(self.features)).astype(np.float32)

        logger.info(f"Dataset loaded: {len(self)} samples, {self.features.shape[1]} features")
        logger.info(f"Strategy distribution: {np.bincount(self.targets, minlength=8).tolist()}")

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequential features for regime detector."""
        n_samples = len(features)
        n_features = features.shape[1]
        sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)

        for i in range(n_samples):
            start_idx = max(0, i - self.sequence_length + 1)
            seq_len = i - start_idx + 1
            sequences[i, -seq_len:] = features[start_idx:i+1]
            # Pad beginning with first value if needed
            if seq_len < self.sequence_length:
                sequences[i, :-seq_len] = features[start_idx]

        return sequences

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'market_features': torch.from_numpy(self.features[idx]),
            'regime_features': torch.from_numpy(self.regime_features[idx]),
            'target': torch.tensor(self.targets[idx], dtype=torch.long),
            'pnl': torch.tensor(self.pnl[idx], dtype=torch.float32),
            'regime': torch.tensor(self.regimes[idx], dtype=torch.long),
        }


class TrainingCallback:
    """Callback for training events."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Called at end of each epoch. Return False to stop training."""
        return True

    def on_batch_end(self, batch: int, loss: float) -> None:
        """Called at end of each batch."""
        pass


class EarlyStopping(TrainingCallback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> bool:
        val_loss = metrics.get('val_loss', metrics.get('train_loss', float('inf')))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")

        return self.counter < self.patience


class RegimeAwareTRMTrainer:
    """Trainer for Regime-Aware TRM."""

    def __init__(
        self,
        model: RegimeAwareTRM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cpu',
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints/regime_aware_trm')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = callbacks or []

        # Optimizer
        if optimizer is None:
            if METAGROKFAST_AVAILABLE:
                self.optimizer = MetaGrokFast.for_trm(model.parameters())
                logger.info("Using MetaGrokFast optimizer")
            else:
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                logger.info("Using AdamW optimizer")
        else:
            self.optimizer = optimizer

        # Loss function
        if loss_fn is None:
            if FOCAL_LOSS_AVAILABLE:
                # Compute class weights from training data
                all_targets = torch.cat([batch['target'] for batch in train_loader])
                class_weights = compute_class_weights_from_labels(all_targets)
                self.loss_fn = TRMFocalLoss(gamma=2.0, class_weights=class_weights)
                logger.info("Using TRMFocalLoss with computed class weights")
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                logger.info("Using standard CrossEntropyLoss")
        else:
            self.loss_fn = loss_fn

        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            market_features = batch['market_features'].to(self.device)
            regime_features = batch['regime_features'].to(self.device)
            targets = batch['target'].to(self.device)
            pnl = batch['pnl'].to(self.device)

            self.optimizer.zero_grad()

            out = self.model(market_features, regime_features)

            if FOCAL_LOSS_AVAILABLE:
                loss = self.loss_fn(
                    out['strategy_logits'],
                    out['halt_logits'].squeeze(-1),
                    targets,
                    pnl
                )
            else:
                loss = self.loss_fn(out['strategy_logits'], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item())

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch in self.val_loader:
            market_features = batch['market_features'].to(self.device)
            regime_features = batch['regime_features'].to(self.device)
            targets = batch['target'].to(self.device)
            pnl = batch['pnl'].to(self.device)

            out = self.model(market_features, regime_features)

            if FOCAL_LOSS_AVAILABLE:
                loss = self.loss_fn(
                    out['strategy_logits'],
                    out['halt_logits'].squeeze(-1),
                    targets,
                    pnl
                )
            else:
                loss = self.loss_fn(out['strategy_logits'], targets)

            total_loss += loss.item()
            n_batches += 1

            predictions = out['strategy_logits'].argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        n_epochs: int = 100,
        log_interval: int = 10
    ) -> Dict[str, List[float]]:
        """Train for multiple epochs."""
        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader) if self.val_loader else 0}")

        # MLflow tracking
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("regime_aware_trm")
            mlflow.start_run()
            mlflow.log_params({
                'n_epochs': n_epochs,
                'batch_size': self.train_loader.batch_size,
                'optimizer': type(self.optimizer).__name__,
            })

        for epoch in range(n_epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            # Log metrics
            if epoch % log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.3f}"
                )

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, step=epoch)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pt', epoch, val_loss, val_accuracy)

            # Callbacks
            metrics = {'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy}
            for callback in self.callbacks:
                if not callback.on_epoch_end(epoch, metrics):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Save final model
        self.save_checkpoint('final.pt', n_epochs - 1, val_loss, val_accuracy)

        if MLFLOW_AVAILABLE:
            mlflow.end_run()

        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
        val_accuracy: float
    ) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'config': self.model.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train Regime-Aware TRM')
    parser.add_argument('--data-path', type=str, default='data/regimes/multi_regime_labeled_2010_2024.parquet',
                        help='Path to labeled data parquet')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/regime_aware_trm')
    parser.add_argument('--pretrained-trm', type=str, default=None, help='Path to pretrained TRM')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')

    args = parser.parse_args()

    if not MODEL_AVAILABLE:
        logger.error("RegimeAwareTRM not available. Cannot train.")
        return 1

    # Check data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run: python scripts/data/download_multi_regime_data.py first")
        return 1

    # Load dataset
    dataset = RegimeAwareDataset(data_path)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create model
    config = RegimeAwareTRMConfig(
        hidden_dim=args.hidden_dim,
        device=args.device
    )

    if args.pretrained_trm and Path(args.pretrained_trm).exists():
        model = RegimeAwareTRM.from_pretrained(args.pretrained_trm, config=config)
    else:
        model = RegimeAwareTRM(config)

    # Create trainer
    callbacks = [EarlyStopping(patience=args.patience)]
    trainer = RegimeAwareTRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        callbacks=callbacks
    )

    # Train
    history = trainer.train(n_epochs=args.epochs, log_interval=5)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
