"""
Train Regime-Aware TRM with Anticipatory Labels

Key Training Objectives:
1. PREDICT crises BEFORE they happen (not just react)
2. REWARD early defensive positioning
3. LEARN optimal strategy switching timing
4. MULTI-HORIZON prediction (today + 5d + 10d + 20d)

Loss Function:
- Primary: CrossEntropy on anticipatory_strategy
- Secondary: CrossEntropy on future strategies (5d, 10d, 20d)
- Reward: Weighted by normalized_reward (early detection bonus)
- Transition: Extra loss on strategy change points

Usage:
    python scripts/training/train_anticipatory_trm.py --epochs 150
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model components
try:
    from src.models.regime_aware_trm import RegimeAwareTRM, RegimeAwareTRMConfig
    MODEL_AVAILABLE = True
except ImportError:
    logger.error("RegimeAwareTRM not available")
    MODEL_AVAILABLE = False

try:
    from src.training.focal_loss import FocalLoss, compute_class_weights_from_labels
    FOCAL_AVAILABLE = True
except ImportError:
    logger.warning("FocalLoss not available")
    FOCAL_AVAILABLE = False

try:
    from src.training.meta_grokfast import MetaGrokFast
    METAGROKFAST_AVAILABLE = True
except ImportError:
    logger.warning("MetaGrokFast not available")
    METAGROKFAST_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AnticipatoryDataset(Dataset):
    """Dataset with anticipatory labels and multi-horizon targets."""

    def __init__(self, data_path: Path, sequence_length: int = 20):
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas required")

        self.sequence_length = sequence_length

        logger.info(f"Loading anticipatory data from {data_path}")
        df = pd.read_parquet(data_path)

        # Feature columns (from regime data)
        feature_cols = [
            'vix_level', 'spy_return_5d', 'spy_return_20d',
            'spy_volatility_20d', 'spy_tlt_corr_20d'
        ]
        available_cols = [c for c in feature_cols if c in df.columns]

        self.features = df[available_cols].values.astype(np.float32)

        # Pad to 10 features for TRM
        if self.features.shape[1] < 10:
            padding = np.zeros((len(self.features), 10 - self.features.shape[1]), dtype=np.float32)
            self.features = np.concatenate([self.features, padding], axis=1)

        # Main target: anticipatory strategy
        self.targets = df['anticipatory_strategy'].values.astype(np.int64)

        # Multi-horizon targets
        self.target_5d = df['strategy_5d_ahead'].values.astype(np.int64)
        self.target_10d = df['strategy_10d_ahead'].values.astype(np.int64)
        self.target_20d = df['strategy_20d_ahead'].values.astype(np.int64)

        # Early warning labels
        self.early_warning_5d = df['crisis_in_5d'].values.astype(np.float32)
        self.early_warning_10d = df['crisis_in_10d'].values.astype(np.float32)
        self.early_warning_20d = df['crisis_in_20d'].values.astype(np.float32)

        # Reward signal (for RL-style weighting)
        self.rewards = df['normalized_reward'].values.astype(np.float32)

        # Transition labels
        self.is_transition = (df['transition_type'] != 'none').values.astype(np.float32)

        # Create regime feature sequences
        self.regime_features = self._create_sequences(self.features[:, :5])

        logger.info(f"Dataset: {len(self)} samples")
        logger.info(f"Early warnings (5d/10d/20d): {self.early_warning_5d.sum():.0f}/{self.early_warning_10d.sum():.0f}/{self.early_warning_20d.sum():.0f}")

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequential features for regime detector."""
        n_samples = len(features)
        n_features = features.shape[1]
        sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)

        for i in range(n_samples):
            start_idx = max(0, i - self.sequence_length + 1)
            seq_len = i - start_idx + 1
            sequences[i, -seq_len:] = features[start_idx:i+1]
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
            'target_5d': torch.tensor(self.target_5d[idx], dtype=torch.long),
            'target_10d': torch.tensor(self.target_10d[idx], dtype=torch.long),
            'target_20d': torch.tensor(self.target_20d[idx], dtype=torch.long),
            'early_warning_5d': torch.tensor(self.early_warning_5d[idx], dtype=torch.float),
            'early_warning_10d': torch.tensor(self.early_warning_10d[idx], dtype=torch.float),
            'early_warning_20d': torch.tensor(self.early_warning_20d[idx], dtype=torch.float),
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float),
            'is_transition': torch.tensor(self.is_transition[idx], dtype=torch.float),
        }


class AnticipatoryLoss(nn.Module):
    """
    Multi-objective loss for anticipatory training.

    Components:
    1. Strategy loss: Predict correct anticipatory strategy
    2. Horizon losses: Predict future strategies (5d, 10d, 20d)
    3. Early warning loss: Binary classification for crisis detection
    4. Transition loss: Extra weight on strategy change points
    5. Reward weighting: Scale loss by anticipation reward
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        lambda_strategy: float = 1.0,
        lambda_horizon: float = 0.3,
        lambda_early_warning: float = 0.5,
        lambda_transition: float = 0.2,
        gamma: float = 2.0,  # Focal loss gamma
    ):
        super().__init__()
        self.lambda_strategy = lambda_strategy
        self.lambda_horizon = lambda_horizon
        self.lambda_early_warning = lambda_early_warning
        self.lambda_transition = lambda_transition

        # Strategy classification loss
        if FOCAL_AVAILABLE and class_weights is not None:
            self.strategy_loss = FocalLoss(gamma=gamma, alpha=class_weights, reduction='none')
        else:
            self.strategy_loss = nn.CrossEntropyLoss(reduction='none')

        # Horizon prediction losses
        self.horizon_loss = nn.CrossEntropyLoss(reduction='none')

        # Early warning loss (binary)
        self.early_warning_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        strategy_logits: torch.Tensor,
        early_warning_logits: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute anticipatory loss.

        Args:
            strategy_logits: (batch, 8) strategy predictions
            early_warning_logits: (batch, 1) crisis probability
            targets: Dict with target, target_5d/10d/20d, early_warning_*, reward, is_transition

        Returns:
            Total loss
        """
        batch_size = strategy_logits.shape[0]

        # 1. Primary strategy loss
        strategy_loss = self.strategy_loss(strategy_logits, targets['target'])

        # 2. Horizon prediction losses (use same logits - model should predict future)
        loss_5d = self.horizon_loss(strategy_logits, targets['target_5d'])
        loss_10d = self.horizon_loss(strategy_logits, targets['target_10d'])
        loss_20d = self.horizon_loss(strategy_logits, targets['target_20d'])
        horizon_loss = (loss_5d + loss_10d + loss_20d) / 3.0

        # 3. Early warning loss
        # Use 10d window as primary early warning target
        ew_loss = self.early_warning_loss(
            early_warning_logits.squeeze(-1),
            targets['early_warning_10d']
        )

        # 4. Transition weighting: higher loss on transition points
        transition_weight = 1.0 + targets['is_transition'] * 2.0  # 3x weight on transitions

        # 5. Reward weighting: scale by anticipation reward
        # Higher reward = lower loss (we got it right early)
        # Transform reward to loss weight: low reward -> high weight
        reward_weight = 2.0 - targets['reward']  # Reward in [-1, 1], weight in [1, 3]
        reward_weight = torch.clamp(reward_weight, 0.5, 3.0)

        # Combine weights
        sample_weight = transition_weight * reward_weight

        # Weighted loss
        total_loss = (
            self.lambda_strategy * (strategy_loss * sample_weight).mean() +
            self.lambda_horizon * (horizon_loss * sample_weight).mean() +
            self.lambda_early_warning * ew_loss.mean()
        )

        if return_components:
            return {
                'total': total_loss,
                'strategy': strategy_loss.mean(),
                'horizon': horizon_loss.mean(),
                'early_warning': ew_loss.mean(),
                'avg_sample_weight': sample_weight.mean(),
            }

        return total_loss


class AnticipatoryTRMTrainer:
    """Trainer for anticipatory regime-aware TRM."""

    def __init__(
        self,
        model: RegimeAwareTRM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        checkpoint_dir: Optional[Path] = None,
        learning_rate: float = 5e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints/anticipatory_trm')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        if METAGROKFAST_AVAILABLE:
            self.optimizer = MetaGrokFast.for_trm(model.parameters())
            logger.info("Using MetaGrokFast optimizer")
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            logger.info("Using AdamW optimizer")

        # Loss function with class weights
        all_targets = torch.cat([batch['target'] for batch in train_loader])
        if FOCAL_AVAILABLE:
            class_weights = compute_class_weights_from_labels(all_targets)
            self.loss_fn = AnticipatoryLoss(class_weights=class_weights)
            logger.info("Using AnticipatoryLoss with focal loss and class weights")
        else:
            self.loss_fn = AnticipatoryLoss()

        self.best_val_loss = float('inf')
        self.best_val_early_warning_acc = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_strategy_acc': [], 'val_early_warning_acc': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_strategy_loss = 0.0
        total_ew_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            market_features = batch['market_features'].to(self.device)
            regime_features = batch['regime_features'].to(self.device)

            targets = {
                'target': batch['target'].to(self.device),
                'target_5d': batch['target_5d'].to(self.device),
                'target_10d': batch['target_10d'].to(self.device),
                'target_20d': batch['target_20d'].to(self.device),
                'early_warning_5d': batch['early_warning_5d'].to(self.device),
                'early_warning_10d': batch['early_warning_10d'].to(self.device),
                'early_warning_20d': batch['early_warning_20d'].to(self.device),
                'reward': batch['reward'].to(self.device),
                'is_transition': batch['is_transition'].to(self.device),
            }

            self.optimizer.zero_grad()

            out = self.model(market_features, regime_features)

            loss_dict = self.loss_fn(
                out['strategy_logits'],
                out['crisis_prob'],
                targets,
                return_components=True
            )

            loss = loss_dict['total']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_strategy_loss += loss_dict['strategy'].item()
            total_ew_loss += loss_dict['early_warning'].item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'strategy_loss': total_strategy_loss / n_batches,
            'ew_loss': total_ew_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct_strategy = 0
        correct_early_warning = 0
        total = 0
        total_early_warning = 0
        n_batches = 0

        for batch in self.val_loader:
            market_features = batch['market_features'].to(self.device)
            regime_features = batch['regime_features'].to(self.device)

            targets = {
                'target': batch['target'].to(self.device),
                'target_5d': batch['target_5d'].to(self.device),
                'target_10d': batch['target_10d'].to(self.device),
                'target_20d': batch['target_20d'].to(self.device),
                'early_warning_5d': batch['early_warning_5d'].to(self.device),
                'early_warning_10d': batch['early_warning_10d'].to(self.device),
                'early_warning_20d': batch['early_warning_20d'].to(self.device),
                'reward': batch['reward'].to(self.device),
                'is_transition': batch['is_transition'].to(self.device),
            }

            out = self.model(market_features, regime_features)

            loss = self.loss_fn(out['strategy_logits'], out['crisis_prob'], targets)
            total_loss += loss.item()
            n_batches += 1

            # Strategy accuracy
            pred_strategy = out['strategy_logits'].argmax(dim=-1)
            correct_strategy += (pred_strategy == targets['target']).sum().item()
            total += targets['target'].size(0)

            # Early warning accuracy (on samples that have early warning)
            ew_target = targets['early_warning_10d']
            ew_pred = (out['crisis_prob'].squeeze(-1) > 0.3).float()

            # Count correct predictions on positive cases
            positive_mask = ew_target > 0.5
            if positive_mask.sum() > 0:
                correct_early_warning += ((ew_pred == ew_target) & positive_mask).sum().item()
                total_early_warning += positive_mask.sum().item()

        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0,
            'strategy_acc': correct_strategy / total if total > 0 else 0,
            'early_warning_acc': correct_early_warning / total_early_warning if total_early_warning > 0 else 0,
            'early_warning_samples': total_early_warning,
        }

    def train(self, n_epochs: int = 150, log_interval: int = 10) -> Dict[str, List[float]]:
        """Full training loop."""
        logger.info(f"Starting anticipatory training for {n_epochs} epochs")

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_strategy_acc'].append(val_metrics.get('strategy_acc', 0))
            self.history['val_early_warning_acc'].append(val_metrics.get('early_warning_acc', 0))

            if epoch % log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics.get('loss', 0):.4f}, "
                    f"strategy_acc={val_metrics.get('strategy_acc', 0):.3f}, "
                    f"ew_acc={val_metrics.get('early_warning_acc', 0):.3f}"
                )

            # Save best model (by validation loss)
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss.pt', epoch, val_metrics)

            # Save best early warning model
            ew_acc = val_metrics.get('early_warning_acc', 0)
            if ew_acc > self.best_val_early_warning_acc:
                self.best_val_early_warning_acc = ew_acc
                self.save_checkpoint('best_early_warning.pt', epoch, val_metrics)

        # Save final
        self.save_checkpoint('final.pt', n_epochs - 1, val_metrics)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict) -> None:
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.model.config,
        }, path)
        logger.info(f"Saved {filename} (epoch {epoch})")


def main():
    parser = argparse.ArgumentParser(description='Train Anticipatory TRM')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    if not MODEL_AVAILABLE:
        logger.error("Model not available")
        return 1

    # Load anticipatory data
    data_dir = Path('data/training/anticipatory')
    if not (data_dir / 'train_anticipatory.parquet').exists():
        logger.error("Anticipatory data not found. Run create_anticipatory_labels.py first.")
        return 1

    train_dataset = AnticipatoryDataset(data_dir / 'train_anticipatory.parquet')
    val_dataset = AnticipatoryDataset(data_dir / 'val_anticipatory.parquet')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    config = RegimeAwareTRMConfig(
        hidden_dim=args.hidden_dim,
        crisis_threshold=0.3,
        device=args.device
    )
    model = RegimeAwareTRM(config)

    # Train
    trainer = AnticipatoryTRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr
    )

    history = trainer.train(n_epochs=args.epochs, log_interval=10)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ANTICIPATORY TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best val loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best early warning accuracy: {trainer.best_val_early_warning_acc:.3f}")
    logger.info(f"Final strategy accuracy: {history['val_strategy_acc'][-1]:.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
