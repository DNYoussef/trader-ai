"""
Train Crisis Predictor - Phase 1

Target: >80% accuracy at predicting Black Swan events at:
- 30 days before
- 7 days before
- 1 day before

Only proceed to Phase 2 (strategy training with rewards) when
all three horizons achieve >80% accuracy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from src.models.crisis_predictor import (
        CrisisPredictor,
        CrisisPredictorConfig,
        CrisisPredictionLoss,
        compute_crisis_metrics
    )
    CRISIS_PREDICTOR_AVAILABLE = True
except ImportError:
    logger.error("CrisisPredictor not available")
    CRISIS_PREDICTOR_AVAILABLE = False


# Black Swan events with exact dates
BLACK_SWAN_EVENTS = [
    ('2010-05-06', 'Flash Crash'),
    ('2011-08-05', 'US Downgrade'),
    ('2015-08-24', 'China Black Monday'),
    ('2018-02-05', 'Volmageddon'),
    ('2018-10-10', 'Q4 2018 Selloff'),
    ('2020-02-24', 'COVID Crash'),
    ('2023-03-10', 'SVB Collapse'),
]


class CrisisDataset(Dataset):
    """Dataset for crisis prediction training."""

    def __init__(
        self,
        data_path: Path,
        sequence_length: int = 60,
        horizons: List[int] = [30, 7, 1]
    ):
        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas required")

        self.sequence_length = sequence_length
        self.horizons = horizons

        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Feature columns
        feature_cols = [
            'vix_level', 'spy_return_5d', 'spy_return_20d',
            'spy_volatility_20d', 'spy_tlt_corr_20d'
        ]
        available_cols = [c for c in feature_cols if c in df.columns]

        self.features = df[available_cols].values.astype(np.float32)
        self.dates = df['date'].values

        # Pad to 10 features
        if self.features.shape[1] < 10:
            padding = np.zeros((len(self.features), 10 - self.features.shape[1]), dtype=np.float32)
            self.features = np.concatenate([self.features, padding], axis=1)

        # Create crisis labels for each horizon
        self.crisis_labels = self._create_crisis_labels(df)

        # Create sequences
        self.valid_indices = list(range(sequence_length, len(self.features) - max(horizons)))

        logger.info(f"Dataset: {len(self.valid_indices)} valid samples")
        for h in horizons:
            n_crisis = self.crisis_labels[h].sum()
            logger.info(f"  Crisis in {h}d: {n_crisis} ({100*n_crisis/len(self.crisis_labels[h]):.2f}%)")

    def _create_crisis_labels(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Create binary crisis labels for each horizon."""
        n_samples = len(df)
        labels = {h: np.zeros(n_samples, dtype=np.float32) for h in self.horizons}

        dates = pd.to_datetime(df['date'])

        for crisis_date_str, event_name in BLACK_SWAN_EVENTS:
            crisis_date = pd.to_datetime(crisis_date_str)

            for horizon in self.horizons:
                # Mark days where crisis is within horizon
                warning_start = crisis_date - timedelta(days=horizon)

                mask = (dates >= warning_start) & (dates < crisis_date)
                labels[horizon][mask.values] = 1.0

                n_marked = mask.sum()
                if n_marked > 0:
                    logger.debug(f"Marked {n_marked} samples for {event_name} at {horizon}d horizon")

        return labels

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        # Get sequence ending at actual_idx
        start_idx = actual_idx - self.sequence_length
        features_seq = self.features[start_idx:actual_idx]

        # Get crisis labels
        crisis_labels = {
            f'crisis_{h}d': torch.tensor(self.crisis_labels[h][actual_idx], dtype=torch.float32)
            for h in self.horizons
        }

        return {
            'features': torch.from_numpy(features_seq),
            'sample_idx': torch.tensor(actual_idx, dtype=torch.long),
            **crisis_labels
        }


class CrisisPredictorTrainer:
    """Trainer for crisis prediction model."""

    def __init__(
        self,
        model: CrisisPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints/crisis_predictor')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.loss_fn = CrisisPredictionLoss()

        self.best_accuracy = {30: 0.0, 7: 0.0, 1: 0.0}
        self.history = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            targets = {
                'crisis_30d': batch['crisis_30d'].to(self.device),
                'crisis_7d': batch['crisis_7d'].to(self.device),
                'crisis_1d': batch['crisis_1d'].to(self.device),
            }

            self.optimizer.zero_grad()

            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'train_loss': total_loss / n_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate and compute metrics."""
        self.model.eval()

        all_predictions = {'prob_30d': [], 'prob_7d': [], 'prob_1d': []}
        all_targets = {'crisis_30d': [], 'crisis_7d': [], 'crisis_1d': []}
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            targets = {
                'crisis_30d': batch['crisis_30d'].to(self.device),
                'crisis_7d': batch['crisis_7d'].to(self.device),
                'crisis_1d': batch['crisis_1d'].to(self.device),
            }

            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            for key in all_predictions:
                all_predictions[key].append(predictions[key].cpu())
            for key in all_targets:
                all_targets[key].append(targets[key].cpu())

        # Concatenate all predictions
        all_predictions = {k: torch.cat(v) for k, v in all_predictions.items()}
        all_targets = {k: torch.cat(v) for k, v in all_targets.items()}

        # Compute metrics
        metrics = compute_crisis_metrics(all_predictions, all_targets)
        metrics['val_loss'] = total_loss / n_batches

        return metrics

    def train(
        self,
        n_epochs: int = 200,
        target_accuracy: float = 0.80,
        log_interval: int = 10
    ) -> Tuple[bool, Dict]:
        """
        Train until >80% accuracy at all horizons or max epochs.

        Returns:
            (success, final_metrics)
        """
        logger.info(f"Training crisis predictor for up to {n_epochs} epochs")
        logger.info(f"Target accuracy: {target_accuracy*100:.0f}% at all horizons")

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step(val_metrics['val_loss'])
            self.history.append({**train_metrics, **val_metrics, 'epoch': epoch})

            # Check accuracy at each horizon
            acc_30d = val_metrics['accuracy_30d']
            acc_7d = val_metrics['accuracy_7d']
            acc_1d = val_metrics['accuracy_1d']

            # Update best
            self.best_accuracy[30] = max(self.best_accuracy[30], acc_30d)
            self.best_accuracy[7] = max(self.best_accuracy[7], acc_7d)
            self.best_accuracy[1] = max(self.best_accuracy[1], acc_1d)

            if epoch % log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: loss={val_metrics['val_loss']:.4f} | "
                    f"Acc 30d={acc_30d*100:.1f}% 7d={acc_7d*100:.1f}% 1d={acc_1d*100:.1f}%"
                )

            # Check recall (catching crises matters more than accuracy)
            recall_30d = val_metrics.get('recall_30d', 0)
            recall_7d = val_metrics.get('recall_7d', 0)
            recall_1d = val_metrics.get('recall_1d', 0)

            # F1 score (balance precision and recall)
            f1_30d = val_metrics.get('f1_30d', 0)
            f1_7d = val_metrics.get('f1_7d', 0)
            f1_1d = val_metrics.get('f1_1d', 0)

            if epoch % log_interval == 0:
                logger.info(
                    f"         Recall 30d={recall_30d*100:.1f}% 7d={recall_7d*100:.1f}% 1d={recall_1d*100:.1f}%"
                )
                logger.info(
                    f"         F1     30d={f1_30d*100:.1f}% 7d={f1_7d*100:.1f}% 1d={f1_1d*100:.1f}%"
                )

            # Target: RECALL >80% (catching crises) not just accuracy
            if recall_30d >= target_accuracy and recall_7d >= target_accuracy and recall_1d >= target_accuracy:
                logger.info(f"\n{'='*60}")
                logger.info(f"TARGET ACHIEVED at epoch {epoch}!")
                logger.info(f"RECALL 30d: {recall_30d*100:.1f}% | 7d: {recall_7d*100:.1f}% | 1d: {recall_1d*100:.1f}%")
                logger.info(f"{'='*60}\n")

                self.save_checkpoint('target_achieved.pt', epoch, val_metrics)
                return True, val_metrics

            # Save best model
            if acc_30d >= self.best_accuracy[30]:
                self.save_checkpoint('best.pt', epoch, val_metrics)

        # Training complete without hitting target
        logger.warning(f"\nMax epochs reached. Best accuracies:")
        logger.warning(f"30d: {self.best_accuracy[30]*100:.1f}%")
        logger.warning(f"7d: {self.best_accuracy[7]*100:.1f}%")
        logger.warning(f"1d: {self.best_accuracy[1]*100:.1f}%")

        return False, val_metrics

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
        logger.info(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train Crisis Predictor')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--target-accuracy', type=float, default=0.80)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-path', type=str, default='data/training/merged_regime_blackswan.parquet')

    args = parser.parse_args()

    if not CRISIS_PREDICTOR_AVAILABLE:
        logger.error("CrisisPredictor not available")
        return 1

    # Check data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return 1

    # Create datasets
    full_dataset = CrisisDataset(data_path, sequence_length=60)

    # Split 80/20
    n_train = int(len(full_dataset) * 0.8)
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train: {n_train}, Val: {n_val}")

    # Create model
    config = CrisisPredictorConfig(
        input_dim=10,
        hidden_dim=256,
        sequence_length=60,
        device=args.device
    )
    model = CrisisPredictor(config)

    # Train
    trainer = CrisisPredictorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr
    )

    success, final_metrics = trainer.train(
        n_epochs=args.epochs,
        target_accuracy=args.target_accuracy
    )

    # Summary
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("PHASE 1 COMPLETE - Ready for Phase 2 (Strategy Training)")
    else:
        logger.info("PHASE 1 INCOMPLETE - Need more training or data")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
