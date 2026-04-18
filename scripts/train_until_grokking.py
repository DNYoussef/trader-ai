"""
TRM Grokking-Aware Training with MetaGrokFast

Runs training until grokking signals are detected:
1. MEMORIZATION PHASE: Train acc >> Val acc (generalization gap)
2. PLATEAU PHASE: Val loss stagnates for extended period
3. GROKKING PHASE: Sudden val loss drop + gap closing

Uses MetaGrokFast optimizer combining:
- GrokFast: EMA gradient filtering (time-spectrum)
- Muon: Newton-Schulz orthogonalization (space-geometry)
- Bigeometric: g * |g|^(2k-1) transform (scale-adaptive)
- k(L): -0.0137*log10(L)+0.1593 (MOO-verified)

Usage:
    python scripts/train_until_grokking.py --data_path data/trm_training/black_swan_labels.parquet
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import argparse

# TRM imports
from src.models.trm_model import TinyRecursiveModel
from src.training.trm_data_loader import TRMDataModule
from src.training.meta_grokfast import (
    MetaGrokFast, TRM_CONFIG, TRM_AGGRESSIVE_CONFIG,
    TRM_PAPER_CONFIG, TRM_ENHANCED_CONFIG, MetaGrokfastConfig
)

# Rich loss functions
try:
    from src.training.trm_loss_functions import NNCTRMLoss, TRMLoss as RichTRMLoss
    RICH_LOSS_AVAILABLE = True
except ImportError:
    RICH_LOSS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GrokingMetrics:
    """Tracks metrics for grokking detection."""
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_top3_acc: float = 0.0
    generalization_gap: float = 0.0  # train_acc - val_acc
    val_loss_ema: float = 0.0        # Smoothed val loss for plateau detection
    grok_score: float = 0.0          # Composite grokking signal


@dataclass
class GrokingState:
    """State machine for grokking detection."""
    phase: str = "warmup"  # warmup -> memorization -> plateau -> grokking -> converged

    # Plateau detection
    plateau_start_epoch: int = -1
    plateau_val_loss: float = float('inf')
    epochs_in_plateau: int = 0

    # Grokking detection
    pre_grok_val_loss: float = float('inf')
    grok_detected_epoch: int = -1
    grok_magnitude: float = 0.0

    # Best metrics
    best_val_acc: float = 0.0
    best_val_loss: float = float('inf')
    best_epoch: int = 0

    # History for analysis
    metrics_history: List[GrokingMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            'phase': self.phase,
            'plateau_start_epoch': self.plateau_start_epoch,
            'epochs_in_plateau': self.epochs_in_plateau,
            'grok_detected_epoch': self.grok_detected_epoch,
            'grok_magnitude': self.grok_magnitude,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }


class GrokingDetector:
    """
    Detects grokking signals during training.

    Grokking is characterized by:
    1. Extended plateau in validation loss
    2. Sudden drop in validation loss (phase transition)
    3. Closing of train/val accuracy gap

    Detection thresholds:
    - PLATEAU: val_loss improvement < 1% for >= plateau_patience epochs
    - GROKKING: val_loss drop > grok_threshold after plateau
    - CONVERGED: val_acc > convergence_acc OR epochs > max_epochs
    """

    def __init__(
        self,
        plateau_patience: int = 15,      # Epochs of stagnation before plateau
        plateau_threshold: float = 0.01, # Min improvement to exit plateau (1%)
        grok_threshold: float = 0.10,    # Min drop to detect grokking (10%)
        convergence_acc: float = 85.0,   # Val acc to declare convergence
        max_epochs: int = 1000,          # Hard limit
        ema_alpha: float = 0.9,          # EMA smoothing for val loss
        warmup_epochs: int = 5,          # Initial warmup before detection
    ):
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.grok_threshold = grok_threshold
        self.convergence_acc = convergence_acc
        self.max_epochs = max_epochs
        self.ema_alpha = ema_alpha
        self.warmup_epochs = warmup_epochs

        self.state = GrokingState()
        self.val_loss_history = deque(maxlen=50)
        self.val_loss_ema = None

    def update(self, metrics: GrokingMetrics) -> Tuple[str, bool]:
        """
        Update state based on new metrics.

        Returns:
            Tuple of (phase_name, should_stop)
        """
        self.state.metrics_history.append(metrics)
        self.val_loss_history.append(metrics.val_loss)

        # Update EMA
        if self.val_loss_ema is None:
            self.val_loss_ema = metrics.val_loss
        else:
            self.val_loss_ema = self.ema_alpha * self.val_loss_ema + (1 - self.ema_alpha) * metrics.val_loss

        metrics.val_loss_ema = self.val_loss_ema
        metrics.generalization_gap = metrics.train_acc - metrics.val_acc

        # Track best
        if metrics.val_acc > self.state.best_val_acc:
            self.state.best_val_acc = metrics.val_acc
            self.state.best_epoch = metrics.epoch
        if metrics.val_loss < self.state.best_val_loss:
            self.state.best_val_loss = metrics.val_loss

        # State machine transitions
        should_stop = False

        if metrics.epoch < self.warmup_epochs:
            self.state.phase = "warmup"

        elif self.state.phase in ("warmup", "memorization"):
            # Check for memorization (train >> val)
            if metrics.generalization_gap > 20.0:
                self.state.phase = "memorization"

            # Check for plateau entry
            if len(self.val_loss_history) >= 5:
                recent_std = np.std(list(self.val_loss_history)[-10:])
                recent_mean = np.mean(list(self.val_loss_history)[-10:])

                if recent_std / (recent_mean + 1e-8) < self.plateau_threshold:
                    self.state.phase = "plateau"
                    self.state.plateau_start_epoch = metrics.epoch
                    self.state.plateau_val_loss = metrics.val_loss
                    self.state.epochs_in_plateau = 0
                    logger.info(f"[PLATEAU DETECTED] Epoch {metrics.epoch}, Val Loss={metrics.val_loss:.4f}")

        elif self.state.phase == "plateau":
            self.state.epochs_in_plateau = metrics.epoch - self.state.plateau_start_epoch

            # Check for grokking (sudden improvement)
            improvement = (self.state.plateau_val_loss - metrics.val_loss) / (self.state.plateau_val_loss + 1e-8)

            if improvement > self.grok_threshold:
                self.state.phase = "grokking"
                self.state.grok_detected_epoch = metrics.epoch
                self.state.grok_magnitude = improvement
                self.state.pre_grok_val_loss = self.state.plateau_val_loss
                logger.info(f"[GROKKING DETECTED] Epoch {metrics.epoch}")
                logger.info(f"  Val Loss: {self.state.plateau_val_loss:.4f} -> {metrics.val_loss:.4f} ({improvement*100:.1f}% drop)")
                logger.info(f"  Plateau duration: {self.state.epochs_in_plateau} epochs")

        elif self.state.phase == "grokking":
            # Check for convergence after grokking
            if metrics.val_acc >= self.convergence_acc:
                self.state.phase = "converged"
                should_stop = True
                logger.info(f"[CONVERGED] Val Acc={metrics.val_acc:.2f}% reached target {self.convergence_acc}%")
            elif metrics.generalization_gap < 5.0:  # Gap closed
                self.state.phase = "converged"
                should_stop = True
                logger.info(f"[CONVERGED] Generalization gap closed to {metrics.generalization_gap:.1f}%")

        # Hard limits
        if metrics.epoch >= self.max_epochs:
            should_stop = True
            logger.info(f"[MAX EPOCHS] Reached {self.max_epochs} epochs")

        if metrics.val_acc >= self.convergence_acc:
            should_stop = True
            logger.info(f"[CONVERGED] Val Acc={metrics.val_acc:.2f}%")

        # Compute grok score (composite signal)
        metrics.grok_score = self._compute_grok_score(metrics)

        return self.state.phase, should_stop

    def _compute_grok_score(self, metrics: GrokingMetrics) -> float:
        """
        Compute composite grokking score.

        Components:
        - Gap closing rate
        - Val loss improvement rate
        - Stability after plateau
        """
        if len(self.state.metrics_history) < 10:
            return 0.0

        recent = self.state.metrics_history[-10:]

        # Gap closing (positive = closing)
        gap_delta = recent[0].generalization_gap - recent[-1].generalization_gap
        gap_score = max(0, gap_delta / 20.0)  # Normalized

        # Loss improvement
        loss_delta = recent[0].val_loss - recent[-1].val_loss
        loss_score = max(0, loss_delta / recent[0].val_loss) if recent[0].val_loss > 0 else 0

        # Stability (low variance = stable)
        val_losses = [m.val_loss for m in recent]
        stability = 1.0 - min(1.0, np.std(val_losses) / (np.mean(val_losses) + 1e-8))

        return 0.4 * gap_score + 0.4 * loss_score + 0.2 * stability


class GrokingTrainer:
    """
    Grokking-aware trainer with MetaGrokFast optimizer.

    Supports paper-matched config for proper grokking experiments.

    Runs training until grokking signals are detected.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        aggressive: bool = False,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm

        # MetaGrokFast optimizer
        self.optimizer = MetaGrokFast.for_trm(
            model.parameters(),
            aggressive=aggressive
        )
        self.config_name = 'aggressive' if aggressive else 'conservative'
        logger.info(f"Optimizer: MetaGrokFast ({self.config_name})")
        logger.info(f"  lr={self.optimizer.config.lr}, lambda={self.optimizer.config.grokfast_lambda}")
        logger.info(f"  weight_decay={self.optimizer.config.weight_decay}")
        logger.info(f"  bigeometric={self.optimizer.config.use_bigeometric}, muon={self.optimizer.config.use_muon}")

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if RICH_LOSS_AVAILABLE:
            self.criterion = NNCTRMLoss(class_weights=class_weights)
            logger.info("Loss: NNCTRMLoss (asymmetric exp(-pnl/k) weighting)")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Loss: CrossEntropyLoss")

        # Grokking detector
        self.detector = GrokingDetector()

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, targets, pnl in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            pnl = pnl.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(features)
            logits = output['strategy_logits']
            halt_logits = output['halt_probability']

            # Compute loss
            if RICH_LOSS_AVAILABLE and hasattr(self.criterion, 'forward'):
                try:
                    loss = self.criterion(logits, halt_logits, targets, pnl)
                except TypeError:
                    loss = nn.functional.cross_entropy(logits, targets)
            else:
                loss = self.criterion(logits, targets)

            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        top3_correct = 0
        total = 0

        for features, targets, pnl in self.val_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            pnl = pnl.to(self.device)

            output = self.model(features)
            logits = output['strategy_logits']
            halt_logits = output['halt_probability']

            # Loss
            if RICH_LOSS_AVAILABLE and hasattr(self.criterion, 'forward'):
                try:
                    loss = self.criterion(logits, halt_logits, targets, pnl)
                except TypeError:
                    loss = nn.functional.cross_entropy(logits, targets)
            else:
                loss = self.criterion(logits, targets)

            total_loss += loss.item()

            # Top-1
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()

            # Top-3
            _, top3_pred = logits.topk(min(logits.size(1), 3), dim=1)
            for i in range(targets.size(0)):
                if targets[i] in top3_pred[i]:
                    top3_correct += 1

            total += targets.size(0)

        return (
            total_loss / len(self.val_loader),
            100.0 * correct / total,
            100.0 * top3_correct / total
        )

    def train_until_grokking(
        self,
        checkpoint_dir: Optional[Path] = None,
        save_every: int = 50,
        start_epoch: int = 0,
    ) -> Dict:
        """
        Train until grokking signals are detected.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            start_epoch: Epoch to start from (for resumption)

        Returns:
            Training summary with metrics and state.
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("GROKKING-AWARE TRAINING WITH METAGROKFAST")
        if start_epoch > 0:
            print(f"RESUMING FROM EPOCH {start_epoch}")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)
        print("\nPhases: warmup -> memorization -> plateau -> grokking -> converged")
        print("Monitoring: val_loss plateau, sudden drops, gap closing")
        print("=" * 70 + "\n")

        epoch = start_epoch
        should_stop = False

        while not should_stop:
            epoch += 1

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_top3 = self.validate()

            # Create metrics
            metrics = GrokingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_top3_acc=val_top3,
            )

            # Update detector
            phase, should_stop = self.detector.update(metrics)

            # Get optimizer stats
            opt_stats = self.optimizer.get_stats() if hasattr(self.optimizer, 'get_stats') else {}

            # Print progress
            gap = metrics.generalization_gap
            phase_str = f"[{phase.upper():^12}]"

            print(f"Epoch {epoch:4d} {phase_str} | "
                  f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
                  f"Gap: {gap:+.1f}% | "
                  f"Grok: {metrics.grok_score:.3f}")

            # Save checkpoint
            if checkpoint_dir and (epoch % save_every == 0 or should_stop):
                self._save_checkpoint(checkpoint_dir, epoch, metrics)

            # Special logging for phase transitions
            if phase == "plateau" and self.detector.state.epochs_in_plateau == 0:
                print(f"\n{'*' * 50}")
                print(f"* PLATEAU ENTERED at epoch {epoch}")
                print(f"* Val Loss: {val_loss:.4f}")
                print(f"* Waiting for grokking signal...")
                print(f"{'*' * 50}\n")

            elif phase == "grokking" and self.detector.state.grok_detected_epoch == epoch:
                print(f"\n{'!' * 50}")
                print(f"! GROKKING DETECTED at epoch {epoch}")
                print(f"! Val Loss: {self.detector.state.pre_grok_val_loss:.4f} -> {val_loss:.4f}")
                print(f"! Improvement: {self.detector.state.grok_magnitude*100:.1f}%")
                print(f"! Plateau duration: {self.detector.state.epochs_in_plateau} epochs")
                print(f"{'!' * 50}\n")

        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final Phase: {self.detector.state.phase}")
        print(f"Total Epochs: {epoch}")
        print(f"Best Val Acc: {self.detector.state.best_val_acc:.2f}% (epoch {self.detector.state.best_epoch})")
        print(f"Best Val Loss: {self.detector.state.best_val_loss:.4f}")

        if self.detector.state.grok_detected_epoch > 0:
            print(f"\nGrokking Details:")
            print(f"  Detected at epoch: {self.detector.state.grok_detected_epoch}")
            print(f"  Plateau duration: {self.detector.state.epochs_in_plateau} epochs")
            print(f"  Magnitude: {self.detector.state.grok_magnitude*100:.1f}% val loss drop")

        print("=" * 70 + "\n")

        # Save final checkpoint
        if checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, epoch, metrics, final=True)
            self._save_training_log(checkpoint_dir)

        return {
            'final_epoch': epoch,
            'state': self.detector.state.to_dict(),
            'metrics_history': [asdict(m) for m in self.detector.state.metrics_history],
        }

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, metrics: GrokingMetrics, final: bool = False):
        """Save model checkpoint."""
        filename = "final_model.pt" if final else f"checkpoint_epoch_{epoch}.pt"
        path = checkpoint_dir / filename

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': asdict(metrics),
            'state': self.detector.state.to_dict(),
        }, path)

        if final:
            logger.info(f"Saved final model to {path}")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Load model and optimizer state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Starting epoch number (checkpoint epoch + 1)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model state")

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")

        # Load detector state
        if 'state' in checkpoint:
            state = checkpoint['state']
            self.detector.state.phase = state.get('phase', 'warmup')
            self.detector.state.plateau_start_epoch = state.get('plateau_start_epoch', -1)
            self.detector.state.epochs_in_plateau = state.get('epochs_in_plateau', 0)
            self.detector.state.grok_detected_epoch = state.get('grok_detected_epoch', -1)
            self.detector.state.grok_magnitude = state.get('grok_magnitude', 0.0)
            self.detector.state.best_val_acc = state.get('best_val_acc', 0.0)
            self.detector.state.best_val_loss = state.get('best_val_loss', float('inf'))
            self.detector.state.best_epoch = state.get('best_epoch', 0)
            logger.info(f"Loaded detector state: phase={self.detector.state.phase}, best_val_acc={self.detector.state.best_val_acc:.2f}%")

        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

        return start_epoch

    def _save_training_log(self, checkpoint_dir: Path):
        """Save training log as JSON."""
        path = checkpoint_dir / "training_log.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'state': self.detector.state.to_dict(),
            'metrics_history': [asdict(m) for m in self.detector.state.metrics_history],
            'config': {
                'optimizer': 'MetaGrokFast',
                'lr': self.optimizer.config.lr,
                'grokfast_lambda': self.optimizer.config.grokfast_lambda,
                'use_bigeometric': self.optimizer.config.use_bigeometric,
                'use_muon': self.optimizer.config.use_muon,
            }
        }

        with open(path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved training log to {path}")


def main():
    parser = argparse.ArgumentParser(description='TRM Grokking-Aware Training')
    parser.add_argument('--data_path', type=str,
                        default='data/trm_training/black_swan_labels.parquet',
                        help='Path to training data')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='models/trm_grokking',
                        help='Directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=96,
                        help='Hidden dimension (default 96 for 59.5:1 param ratio)')
    parser.add_argument('--binary_classification', action='store_true',
                        help='Use binary classification (positive vs negative return)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Use aggressive MetaGrokFast config')
    parser.add_argument('--paper', action='store_true',
                        help='Use TRM paper config (lr=1e-4, wd=1.0, lambda=2.0, no enhancements)')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use ENHANCED config (paper params + Muon + Bigeometric)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override weight decay')
    parser.add_argument('--grokfast_lambda', type=float, default=None,
                        help='Override GrokFast lambda')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum epochs')
    parser.add_argument('--convergence_acc', type=float, default=80.0,
                        help='Val accuracy to declare convergence')
    parser.add_argument('--plateau_patience', type=int, default=20,
                        help='Epochs to wait in plateau')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., models/trm_grokking/final_model.pt)')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data
    data_path = project_root / args.data_path
    logger.info(f"Loading data from {data_path}")

    # Determine number of classes
    num_classes = 2 if args.binary_classification else 8

    data_module = TRMDataModule(
        data_path=data_path,
        random_seed=args.seed,
        binary_classification=args.binary_classification
    )
    train_loader, val_loader, test_loader = data_module.create_dataloaders(
        batch_size=args.batch_size,
        shuffle=True
    )

    # Class weights for imbalanced data
    class_weights = data_module.compute_class_weights(num_classes=num_classes)
    logger.info(f"Class weights: {class_weights}")

    # Model
    model = TinyRecursiveModel(
        input_dim=len(data_module.train_dataset.features[0]),
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_latent_steps=6,
        num_recursion_cycles=3,
    )

    # Determine config
    if args.enhanced:
        config = TRM_ENHANCED_CONFIG
        config_name = "ENHANCED (paper params + Muon + Bigeometric)"
    elif args.paper:
        config = TRM_PAPER_CONFIG
        config_name = "TRM_PAPER (lr=1e-4, wd=1.0, lambda=2.0, vanilla)"
    elif args.aggressive:
        config = TRM_AGGRESSIVE_CONFIG
        config_name = "AGGRESSIVE"
    else:
        config = TRM_CONFIG
        config_name = "CONSERVATIVE"

    # Override config values if provided
    if args.lr is not None:
        config.lr = args.lr
        config.muon_lr = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.grokfast_lambda is not None:
        config.grokfast_lambda = args.grokfast_lambda

    logger.info(f"Using config: {config_name}")
    logger.info(f"  lr={config.lr}, weight_decay={config.weight_decay}, lambda={config.grokfast_lambda}")

    # Trainer with custom config
    trainer = GrokingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        aggressive=False,  # We handle config ourselves now
    )
    # Override optimizer with our custom config
    trainer.optimizer = MetaGrokFast(model.parameters(), config=config)
    trainer.config_name = config_name
    logger.info(f"[ACTUAL CONFIG] lr={config.lr}, wd={config.weight_decay}, lambda={config.grokfast_lambda}")
    logger.info(f"[ACTUAL CONFIG] bigeometric={config.use_bigeometric}, muon={config.use_muon}")

    # Update detector settings
    trainer.detector.max_epochs = args.max_epochs
    trainer.detector.convergence_acc = args.convergence_acc
    trainer.detector.plateau_patience = args.plateau_patience

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        start_epoch = trainer.load_checkpoint(resume_path)

    # Train until grokking
    checkpoint_dir = project_root / args.checkpoint_dir
    result = trainer.train_until_grokking(
        checkpoint_dir=checkpoint_dir,
        save_every=25,
        start_epoch=start_epoch,
    )

    # Save normalization params for inference
    data_module.save_normalization_params(checkpoint_dir / "normalization_params.json")

    logger.info("Training complete!")
    return result


if __name__ == '__main__':
    main()
