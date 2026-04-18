"""
TRM Optimized Training - MOO-tuned hyperparameters + Augmentation + LR Scheduling

Uses:
1. MOO-optimized hyperparameters: hidden=512, T=3, n=6, lr=1e-3
2. Data augmentation: noise, masking, mixup
3. Cosine annealing LR scheduler with warmup
4. Extended training: 50 epochs

Expected improvement: 18% -> 25%+ validation accuracy

SOURCE: MOO hyperparameter search (results/trm_hyperparameter_moo.json)
"""

import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.trm_model import TinyRecursiveModel
from training.trm_data_loader import TRMDataModule
from training.trm_trainer import TRMTrainer, GrokFastOptimizer
from training.trm_augmentation import (
    TRMDataAugmenter,
    AugmentationConfig,
    AugmentedTRMDataset,
    get_default_augmentation_config
)

# Setup logging
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'trm_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# MOO-optimized hyperparameters
OPTIMIZED_CONFIG = {
    'hidden_dim': 512,      # MOO found smaller is better for this dataset
    'T': 3,                 # Recursion cycles
    'n': 6,                 # Latent steps
    'learning_rate': 1e-3,  # Higher LR works with smaller model
    'batch_size': 256,
    'epochs': 100,          # Extended training for grokking
    'warmup_epochs': 5,     # LR warmup
    'patience': 25,         # Longer patience to see grokking
    'optimizer_type': 'metagrokfast',  # Use METAGROKFAST for grokking
}


def get_lr_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 3):
    """
    Create cosine annealing LR scheduler with linear warmup.

    Args:
        optimizer: The optimizer
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        LR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return LambdaLR(optimizer, lr_lambda)


class OptimizedTRMTrainer(TRMTrainer):
    """
    Enhanced TRM Trainer with LR scheduling and augmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        augmenter: TRMDataAugmenter = None,
        num_epochs: int = 50,
        warmup_epochs: int = 3,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        self.augmenter = augmenter
        self.num_epochs = num_epochs

        # Create LR scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer,
            num_epochs,
            warmup_epochs
        )

    def train_epoch(self):
        """Train one epoch with augmentation."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (features, targets, pnl) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            pnl = pnl.to(self.device)

            # Apply augmentation if enabled
            if self.augmenter is not None:
                features, targets, pnl = self.augmenter.augment_batch(
                    features, targets, pnl
                )

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(features, T=self.T, n=self.n)
            logits = output['strategy_logits']
            halt_logits = output['halt_probability']

            # Compute loss
            try:
                loss = self.criterion(logits, halt_logits, targets, pnl)
            except:
                loss = self.criterion(logits, targets, self.model, self.T, self.n)

            # Backward pass with gradient clipping
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }

    def fit(self, num_epochs: int, patience: int = 10, checkpoint_dir=None):
        """Train with LR scheduling."""
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0

        print(f"\n{'='*60}")
        print(f"Optimized TRM Training (T={self.T}, n={self.n})")
        print(f"Device: {self.device}")
        print(f"Augmentation: {'Enabled' if self.augmenter else 'Disabled'}")
        print(f"LR Schedule: Cosine annealing with warmup")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Get current LR
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # Step scheduler
            self.scheduler.step()

            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val Top-3: {val_metrics['top3_accuracy']:.2f}%")

            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                epochs_without_improvement = 0

                print(f"[BEST] New best: {self.best_val_acc:.2f}%")

                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / "best_model_optimized.pt",
                        epoch,
                        val_metrics
                    )
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        print(f"{'='*60}\n")

        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'total_epochs': epoch + 1
        }


def main():
    """Run optimized TRM training."""
    print("=" * 70)
    print("TRM OPTIMIZED TRAINING")
    print("MOO-tuned hyperparameters + Augmentation + LR Scheduling")
    print("=" * 70)
    print()

    # Configuration
    config = OPTIMIZED_CONFIG
    print("[1/6] Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load data
    print("[2/6] Loading training data...")
    data_path = Path(__file__).parent.parent.parent / 'data' / 'trm_training' / 'black_swan_labels.parquet'

    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return

    data_module = TRMDataModule(data_path=str(data_path), random_seed=42)
    train_loader, val_loader, test_loader = data_module.create_dataloaders(
        batch_size=config['batch_size'],
        shuffle=True
    )

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print()

    # Class weights
    print("[3/6] Computing class weights...")
    class_weights = data_module.compute_class_weights(num_classes=8, max_weight=10.0)
    print(f"  Weights: {class_weights.tolist()}")
    print()

    # Create model with MOO-optimized hyperparameters
    print("[4/6] Creating model...")
    model = TinyRecursiveModel(
        input_dim=10,
        hidden_dim=config['hidden_dim'],
        output_dim=8,
        num_latent_steps=config['n'],
        num_recursion_cycles=config['T']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Architecture: {config['hidden_dim']} hidden, T={config['T']}, n={config['n']}")
    print()

    # Setup augmentation
    print("[5/6] Setting up augmentation...")
    aug_config = get_default_augmentation_config()
    augmenter = TRMDataAugmenter(aug_config)
    print(f"  Noise: std={aug_config.noise_std}, prob={aug_config.noise_prob}")
    print(f"  Mask: prob={aug_config.mask_prob}, max_features={aug_config.mask_max_features}")
    print(f"  Mixup: alpha={aug_config.mixup_alpha}, prob={aug_config.mixup_prob}")
    print()

    # Create trainer
    print("[6/6] Creating trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = OptimizedTRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        augmenter=augmenter,
        num_epochs=config['epochs'],
        warmup_epochs=config['warmup_epochs'],
        device=device,
        lr=config['learning_rate'],
        loss_type='nnc',
        max_grad_norm=1.0,
        class_weights=class_weights,
        T=config['T'],
        n=config['n']
    )

    print(f"  Device: {device}")
    print(f"  Optimizer: GrokFast + Cosine LR")
    print(f"  Loss: NNC asymmetric")
    print()

    # Train
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    results = trainer.fit(
        num_epochs=config['epochs'],
        patience=config['patience'],
        checkpoint_dir=checkpoint_dir
    )

    # Save results
    results_path = Path(__file__).parent.parent.parent / 'results' / 'trm_optimized_training.json'
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'best_val_acc': results['best_val_acc'],
            'best_epoch': results['best_epoch'],
            'total_epochs': results['total_epochs'],
            'train_history': [{'loss': h['loss'], 'accuracy': h['accuracy']} for h in results['train_history']],
            'val_history': [{'loss': h['loss'], 'accuracy': h['accuracy'], 'top3': h['top3_accuracy']} for h in results['val_history']],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"Results saved to {results_path}")

    # Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    model.eval()
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for features, targets, pnl in test_loader:
            features = features.to(device)
            targets = targets.to(device)

            output = model(features, T=config['T'], n=config['n'])
            logits = output['strategy_logits']

            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()

            _, top3_pred = logits.topk(3, dim=1)
            top3_correct += sum(targets[i] in top3_pred[i] for i in range(targets.size(0)))

            total += targets.size(0)

    test_acc = 100.0 * correct / total
    test_top3 = 100.0 * top3_correct / total

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Top-3 Accuracy: {test_top3:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
