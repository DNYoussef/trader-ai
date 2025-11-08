"""
TRM Trainer - Phase 2 Implementation
Training orchestration with GrokFast optimizer and comprehensive training loop.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import json


class TRMLoss(nn.Module):
    """
    TRM Loss Function - Combines cross-entropy with complexity penalty.

    Loss = CrossEntropy(predictions, targets) + λ * complexity_penalty

    Supports class weighting to handle imbalanced datasets.
    """

    def __init__(self, complexity_weight: float = 0.01, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.complexity_weight = complexity_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        T: Optional[int] = None,
        n: int = 6
    ) -> torch.Tensor:
        """
        Compute loss with optional complexity penalty.

        Args:
            logits: [batch, num_classes] model predictions
            targets: [batch] target labels
            model: TinyRecursiveModel instance (for complexity)
            T: Recursion depth used
            n: Base operations per layer

        Returns:
            loss: Scalar loss value
        """
        # Base cross-entropy loss
        ce = self.ce_loss(logits, targets)

        # Skip complexity penalty for now (model.get_complexity not implemented)
        # Future: add complexity penalty based on T * n recursion depth

        return ce


class GrokFastOptimizer:
    """
    GrokFast Optimizer Wrapper - Accelerates grokking via gradient filtering.

    Algorithm:
    1. Maintain EMA of gradients: ema = α * ema + (1-α) * grad
    2. Filter gradients: grad_new = grad + λ * (grad - ema)
    3. Apply filtered gradients to base optimizer

    Paper: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
    """

    def __init__(
        self,
        params,
        base_optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        alpha: float = 0.98,
        lambda_: float = 0.1,
        weight_decay: float = 0.01
    ):
        """
        Initialize GrokFast optimizer.

        Args:
            params: Model parameters
            base_optimizer: Base optimizer (defaults to AdamW)
            lr: Learning rate
            alpha: EMA decay rate (0.98 recommended)
            lambda_: Gradient filter strength (0.1 recommended)
            weight_decay: Weight decay for AdamW
        """
        self.params = list(params)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.lr = lr

        # Initialize base optimizer
        if base_optimizer is None:
            self.optimizer = AdamW(self.params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = base_optimizer

        # Initialize EMA storage for gradients
        self.grad_ema = {}
        for i, param in enumerate(self.params):
            if param.requires_grad:
                self.grad_ema[i] = torch.zeros_like(param.data)

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform optimization step with gradient filtering.

        Applies GrokFast filtering before calling base optimizer step.
        """
        # Apply gradient filtering
        for i, param in enumerate(self.params):
            if param.requires_grad and param.grad is not None:
                # Update EMA: ema = α * ema + (1-α) * grad
                self.grad_ema[i] = (
                    self.alpha * self.grad_ema[i] +
                    (1 - self.alpha) * param.grad.data
                )

                # Filter: grad_new = grad + λ * (grad - ema)
                filtered_grad = (
                    param.grad.data +
                    self.lambda_ * (param.grad.data - self.grad_ema[i])
                )

                # Replace gradient
                param.grad.data = filtered_grad

        # Apply base optimizer step
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'grad_ema': self.grad_ema,
            'alpha': self.alpha,
            'lambda': self.lambda_
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.grad_ema = state_dict['grad_ema']
        self.alpha = state_dict['alpha']
        self.lambda_ = state_dict['lambda']


class TRMTrainer:
    """
    TRM Trainer - Comprehensive training orchestration.

    Features:
    - GrokFast optimizer integration
    - Progress tracking with tqdm
    - Early stopping by validation accuracy
    - Model checkpointing
    - Metrics logging (loss, accuracy, top-3 accuracy)
    - Device agnostic (CPU/GPU)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        alpha: float = 0.98,
        lambda_: float = 0.1,
        weight_decay: float = 0.01,
        T: int = 3,
        n: int = 6,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize TRM trainer.

        Args:
            model: TinyRecursiveModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training (auto-detect if None)
            lr: Learning rate
            alpha: GrokFast EMA decay
            lambda_: GrokFast filter strength
            weight_decay: Weight decay
            T: Recursion depth for forward pass
            n: Base operations per layer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.T = T
        self.n = n

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = GrokFastOptimizer(
            self.model.parameters(),
            lr=lr,
            alpha=alpha,
            lambda_=lambda_,
            weight_decay=weight_decay
        )
        # Move class weights to device if provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = TRMLoss(class_weights=class_weights)

        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            metrics: Dict with 'loss' and 'accuracy'
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (features, targets, pnl) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            # pnl = pnl.to(self.device)  # Not needed for now, but available for loss weighting

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(features, T=self.T, n=self.n)
            logits = output['strategy_logits']  # Extract logits from dict

            # Compute loss
            loss = self.criterion(logits, targets, self.model, self.T, self.n)

            # Backward pass
            loss.backward()
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

        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            metrics: Dict with 'loss', 'accuracy', 'top3_accuracy'
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        top3_correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for features, targets, pnl in pbar:
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                # pnl = pnl.to(self.device)  # Not needed for now

                # Forward pass
                output = self.model(features, T=self.T, n=self.n)
                logits = output['strategy_logits']  # Extract logits from dict

                # Compute loss
                loss = self.criterion(logits, targets, self.model, self.T, self.n)

                # Track metrics
                total_loss += loss.item()

                # Top-1 accuracy
                pred = logits.argmax(dim=1)
                correct += (pred == targets).sum().item()

                # Top-3 accuracy
                _, top3_pred = logits.topk(3, dim=1)
                top3_correct += sum(
                    targets[i] in top3_pred[i] for i in range(targets.size(0))
                )

                total += targets.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss / len(pbar):.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100.0 * correct / total,
            'top3_accuracy': 100.0 * top3_correct / total
        }

        return metrics

    def fit(
        self,
        num_epochs: int,
        patience: int = 10,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train model with early stopping.

        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience (epochs)
            checkpoint_dir: Directory for saving checkpoints

        Returns:
            training_summary: Dict with training history and best metrics
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0

        print(f"\n{'='*60}")
        print(f"Training TRM Model (T={self.T}, n={self.n})")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # Print metrics
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                  f"Val Top-3 Acc: {val_metrics['top3_accuracy']:.2f}%")

            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                epochs_without_improvement = 0

                print(f"[BEST] New best validation accuracy: {self.best_val_acc:.2f}%")

                # Save checkpoint
                if checkpoint_dir is not None:
                    self.save_checkpoint(
                        checkpoint_dir / "best_model.pt",
                        epoch,
                        val_metrics
                    )
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")
        print(f"{'='*60}\n")

        training_summary = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'total_epochs': epoch + 1
        }

        return training_summary

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Save model checkpoint.

        Args:
            path: Checkpoint file path
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'hyperparameters': {
                'T': self.T,
                'n': self.n
            }
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.T = checkpoint['hyperparameters']['T']
        self.n = checkpoint['hyperparameters']['n']

        print(f"Checkpoint loaded from {path}")
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best val accuracy: {self.best_val_acc:.2f}%")
