"""
TRM Loss Functions for Trade Reasoning Module

Implements 3-component loss function:
1. Task Loss: Cross-entropy for 8-way strategy classification
2. Halt Loss: Binary cross-entropy for halting decisions
3. Profit-Weighted Loss: Task loss weighted by trade profitability

Author: Loss Functions Implementation Agent
Date: 2025-11-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def compute_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute cross-entropy loss for 8-way strategy classification.

    Args:
        logits: Predicted logits, shape (batch_size, num_classes=8)
        labels: Ground truth labels, shape (batch_size,)
        class_weights: Optional class weights for imbalanced data, shape (num_classes,)
        reduction: How to reduce the loss ('none', 'mean', 'sum')

    Returns:
        loss: Scalar loss if reduction != 'none', else shape (batch_size,)

    Formula:
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction=reduction)
    """
    # Validate inputs
    assert logits.ndim == 2, f"Logits must be 2D (batch, classes), got {logits.shape}"
    assert labels.ndim == 1, f"Labels must be 1D (batch,), got {labels.shape}"
    assert logits.size(0) == labels.size(0), "Batch size mismatch"
    assert logits.size(1) == 8, f"Expected 8 classes, got {logits.size(1)}"

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction=reduction
    )

    return loss


def compute_halt_loss(
    halt_logits: torch.Tensor,
    task_logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_threshold: float = 0.7,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for halting decisions.

    Dynamic halt targets: halt=1 if model is confident AND correct.

    Args:
        halt_logits: Raw halt logits, shape (batch_size, 1) or (batch_size,)
        task_logits: Task prediction logits, shape (batch_size, num_classes=8)
        labels: Ground truth labels, shape (batch_size,)
        confidence_threshold: Threshold for "confident" prediction (default 0.7)
        reduction: How to reduce the loss ('none', 'mean', 'sum')

    Returns:
        loss: Scalar loss if reduction != 'none', else shape (batch_size,)

    Formula:
        halt_target = (max(softmax(task_logits)) > threshold AND pred == label).float()
        loss = BCE_with_logits(halt_logits, halt_target)
    """
    # Validate inputs
    batch_size = task_logits.size(0)
    assert labels.size(0) == batch_size, "Batch size mismatch"

    # Reshape halt_logits if needed
    if halt_logits.ndim == 2:
        halt_logits = halt_logits.squeeze(-1)
    assert halt_logits.ndim == 1, f"Halt logits must be 1D after squeeze, got {halt_logits.shape}"

    # Compute predictions and confidence
    probs = F.softmax(task_logits, dim=-1)  # (batch_size, 8)
    max_probs, predictions = probs.max(dim=-1)  # (batch_size,)

    # Create dynamic halt targets
    # halt=1 if: (1) confident (max_prob > threshold) AND (2) correct (pred == label)
    is_confident = max_probs > confidence_threshold
    is_correct = predictions == labels
    halt_targets = (is_confident & is_correct).float()  # (batch_size,)

    # Compute BCE loss with logits
    loss = F.binary_cross_entropy_with_logits(
        halt_logits,
        halt_targets,
        reduction=reduction
    )

    return loss


def compute_profit_weighted_loss(
    task_logits: torch.Tensor,
    labels: torch.Tensor,
    pnl: torch.Tensor,
    pnl_scale: float = 0.05,
    class_weights: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute task loss weighted by trade profitability.

    Profitable trades get lower weight (already good), unprofitable trades get higher weight.

    Args:
        task_logits: Predicted logits, shape (batch_size, num_classes=8)
        labels: Ground truth labels, shape (batch_size,)
        pnl: Profit/loss per sample, shape (batch_size,)
        pnl_scale: Scaling factor for tanh (default 0.05 = 5%)
        class_weights: Optional class weights, shape (num_classes,)
        reduction: How to reduce the loss ('none', 'mean', 'sum')

    Returns:
        loss: Weighted task loss

    Formula:
        profit_weight = 1 - tanh(pnl / pnl_scale)
        weighted_loss = profit_weight * task_loss

    Examples:
        pnl = +0.10 (10% profit) → weight ≈ -0.87 (discourage overfitting to luck)
        pnl = +0.05 (5% profit)  → weight ≈ 0.00 (neutral)
        pnl = 0.00 (breakeven)   → weight = 1.00 (normal learning)
        pnl = -0.05 (5% loss)    → weight ≈ 2.00 (learn from mistakes)
        pnl = -0.10 (10% loss)   → weight ≈ 2.87 (strongly learn from big losses)
    """
    # Validate inputs
    batch_size = task_logits.size(0)
    assert labels.size(0) == batch_size, "Batch size mismatch"
    assert pnl.size(0) == batch_size, "PnL batch size mismatch"

    # Compute base task loss per sample (no reduction yet)
    task_loss = compute_task_loss(
        task_logits,
        labels,
        class_weights=class_weights,
        reduction='none'
    )  # (batch_size,)

    # Compute profit weights
    # weight = 1 - tanh(pnl / pnl_scale)
    # Positive PnL → negative weight (discourage)
    # Negative PnL → positive weight (encourage learning)
    profit_weights = 1.0 - torch.tanh(pnl / pnl_scale)  # (batch_size,)

    # Apply profit weighting
    weighted_loss = profit_weights * task_loss  # (batch_size,)

    # Apply final reduction
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    elif reduction == 'none':
        return weighted_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class TRMLoss(nn.Module):
    """
    Combined TRM loss function with 3 components:
    1. Task loss (cross-entropy for strategy classification)
    2. Halt loss (BCE for halting decisions)
    3. Profit-weighted task loss (learn from mistakes)

    Total loss:
        L = lambda_halt * L_halt + lambda_profit * L_profit_weighted_task

    Default hyperparameters:
        lambda_halt = 0.01 (small contribution from halt)
        lambda_profit = 1.0 (main learning signal)
    """

    def __init__(
        self,
        lambda_halt: float = 0.01,
        lambda_profit: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.7,
        pnl_scale: float = 0.05
    ):
        """
        Initialize TRM loss function.

        Args:
            lambda_halt: Weight for halt loss component
            lambda_profit: Weight for profit-weighted task loss
            class_weights: Optional class weights for imbalanced data, shape (8,)
            confidence_threshold: Threshold for halt target (default 0.7)
            pnl_scale: Scaling factor for profit weighting (default 0.05)
        """
        super().__init__()

        self.lambda_halt = lambda_halt
        self.lambda_profit = lambda_profit
        self.confidence_threshold = confidence_threshold
        self.pnl_scale = pnl_scale

        # Register class weights as buffer (not a parameter)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        task_logits: torch.Tensor,
        halt_logits: torch.Tensor,
        labels: torch.Tensor,
        pnl: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute combined TRM loss.

        Args:
            task_logits: Task prediction logits, shape (batch_size, 8)
            halt_logits: Halt decision logits, shape (batch_size, 1) or (batch_size,)
            labels: Ground truth labels, shape (batch_size,)
            pnl: Profit/loss per sample, shape (batch_size,)
            return_components: If True, return dict with all loss components

        Returns:
            loss: Scalar total loss, or dict with components if return_components=True
        """
        # Compute halt loss
        halt_loss = compute_halt_loss(
            halt_logits,
            task_logits,
            labels,
            confidence_threshold=self.confidence_threshold,
            reduction='mean'
        )

        # Compute profit-weighted task loss
        profit_weighted_task_loss = compute_profit_weighted_loss(
            task_logits,
            labels,
            pnl,
            pnl_scale=self.pnl_scale,
            class_weights=self.class_weights,
            reduction='mean'
        )

        # Combine losses
        total_loss = (
            self.lambda_halt * halt_loss +
            self.lambda_profit * profit_weighted_task_loss
        )

        if return_components:
            # Also compute unweighted task loss for monitoring
            task_loss = compute_task_loss(
                task_logits,
                labels,
                class_weights=self.class_weights,
                reduction='mean'
            )

            return {
                'total_loss': total_loss,
                'halt_loss': halt_loss,
                'task_loss': task_loss,
                'profit_weighted_task_loss': profit_weighted_task_loss,
                'lambda_halt': self.lambda_halt,
                'lambda_profit': self.lambda_profit
            }

        return total_loss

    def extra_repr(self) -> str:
        """String representation of loss function."""
        return (
            f"lambda_halt={self.lambda_halt}, "
            f"lambda_profit={self.lambda_profit}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"pnl_scale={self.pnl_scale}"
        )


def test_loss_functions():
    """Quick sanity test for loss functions."""
    batch_size = 4
    num_classes = 8

    # Create dummy data
    task_logits = torch.randn(batch_size, num_classes)
    halt_logits = torch.randn(batch_size, 1)
    labels = torch.randint(0, num_classes, (batch_size,))
    pnl = torch.tensor([0.10, 0.05, 0.0, -0.10])  # 10% profit, 5% profit, breakeven, 10% loss

    print("=" * 60)
    print("TRM Loss Functions Sanity Test")
    print("=" * 60)

    # Test individual components
    print("\n1. Task Loss (Cross-Entropy):")
    task_loss = compute_task_loss(task_logits, labels)
    print(f"   Loss: {task_loss.item():.4f}")

    print("\n2. Halt Loss (BCE):")
    halt_loss = compute_halt_loss(halt_logits, task_logits, labels)
    print(f"   Loss: {halt_loss.item():.4f}")

    print("\n3. Profit-Weighted Task Loss:")
    profit_loss = compute_profit_weighted_loss(task_logits, labels, pnl)
    print(f"   Loss: {profit_loss.item():.4f}")
    print(f"   PnL values: {pnl.tolist()}")
    print(f"   Profit weights: {(1.0 - torch.tanh(pnl / 0.05)).tolist()}")

    # Test combined loss
    print("\n4. Combined TRM Loss:")
    trm_loss = TRMLoss(lambda_halt=0.01, lambda_profit=1.0)
    print(f"   {trm_loss}")

    total_loss = trm_loss(task_logits, halt_logits, labels, pnl)
    print(f"   Total Loss: {total_loss.item():.4f}")

    # Test with components
    print("\n5. Loss Components:")
    components = trm_loss(task_logits, halt_logits, labels, pnl, return_components=True)
    for key, value in components.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.item():.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_loss_functions()
