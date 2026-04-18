"""
Focal Loss for Imbalanced Strategy Classification

Focal Loss (Lin et al., 2017) addresses class imbalance by down-weighting
easy examples and focusing on hard examples. Critical for TRM training
where crisis strategies (0, 6) are rare but important.

Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Parameters:
- gamma: Focusing parameter (default 2.0)
  - gamma=0: Standard cross-entropy
  - gamma>0: Down-weight easy examples
  - gamma=2: Recommended for moderate imbalance

- alpha: Class weight (optional)
  - Can be computed from inverse class frequency
  - Capped at 10x to prevent overfitting rare classes
"""

from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for imbalanced classification.

    Focal Loss reduces the relative loss for well-classified examples (p > 0.5),
    putting more focus on hard, misclassified examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
            alpha: Class weights tensor (n_classes,). If None, uniform weights.
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (batch, n_classes) raw model outputs
            targets: (batch,) integer class labels

        Returns:
            Scalar loss if reduction='mean'/'sum', else (batch,) losses
        """
        n_classes = logits.shape[-1]
        device = logits.device

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            with torch.no_grad():
                smoothed = torch.full_like(
                    logits,
                    self.label_smoothing / (n_classes - 1)
                )
                smoothed.scatter_(
                    1,
                    targets.unsqueeze(1),
                    1.0 - self.label_smoothing
                )
        else:
            smoothed = None

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy loss
        if smoothed is not None:
            # Use smoothed targets
            ce_loss = -torch.sum(smoothed * F.log_softmax(logits, dim=-1), dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TRMFocalLoss(nn.Module):
    """
    Combined loss for TRM training with focal loss.

    Combines:
    1. Focal Loss for strategy classification (handles imbalance)
    2. BCE for halt signal
    3. NNC profit-weighted loss for reward/punishment

    Total = lambda_focal * L_focal + lambda_halt * L_halt + lambda_profit * L_profit
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        lambda_focal: float = 1.0,
        lambda_halt: float = 0.5,
        lambda_profit: float = 0.3,
        k_gain: float = 0.05,
        k_loss: float = 0.02,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            gamma: Focal loss gamma parameter
            class_weights: Per-class weights (8,). Computed from inverse freq if None.
            lambda_focal: Weight for focal (strategy) loss
            lambda_halt: Weight for halt BCE loss
            lambda_profit: Weight for profit-weighted loss
            k_gain: k parameter for profitable trades (NNC)
            k_loss: k parameter for losing trades (NNC, smaller = more punishment)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.lambda_halt = lambda_halt
        self.lambda_profit = lambda_profit
        self.k_gain = k_gain
        self.k_loss = k_loss

        self.focal_loss = FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            reduction='none',
            label_smoothing=label_smoothing
        )
        self.halt_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        strategy_logits: torch.Tensor,
        halt_logits: torch.Tensor,
        targets: torch.Tensor,
        pnl: torch.Tensor,
        halt_targets: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined TRM focal loss.

        Args:
            strategy_logits: (batch, 8) raw strategy predictions
            halt_logits: (batch,) or (batch, 1) halt signal logits
            targets: (batch,) integer strategy labels
            pnl: (batch,) profit/loss for each sample
            halt_targets: (batch,) optional halt targets (1 = should halt)
            return_components: If True, return dict with individual losses

        Returns:
            Combined loss scalar, or dict with components if return_components
        """
        batch_size = strategy_logits.shape[0]
        device = strategy_logits.device

        # Ensure halt_logits is (batch,)
        if halt_logits.dim() == 2:
            halt_logits = halt_logits.squeeze(-1)

        # 1. Focal loss for strategy classification
        focal = self.focal_loss(strategy_logits, targets)  # (batch,)

        # 2. Profit-weighted loss using NNC asymmetric k
        k = torch.where(pnl >= 0, self.k_gain, self.k_loss)
        profit_weight = torch.exp(torch.clamp(-pnl / k, -10, 10))
        profit_weighted_focal = focal * profit_weight

        # 3. Halt loss
        if halt_targets is None:
            # Default: should halt when prediction is wrong AND trade lost money
            with torch.no_grad():
                predictions = strategy_logits.argmax(dim=-1)
                wrong = predictions != targets
                lost_money = pnl < 0
                halt_targets = (wrong & lost_money).float()

        halt = self.halt_loss(halt_logits, halt_targets)  # (batch,)

        # Combine losses
        total = (
            self.lambda_focal * profit_weighted_focal +
            self.lambda_halt * halt
        ).mean()

        if return_components:
            return {
                'total': total,
                'focal': focal.mean(),
                'focal_weighted': profit_weighted_focal.mean(),
                'halt': halt.mean(),
                'profit_weight_mean': profit_weight.mean(),
            }

        return total


def compute_class_weights(
    strategy_counts: Dict[int, int],
    n_classes: int = 8,
    max_weight: float = 10.0,
    use_sqrt: bool = True
) -> torch.Tensor:
    """
    Compute class weights from strategy sample counts.

    Uses inverse frequency with optional sqrt dampening and capping.

    Args:
        strategy_counts: Dict mapping strategy index to sample count
        n_classes: Total number of classes
        max_weight: Maximum weight cap (prevents overfitting rare classes)
        use_sqrt: If True, use sqrt of inverse frequency (dampened)

    Returns:
        weights: (n_classes,) tensor of class weights
    """
    # Get counts, defaulting to 1 for missing classes
    counts = torch.tensor([
        max(strategy_counts.get(i, 1), 1)
        for i in range(n_classes)
    ], dtype=torch.float32)

    # Compute inverse frequency
    total = counts.sum()
    inverse_freq = total / counts

    # Optionally apply sqrt dampening
    if use_sqrt:
        inverse_freq = torch.sqrt(inverse_freq)

    # Normalize so min weight = 1.0
    weights = inverse_freq / inverse_freq.min()

    # Cap at max_weight
    weights = torch.clamp(weights, max=max_weight)

    logger.info(f"Class weights: {weights.tolist()}")

    return weights


def compute_class_weights_from_labels(
    labels: torch.Tensor,
    n_classes: int = 8,
    max_weight: float = 10.0,
    use_sqrt: bool = True
) -> torch.Tensor:
    """
    Compute class weights directly from label tensor.

    Args:
        labels: (N,) tensor of integer labels
        n_classes: Total number of classes
        max_weight: Maximum weight cap
        use_sqrt: Use sqrt dampening

    Returns:
        weights: (n_classes,) tensor of class weights
    """
    # Count occurrences
    counts = {}
    for i in range(n_classes):
        counts[i] = int((labels == i).sum().item())

    return compute_class_weights(counts, n_classes, max_weight, use_sqrt)
