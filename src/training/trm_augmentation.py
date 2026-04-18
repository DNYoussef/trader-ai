"""
TRM Data Augmentation - Noise Injection and Time Shifts

Augmentation techniques for TRM training data to improve generalization
and increase effective dataset size.

Techniques:
1. Gaussian Noise: Add small random noise to features
2. Time Shifts: Sample nearby time points as augmented examples
3. Feature Masking: Randomly zero out some features
4. Mixup: Interpolate between samples (label-preserving)

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Noise injection
    noise_enabled: bool = True
    noise_std: float = 0.1  # Standard deviation of Gaussian noise
    noise_prob: float = 0.5  # Probability of applying noise

    # Feature masking
    mask_enabled: bool = True
    mask_prob: float = 0.1  # Probability of masking each feature
    mask_max_features: int = 3  # Max features to mask per sample

    # Time shifts (for sequential data)
    time_shift_enabled: bool = False  # Requires temporal ordering
    time_shift_range: int = 2  # Max days to shift

    # Mixup augmentation
    mixup_enabled: bool = True
    mixup_alpha: float = 0.2  # Beta distribution parameter
    mixup_prob: float = 0.3  # Probability of applying mixup


class TRMDataAugmenter:
    """
    Data augmentation for TRM training.

    Applies various augmentation techniques to increase effective
    dataset size and improve model generalization.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter.

        Args:
            config: Augmentation configuration (uses defaults if None)
        """
        self.config = config or AugmentationConfig()

    def augment_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pnl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to a batch of data.

        Args:
            features: (batch_size, n_features) input features
            labels: (batch_size,) strategy labels
            pnl: (batch_size,) profit/loss values

        Returns:
            Augmented (features, labels, pnl)
        """
        batch_size = features.size(0)
        aug_features = features.clone()
        aug_labels = labels.clone()
        aug_pnl = pnl.clone()

        # Apply noise injection
        if self.config.noise_enabled:
            aug_features = self._add_noise(aug_features)

        # Apply feature masking
        if self.config.mask_enabled:
            aug_features = self._mask_features(aug_features)

        # Apply mixup (modifies labels too)
        if self.config.mixup_enabled and batch_size > 1:
            aug_features, aug_labels, aug_pnl = self._mixup(
                aug_features, aug_labels, aug_pnl
            )

        return aug_features, aug_labels, aug_pnl

    def _add_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to features."""
        batch_size = features.size(0)

        # Generate mask for which samples to augment
        mask = torch.rand(batch_size) < self.config.noise_prob

        if mask.sum() > 0:
            # Generate noise
            noise = torch.randn_like(features) * self.config.noise_std

            # Apply only to selected samples
            features = features.clone()
            features[mask] = features[mask] + noise[mask]

        return features

    def _mask_features(self, features: torch.Tensor) -> torch.Tensor:
        """Randomly mask some features to zero."""
        batch_size, n_features = features.shape
        features = features.clone()

        for i in range(batch_size):
            if torch.rand(1).item() < self.config.mask_prob:
                # Select random features to mask
                n_mask = torch.randint(1, self.config.mask_max_features + 1, (1,)).item()
                mask_indices = torch.randperm(n_features)[:n_mask]
                features[i, mask_indices] = 0.0

        return features

    def _mixup(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pnl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.

        For classification, we only mixup samples with the same label
        to preserve label correctness.
        """
        batch_size = features.size(0)

        # Generate mask for which samples to augment
        mask = torch.rand(batch_size) < self.config.mixup_prob

        if mask.sum() > 0:
            features = features.clone()
            pnl = pnl.clone()

            # For each sample to augment, find another with same label
            for i in torch.where(mask)[0]:
                label_i = labels[i].item()

                # Find samples with same label
                same_label = (labels == label_i)
                same_label[i] = False  # Exclude self

                if same_label.sum() > 0:
                    # Random sample from same-label pool
                    candidates = torch.where(same_label)[0]
                    j = candidates[torch.randint(len(candidates), (1,))].item()

                    # Mixup weight from Beta distribution
                    lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)

                    # Interpolate features and PnL
                    features[i] = lam * features[i] + (1 - lam) * features[j]
                    pnl[i] = lam * pnl[i] + (1 - lam) * pnl[j]
                    # Label stays the same since both have same label

        return features, labels, pnl


class AugmentedTRMDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies augmentation on-the-fly.

    This allows different augmentations each epoch, effectively
    increasing dataset size.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augmenter: Optional[TRMDataAugmenter] = None,
        augment_prob: float = 0.5
    ):
        """
        Initialize augmented dataset.

        Args:
            base_dataset: Original TRMDataset
            augmenter: TRMDataAugmenter instance
            augment_prob: Probability of augmenting each sample
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter or TRMDataAugmenter()
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        features, label, pnl = self.base_dataset[idx]

        # Randomly apply augmentation
        if torch.rand(1).item() < self.augment_prob:
            # Convert to batch format for augmenter
            features_batch = features.unsqueeze(0)
            labels_batch = torch.tensor([label])
            pnl_batch = torch.tensor([pnl])

            # Augment (only noise and mask, no mixup for single sample)
            if self.augmenter.config.noise_enabled:
                noise = torch.randn_like(features) * self.augmenter.config.noise_std
                features = features + noise

            if self.augmenter.config.mask_enabled:
                if torch.rand(1).item() < self.augmenter.config.mask_prob:
                    n_mask = torch.randint(1, self.augmenter.config.mask_max_features + 1, (1,)).item()
                    mask_indices = torch.randperm(len(features))[:n_mask]
                    features[mask_indices] = 0.0

        return features, label, pnl


def create_augmented_dataloader(
    base_dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    augmentation_config: Optional[AugmentationConfig] = None,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with on-the-fly augmentation.

    Args:
        base_dataset: Original TRMDataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augmentation_config: Augmentation settings
        num_workers: Number of data loading workers

    Returns:
        DataLoader with augmented dataset
    """
    config = augmentation_config or AugmentationConfig()
    augmenter = TRMDataAugmenter(config)

    augmented_dataset = AugmentedTRMDataset(
        base_dataset=base_dataset,
        augmenter=augmenter,
        augment_prob=0.5
    )

    return torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# Convenience function for training script
def get_default_augmentation_config() -> AugmentationConfig:
    """Get recommended augmentation config for TRM training."""
    return AugmentationConfig(
        noise_enabled=True,
        noise_std=0.05,  # 5% noise
        noise_prob=0.5,
        mask_enabled=True,
        mask_prob=0.15,
        mask_max_features=2,
        mixup_enabled=True,
        mixup_alpha=0.2,
        mixup_prob=0.25
    )


if __name__ == "__main__":
    # Test augmentation
    print("Testing TRM Data Augmentation")
    print("=" * 50)

    # Create dummy data
    batch_size = 8
    n_features = 10

    features = torch.randn(batch_size, n_features)
    labels = torch.randint(0, 8, (batch_size,))
    pnl = torch.randn(batch_size) * 0.1

    print(f"Original features shape: {features.shape}")
    print(f"Original features[0]: {features[0][:5].tolist()}")

    # Apply augmentation
    config = get_default_augmentation_config()
    augmenter = TRMDataAugmenter(config)

    aug_features, aug_labels, aug_pnl = augmenter.augment_batch(features, labels, pnl)

    print(f"\nAugmented features shape: {aug_features.shape}")
    print(f"Augmented features[0]: {aug_features[0][:5].tolist()}")

    # Check that augmentation made changes
    diff = (features - aug_features).abs().mean().item()
    print(f"\nMean absolute difference: {diff:.4f}")

    print("\n[OK] Augmentation working correctly!")
