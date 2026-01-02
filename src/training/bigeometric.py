"""
Bigeometric Calculus - Gradient Transformation for MetaGrokFast.

SOURCE: the-agent-maker/src/cross_phase/meta_calculus/bigeometric.py

Key Property: D_BG[x^n] = e^n (independent of x!)
This makes power-law gradient explosions BOUNDED.

Gradient Transform: g_meta = g * |g|^(2k-1)
- When k < 0.5: dampens large gradients (exponent < 0)
- When k = 0.5: identity (exponent = 0, classical)
- When k > 0.5: amplifies large gradients (exponent > 0)

Benefits:
- Prevents gradient explosion without clipping
- Preserves gradient direction
- Scale-adaptive via k(L) formula
"""

import torch
from typing import Optional, Union, Tuple
from dataclasses import dataclass

from .k_formula import compute_k, k_from_gradient, KFormulaConfig


@dataclass
class BigeometricConfig:
    """Configuration for bigeometric operations."""
    eps: float = 1e-8
    max_magnitude: float = 1e6
    k_min: float = 0.0
    k_max: float = 1.0
    use_adaptive_k: bool = True
    k_formula_config: Optional[KFormulaConfig] = None


class BigeometricTransform:
    """
    Bigeometric gradient transformation for neural network training.

    Transform: g_meta = g * |g|^(2k-1)

    Provides:
    - Bounded gradients (prevents explosion)
    - Direction preservation (unlike clipping)
    - Scale-adaptive behavior via k(L)
    """

    def __init__(self, config: Optional[BigeometricConfig] = None):
        self.config = config or BigeometricConfig()

    def transform(
        self,
        grad: torch.Tensor,
        k: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Apply bigeometric transformation to gradient.

        Formula: g_meta = g * |g|^(2k-1)

        Args:
            grad: Input gradient tensor
            k: Meta-parameter (if None, computed adaptively)

        Returns:
            Transformed gradient
        """
        if k is None:
            if self.config.use_adaptive_k:
                k = k_from_gradient(grad, self.config.k_formula_config)
            else:
                k = 0.5  # Identity transform

        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=grad.device, dtype=grad.dtype)

        # g_meta = g * |g|^(2k-1)
        abs_grad = torch.abs(grad).clamp(min=self.config.eps)
        exponent = 2 * k - 1

        scale = abs_grad ** exponent
        scale = scale.clamp(max=self.config.max_magnitude)

        return grad * scale


def bigeometric_gradient_transform(
    grad: torch.Tensor,
    k: Optional[Union[float, torch.Tensor]] = None,
    config: Optional[BigeometricConfig] = None
) -> torch.Tensor:
    """
    Transform gradient using bigeometric calculus.

    This is the main function for MetaGrokFast integration.

    Args:
        grad: Input gradient
        k: Meta-parameter (adaptive if None)
        config: Bigeometric configuration

    Returns:
        Transformed gradient with controlled magnitude
    """
    transform = BigeometricTransform(config)
    return transform.transform(grad, k)


def bigeometric_gradient_with_stats(
    grad: torch.Tensor,
    k: Optional[Union[float, torch.Tensor]] = None,
    config: Optional[BigeometricConfig] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Transform gradient and return statistics for monitoring.

    Args:
        grad: Input gradient
        k: Meta-parameter
        config: Configuration

    Returns:
        Tuple of (transformed_gradient, statistics_dict)
    """
    config = config or BigeometricConfig()

    if k is None and config.use_adaptive_k:
        k = k_from_gradient(grad, config.k_formula_config)
    elif k is None:
        k = 0.5

    k_val = k.item() if isinstance(k, torch.Tensor) else k

    orig_norm = torch.norm(grad).item()
    orig_max = torch.abs(grad).max().item()

    transform = BigeometricTransform(config)
    grad_meta = transform.transform(grad, k)

    meta_norm = torch.norm(grad_meta).item()
    meta_max = torch.abs(grad_meta).max().item()

    stats = {
        "k": k_val,
        "original_norm": orig_norm,
        "transformed_norm": meta_norm,
        "original_max": orig_max,
        "transformed_max": meta_max,
        "compression_ratio": orig_norm / (meta_norm + 1e-8),
    }

    return grad_meta, stats


# Log-space operations for weight manipulation
def to_log_space(tensor: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform tensor to log-space (bigeometric domain)."""
    signs = torch.sign(tensor)
    log_magnitudes = torch.log(torch.abs(tensor) + eps)
    return log_magnitudes, signs


def from_log_space(log_magnitudes: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Transform from log-space back to linear space."""
    return signs * torch.exp(log_magnitudes)
