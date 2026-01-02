"""
k(L) Formula - Scale-adaptive k parameter from Meta-Calculus MOO.

SOURCE: the-agent-maker/src/cross_phase/meta_calculus/k_formula.py
WARNING: Coefficients K_SLOPE=-0.0137, K_INTERCEPT=0.1593 are from PHYSICS domain
         (R^2 = 0.71, p = 0.008 in original meta-calculus MOO context)
         NOT YET VALIDATED for ML loss landscapes in trader-ai

Formula: k(L) = -0.0137 * log10(L) + 0.1593

CRITICAL NOTES:
1. log10 vs ln matters: log10(100) = 2, ln(100) = 4.6 (35% error if wrong)
2. Coefficients may need ML-specific tuning (see STREAM4-FIXES.md)
3. For production, consider conservative fixed k = 0.15 until validated

The k parameter provides scale-dependent adaptation:
- Higher gradient magnitude -> lower k -> behavior depends on domain
- Lower gradient magnitude -> higher k -> behavior depends on domain

Applications in Trader-AI:
- TRM gradient filtering strength
- NNC-adjusted learning rates
- Loss weighting by trade magnitude
"""

import torch
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass


# Verified coefficients from meta-calculus MOO optimization
K_SLOPE = -0.0137
K_INTERCEPT = 0.1593
K_MIN = 0.0
K_MAX = 1.0


@dataclass
class KFormulaConfig:
    """Configuration for k(L) formula behavior."""
    slope: float = K_SLOPE
    intercept: float = K_INTERCEPT
    k_min: float = K_MIN
    k_max: float = K_MAX
    log_base: float = 10.0
    eps: float = 1e-8
    gradient_scale: float = 1.0
    layer_scale: float = 1.0


def compute_k(
    L: Union[float, np.ndarray, torch.Tensor],
    config: Optional[KFormulaConfig] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute k parameter from scale L.

    Formula: k(L) = -0.0137 * log10(L) + 0.1593

    Args:
        L: Scale parameter (gradient magnitude, layer index, etc.)
        config: Optional configuration

    Returns:
        k value(s) clamped to [k_min, k_max]
    """
    if config is None:
        config = KFormulaConfig()

    if isinstance(L, torch.Tensor):
        return _compute_k_torch(L, config)
    elif isinstance(L, np.ndarray):
        return _compute_k_numpy(L, config)
    else:
        return _compute_k_scalar(float(L), config)


def _compute_k_scalar(L: float, config: KFormulaConfig) -> float:
    """Scalar implementation of k(L)."""
    L_safe = max(L, config.eps)
    log_L = np.log10(L_safe)
    k = config.slope * log_L + config.intercept
    return max(config.k_min, min(config.k_max, k))


def _compute_k_numpy(L: np.ndarray, config: KFormulaConfig) -> np.ndarray:
    """NumPy implementation of k(L)."""
    L_safe = np.maximum(L, config.eps)
    log_L = np.log10(L_safe)
    k = config.slope * log_L + config.intercept
    return np.clip(k, config.k_min, config.k_max)


def _compute_k_torch(L: torch.Tensor, config: KFormulaConfig) -> torch.Tensor:
    """PyTorch implementation of k(L)."""
    L_safe = torch.clamp(L, min=config.eps)
    log_L = torch.log10(L_safe)
    k = config.slope * log_L + config.intercept
    return torch.clamp(k, config.k_min, config.k_max)


def k_from_gradient(
    grad: Union[torch.Tensor, np.ndarray],
    config: Optional[KFormulaConfig] = None
) -> Union[torch.Tensor, float]:
    """
    Compute k from gradient magnitude for MetaGrokFast.

    Args:
        grad: Gradient tensor or array
        config: Optional configuration

    Returns:
        k value for gradient transformation
    """
    if config is None:
        config = KFormulaConfig()

    if isinstance(grad, torch.Tensor):
        L = torch.norm(grad) * config.gradient_scale
    else:
        L = np.linalg.norm(grad) * config.gradient_scale

    return compute_k(L, config)


def k_from_layer_index(
    layer_idx: int,
    total_layers: int,
    config: Optional[KFormulaConfig] = None
) -> float:
    """
    Compute k from layer position for layer-wise operations.

    Early layers -> higher k -> more conservative
    Later layers -> lower k -> more aggressive

    Args:
        layer_idx: Current layer index (0-based)
        total_layers: Total number of layers
        config: Optional configuration

    Returns:
        k value for this layer
    """
    if config is None:
        config = KFormulaConfig()

    L = max(0.01, (layer_idx + 1) / total_layers) * config.layer_scale
    return compute_k(L, config)


def k_from_pnl(
    pnl: float,
    scale: float = 0.05,
    config: Optional[KFormulaConfig] = None
) -> float:
    """
    Compute k from PnL for loss weighting.

    Large PnL (positive or negative) -> lower k -> more learning
    Small PnL -> higher k -> less learning

    Args:
        pnl: Profit/loss value
        scale: Scaling factor
        config: Optional configuration

    Returns:
        k value for loss weighting
    """
    if config is None:
        config = KFormulaConfig()

    L = max(config.eps, abs(pnl) / scale)
    return compute_k(L, config)
