"""
Scheme-Robust Strategy Selector.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 5

Selects trading strategies requiring 60% model consensus.
On disagreement, defaults to balanced_safe (strategy 2).

Key Design:
- Multi-scheme validation: signals must agree across NNC and classical
- 60% consensus threshold
- Defensive default on disagreement
- Geometric mean confidence for robustness
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# NNC imports
try:
    from src.utils.multiplicative import GeometricOperations
    from src.utils.nnc_feature_flags import get_flag
    NNC_AVAILABLE = True
except ImportError:
    NNC_AVAILABLE = False

from src.strategies.robust_signals import (
    ModelPrediction,
    SignalResult,
    RobustSignalGenerator,
    STRATEGY_NAMES,
    DEFAULT_STRATEGY,
)

logger = logging.getLogger(__name__)


@dataclass
class SchemeValidation:
    """Validation result across calculation schemes."""
    classical_strategy: int
    nnc_strategy: int
    schemes_agree: bool
    classical_confidence: float
    nnc_confidence: float
    geo_mean_confidence: float
    divergence_pct: float


@dataclass
class RobustSelection:
    """Final robust strategy selection."""
    strategy_idx: int
    strategy_name: str
    confidence: float
    is_consensus: bool
    is_default: bool
    is_scheme_robust: bool
    scheme_validation: Optional[SchemeValidation]
    model_consensus_pct: float
    selection_time_ms: float
    reason: str


class SchemeRobustSelector:
    """
    Selects strategies that are robust across calculation schemes.

    A signal is "scheme-robust" if:
    1. Models agree (60%+ consensus)
    2. Classical and NNC calculations agree
    3. Confidence exceeds threshold

    On any disagreement, defaults to balanced_safe (strategy 2).
    """

    def __init__(
        self,
        consensus_threshold: float = 0.60,
        scheme_divergence_threshold: float = 0.10,
        min_confidence: float = 0.50,
        default_strategy: int = DEFAULT_STRATEGY,
    ):
        """
        Initialize scheme-robust selector.

        Args:
            consensus_threshold: Required model agreement (default 60%)
            scheme_divergence_threshold: Max allowed scheme divergence (default 10%)
            min_confidence: Minimum confidence to count vote
            default_strategy: Strategy on disagreement (balanced_safe)
        """
        self._consensus_threshold = consensus_threshold
        self._scheme_divergence_threshold = scheme_divergence_threshold
        self._min_confidence = min_confidence
        self._default_strategy = default_strategy

        self._signal_generator = RobustSignalGenerator(
            consensus_threshold=consensus_threshold,
            default_strategy=default_strategy,
            min_confidence=min_confidence,
        )

    def select(
        self,
        predictions: List[ModelPrediction],
        features: Optional[Dict[str, float]] = None,
    ) -> RobustSelection:
        """
        Select strategy with scheme-robustness validation.

        Args:
            predictions: Model predictions
            features: Optional market features for scheme validation

        Returns:
            RobustSelection with validated strategy
        """
        start_time = datetime.now()

        # Step 1: Model consensus
        signal = self._signal_generator.generate_signal(predictions)

        # Step 2: Scheme validation (if features provided and NNC available)
        scheme_validation = None
        is_scheme_robust = True

        if features and NNC_AVAILABLE:
            scheme_validation = self._validate_schemes(predictions, features)
            is_scheme_robust = scheme_validation.schemes_agree

        # Step 3: Final decision
        if signal.is_consensus and is_scheme_robust:
            strategy_idx = signal.strategy_idx
            is_default = False
            reason = "Consensus reached with scheme validation"
        elif signal.is_consensus and not is_scheme_robust:
            # Models agree but schemes diverge - use default
            strategy_idx = self._default_strategy
            is_default = True
            reason = f"Scheme divergence: {scheme_validation.divergence_pct:.1%}"
        else:
            # No model consensus
            strategy_idx = self._default_strategy
            is_default = True
            reason = f"No consensus: {signal.consensus_pct:.1%} < {self._consensus_threshold:.1%}"

        end_time = datetime.now()
        selection_time_ms = (end_time - start_time).total_seconds() * 1000

        return RobustSelection(
            strategy_idx=strategy_idx,
            strategy_name=STRATEGY_NAMES[strategy_idx],
            confidence=signal.confidence if not is_default else 0.5,
            is_consensus=signal.is_consensus,
            is_default=is_default,
            is_scheme_robust=is_scheme_robust,
            scheme_validation=scheme_validation,
            model_consensus_pct=signal.consensus_pct,
            selection_time_ms=selection_time_ms,
            reason=reason,
        )

    def _validate_schemes(
        self,
        predictions: List[ModelPrediction],
        features: Dict[str, float],
    ) -> SchemeValidation:
        """
        Validate that classical and NNC schemes agree.

        Args:
            predictions: Model predictions
            features: Market features

        Returns:
            SchemeValidation result
        """
        # Classical: arithmetic mean of confidences
        confidences = [p.confidence for p in predictions]
        classical_conf = np.mean(confidences) if confidences else 0.0

        # NNC: geometric mean of confidences
        if NNC_AVAILABLE:
            nnc_conf = GeometricOperations.geometric_mean(confidences) if confidences else 0.0
        else:
            nnc_conf = np.exp(np.mean(np.log(np.clip(confidences, 1e-10, 1.0)))) if confidences else 0.0

        # Strategy by classical (argmax of votes)
        vote_counts = {}
        for p in predictions:
            if p.strategy_idx not in vote_counts:
                vote_counts[p.strategy_idx] = 0
            vote_counts[p.strategy_idx] += 1

        classical_strategy = max(vote_counts.keys(), key=lambda k: vote_counts[k]) if vote_counts else self._default_strategy

        # Strategy by NNC (weighted by geometric confidence)
        weighted_votes = {}
        for p in predictions:
            weight = p.confidence  # Use confidence as weight
            if p.strategy_idx not in weighted_votes:
                weighted_votes[p.strategy_idx] = 0.0
            weighted_votes[p.strategy_idx] += weight

        nnc_strategy = max(weighted_votes.keys(), key=lambda k: weighted_votes[k]) if weighted_votes else self._default_strategy

        # Divergence
        divergence_pct = abs(classical_conf - nnc_conf)

        schemes_agree = (
            classical_strategy == nnc_strategy and
            divergence_pct <= self._scheme_divergence_threshold
        )

        # Geometric mean of both confidences
        geo_mean_conf = np.sqrt(classical_conf * nnc_conf)

        return SchemeValidation(
            classical_strategy=classical_strategy,
            nnc_strategy=nnc_strategy,
            schemes_agree=schemes_agree,
            classical_confidence=float(classical_conf),
            nnc_confidence=float(nnc_conf),
            geo_mean_confidence=float(geo_mean_conf),
            divergence_pct=float(divergence_pct),
        )

    def select_with_fallback_chain(
        self,
        predictions: List[ModelPrediction],
        features: Optional[Dict[str, float]] = None,
    ) -> RobustSelection:
        """
        Select with progressive fallback chain.

        Fallback order:
        1. Full consensus (60%+ models, schemes agree) -> selected strategy
        2. Model consensus only (60%+ models) -> selected strategy (lower confidence)
        3. Simple majority (>50%) -> selected strategy (even lower confidence)
        4. No majority -> balanced_safe (default)

        Args:
            predictions: Model predictions
            features: Optional market features

        Returns:
            RobustSelection with fallback handling
        """
        start_time = datetime.now()

        if not predictions:
            return self._create_default_selection(start_time, "No predictions")

        # Count votes
        vote_counts: Dict[int, int] = {}
        for p in predictions:
            if p.strategy_idx not in vote_counts:
                vote_counts[p.strategy_idx] = 0
            vote_counts[p.strategy_idx] += 1

        total_votes = len(predictions)
        winner_idx = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        winner_pct = vote_counts[winner_idx] / total_votes

        # Calculate confidences
        winner_preds = [p for p in predictions if p.strategy_idx == winner_idx]
        confidences = [p.confidence for p in winner_preds]

        if NNC_AVAILABLE:
            geo_conf = GeometricOperations.geometric_mean(confidences) if confidences else 0.0
        else:
            geo_conf = np.exp(np.mean(np.log(np.clip(confidences, 1e-10, 1.0)))) if confidences else 0.0

        # Scheme validation
        scheme_validation = None
        is_scheme_robust = True
        if features and NNC_AVAILABLE:
            scheme_validation = self._validate_schemes(predictions, features)
            is_scheme_robust = scheme_validation.schemes_agree

        # Fallback chain
        if winner_pct >= self._consensus_threshold and is_scheme_robust:
            # Full consensus
            strategy_idx = winner_idx
            confidence = geo_conf
            is_default = False
            reason = f"Full consensus: {winner_pct:.1%} agreement, schemes match"
        elif winner_pct >= self._consensus_threshold:
            # Model consensus only
            strategy_idx = winner_idx
            confidence = geo_conf * 0.8  # Reduce confidence
            is_default = False
            reason = f"Model consensus: {winner_pct:.1%}, but scheme divergence"
        elif winner_pct > 0.50:
            # Simple majority
            strategy_idx = winner_idx
            confidence = geo_conf * 0.6  # Lower confidence
            is_default = False
            reason = f"Simple majority: {winner_pct:.1%}"
        else:
            # No majority - default
            strategy_idx = self._default_strategy
            confidence = 0.5
            is_default = True
            reason = f"No majority: {winner_pct:.1%}"

        end_time = datetime.now()
        selection_time_ms = (end_time - start_time).total_seconds() * 1000

        return RobustSelection(
            strategy_idx=strategy_idx,
            strategy_name=STRATEGY_NAMES[strategy_idx],
            confidence=float(confidence),
            is_consensus=winner_pct >= self._consensus_threshold,
            is_default=is_default,
            is_scheme_robust=is_scheme_robust,
            scheme_validation=scheme_validation,
            model_consensus_pct=winner_pct,
            selection_time_ms=selection_time_ms,
            reason=reason,
        )

    def _create_default_selection(
        self,
        start_time: datetime,
        reason: str,
    ) -> RobustSelection:
        """Create default selection result."""
        end_time = datetime.now()
        selection_time_ms = (end_time - start_time).total_seconds() * 1000

        return RobustSelection(
            strategy_idx=self._default_strategy,
            strategy_name=STRATEGY_NAMES[self._default_strategy],
            confidence=0.5,
            is_consensus=False,
            is_default=True,
            is_scheme_robust=False,
            scheme_validation=None,
            model_consensus_pct=0.0,
            selection_time_ms=selection_time_ms,
            reason=reason,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_robust_strategy(
    predictions: List[ModelPrediction],
    consensus_threshold: float = 0.60,
) -> RobustSelection:
    """
    Convenience function for robust strategy selection.

    Args:
        predictions: Model predictions
        consensus_threshold: Required agreement percentage

    Returns:
        RobustSelection result
    """
    selector = SchemeRobustSelector(consensus_threshold=consensus_threshold)
    return selector.select(predictions)


def get_default_strategy() -> Tuple[int, str]:
    """Get the default strategy (balanced_safe)."""
    return (DEFAULT_STRATEGY, STRATEGY_NAMES[DEFAULT_STRATEGY])
