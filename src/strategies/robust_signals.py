"""
Robust Signals - Weighted consensus signal generation.

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1 Phase 5

Generates trading signals that require multi-model consensus.
Uses geometric mean confidence (NNC) instead of arithmetic mean.

Key Design:
- Signals require 60% model consensus
- On disagreement, default to balanced_safe (strategy 2)
- Geometric mean confidence is more robust to outliers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# NNC imports
try:
    from src.utils.multiplicative import GeometricOperations
    NNC_AVAILABLE = True
except ImportError:
    NNC_AVAILABLE = False

logger = logging.getLogger(__name__)


# Strategy indices (from strategy_labeler.py)
STRATEGY_NAMES = [
    'ultra_defensive',    # 0: 20/50/30 SPY/TLT/Cash
    'defensive',          # 1: 40/30/30
    'balanced_safe',      # 2: 60/20/20 - DEFAULT ON DISAGREEMENT
    'balanced_growth',    # 3: 70/20/10
    'growth',             # 4: 80/15/5
    'aggressive_growth',  # 5: 90/10/0
    'contrarian_long',    # 6: 85/15/0
    'tactical_opportunity' # 7: 75/25/0
]

DEFAULT_STRATEGY = 2  # balanced_safe


@dataclass
class SignalResult:
    """Result of signal generation."""
    strategy_idx: int
    strategy_name: str
    confidence: float
    consensus_pct: float
    is_consensus: bool
    is_default: bool
    model_votes: Dict[str, int]
    model_confidences: Dict[str, float]
    geo_mean_confidence: float
    arith_mean_confidence: float
    timestamp: datetime
    generation_time_ms: float


@dataclass
class ModelPrediction:
    """Single model's prediction."""
    model_name: str
    strategy_idx: int
    confidence: float
    raw_logits: Optional[np.ndarray] = None


class RobustSignalGenerator:
    """
    Generate trading signals with multi-model consensus.

    Uses geometric mean confidence (NNC) for robustness.
    Requires 60% model consensus, else defaults to balanced_safe.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.60,
        default_strategy: int = DEFAULT_STRATEGY,
        min_confidence: float = 0.50,
    ):
        """
        Initialize robust signal generator.

        Args:
            consensus_threshold: Required agreement percentage (default 60%)
            default_strategy: Strategy on disagreement (default balanced_safe)
            min_confidence: Minimum confidence to count a vote
        """
        self._consensus_threshold = consensus_threshold
        self._default_strategy = default_strategy
        self._min_confidence = min_confidence

    def generate_signal(
        self,
        predictions: List[ModelPrediction],
    ) -> SignalResult:
        """
        Generate consensus signal from multiple model predictions.

        Args:
            predictions: List of ModelPrediction from different models

        Returns:
            SignalResult with consensus strategy or default
        """
        start_time = datetime.now()

        if not predictions:
            return self._create_default_result(start_time, "No predictions")

        # Count votes by strategy
        votes: Dict[int, List[ModelPrediction]] = {}
        for pred in predictions:
            if pred.confidence >= self._min_confidence:
                if pred.strategy_idx not in votes:
                    votes[pred.strategy_idx] = []
                votes[pred.strategy_idx].append(pred)

        if not votes:
            return self._create_default_result(start_time, "No confident votes")

        # Find winning strategy
        total_votes = sum(len(v) for v in votes.values())
        winner_idx = max(votes.keys(), key=lambda k: len(votes[k]))
        winner_votes = len(votes[winner_idx])
        consensus_pct = winner_votes / total_votes

        # Check consensus threshold
        is_consensus = consensus_pct >= self._consensus_threshold

        if is_consensus:
            strategy_idx = winner_idx
            is_default = False
        else:
            strategy_idx = self._default_strategy
            is_default = True

        # Calculate confidences
        winner_preds = votes.get(winner_idx, [])
        confidences = [p.confidence for p in winner_preds]

        if confidences:
            if NNC_AVAILABLE:
                geo_mean_conf = GeometricOperations.geometric_mean(confidences)
            else:
                geo_mean_conf = np.exp(np.mean(np.log(np.clip(confidences, 1e-10, 1.0))))
            arith_mean_conf = np.mean(confidences)
        else:
            geo_mean_conf = 0.0
            arith_mean_conf = 0.0

        # Build result
        model_votes = {STRATEGY_NAMES[k]: len(v) for k, v in votes.items()}
        model_confidences = {p.model_name: p.confidence for p in predictions}

        end_time = datetime.now()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000

        return SignalResult(
            strategy_idx=strategy_idx,
            strategy_name=STRATEGY_NAMES[strategy_idx],
            confidence=geo_mean_conf if is_consensus else 0.5,
            consensus_pct=consensus_pct,
            is_consensus=is_consensus,
            is_default=is_default,
            model_votes=model_votes,
            model_confidences=model_confidences,
            geo_mean_confidence=geo_mean_conf,
            arith_mean_confidence=arith_mean_conf,
            timestamp=start_time,
            generation_time_ms=generation_time_ms,
        )

    def _create_default_result(
        self,
        timestamp: datetime,
        reason: str,
    ) -> SignalResult:
        """Create default result when no consensus possible."""
        end_time = datetime.now()
        generation_time_ms = (end_time - timestamp).total_seconds() * 1000

        logger.warning(f"Defaulting to balanced_safe: {reason}")

        return SignalResult(
            strategy_idx=self._default_strategy,
            strategy_name=STRATEGY_NAMES[self._default_strategy],
            confidence=0.5,
            consensus_pct=0.0,
            is_consensus=False,
            is_default=True,
            model_votes={},
            model_confidences={},
            geo_mean_confidence=0.0,
            arith_mean_confidence=0.0,
            timestamp=timestamp,
            generation_time_ms=generation_time_ms,
        )

    def generate_weighted_signal(
        self,
        predictions: List[ModelPrediction],
        weights: Optional[Dict[str, float]] = None,
    ) -> SignalResult:
        """
        Generate weighted consensus signal.

        Models with higher weights have more influence.

        Args:
            predictions: List of model predictions
            weights: Optional dict of model_name -> weight (default: equal)

        Returns:
            SignalResult with weighted consensus
        """
        start_time = datetime.now()

        if not predictions:
            return self._create_default_result(start_time, "No predictions")

        # Default to equal weights
        if weights is None:
            weights = {p.model_name: 1.0 for p in predictions}

        # Weighted votes by strategy
        weighted_votes: Dict[int, float] = {}
        for pred in predictions:
            if pred.confidence >= self._min_confidence:
                weight = weights.get(pred.model_name, 1.0)
                weighted_vote = weight * pred.confidence
                if pred.strategy_idx not in weighted_votes:
                    weighted_votes[pred.strategy_idx] = 0.0
                weighted_votes[pred.strategy_idx] += weighted_vote

        if not weighted_votes:
            return self._create_default_result(start_time, "No confident weighted votes")

        # Find winner by weighted sum
        total_weight = sum(weighted_votes.values())
        winner_idx = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        consensus_pct = weighted_votes[winner_idx] / total_weight

        is_consensus = consensus_pct >= self._consensus_threshold

        if is_consensus:
            strategy_idx = winner_idx
            is_default = False
        else:
            strategy_idx = self._default_strategy
            is_default = True

        # Calculate confidences from predictions voting for winner
        winner_preds = [p for p in predictions if p.strategy_idx == winner_idx]
        confidences = [p.confidence for p in winner_preds]

        if confidences and NNC_AVAILABLE:
            geo_mean_conf = GeometricOperations.geometric_mean(confidences)
        elif confidences:
            geo_mean_conf = np.exp(np.mean(np.log(np.clip(confidences, 1e-10, 1.0))))
        else:
            geo_mean_conf = 0.0

        arith_mean_conf = np.mean(confidences) if confidences else 0.0

        model_votes = {
            STRATEGY_NAMES[k]: v for k, v in weighted_votes.items()
        }
        model_confidences = {p.model_name: p.confidence for p in predictions}

        end_time = datetime.now()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000

        return SignalResult(
            strategy_idx=strategy_idx,
            strategy_name=STRATEGY_NAMES[strategy_idx],
            confidence=geo_mean_conf if is_consensus else 0.5,
            consensus_pct=consensus_pct,
            is_consensus=is_consensus,
            is_default=is_default,
            model_votes=model_votes,
            model_confidences=model_confidences,
            geo_mean_confidence=geo_mean_conf,
            arith_mean_confidence=arith_mean_conf,
            timestamp=start_time,
            generation_time_ms=generation_time_ms,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_consensus_signal(
    predictions: List[ModelPrediction],
    consensus_threshold: float = 0.60,
) -> SignalResult:
    """
    Convenience function to generate consensus signal.

    Args:
        predictions: List of model predictions
        consensus_threshold: Required agreement percentage

    Returns:
        SignalResult with consensus or default strategy
    """
    generator = RobustSignalGenerator(consensus_threshold=consensus_threshold)
    return generator.generate_signal(predictions)


def create_prediction(
    model_name: str,
    strategy_idx: int,
    confidence: float,
) -> ModelPrediction:
    """
    Create a ModelPrediction.

    Args:
        model_name: Name of the model
        strategy_idx: Predicted strategy index (0-7)
        confidence: Confidence score (0-1)

    Returns:
        ModelPrediction instance
    """
    return ModelPrediction(
        model_name=model_name,
        strategy_idx=strategy_idx,
        confidence=confidence,
    )
