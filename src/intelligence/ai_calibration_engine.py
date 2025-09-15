"""
AI Self-Calibrating Decision Engine
Implements mathematical framework for AI to learn its own risk tolerances and decision curves.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class AIPrediction:
    """Single AI prediction with outcome tracking"""
    id: str
    timestamp: datetime
    prediction: float  # 0-1 probability
    confidence: float  # 0-1 confidence level
    actual_outcome: Optional[bool] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class AIUtilityParameters:
    """AI's learned utility function parameters"""
    risk_aversion: float = 0.5  # γ parameter in CRRA utility
    loss_aversion: float = 2.0  # Prospect theory loss aversion multiplier
    kelly_safety_factor: float = 0.25  # k ∈ [0.2, 0.5]
    confidence_threshold: float = 0.7  # Minimum confidence for action
    learning_rate: float = 0.01
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AICalibrationMetrics:
    """AI's calibration performance metrics"""
    total_predictions: int = 0
    resolved_predictions: int = 0
    overall_accuracy: float = 0.0
    brier_score: float = 1.0  # Lower is better
    log_loss: float = np.inf  # Lower is better
    calibration_error: float = 1.0  # Lower is better
    pit_p_value: float = 0.0  # Higher is better (should be > 0.05)
    confidence_bins: Dict[str, Dict[str, float]] = field(default_factory=dict)

class AICalibrationEngine:
    """
    Core AI self-calibration system implementing mathematical framework:
    - Brier scoring for prediction accuracy
    - PIT testing for calibration quality
    - Dynamic utility function learning
    - Kelly-lite position sizing with learned parameters
    """

    def __init__(self, persistence_path: str = "data/ai_calibration.pkl"):
        self.persistence_path = persistence_path
        self.predictions: List[AIPrediction] = []
        self.utility_params = AIUtilityParameters()
        self.calibration_metrics = AICalibrationMetrics()

        # Load existing calibration data if available
        self._load_calibration_data()

    def make_prediction(self,
                       prediction_value: float,
                       confidence: float,
                       context: Dict[str, Any] = None) -> str:
        """
        AI makes a prediction and records it for later calibration

        Args:
            prediction_value: 0-1 probability of positive outcome
            confidence: 0-1 confidence level in this prediction
            context: Additional context for the prediction

        Returns:
            prediction_id: Unique identifier for tracking
        """
        prediction_id = f"ai_pred_{len(self.predictions)}_{datetime.now().isoformat()}"

        prediction = AIPrediction(
            id=prediction_id,
            timestamp=datetime.now(),
            prediction=prediction_value,
            confidence=confidence,
            context=context or {}
        )

        self.predictions.append(prediction)
        logger.info(f"AI made prediction {prediction_id}: {prediction_value:.3f} (conf: {confidence:.3f})")

        # Auto-save after each prediction
        self._save_calibration_data()

        return prediction_id

    def resolve_prediction(self, prediction_id: str, actual_outcome: bool) -> bool:
        """
        Resolve a prediction with actual outcome and update AI calibration

        Args:
            prediction_id: ID from make_prediction
            actual_outcome: True/False actual result

        Returns:
            success: Whether resolution was successful
        """
        for prediction in self.predictions:
            if prediction.id == prediction_id:
                prediction.actual_outcome = actual_outcome
                prediction.resolved = True
                prediction.resolution_timestamp = datetime.now()

                logger.info(f"Resolved prediction {prediction_id}: {actual_outcome}")

                # Update AI calibration metrics
                self._update_calibration_metrics()

                # Update AI utility parameters based on performance
                self._update_utility_parameters(prediction, actual_outcome)

                # Save updated calibration
                self._save_calibration_data()

                return True

        logger.warning(f"Prediction {prediction_id} not found for resolution")
        return False

    def calculate_ai_utility(self, outcome: float, baseline: float = 0.0) -> float:
        """
        Calculate AI's personal utility using learned parameters

        Uses CRRA utility function: U(x) = x^(1-γ) / (1-γ)
        With prospect theory for losses
        """
        if outcome >= baseline:
            # Gains: CRRA utility
            if self.utility_params.risk_aversion == 1.0:
                return np.log(outcome + 1e-8)  # Log utility at γ=1
            else:
                return ((outcome + 1e-8) ** (1 - self.utility_params.risk_aversion)) / (1 - self.utility_params.risk_aversion)
        else:
            # Losses: Prospect theory with loss aversion
            loss = baseline - outcome
            return -self.utility_params.loss_aversion * (loss ** (1 - self.utility_params.risk_aversion))

    def calculate_ai_kelly_fraction(self, expected_return: float, variance: float) -> float:
        """
        Calculate AI's personal Kelly fraction with learned safety factor

        f* = μ/σ² (full Kelly)
        f = k * f* where k is AI's learned safety factor
        """
        if variance <= 0:
            return 0.0

        full_kelly = expected_return / variance
        ai_kelly = self.utility_params.kelly_safety_factor * full_kelly

        # Ensure reasonable bounds
        return max(0.0, min(0.5, ai_kelly))

    def get_ai_confidence_adjustment(self, stated_confidence: float) -> float:
        """
        Adjust stated confidence based on AI's historical calibration accuracy

        Returns calibrated confidence level
        """
        if not self.calibration_metrics.confidence_bins:
            return stated_confidence

        # Find closest calibration bin
        closest_bin = min(
            self.calibration_metrics.confidence_bins.keys(),
            key=lambda x: abs(float(x) - stated_confidence)
        )

        bin_data = self.calibration_metrics.confidence_bins[closest_bin]

        if bin_data.get('count', 0) >= 10:  # Require minimum sample size
            return bin_data['accuracy']

        return stated_confidence

    def calculate_brier_score(self) -> float:
        """
        Calculate Brier score for AI's predictions: (p - y)²
        Lower is better (0 = perfect, 1 = worst possible)
        """
        resolved_predictions = [p for p in self.predictions if p.resolved]

        if not resolved_predictions:
            return 1.0

        brier_scores = []
        for pred in resolved_predictions:
            outcome = 1.0 if pred.actual_outcome else 0.0
            brier_score = (pred.prediction - outcome) ** 2
            brier_scores.append(brier_score)

        return np.mean(brier_scores)

    def calculate_log_loss(self) -> float:
        """
        Calculate log loss for AI's predictions: -[y*ln(p) + (1-y)*ln(1-p)]
        Lower is better
        """
        resolved_predictions = [p for p in self.predictions if p.resolved]

        if not resolved_predictions:
            return np.inf

        log_losses = []
        for pred in resolved_predictions:
            outcome = 1.0 if pred.actual_outcome else 0.0
            # Clip predictions to avoid log(0)
            p = max(1e-15, min(1-1e-15, pred.prediction))
            log_loss = -(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))
            log_losses.append(log_loss)

        return np.mean(log_losses)

    def perform_pit_test(self) -> float:
        """
        Perform Probability Integral Transform test for calibration

        Returns p-value from Kolmogorov-Smirnov test
        Higher p-value (>0.05) indicates good calibration
        """
        resolved_predictions = [p for p in self.predictions if p.resolved]

        if len(resolved_predictions) < 10:
            return 0.0

        # Calculate PIT values: u = F(y) where F is the predicted CDF
        pit_values = []
        for pred in resolved_predictions:
            if pred.actual_outcome:
                # Outcome was positive, u = predicted probability
                pit_values.append(pred.prediction)
            else:
                # Outcome was negative, u = 1 - predicted probability
                pit_values.append(1 - pred.prediction)

        # Test if PIT values are uniform using KS test
        ks_statistic, p_value = stats.kstest(pit_values, 'uniform')

        return p_value

    def _update_calibration_metrics(self):
        """Update all calibration metrics based on current predictions"""
        resolved_predictions = [p for p in self.predictions if p.resolved]

        if not resolved_predictions:
            return

        # Basic metrics
        self.calibration_metrics.total_predictions = len(self.predictions)
        self.calibration_metrics.resolved_predictions = len(resolved_predictions)

        # Accuracy
        correct = sum(1 for p in resolved_predictions
                     if (p.prediction >= 0.5) == p.actual_outcome)
        self.calibration_metrics.overall_accuracy = correct / len(resolved_predictions)

        # Scoring metrics
        self.calibration_metrics.brier_score = self.calculate_brier_score()
        self.calibration_metrics.log_loss = self.calculate_log_loss()
        self.calibration_metrics.pit_p_value = self.perform_pit_test()

        # Confidence bin analysis
        self._update_confidence_bins()

        # Calculate overall calibration error
        if self.calibration_metrics.confidence_bins:
            calibration_errors = [
                abs(bin_data['accuracy'] - bin_data['target_confidence'])
                for bin_data in self.calibration_metrics.confidence_bins.values()
                if bin_data['count'] >= 5
            ]
            self.calibration_metrics.calibration_error = np.mean(calibration_errors) if calibration_errors else 1.0

    def _update_confidence_bins(self):
        """Update calibration accuracy by confidence level"""
        resolved_predictions = [p for p in self.predictions if p.resolved]

        if not resolved_predictions:
            return

        # Define confidence bins
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_data = {}

        for bin_center in confidence_bins:
            # Find predictions within ±0.05 of this confidence level
            bin_predictions = [
                p for p in resolved_predictions
                if abs(p.confidence - bin_center) <= 0.05
            ]

            if bin_predictions:
                correct = sum(1 for p in bin_predictions
                             if (p.prediction >= 0.5) == p.actual_outcome)
                accuracy = correct / len(bin_predictions)

                bin_data[str(bin_center)] = {
                    'accuracy': accuracy,
                    'count': len(bin_predictions),
                    'target_confidence': bin_center,
                    'calibration_error': abs(accuracy - bin_center)
                }

        self.calibration_metrics.confidence_bins = bin_data

    def _update_utility_parameters(self, prediction: AIPrediction, actual_outcome: bool):
        """
        Update AI's utility parameters based on prediction performance

        Implements learning algorithm to adjust risk aversion and Kelly factor
        """
        was_correct = (prediction.prediction >= 0.5) == actual_outcome
        confidence_error = abs(prediction.confidence - (1.0 if was_correct else 0.0))

        # Update risk aversion based on performance
        if was_correct:
            # Correct prediction - become slightly less risk averse
            adjustment = -self.utility_params.learning_rate * confidence_error
        else:
            # Incorrect prediction - become more risk averse
            adjustment = self.utility_params.learning_rate * (1 + confidence_error)

        self.utility_params.risk_aversion = max(
            0.1, min(2.0, self.utility_params.risk_aversion + adjustment)
        )

        # Update Kelly safety factor based on Brier score
        current_brier = self.calibration_metrics.brier_score
        kelly_adjustment = -self.utility_params.learning_rate * current_brier

        self.utility_params.kelly_safety_factor = max(
            0.1, min(0.5, self.utility_params.kelly_safety_factor + kelly_adjustment)
        )

        # Update confidence threshold based on calibration error
        if self.calibration_metrics.calibration_error < 0.1:
            # Well calibrated - can be more confident
            self.utility_params.confidence_threshold = max(
                0.5, self.utility_params.confidence_threshold - 0.01
            )
        else:
            # Poorly calibrated - require higher confidence
            self.utility_params.confidence_threshold = min(
                0.9, self.utility_params.confidence_threshold + 0.01
            )

        self.utility_params.last_updated = datetime.now()

        logger.info(f"Updated AI parameters: risk_aversion={self.utility_params.risk_aversion:.3f}, "
                   f"kelly_factor={self.utility_params.kelly_safety_factor:.3f}, "
                   f"confidence_threshold={self.utility_params.confidence_threshold:.3f}")

    def get_ai_decision_confidence(self, base_confidence: float) -> float:
        """
        Get AI's adjusted decision confidence based on calibration history
        """
        # Start with base confidence
        adjusted_confidence = base_confidence

        # Adjust based on historical calibration
        calibrated_confidence = self.get_ai_confidence_adjustment(base_confidence)

        # Blend base and calibrated confidence
        blend_factor = min(1.0, self.calibration_metrics.resolved_predictions / 50.0)
        adjusted_confidence = (1 - blend_factor) * base_confidence + blend_factor * calibrated_confidence

        # Apply confidence threshold
        if adjusted_confidence < self.utility_params.confidence_threshold:
            return 0.0  # Below threshold - no decision

        return adjusted_confidence

    def export_calibration_report(self) -> Dict[str, Any]:
        """Export comprehensive calibration report for UI display"""
        return {
            'utility_parameters': {
                'risk_aversion': self.utility_params.risk_aversion,
                'loss_aversion': self.utility_params.loss_aversion,
                'kelly_safety_factor': self.utility_params.kelly_safety_factor,
                'confidence_threshold': self.utility_params.confidence_threshold,
                'last_updated': self.utility_params.last_updated.isoformat()
            },
            'calibration_metrics': {
                'total_predictions': self.calibration_metrics.total_predictions,
                'resolved_predictions': self.calibration_metrics.resolved_predictions,
                'overall_accuracy': self.calibration_metrics.overall_accuracy,
                'brier_score': self.calibration_metrics.brier_score,
                'log_loss': self.calibration_metrics.log_loss,
                'calibration_error': self.calibration_metrics.calibration_error,
                'pit_p_value': self.calibration_metrics.pit_p_value,
                'confidence_bins': self.calibration_metrics.confidence_bins
            },
            'recent_predictions': [
                {
                    'id': p.id,
                    'timestamp': p.timestamp.isoformat(),
                    'prediction': p.prediction,
                    'confidence': p.confidence,
                    'actual_outcome': p.actual_outcome,
                    'resolved': p.resolved
                }
                for p in self.predictions[-10:]  # Last 10 predictions
            ]
        }

    def _save_calibration_data(self):
        """Save calibration data to disk"""
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

            data = {
                'predictions': self.predictions,
                'utility_params': self.utility_params,
                'calibration_metrics': self.calibration_metrics
            }

            with open(self.persistence_path, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")

    def _load_calibration_data(self):
        """Load calibration data from disk"""
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'rb') as f:
                    data = pickle.load(f)

                self.predictions = data.get('predictions', [])
                self.utility_params = data.get('utility_params', AIUtilityParameters())
                self.calibration_metrics = data.get('calibration_metrics', AICalibrationMetrics())

                logger.info(f"Loaded {len(self.predictions)} predictions from calibration data")

        except Exception as e:
            logger.warning(f"Could not load calibration data: {e}")
            # Initialize with defaults
            self.predictions = []
            self.utility_params = AIUtilityParameters()
            self.calibration_metrics = AICalibrationMetrics()

# Global AI calibration engine instance
ai_calibration_engine = AICalibrationEngine()