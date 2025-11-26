"""
Simple Brier Score Tracker for Position Sizing

Tracks prediction accuracy using Brier Score and adjusts Kelly criterion position sizing.
The Brier score measures the mean squared difference between predicted probabilities and actual outcomes.

Formula: brier_score = mean((forecast - outcome)^2)
- Lower scores indicate better predictions (0 = perfect, 1 = worst possible)
- Integrated with Kelly sizing: adjusted_kelly = base_kelly * (1 - brier_score)
"""

import numpy as np
from typing import List
from datetime import datetime


class BrierTracker:
    """Simple Brier score tracker with Kelly criterion integration"""

    def __init__(self, window_size: int = 100):
        """
        Initialize Brier tracker

        Args:
            window_size: Number of recent predictions to include in score calculation
        """
        self.window_size = window_size
        self.forecasts: List[float] = []
        self.outcomes: List[float] = []
        self.timestamps: List[datetime] = []

    def record_prediction(self, forecast: float, actual_outcome: float) -> None:
        """
        Record a prediction and its outcome

        Args:
            forecast: Predicted probability (0.0 to 1.0)
            actual_outcome: Actual binary outcome (0.0 or 1.0)
        """
        # Validate inputs
        if not (0.0 <= forecast <= 1.0):
            raise ValueError(f"Forecast must be between 0.0 and 1.0, got {forecast}")
        if actual_outcome not in [0.0, 1.0]:
            raise ValueError(f"Actual outcome must be 0.0 or 1.0, got {actual_outcome}")

        # Add new prediction
        self.forecasts.append(forecast)
        self.outcomes.append(actual_outcome)
        self.timestamps.append(datetime.now())

        # Maintain window size
        if len(self.forecasts) > self.window_size:
            self.forecasts.pop(0)
            self.outcomes.pop(0)
            self.timestamps.pop(0)

    def get_brier_score(self) -> float:
        """
        Calculate current Brier score

        Returns:
            Brier score (lower is better, 0 = perfect, 1 = worst)
            Returns 0.5 (neutral) if no predictions recorded
        """
        if len(self.forecasts) == 0:
            return 0.5  # Neutral score when no data

        forecasts_array = np.array(self.forecasts)
        outcomes_array = np.array(self.outcomes)

        # Brier score formula: mean((forecast - outcome)^2)
        squared_errors = (forecasts_array - outcomes_array) ** 2
        brier_score = np.mean(squared_errors)

        return float(brier_score)

    def get_prediction_count(self) -> int:
        """Get number of predictions recorded"""
        return len(self.forecasts)

    def get_accuracy(self) -> float:
        """
        Calculate simple accuracy (percentage of correct binary predictions)

        Returns:
            Accuracy as decimal (0.0 to 1.0)
        """
        if len(self.forecasts) == 0:
            return 0.5

        # Convert forecasts to binary predictions (threshold at 0.5)
        binary_predictions = [1.0 if f >= 0.5 else 0.0 for f in self.forecasts]
        correct_predictions = sum(1 for pred, actual in zip(binary_predictions, self.outcomes) if pred == actual)

        return correct_predictions / len(self.forecasts)

    def adjust_kelly_sizing(self, base_kelly: float) -> float:
        """
        Adjust Kelly criterion position sizing based on Brier score

        Args:
            base_kelly: Base Kelly criterion percentage (0.0 to 1.0)

        Returns:
            Adjusted Kelly percentage considering prediction accuracy
        """
        if base_kelly <= 0:
            return 0.0

        brier_score = self.get_brier_score()

        # Adjustment factor: (1 - brier_score)
        # Perfect predictions (brier=0) -> full Kelly size
        # Worst predictions (brier=1) -> zero position size
        # Random predictions (brier=0.5) -> 50% Kelly size
        adjustment_factor = 1.0 - brier_score

        # Ensure adjustment factor is non-negative
        adjustment_factor = max(0.0, adjustment_factor)

        adjusted_kelly = base_kelly * adjustment_factor

        return adjusted_kelly

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with Brier score, accuracy, and prediction count
        """
        return {
            'brier_score': self.get_brier_score(),
            'accuracy': self.get_accuracy(),
            'prediction_count': self.get_prediction_count(),
            'last_prediction_time': self.timestamps[-1] if self.timestamps else None
        }

    def reset(self) -> None:
        """Clear all recorded predictions"""
        self.forecasts.clear()
        self.outcomes.clear()
        self.timestamps.clear()


# Simple example usage
if __name__ == "__main__":
    # Create tracker
    tracker = BrierTracker(window_size=50)

    # Example predictions and outcomes
    examples = [
        (0.8, 1.0),  # High confidence, correct
        (0.3, 0.0),  # Low confidence, correct
        (0.9, 0.0),  # High confidence, wrong
        (0.2, 1.0),  # Low confidence, wrong
        (0.6, 1.0),  # Medium confidence, correct
    ]

    # Record predictions
    for forecast, outcome in examples:
        tracker.record_prediction(forecast, outcome)

    # Show results
    stats = tracker.get_stats()
    print(f"Brier Score: {stats['brier_score']:.3f}")
    print(f"Accuracy: {stats['accuracy']:.3f}")
    print(f"Predictions: {stats['prediction_count']}")

    # Test Kelly adjustment
    base_kelly = 0.25  # 25% base Kelly sizing
    adjusted_kelly = tracker.adjust_kelly_sizing(base_kelly)
    print(f"Base Kelly: {base_kelly:.3f}")
    print(f"Adjusted Kelly: {adjusted_kelly:.3f}")
    print(f"Adjustment Factor: {adjusted_kelly/base_kelly:.3f}")