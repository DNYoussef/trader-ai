"""
Kelly Criterion with Brier Score Integration

Simple integration of Brier score tracking with Kelly position sizing.
Adjusts position sizes based on prediction accuracy to implement "Position > Opinion" principle.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from performance.simple_brier import BrierTracker
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class KellyWithBrier:
    """Kelly Criterion enhanced with Brier score accuracy tracking"""

    def __init__(self, brier_window_size: int = 100, max_kelly: float = 0.25):
        """
        Initialize Kelly calculator with Brier score tracking

        Args:
            brier_window_size: Number of recent predictions for Brier score calculation
            max_kelly: Maximum Kelly percentage allowed (safety cap)
        """
        self.brier_tracker = BrierTracker(window_size=brier_window_size)
        self.max_kelly = max_kelly
        self.prediction_count = 0

    def calculate_kelly_base(self, win_prob: float, win_payoff: float, loss_payoff: float = 1.0) -> float:
        """
        Calculate base Kelly criterion percentage

        Args:
            win_prob: Probability of winning (0.0 to 1.0)
            win_payoff: Payoff ratio for wins (e.g., 2.0 for 2:1 odds)
            loss_payoff: Payoff ratio for losses (typically 1.0)

        Returns:
            Base Kelly percentage before Brier adjustment
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        loss_prob = 1.0 - win_prob

        # Kelly formula: (bp - q) / b
        # where b = win_payoff, p = win_prob, q = loss_prob
        numerator = (win_payoff * win_prob) - loss_prob
        denominator = win_payoff

        if denominator <= 0 or numerator <= 0:
            return 0.0

        kelly_base = numerator / denominator

        # Apply safety cap
        kelly_base = min(kelly_base, self.max_kelly)

        return max(0.0, kelly_base)

    def record_prediction_outcome(self, predicted_prob: float, actual_outcome: bool) -> None:
        """
        Record a prediction and its actual outcome for Brier score tracking

        Args:
            predicted_prob: Predicted probability (0.0 to 1.0)
            actual_outcome: Whether prediction was correct (True/False)
        """
        outcome_value = 1.0 if actual_outcome else 0.0
        self.brier_tracker.record_prediction(predicted_prob, outcome_value)
        self.prediction_count += 1

        if self.prediction_count % 10 == 0:
            stats = self.brier_tracker.get_stats()
            logger.info(f"Brier stats after {self.prediction_count} predictions: "
                       f"Score={stats['brier_score']:.3f}, Accuracy={stats['accuracy']:.3f}")

    def calculate_adjusted_kelly(self, win_prob: float, win_payoff: float, loss_payoff: float = 1.0) -> Dict:
        """
        Calculate Kelly percentage adjusted for prediction accuracy

        Args:
            win_prob: Probability of winning (0.0 to 1.0)
            win_payoff: Payoff ratio for wins
            loss_payoff: Payoff ratio for losses

        Returns:
            Dictionary with base Kelly, adjusted Kelly, and Brier stats
        """
        # Calculate base Kelly
        base_kelly = self.calculate_kelly_base(win_prob, win_payoff, loss_payoff)

        # Get Brier score adjustment
        adjusted_kelly = self.brier_tracker.adjust_kelly_sizing(base_kelly)

        # Get current stats
        brier_stats = self.brier_tracker.get_stats()

        return {
            'base_kelly': base_kelly,
            'adjusted_kelly': adjusted_kelly,
            'adjustment_factor': adjusted_kelly / base_kelly if base_kelly > 0 else 0.0,
            'brier_score': brier_stats['brier_score'],
            'accuracy': brier_stats['accuracy'],
            'prediction_count': brier_stats['prediction_count']
        }

    def get_position_size(self, account_balance: float, win_prob: float,
                         win_payoff: float, loss_payoff: float = 1.0) -> Tuple[float, Dict]:
        """
        Calculate actual position size in dollars

        Args:
            account_balance: Total account balance
            win_prob: Probability of winning
            win_payoff: Payoff ratio for wins
            loss_payoff: Payoff ratio for losses

        Returns:
            Tuple of (position_size_dollars, calculation_details)
        """
        kelly_result = self.calculate_adjusted_kelly(win_prob, win_payoff, loss_payoff)

        position_size = account_balance * kelly_result['adjusted_kelly']

        return position_size, kelly_result

    def reset_tracker(self) -> None:
        """Reset Brier score tracker"""
        self.brier_tracker.reset()
        self.prediction_count = 0
        logger.info("Brier tracker reset")

    def get_current_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.brier_tracker.get_stats()
        stats['total_predictions'] = self.prediction_count
        return stats


# Simple test/demo
def demo_kelly_brier():
    """Demonstrate Kelly with Brier score integration"""
    print("=== Kelly Criterion with Brier Score Demo ===\n")

    # Initialize calculator
    kelly_calc = KellyWithBrier(brier_window_size=20, max_kelly=0.25)

    # Test parameters
    account_balance = 100000  # $100k account

    print("Scenario: Trading with initial 60% win rate, 1.5:1 payoff ratio")
    print("Recording prediction outcomes to build Brier score history...\n")

    # Simulate some predictions and outcomes
    import random
    random.seed(42)  # For reproducible results

    predictions_and_outcomes = [
        # Initial good predictions
        (0.7, True), (0.8, True), (0.6, True), (0.9, False),  # Mostly correct
        (0.3, False), (0.2, False), (0.4, True), (0.1, False),
        # Some poor predictions
        (0.9, False), (0.8, False), (0.7, False),  # Overconfident misses
        # Recovery
        (0.6, True), (0.7, True), (0.5, True), (0.4, False)
    ]

    # Record predictions
    for i, (pred_prob, outcome) in enumerate(predictions_and_outcomes):
        kelly_calc.record_prediction_outcome(pred_prob, outcome)

        # Calculate position for current trade
        position_size, details = kelly_calc.get_position_size(
            account_balance=account_balance,
            win_prob=0.6,  # Assume 60% win rate for position sizing
            win_payoff=1.5,  # 1.5:1 payoff
            loss_payoff=1.0
        )

        if i % 5 == 4:  # Print every 5th prediction
            print(f"After {i+1} predictions:")
            print(f"  Brier Score: {details['brier_score']:.3f}")
            print(f"  Accuracy: {details['accuracy']:.3f}")
            print(f"  Base Kelly: {details['base_kelly']:.3f} ({details['base_kelly']*100:.1f}%)")
            print(f"  Adjusted Kelly: {details['adjusted_kelly']:.3f} ({details['adjusted_kelly']*100:.1f}%)")
            print(f"  Adjustment Factor: {details['adjustment_factor']:.3f}")
            print(f"  Position Size: ${position_size:,.0f}")
            print(f"  Position % of Account: {position_size/account_balance*100:.1f}%")
            print()

    print("=== Final Results ===")
    final_stats = kelly_calc.get_current_stats()
    print(f"Final Brier Score: {final_stats['brier_score']:.3f} (lower is better)")
    print(f"Final Accuracy: {final_stats['accuracy']:.3f}")
    print(f"Total Predictions: {final_stats['prediction_count']}")

    # Show impact on position sizing
    final_position, final_details = kelly_calc.get_position_size(
        account_balance=account_balance,
        win_prob=0.6,
        win_payoff=1.5
    )

    print("\nPosition Sizing Impact:")
    print(f"  Without Brier adjustment: ${account_balance * final_details['base_kelly']:,.0f}")
    print(f"  With Brier adjustment: ${final_position:,.0f}")
    print(f"  Risk reduction: {(1 - final_details['adjustment_factor'])*100:.1f}%")


if __name__ == "__main__":
    demo_kelly_brier()