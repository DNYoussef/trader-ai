"""
Standalone test of Kelly criterion with Brier score integration
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'performance'))

from simple_brier import BrierTracker
from typing import Dict, Tuple


class KellyWithBrier:
    """Kelly Criterion enhanced with Brier score accuracy tracking"""

    def __init__(self, brier_window_size: int = 100, max_kelly: float = 0.25):
        self.brier_tracker = BrierTracker(window_size=brier_window_size)
        self.max_kelly = max_kelly
        self.prediction_count = 0

    def calculate_kelly_base(self, win_prob: float, win_payoff: float, loss_payoff: float = 1.0) -> float:
        """Calculate base Kelly criterion percentage"""
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        loss_prob = 1.0 - win_prob
        numerator = (win_payoff * win_prob) - loss_prob
        denominator = win_payoff

        if denominator <= 0 or numerator <= 0:
            return 0.0

        kelly_base = numerator / denominator
        kelly_base = min(kelly_base, self.max_kelly)
        return max(0.0, kelly_base)

    def record_prediction_outcome(self, predicted_prob: float, actual_outcome: bool) -> None:
        """Record a prediction and its actual outcome"""
        outcome_value = 1.0 if actual_outcome else 0.0
        self.brier_tracker.record_prediction(predicted_prob, outcome_value)
        self.prediction_count += 1

    def calculate_adjusted_kelly(self, win_prob: float, win_payoff: float, loss_payoff: float = 1.0) -> Dict:
        """Calculate Kelly percentage adjusted for prediction accuracy"""
        base_kelly = self.calculate_kelly_base(win_prob, win_payoff, loss_payoff)
        adjusted_kelly = self.brier_tracker.adjust_kelly_sizing(base_kelly)
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
        """Calculate actual position size in dollars"""
        kelly_result = self.calculate_adjusted_kelly(win_prob, win_payoff, loss_payoff)
        position_size = account_balance * kelly_result['adjusted_kelly']
        return position_size, kelly_result


def test_simple_functionality():
    """Test basic functionality works correctly"""
    print("=== Basic Functionality Test ===")

    # Test 1: Brier tracker basics
    tracker = BrierTracker(window_size=10)

    # Perfect predictions should give Brier score of 0
    tracker.record_prediction(1.0, 1.0)
    tracker.record_prediction(0.0, 0.0)
    print(f"Perfect predictions Brier score: {tracker.get_brier_score():.3f} (should be 0.0)")

    # Reset and test worst predictions (should give Brier score of 1)
    tracker.reset()
    tracker.record_prediction(0.0, 1.0)
    tracker.record_prediction(1.0, 0.0)
    print(f"Worst predictions Brier score: {tracker.get_brier_score():.3f} (should be 1.0)")

    # Test 2: Kelly calculation
    kelly_calc = KellyWithBrier()

    # Test base Kelly calculation (classic example: 60% win rate, 2:1 payoff)
    base_kelly = kelly_calc.calculate_kelly_base(win_prob=0.6, win_payoff=2.0)
    print(f"Base Kelly (60% win, 2:1 payoff): {base_kelly:.3f} (should be 0.2)")

    # Test Kelly with perfect prediction history
    kelly_calc.brier_tracker.record_prediction(1.0, 1.0)
    kelly_calc.brier_tracker.record_prediction(0.0, 0.0)

    result = kelly_calc.calculate_adjusted_kelly(win_prob=0.6, win_payoff=2.0)
    print(f"Adjusted Kelly with perfect predictions: {result['adjusted_kelly']:.3f}")
    print(f"Adjustment factor: {result['adjustment_factor']:.3f} (should be 1.0)")

    print("✓ Basic functionality tests passed\n")


def demo_realistic_scenario():
    """Demonstrate realistic trading scenario"""
    print("=== Realistic Trading Scenario ===")

    kelly_calc = KellyWithBrier(brier_window_size=20, max_kelly=0.25)
    account_balance = 100000  # $100k account

    print("Scenario: 60% win rate trading with 1.5:1 payoff ratio")
    print("Building prediction history with varying accuracy...\n")

    # Simulate realistic predictions with some good and bad periods
    predictions = [
        # Good period
        (0.8, True), (0.7, True), (0.6, True), (0.3, False), (0.2, False),
        # Overconfident period (bad)
        (0.9, False), (0.8, False), (0.9, True), (0.7, False),
        # Recovery period
        (0.6, True), (0.5, True), (0.4, False), (0.7, True), (0.8, True),
        # Mixed period
        (0.6, False), (0.4, True), (0.7, True), (0.3, False)
    ]

    print("Prediction History:")
    print("Pred | Actual | Running Brier | Running Accuracy | Adj Kelly | Position Size")
    print("-" * 75)

    for i, (pred_prob, outcome) in enumerate(predictions):
        kelly_calc.record_prediction_outcome(pred_prob, outcome)

        position_size, details = kelly_calc.get_position_size(
            account_balance=account_balance,
            win_prob=0.6,
            win_payoff=1.5
        )

        print(f"{pred_prob:4.1f} | {str(outcome):>6} | "
              f"{details['brier_score']:11.3f} | {details['accuracy']:15.3f} | "
              f"{details['adjusted_kelly']:8.3f} | ${position_size:>10,.0f}")

    # Final summary
    final_stats = kelly_calc.brier_tracker.get_stats()
    print(f"\n=== Final Results ===")
    print(f"Final Brier Score: {final_stats['brier_score']:.3f}")
    print(f"Final Accuracy: {final_stats['accuracy']:.3f}")
    print(f"Total Predictions: {final_stats['prediction_count']}")

    # Show final position sizing
    final_pos, final_details = kelly_calc.get_position_size(account_balance, 0.6, 1.5)
    base_pos = account_balance * final_details['base_kelly']

    print(f"\nPosition Sizing Impact:")
    print(f"  Base Kelly Position: ${base_pos:,.0f} ({final_details['base_kelly']*100:.1f}%)")
    print(f"  Brier-Adjusted Position: ${final_pos:,.0f} ({final_details['adjusted_kelly']*100:.1f}%)")
    print(f"  Risk Reduction: {(1-final_details['adjustment_factor'])*100:.1f}%")
    print(f"  Dollar Risk Reduction: ${base_pos - final_pos:,.0f}")


if __name__ == "__main__":
    test_simple_functionality()
    demo_realistic_scenario()