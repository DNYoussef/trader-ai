"""
Simple demo of Kelly criterion with Brier score integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'performance'))

from simple_brier import BrierTracker


def demo_brier_kelly_integration():
    """Demonstrate the working Brier-Kelly integration"""
    print("=== Kelly Criterion + Brier Score Integration Demo ===\n")

    # Initialize Brier tracker
    tracker = BrierTracker(window_size=20)

    # Kelly calculation function
    def calculate_kelly(win_prob, win_payoff, loss_payoff=1.0):
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        loss_prob = 1.0 - win_prob
        numerator = (win_payoff * win_prob) - loss_prob
        denominator = win_payoff
        if denominator <= 0 or numerator <= 0:
            return 0.0
        kelly = numerator / denominator
        return min(kelly, 0.25)  # Cap at 25%

    # Trading parameters
    account_balance = 100000  # $100k
    win_rate = 0.6  # 60% win rate
    payoff_ratio = 1.5  # 1.5:1 payoff

    # Calculate base Kelly
    base_kelly = calculate_kelly(win_rate, payoff_ratio)
    print(f"Base Kelly calculation: {base_kelly:.3f} ({base_kelly*100:.1f}%)")
    print(f"Base position size: ${account_balance * base_kelly:,.0f}\n")

    # Simulate trading with predictions
    print("Simulating trading with prediction tracking...")
    print("Format: [Prediction -> Outcome] Brier Score | Adjusted Kelly | Position Size")
    print("-" * 70)

    # Trading scenarios
    trades = [
        (0.8, True),   # High confidence, correct
        (0.7, True),   # Good prediction
        (0.6, False),  # Moderate confidence, wrong
        (0.9, False),  # Overconfident, wrong
        (0.8, False),  # Overconfident, wrong
        (0.5, True),   # Modest confidence, correct
        (0.4, False),  # Low confidence, correct
        (0.7, True),   # Recovery
        (0.6, True),   # Building confidence
        (0.8, True),   # Strong finish
    ]

    for i, (prediction, outcome) in enumerate(trades, 1):
        # Record the prediction outcome
        tracker.record_prediction(prediction, float(outcome))

        # Get current Brier score and adjust Kelly
        brier_score = tracker.get_brier_score()
        adjusted_kelly = tracker.adjust_kelly_sizing(base_kelly)
        position_size = account_balance * adjusted_kelly

        outcome_str = "WIN " if outcome else "LOSS"
        print(f"Trade {i:2d}: [{prediction:.1f} -> {outcome_str}] "
              f"Brier: {brier_score:.3f} | "
              f"Kelly: {adjusted_kelly:.3f} | "
              f"Size: ${position_size:>8,.0f}")

    # Final results
    print("\n=== Final Results ===")
    final_stats = tracker.get_stats()
    final_adjusted = tracker.adjust_kelly_sizing(base_kelly)

    print(f"Final Brier Score: {final_stats['brier_score']:.3f} (0=perfect, 1=worst)")
    print(f"Final Accuracy: {final_stats['accuracy']:.3f} ({final_stats['accuracy']*100:.1f}%)")
    print(f"Total Trades: {final_stats['prediction_count']}")

    print(f"\nPosition Sizing Impact:")
    print(f"  Base Kelly (no adjustment): ${account_balance * base_kelly:>10,.0f}")
    print(f"  Brier-adjusted Kelly:       ${account_balance * final_adjusted:>10,.0f}")
    adjustment_factor = final_adjusted / base_kelly
    print(f"  Adjustment Factor: {adjustment_factor:.3f}")
    print(f"  Risk Reduction: {(1-adjustment_factor)*100:.1f}%")

    print(f"\nKey Insight: Position size scales with prediction accuracy")
    print(f"Better predictions = larger positions, worse predictions = smaller positions")


if __name__ == "__main__":
    demo_brier_kelly_integration()