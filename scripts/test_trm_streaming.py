"""
Test TRM Streaming Integration
Quick validation that TRM can make real-time predictions
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from intelligence.trm_streaming_integration import TRMStreamingPredictor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    print("="*80)
    print("TRM STREAMING TEST")
    print("="*80)
    print()

    # Create predictor (5 second intervals for testing)
    predictor = TRMStreamingPredictor(update_interval=5)

    print(f"[OK] Loaded TRM model")
    print(f"[OK] Loaded normalization parameters")
    print(f"[OK] Update interval: {predictor.update_interval}s")
    print()
    print("Starting streaming predictions for 30 seconds...")
    print("(Press Ctrl+C to stop early)")
    print()

    # Callback to display predictions
    async def display_prediction(prediction):
        if 'error' in prediction:
            print(f"\n[ERROR] {prediction['error']}")
            return

        print(f"\n{'='*60}")
        print(f"Time: {prediction['timestamp']}")
        print(f"Strategy: {prediction['strategy_name']} (ID: {prediction['strategy_id']})")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"\nTop 3 Strategy Probabilities:")

        probs = prediction['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

        for i, (name, prob) in enumerate(sorted_probs, 1):
            bar = '|' * int(prob * 40)
            print(f"  {i}. {name:25s}: {bar} {prob:.2%}")

        print(f"\nHalt Probability: {prediction['halt_probability']:.4f}")
        print(f"{'='*60}")

    try:
        # Start streaming
        stream_task = asyncio.create_task(
            predictor.stream_predictions(callback=display_prediction)
        )

        # Run for 30 seconds
        await asyncio.sleep(30)

        # Stop streaming
        predictor.stop_streaming()
        await stream_task

    except KeyboardInterrupt:
        predictor.stop_streaming()
        print("\n\n[WARN] Streaming stopped by user")

    # Show summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    summary = predictor.get_prediction_summary()

    print(f"\nTotal Predictions: {summary.get('total_predictions', 0)}")
    print(f"Average Confidence: {summary.get('average_confidence', 0):.2%}")

    if 'strategy_distribution' in summary:
        print(f"\nStrategy Distribution:")
        for strategy, count in summary['strategy_distribution'].items():
            print(f"  {strategy:25s}: {count} times")

    print()

if __name__ == "__main__":
    asyncio.run(main())
