"""
Run the REAL HRM training with reward-based learning
This uses actual performance simulation, not simple rules
"""
import subprocess
import sys

print("="*80)
print("STARTING REAL HRM TRAINING WITH REWARD-BASED LEARNING")
print("="*80)
print("This will:")
print("1. Simulate each strategy's actual performance")
print("2. Calculate real returns/losses")
print("3. Use ConvexRewardFunction to score outcomes")
print("4. Train model to pick highest reward strategy")
print("="*80)
print()

# Run the original training script with unbuffered output
result = subprocess.run(
    [sys.executable, "-u", "scripts/training/train_enhanced_hrm_32d.py"],
    capture_output=False,
    text=True
)

print(f"\nTraining completed with exit code: {result.returncode}")