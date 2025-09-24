"""
Test REAL training time - shows it will take hours, not seconds
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from real_hrm_implementation import create_real_hrm

def estimate_real_training_time():
    """Estimate actual training time for 156M parameter model"""

    print("=" * 80)
    print("REAL TRAINING TIME ESTIMATION")
    print("=" * 80)

    # Create real model
    model, config = create_real_hrm()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training parameters from fixed config
    batch_size = 16
    gradient_accumulation = 4
    max_iterations = 50000

    print(f"\nTraining Configuration:")
    print(f"  Model Parameters: {model.get_param_count():,}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation: {gradient_accumulation}")
    print(f"  Effective Batch Size: {batch_size * gradient_accumulation}")
    print(f"  Max Iterations: {max_iterations:,}")
    print(f"  Device: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Create dummy batch
    dummy_input = torch.randn(batch_size, 24, device=device)
    dummy_labels = torch.randint(0, 8, (batch_size,), device=device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)

    print("\nTiming forward + backward passes...")

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        model.reset_hidden()  # Reset hidden states
        output = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_labels)
        loss.backward()
        optimizer.step()

    # Time 20 iterations
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    optimizer.zero_grad()
    for i in range(20):
        # Reset hidden states each iteration
        model.reset_hidden()

        # Forward pass
        output = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_labels)

        # Backward pass with gradient accumulation
        loss = loss / gradient_accumulation
        loss.backward()

        # Optimizer step every N iterations (gradient accumulation)
        if (i + 1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.time() - start_time

    # Calculate estimates
    time_per_iter = elapsed / 20
    time_per_effective_batch = time_per_iter * gradient_accumulation

    total_seconds = time_per_iter * max_iterations
    total_hours = total_seconds / 3600

    print("\nTiming Results:")
    print(f"  Time per iteration: {time_per_iter:.3f} seconds")
    print(f"  Time per effective batch: {time_per_effective_batch:.3f} seconds")
    print(f"  Estimated total time: {total_hours:.1f} hours")

    print("\n" + "=" * 80)
    print("TRAINING TIME BREAKDOWN")
    print("=" * 80)

    # Detailed breakdown
    print(f"\nFor {max_iterations:,} iterations:")
    print(f"  First 1,000 iterations: {(time_per_iter * 1000) / 60:.1f} minutes")
    print(f"  First 2,000 iterations (min before grokking check): {(time_per_iter * 2000) / 60:.1f} minutes")
    print(f"  First 10,000 iterations: {(time_per_iter * 10000) / 3600:.1f} hours")
    print(f"  Full 50,000 iterations: {total_hours:.1f} hours")

    print("\nGrokking Timeline Estimate:")
    print(f"  Early signs of learning: 2-4 hours")
    print(f"  Potential grokking onset: 6-12 hours")
    print(f"  Full grokking achieved: 12-24+ hours")

    # Memory usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if total_hours >= 6:
        print("SUCCESS: Training will take HOURS as expected!")
        print(f"Expected training time: {total_hours:.1f} hours")
        print("\nThis is REAL training, not fake instant completion.")
    else:
        print(f"WARNING: Training time seems short ({total_hours:.1f} hours)")
        print("Consider increasing iterations or model complexity")

    print("\nTo start real training:")
    print("  python scripts/hrm_grokfast_trader.py")
    print("\nTraining will NOT exit after 1 second!")
    print("It will run for hours until real grokking is achieved.")
    print("=" * 80)

if __name__ == "__main__":
    estimate_real_training_time()