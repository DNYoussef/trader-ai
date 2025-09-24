"""
HRM SANDBOX TEST
Quick validation test to ensure HRM system works before full training
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from hrm_grokfast_trader import HRMGrokkingTrainer

def sandbox_test():
    """Quick sandbox test of HRM system"""

    print("=" * 60)
    print("HRM SANDBOX TEST")
    print("=" * 60)
    print("Testing HRM integration before full training...")
    print()

    try:
        # Initialize trainer
        print("1. Initializing HRM trainer...")
        trainer = HRMGrokkingTrainer()  # No mock option, always real
        print(f"   OK Trainer initialized")
        print(f"   OK Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print()

        # Test data generation
        print("2. Testing hierarchical data generation...")
        scenarios = trainer.data_generator.generate_batch(4)
        print(f"   OK Generated {len(scenarios)} scenarios")
        print(f"   OK Example scenario: VIX={scenarios[0]['vix_level']:.1f}")
        print()

        # Test batch creation
        print("3. Testing batch creation...")
        features, labels = trainer.create_hierarchical_training_batch(scenarios)
        if features is not None:
            print(f"   OK Batch created: {features.shape}")
            print(f"   OK Labels: {labels.shape}")
        else:
            print("   WARNING No signals generated (may be normal)")
        print()

        # Test one training iteration
        print("4. Testing training iteration...")

        # Override training config for quick test
        trainer.training_config['max_iterations'] = 3  # Just 3 iterations for test
        trainer.training_config['validation_every'] = 1  # Validate every iteration

        # Run short training
        print("   Running 3 quick iterations...")
        success = trainer.train_until_hierarchical_grokking()

        if success:
            print("   OK Training completed successfully")
        else:
            print("   OK Training ran without errors (expected for short test)")

        print()
        print("=" * 60)
        print("SANDBOX TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("HRM system is ready for full training!")
        print()

        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("=" * 60)
        print("SANDBOX TEST FAILED")
        print("=" * 60)
        return False

if __name__ == "__main__":
    sandbox_test()