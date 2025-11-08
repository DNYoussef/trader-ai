"""
Debug version of optimized training to identify why no iterations execute
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import numpy as np
import torch
import torch.nn as nn

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.strategies.enhanced_market_state import create_enhanced_market_state
from src.intelligence.enhanced_hrm_features import EnhancedHRMFeatureEngine

print("="*60)
print("DEBUGGING OPTIMIZED TRAINING ISSUES")
print("="*60)

# Test 1: Can we create market states?
print("\n1. Testing market state generation...")
try:
    scenario = create_enhanced_market_state('SPY')
    print(f"✓ Market state created: {list(scenario.keys())}")
    print(f"  VIX: {scenario.get('vix_level', 'N/A')}")
    print(f"  Price: {scenario.get('price', 'N/A')}")
except Exception as e:
    print(f"✗ Market state failed: {e}")
    sys.exit(1)

# Test 2: Can we create enhanced features?
print("\n2. Testing enhanced feature generation...")
try:
    feature_engine = EnhancedHRMFeatureEngine()
    enhanced_features = feature_engine.generate_features(scenario)
    print(f"✓ Enhanced features created: shape {enhanced_features.combined_features.shape}")
    print(f"  Feature sample: {enhanced_features.combined_features[:5]}")
except Exception as e:
    print(f"✗ Enhanced features failed: {e}")
    sys.exit(1)

# Test 3: Can we generate a single batch?
print("\n3. Testing single batch generation...")

class DebugDataGenerator:
    def __init__(self):
        self.feature_engine = EnhancedHRMFeatureEngine()
        self.strategies = [
            'crisis_alpha', 'tail_hedge', 'volatility_harvest', 'event_catalyst',
            'correlation_breakdown', 'inequality_arbitrage', 'momentum_explosion', 'mean_reversion'
        ]

    def generate_single_sample(self):
        """Generate a single sample for debugging"""
        try:
            # Generate scenario
            scenario = create_enhanced_market_state('SPY')

            # Generate features
            enhanced_features = self.feature_engine.generate_features(scenario)

            # Simple strategy selection (just pick random for debugging)
            best_idx = np.random.randint(0, 8)

            return enhanced_features.combined_features, best_idx

        except Exception as e:
            print(f"  Sample generation error: {e}")
            return None, None

    def generate_debug_batch(self, batch_size=4):  # Small batch for debugging
        """Generate small debug batch"""
        print(f"    Attempting to generate batch of size {batch_size}...")

        batch_features = []
        batch_labels = []

        for i in range(batch_size):
            print(f"      Sample {i+1}/{batch_size}...", end=" ")
            features, label = self.generate_single_sample()

            if features is not None and label is not None:
                batch_features.append(features)
                batch_labels.append(label)
                print("✓")
            else:
                print("✗")

        if len(batch_features) == 0:
            print(f"    ✗ No valid samples generated!")
            return None, None

        print(f"    ✓ Generated {len(batch_features)} valid samples")

        # Convert to tensors
        try:
            features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
            labels_tensor = torch.tensor(np.array(batch_labels), dtype=torch.long)

            print(f"    ✓ Tensors created: features {features_tensor.shape}, labels {labels_tensor.shape}")
            return features_tensor, labels_tensor

        except Exception as e:
            print(f"    ✗ Tensor conversion failed: {e}")
            return None, None

try:
    debug_generator = DebugDataGenerator()
    features, labels = debug_generator.generate_debug_batch()

    if features is not None:
        print(f"✓ Debug batch created successfully!")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
    else:
        print("✗ Debug batch generation failed!")
        sys.exit(1)

except Exception as e:
    print(f"✗ Debug batch generation crashed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Can we create the model and do a forward pass?
print("\n4. Testing model forward pass...")

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 64)
        self.output = nn.Linear(64, 8)

    def forward(self, x):
        x = self.projection(x)
        x = torch.relu(x)
        return self.output(x)

try:
    model = SimpleTestModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)

    print(f"✓ Model created on device: {device}")

    # Forward pass
    with torch.no_grad():
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels)

    print(f"✓ Forward pass successful!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test torch.compile
print("\n5. Testing torch.compile...")
try:
    compiled_model = torch.compile(model, mode='reduce-overhead')

    with torch.no_grad():
        compiled_logits = compiled_model(features)
        compiled_loss = nn.functional.cross_entropy(compiled_logits, labels)

    print(f"✓ torch.compile successful!")
    print(f"  Compiled loss: {compiled_loss.item():.4f}")

except Exception as e:
    print(f"✗ torch.compile failed: {e}")
    import traceback
    traceback.print_exc()
    print("  (This might be expected on some systems)")

print("\n" + "="*60)
print("DEBUG RESULTS:")
print("- Market state generation: ✓")
print("- Enhanced features: ✓")
print("- Batch generation: ✓")
print("- Model forward pass: ✓")
print("- torch.compile: Check above")
print("\nThe optimized training should work. Issue might be in:")
print("1. Training loop logic")
print("2. Batch generation scaling to larger sizes")
print("3. Exception handling masking real errors")
print("="*60)