"""Verify the trained HRM model"""
import torch
import json
from pathlib import Path

# Check model file
model_path = Path("models/enhanced_hrm_32d_grokfast.pth")
history_path = Path("models/enhanced_hrm_32d_history.json")

print("Model Verification")
print("="*50)

# Check files exist
print(f"Model file exists: {model_path.exists()}")
print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"History file exists: {history_path.exists()}")
print()

# Load and check history
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    print("Training History:")
    print(f"Iterations: {history.get('iterations', 'N/A')}")
    print(f"Final accuracy: {history.get('accuracies', ['N/A'])[-1] if 'accuracies' in history else 'N/A'}")
    print(f"Final loss: {history.get('losses', ['N/A'])[-1] if 'losses' in history else 'N/A'}")
    print()

# Try to load model
if model_path.exists():
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("Model checkpoint loaded successfully!")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")

        if 'best_accuracy' in checkpoint:
            print(f"Best accuracy from checkpoint: {checkpoint['best_accuracy']}")
        if 'iteration' in checkpoint:
            print(f"Training iteration: {checkpoint['iteration']}")

    except Exception as e:
        print(f"Error loading model: {e}")