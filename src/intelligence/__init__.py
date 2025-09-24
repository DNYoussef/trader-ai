"""
ML Intelligence System
Production-ready machine learning pipeline for financial trading
"""

__version__ = "1.0.0"
__author__ = "AI Trading Intelligence Team"

# Core ML Components
try:
    from .training.trainer import ModelTrainer
    from .models.registry import ModelRegistry
    from .data.processor import DataProcessor
    from .prediction.predictor import Predictor

    # ML Intelligence Components
    ML_COMPONENTS = [
        "ModelTrainer",
        "ModelRegistry",
        "DataProcessor",
        "Predictor"
    ]
except ImportError as e:
    print(f"Warning: ML components not available: {e}")
    ML_COMPONENTS = []

# Legacy Intelligence Components - Simplified to avoid import errors
# These components are no longer actively used in the main system
LEGACY_COMPONENTS = []

# Export all available components
__all__ = ML_COMPONENTS + LEGACY_COMPONENTS