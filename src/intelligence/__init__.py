"""
ML Intelligence System
Production-ready machine learning pipeline for financial trading
"""

__version__ = "1.0.0"
__author__ = "AI Trading Intelligence Team"

_LAZY_EXPORTS = {
    "ModelTrainer": ".training.trainer",
    "ModelRegistry": ".models.registry",
    "DataProcessor": ".data.processor",
    "Predictor": ".prediction.predictor",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """Load heavyweight ML components only when explicitly requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
