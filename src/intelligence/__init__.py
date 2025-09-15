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

# Legacy Intelligence Components (if available)
try:
    from .risk_pattern_engine import (
        RiskPatternEngine,
        PatternAnalyzer,
        RiskLevel,
        DetectionResult,
        AlertMetadata
    )

    from .ai_alert_system import (
        AIAlertSystem,
        AlertType,
        AlertPriority,
        AlertClassifier,
        ContextualAlertGenerator
    )

    from .alert_orchestrator import (
        AlertOrchestrator,
        AlertQueue,
        AlertRouter,
        AlertPriorityManager,
        AlertDeduplicator
    )

    from .predictive_warning_system import (
        PredictiveWarningSystem,
        WarningPredictor,
        PatternSequenceAnalyzer,
        TimeSeriesForecaster,
        RiskProbabilityCalculator
    )

    from .context_filter import (
        ContextFilter,
        MarketContextAnalyzer,
        AlertContextEnricher,
        ContextualRelevanceScorer,
        TemporalContextManager
    )

    LEGACY_COMPONENTS = [
        'RiskPatternEngine', 'PatternAnalyzer', 'RiskLevel',
        'DetectionResult', 'AlertMetadata',
        'AIAlertSystem', 'AlertType', 'AlertPriority',
        'AlertClassifier', 'ContextualAlertGenerator',
        'AlertOrchestrator', 'AlertQueue', 'AlertRouter',
        'AlertPriorityManager', 'AlertDeduplicator',
        'PredictiveWarningSystem', 'WarningPredictor',
        'PatternSequenceAnalyzer', 'TimeSeriesForecaster',
        'RiskProbabilityCalculator',
        'ContextFilter', 'MarketContextAnalyzer',
        'AlertContextEnricher', 'ContextualRelevanceScorer',
        'TemporalContextManager'
    ]
except ImportError as e:
    print(f"Warning: Legacy intelligence components not available: {e}")
    LEGACY_COMPONENTS = []

# Export all available components
__all__ = ML_COMPONENTS + LEGACY_COMPONENTS