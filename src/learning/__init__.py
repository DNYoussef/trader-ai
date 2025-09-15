"""
Continuous Learning System for Gary×Taleb Trading System

A comprehensive continuous learning infrastructure that includes:
- Automated model retraining pipeline
- Performance feedback loops
- A/B testing framework
- Strategy adaptation engine
- Real-time performance monitoring
- Automated rollback system
- Online learning algorithms
- System orchestration

Key Components:
- ContinuousLearner: Automated retraining with performance validation
- PerformanceFeedback: Real-time feedback tracking actual vs predicted returns
- ABTestingFramework: Production A/B testing with statistical significance
- StrategyAdaptationEngine: Adaptive strategy optimization for market conditions
- PerformanceMonitor: Real-time monitoring with degradation detection
- LearningOrchestrator: Central coordination of all learning components
- PerformanceAnalyzer: Advanced P&L correlation and attribution analysis

Usage:
    from src.learning import LearningOrchestrator, OrchestrationConfig

    config = OrchestrationConfig()
    orchestrator = LearningOrchestrator(config)
    orchestrator.start_orchestration()

    # Record trading performance
    orchestrator.record_trading_performance(
        model_id="gary_dpi_v1",
        actual_return=0.02,
        predicted_return=0.015,
        trade_data={...},
        market_data={...}
    )
"""

from .pipeline.continuous_learner import (
    ContinuousLearner,
    LearningConfig,
    ModelPerformance,
    RetrainingResult
)

from .feedback.performance_feedback import (
    PerformanceFeedback,
    FeedbackMetrics,
    FeedbackSignal,
    PerformanceWindow
)

from .testing.ab_testing import (
    ABTestingFramework,
    ExperimentConfig,
    VariantConfig,
    ExperimentStatus,
    TrafficSplitType,
    StatisticalResult,
    ExperimentSummary
)

from .adaptation.strategy_adaptation import (
    StrategyAdaptationEngine,
    MarketRegime,
    StrategyParameters,
    AdaptationSignal,
    PerformanceContext
)

from .monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceThreshold,
    PerformanceAlert,
    ModelHealthStatus,
    PerformanceMetrics as MonitoringMetrics
)

from .analysis.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics as AnalysisMetrics,
    PnLAttribution,
    RegimePerformance,
    CorrelationAnalysis
)

from .orchestration.learning_orchestrator import (
    LearningOrchestrator,
    OrchestrationConfig,
    SystemHealth,
    CoordinationAction
)

__all__ = [
    # Core orchestration
    'LearningOrchestrator',
    'OrchestrationConfig',
    'SystemHealth',
    'CoordinationAction',

    # Continuous learning
    'ContinuousLearner',
    'LearningConfig',
    'ModelPerformance',
    'RetrainingResult',

    # Performance feedback
    'PerformanceFeedback',
    'FeedbackMetrics',
    'FeedbackSignal',
    'PerformanceWindow',

    # A/B testing
    'ABTestingFramework',
    'ExperimentConfig',
    'VariantConfig',
    'ExperimentStatus',
    'TrafficSplitType',
    'StatisticalResult',
    'ExperimentSummary',

    # Strategy adaptation
    'StrategyAdaptationEngine',
    'MarketRegime',
    'StrategyParameters',
    'AdaptationSignal',
    'PerformanceContext',

    # Performance monitoring
    'PerformanceMonitor',
    'PerformanceThreshold',
    'PerformanceAlert',
    'ModelHealthStatus',
    'MonitoringMetrics',

    # Performance analysis
    'PerformanceAnalyzer',
    'AnalysisMetrics',
    'PnLAttribution',
    'RegimePerformance',
    'CorrelationAnalysis'
]

# Version information
__version__ = "1.0.0"
__author__ = "Gary×Taleb Trading System"
__description__ = "Continuous Learning System for Algorithmic Trading"

# Configuration defaults
DEFAULT_CONFIG = OrchestrationConfig(
    continuous_learning_enabled=True,
    performance_feedback_enabled=True,
    ab_testing_enabled=True,
    strategy_adaptation_enabled=True,
    performance_monitoring_enabled=True,
    auto_rollback_enabled=True,
    orchestration_interval_seconds=60,
    health_check_interval_seconds=300,
    critical_performance_threshold=-0.2,
    auto_intervention_threshold=-0.15,
    rollback_trigger_threshold=-0.25
)

def create_learning_system(config: OrchestrationConfig = None) -> LearningOrchestrator:
    """
    Create and configure a complete learning system

    Args:
        config: Optional orchestration configuration

    Returns:
        Configured LearningOrchestrator instance
    """
    if config is None:
        config = DEFAULT_CONFIG

    return LearningOrchestrator(config)