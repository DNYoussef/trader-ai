"""
Gary×Taleb Trading Strategy Performance Analysis System
Phase 4 Implementation - Complete Performance Benchmarking Suite

This module provides comprehensive performance analysis capabilities for the Gary×Taleb
trading strategy, combining Gary's Dynamic Performance Indicator (DPI) methodology
with Nassim Taleb's antifragility principles.

Key Components:
- Benchmarking Framework: Compare against traditional strategies
- Performance Metrics Engine: Advanced metrics with Sharpe optimization
- Strategy Comparison: Statistical significance testing
- Drawdown Analysis: Risk management and prevention
- Real-time Tracking: Live performance monitoring
- Return Attribution: Source analysis of performance
- Optimization Engine: Parameter tuning and recommendations
- Statistical Validation: Comprehensive testing framework
- Antifragility Analyzer: Taleb-inspired metrics integration
- Reporting System: Professional visualization and reports

Author: Performance Analysis System
Version: 4.0.0
"""

from .benchmarking.BenchmarkFramework import (
    BenchmarkFramework,
    BaseStrategy,
    BuyAndHoldStrategy,
    MovingAverageCrossover,
    MeanReversionStrategy,
    GaryTalebStrategy,
    PerformanceMetrics
)

from .metrics.PerformanceEngine import (
    PerformanceEngine,
    RiskMetrics,
    ReturnMetrics,
    RatioMetrics,
    TalebMetrics,
    DPIMetrics
)

from .comparison.StrategyComparison import (
    StrategyComparison,
    StatisticalTest,
    ComparisonResult
)

from .risk.DrawdownAnalysis import (
    DrawdownAnalysis,
    DrawdownEvent,
    DrawdownStats,
    RiskAlert
)

from .tracking.RealTimeTracker import (
    RealTimeTracker,
    PerformanceSnapshot,
    OptimizationSignal,
    AlertCondition
)

from .attribution.ReturnAttribution import (
    ReturnAttribution,
    AttributionComponent,
    AttributionResult
)

from .optimization.PerformanceOptimizer import (
    PerformanceOptimizer,
    OptimizationParameter,
    OptimizationResult,
    OptimizationRecommendation
)

from .validation.StatisticalValidator import (
    StatisticalValidator,
    ValidationResult
)

from .analytics.AntifragilityAnalyzer import (
    AntifragilityAnalyzer,
    AntifragilityMetrics,
    DPIMetrics as AnalyticsDPIMetrics,
    IntegratedAnalysis
)

from .reporting.PerformanceReporter import (
    PerformanceReporter,
    ReportSection,
    ReportConfig
)

__version__ = "4.0.0"

__all__ = [
    # Benchmarking
    "BenchmarkFramework",
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossover",
    "MeanReversionStrategy",
    "GaryTalebStrategy",
    "PerformanceMetrics",

    # Metrics Engine
    "PerformanceEngine",
    "RiskMetrics",
    "ReturnMetrics",
    "RatioMetrics",
    "TalebMetrics",
    "DPIMetrics",

    # Strategy Comparison
    "StrategyComparison",
    "StatisticalTest",
    "ComparisonResult",

    # Risk Analysis
    "DrawdownAnalysis",
    "DrawdownEvent",
    "DrawdownStats",
    "RiskAlert",

    # Real-time Tracking
    "RealTimeTracker",
    "PerformanceSnapshot",
    "OptimizationSignal",
    "AlertCondition",

    # Return Attribution
    "ReturnAttribution",
    "AttributionComponent",
    "AttributionResult",

    # Optimization
    "PerformanceOptimizer",
    "OptimizationParameter",
    "OptimizationResult",
    "OptimizationRecommendation",

    # Statistical Validation
    "StatisticalValidator",
    "ValidationResult",

    # Antifragility Analytics
    "AntifragilityAnalyzer",
    "AntifragilityMetrics",
    "AnalyticsDPIMetrics",
    "IntegratedAnalysis",

    # Reporting
    "PerformanceReporter",
    "ReportSection",
    "ReportConfig"
]

# Required for comprehensive analysis helper function
import pandas as pd

# Quick access functions for common workflows
def create_comprehensive_analysis(returns, equity_curve, initial_capital=200.0):
    """
    Create a comprehensive performance analysis using all available tools.

    Args:
        returns: pd.Series of strategy returns
        equity_curve: pd.Series of strategy equity curve
        initial_capital: float, initial capital amount

    Returns:
        dict: Complete analysis results including all metrics and recommendations
    """

    # Initialize all analyzers
    benchmark_framework = BenchmarkFramework(initial_capital)
    performance_engine = PerformanceEngine()
    StrategyComparison()
    drawdown_analyzer = DrawdownAnalysis()
    antifragility_analyzer = AntifragilityAnalyzer()
    validator = StatisticalValidator()

    # Create sample data for benchmarking
    sample_data = pd.DataFrame({
        'Close': equity_curve / (equity_curve.iloc[0] / 100),  # Normalize to price-like data
        'Volume': [1000] * len(equity_curve)
    }, index=equity_curve.index)

    # Run comprehensive analysis
    results = {}

    # 1. Benchmarking
    benchmark_results = benchmark_framework.run_benchmark(sample_data)
    results['benchmarking'] = {
        'results': benchmark_results,
        'summary': benchmark_framework.get_performance_summary(),
        'report': benchmark_framework.generate_benchmark_report()
    }

    # 2. Performance Metrics
    performance_metrics = performance_engine.calculate_comprehensive_metrics(
        returns, equity_curve
    )
    results['performance_metrics'] = performance_metrics

    # 3. Drawdown Analysis
    drawdown_events, drawdown_stats = drawdown_analyzer.analyze_drawdowns(equity_curve, returns)
    results['drawdown_analysis'] = {
        'events': drawdown_events,
        'statistics': drawdown_stats,
        'report': drawdown_analyzer.generate_drawdown_report(drawdown_events, drawdown_stats)
    }

    # 4. Antifragility Analysis
    antifragility_metrics = antifragility_analyzer.calculate_antifragility_metrics(returns)
    dpi_metrics = antifragility_analyzer.calculate_dpi_metrics(returns)
    integrated_analysis = antifragility_analyzer.perform_integrated_analysis(
        returns, antifragility_metrics, dpi_metrics
    )
    results['antifragility_analysis'] = {
        'antifragility_metrics': antifragility_metrics,
        'dpi_metrics': dpi_metrics,
        'integrated_analysis': integrated_analysis,
        'report': antifragility_analyzer.generate_comprehensive_report(integrated_analysis, returns)
    }

    # 5. Statistical Validation
    validation_result = validator.validate_strategy_performance(returns, equity_curve)
    results['statistical_validation'] = {
        'result': validation_result,
        'report': validator.generate_validation_report(validation_result)
    }

    return results

def create_gary_taleb_report(returns, equity_curve, output_path="gary_taleb_report.html"):
    """
    Create a comprehensive Gary×Taleb performance report.

    Args:
        returns: pd.Series of strategy returns
        equity_curve: pd.Series of strategy equity curve
        output_path: str, path for the output HTML report

    Returns:
        str: Path to the generated report
    """

    # Run comprehensive analysis
    analysis_results = create_comprehensive_analysis(returns, equity_curve)

    # Initialize reporter
    config = ReportConfig(
        title="Gary×Taleb Trading Strategy Performance Report",
        color_scheme="professional",
        include_executive_summary=True
    )

    reporter = PerformanceReporter(config)

    # Extract key metrics for executive summary
    perf_metrics = analysis_results['performance_metrics']
    antifragility = analysis_results['antifragility_analysis']

    key_metrics = {
        'sharpe_ratio': perf_metrics['ratios'].sharpe_ratio,
        'max_drawdown': analysis_results['drawdown_analysis']['statistics'].max_drawdown,
        'dpi_score': antifragility['dpi_metrics'].composite_dpi_score,
        'antifragility_score': antifragility['antifragility_metrics'].convexity_score,
        'synergy_score': antifragility['integrated_analysis'].synergy_score,
        'overall_score': antifragility['integrated_analysis'].combined_effectiveness
    }

    performance_data = {
        'total_return': perf_metrics['returns'].total_return,
        'annualized_return': perf_metrics['returns'].annualized_return
    }

    recommendations = antifragility['integrated_analysis'].optimization_opportunities

    # Add report sections
    reporter.add_section(reporter.create_executive_summary(
        performance_data, key_metrics, recommendations
    ))

    reporter.add_section(reporter.create_performance_overview(
        returns, equity_curve
    ))

    reporter.add_section(reporter.create_gary_taleb_analysis(
        antifragility['antifragility_metrics'].__dict__,
        antifragility['dpi_metrics'].__dict__,
        antifragility['integrated_analysis'].__dict__
    ))

    # Generate HTML report
    report_path = reporter.generate_html_report(output_path)

    return report_path

# Configuration constants
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BENCHMARK_RETURN = 0.10

# Gary×Taleb specific constants
GARY_DPI_WEIGHTS = {
    'momentum': 0.25,
    'stability': 0.20,
    'consistency': 0.20,
    'adaptability': 0.15,
    'signal_quality': 0.10,
    'execution_efficiency': 0.10
}

TALEB_ANTIFRAGILITY_WEIGHTS = {
    'convexity': 0.25,
    'volatility_benefit': 0.20,
    'asymmetry_ratio': 0.15,
    'black_swan_protection': 0.15,
    'barbell_efficiency': 0.10,
    'optionality_ratio': 0.10,
    'resilience_factor': 0.05
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.6,
    'moderate': 0.4,
    'poor': 0.2
}

print(f"Gary×Taleb Performance Analysis System v{__version__} loaded successfully")
print("Available modules: Benchmarking, Metrics, Comparison, Risk, Tracking, Attribution, Optimization, Validation, Analytics, Reporting")
print("Use create_comprehensive_analysis() for full analysis or create_gary_taleb_report() for HTML reports")