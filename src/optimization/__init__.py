"""
Multi-Objective Optimization (MOO) package for trader-ai.

This package implements the GlobalMOO + Pymoo woven strategy:
1. GlobalMOO (cloud, best-in-class) finds outer Pareto edges
2. Pymoo (local, fast) searches within confined space
3. Fallback to Pymoo-only if GlobalMOO unavailable

SOURCE: NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1
        meta_calculus/moo_integration.py pattern

Modules:
- trading_oracle: Black-box evaluator for trading objectives
- pymoo_adapter: NSGA-II wrapper for local optimization
- globalmoo_client: Cloud API client (optional)
- robust_pipeline: Ensemble + caching + fallback

Usage:
    from src.optimization import PortfolioOracle, RobustMOOPipeline

    oracle = PortfolioOracle(returns, cov_matrix)
    pipeline = RobustMOOPipeline()
    result = pipeline.optimize(oracle)
"""

__version__ = "1.0.0"
__source__ = "NNC-MOO-UNIFIED-IMPLEMENTATION-PLAN.md v2.1"

# Import core classes
from src.optimization.trading_oracle import (
    TradingOracle,
    PortfolioOracle,
    StrategySelectionOracle,
    GateProgressionOracle,
    ObjectiveSpec,
    ConstraintSpec,
    OracleResult,
    create_oracle,
)

from src.optimization.pymoo_adapter import (
    PymooAdapter,
    PymooConfig,
    OptimizationResult,
    run_nsga2,
    PYMOO_AVAILABLE,
)

from src.optimization.globalmoo_client import (
    GlobalMOOClient,
    GlobalMOOConfig,
    MockGlobalMOOClient,
    get_globalmoo_client,
)

from src.optimization.robust_pipeline import (
    WovenMOOPipeline,
    RobustMOOPipeline,
    WovenConfig,
    ResultCache,
    run_woven_optimization,
    run_pymoo_only,
)

__all__ = [
    # Oracle classes
    "TradingOracle",
    "PortfolioOracle",
    "StrategySelectionOracle",
    "GateProgressionOracle",
    "ObjectiveSpec",
    "ConstraintSpec",
    "OracleResult",
    "create_oracle",
    # Pymoo
    "PymooAdapter",
    "PymooConfig",
    "OptimizationResult",
    "run_nsga2",
    "PYMOO_AVAILABLE",
    # GlobalMOO
    "GlobalMOOClient",
    "GlobalMOOConfig",
    "MockGlobalMOOClient",
    "get_globalmoo_client",
    # Pipeline
    "WovenMOOPipeline",
    "RobustMOOPipeline",
    "WovenConfig",
    "ResultCache",
    "run_woven_optimization",
    "run_pymoo_only",
]
