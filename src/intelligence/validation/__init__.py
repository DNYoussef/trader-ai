"""
Validation Package

Provides MCPT (Monte Carlo Permutation Testing) validation for trading strategies.

Components:
- objectives: Objective functions (profit_factor, sharpe, sortino, calmar)
- mcpt_validator: MCPT validation wrapper
"""

from .objectives import (
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    compute_all_metrics,
    get_objective_function,
)

from .mcpt_validator import (
    MCPTValidator,
    MCPTResult,
    passes_gate,
)

__all__ = [
    # Objectives
    'profit_factor',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'max_drawdown',
    'compute_all_metrics',
    'get_objective_function',
    # MCPT
    'MCPTValidator',
    'MCPTResult',
    'passes_gate',
]
