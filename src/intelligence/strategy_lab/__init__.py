"""
Strategy Lab Package

Provides infrastructure for strategy selection and creation with MCPT validation.

Components:
- strategy_spec: StrategySpec dataclass for strategy definitions
- signal_interface: Position signal generation and return calculation
- strategy_selector: Model to select optimal strategy from 8 candidates
"""

from .strategy_spec import (
    StrategySpec,
    EXISTING_STRATEGIES,
    get_strategy_spec,
    get_strategy_by_name,
)
from .signal_interface import (
    build_position_signal,
    strategy_returns,
    equity_curve,
    position_from_signal_function,
    compare_signals,
)
from .strategy_selector import (
    StrategySelector,
    SelectionResult,
    SelectorMetrics,
    STRATEGY_NAMES,
)

__all__ = [
    # Strategy specs
    'StrategySpec',
    'EXISTING_STRATEGIES',
    'get_strategy_spec',
    'get_strategy_by_name',
    # Signal interface
    'build_position_signal',
    'strategy_returns',
    'equity_curve',
    'position_from_signal_function',
    'compare_signals',
    # Strategy selector
    'StrategySelector',
    'SelectionResult',
    'SelectorMetrics',
    'STRATEGY_NAMES',
]
