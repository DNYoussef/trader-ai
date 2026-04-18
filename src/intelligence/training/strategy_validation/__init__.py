"""
Strategy Validation Battery

Comprehensive validation suite for trading strategies including:
- Backtest core computations
- Parameter sensitivity analysis
- Stress tests (slippage, delay, fees)
- Monte Carlo simulations
- MCPT integration

All validation must pass before a strategy can be promoted to higher gates.
"""

from .backtest_core import (
    BacktestAssumptions,
    compute_bar_returns,
    objective_profit_factor,
    objective_sharpe,
    objective_max_drawdown,
    objective_calmar,
)

from .parameter_sensitivity import (
    run_parameter_grid,
    compute_cliff_score,
    export_heatmap_artifact,
)

from .stress_tests import (
    stress_slippage,
    stress_delay,
    stress_fee,
    stress_combined,
)

from .monte_carlo import (
    shuffle_bar_returns_mc,
    block_bootstrap_mc,
    compute_ruin_probability,
)

from .validation_battery import (
    ValidationBattery,
    ValidationResult,
    ValidationConfig,
    run_full_validation,
)

__all__ = [
    # Backtest core
    'BacktestAssumptions',
    'compute_bar_returns',
    'objective_profit_factor',
    'objective_sharpe',
    'objective_max_drawdown',
    'objective_calmar',
    # Parameter sensitivity
    'run_parameter_grid',
    'compute_cliff_score',
    'export_heatmap_artifact',
    # Stress tests
    'stress_slippage',
    'stress_delay',
    'stress_fee',
    'stress_combined',
    # Monte Carlo
    'shuffle_bar_returns_mc',
    'block_bootstrap_mc',
    'compute_ruin_probability',
    # Validation battery
    'ValidationBattery',
    'ValidationResult',
    'ValidationConfig',
    'run_full_validation',
]
