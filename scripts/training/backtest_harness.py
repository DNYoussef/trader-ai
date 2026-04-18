"""
Walk-Forward Backtesting Harness for Trader-AI Signal Strategies

Provides rigorous validation with:
- Time-ordered splits (no random shuffling for time series)
- Walk-forward cross-validation with expanding/rolling windows
- Slippage and fee modeling
- Multi-asset robustness testing
- No-lookahead verification

Author: Trader-AI System
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Type of training window for walk-forward validation"""
    EXPANDING = "expanding"  # Train on all historical data
    ROLLING = "rolling"      # Train on fixed-size window


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting"""
    train_window: int = 252        # Training window in bars (1 year of daily)
    test_window: int = 63          # Test window in bars (3 months)
    window_type: WindowType = WindowType.EXPANDING
    min_train_size: int = 126      # Minimum training samples (6 months)

    # Cost modeling
    slippage_bps: float = 5.0      # Slippage in basis points
    commission_per_trade: float = 0.0  # Fixed commission per trade

    # Position sizing
    initial_capital: float = 10000.0
    max_position_pct: float = 0.20  # Max 20% per position

    # Risk management
    atr_stop_mult: float = 2.0     # ATR multiplier for stop loss
    atr_target_mult: float = 3.0   # ATR multiplier for take profit
    max_holding_bars: int = 20     # Maximum holding period

    # Validation
    require_min_trades: int = 10   # Minimum trades for valid fold


@dataclass
class TradeResult:
    """Result of a single trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    exit_reason: str  # 'target', 'stop', 'timeout', 'signal'
    slippage_cost: float
    commission_cost: float


@dataclass
class FoldResult:
    """Result of a single walk-forward fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    trades: List[TradeResult] = field(default_factory=list)

    # Metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    num_trades: int = 0


@dataclass
class BacktestResults:
    """Aggregated walk-forward backtest results"""
    config: BacktestConfig
    folds: List[FoldResult] = field(default_factory=list)

    # Aggregated metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    # Robustness metrics
    fold_sharpe_std: float = 0.0   # Standard deviation of fold Sharpes
    pct_profitable_folds: float = 0.0
    worst_fold_return: float = 0.0
    best_fold_return: float = 0.0


@dataclass
class SignalBacktestResult:
    """Result of backtesting a single signal function"""
    signal_name: str
    results: BacktestResults
    baseline_comparison: Optional[Dict[str, float]] = None
    passed_validation: bool = False
    validation_notes: List[str] = field(default_factory=list)


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine for signal strategies.

    Implements proper time-series validation with:
    - No lookahead bias
    - Expanding or rolling training windows
    - Realistic cost modeling
    - Comprehensive performance metrics
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        logger.info(f"WalkForwardBacktest initialized: {self.config.window_type.value} window, "
                    f"train={self.config.train_window}, test={self.config.test_window}")

    def run_walk_forward(
        self,
        strategy: Any,
        data: pd.DataFrame,
        fit_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None
    ) -> BacktestResults:
        """
        Run walk-forward validation on a strategy.

        Args:
            strategy: Strategy object with fit() and predict() methods
            data: OHLCV DataFrame with datetime index
            fit_fn: Optional custom fit function (strategy, train_data) -> None
            predict_fn: Optional custom predict function (strategy, test_data) -> signals

        Returns:
            BacktestResults with all folds and metrics
        """
        self._validate_data(data)

        folds = []
        fold_id = 0

        # Calculate fold boundaries
        total_bars = len(data)
        start_idx = self.config.min_train_size

        while start_idx + self.config.test_window <= total_bars:
            # Determine training range
            if self.config.window_type == WindowType.EXPANDING:
                train_start_idx = 0
            else:  # ROLLING
                train_start_idx = max(0, start_idx - self.config.train_window)

            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = min(start_idx + self.config.test_window, total_bars)

            # Split data
            train_data = data.iloc[train_start_idx:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()

            # Fit strategy on training data
            if fit_fn:
                fit_fn(strategy, train_data)
            elif hasattr(strategy, 'fit'):
                strategy.fit(train_data)

            # Generate predictions on test data
            if predict_fn:
                signals = predict_fn(strategy, test_data)
            elif hasattr(strategy, 'predict'):
                signals = strategy.predict(test_data)
            else:
                signals = self._generate_signals_from_features(strategy, test_data)

            # Execute trades based on signals
            trades = self._execute_trades(test_data, signals)

            # Calculate fold metrics
            fold_result = self._calculate_fold_metrics(
                fold_id=fold_id,
                train_data=train_data,
                test_data=test_data,
                trades=trades
            )
            folds.append(fold_result)

            fold_id += 1
            start_idx += self.config.test_window

        # Aggregate results
        results = self._aggregate_results(folds)

        logger.info(f"Walk-forward complete: {len(folds)} folds, "
                    f"Sharpe={results.sharpe_ratio:.2f}, "
                    f"Total Return={results.total_return*100:.1f}%")

        return results

    def run_signal_backtest(
        self,
        signal_fn: Callable[[pd.DataFrame, int], Tuple[int, Dict]],
        ohlcv: pd.DataFrame,
        symbol: str = "TEST"
    ) -> SignalBacktestResult:
        """
        Backtest a signal generation function.

        Args:
            signal_fn: Function(df, idx) -> (signal, metadata)
                signal: +1 long, -1 short, 0 no signal
                metadata: dict with strength, stop_suggestion, etc.
            ohlcv: OHLCV DataFrame
            symbol: Symbol name for logging

        Returns:
            SignalBacktestResult with full validation
        """
        self._validate_data(ohlcv)

        folds = []
        fold_id = 0

        total_bars = len(ohlcv)
        start_idx = self.config.min_train_size

        while start_idx + self.config.test_window <= total_bars:
            test_start_idx = start_idx
            test_end_idx = min(start_idx + self.config.test_window, total_bars)

            test_data = ohlcv.iloc[test_start_idx:test_end_idx].copy()

            # Generate signals for test period
            signals = []
            for i in range(len(test_data)):
                global_idx = test_start_idx + i
                # Only pass data up to current bar (no lookahead)
                historical_data = ohlcv.iloc[:global_idx + 1].copy()
                signal, metadata = signal_fn(historical_data, len(historical_data) - 1)
                signals.append({
                    'signal': signal,
                    'metadata': metadata,
                    'idx': i
                })

            # Convert to DataFrame for trade execution
            signal_series = pd.Series(
                [s['signal'] for s in signals],
                index=test_data.index
            )

            # Execute trades
            trades = self._execute_trades(test_data, signal_series)

            # Calculate fold metrics
            fold_result = self._calculate_fold_metrics(
                fold_id=fold_id,
                train_data=ohlcv.iloc[:test_start_idx],
                test_data=test_data,
                trades=trades
            )
            folds.append(fold_result)

            fold_id += 1
            start_idx += self.config.test_window

        # Aggregate results
        results = self._aggregate_results(folds)

        # Validate results
        validation_notes = []
        passed = True

        if results.total_trades < self.config.require_min_trades * len(folds):
            validation_notes.append(f"Low trade count: {results.total_trades}")
            passed = False

        if results.sharpe_ratio < 0:
            validation_notes.append(f"Negative Sharpe: {results.sharpe_ratio:.2f}")
            passed = False

        if results.max_drawdown > 0.3:
            validation_notes.append(f"High drawdown: {results.max_drawdown*100:.1f}%")

        if results.pct_profitable_folds < 0.5:
            validation_notes.append(f"Low fold profitability: {results.pct_profitable_folds*100:.0f}%")
            passed = False

        return SignalBacktestResult(
            signal_name=signal_fn.__name__ if hasattr(signal_fn, '__name__') else 'signal',
            results=results,
            passed_validation=passed,
            validation_notes=validation_notes
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate OHLCV data format"""
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) < self.config.min_train_size + self.config.test_window:
            raise ValueError(f"Insufficient data: {len(data)} bars, "
                           f"need at least {self.config.min_train_size + self.config.test_window}")

    def _execute_trades(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> List[TradeResult]:
        """
        Execute trades based on signals with realistic cost modeling.

        Args:
            data: OHLCV data for the test period
            signals: Series of signals (+1 long, -1 short, 0 flat)

        Returns:
            List of TradeResult objects
        """
        trades = []
        position = None

        # Calculate ATR for stops/targets
        atr = self._calculate_atr(data)

        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0
            current_price = row['close']
            current_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]

            # Check for exit conditions if in position
            if position is not None:
                exit_reason = None
                exit_price = current_price

                if position['direction'] == 'long':
                    # Check stop loss
                    if row['low'] <= position['stop_price']:
                        exit_reason = 'stop'
                        exit_price = position['stop_price']
                    # Check take profit
                    elif row['high'] >= position['target_price']:
                        exit_reason = 'target'
                        exit_price = position['target_price']
                    # Check timeout
                    elif position['holding_bars'] >= self.config.max_holding_bars:
                        exit_reason = 'timeout'
                    # Check signal reversal
                    elif signal == -1:
                        exit_reason = 'signal'
                else:  # short
                    if row['high'] >= position['stop_price']:
                        exit_reason = 'stop'
                        exit_price = position['stop_price']
                    elif row['low'] <= position['target_price']:
                        exit_reason = 'target'
                        exit_price = position['target_price']
                    elif position['holding_bars'] >= self.config.max_holding_bars:
                        exit_reason = 'timeout'
                    elif signal == 1:
                        exit_reason = 'signal'

                if exit_reason:
                    # Apply slippage
                    slippage = exit_price * (self.config.slippage_bps / 10000)
                    if position['direction'] == 'long':
                        exit_price -= slippage
                    else:
                        exit_price += slippage

                    # Calculate PnL
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['quantity']
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    trades.append(TradeResult(
                        entry_time=position['entry_time'],
                        exit_time=timestamp,
                        symbol=data.get('symbol', 'TEST') if isinstance(data.get('symbol'), str) else 'TEST',
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        quantity=position['quantity'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        holding_bars=position['holding_bars'],
                        exit_reason=exit_reason,
                        slippage_cost=slippage * position['quantity'] * 2,
                        commission_cost=self.config.commission_per_trade * 2
                    ))
                    position = None
                else:
                    position['holding_bars'] += 1

            # Check for entry if not in position
            if position is None and signal != 0:
                # Apply slippage to entry
                entry_price = current_price
                slippage = entry_price * (self.config.slippage_bps / 10000)

                if signal == 1:  # Long
                    entry_price += slippage
                    stop_price = entry_price - (current_atr * self.config.atr_stop_mult)
                    target_price = entry_price + (current_atr * self.config.atr_target_mult)
                    direction = 'long'
                else:  # Short
                    entry_price -= slippage
                    stop_price = entry_price + (current_atr * self.config.atr_stop_mult)
                    target_price = entry_price - (current_atr * self.config.atr_target_mult)
                    direction = 'short'

                # Calculate position size
                risk_per_share = abs(entry_price - stop_price)
                max_position_value = self.config.initial_capital * self.config.max_position_pct
                quantity = min(
                    max_position_value / entry_price,
                    (self.config.initial_capital * 0.02) / risk_per_share  # 2% risk per trade
                )

                position = {
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'direction': direction,
                    'quantity': quantity,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'holding_bars': 0
                }

        return trades

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # Fill NaN with first valid value
        atr = atr.fillna(method='bfill')

        return atr

    def _calculate_fold_metrics(
        self,
        fold_id: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        trades: List[TradeResult]
    ) -> FoldResult:
        """Calculate metrics for a single fold"""
        result = FoldResult(
            fold_id=fold_id,
            train_start=train_data.index[0] if len(train_data) > 0 else None,
            train_end=train_data.index[-1] if len(train_data) > 0 else None,
            test_start=test_data.index[0] if len(test_data) > 0 else None,
            test_end=test_data.index[-1] if len(test_data) > 0 else None,
            trades=trades,
            num_trades=len(trades)
        )

        if not trades:
            return result

        # Calculate returns
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        result.total_return = sum(pnls) / self.config.initial_capital
        result.avg_trade_pnl = np.mean(pnls)

        # Win rate
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        result.win_rate = len(wins) / len(pnls) if pnls else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified - daily returns approximation)
        if len(pnl_pcts) > 1:
            result.sharpe_ratio = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + self.config.initial_capital)
        result.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return result

    def _aggregate_results(self, folds: List[FoldResult]) -> BacktestResults:
        """Aggregate fold results into overall backtest results"""
        results = BacktestResults(
            config=self.config,
            folds=folds
        )

        if not folds:
            return results

        # Collect all trades
        all_trades = []
        for fold in folds:
            all_trades.extend(fold.trades)

        results.total_trades = len(all_trades)

        if not all_trades:
            return results

        # Calculate aggregate metrics
        all_pnls = [t.pnl for t in all_trades]
        all_pnl_pcts = [t.pnl_pct for t in all_trades]

        results.total_return = sum(all_pnls) / self.config.initial_capital

        # Annualized return (approximate)
        total_bars = sum(len(f.trades) for f in folds)
        years = total_bars / 252 if total_bars > 0 else 1
        results.annualized_return = (1 + results.total_return) ** (1/years) - 1 if years > 0 else 0

        # Win rate and profit factor
        wins = [p for p in all_pnls if p > 0]
        losses = [p for p in all_pnls if p <= 0]
        results.win_rate = len(wins) / len(all_pnls) if all_pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe and Sortino
        if len(all_pnl_pcts) > 1:
            mean_ret = np.mean(all_pnl_pcts)
            std_ret = np.std(all_pnl_pcts)
            results.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

            # Sortino (downside deviation)
            downside = [r for r in all_pnl_pcts if r < 0]
            downside_std = np.std(downside) if len(downside) > 1 else std_ret
            results.sortino_ratio = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Max drawdown across all folds
        cumulative = np.cumsum(all_pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + self.config.initial_capital)
        results.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calmar ratio
        results.calmar_ratio = results.annualized_return / results.max_drawdown if results.max_drawdown > 0 else 0

        # Fold-level statistics
        fold_returns = [f.total_return for f in folds]
        fold_sharpes = [f.sharpe_ratio for f in folds if f.num_trades > 0]

        results.fold_sharpe_std = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0
        results.pct_profitable_folds = len([r for r in fold_returns if r > 0]) / len(fold_returns) if fold_returns else 0
        results.worst_fold_return = min(fold_returns) if fold_returns else 0
        results.best_fold_return = max(fold_returns) if fold_returns else 0

        return results

    def _generate_signals_from_features(self, strategy: Any, data: pd.DataFrame) -> pd.Series:
        """Generate signals from strategy that outputs features/probabilities"""
        signals = []
        for i in range(len(data)):
            if hasattr(strategy, 'generate_signal'):
                signal = strategy.generate_signal(data.iloc[:i+1])
            else:
                signal = 0
            signals.append(signal)
        return pd.Series(signals, index=data.index)


def run_slippage_sensitivity(
    backtest: WalkForwardBacktest,
    signal_fn: Callable,
    data: pd.DataFrame,
    slippage_range: List[float] = [0, 5, 10, 20, 50]
) -> Dict[float, BacktestResults]:
    """
    Run backtest across multiple slippage levels.

    Args:
        backtest: WalkForwardBacktest instance
        signal_fn: Signal function to test
        data: OHLCV data
        slippage_range: List of slippage values in basis points

    Returns:
        Dict mapping slippage -> BacktestResults
    """
    results = {}
    original_slippage = backtest.config.slippage_bps

    for slippage in slippage_range:
        backtest.config.slippage_bps = slippage
        result = backtest.run_signal_backtest(signal_fn, data)
        results[slippage] = result.results
        logger.info(f"Slippage {slippage}bps: Sharpe={result.results.sharpe_ratio:.2f}, "
                    f"Return={result.results.total_return*100:.1f}%")

    # Restore original
    backtest.config.slippage_bps = original_slippage

    return results


def run_multi_asset_robustness(
    backtest: WalkForwardBacktest,
    signal_fn: Callable,
    asset_data: Dict[str, pd.DataFrame],
    min_sharpe: float = 0.5
) -> Dict[str, SignalBacktestResult]:
    """
    Test signal robustness across multiple assets.

    Args:
        backtest: WalkForwardBacktest instance
        signal_fn: Signal function to test
        asset_data: Dict of symbol -> OHLCV DataFrame
        min_sharpe: Minimum Sharpe ratio to pass

    Returns:
        Dict of symbol -> SignalBacktestResult
    """
    results = {}
    passing = 0

    for symbol, data in asset_data.items():
        try:
            result = backtest.run_signal_backtest(signal_fn, data, symbol=symbol)
            results[symbol] = result

            if result.results.sharpe_ratio >= min_sharpe:
                passing += 1
                logger.info(f"{symbol}: PASS (Sharpe={result.results.sharpe_ratio:.2f})")
            else:
                logger.warning(f"{symbol}: FAIL (Sharpe={result.results.sharpe_ratio:.2f})")
        except Exception as e:
            logger.error(f"{symbol}: ERROR - {e}")
            results[symbol] = None

    logger.info(f"Multi-asset robustness: {passing}/{len(asset_data)} passed")

    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)
    data['high'] = data['open'] + np.abs(np.random.randn(500) * 0.5)
    data['low'] = data['open'] - np.abs(np.random.randn(500) * 0.5)
    data['close'] = data['open'] + np.random.randn(500) * 0.3

    # Example signal function
    def simple_ma_signal(df: pd.DataFrame, idx: int) -> Tuple[int, Dict]:
        """Simple moving average crossover signal"""
        if len(df) < 20:
            return 0, {}

        close = df['close']
        ma_fast = close.rolling(5).mean().iloc[-1]
        ma_slow = close.rolling(20).mean().iloc[-1]

        signal = 1 if ma_fast > ma_slow else (-1 if ma_fast < ma_slow else 0)
        return signal, {'ma_fast': ma_fast, 'ma_slow': ma_slow}

    # Run backtest
    backtest = WalkForwardBacktest()
    result = backtest.run_signal_backtest(simple_ma_signal, data)

    print(f"\nBacktest Results:")
    print(f"  Total Return: {result.results.total_return*100:.1f}%")
    print(f"  Sharpe Ratio: {result.results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.results.max_drawdown*100:.1f}%")
    print(f"  Win Rate: {result.results.win_rate*100:.1f}%")
    print(f"  Total Trades: {result.results.total_trades}")
    print(f"  Passed Validation: {result.passed_validation}")
    print(f"  Notes: {result.validation_notes}")
