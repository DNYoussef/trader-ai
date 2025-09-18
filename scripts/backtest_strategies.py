"""
Black Swan Strategy Backtesting Engine
Comprehensive historical simulation and performance analysis

Tests strategies across 30 years of market data to measure:
- Black swan capture rate
- Convexity achieved
- Risk-adjusted returns
- Maximum drawdowns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from src.strategies.black_swan_strategies import (
    BlackSwanStrategyToolbox,
    MarketState,
    StrategySignal
)
from src.strategies.convex_reward_function import (
    ConvexRewardFunction,
    TradeOutcome
)
from src.data.historical_data_manager import HistoricalDataManager
from src.data.black_swan_labeler import BlackSwanLabeler

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    position_sizing: str = 'fixed'  # 'fixed', 'kelly', 'volatility'
    max_position_pct: float = 0.20  # 20% max per position
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    rebalance_frequency: int = 20  # Days between rebalancing
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage

@dataclass
class BacktestResult:
    """Results from a backtest"""
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    black_swan_captures: int
    convexity_ratio: float
    calmar_ratio: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    equity_curve: pd.Series
    trade_history: List[Dict]
    metrics: Dict[str, Any]


class StrategyBacktester:
    """Backtests individual strategies on historical data"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.convex_reward = ConvexRewardFunction()
        self.trades = []
        self.equity_curve = []
        self.current_positions = {}

    def backtest_strategy(self,
                         strategy_name: str,
                         data: pd.DataFrame,
                         start_date: str = None,
                         end_date: str = None) -> BacktestResult:
        """Run backtest for a single strategy"""

        logger.info(f"Backtesting {strategy_name} strategy...")

        # Filter date range
        data['date'] = pd.to_datetime(data['date'])
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]

        # Initialize
        self.trades = []
        self.equity_curve = []
        self.current_positions = {}
        capital = self.config.initial_capital

        # Get unique dates
        dates = sorted(data['date'].unique())

        # Create strategy toolbox
        toolbox = BlackSwanStrategyToolbox()
        strategy = toolbox.strategies.get(strategy_name)

        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return None

        # Simulate trading
        for i, date in enumerate(dates):
            # Get market state
            market_state = self._get_market_state(data, date)

            # Get historical data for analysis
            hist_data = data[data['date'] <= date].tail(100)

            # Generate signal
            signal = strategy.analyze(market_state, hist_data)

            if signal and signal.action != 'hold':
                # Execute trade
                trade = self._execute_trade(signal, date, capital, data)
                if trade:
                    self.trades.append(trade)
                    capital = trade['capital_after']

            # Update positions
            capital = self._update_positions(date, capital, data)

            # Record equity
            self.equity_curve.append({
                'date': date,
                'capital': capital,
                'return': (capital / self.config.initial_capital - 1)
            })

            # Rebalance periodically
            if i % self.config.rebalance_frequency == 0:
                capital = self._rebalance_portfolio(date, capital)

        # Calculate metrics
        result = self._calculate_metrics(strategy_name)

        logger.info(f"Backtest complete for {strategy_name}: "
                   f"Return={result.total_return:.2%}, "
                   f"Sharpe={result.sharpe_ratio:.2f}")

        return result

    def _get_market_state(self, data: pd.DataFrame, date: pd.Timestamp) -> MarketState:
        """Extract market state for a specific date"""

        recent_data = data[data['date'] <= date].tail(60)

        # SPY data
        spy_data = recent_data[recent_data['symbol'] == 'SPY']
        if not spy_data.empty:
            returns_5d = spy_data['returns'].tail(5).sum() if len(spy_data) >= 5 else 0
            returns_20d = spy_data['returns'].tail(20).sum() if len(spy_data) >= 20 else 0
        else:
            returns_5d = returns_20d = 0

        # VIX data
        vix_data = recent_data[recent_data['symbol'] == 'VIX']
        if not vix_data.empty and 'close' in vix_data.columns:
            vix_level = vix_data['close'].iloc[-1]
            vix_percentile = (vix_data['close'].iloc[-1] <= vix_data['close']).mean()
        else:
            vix_level = 20
            vix_percentile = 0.5

        # Market breadth (simplified)
        symbols_up = (recent_data.groupby('symbol')['returns'].last() > 0).sum()
        total_symbols = recent_data['symbol'].nunique()
        market_breadth = symbols_up / total_symbols if total_symbols > 0 else 0.5

        # Correlation (simplified)
        if total_symbols > 5:
            pivot = recent_data.pivot(index='date', columns='symbol', values='returns')
            if len(pivot) >= 20:
                corr_matrix = pivot.tail(20).corr()
                avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            else:
                avg_correlation = 0.5
        else:
            avg_correlation = 0.5

        # Volume ratio
        if 'volume' in spy_data.columns:
            current_vol = spy_data['volume'].iloc[-1] if not spy_data.empty else 1
            avg_vol = spy_data['volume'].mean() if not spy_data.empty else 1
            volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        else:
            volume_ratio = 1

        # Determine regime
        if vix_level > 30:
            regime = 'crisis'
        elif vix_level > 20:
            regime = 'volatile'
        else:
            regime = 'normal'

        return MarketState(
            timestamp=date.to_pydatetime(),
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            spy_returns_5d=returns_5d,
            spy_returns_20d=returns_20d,
            put_call_ratio=1.0,  # Placeholder
            market_breadth=market_breadth,
            correlation=avg_correlation,
            volume_ratio=volume_ratio,
            regime=regime
        )

    def _execute_trade(self,
                      signal: StrategySignal,
                      date: pd.Timestamp,
                      capital: float,
                      data: pd.DataFrame) -> Optional[Dict]:
        """Execute a trade based on signal"""

        # Calculate position size
        position_size = min(
            capital * float(signal.allocation_pct),
            capital * self.config.max_position_pct
        )

        # Apply transaction costs
        cost = position_size * self.config.transaction_cost_pct
        position_size -= cost

        # Get entry price (with slippage)
        symbol_data = data[(data['date'] == date) & (data['symbol'] == 'SPY')]
        if symbol_data.empty:
            return None

        entry_price = symbol_data['close'].iloc[0]
        if signal.action == 'buy':
            entry_price *= (1 + self.config.slippage_pct)
        else:
            entry_price *= (1 - self.config.slippage_pct)

        # Create trade record
        trade = {
            'entry_date': date,
            'exit_date': None,
            'symbol': signal.symbol,
            'strategy': signal.strategy_name,
            'action': signal.action,
            'entry_price': entry_price,
            'exit_price': None,
            'position_size': position_size,
            'shares': position_size / entry_price,
            'stop_loss': entry_price * (1 - self.config.stop_loss_pct),
            'take_profit': entry_price * (1 + self.config.take_profit_pct),
            'capital_before': capital,
            'capital_after': capital - position_size - cost,
            'return': None,
            'is_open': True
        }

        # Store position
        self.current_positions[f"{signal.symbol}_{date}"] = trade

        return trade

    def _update_positions(self, date: pd.Timestamp, capital: float, data: pd.DataFrame) -> float:
        """Update open positions and check exit conditions"""

        for position_id, position in list(self.current_positions.items()):
            if not position['is_open']:
                continue

            # Get current price
            symbol_data = data[(data['date'] == date) & (data['symbol'] == 'SPY')]
            if symbol_data.empty:
                continue

            current_price = symbol_data['close'].iloc[0]

            # Check exit conditions
            should_exit = False
            exit_reason = None

            if current_price <= position['stop_loss']:
                should_exit = True
                exit_reason = 'stop_loss'
            elif current_price >= position['take_profit']:
                should_exit = True
                exit_reason = 'take_profit'
            elif (date - position['entry_date']).days > 20:  # Time exit
                should_exit = True
                exit_reason = 'time_exit'

            if should_exit:
                # Close position
                exit_price = current_price * (1 - self.config.slippage_pct)
                position['exit_date'] = date
                position['exit_price'] = exit_price
                position['return'] = (exit_price - position['entry_price']) / position['entry_price']
                position['is_open'] = False

                # Update capital
                capital += position['shares'] * exit_price
                capital -= position['shares'] * exit_price * self.config.transaction_cost_pct

                # Remove from current positions
                del self.current_positions[position_id]

        return capital

    def _rebalance_portfolio(self, date: pd.Timestamp, capital: float) -> float:
        """Rebalance portfolio positions"""
        # Simplified rebalancing - close positions that are too large
        for position_id, position in list(self.current_positions.items()):
            if position['position_size'] > capital * self.config.max_position_pct * 1.5:
                # Reduce position
                # This is simplified - would be more complex in production
                pass

        return capital

    def _calculate_metrics(self, strategy_name: str) -> BacktestResult:
        """Calculate backtest metrics"""

        if not self.equity_curve:
            return BacktestResult(
                strategy_name=strategy_name,
                total_return=0,
                annualized_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                black_swan_captures=0,
                convexity_ratio=0,
                calmar_ratio=0,
                avg_trade_duration=0,
                best_trade=0,
                worst_trade=0,
                equity_curve=pd.Series(),
                trade_history=self.trades,
                metrics={}
            )

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # Calculate returns
        equity_df['daily_return'] = equity_df['capital'].pct_change()
        total_return = (equity_df['capital'].iloc[-1] / self.config.initial_capital) - 1

        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

        # Sharpe ratio
        daily_returns = equity_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        closed_trades = [t for t in self.trades if not t.get('is_open', True)]
        if closed_trades:
            returns = [t['return'] for t in closed_trades if t['return'] is not None]
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r < 0]

            win_rate = len(winning_trades) / len(returns) if returns else 0

            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
            profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else float('inf')

            best_trade = max(returns) if returns else 0
            worst_trade = min(returns) if returns else 0

            # Trade duration
            durations = [(t['exit_date'] - t['entry_date']).days for t in closed_trades
                        if t.get('exit_date') and t.get('entry_date')]
            avg_duration = np.mean(durations) if durations else 0
        else:
            win_rate = profit_factor = best_trade = worst_trade = avg_duration = 0

        # Black swan captures (trades with >50% return)
        black_swan_captures = len([r for r in returns if r > 0.5]) if closed_trades else 0

        # Convexity ratio (average upside / average downside)
        if winning_trades and losing_trades:
            convexity_ratio = np.mean(winning_trades) / abs(np.mean(losing_trades))
        else:
            convexity_ratio = 0

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(closed_trades),
            black_swan_captures=black_swan_captures,
            convexity_ratio=convexity_ratio,
            calmar_ratio=calmar,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            equity_curve=equity_df['capital'],
            trade_history=self.trades,
            metrics={
                'total_days': days,
                'volatility': daily_returns.std() * np.sqrt(252),
                'skew': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis()
            }
        )


class PortfolioBacktester:
    """Backtests portfolio of strategies"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_backtester = StrategyBacktester(config)
        self.portfolio_results = {}

    def backtest_portfolio(self,
                          strategy_weights: Dict[str, float],
                          data: pd.DataFrame,
                          start_date: str = None,
                          end_date: str = None) -> Dict[str, Any]:
        """Backtest a portfolio of strategies"""

        logger.info("Backtesting portfolio of strategies...")

        # Backtest each strategy
        for strategy_name, weight in strategy_weights.items():
            if weight > 0:
                result = self.strategy_backtester.backtest_strategy(
                    strategy_name, data, start_date, end_date
                )
                if result:
                    self.portfolio_results[strategy_name] = {
                        'weight': weight,
                        'result': result
                    }

        # Combine results
        portfolio_metrics = self._combine_portfolio_results()

        return portfolio_metrics

    def _combine_portfolio_results(self) -> Dict[str, Any]:
        """Combine individual strategy results into portfolio metrics"""

        if not self.portfolio_results:
            return {}

        # Weight-adjusted returns
        total_return = sum(
            r['weight'] * r['result'].total_return
            for r in self.portfolio_results.values()
        )

        # Weight-adjusted Sharpe
        sharpe = sum(
            r['weight'] * r['result'].sharpe_ratio
            for r in self.portfolio_results.values()
        )

        # Total trades
        total_trades = sum(
            r['result'].total_trades
            for r in self.portfolio_results.values()
        )

        # Black swan captures
        black_swan_captures = sum(
            r['result'].black_swan_captures
            for r in self.portfolio_results.values()
        )

        # Average convexity
        avg_convexity = np.mean([
            r['result'].convexity_ratio
            for r in self.portfolio_results.values()
        ])

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'total_trades': total_trades,
            'black_swan_captures': black_swan_captures,
            'avg_convexity': avg_convexity,
            'strategy_results': self.portfolio_results
        }


def plot_backtest_results(results: List[BacktestResult], save_path: Path = None):
    """Plot backtest results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Equity curves
    ax = axes[0, 0]
    for result in results:
        if not result.equity_curve.empty:
            ax.plot(result.equity_curve.index, result.equity_curve.values,
                   label=result.strategy_name, alpha=0.7)
    ax.set_title('Equity Curves')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Returns distribution
    ax = axes[0, 1]
    returns_data = []
    labels = []
    for result in results:
        if result.trade_history:
            returns = [t['return'] for t in result.trade_history
                      if t.get('return') is not None]
            if returns:
                returns_data.append(returns)
                labels.append(result.strategy_name)

    if returns_data:
        ax.boxplot(returns_data, labels=labels)
        ax.set_title('Returns Distribution')
        ax.set_ylabel('Return')
        ax.grid(True, alpha=0.3)

    # Risk-Return scatter
    ax = axes[1, 0]
    for result in results:
        ax.scatter(result.metrics.get('volatility', 0) * 100,
                  result.annualized_return * 100,
                  s=100, alpha=0.6, label=result.strategy_name)
    ax.set_title('Risk-Return Profile')
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics comparison
    ax = axes[1, 1]
    metrics_df = pd.DataFrame({
        'Sharpe': [r.sharpe_ratio for r in results],
        'Sortino': [r.sortino_ratio for r in results],
        'Calmar': [r.calmar_ratio for r in results]
    }, index=[r.strategy_name for r in results])

    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title('Risk-Adjusted Returns')
    ax.set_ylabel('Ratio')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def main():
    """Main backtest execution"""

    logger.info("Starting Strategy Backtesting Engine")

    # Configuration
    config = BacktestConfig()

    # Load data
    logger.info("Loading historical data...")
    manager = HistoricalDataManager()
    data = manager.get_training_data(
        start_date="2015-01-01",
        end_date="2024-12-31"
    )

    if data.empty:
        logger.error("No data available for backtesting")
        return False

    # Backtest individual strategies
    backtester = StrategyBacktester(config)
    strategies = ['tail_hedge', 'volatility_harvest', 'crisis_alpha',
                 'momentum_explosion', 'mean_reversion']

    results = []
    for strategy in strategies:
        result = backtester.backtest_strategy(strategy, data)
        if result:
            results.append(result)

            logger.info(f"\n{strategy} Results:")
            logger.info(f"  Total Return: {result.total_return:.2%}")
            logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"  Win Rate: {result.win_rate:.2%}")
            logger.info(f"  Black Swan Captures: {result.black_swan_captures}")

    # Test portfolio
    logger.info("\nBacktesting Portfolio...")
    portfolio_backtester = PortfolioBacktester(config)

    weights = {
        'tail_hedge': 0.20,
        'volatility_harvest': 0.15,
        'crisis_alpha': 0.25,
        'momentum_explosion': 0.20,
        'mean_reversion': 0.20
    }

    portfolio_results = portfolio_backtester.backtest_portfolio(weights, data)

    logger.info("\nPortfolio Results:")
    logger.info(f"  Total Return: {portfolio_results['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {portfolio_results['sharpe_ratio']:.2f}")
    logger.info(f"  Total Trades: {portfolio_results['total_trades']}")
    logger.info(f"  Black Swan Captures: {portfolio_results['black_swan_captures']}")
    logger.info(f"  Average Convexity: {portfolio_results['avg_convexity']:.2f}")

    # Plot results
    if results:
        plot_path = Path("reports/backtest_results.png")
        plot_path.parent.mkdir(exist_ok=True)
        plot_backtest_results(results, plot_path)

    logger.info("\n✅ Backtesting complete!")
    logger.info("📊 Results show strategy performance across market conditions")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)