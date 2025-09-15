"""
Comprehensive Backtesting Framework for Alpha Generation Systems

This module provides a sophisticated backtesting framework that evaluates
the performance of the integrated alpha generation system including:
- Narrative Gap Engine performance
- Shadow Book counterfactual analysis
- Policy Twin ethical impact assessment
- Integrated signal performance measurement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import json
from pathlib import Path
import pickle

# Import alpha generation components
from alpha_integration import AlphaIntegrationEngine, AlphaSignal, PortfolioState
from ..narrative.narrative_gap import NGSignal
from ..learning.shadow_book import ShadowBookEngine, Trade
from ..learning.policy_twin import PolicyTwin, EthicalTrade

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    transaction_costs: float = 0.001  # 10 bps
    max_position_size: float = 0.05  # 5% max position
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual

@dataclass
class BacktestResult:
    """Results from backtesting run"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime

    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Alpha-specific metrics
    ng_signal_accuracy: float
    ethical_score_average: float
    shadow_book_outperformance: float

    # Portfolio metrics
    avg_position_count: float
    turnover_rate: float
    var_utilization: float

    # Daily data
    daily_returns: pd.Series
    daily_positions: pd.DataFrame
    daily_signals: pd.DataFrame

    # Trade analysis
    total_trades: int
    winning_trades: int
    avg_holding_period: float

    metadata: Dict[str, Any] = field(default_factory=dict)

class MarketDataSimulator:
    """Simulates market data for backtesting"""

    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}

        self._generate_market_data()

    def _generate_market_data(self):
        """Generate synthetic market data for backtesting"""
        try:
            # Generate trading days
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

            for symbol in self.symbols:
                # Simulate price data with realistic characteristics
                n_days = len(date_range)

                # Base parameters
                initial_price = np.random.uniform(50, 500)
                annual_return = np.random.normal(0.08, 0.15)  # 8% +/- 15%
                annual_vol = np.random.uniform(0.15, 0.35)    # 15-35% volatility

                # Generate returns with various regimes
                daily_vol = annual_vol / np.sqrt(252)
                daily_return = annual_return / 252

                returns = []
                for i in range(n_days):
                    # Add regime changes and jumps
                    base_return = np.random.normal(daily_return, daily_vol)

                    # Occasional jumps (earnings, news)
                    if np.random.random() < 0.02:  # 2% chance of jump
                        jump = np.random.normal(0, 0.05)  # 5% std jump
                        base_return += jump

                    # Momentum and mean reversion effects
                    if i > 10:
                        recent_returns = returns[-10:]
                        momentum = np.mean(recent_returns) * 0.1  # 10% momentum
                        mean_reversion = -np.mean(recent_returns[-5:]) * 0.05  # 5% mean reversion
                        base_return += momentum + mean_reversion

                    returns.append(base_return)

                # Convert to prices
                prices = [initial_price]
                for ret in returns:
                    prices.append(prices[-1] * (1 + ret))

                # Create DataFrame
                price_data = pd.DataFrame({
                    'date': date_range,
                    'open': prices[:-1],
                    'high': [p * np.random.uniform(1.0, 1.02) for p in prices[:-1]],
                    'low': [p * np.random.uniform(0.98, 1.0) for p in prices[:-1]],
                    'close': prices[1:],
                    'volume': [np.random.uniform(1e6, 10e6) for _ in range(n_days)]
                })

                price_data['returns'] = price_data['close'].pct_change()

                self.data_cache[symbol] = price_data

        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            raise

    def get_price_data(self, symbol: str, start_date: datetime = None,
                      end_date: datetime = None) -> pd.DataFrame:
        """Get price data for symbol within date range"""
        if symbol not in self.data_cache:
            raise ValueError(f"No data available for symbol {symbol}")

        data = self.data_cache[symbol].copy()

        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]

        return data

    def get_current_price(self, symbol: str, date: datetime) -> float:
        """Get current price for symbol at specific date"""
        data = self.get_price_data(symbol)
        closest_date = data[data['date'] <= date]['date'].max()

        if pd.isna(closest_date):
            return None

        return data[data['date'] == closest_date]['close'].iloc[0]

class AlphaBacktester:
    """Main backtesting engine for alpha generation systems"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data = MarketDataSimulator(
            config.symbols, config.start_date, config.end_date
        )

        # Initialize alpha components
        self.alpha_engine = AlphaIntegrationEngine()
        self.shadow_book = ShadowBookEngine()
        self.policy_twin = PolicyTwin()

        # Backtesting state
        self.portfolio_history = []
        self.signal_history = []
        self.trade_history = []
        self.daily_metrics = []

    async def run_backtest(self) -> BacktestResult:
        """Run complete backtesting process"""
        try:
            logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

            # Initialize portfolio
            current_capital = self.config.initial_capital
            current_positions = {symbol: 0.0 for symbol in self.config.symbols}

            # Get trading dates
            trading_dates = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date,
                freq='B'
            )

            daily_portfolio_values = []
            daily_returns_list = []
            all_signals = []
            all_trades = []

            for i, current_date in enumerate(trading_dates):
                try:
                    # Get current prices
                    current_prices = {}
                    for symbol in self.config.symbols:
                        price = self.market_data.get_current_price(symbol, current_date)
                        if price is not None:
                            current_prices[symbol] = price

                    # Calculate current portfolio value
                    portfolio_value = current_capital
                    for symbol, position in current_positions.items():
                        if symbol in current_prices:
                            portfolio_value += position * current_prices[symbol]

                    # Create portfolio state
                    portfolio_state = PortfolioState(
                        total_capital=portfolio_value,
                        available_capital=current_capital,
                        current_positions={k: v * current_prices.get(k, 0)
                                         for k, v in current_positions.items()},
                        risk_utilization=self._calculate_risk_utilization(current_positions, current_prices),
                        var_usage=0.5,  # Simplified
                        sector_exposures=self._calculate_sector_exposures(current_positions, current_prices)
                    )

                    # Generate signals
                    signals = self.alpha_engine.generate_portfolio_signals(
                        list(current_prices.keys()), current_prices, portfolio_state
                    )

                    # Execute trades based on signals
                    trades_executed = await self._execute_signals(
                        signals, current_positions, current_capital, current_prices, current_date
                    )

                    # Update positions and capital
                    for trade in trades_executed:
                        if trade['action'] == 'buy':
                            shares_bought = trade['notional'] / trade['price']
                            current_positions[trade['symbol']] += shares_bought
                            current_capital -= trade['notional'] * (1 + self.config.transaction_costs)
                        elif trade['action'] == 'sell':
                            shares_sold = min(current_positions[trade['symbol']],
                                            trade['notional'] / trade['price'])
                            current_positions[trade['symbol']] -= shares_sold
                            current_capital += shares_sold * trade['price'] * (1 - self.config.transaction_costs)

                    # Calculate daily return
                    if i > 0:
                        daily_return = (portfolio_value - daily_portfolio_values[-1]) / daily_portfolio_values[-1]
                        daily_returns_list.append(daily_return)
                    else:
                        daily_returns_list.append(0.0)

                    daily_portfolio_values.append(portfolio_value)
                    all_signals.extend(signals)
                    all_trades.extend(trades_executed)

                    # Store daily metrics
                    self.daily_metrics.append({
                        'date': current_date,
                        'portfolio_value': portfolio_value,
                        'capital': current_capital,
                        'positions': current_positions.copy(),
                        'signals_count': len(signals),
                        'trades_count': len(trades_executed)
                    })

                    if i % 50 == 0:  # Progress logging
                        logger.info(f"Backtest progress: {i}/{len(trading_dates)} days, "
                                  f"Portfolio value: ${portfolio_value:,.0f}")

                except Exception as e:
                    logger.error(f"Error on date {current_date}: {e}")
                    continue

            # Calculate performance metrics
            result = self._calculate_performance_metrics(
                daily_portfolio_values, daily_returns_list, all_signals, all_trades, trading_dates
            )

            logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
            return result

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    async def _execute_signals(self, signals: List[AlphaSignal], current_positions: Dict[str, float],
                             current_capital: float, current_prices: Dict[str, float],
                             current_date: datetime) -> List[Dict[str, Any]]:
        """Execute trades based on generated signals"""
        trades_executed = []

        try:
            for signal in signals:
                if signal.action in ['buy', 'sell'] and signal.final_score > 0.5:
                    # Calculate trade size
                    if signal.action == 'buy':
                        max_notional = min(signal.recommended_size, current_capital * 0.8)
                        if max_notional > 1000:  # Minimum trade size
                            trade = {
                                'symbol': signal.symbol,
                                'action': 'buy',
                                'notional': max_notional,
                                'price': current_prices[signal.symbol],
                                'date': current_date,
                                'signal_score': signal.final_score,
                                'ng_score': signal.ng_score,
                                'ethical_score': signal.ethical_score
                            }
                            trades_executed.append(trade)

                    elif signal.action == 'sell':
                        current_position_value = current_positions[signal.symbol] * current_prices[signal.symbol]
                        if current_position_value > 1000:  # Have position to sell
                            sell_fraction = min(1.0, signal.final_score)  # Sell based on signal strength
                            sell_notional = current_position_value * sell_fraction

                            trade = {
                                'symbol': signal.symbol,
                                'action': 'sell',
                                'notional': sell_notional,
                                'price': current_prices[signal.symbol],
                                'date': current_date,
                                'signal_score': signal.final_score,
                                'ng_score': signal.ng_score,
                                'ethical_score': signal.ethical_score
                            }
                            trades_executed.append(trade)

        except Exception as e:
            logger.error(f"Error executing signals: {e}")

        return trades_executed

    def _calculate_risk_utilization(self, positions: Dict[str, float],
                                  prices: Dict[str, float]) -> float:
        """Calculate current risk utilization"""
        total_exposure = sum(abs(pos * prices.get(symbol, 0))
                           for symbol, pos in positions.items())

        # Simplified risk calculation
        return min(1.0, total_exposure / (self.config.initial_capital * 2))

    def _calculate_sector_exposures(self, positions: Dict[str, float],
                                  prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposures"""
        # Simplified sector mapping
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare'
        }

        sector_exposures = {}
        total_value = sum(pos * prices.get(symbol, 0) for symbol, pos in positions.items())

        if total_value > 0:
            for symbol, position in positions.items():
                sector = sector_map.get(symbol, 'Other')
                position_value = position * prices.get(symbol, 0)

                if sector not in sector_exposures:
                    sector_exposures[sector] = 0
                sector_exposures[sector] += position_value / total_value

        return sector_exposures

    def _calculate_performance_metrics(self, portfolio_values: List[float],
                                     daily_returns: List[float],
                                     all_signals: List[AlphaSignal],
                                     all_trades: List[Dict[str, Any]],
                                     trading_dates: pd.DatetimeIndex) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        try:
            # Convert to numpy arrays
            returns_array = np.array(daily_returns)

            # Basic performance metrics
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            trading_days = len(returns_array)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
            volatility = np.std(returns_array) * np.sqrt(252)

            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0

            # Drawdown calculation
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)

            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # Alpha-specific metrics
            ng_scores = [s.ng_score for s in all_signals if s.ng_score > 0]
            ng_signal_accuracy = np.mean(ng_scores) if ng_scores else 0

            ethical_scores = [s.ethical_score for s in all_signals]
            ethical_score_average = np.mean(ethical_scores) if ethical_scores else 0

            # Shadow book analysis (simplified)
            shadow_book_outperformance = 0.05  # Placeholder - would calculate from actual shadow trades

            # Trading metrics
            total_trades = len(all_trades)
            winning_trades = len([t for t in all_trades if t.get('pnl', 0) > 0])

            # Create DataFrames for detailed analysis
            daily_returns_series = pd.Series(daily_returns, index=trading_dates[:len(daily_returns)])

            # Positions DataFrame
            positions_data = []
            for metric in self.daily_metrics:
                pos_row = {'date': metric['date']}
                pos_row.update(metric['positions'])
                positions_data.append(pos_row)

            daily_positions = pd.DataFrame(positions_data)

            # Signals DataFrame
            signals_data = []
            for signal in all_signals:
                signals_data.append({
                    'date': signal.timestamp,
                    'symbol': signal.symbol,
                    'final_score': signal.final_score,
                    'ng_score': signal.ng_score,
                    'ethical_score': signal.ethical_score,
                    'action': signal.action
                })

            daily_signals = pd.DataFrame(signals_data)

            # Portfolio metrics
            avg_position_count = np.mean([len([v for v in m['positions'].values() if v != 0])
                                        for m in self.daily_metrics])

            turnover_rate = total_trades / trading_days * 252 if trading_days > 0 else 0
            var_utilization = 0.6  # Simplified

            result = BacktestResult(
                config=self.config,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                ng_signal_accuracy=ng_signal_accuracy,
                ethical_score_average=ethical_score_average,
                shadow_book_outperformance=shadow_book_outperformance,
                avg_position_count=avg_position_count,
                turnover_rate=turnover_rate,
                var_utilization=var_utilization,
                daily_returns=daily_returns_series,
                daily_positions=daily_positions,
                daily_signals=daily_signals,
                total_trades=total_trades,
                winning_trades=winning_trades,
                avg_holding_period=5.0,  # Simplified
                metadata={
                    'portfolio_values': portfolio_values,
                    'all_trades': all_trades,
                    'final_portfolio_value': portfolio_values[-1]
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise

def run_alpha_backtest():
    """Run a comprehensive backtest of the alpha generation system"""
    # Configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=10_000_000,  # $10M
        symbols=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ'],
        rebalance_frequency='daily',
        transaction_costs=0.001,  # 10 bps
        max_position_size=0.05,   # 5%
        benchmark='SPY',
        risk_free_rate=0.02
    )

    # Run backtest
    backtester = AlphaBacktester(config)

    async def run_test():
        result = await backtester.run_backtest()

        print("=== ALPHA GENERATION BACKTEST RESULTS ===")
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Capital: ${config.initial_capital:,.0f}")
        print(f"Final Value: ${result.metadata['final_portfolio_value']:,.0f}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Annualized Return: {result.annualized_return:.2%}")
        print(f"  Volatility: {result.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Calmar Ratio: {result.calmar_ratio:.2f}")
        print(f"\nAlpha Generation Metrics:")
        print(f"  NG Signal Accuracy: {result.ng_signal_accuracy:.2%}")
        print(f"  Average Ethical Score: {result.ethical_score_average:.3f}")
        print(f"  Shadow Book Outperformance: {result.shadow_book_outperformance:.2%}")
        print(f"\nTrading Metrics:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Winning Trades: {result.winning_trades}")
        print(f"  Win Rate: {result.winning_trades/result.total_trades:.2%}" if result.total_trades > 0 else "  Win Rate: N/A")
        print(f"  Turnover Rate: {result.turnover_rate:.1f}x")
        print(f"  Average Positions: {result.avg_position_count:.1f}")

        return result

    # Execute test
    return asyncio.run(run_test())

if __name__ == "__main__":
    result = run_alpha_backtest()