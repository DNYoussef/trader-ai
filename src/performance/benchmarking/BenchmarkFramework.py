"""
Gary×Taleb Trading Strategy Benchmarking Framework
Performance analysis and comparison system for antifragile trading strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for trading strategies"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    value_at_risk_95: float
    conditional_var_95: float
    skewness: float
    kurtosis: float
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    recovery_factor: float
    ulcer_index: float
    antifragility_score: float
    dpi_alignment_score: float

    # Risk-adjusted metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    tracking_error: float = 0.0

    # Additional Taleb-inspired metrics
    black_swan_protection: float = 0.0
    convexity_score: float = 0.0
    optionality_ratio: float = 0.0

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.trades: List[Dict] = []
        self.equity_curve: pd.Series = pd.Series()

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the strategy"""
        pass

    @abstractmethod
    def backtest(self, data: pd.DataFrame, initial_capital: float = 200.0) -> Dict:
        """Run backtest for the strategy"""
        pass

class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold benchmark strategy"""

    def __init__(self):
        super().__init__("Buy & Hold", "Simple buy and hold strategy")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at start
        return signals

    def backtest(self, data: pd.DataFrame, initial_capital: float = 200.0) -> Dict:
        returns = data['Close'].pct_change().fillna(0)
        self.equity_curve = (1 + returns).cumprod() * initial_capital

        return {
            'equity_curve': self.equity_curve,
            'returns': returns,
            'trades': [{'type': 'buy', 'price': data['Close'].iloc[0], 'date': data.index[0]}]
        }

class MovingAverageCrossover(BaseStrategy):
    """Moving average crossover strategy"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(f"MA Crossover ({fast_period}/{slow_period})",
                        f"Moving average crossover strategy with {fast_period} and {slow_period} periods")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data['Close'].rolling(window=self.fast_period).mean()
        slow_ma = data['Close'].rolling(window=self.slow_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma <= slow_ma] = -1

        return signals.fillna(0)

    def backtest(self, data: pd.DataFrame, initial_capital: float = 200.0) -> Dict:
        signals = self.generate_signals(data)
        returns = data['Close'].pct_change().fillna(0)

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        self.equity_curve = (1 + strategy_returns).cumprod() * initial_capital

        # Track trades
        trades = []
        position = 0
        for i, signal in enumerate(signals):
            if signal != position:
                trades.append({
                    'type': 'buy' if signal > position else 'sell',
                    'price': data['Close'].iloc[i],
                    'date': data.index[i],
                    'signal': signal
                })
                position = signal

        return {
            'equity_curve': self.equity_curve,
            'returns': strategy_returns,
            'trades': trades
        }

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using RSI"""

    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"Mean Reversion (RSI {rsi_period})",
                        f"Mean reversion strategy using RSI with {oversold}/{overbought} levels")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi = self.calculate_rsi(data['Close'])

        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1   # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought

        return signals.fillna(0)

    def backtest(self, data: pd.DataFrame, initial_capital: float = 200.0) -> Dict:
        signals = self.generate_signals(data)
        returns = data['Close'].pct_change().fillna(0)

        strategy_returns = signals.shift(1) * returns
        self.equity_curve = (1 + strategy_returns).cumprod() * initial_capital

        trades = []
        position = 0
        for i, signal in enumerate(signals):
            if signal != position:
                trades.append({
                    'type': 'buy' if signal > position else 'sell',
                    'price': data['Close'].iloc[i],
                    'date': data.index[i],
                    'signal': signal
                })
                position = signal

        return {
            'equity_curve': self.equity_curve,
            'returns': strategy_returns,
            'trades': trades
        }

class GaryTalebStrategy(BaseStrategy):
    """Gary×Taleb antifragile trading strategy implementation"""

    def __init__(self, dpi_threshold: float = 0.7, antifragility_factor: float = 1.5):
        super().__init__("Gary×Taleb Strategy",
                        "Antifragile trading strategy combining Gary's DPI and Taleb's principles")
        self.dpi_threshold = dpi_threshold
        self.antifragility_factor = antifragility_factor

    def calculate_dpi_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Gary's DPI (Dynamic Performance Indicator) score"""
        # Momentum component
        momentum = data['Close'].pct_change(20)

        # Volatility component (lower is better for stability)
        volatility = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()

        # Volume confirmation
        volume_sma = data.get('Volume', pd.Series(1, index=data.index)).rolling(20).mean()
        volume_ratio = data.get('Volume', pd.Series(1, index=data.index)) / volume_sma

        # DPI score calculation
        dpi_score = (momentum * 0.4) + ((1 - volatility) * 0.3) + (volume_ratio * 0.3)
        return dpi_score.fillna(0)

    def calculate_antifragility_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Taleb-inspired antifragility score"""
        returns = data['Close'].pct_change()

        # Convexity measure (positive skew preference)
        rolling_skew = returns.rolling(20).skew()

        # Tail protection (looking for positive tail events)
        rolling_kurt = returns.rolling(20).kurt()

        # Volatility clustering (antifragile systems benefit from volatility)
        vol_clustering = returns.rolling(5).std() / returns.rolling(20).std()

        antifragility = (rolling_skew * 0.4) + (rolling_kurt * 0.3) + (vol_clustering * 0.3)
        return antifragility.fillna(0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        dpi_score = self.calculate_dpi_score(data)
        antifragility_score = self.calculate_antifragility_score(data)

        # Combined signal
        combined_score = (dpi_score + antifragility_score * self.antifragility_factor) / 2

        signals = pd.Series(0, index=data.index)
        signals[combined_score > self.dpi_threshold] = 1
        signals[combined_score < -self.dpi_threshold] = -1

        return signals.fillna(0)

    def backtest(self, data: pd.DataFrame, initial_capital: float = 200.0) -> Dict:
        signals = self.generate_signals(data)
        returns = data['Close'].pct_change().fillna(0)

        # Apply position sizing based on signal strength
        dpi_score = self.calculate_dpi_score(data)
        antifragility_score = self.calculate_antifragility_score(data)
        combined_score = (dpi_score + antifragility_score * self.antifragility_factor) / 2

        # Dynamic position sizing
        position_size = np.abs(combined_score) * signals
        position_size = np.clip(position_size, -1, 1)  # Limit leverage

        strategy_returns = position_size.shift(1) * returns
        self.equity_curve = (1 + strategy_returns).cumprod() * initial_capital

        trades = []
        position = 0
        for i, signal in enumerate(signals):
            if signal != position:
                trades.append({
                    'type': 'buy' if signal > position else 'sell',
                    'price': data['Close'].iloc[i],
                    'date': data.index[i],
                    'signal': signal,
                    'dpi_score': dpi_score.iloc[i],
                    'antifragility_score': antifragility_score.iloc[i],
                    'position_size': position_size.iloc[i]
                })
                position = signal

        return {
            'equity_curve': self.equity_curve,
            'returns': strategy_returns,
            'trades': trades,
            'dpi_scores': dpi_score,
            'antifragility_scores': antifragility_score
        }

class BenchmarkFramework:
    """Comprehensive benchmarking framework for trading strategies"""

    def __init__(self, initial_capital: float = 200.0):
        self.initial_capital = initial_capital
        self.strategies: Dict[str, BaseStrategy] = {}
        self.results: Dict[str, Dict] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}

        # Initialize baseline strategies
        self._initialize_baseline_strategies()

    def _initialize_baseline_strategies(self):
        """Initialize baseline comparison strategies"""
        self.strategies = {
            'buy_hold': BuyAndHoldStrategy(),
            'ma_crossover_fast': MovingAverageCrossover(5, 20),
            'ma_crossover_slow': MovingAverageCrossover(20, 50),
            'mean_reversion': MeanReversionStrategy(),
            'gary_taleb': GaryTalebStrategy()
        }

    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a custom strategy to the benchmark"""
        self.strategies[name] = strategy

    def run_benchmark(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Run benchmark tests for all strategies"""
        print("Running comprehensive benchmark tests...")

        for name, strategy in self.strategies.items():
            print(f"Testing {strategy.name}...")
            try:
                result = strategy.backtest(data, self.initial_capital)
                self.results[name] = result

                # Calculate performance metrics
                self.performance_metrics[name] = self._calculate_metrics(
                    result['returns'], result['equity_curve']
                )

            except Exception as e:
                print(f"Error testing {strategy.name}: {str(e)}")
                continue

        return self.results

    def _calculate_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        # Basic return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_volatility if downside_volatility > 0 else 0

        # Drawdown metrics
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                drawdown_duration = max(drawdown_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR and CVaR
        var_95 = np.percentile(returns.dropna(), 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0

        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurt()

        # Trade metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns.dropna()) if len(returns.dropna()) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win * len(positive_returns) / (avg_loss * len(negative_returns))) if avg_loss != 0 else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Ulcer Index
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())

        # Recovery Factor
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Omega Ratio (simplified)
        threshold_return = 0.0
        gains = returns[returns > threshold_return].sum()
        losses = abs(returns[returns <= threshold_return].sum())
        omega_ratio = gains / losses if losses > 0 else float('inf')

        # Tail Ratio
        tail_ratio = abs(np.percentile(returns.dropna(), 95) / np.percentile(returns.dropna(), 5)) if np.percentile(returns.dropna(), 5) != 0 else 0

        # Antifragility Score (Taleb-inspired)
        # Measures how much the strategy benefits from volatility
        vol_periods = returns.rolling(20).std()
        return_periods = returns.rolling(20).mean()
        correlation = vol_periods.corr(return_periods) if not vol_periods.isna().all() and not return_periods.isna().all() else 0
        antifragility_score = max(0, correlation)  # Positive correlation means antifragile

        # DPI Alignment Score (Gary-inspired)
        # Measures consistency of performance
        rolling_returns = returns.rolling(20).mean()
        consistency = 1 - rolling_returns.std() / abs(rolling_returns.mean()) if rolling_returns.mean() != 0 else 0
        dpi_alignment_score = max(0, min(1, consistency))

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            tail_ratio=tail_ratio,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            antifragility_score=antifragility_score,
            dpi_alignment_score=dpi_alignment_score
        )

    def get_performance_summary(self) -> pd.DataFrame:
        """Get comprehensive performance summary table"""
        if not self.performance_metrics:
            return pd.DataFrame()

        summary_data = []
        for name, metrics in self.performance_metrics.items():
            strategy_name = self.strategies[name].name
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.3f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.3f}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Antifragility': f"{metrics.antifragility_score:.3f}",
                'DPI Alignment': f"{metrics.dpi_alignment_score:.3f}"
            })

        df = pd.DataFrame(summary_data)
        return df.sort_values('Sharpe Ratio', ascending=False)

    def plot_equity_curves(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot equity curves for all strategies"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Gary×Taleb Strategy Benchmarking Results', fontsize=16, fontweight='bold')

        # Equity curves
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            strategy_name = self.strategies[name].name
            equity_curve = result['equity_curve']
            ax1.plot(equity_curve.index, equity_curve.values, label=strategy_name, linewidth=2)

        ax1.set_title('Equity Curves Comparison', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown comparison
        ax2 = axes[0, 1]
        for name, result in self.results.items():
            strategy_name = self.strategies[name].name
            equity_curve = result['equity_curve']
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=strategy_name)

        ax2.set_title('Drawdown Comparison', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Risk-Return scatter
        ax3 = axes[1, 0]
        returns = []
        risks = []
        names = []

        for name, metrics in self.performance_metrics.items():
            returns.append(metrics.annualized_return)
            risks.append(metrics.volatility)
            names.append(self.strategies[name].name)

        scatter = ax3.scatter(risks, returns, s=100, alpha=0.7, c=range(len(names)), cmap='viridis')

        for i, name in enumerate(names):
            ax3.annotate(name, (risks[i], returns[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        ax3.set_title('Risk-Return Profile', fontweight='bold')
        ax3.set_xlabel('Volatility (Risk)')
        ax3.set_ylabel('Annualized Return')
        ax3.grid(True, alpha=0.3)

        # Performance metrics radar
        ax4 = axes[1, 1]

        # Select key metrics for radar chart
        gary_taleb_metrics = self.performance_metrics.get('gary_taleb')
        if gary_taleb_metrics:
            metrics_names = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                           'Win Rate', 'Antifragility', 'DPI Alignment']

            # Normalize metrics for radar chart
            values = [
                min(gary_taleb_metrics.sharpe_ratio / 3, 1),  # Normalize Sharpe to 0-1
                min(gary_taleb_metrics.sortino_ratio / 3, 1),
                min(gary_taleb_metrics.calmar_ratio / 5, 1),
                gary_taleb_metrics.win_rate,
                gary_taleb_metrics.antifragility_score,
                gary_taleb_metrics.dpi_alignment_score
            ]

            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))

            ax4.plot(angles, values, 'o-', linewidth=2, label='Gary×Taleb Strategy')
            ax4.fill(angles, values, alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_names)
            ax4.set_ylim(0, 1)
            ax4.set_title('Gary×Taleb Strategy Profile', fontweight='bold')
            ax4.grid(True)

        plt.tight_layout()
        plt.show()

        return fig

    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        if not self.performance_metrics:
            return "No benchmark results available. Run benchmark first."

        report = []
        report.append("=" * 80)
        report.append("GARY×TALEB TRADING STRATEGY BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Capital: ${self.initial_capital}")
        report.append("")

        # Performance summary
        summary_df = self.get_performance_summary()
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(summary_df.to_string(index=False))
        report.append("")

        # Gary×Taleb strategy analysis
        if 'gary_taleb' in self.performance_metrics:
            gt_metrics = self.performance_metrics['gary_taleb']
            report.append("GARY×TALEB STRATEGY DETAILED ANALYSIS")
            report.append("-" * 45)
            report.append(f"Total Return: {gt_metrics.total_return:.2%}")
            report.append(f"Annualized Return: {gt_metrics.annualized_return:.2%}")
            report.append(f"Sharpe Ratio: {gt_metrics.sharpe_ratio:.3f}")
            report.append(f"Maximum Drawdown: {gt_metrics.max_drawdown:.2%}")
            report.append(f"Antifragility Score: {gt_metrics.antifragility_score:.3f}")
            report.append(f"DPI Alignment Score: {gt_metrics.dpi_alignment_score:.3f}")
            report.append("")

            # Comparison with best performer
            best_sharpe = max(self.performance_metrics.values(), key=lambda x: x.sharpe_ratio)
            if gt_metrics.sharpe_ratio >= best_sharpe.sharpe_ratio:
                report.append("🏆 Gary×Taleb strategy OUTPERFORMS all benchmark strategies!")
            else:
                improvement_needed = ((best_sharpe.sharpe_ratio / gt_metrics.sharpe_ratio) - 1) * 100
                report.append(f"Gary×Taleb strategy needs {improvement_needed:.1f}% Sharpe improvement to match best performer")
            report.append("")

        # Risk analysis
        report.append("RISK ANALYSIS")
        report.append("-" * 20)
        for name, metrics in self.performance_metrics.items():
            strategy_name = self.strategies[name].name
            report.append(f"{strategy_name}:")
            report.append(f"  VaR (95%): {metrics.value_at_risk_95:.2%}")
            report.append(f"  CVaR (95%): {metrics.conditional_var_95:.2%}")
            report.append(f"  Max Drawdown Duration: {metrics.max_drawdown_duration} periods")
            report.append(f"  Ulcer Index: {metrics.ulcer_index:.3f}")
            report.append("")

        # Recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 35)

        if 'gary_taleb' in self.performance_metrics:
            gt_metrics = self.performance_metrics['gary_taleb']

            if gt_metrics.sharpe_ratio < 1.0:
                report.append("• Consider increasing position sizing during high-conviction signals")

            if gt_metrics.max_drawdown < -0.15:
                report.append("• Implement additional risk management stops")

            if gt_metrics.win_rate < 0.5:
                report.append("• Review signal generation logic for better accuracy")

            if gt_metrics.antifragility_score < 0.3:
                report.append("• Enhance volatility-based position sizing")

            report.append("• Continue monitoring for black swan events")
            report.append("• Regular rebalancing of DPI parameters")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Load sample data for testing
    print("Initializing Gary×Taleb Benchmarking Framework...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * (1 + returns).cumprod()
    volumes = np.random.lognormal(10, 0.5, len(dates))

    sample_data = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    # Run benchmark
    framework = BenchmarkFramework(initial_capital=200.0)
    results = framework.run_benchmark(sample_data)

    # Generate report
    print("\n" + framework.generate_benchmark_report())

    # Plot results
    framework.plot_equity_curves()