"""
Trading Performance Analyzer for Gary×Taleb Trading System

Advanced performance analysis with P&L correlation, attribution analysis,
regime-based performance evaluation, and comprehensive trading metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

@dataclass
class PnLAttribution:
    """P&L attribution analysis"""
    total_pnl: float
    model_contribution: float
    timing_contribution: float
    sizing_contribution: float
    market_contribution: float
    execution_contribution: float
    unexplained_contribution: float
    attribution_confidence: float

@dataclass
class RegimePerformance:
    """Performance metrics by market regime"""
    regime_name: str
    total_trades: int
    total_pnl: float
    win_rate: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    gary_dpi: float
    taleb_antifragility: float
    alpha: float
    beta: float
    information_ratio: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime

    # Basic metrics
    total_trades: int
    total_pnl: float
    total_return: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float

    # Gary×Taleb metrics
    gary_dpi: float
    taleb_antifragility: float
    stress_performance: float
    volatility_clustering: float

    # Model performance
    prediction_accuracy: float
    direction_accuracy: float
    correlation_actual_predicted: float
    r_squared: float
    information_coefficient: float

    # Execution metrics
    execution_quality: float
    slippage_impact: float
    transaction_cost_impact: float
    timing_efficiency: float

@dataclass
class CorrelationAnalysis:
    """Correlation analysis between predictions and returns"""
    pearson_correlation: float
    spearman_correlation: float
    kendall_tau: float
    r_squared: float
    significance_level: float
    confidence_interval: Tuple[float, float]
    rolling_correlation: List[float]
    regime_correlations: Dict[str, float]

class PerformanceAnalyzer:
    """
    Advanced trading performance analyzer with P&L correlation,
    attribution analysis, and comprehensive performance metrics.
    """

    def __init__(self, analysis_window_days: int = 30):
        self.analysis_window_days = analysis_window_days
        self.logger = self._setup_logging()

        # Database setup
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/performance_analyzer.db")
        self._init_database()

        # Market regimes
        self.market_regimes = ['low_volatility', 'high_volatility', 'trending_up', 'trending_down', 'sideways']

        # Benchmark data (would be loaded from external source)
        self.benchmark_returns = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance analyzer"""
        logger = logging.getLogger('PerformanceAnalyzer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/performance_analyzer.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for performance analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    predicted_return REAL,
                    actual_return REAL,
                    position_size REAL,
                    holding_period REAL,
                    entry_price REAL,
                    exit_price REAL,
                    transaction_costs REAL,
                    slippage REAL,
                    market_regime TEXT,
                    volatility REAL,
                    confidence_score REAL,
                    trade_pnl REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    attribution_json TEXT,
                    correlation_analysis_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS regime_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    regime_name TEXT NOT NULL,
                    performance_json TEXT NOT NULL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS pnl_attribution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_pnl REAL,
                    model_contribution REAL,
                    timing_contribution REAL,
                    sizing_contribution REAL,
                    market_contribution REAL,
                    execution_contribution REAL,
                    unexplained_contribution REAL,
                    attribution_confidence REAL
                )
            ''')

    def record_trade_performance(self, model_id: str, trade_data: Dict[str, Any]) -> bool:
        """
        Record trade performance data

        Args:
            model_id: Model identifier
            trade_data: Trade performance data

        Returns:
            success: Whether recording was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trading_performance (
                        timestamp, model_id, trade_id, predicted_return, actual_return,
                        position_size, holding_period, entry_price, exit_price,
                        transaction_costs, slippage, market_regime, volatility,
                        confidence_score, trade_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_id,
                    trade_data.get('trade_id', ''),
                    trade_data.get('predicted_return', 0.0),
                    trade_data.get('actual_return', 0.0),
                    trade_data.get('position_size', 0.0),
                    trade_data.get('holding_period', 0.0),
                    trade_data.get('entry_price', 0.0),
                    trade_data.get('exit_price', 0.0),
                    trade_data.get('transaction_costs', 0.0),
                    trade_data.get('slippage', 0.0),
                    trade_data.get('market_regime', 'unknown'),
                    trade_data.get('volatility', 0.0),
                    trade_data.get('confidence_score', 0.5),
                    trade_data.get('trade_pnl', 0.0)
                ))

            return True

        except Exception as e:
            self.logger.error(f"Error recording trade performance: {e}")
            return False

    def analyze_performance(self, model_id: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Optional[PerformanceMetrics]:
        """
        Analyze comprehensive trading performance

        Args:
            model_id: Model identifier
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            PerformanceMetrics object
        """
        try:
            # Set default date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.analysis_window_days)

            # Load trade data
            trade_data = self._load_trade_data(model_id, start_date, end_date)

            if not trade_data:
                self.logger.warning(f"No trade data found for model {model_id}")
                return None

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(trade_data, start_date, end_date)

            # Save analysis results
            self._save_performance_analysis(model_id, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing performance for model {model_id}: {e}")
            return None

    def _load_trade_data(self, model_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load trade data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM trading_performance
                    WHERE model_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''

                df = pd.read_sql_query(query, conn, params=(
                    model_id,
                    start_date.isoformat(),
                    end_date.isoformat()
                ))

                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                return df

        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")
            return pd.DataFrame()

    def _calculate_performance_metrics(self, trade_data: pd.DataFrame,
                                     start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if trade_data.empty:
                return self._create_empty_metrics(start_date, end_date)

            # Basic calculations
            total_trades = len(trade_data)
            actual_returns = trade_data['actual_return'].values
            predicted_returns = trade_data['predicted_return'].values
            trade_pnls = trade_data['trade_pnl'].values

            total_pnl = np.sum(trade_pnls)
            total_return = np.sum(actual_returns)

            # Win/loss metrics
            winning_trades = actual_returns > 0
            win_rate = np.mean(winning_trades)

            wins = actual_returns[winning_trades]
            losses = actual_returns[~winning_trades]

            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            max_win = np.max(wins) if len(wins) > 0 else 0
            max_loss = np.min(losses) if len(losses) > 0 else 0

            # Profit factor
            gross_profit = np.sum(wins) if len(wins) > 0 else 0
            gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Risk metrics
            volatility = np.std(actual_returns) if len(actual_returns) > 1 else 0

            # Sharpe ratio
            risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
            excess_returns = actual_returns - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

            # Sortino ratio
            downside_returns = actual_returns[actual_returns < 0]
            downside_volatility = np.std(downside_returns) if len(downside_returns) > 1 else volatility
            sortino_ratio = np.mean(excess_returns) / downside_volatility if downside_volatility > 0 else 0

            # Drawdown calculation
            cumulative_returns = np.cumprod(1 + actual_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Calmar ratio
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # VaR and CVaR
            var_95 = np.percentile(actual_returns, 5)
            cvar_95 = np.mean(actual_returns[actual_returns <= var_95])

            # Gary×Taleb specific metrics
            gary_dpi = self._calculate_gary_dpi(actual_returns)
            taleb_antifragility = self._calculate_taleb_antifragility(actual_returns)
            stress_performance = self._calculate_stress_performance(trade_data)
            volatility_clustering = self._calculate_volatility_clustering(trade_data)

            # Model performance metrics
            prediction_accuracy = np.mean(np.abs(actual_returns - predicted_returns))
            direction_accuracy = np.mean((actual_returns > 0) == (predicted_returns > 0))

            correlation_actual_predicted, _ = pearsonr(actual_returns, predicted_returns) if len(actual_returns) > 1 else (0, 1)
            r_squared = r2_score(actual_returns, predicted_returns) if len(actual_returns) > 1 else 0

            # Information coefficient (rank correlation)
            information_coefficient, _ = spearmanr(actual_returns, predicted_returns) if len(actual_returns) > 1 else (0, 1)

            # Execution metrics
            execution_quality = self._calculate_execution_quality(trade_data)
            slippage_impact = np.mean(trade_data['slippage'].values)
            transaction_cost_impact = np.mean(trade_data['transaction_costs'].values)
            timing_efficiency = self._calculate_timing_efficiency(trade_data)

            return PerformanceMetrics(
                timestamp=datetime.now(),
                period_start=start_date,
                period_end=end_date,
                total_trades=total_trades,
                total_pnl=total_pnl,
                total_return=total_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_win=max_win,
                max_loss=max_loss,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                gary_dpi=gary_dpi,
                taleb_antifragility=taleb_antifragility,
                stress_performance=stress_performance,
                volatility_clustering=volatility_clustering,
                prediction_accuracy=prediction_accuracy,
                direction_accuracy=direction_accuracy,
                correlation_actual_predicted=correlation_actual_predicted,
                r_squared=r_squared,
                information_coefficient=information_coefficient,
                execution_quality=execution_quality,
                slippage_impact=slippage_impact,
                transaction_cost_impact=transaction_cost_impact,
                timing_efficiency=timing_efficiency
            )

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_metrics(start_date, end_date)

    def _calculate_gary_dpi(self, returns: np.ndarray) -> float:
        """Calculate Gary's Dynamic Performance Index"""
        if len(returns) == 0:
            return 0.0

        avg_return = np.mean(returns)
        win_rate = np.mean(returns > 0)
        volatility = np.std(returns)

        # Max drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        denominator = max_drawdown + volatility
        gary_dpi = (avg_return * win_rate) / denominator if denominator > 0 else 0

        return gary_dpi

    def _calculate_taleb_antifragility(self, returns: np.ndarray) -> float:
        """Calculate Taleb's antifragility score"""
        if len(returns) < 4:
            return 0.0

        mean_return = np.mean(returns)
        volatility = np.std(returns)

        # Identify stress periods (high volatility)
        volatility_threshold = np.percentile(np.abs(returns - mean_return), 75)
        stress_periods = np.abs(returns - mean_return) > volatility_threshold

        if np.sum(stress_periods) == 0:
            return 0.0

        stress_returns = returns[stress_periods]
        normal_returns = returns[~stress_periods]

        stress_performance = np.mean(stress_returns) if len(stress_returns) > 0 else 0
        normal_performance = np.mean(normal_returns) if len(normal_returns) > 0 else 0

        # Antifragility: positive when stress performance exceeds normal
        antifragility = (stress_performance - normal_performance) / (volatility + 1e-8)

        return antifragility

    def _calculate_stress_performance(self, trade_data: pd.DataFrame) -> float:
        """Calculate performance during stress periods"""
        if trade_data.empty:
            return 0.0

        # Identify high volatility periods
        volatility_threshold = trade_data['volatility'].quantile(0.8)
        stress_trades = trade_data[trade_data['volatility'] > volatility_threshold]

        if stress_trades.empty:
            return 0.0

        stress_returns = stress_trades['actual_return'].values
        return np.mean(stress_returns)

    def _calculate_volatility_clustering(self, trade_data: pd.DataFrame) -> float:
        """Calculate volatility clustering measure"""
        if len(trade_data) < 10:
            return 0.0

        volatilities = trade_data['volatility'].values

        # Calculate autocorrelation of squared returns as proxy for clustering
        squared_vol = volatilities ** 2
        if len(squared_vol) > 1:
            autocorr = np.corrcoef(squared_vol[:-1], squared_vol[1:])[0, 1]
            return autocorr if not np.isnan(autocorr) else 0.0

        return 0.0

    def _calculate_execution_quality(self, trade_data: pd.DataFrame) -> float:
        """Calculate execution quality score"""
        if trade_data.empty:
            return 0.0

        # Simple execution quality based on slippage and costs
        avg_slippage = trade_data['slippage'].mean()
        avg_costs = trade_data['transaction_costs'].mean()

        # Lower slippage and costs = higher quality
        execution_quality = 1.0 - (avg_slippage + avg_costs)
        return max(0.0, min(1.0, execution_quality))

    def _calculate_timing_efficiency(self, trade_data: pd.DataFrame) -> float:
        """Calculate timing efficiency score"""
        if trade_data.empty:
            return 0.0

        # Efficiency based on holding period vs returns
        holding_periods = trade_data['holding_period'].values
        returns = trade_data['actual_return'].values

        if len(returns) > 1 and np.std(holding_periods) > 0:
            # Negative correlation suggests efficient timing (shorter holds for better returns)
            correlation, _ = pearsonr(holding_periods, returns)
            efficiency = -correlation if not np.isnan(correlation) else 0.0
            return max(-1.0, min(1.0, efficiency))

        return 0.0

    def _create_empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Create empty metrics object"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            total_trades=0,
            total_pnl=0.0,
            total_return=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            gary_dpi=0.0,
            taleb_antifragility=0.0,
            stress_performance=0.0,
            volatility_clustering=0.0,
            prediction_accuracy=0.0,
            direction_accuracy=0.0,
            correlation_actual_predicted=0.0,
            r_squared=0.0,
            information_coefficient=0.0,
            execution_quality=0.0,
            slippage_impact=0.0,
            transaction_cost_impact=0.0,
            timing_efficiency=0.0
        )

    def analyze_pnl_attribution(self, model_id: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Optional[PnLAttribution]:
        """
        Analyze P&L attribution across different factors

        Args:
            model_id: Model identifier
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            PnLAttribution object
        """
        try:
            # Set default date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.analysis_window_days)

            # Load trade data
            trade_data = self._load_trade_data(model_id, start_date, end_date)

            if trade_data.empty:
                return None

            total_pnl = trade_data['trade_pnl'].sum()

            # Model contribution (based on prediction accuracy)
            predicted_returns = trade_data['predicted_return'].values
            actual_returns = trade_data['actual_return'].values
            position_sizes = trade_data['position_size'].values

            # Calculate what P&L would have been with perfect predictions
            perfect_pnl = np.sum(np.abs(actual_returns) * position_sizes)
            predicted_pnl = np.sum(predicted_returns * position_sizes)
            model_contribution = predicted_pnl / perfect_pnl if perfect_pnl > 0 else 0

            # Timing contribution (based on holding periods)
            timing_contribution = self._calculate_timing_contribution(trade_data)

            # Sizing contribution (based on position sizing effectiveness)
            sizing_contribution = self._calculate_sizing_contribution(trade_data)

            # Market contribution (based on overall market performance)
            market_contribution = self._calculate_market_contribution(trade_data)

            # Execution contribution (slippage and costs)
            execution_costs = trade_data['slippage'].sum() + trade_data['transaction_costs'].sum()
            execution_contribution = -execution_costs / abs(total_pnl) if total_pnl != 0 else 0

            # Unexplained contribution
            explained_pnl = (model_contribution + timing_contribution + sizing_contribution +
                           market_contribution + execution_contribution)
            unexplained_contribution = 1.0 - explained_pnl

            # Attribution confidence (based on data quality and completeness)
            attribution_confidence = self._calculate_attribution_confidence(trade_data)

            attribution = PnLAttribution(
                total_pnl=total_pnl,
                model_contribution=model_contribution,
                timing_contribution=timing_contribution,
                sizing_contribution=sizing_contribution,
                market_contribution=market_contribution,
                execution_contribution=execution_contribution,
                unexplained_contribution=unexplained_contribution,
                attribution_confidence=attribution_confidence
            )

            # Save attribution analysis
            self._save_pnl_attribution(model_id, attribution, start_date, end_date)

            return attribution

        except Exception as e:
            self.logger.error(f"Error analyzing P&L attribution: {e}")
            return None

    def _calculate_timing_contribution(self, trade_data: pd.DataFrame) -> float:
        """Calculate timing contribution to P&L"""
        try:
            # Simplified timing analysis based on holding period efficiency
            returns = trade_data['actual_return'].values
            holding_periods = trade_data['holding_period'].values

            if len(returns) > 1:
                # Calculate efficiency as return per unit time
                efficiency = returns / (holding_periods + 1e-8)
                avg_efficiency = np.mean(efficiency)

                # Normalize to contribution percentage
                timing_contribution = np.tanh(avg_efficiency) * 0.2  # Cap at 20%
                return timing_contribution

        except Exception as e:
            self.logger.error(f"Error calculating timing contribution: {e}")

        return 0.0

    def _calculate_sizing_contribution(self, trade_data: pd.DataFrame) -> float:
        """Calculate position sizing contribution to P&L"""
        try:
            # Analyze if larger positions were taken on better trades
            returns = trade_data['actual_return'].values
            sizes = trade_data['position_size'].values

            if len(returns) > 1:
                # Correlation between position size and returns
                correlation, _ = pearsonr(sizes, returns)

                if not np.isnan(correlation):
                    # Positive correlation suggests good sizing
                    sizing_contribution = correlation * 0.15  # Cap at 15%
                    return sizing_contribution

        except Exception as e:
            self.logger.error(f"Error calculating sizing contribution: {e}")

        return 0.0

    def _calculate_market_contribution(self, trade_data: pd.DataFrame) -> float:
        """Calculate market contribution to P&L"""
        try:
            # Simplified market contribution based on regime performance
            regime_returns = {}

            for regime in self.market_regimes:
                regime_trades = trade_data[trade_data['market_regime'] == regime]
                if not regime_trades.empty:
                    regime_returns[regime] = regime_trades['actual_return'].mean()

            if regime_returns:
                # Market contribution is the average regime performance
                market_contribution = np.mean(list(regime_returns.values())) * 0.3  # Cap at 30%
                return market_contribution

        except Exception as e:
            self.logger.error(f"Error calculating market contribution: {e}")

        return 0.0

    def _calculate_attribution_confidence(self, trade_data: pd.DataFrame) -> float:
        """Calculate confidence in attribution analysis"""
        try:
            # Base confidence on data completeness and quality
            total_trades = len(trade_data)

            # Check for missing data
            missing_data_penalty = 0
            for col in ['predicted_return', 'actual_return', 'position_size', 'market_regime']:
                missing_ratio = trade_data[col].isnull().sum() / total_trades
                missing_data_penalty += missing_ratio * 0.2

            # Base confidence
            base_confidence = 0.8

            # Adjust for sample size
            sample_size_factor = min(1.0, total_trades / 100)  # Full confidence at 100+ trades

            confidence = (base_confidence - missing_data_penalty) * sample_size_factor

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating attribution confidence: {e}")
            return 0.5

    def analyze_correlation(self, model_id: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Optional[CorrelationAnalysis]:
        """
        Analyze correlation between predictions and actual returns

        Args:
            model_id: Model identifier
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            CorrelationAnalysis object
        """
        try:
            # Set default date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.analysis_window_days)

            # Load trade data
            trade_data = self._load_trade_data(model_id, start_date, end_date)

            if trade_data.empty or len(trade_data) < 2:
                return None

            predicted_returns = trade_data['predicted_return'].values
            actual_returns = trade_data['actual_return'].values

            # Calculate correlations
            pearson_corr, pearson_p = pearsonr(predicted_returns, actual_returns)
            spearman_corr, spearman_p = spearmanr(predicted_returns, actual_returns)
            kendall_tau, kendall_p = stats.kendalltau(predicted_returns, actual_returns)

            # R-squared
            r_squared = r2_score(actual_returns, predicted_returns)

            # Significance level (using Pearson p-value)
            significance_level = pearson_p

            # Confidence interval for Pearson correlation
            n = len(predicted_returns)
            z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr))
            se = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(0.975)  # 95% confidence interval

            ci_lower = np.tanh(z_score - z_critical * se)
            ci_upper = np.tanh(z_score + z_critical * se)
            confidence_interval = (ci_lower, ci_upper)

            # Rolling correlation (if enough data)
            rolling_correlation = []
            if len(trade_data) >= 20:
                window_size = min(20, len(trade_data) // 5)
                for i in range(window_size, len(trade_data)):
                    window_pred = predicted_returns[i-window_size:i]
                    window_actual = actual_returns[i-window_size:i]
                    corr, _ = pearsonr(window_pred, window_actual)
                    rolling_correlation.append(corr if not np.isnan(corr) else 0.0)

            # Regime-based correlations
            regime_correlations = {}
            for regime in self.market_regimes:
                regime_trades = trade_data[trade_data['market_regime'] == regime]
                if len(regime_trades) >= 2:
                    regime_pred = regime_trades['predicted_return'].values
                    regime_actual = regime_trades['actual_return'].values
                    regime_corr, _ = pearsonr(regime_pred, regime_actual)
                    regime_correlations[regime] = regime_corr if not np.isnan(regime_corr) else 0.0

            correlation_analysis = CorrelationAnalysis(
                pearson_correlation=pearson_corr if not np.isnan(pearson_corr) else 0.0,
                spearman_correlation=spearman_corr if not np.isnan(spearman_corr) else 0.0,
                kendall_tau=kendall_tau if not np.isnan(kendall_tau) else 0.0,
                r_squared=r_squared,
                significance_level=significance_level,
                confidence_interval=confidence_interval,
                rolling_correlation=rolling_correlation,
                regime_correlations=regime_correlations
            )

            return correlation_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing correlation: {e}")
            return None

    def analyze_regime_performance(self, model_id: str,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, RegimePerformance]:
        """
        Analyze performance by market regime

        Args:
            model_id: Model identifier
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dictionary of regime performance
        """
        try:
            # Set default date range
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=self.analysis_window_days)

            # Load trade data
            trade_data = self._load_trade_data(model_id, start_date, end_date)

            if trade_data.empty:
                return {}

            regime_performance = {}

            for regime in self.market_regimes:
                regime_trades = trade_data[trade_data['market_regime'] == regime]

                if regime_trades.empty:
                    continue

                # Calculate regime-specific metrics
                returns = regime_trades['actual_return'].values
                pnls = regime_trades['trade_pnl'].values

                total_trades = len(regime_trades)
                total_pnl = np.sum(pnls)
                win_rate = np.mean(returns > 0)
                avg_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0

                # Sharpe ratio
                excess_returns = returns - 0.02/252  # Risk-free rate
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

                # Max drawdown
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)

                # Gary DPI and Taleb antifragility
                gary_dpi = self._calculate_gary_dpi(returns)
                taleb_antifragility = self._calculate_taleb_antifragility(returns)

                # Alpha and beta (simplified, would need benchmark data)
                alpha = avg_return - 0.02/252  # Excess return over risk-free
                beta = 1.0  # Simplified assumption

                # Information ratio
                tracking_error = volatility
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0

                regime_performance[regime] = RegimePerformance(
                    regime_name=regime,
                    total_trades=total_trades,
                    total_pnl=total_pnl,
                    win_rate=win_rate,
                    avg_return=avg_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    gary_dpi=gary_dpi,
                    taleb_antifragility=taleb_antifragility,
                    alpha=alpha,
                    beta=beta,
                    information_ratio=information_ratio
                )

            # Save regime performance analysis
            for regime, performance in regime_performance.items():
                self._save_regime_performance(model_id, performance)

            return regime_performance

        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            return {}

    def _save_performance_analysis(self, model_id: str, metrics: PerformanceMetrics):
        """Save performance analysis to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_analysis (
                        analysis_date, period_start, period_end, model_id, metrics_json
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    metrics.period_start.isoformat(),
                    metrics.period_end.isoformat(),
                    model_id,
                    json.dumps(asdict(metrics), default=str)
                ))

        except Exception as e:
            self.logger.error(f"Error saving performance analysis: {e}")

    def _save_pnl_attribution(self, model_id: str, attribution: PnLAttribution,
                             start_date: datetime, end_date: datetime):
        """Save P&L attribution to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO pnl_attribution (
                        analysis_date, model_id, period_start, period_end, total_pnl,
                        model_contribution, timing_contribution, sizing_contribution,
                        market_contribution, execution_contribution, unexplained_contribution,
                        attribution_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_id,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    attribution.total_pnl,
                    attribution.model_contribution,
                    attribution.timing_contribution,
                    attribution.sizing_contribution,
                    attribution.market_contribution,
                    attribution.execution_contribution,
                    attribution.unexplained_contribution,
                    attribution.attribution_confidence
                ))

        except Exception as e:
            self.logger.error(f"Error saving P&L attribution: {e}")

    def _save_regime_performance(self, model_id: str, performance: RegimePerformance):
        """Save regime performance to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO regime_performance (
                        analysis_date, model_id, regime_name, performance_json
                    ) VALUES (?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_id,
                    performance.regime_name,
                    json.dumps(asdict(performance), default=str)
                ))

        except Exception as e:
            self.logger.error(f"Error saving regime performance: {e}")

    def get_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get latest performance analysis
            performance_metrics = self.analyze_performance(model_id)
            pnl_attribution = self.analyze_pnl_attribution(model_id)
            correlation_analysis = self.analyze_correlation(model_id)
            regime_performance = self.analyze_regime_performance(model_id)

            summary = {
                'model_id': model_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_metrics': asdict(performance_metrics) if performance_metrics else None,
                'pnl_attribution': asdict(pnl_attribution) if pnl_attribution else None,
                'correlation_analysis': asdict(correlation_analysis) if correlation_analysis else None,
                'regime_performance': {k: asdict(v) for k, v in regime_performance.items()},
                'summary_statistics': self._generate_summary_statistics(performance_metrics, regime_performance)
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}

    def _generate_summary_statistics(self, metrics: Optional[PerformanceMetrics],
                                   regime_performance: Dict[str, RegimePerformance]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not metrics:
            return {}

        # Best and worst performing regimes
        best_regime = None
        worst_regime = None
        best_gary_dpi = float('-inf')
        worst_gary_dpi = float('inf')

        for regime_name, performance in regime_performance.items():
            if performance.gary_dpi > best_gary_dpi:
                best_gary_dpi = performance.gary_dpi
                best_regime = regime_name
            if performance.gary_dpi < worst_gary_dpi:
                worst_gary_dpi = performance.gary_dpi
                worst_regime = regime_name

        return {
            'overall_gary_dpi': metrics.gary_dpi,
            'overall_taleb_antifragility': metrics.taleb_antifragility,
            'overall_sharpe_ratio': metrics.sharpe_ratio,
            'total_return': metrics.total_return,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'prediction_correlation': metrics.correlation_actual_predicted,
            'best_regime': best_regime,
            'worst_regime': worst_regime,
            'regime_count': len(regime_performance)
        }

if __name__ == "__main__":
    # Example usage
    analyzer = PerformanceAnalyzer(analysis_window_days=30)

    print("Performance analyzer initialized...")
    print("Use analyzer.record_trade_performance() to record trade results")
    print("Use analyzer.analyze_performance() to get comprehensive analysis")
    print("Use analyzer.get_performance_summary() to get complete summary")