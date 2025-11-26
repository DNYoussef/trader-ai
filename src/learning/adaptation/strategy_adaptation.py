"""
Strategy Adaptation Engine for Gary×Taleb Trading System

Adaptive strategy optimization that learns from market conditions,
adjusts parameters based on performance feedback, and optimizes
Gary DPI and Taleb antifragility in real-time.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import sqlite3
from pathlib import Path
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_id: str
    name: str
    description: str
    volatility_range: Tuple[float, float]
    trend_strength_range: Tuple[float, float]
    correlation_range: Tuple[float, float]
    characteristics: Dict[str, Any]

@dataclass
class StrategyParameters:
    """Strategy parameter configuration"""
    strategy_id: str
    regime_id: str
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    confidence_score: float
    adaptation_count: int

@dataclass
class AdaptationSignal:
    """Signal for strategy adaptation"""
    signal_type: str  # 'parameter_adjustment', 'regime_change', 'performance_degradation'
    urgency: str  # 'low', 'medium', 'high', 'critical'
    strategy_id: str
    current_regime: str
    suggested_parameters: Dict[str, float]
    expected_improvement: float
    confidence: float
    reasoning: str

@dataclass
class PerformanceContext:
    """Context for performance evaluation"""
    timestamp: datetime
    market_regime: str
    volatility: float
    trend_strength: float
    correlation: float
    volume: float
    spread: float
    gary_dpi: float
    taleb_antifragility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

class StrategyAdaptationEngine:
    """
    Adaptive strategy optimization engine that learns from market conditions
    and continuously optimizes trading strategies for maximum performance.
    """

    def __init__(self, adaptation_window_hours: int = 24, min_samples: int = 20):
        self.adaptation_window_hours = adaptation_window_hours
        self.min_samples = min_samples
        self.logger = self._setup_logging()

        # Database setup
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/strategy_adaptation.db")
        self._init_database()

        # Market regime detection
        self.market_regimes = self._initialize_market_regimes()
        self.current_regime = "neutral"

        # Strategy parameters by regime
        self.strategy_parameters: Dict[str, Dict[str, StrategyParameters]] = {}
        self.parameter_history: Dict[str, deque] = {}

        # Performance tracking
        self.performance_buffer = deque(maxlen=1000)
        self.adaptation_signals: List[AdaptationSignal] = []

        # Online learning components
        self.regime_classifier = None
        self.parameter_optimizer = None
        self.performance_predictor = None

        # Threading
        self.is_running = False
        self.adaptation_thread = None

        # Load existing configurations
        self._load_existing_parameters()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for strategy adaptation"""
        logger = logging.getLogger('StrategyAdaptation')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/strategy_adaptation.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for adaptation tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    regime_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    volatility_min REAL,
                    volatility_max REAL,
                    trend_min REAL,
                    trend_max REAL,
                    correlation_min REAL,
                    correlation_max REAL,
                    characteristics_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    regime_id TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    performance_metrics_json TEXT,
                    last_updated TEXT NOT NULL,
                    confidence_score REAL,
                    adaptation_count INTEGER DEFAULT 0
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    volatility REAL,
                    trend_strength REAL,
                    correlation REAL,
                    volume REAL,
                    spread REAL,
                    gary_dpi REAL,
                    taleb_antifragility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    current_regime TEXT NOT NULL,
                    suggested_parameters_json TEXT,
                    expected_improvement REAL,
                    confidence REAL,
                    reasoning TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    regime_id TEXT NOT NULL,
                    old_parameters_json TEXT,
                    new_parameters_json TEXT,
                    performance_before REAL,
                    performance_after REAL,
                    success BOOLEAN
                )
            ''')

    def _initialize_market_regimes(self) -> Dict[str, MarketRegime]:
        """Initialize predefined market regimes"""
        regimes = {
            'low_volatility': MarketRegime(
                regime_id='low_volatility',
                name='Low Volatility',
                description='Calm market conditions with low volatility',
                volatility_range=(0.0, 0.015),
                trend_strength_range=(-0.01, 0.01),
                correlation_range=(0.3, 0.7),
                characteristics={'risk_preference': 'moderate', 'mean_reversion': True}
            ),
            'high_volatility': MarketRegime(
                regime_id='high_volatility',
                name='High Volatility',
                description='Turbulent market with high volatility',
                volatility_range=(0.04, 1.0),
                trend_strength_range=(-0.1, 0.1),
                correlation_range=(0.1, 0.9),
                characteristics={'risk_preference': 'conservative', 'momentum': True}
            ),
            'trending_up': MarketRegime(
                regime_id='trending_up',
                name='Upward Trend',
                description='Strong upward trending market',
                volatility_range=(0.01, 0.05),
                trend_strength_range=(0.02, 1.0),
                correlation_range=(0.5, 0.9),
                characteristics={'risk_preference': 'aggressive', 'trend_following': True}
            ),
            'trending_down': MarketRegime(
                regime_id='trending_down',
                name='Downward Trend',
                description='Strong downward trending market',
                volatility_range=(0.01, 0.05),
                trend_strength_range=(-1.0, -0.02),
                correlation_range=(0.5, 0.9),
                characteristics={'risk_preference': 'defensive', 'contrarian': True}
            ),
            'sideways': MarketRegime(
                regime_id='sideways',
                name='Sideways Market',
                description='Range-bound market with no clear direction',
                volatility_range=(0.005, 0.03),
                trend_strength_range=(-0.015, 0.015),
                correlation_range=(0.2, 0.6),
                characteristics={'risk_preference': 'balanced', 'mean_reversion': True}
            )
        }

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            for regime in regimes.values():
                conn.execute('''
                    INSERT OR REPLACE INTO market_regimes (
                        regime_id, name, description, volatility_min, volatility_max,
                        trend_min, trend_max, correlation_min, correlation_max, characteristics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    regime.regime_id,
                    regime.name,
                    regime.description,
                    regime.volatility_range[0],
                    regime.volatility_range[1],
                    regime.trend_strength_range[0],
                    regime.trend_strength_range[1],
                    regime.correlation_range[0],
                    regime.correlation_range[1],
                    json.dumps(regime.characteristics)
                ))

        return regimes

    def _load_existing_parameters(self):
        """Load existing strategy parameters from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT strategy_id, regime_id, parameters_json, performance_metrics_json,
                           last_updated, confidence_score, adaptation_count
                    FROM strategy_parameters
                    WHERE id IN (
                        SELECT MAX(id) FROM strategy_parameters
                        GROUP BY strategy_id, regime_id
                    )
                '''
                cursor = conn.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    strategy_id = row[0]
                    regime_id = row[1]
                    parameters = json.loads(row[2])
                    performance_metrics = json.loads(row[3]) if row[3] else {}
                    last_updated = datetime.fromisoformat(row[4])
                    confidence_score = row[5] or 0.5
                    adaptation_count = row[6] or 0

                    if strategy_id not in self.strategy_parameters:
                        self.strategy_parameters[strategy_id] = {}

                    self.strategy_parameters[strategy_id][regime_id] = StrategyParameters(
                        strategy_id=strategy_id,
                        regime_id=regime_id,
                        parameters=parameters,
                        performance_metrics=performance_metrics,
                        last_updated=last_updated,
                        confidence_score=confidence_score,
                        adaptation_count=adaptation_count
                    )

                self.logger.info(f"Loaded parameters for {len(self.strategy_parameters)} strategies")

        except Exception as e:
            self.logger.error(f"Error loading existing parameters: {e}")

    def start_adaptation_engine(self):
        """Start the strategy adaptation engine"""
        if self.is_running:
            self.logger.warning("Strategy adaptation engine already running")
            return

        self.is_running = True
        self.logger.info("Starting strategy adaptation engine")

        # Start background adaptation
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()

        self.logger.info("Strategy adaptation engine started")

    def stop_adaptation_engine(self):
        """Stop the strategy adaptation engine"""
        self.is_running = False
        self.logger.info("Strategy adaptation engine stopped")

    def _adaptation_loop(self):
        """Main adaptation loop"""
        while self.is_running:
            try:
                # Detect current market regime
                self._update_market_regime()

                # Check for adaptation signals
                self._generate_adaptation_signals()

                # Process adaptation signals
                self._process_adaptation_signals()

                # Update online learning models
                self._update_learning_models()

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {e}")
                time.sleep(60)

    def record_performance(self, strategy_id: str, market_data: Dict[str, float],
                          performance_metrics: Dict[str, float]) -> bool:
        """
        Record performance data for strategy adaptation

        Args:
            strategy_id: Strategy identifier
            market_data: Current market conditions
            performance_metrics: Performance metrics achieved

        Returns:
            success: Whether recording was successful
        """
        try:
            # Detect current market regime
            current_regime = self.detect_market_regime(market_data)

            # Create performance context
            context = PerformanceContext(
                timestamp=datetime.now(),
                market_regime=current_regime,
                volatility=market_data.get('volatility', 0.02),
                trend_strength=market_data.get('trend_strength', 0.0),
                correlation=market_data.get('correlation', 0.5),
                volume=market_data.get('volume', 1.0),
                spread=market_data.get('spread', 0.001),
                gary_dpi=performance_metrics.get('gary_dpi', 0.0),
                taleb_antifragility=performance_metrics.get('taleb_antifragility', 0.0),
                sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                max_drawdown=performance_metrics.get('max_drawdown', 0.0),
                win_rate=performance_metrics.get('win_rate', 0.5)
            )

            # Add to performance buffer
            self.performance_buffer.append(context)

            # Save to database
            self._save_performance_context(context)

            # Check if adaptation is needed
            self._check_adaptation_trigger(strategy_id, context)

            return True

        except Exception as e:
            self.logger.error(f"Error recording performance: {e}")
            return False

    def detect_market_regime(self, market_data: Dict[str, float]) -> str:
        """
        Detect current market regime based on market conditions

        Args:
            market_data: Current market conditions

        Returns:
            regime_id: Detected market regime
        """
        try:
            volatility = market_data.get('volatility', 0.02)
            trend_strength = market_data.get('trend_strength', 0.0)
            correlation = market_data.get('correlation', 0.5)

            # Rule-based regime detection
            for regime_id, regime in self.market_regimes.items():
                if (regime.volatility_range[0] <= volatility <= regime.volatility_range[1] and
                    regime.trend_strength_range[0] <= trend_strength <= regime.trend_strength_range[1] and
                    regime.correlation_range[0] <= correlation <= regime.correlation_range[1]):
                    return regime_id

            # Default to sideways if no match
            return 'sideways'

        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'sideways'

    def _update_market_regime(self):
        """Update current market regime"""
        try:
            if len(self.performance_buffer) == 0:
                return

            # Get recent market data
            recent_context = list(self.performance_buffer)[-10:]  # Last 10 observations

            if recent_context:
                # Calculate average market conditions
                avg_volatility = np.mean([c.volatility for c in recent_context])
                avg_trend = np.mean([c.trend_strength for c in recent_context])
                avg_correlation = np.mean([c.correlation for c in recent_context])

                market_data = {
                    'volatility': avg_volatility,
                    'trend_strength': avg_trend,
                    'correlation': avg_correlation
                }

                new_regime = self.detect_market_regime(market_data)

                if new_regime != self.current_regime:
                    self.logger.info(f"Market regime changed: {self.current_regime} -> {new_regime}")
                    self.current_regime = new_regime

                    # Generate regime change signal
                    self._generate_regime_change_signal(new_regime)

        except Exception as e:
            self.logger.error(f"Error updating market regime: {e}")

    def _generate_regime_change_signal(self, new_regime: str):
        """Generate adaptation signal for regime change"""
        for strategy_id in self.strategy_parameters.keys():
            if new_regime not in self.strategy_parameters[strategy_id]:
                # No parameters for this regime, use default or interpolate
                suggested_parameters = self._interpolate_parameters(strategy_id, new_regime)
            else:
                # Use existing parameters for this regime
                suggested_parameters = self.strategy_parameters[strategy_id][new_regime].parameters

            signal = AdaptationSignal(
                signal_type='regime_change',
                urgency='high',
                strategy_id=strategy_id,
                current_regime=new_regime,
                suggested_parameters=suggested_parameters,
                expected_improvement=0.05,  # Placeholder
                confidence=0.8,
                reasoning=f"Market regime changed to {new_regime}"
            )

            self.adaptation_signals.append(signal)
            self._save_adaptation_signal(signal)

    def _interpolate_parameters(self, strategy_id: str, target_regime: str) -> Dict[str, float]:
        """Interpolate parameters for a regime without existing configuration"""
        try:
            if strategy_id not in self.strategy_parameters:
                return self._get_default_parameters(strategy_id)

            # Get parameters from existing regimes
            existing_params = list(self.strategy_parameters[strategy_id].values())

            if not existing_params:
                return self._get_default_parameters(strategy_id)

            # Simple interpolation - average existing parameters
            param_names = existing_params[0].parameters.keys()
            interpolated = {}

            for param_name in param_names:
                values = [p.parameters[param_name] for p in existing_params]
                interpolated[param_name] = np.mean(values)

            # Adjust based on regime characteristics
            regime_characteristics = self.market_regimes[target_regime].characteristics

            if regime_characteristics.get('risk_preference') == 'conservative':
                interpolated['position_size'] = interpolated.get('position_size', 0.02) * 0.7
                interpolated['stop_loss'] = interpolated.get('stop_loss', 0.02) * 0.8

            elif regime_characteristics.get('risk_preference') == 'aggressive':
                interpolated['position_size'] = interpolated.get('position_size', 0.02) * 1.3
                interpolated['take_profit'] = interpolated.get('take_profit', 0.03) * 1.2

            return interpolated

        except Exception as e:
            self.logger.error(f"Error interpolating parameters: {e}")
            return self._get_default_parameters(strategy_id)

    def _get_default_parameters(self, strategy_id: str) -> Dict[str, float]:
        """Get default parameters for a strategy"""
        defaults = {
            'gary_dpi_momentum': {
                'position_size': 0.02,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'momentum_threshold': 0.01,
                'volatility_factor': 1.0,
                'correlation_threshold': 0.5
            },
            'taleb_antifragile': {
                'position_size': 0.015,
                'stop_loss': 0.025,
                'take_profit': 0.04,
                'stress_multiplier': 1.5,
                'antifragility_threshold': 0.02,
                'volatility_boost': 1.2
            },
            'risk_parity': {
                'position_size': 0.01,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'risk_budget': 0.1,
                'rebalance_threshold': 0.05,
                'diversification_factor': 0.8
            }
        }

        return defaults.get(strategy_id, {
            'position_size': 0.02,
            'stop_loss': 0.02,
            'take_profit': 0.03
        })

    def _generate_adaptation_signals(self):
        """Generate adaptation signals based on performance"""
        try:
            if len(self.performance_buffer) < self.min_samples:
                return

            # Get recent performance data
            recent_performance = list(self.performance_buffer)[-self.min_samples:]

            # Group by strategy (assuming strategy info is available)
            # For now, analyze overall performance

            # Check for performance degradation
            gary_dpi_values = [p.gary_dpi for p in recent_performance]
            antifragility_values = [p.taleb_antifragility for p in recent_performance]

            # Calculate trends
            gary_dpi_trend = self._calculate_trend(gary_dpi_values)
            antifragility_trend = self._calculate_trend(antifragility_values)

            # Generate signals for performance degradation
            if gary_dpi_trend < -0.01:  # Negative trend
                self._generate_performance_degradation_signal('gary_dpi', gary_dpi_trend)

            if antifragility_trend < -0.01:
                self._generate_performance_degradation_signal('taleb_antifragility', antifragility_trend)

            # Check for parameter optimization opportunities
            self._check_optimization_opportunities(recent_performance)

        except Exception as e:
            self.logger.error(f"Error generating adaptation signals: {e}")

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _generate_performance_degradation_signal(self, metric_name: str, trend: float):
        """Generate signal for performance degradation"""
        urgency = 'critical' if trend < -0.05 else 'high'

        # For now, generate signal for all strategies
        for strategy_id in self.strategy_parameters.keys():
            signal = AdaptationSignal(
                signal_type='performance_degradation',
                urgency=urgency,
                strategy_id=strategy_id,
                current_regime=self.current_regime,
                suggested_parameters=self._optimize_parameters_for_performance(strategy_id, metric_name),
                expected_improvement=abs(trend),
                confidence=0.7,
                reasoning=f"{metric_name} showing negative trend: {trend:.4f}"
            )

            self.adaptation_signals.append(signal)
            self._save_adaptation_signal(signal)

    def _optimize_parameters_for_performance(self, strategy_id: str, metric_name: str) -> Dict[str, float]:
        """Optimize parameters to improve specific performance metric"""
        try:
            if strategy_id not in self.strategy_parameters:
                return self._get_default_parameters(strategy_id)

            current_regime = self.current_regime
            if current_regime not in self.strategy_parameters[strategy_id]:
                return self._get_default_parameters(strategy_id)

            current_params = self.strategy_parameters[strategy_id][current_regime].parameters

            # Simple parameter adjustment based on metric
            optimized_params = current_params.copy()

            if metric_name == 'gary_dpi':
                # Adjust parameters to improve Gary DPI
                optimized_params['position_size'] = max(0.005, current_params.get('position_size', 0.02) * 0.9)
                optimized_params['stop_loss'] = max(0.01, current_params.get('stop_loss', 0.02) * 0.95)

            elif metric_name == 'taleb_antifragility':
                # Adjust parameters to improve antifragility
                optimized_params['volatility_factor'] = current_params.get('volatility_factor', 1.0) * 1.1
                optimized_params['stress_multiplier'] = current_params.get('stress_multiplier', 1.0) * 1.2

            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return self._get_default_parameters(strategy_id)

    def _check_optimization_opportunities(self, recent_performance: List[PerformanceContext]):
        """Check for parameter optimization opportunities"""
        try:
            # Analyze performance variance
            gary_dpi_variance = np.var([p.gary_dpi for p in recent_performance])
            sharpe_variance = np.var([p.sharpe_ratio for p in recent_performance])

            # High variance might indicate suboptimal parameters
            if gary_dpi_variance > 0.01 or sharpe_variance > 0.05:
                for strategy_id in self.strategy_parameters.keys():
                    signal = AdaptationSignal(
                        signal_type='parameter_adjustment',
                        urgency='medium',
                        strategy_id=strategy_id,
                        current_regime=self.current_regime,
                        suggested_parameters=self._suggest_variance_reduction_parameters(strategy_id),
                        expected_improvement=0.02,
                        confidence=0.6,
                        reasoning="High performance variance detected"
                    )

                    self.adaptation_signals.append(signal)
                    self._save_adaptation_signal(signal)

        except Exception as e:
            self.logger.error(f"Error checking optimization opportunities: {e}")

    def _suggest_variance_reduction_parameters(self, strategy_id: str) -> Dict[str, float]:
        """Suggest parameters to reduce performance variance"""
        try:
            if (strategy_id not in self.strategy_parameters or
                self.current_regime not in self.strategy_parameters[strategy_id]):
                return self._get_default_parameters(strategy_id)

            current_params = self.strategy_parameters[strategy_id][self.current_regime].parameters
            adjusted_params = current_params.copy()

            # Reduce position size to decrease variance
            adjusted_params['position_size'] = current_params.get('position_size', 0.02) * 0.8

            # Tighten stop loss
            adjusted_params['stop_loss'] = current_params.get('stop_loss', 0.02) * 0.9

            # Adjust take profit
            adjusted_params['take_profit'] = current_params.get('take_profit', 0.03) * 0.95

            return adjusted_params

        except Exception as e:
            self.logger.error(f"Error suggesting variance reduction parameters: {e}")
            return self._get_default_parameters(strategy_id)

    def _process_adaptation_signals(self):
        """Process pending adaptation signals"""
        try:
            # Sort signals by urgency and confidence
            sorted_signals = sorted(
                self.adaptation_signals,
                key=lambda s: (s.urgency == 'critical', s.urgency == 'high', s.confidence),
                reverse=True
            )

            for signal in sorted_signals[:5]:  # Process top 5 signals
                self._apply_adaptation_signal(signal)

            # Clear processed signals
            self.adaptation_signals.clear()

        except Exception as e:
            self.logger.error(f"Error processing adaptation signals: {e}")

    def _apply_adaptation_signal(self, signal: AdaptationSignal):
        """Apply an adaptation signal"""
        try:
            strategy_id = signal.strategy_id
            regime_id = signal.current_regime

            # Get current parameters
            old_parameters = None
            if (strategy_id in self.strategy_parameters and
                regime_id in self.strategy_parameters[strategy_id]):
                old_parameters = self.strategy_parameters[strategy_id][regime_id].parameters.copy()

            # Apply new parameters
            new_parameters = signal.suggested_parameters

            # Update strategy parameters
            if strategy_id not in self.strategy_parameters:
                self.strategy_parameters[strategy_id] = {}

            self.strategy_parameters[strategy_id][regime_id] = StrategyParameters(
                strategy_id=strategy_id,
                regime_id=regime_id,
                parameters=new_parameters,
                performance_metrics={},
                last_updated=datetime.now(),
                confidence_score=signal.confidence,
                adaptation_count=(self.strategy_parameters[strategy_id][regime_id].adaptation_count + 1
                                if regime_id in self.strategy_parameters[strategy_id] else 1)
            )

            # Save to database
            self._save_strategy_parameters(self.strategy_parameters[strategy_id][regime_id])

            # Log adaptation
            self._log_adaptation(strategy_id, regime_id, old_parameters, new_parameters, signal)

            self.logger.info(f"Applied adaptation for {strategy_id} in {regime_id}: {signal.reasoning}")

        except Exception as e:
            self.logger.error(f"Error applying adaptation signal: {e}")

    def _check_adaptation_trigger(self, strategy_id: str, context: PerformanceContext):
        """Check if adaptation should be triggered for a strategy"""
        try:
            # Simple trigger based on performance threshold
            if context.gary_dpi < -0.1 or context.taleb_antifragility < -0.2:
                # Generate immediate adaptation signal
                signal = AdaptationSignal(
                    signal_type='performance_degradation',
                    urgency='high',
                    strategy_id=strategy_id,
                    current_regime=context.market_regime,
                    suggested_parameters=self._emergency_parameter_adjustment(strategy_id),
                    expected_improvement=0.1,
                    confidence=0.8,
                    reasoning=f"Emergency adaptation: Gary DPI={context.gary_dpi:.4f}, Antifragility={context.taleb_antifragility:.4f}"
                )

                self.adaptation_signals.append(signal)
                self._save_adaptation_signal(signal)

        except Exception as e:
            self.logger.error(f"Error checking adaptation trigger: {e}")

    def _emergency_parameter_adjustment(self, strategy_id: str) -> Dict[str, float]:
        """Emergency parameter adjustment for poor performance"""
        base_params = self._get_default_parameters(strategy_id)

        # Conservative adjustments
        emergency_params = {
            'position_size': min(base_params.get('position_size', 0.02), 0.01),
            'stop_loss': min(base_params.get('stop_loss', 0.02), 0.015),
            'take_profit': max(base_params.get('take_profit', 0.03), 0.02),
        }

        # Merge with other parameters
        emergency_params.update({k: v for k, v in base_params.items() if k not in emergency_params})

        return emergency_params

    def _update_learning_models(self):
        """Update online learning models"""
        try:
            if len(self.performance_buffer) < 50:  # Need sufficient data
                return

            # Prepare training data
            X, y_gary, y_antifragility = self._prepare_training_data()

            if len(X) < 10:
                return

            # Update performance predictor
            if self.performance_predictor is None:
                self.performance_predictor = RandomForestRegressor(n_estimators=50, random_state=42)

            # Train on Gary DPI prediction
            self.performance_predictor.fit(X, y_gary)

            self.logger.info("Updated online learning models")

        except Exception as e:
            self.logger.error(f"Error updating learning models: {e}")

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for online learning"""
        recent_data = list(self.performance_buffer)[-100:]  # Last 100 observations

        features = []
        gary_targets = []
        antifragility_targets = []

        for context in recent_data:
            feature_vector = [
                context.volatility,
                context.trend_strength,
                context.correlation,
                context.volume,
                context.spread
            ]

            features.append(feature_vector)
            gary_targets.append(context.gary_dpi)
            antifragility_targets.append(context.taleb_antifragility)

        return np.array(features), np.array(gary_targets), np.array(antifragility_targets)

    def _save_performance_context(self, context: PerformanceContext):
        """Save performance context to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_context (
                    timestamp, market_regime, volatility, trend_strength, correlation,
                    volume, spread, gary_dpi, taleb_antifragility, sharpe_ratio,
                    max_drawdown, win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                context.timestamp.isoformat(),
                context.market_regime,
                context.volatility,
                context.trend_strength,
                context.correlation,
                context.volume,
                context.spread,
                context.gary_dpi,
                context.taleb_antifragility,
                context.sharpe_ratio,
                context.max_drawdown,
                context.win_rate
            ))

    def _save_strategy_parameters(self, params: StrategyParameters):
        """Save strategy parameters to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO strategy_parameters (
                    strategy_id, regime_id, parameters_json, performance_metrics_json,
                    last_updated, confidence_score, adaptation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                params.strategy_id,
                params.regime_id,
                json.dumps(params.parameters),
                json.dumps(params.performance_metrics),
                params.last_updated.isoformat(),
                params.confidence_score,
                params.adaptation_count
            ))

    def _save_adaptation_signal(self, signal: AdaptationSignal):
        """Save adaptation signal to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO adaptation_signals (
                    timestamp, signal_type, urgency, strategy_id, current_regime,
                    suggested_parameters_json, expected_improvement, confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                signal.signal_type,
                signal.urgency,
                signal.strategy_id,
                signal.current_regime,
                json.dumps(signal.suggested_parameters),
                signal.expected_improvement,
                signal.confidence,
                signal.reasoning
            ))

    def _log_adaptation(self, strategy_id: str, regime_id: str,
                       old_parameters: Optional[Dict[str, float]],
                       new_parameters: Dict[str, float],
                       signal: AdaptationSignal):
        """Log adaptation history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO adaptation_history (
                    timestamp, strategy_id, regime_id, old_parameters_json,
                    new_parameters_json, performance_before, performance_after, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                strategy_id,
                regime_id,
                json.dumps(old_parameters) if old_parameters else None,
                json.dumps(new_parameters),
                0.0,  # Will be updated later
                0.0,  # Will be updated later
                True  # Assume success for now
            ))

    def get_current_parameters(self, strategy_id: str, regime_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Get current parameters for a strategy"""
        try:
            regime = regime_id or self.current_regime

            if (strategy_id in self.strategy_parameters and
                regime in self.strategy_parameters[strategy_id]):
                return self.strategy_parameters[strategy_id][regime].parameters.copy()

            return self._get_default_parameters(strategy_id)

        except Exception as e:
            self.logger.error(f"Error getting current parameters: {e}")
            return None

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation system status"""
        try:
            summary = {
                'current_regime': self.current_regime,
                'active_strategies': len(self.strategy_parameters),
                'total_adaptations': sum(
                    sum(params.adaptation_count for params in strategy_params.values())
                    for strategy_params in self.strategy_parameters.values()
                ),
                'pending_signals': len(self.adaptation_signals),
                'recent_performance': len(self.performance_buffer),
                'is_running': self.is_running,
                'strategies_by_regime': {}
            }

            # Add strategy details by regime
            for strategy_id, regimes in self.strategy_parameters.items():
                for regime_id, params in regimes.items():
                    if regime_id not in summary['strategies_by_regime']:
                        summary['strategies_by_regime'][regime_id] = []

                    summary['strategies_by_regime'][regime_id].append({
                        'strategy_id': strategy_id,
                        'last_updated': params.last_updated.isoformat(),
                        'confidence_score': params.confidence_score,
                        'adaptation_count': params.adaptation_count
                    })

            return summary

        except Exception as e:
            self.logger.error(f"Error getting adaptation summary: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    adaptation_engine = StrategyAdaptationEngine(adaptation_window_hours=12, min_samples=10)
    adaptation_engine.start_adaptation_engine()

    print("Strategy adaptation engine started...")
    print("Use adaptation_engine.record_performance() to record trading performance")
    print("Use adaptation_engine.get_current_parameters() to get optimized parameters")
    print("Use adaptation_engine.get_adaptation_summary() to get system status")