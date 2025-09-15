"""
Continuous Learning Pipeline for Gary×Taleb Trading System

Automated model retraining pipeline that learns from actual trading performance
with emphasis on Gary DPI optimization and Taleb antifragility principles.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import pickle
import sqlite3
from pathlib import Path
import threading
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

@dataclass
class LearningConfig:
    """Configuration for continuous learning pipeline"""
    retrain_frequency_hours: int = 24
    min_samples_for_retrain: int = 100
    performance_window_days: int = 30
    model_comparison_window_days: int = 7
    max_model_versions: int = 10
    auto_rollback_threshold: float = -0.05  # 5% performance degradation
    data_freshness_hours: int = 6
    feature_importance_threshold: float = 0.01
    cross_validation_folds: int = 5
    ensemble_voting_threshold: float = 0.7

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    timestamp: datetime
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    gary_dpi: float
    taleb_antifragility: float
    actual_returns: List[float]
    predicted_returns: List[float]
    trades_count: int
    avg_holding_period: float

@dataclass
class RetrainingResult:
    """Result of model retraining"""
    success: bool
    model_id: str
    performance: ModelPerformance
    training_time: float
    feature_importance: Dict[str, float]
    validation_scores: List[float]
    improvement_over_previous: float
    error_message: Optional[str] = None

class ContinuousLearner:
    """
    Continuous learning pipeline that automatically retrains models
    based on actual trading performance and market conditions.
    """

    def __init__(self, config: LearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.db_path = Path("C:/Users/17175/Desktop/trader-ai/data/learning_pipeline.db")
        self.models_path = Path("C:/Users/17175/Desktop/trader-ai/models/continuous")
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # Current active models
        self.active_models: Dict[str, Any] = {}
        self.model_performances: Dict[str, List[ModelPerformance]] = {}

        # Threading for background tasks
        self.is_running = False
        self.retrain_thread = None
        self.monitor_thread = None

        # Load existing models
        self._load_existing_models()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for continuous learner"""
        logger = logging.getLogger('ContinuousLearner')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler('C:/Users/17175/Desktop/trader-ai/logs/continuous_learner.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_database(self):
        """Initialize SQLite database for tracking learning pipeline"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    mse REAL,
                    mae REAL,
                    r2 REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    gary_dpi REAL,
                    taleb_antifragility REAL,
                    trades_count INTEGER,
                    avg_holding_period REAL,
                    performance_data TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS retraining_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN,
                    training_time REAL,
                    improvement REAL,
                    feature_importance TEXT,
                    validation_scores TEXT,
                    error_message TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version INTEGER,
                    timestamp TEXT NOT NULL,
                    is_active BOOLEAN,
                    model_path TEXT,
                    config_data TEXT
                )
            ''')

    def _load_existing_models(self):
        """Load existing trained models from disk"""
        try:
            for model_file in self.models_path.glob("*.pkl"):
                model_id = model_file.stem
                model = joblib.load(model_file)
                self.active_models[model_id] = model
                self.logger.info(f"Loaded existing model: {model_id}")
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")

    def start_continuous_learning(self):
        """Start the continuous learning pipeline"""
        if self.is_running:
            self.logger.warning("Continuous learning already running")
            return

        self.is_running = True
        self.logger.info("Starting continuous learning pipeline")

        # Schedule periodic retraining
        schedule.every(self.config.retrain_frequency_hours).hours.do(self._scheduled_retrain)

        # Start background threads
        self.retrain_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)

        self.retrain_thread.start()
        self.monitor_thread.start()

        self.logger.info("Continuous learning pipeline started")

    def stop_continuous_learning(self):
        """Stop the continuous learning pipeline"""
        self.is_running = False
        schedule.clear()
        self.logger.info("Continuous learning pipeline stopped")

    def _run_scheduler(self):
        """Run the scheduled tasks"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _monitor_performance(self):
        """Monitor model performance continuously"""
        while self.is_running:
            try:
                self._check_model_performance()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")

    def _scheduled_retrain(self):
        """Scheduled retraining task"""
        try:
            self.logger.info("Starting scheduled retraining")

            # Check if we have enough new data
            if not self._has_sufficient_data():
                self.logger.info("Insufficient data for retraining, skipping")
                return

            # Perform retraining for all active models
            for model_id in self.active_models.keys():
                result = self.retrain_model(model_id)
                if result.success:
                    self.logger.info(f"Successfully retrained model {model_id}")
                else:
                    self.logger.error(f"Failed to retrain model {model_id}: {result.error_message}")

        except Exception as e:
            self.logger.error(f"Error in scheduled retraining: {e}")

    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient new data for retraining"""
        try:
            # Check for new trading data
            conn = sqlite3.connect("C:/Users/17175/Desktop/trader-ai/data/trading_data.db")

            cutoff_time = datetime.now() - timedelta(hours=self.config.data_freshness_hours)

            query = """
                SELECT COUNT(*) FROM trades
                WHERE timestamp > ? AND status = 'completed'
            """

            cursor = conn.execute(query, (cutoff_time.isoformat(),))
            new_trades = cursor.fetchone()[0]
            conn.close()

            return new_trades >= self.config.min_samples_for_retrain

        except Exception as e:
            self.logger.error(f"Error checking data sufficiency: {e}")
            return False

    def retrain_model(self, model_id: str) -> RetrainingResult:
        """
        Retrain a specific model with latest data

        Args:
            model_id: ID of the model to retrain

        Returns:
            RetrainingResult with training outcome
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting retraining for model {model_id}")

            # Load training data
            X, y = self._load_training_data()

            if len(X) < self.config.min_samples_for_retrain:
                return RetrainingResult(
                    success=False,
                    model_id=model_id,
                    performance=None,
                    training_time=0,
                    feature_importance={},
                    validation_scores=[],
                    improvement_over_previous=0,
                    error_message="Insufficient training data"
                )

            # Create new model instance
            model = self._create_model(model_id)

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')

            # Train on full dataset
            model.fit(X_scaled, y)

            # Calculate feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                feature_importance = dict(zip(feature_names, model.feature_importances_))

            # Evaluate performance
            y_pred = model.predict(X_scaled)
            performance = self._evaluate_model_performance(model_id, y, y_pred)

            # Calculate improvement over previous version
            improvement = self._calculate_improvement(model_id, performance)

            # Save model if improvement is significant
            if improvement > 0 or len(self.active_models) == 0:
                self._save_model(model_id, model, scaler)
                self.active_models[model_id] = {'model': model, 'scaler': scaler}

                # Update model performance tracking
                if model_id not in self.model_performances:
                    self.model_performances[model_id] = []
                self.model_performances[model_id].append(performance)

                # Save to database
                self._save_performance_to_db(performance)

            training_time = time.time() - start_time

            # Save retraining history
            self._save_retraining_history(
                model_id, True, training_time, improvement,
                feature_importance, cv_scores.tolist(), None
            )

            self.logger.info(f"Model {model_id} retrained successfully in {training_time:.2f}s")

            return RetrainingResult(
                success=True,
                model_id=model_id,
                performance=performance,
                training_time=training_time,
                feature_importance=feature_importance,
                validation_scores=cv_scores.tolist(),
                improvement_over_previous=improvement
            )

        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)

            self._save_retraining_history(
                model_id, False, training_time, 0, {}, [], error_msg
            )

            self.logger.error(f"Error retraining model {model_id}: {error_msg}")

            return RetrainingResult(
                success=False,
                model_id=model_id,
                performance=None,
                training_time=training_time,
                feature_importance={},
                validation_scores=[],
                improvement_over_previous=0,
                error_message=error_msg
            )

    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from database"""
        conn = sqlite3.connect("C:/Users/17175/Desktop/trader-ai/data/trading_data.db")

        # Load features and targets for completed trades
        query = """
            SELECT features, actual_return
            FROM trades
            WHERE status = 'completed' AND actual_return IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10000
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Parse features (assuming JSON format)
        features_list = []
        targets = []

        for _, row in df.iterrows():
            try:
                features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
                if isinstance(features, dict):
                    features_list.append(list(features.values()))
                else:
                    features_list.append(features)
                targets.append(row['actual_return'])
            except:
                continue

        X = np.array(features_list)
        y = np.array(targets)

        return X, y

    def _create_model(self, model_id: str):
        """Create a new model instance based on model type"""
        model_configs = {
            'gary_dpi': RandomForestRegressor(n_estimators=100, random_state=42),
            'taleb_antifragile': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ensemble': RandomForestRegressor(n_estimators=200, random_state=42),
            'risk_adjusted': Ridge(alpha=1.0)
        }

        return model_configs.get(model_id, RandomForestRegressor(n_estimators=100, random_state=42))

    def _evaluate_model_performance(self, model_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelPerformance:
        """Evaluate model performance with trading-specific metrics"""

        # Basic ML metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Trading metrics
        returns = y_true
        predicted_returns = y_pred

        # Sharpe ratio
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Gary DPI calculation
        gary_dpi = self._calculate_gary_dpi(returns)

        # Taleb antifragility score
        taleb_antifragility = self._calculate_taleb_antifragility(returns)

        return ModelPerformance(
            model_id=model_id,
            timestamp=datetime.now(),
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            gary_dpi=gary_dpi,
            taleb_antifragility=taleb_antifragility,
            actual_returns=returns.tolist(),
            predicted_returns=predicted_returns.tolist(),
            trades_count=len(returns),
            avg_holding_period=1.0  # Placeholder
        )

    def _calculate_gary_dpi(self, returns: np.ndarray) -> float:
        """Calculate Gary's DPI (Dynamic Performance Index)"""
        if len(returns) == 0:
            return 0.0

        # Gary's DPI: (Average Return * Win Rate) / (Max Drawdown + Volatility)
        avg_return = np.mean(returns)
        win_rate = np.sum(returns > 0) / len(returns)
        volatility = np.std(returns)

        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        denominator = max_drawdown + volatility
        gary_dpi = (avg_return * win_rate) / denominator if denominator > 0 else 0

        return gary_dpi

    def _calculate_taleb_antifragility(self, returns: np.ndarray) -> float:
        """Calculate Taleb's antifragility score"""
        if len(returns) == 0:
            return 0.0

        # Antifragility: Performance improvement during stress
        # Higher score when system benefits from volatility
        volatility = np.std(returns)
        mean_return = np.mean(returns)

        # Identify high volatility periods (top 25%)
        volatility_threshold = np.percentile(np.abs(returns - mean_return), 75)
        stress_periods = np.abs(returns - mean_return) > volatility_threshold

        if np.sum(stress_periods) == 0:
            return 0.0

        stress_returns = returns[stress_periods]
        normal_returns = returns[~stress_periods]

        stress_performance = np.mean(stress_returns) if len(stress_returns) > 0 else 0
        normal_performance = np.mean(normal_returns) if len(normal_returns) > 0 else 0

        # Antifragility score: positive when stress performance exceeds normal
        antifragility = (stress_performance - normal_performance) / (volatility + 1e-8)

        return antifragility

    def _calculate_improvement(self, model_id: str, current_performance: ModelPerformance) -> float:
        """Calculate improvement over previous model version"""
        if model_id not in self.model_performances or len(self.model_performances[model_id]) == 0:
            return 1.0  # First model, consider as improvement

        previous_performance = self.model_performances[model_id][-1]

        # Weighted improvement score
        weights = {
            'gary_dpi': 0.3,
            'taleb_antifragility': 0.25,
            'sharpe_ratio': 0.2,
            'r2': 0.15,
            'profit_factor': 0.1
        }

        improvements = {}
        improvements['gary_dpi'] = (current_performance.gary_dpi - previous_performance.gary_dpi) / (abs(previous_performance.gary_dpi) + 1e-8)
        improvements['taleb_antifragility'] = (current_performance.taleb_antifragility - previous_performance.taleb_antifragility) / (abs(previous_performance.taleb_antifragility) + 1e-8)
        improvements['sharpe_ratio'] = (current_performance.sharpe_ratio - previous_performance.sharpe_ratio) / (abs(previous_performance.sharpe_ratio) + 1e-8)
        improvements['r2'] = (current_performance.r2 - previous_performance.r2) / (abs(previous_performance.r2) + 1e-8)
        improvements['profit_factor'] = (current_performance.profit_factor - previous_performance.profit_factor) / (abs(previous_performance.profit_factor) + 1e-8)

        weighted_improvement = sum(weights[metric] * improvements[metric] for metric in weights)

        return weighted_improvement

    def _save_model(self, model_id: str, model, scaler):
        """Save trained model and scaler to disk"""
        model_data = {
            'model': model,
            'scaler': scaler,
            'timestamp': datetime.now(),
            'model_id': model_id
        }

        model_path = self.models_path / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)

        # Save version info to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_versions (model_id, version, timestamp, is_active, model_path, config_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                1,  # Version numbering logic can be enhanced
                datetime.now().isoformat(),
                True,
                str(model_path),
                json.dumps(asdict(self.config))
            ))

    def _save_performance_to_db(self, performance: ModelPerformance):
        """Save model performance to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_performances (
                    model_id, timestamp, mse, mae, r2, sharpe_ratio, max_drawdown,
                    win_rate, profit_factor, gary_dpi, taleb_antifragility,
                    trades_count, avg_holding_period, performance_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance.model_id,
                performance.timestamp.isoformat(),
                performance.mse,
                performance.mae,
                performance.r2,
                performance.sharpe_ratio,
                performance.max_drawdown,
                performance.win_rate,
                performance.profit_factor,
                performance.gary_dpi,
                performance.taleb_antifragility,
                performance.trades_count,
                performance.avg_holding_period,
                json.dumps(asdict(performance))
            ))

    def _save_retraining_history(self, model_id: str, success: bool, training_time: float,
                               improvement: float, feature_importance: Dict, validation_scores: List,
                               error_message: Optional[str]):
        """Save retraining history to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO retraining_history (
                    model_id, timestamp, success, training_time, improvement,
                    feature_importance, validation_scores, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                datetime.now().isoformat(),
                success,
                training_time,
                improvement,
                json.dumps(feature_importance),
                json.dumps(validation_scores),
                error_message
            ))

    def _check_model_performance(self):
        """Check current model performance and trigger rollback if needed"""
        try:
            for model_id in self.active_models.keys():
                recent_performance = self._get_recent_performance(model_id)

                if recent_performance and recent_performance.gary_dpi < self.config.auto_rollback_threshold:
                    self.logger.warning(f"Model {model_id} performance degraded, considering rollback")
                    self._trigger_automatic_rollback(model_id)

        except Exception as e:
            self.logger.error(f"Error checking model performance: {e}")

    def _get_recent_performance(self, model_id: str) -> Optional[ModelPerformance]:
        """Get recent performance for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM model_performances
                    WHERE model_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (model_id,))
                row = cursor.fetchone()

                if row:
                    # Convert row to ModelPerformance object
                    performance_data = json.loads(row[13])  # performance_data column
                    return ModelPerformance(**performance_data)

        except Exception as e:
            self.logger.error(f"Error getting recent performance: {e}")

        return None

    def _trigger_automatic_rollback(self, model_id: str):
        """Trigger automatic rollback to previous model version"""
        try:
            self.logger.info(f"Triggering automatic rollback for model {model_id}")

            # Load previous version from database
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT model_path FROM model_versions
                    WHERE model_id = ? AND is_active = 0
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (model_id,))
                row = cursor.fetchone()

                if row:
                    previous_model_path = row[0]

                    # Load previous model
                    model_data = joblib.load(previous_model_path)
                    self.active_models[model_id] = model_data

                    self.logger.info(f"Successfully rolled back model {model_id}")
                else:
                    self.logger.warning(f"No previous version found for model {model_id}")

        except Exception as e:
            self.logger.error(f"Error in automatic rollback: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        status = {
            'active_models': list(self.active_models.keys()),
            'is_running': self.is_running,
            'last_retrain_times': {},
            'performance_summary': {}
        }

        for model_id in self.active_models.keys():
            recent_performance = self._get_recent_performance(model_id)
            if recent_performance:
                status['performance_summary'][model_id] = {
                    'gary_dpi': recent_performance.gary_dpi,
                    'taleb_antifragility': recent_performance.taleb_antifragility,
                    'sharpe_ratio': recent_performance.sharpe_ratio,
                    'last_updated': recent_performance.timestamp.isoformat()
                }

        return status

    def force_retrain_all(self) -> List[RetrainingResult]:
        """Force retraining of all active models"""
        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for model_id in self.active_models.keys():
                future = executor.submit(self.retrain_model, model_id)
                futures.append(future)

            for future in futures:
                results.append(future.result())

        return results

if __name__ == "__main__":
    # Example usage
    config = LearningConfig(
        retrain_frequency_hours=12,
        min_samples_for_retrain=50,
        performance_window_days=14,
        auto_rollback_threshold=-0.03
    )

    learner = ContinuousLearner(config)
    learner.start_continuous_learning()

    print("Continuous learning pipeline started...")
    print("Use learner.get_model_status() to check status")
    print("Use learner.force_retrain_all() to force retraining")