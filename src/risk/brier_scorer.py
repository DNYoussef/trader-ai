"""
Brier Score Calibration System for Super-Gary Trading Framework

This module implements sophisticated forecast calibration tracking using Brier scores
to ensure survival-first trading with optimal position sizing based on prediction accuracy.

Mathematical Foundation:
- Brier Score: BS = (1/n) Σ(forecast_i - outcome_i)²
- Perfect calibration: BS = 0, worst case: BS = 1
- Position sizing integration: scale Kelly fraction by calibration quality
- "Skin-in-the-game": risk limits scale with Brier/log-loss performance

Key Features:
- Real-time calibration tracking across all prediction types
- Historical accuracy weighting for position sizing
- Calibration plots and reliability diagrams
- Performance scoreboard with "Position > Opinion" tracking
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CalibrationMetrics:
    """Container for calibration performance metrics"""
    brier_score: float
    log_loss: float
    reliability: float
    resolution: float
    uncertainty: float
    calibration_slope: float
    calibration_intercept: float
    num_predictions: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionRecord:
    """Single prediction record for tracking"""
    prediction_id: str
    forecast: float  # Probability [0,1]
    outcome: Optional[bool]  # Actual result when available
    confidence: float  # Model confidence
    prediction_type: str  # 'direction', 'volatility', 'regime', etc.
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class BrierScorer:
    """
    Advanced Brier Score calibration system for trading predictions

    Implements survival-first position sizing based on forecast accuracy
    with comprehensive calibration tracking and visualization.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Prediction tracking
        self.predictions: Dict[str, PredictionRecord] = {}
        self.calibration_history: List[CalibrationMetrics] = []

        # Performance tracking
        self.prediction_types = ['direction', 'volatility', 'regime', 'drawdown', 'momentum']
        self.type_metrics: Dict[str, List[CalibrationMetrics]] = {
            ptype: [] for ptype in self.prediction_types
        }

        # Position sizing integration
        self.base_kelly_multiplier = self.config.get('base_kelly_multiplier', 0.25)
        self.min_calibration_threshold = self.config.get('min_calibration_threshold', 0.6)
        self.max_position_scale = self.config.get('max_position_scale', 1.0)

        # Data persistence
        self.data_path = Path(self.config.get('data_path', './data/calibration'))
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._initialize_tracking()

    def _default_config(self) -> Dict:
        """Default configuration for calibration system"""
        return {
            'calibration_window': 1000,  # Number of predictions for rolling calibration
            'update_frequency': 100,     # Update calibration every N predictions
            'min_predictions': 50,       # Minimum predictions before using calibration
            'base_kelly_multiplier': 0.25,  # Conservative Kelly multiplier
            'min_calibration_threshold': 0.6,  # Minimum calibration score for full sizing
            'max_position_scale': 1.0,   # Maximum position scale factor
            'persistence_interval': 24,  # Hours between data saves
            'plot_update_interval': 500   # Predictions between plot updates
        }

    def _initialize_tracking(self):
        """Initialize calibration tracking system"""
        try:
            # Load existing data if available
            self._load_calibration_data()
            self.logger.info("Brier scorer initialized with historical data")
        except Exception as e:
            self.logger.warning(f"Could not load historical calibration data: {e}")
            self.logger.info("Starting fresh calibration tracking")

    def add_prediction(self,
                      prediction_id: str,
                      forecast: float,
                      prediction_type: str,
                      confidence: float = 1.0,
                      metadata: Optional[Dict] = None) -> str:
        """
        Add a new prediction to the tracking system

        Args:
            prediction_id: Unique identifier for prediction
            forecast: Probability forecast [0,1]
            prediction_type: Type of prediction ('direction', 'volatility', etc.)
            confidence: Model confidence in prediction [0,1]
            metadata: Additional prediction metadata

        Returns:
            Prediction ID for tracking
        """
        if not 0 <= forecast <= 1:
            raise ValueError(f"Forecast must be between 0 and 1, got {forecast}")

        if prediction_type not in self.prediction_types:
            self.logger.warning(f"Unknown prediction type: {prediction_type}")

        record = PredictionRecord(
            prediction_id=prediction_id,
            forecast=forecast,
            outcome=None,  # Will be set when outcome is known
            confidence=confidence,
            prediction_type=prediction_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.predictions[prediction_id] = record
        self.logger.debug(f"Added prediction {prediction_id}: {forecast:.3f} ({prediction_type})")

        return prediction_id

    def update_outcome(self, prediction_id: str, outcome: bool) -> bool:
        """
        Update prediction with actual outcome

        Args:
            prediction_id: ID of prediction to update
            outcome: True/False actual result

        Returns:
            Success of update
        """
        if prediction_id not in self.predictions:
            self.logger.error(f"Prediction ID {prediction_id} not found")
            return False

        self.predictions[prediction_id].outcome = outcome
        self.logger.debug(f"Updated outcome for {prediction_id}: {outcome}")

        # Check if we should update calibration metrics
        completed_predictions = self._get_completed_predictions()
        if len(completed_predictions) % self.config['update_frequency'] == 0:
            self._update_calibration_metrics()

        return True

    def get_calibration_score(self, prediction_type: Optional[str] = None) -> float:
        """
        Get current calibration score (1 - Brier Score)

        Args:
            prediction_type: Specific type to get score for, or None for overall

        Returns:
            Calibration score [0,1], higher is better
        """
        completed = self._get_completed_predictions(prediction_type)

        if len(completed) < self.config['min_predictions']:
            return 0.5  # Neutral score for insufficient data

        forecasts = [p.forecast for p in completed]
        outcomes = [float(p.outcome) for p in completed]

        # Use recent window for calibration
        window_size = min(len(completed), self.config['calibration_window'])
        forecasts = forecasts[-window_size:]
        outcomes = outcomes[-window_size:]

        try:
            brier_score = brier_score_loss(outcomes, forecasts)
            calibration_score = 1 - brier_score  # Convert to score (higher is better)
            return max(0, calibration_score)  # Ensure non-negative
        except Exception as e:
            self.logger.error(f"Error calculating calibration score: {e}")
            return 0.5

    def get_position_size_multiplier(self,
                                   prediction_type: Optional[str] = None,
                                   base_kelly: float = 1.0) -> float:
        """
        Calculate position size multiplier based on calibration performance

        Args:
            prediction_type: Specific prediction type to evaluate
            base_kelly: Base Kelly fraction to modify

        Returns:
            Position size multiplier [0, max_position_scale]
        """
        calibration_score = self.get_calibration_score(prediction_type)

        # Insufficient data - use conservative sizing
        completed = self._get_completed_predictions(prediction_type)
        if len(completed) < self.config['min_predictions']:
            return self.base_kelly_multiplier * 0.5

        # Scale position by calibration quality
        if calibration_score < self.min_calibration_threshold:
            # Poor calibration - reduce sizing significantly
            scale_factor = (calibration_score / self.min_calibration_threshold) * 0.5
        else:
            # Good calibration - scale up to maximum
            excess_calibration = calibration_score - self.min_calibration_threshold
            max_excess = 1.0 - self.min_calibration_threshold
            scale_factor = 0.5 + (excess_calibration / max_excess) * 0.5

        # Apply base multiplier and cap
        multiplier = base_kelly * scale_factor * self.base_kelly_multiplier
        return min(multiplier, self.max_position_scale)

    def generate_calibration_plot(self,
                                prediction_type: Optional[str] = None,
                                save_path: Optional[str] = None) -> Optional[str]:
        """
        Generate calibration plot (reliability diagram)

        Args:
            prediction_type: Specific type to plot, or None for all
            save_path: Path to save plot, or None to not save

        Returns:
            Path to saved plot or None
        """
        completed = self._get_completed_predictions(prediction_type)

        if len(completed) < 20:  # Need minimum data for meaningful plot
            self.logger.warning("Insufficient data for calibration plot")
            return None

        forecasts = np.array([p.forecast for p in completed])
        outcomes = np.array([float(p.outcome) for p in completed])

        try:
            # Create calibration plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Reliability diagram
            fraction_of_positives, mean_predicted_value = calibration_curve(
                outcomes, forecasts, n_bins=10, normalize=False
            )

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"Calibration ({prediction_type or 'All'})")
            ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Calibration Plot (Reliability Diagram)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Histogram of predictions
            ax2.hist(forecasts, bins=20, alpha=0.7, density=True,
                    label="Prediction Distribution")
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Density")
            ax2.set_title("Prediction Distribution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Calibration plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None

        except Exception as e:
            self.logger.error(f"Error generating calibration plot: {e}")
            return None

    def get_performance_scoreboard(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance scoreboard

        Returns:
            Dictionary with performance metrics and rankings
        """
        scoreboard = {
            'overall_metrics': self._calculate_overall_metrics(),
            'type_breakdown': {},
            'position_effectiveness': self._calculate_position_effectiveness(),
            'recent_performance': self._calculate_recent_performance(),
            'calibration_trend': self._calculate_calibration_trend(),
            'timestamp': datetime.now().isoformat()
        }

        # Breakdown by prediction type
        for ptype in self.prediction_types:
            type_completed = self._get_completed_predictions(ptype)
            if len(type_completed) >= 10:  # Minimum for meaningful metrics
                scoreboard['type_breakdown'][ptype] = {
                    'calibration_score': self.get_calibration_score(ptype),
                    'num_predictions': len(type_completed),
                    'position_multiplier': self.get_position_size_multiplier(ptype),
                    'recent_accuracy': self._calculate_recent_accuracy(ptype),
                    'brier_score': self._calculate_brier_score(ptype)
                }

        return scoreboard

    def _get_completed_predictions(self,
                                 prediction_type: Optional[str] = None) -> List[PredictionRecord]:
        """Get predictions with known outcomes"""
        completed = [p for p in self.predictions.values()
                    if p.outcome is not None]

        if prediction_type:
            completed = [p for p in completed if p.prediction_type == prediction_type]

        # Sort by timestamp
        completed.sort(key=lambda x: x.timestamp)
        return completed

    def _update_calibration_metrics(self):
        """Update calibration metrics for all prediction types"""
        try:
            # Overall metrics
            overall_metrics = self._calculate_calibration_metrics()
            if overall_metrics:
                self.calibration_history.append(overall_metrics)

            # Type-specific metrics
            for ptype in self.prediction_types:
                type_metrics = self._calculate_calibration_metrics(ptype)
                if type_metrics:
                    self.type_metrics[ptype].append(type_metrics)

            # Trim history to prevent memory bloat
            max_history = 1000
            self.calibration_history = self.calibration_history[-max_history:]
            for ptype in self.prediction_types:
                self.type_metrics[ptype] = self.type_metrics[ptype][-max_history:]

            self.logger.debug("Updated calibration metrics")

        except Exception as e:
            self.logger.error(f"Error updating calibration metrics: {e}")

    def _calculate_calibration_metrics(self,
                                     prediction_type: Optional[str] = None) -> Optional[CalibrationMetrics]:
        """Calculate detailed calibration metrics"""
        completed = self._get_completed_predictions(prediction_type)

        if len(completed) < self.config['min_predictions']:
            return None

        forecasts = np.array([p.forecast for p in completed])
        outcomes = np.array([float(p.outcome) for p in completed])

        try:
            # Core metrics
            brier_score = brier_score_loss(outcomes, forecasts)
            log_loss_score = log_loss(outcomes, forecasts, eps=1e-15)

            # Decomposition of Brier score
            outcome_mean = np.mean(outcomes)
            reliability = np.mean((forecasts - outcomes) ** 2)  # Calibration component
            resolution = np.mean((forecasts - outcome_mean) ** 2)  # Resolution component
            uncertainty = outcome_mean * (1 - outcome_mean)  # Uncertainty component

            # Calibration line fit
            try:
                slope, intercept = np.polyfit(forecasts, outcomes, 1)
            except:
                slope, intercept = 0, outcome_mean

            return CalibrationMetrics(
                brier_score=brier_score,
                log_loss=log_loss_score,
                reliability=reliability,
                resolution=resolution,
                uncertainty=uncertainty,
                calibration_slope=slope,
                calibration_intercept=intercept,
                num_predictions=len(completed)
            )

        except Exception as e:
            self.logger.error(f"Error calculating calibration metrics: {e}")
            return None

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        completed = self._get_completed_predictions()

        if len(completed) < 10:
            return {'status': 'insufficient_data', 'num_predictions': len(completed)}

        forecasts = [p.forecast for p in completed]
        outcomes = [float(p.outcome) for p in completed]

        return {
            'total_predictions': len(completed),
            'calibration_score': self.get_calibration_score(),
            'average_confidence': np.mean([p.confidence for p in completed]),
            'brier_score': brier_score_loss(outcomes, forecasts),
            'accuracy': np.mean([(f > 0.5) == o for f, o in zip(forecasts, outcomes)]),
            'position_multiplier': self.get_position_size_multiplier(),
            'data_span_days': (completed[-1].timestamp - completed[0].timestamp).days
        }

    def _calculate_position_effectiveness(self) -> Dict[str, float]:
        """Calculate how effective position sizing has been"""
        # This would integrate with actual trading results
        # For now, return theoretical effectiveness based on calibration
        calibration_score = self.get_calibration_score()

        return {
            'sizing_effectiveness': calibration_score,
            'theoretical_sharpe_improvement': max(0, (calibration_score - 0.5) * 2),
            'risk_adjusted_return': calibration_score * self.get_position_size_multiplier()
        }

    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate performance over recent window"""
        completed = self._get_completed_predictions()

        if len(completed) < 20:
            return {'status': 'insufficient_data'}

        # Use recent 25% of data
        recent_count = max(10, len(completed) // 4)
        recent = completed[-recent_count:]

        forecasts = [p.forecast for p in recent]
        outcomes = [float(p.outcome) for p in recent]

        return {
            'recent_calibration': 1 - brier_score_loss(outcomes, forecasts),
            'recent_accuracy': np.mean([(f > 0.5) == o for f, o in zip(forecasts, outcomes)]),
            'recent_predictions': len(recent),
            'days_span': (recent[-1].timestamp - recent[0].timestamp).days
        }

    def _calculate_calibration_trend(self) -> Dict[str, Any]:
        """Calculate trend in calibration performance"""
        if len(self.calibration_history) < 5:
            return {'status': 'insufficient_history'}

        recent_scores = [1 - m.brier_score for m in self.calibration_history[-10:]]
        older_scores = [1 - m.brier_score for m in self.calibration_history[-20:-10]] if len(self.calibration_history) >= 20 else []

        trend = {
            'recent_average': np.mean(recent_scores),
            'trend_direction': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable'
        }

        if older_scores:
            trend['improvement'] = np.mean(recent_scores) - np.mean(older_scores)
            trend['trend_direction'] = 'improving' if trend['improvement'] > 0.01 else 'declining' if trend['improvement'] < -0.01 else 'stable'

        return trend

    def _calculate_recent_accuracy(self, prediction_type: str) -> float:
        """Calculate recent accuracy for specific prediction type"""
        completed = self._get_completed_predictions(prediction_type)

        if len(completed) < 10:
            return 0.5

        recent_count = max(5, len(completed) // 4)
        recent = completed[-recent_count:]

        forecasts = [p.forecast for p in recent]
        outcomes = [p.outcome for p in recent]

        return np.mean([(f > 0.5) == o for f, o in zip(forecasts, outcomes)])

    def _calculate_brier_score(self, prediction_type: str) -> float:
        """Calculate Brier score for specific prediction type"""
        completed = self._get_completed_predictions(prediction_type)

        if len(completed) < 10:
            return 1.0  # Worst possible score

        forecasts = [p.forecast for p in completed]
        outcomes = [float(p.outcome) for p in completed]

        return brier_score_loss(outcomes, forecasts)

    def _save_calibration_data(self):
        """Save calibration data to disk"""
        try:
            # Save predictions
            predictions_data = {
                pid: {
                    'prediction_id': p.prediction_id,
                    'forecast': p.forecast,
                    'outcome': p.outcome,
                    'confidence': p.confidence,
                    'prediction_type': p.prediction_type,
                    'timestamp': p.timestamp.isoformat(),
                    'metadata': p.metadata
                }
                for pid, p in self.predictions.items()
            }

            with open(self.data_path / 'predictions.json', 'w') as f:
                json.dump(predictions_data, f, indent=2)

            # Save calibration history
            history_data = [
                {
                    'brier_score': m.brier_score,
                    'log_loss': m.log_loss,
                    'reliability': m.reliability,
                    'resolution': m.resolution,
                    'uncertainty': m.uncertainty,
                    'calibration_slope': m.calibration_slope,
                    'calibration_intercept': m.calibration_intercept,
                    'num_predictions': m.num_predictions,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.calibration_history
            ]

            with open(self.data_path / 'calibration_history.json', 'w') as f:
                json.dump(history_data, f, indent=2)

            self.logger.info("Calibration data saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving calibration data: {e}")

    def _load_calibration_data(self):
        """Load calibration data from disk"""
        # Load predictions
        predictions_file = self.data_path / 'predictions.json'
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                predictions_data = json.load(f)

            for pid, data in predictions_data.items():
                self.predictions[pid] = PredictionRecord(
                    prediction_id=data['prediction_id'],
                    forecast=data['forecast'],
                    outcome=data['outcome'],
                    confidence=data['confidence'],
                    prediction_type=data['prediction_type'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    metadata=data['metadata']
                )

        # Load calibration history
        history_file = self.data_path / 'calibration_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)

            self.calibration_history = [
                CalibrationMetrics(
                    brier_score=m['brier_score'],
                    log_loss=m['log_loss'],
                    reliability=m['reliability'],
                    resolution=m['resolution'],
                    uncertainty=m['uncertainty'],
                    calibration_slope=m['calibration_slope'],
                    calibration_intercept=m['calibration_intercept'],
                    num_predictions=m['num_predictions'],
                    timestamp=datetime.fromisoformat(m['timestamp'])
                )
                for m in history_data
            ]

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create calibration system
    scorer = BrierScorer()

    # Simulate some predictions
    np.random.seed(42)
    for i in range(100):
        # Simulate predictions with varying quality
        true_prob = np.random.random()
        noise = np.random.normal(0, 0.2)  # Add some miscalibration
        forecast = np.clip(true_prob + noise, 0.01, 0.99)

        outcome = np.random.random() < true_prob

        pred_id = f"test_prediction_{i}"
        scorer.add_prediction(pred_id, forecast, "direction", confidence=0.8)
        scorer.update_outcome(pred_id, outcome)

    # Generate performance report
    scoreboard = scorer.get_performance_scoreboard()
    print("Calibration Performance Scoreboard:")
    print(f"Overall Calibration Score: {scoreboard['overall_metrics']['calibration_score']:.3f}")
    print(f"Position Size Multiplier: {scoreboard['overall_metrics']['position_multiplier']:.3f}")
    print(f"Total Predictions: {scoreboard['overall_metrics']['total_predictions']}")

    # Generate calibration plot
    scorer.generate_calibration_plot()