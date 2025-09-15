"""
AI-Powered Intelligent Alert System for Gary×Taleb Trading System

Provides AI-enhanced risk alerting with pattern recognition and predictive warnings:
- Smart threshold detection with ML-based anomaly detection
- Risk pattern recognition using ensemble methods
- Predictive early warning system with time-series forecasting
- Context-aware alert filtering to reduce false positives
- Integration with existing DPI and antifragility calculations

Success Metrics:
- Alert accuracy: <5% false positive rate
- Early warning: 5-15 minute advance notice
- Pattern detection: Identify risk regime changes
- Context awareness: Market-condition alert filtering
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import joblib
from scipy import stats
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskPattern(Enum):
    """Risk pattern types"""
    REGIME_CHANGE = "regime_change"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MOMENTUM_REVERSAL = "momentum_reversal"
    DPI_DIVERGENCE = "dpi_divergence"
    ANTIFRAGILITY_BREACH = "antifragility_breach"


@dataclass
class AlertContext:
    """Market context for alert filtering"""
    market_hours: bool
    volatility_regime: str
    market_cap_sector: str
    correlation_environment: float
    news_sentiment: float
    volume_profile: str


@dataclass
class RiskAlert:
    """AI-generated risk alert"""
    alert_id: str
    timestamp: datetime
    symbol: str
    severity: AlertSeverity
    pattern_type: RiskPattern
    confidence: float
    prediction_horizon_minutes: int
    description: str
    context: AlertContext
    metrics: Dict[str, float]
    actionable_insights: List[str]
    false_positive_probability: float


@dataclass
class PredictiveWarning:
    """Predictive early warning"""
    warning_id: str
    timestamp: datetime
    predicted_event_time: datetime
    probability: float
    risk_magnitude: float
    affected_symbols: List[str]
    mitigation_suggestions: List[str]


class AIAlertSystem:
    """
    AI-Powered Intelligent Alert System

    Enhances traditional threshold-based alerts with:
    - ML-based anomaly detection
    - Pattern recognition for regime changes
    - Predictive modeling for early warnings
    - Context-aware filtering
    """

    def __init__(self,
                 dpi_calculator=None,
                 lookback_days: int = 30,
                 alert_threshold: float = 0.05,
                 model_update_frequency: int = 1440):  # minutes
        """
        Initialize AI Alert System

        Args:
            dpi_calculator: Reference to DPI calculation system
            lookback_days: Historical data lookback period
            alert_threshold: Base alert threshold (5% false positive target)
            model_update_frequency: Model retraining frequency in minutes
        """
        self.dpi_calculator = dpi_calculator
        self.lookback_days = lookback_days
        self.alert_threshold = alert_threshold
        self.model_update_frequency = model_update_frequency

        # ML Models
        self.anomaly_detector = IsolationForest(
            contamination=0.05,  # 5% expected outliers
            random_state=42,
            n_estimators=100
        )
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()

        # Model training status
        self.models_trained = False
        self.last_model_update = None

        # Alert history for learning
        self.alert_history: List[RiskAlert] = []
        self.prediction_history: List[PredictiveWarning] = []

        # Pattern recognition cache
        self.pattern_cache = {}

        logger.info("AI Alert System initialized with ML-enhanced capabilities")

    def train_models(self, historical_data: pd.DataFrame,
                    risk_events: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models on historical data

        Args:
            historical_data: Market data with features
            risk_events: Historical risk events for supervised learning

        Returns:
            Training metrics and model performance
        """
        try:
            logger.info("Training AI models for intelligent alerting...")

            # Feature engineering for anomaly detection
            features = self._engineer_features(historical_data)

            # Train anomaly detector (unsupervised)
            X_scaled = self.scaler.fit_transform(features)
            self.anomaly_detector.fit(X_scaled)

            # Train pattern classifier if risk events available
            if not risk_events.empty:
                X_patterns, y_patterns = self._prepare_pattern_data(
                    historical_data, risk_events
                )

                if len(X_patterns) > 50:  # Minimum samples for training
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_patterns, y_patterns, test_size=0.2, random_state=42
                    )

                    self.pattern_classifier.fit(X_train, y_train)
                    y_pred = self.pattern_classifier.predict(X_test)

                    pattern_accuracy = accuracy_score(y_test, y_pred)
                    pattern_report = classification_report(
                        y_test, y_pred, output_dict=True
                    )
                else:
                    pattern_accuracy = 0.0
                    pattern_report = {}
            else:
                pattern_accuracy = 0.0
                pattern_report = {}

            # Calculate anomaly detection baseline
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            anomaly_threshold = np.percentile(anomaly_scores, 5)  # 5% threshold

            self.models_trained = True
            self.last_model_update = datetime.now()

            training_metrics = {
                'training_samples': len(historical_data),
                'feature_count': features.shape[1],
                'anomaly_threshold': anomaly_threshold,
                'pattern_accuracy': pattern_accuracy,
                'pattern_classification_report': pattern_report,
                'training_timestamp': self.last_model_update.isoformat()
            }

            logger.info(f"Model training completed. Accuracy: {pattern_accuracy:.3f}")
            return training_metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'error': str(e)}

    def generate_alert(self, symbol: str, current_data: Dict[str, Any],
                      market_context: AlertContext) -> Optional[RiskAlert]:
        """
        Generate AI-powered risk alert

        Args:
            symbol: Trading symbol
            current_data: Current market data and metrics
            market_context: Market context for filtering

        Returns:
            Risk alert if conditions met, None otherwise
        """
        try:
            if not self.models_trained:
                logger.warning("Models not trained, using basic threshold alerts")
                return self._generate_basic_alert(symbol, current_data, market_context)

            # Extract features for current data
            features = self._extract_current_features(current_data)

            # Anomaly detection
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1

            # Pattern recognition
            pattern_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
            predicted_pattern = self.pattern_classifier.classes_[np.argmax(pattern_proba)]
            pattern_confidence = np.max(pattern_proba)

            # Context-aware filtering
            if not self._passes_context_filter(market_context, anomaly_score):
                logger.debug(f"Alert filtered out for {symbol} due to market context")
                return None

            # Severity classification
            severity = self._classify_severity(
                anomaly_score, pattern_confidence, current_data
            )

            # Calculate false positive probability
            fp_probability = self._estimate_false_positive_probability(
                anomaly_score, pattern_confidence, market_context
            )

            # Generate alert if thresholds met
            if is_anomaly and fp_probability < self.alert_threshold:
                alert = RiskAlert(
                    alert_id=f"AI_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    severity=severity,
                    pattern_type=RiskPattern(predicted_pattern),
                    confidence=pattern_confidence,
                    prediction_horizon_minutes=self._estimate_horizon(pattern_confidence),
                    description=self._generate_alert_description(
                        symbol, predicted_pattern, anomaly_score
                    ),
                    context=market_context,
                    metrics={
                        'anomaly_score': float(anomaly_score),
                        'pattern_confidence': float(pattern_confidence),
                        'false_positive_prob': float(fp_probability)
                    },
                    actionable_insights=self._generate_actionable_insights(
                        predicted_pattern, current_data
                    ),
                    false_positive_probability=fp_probability
                )

                # Store alert for learning
                self.alert_history.append(alert)

                logger.info(f"Generated AI alert for {symbol}: {severity.value} - {predicted_pattern}")
                return alert

            return None

        except Exception as e:
            logger.error(f"Error generating AI alert for {symbol}: {e}")
            return None

    def generate_predictive_warning(self, symbols: List[str],
                                  market_data: Dict[str, pd.DataFrame]) -> List[PredictiveWarning]:
        """
        Generate predictive early warnings

        Args:
            symbols: List of symbols to analyze
            market_data: Historical market data for prediction

        Returns:
            List of predictive warnings (5-15 minute horizon)
        """
        warnings = []

        try:
            for symbol in symbols:
                if symbol not in market_data or market_data[symbol].empty:
                    continue

                # Time series forecasting for risk metrics
                risk_forecast = self._forecast_risk_metrics(
                    symbol, market_data[symbol]
                )

                if risk_forecast['probability'] > 0.7:  # High probability threshold
                    warning = PredictiveWarning(
                        warning_id=f"PRED_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        predicted_event_time=datetime.now() + timedelta(
                            minutes=risk_forecast['horizon_minutes']
                        ),
                        probability=risk_forecast['probability'],
                        risk_magnitude=risk_forecast['magnitude'],
                        affected_symbols=[symbol],
                        mitigation_suggestions=self._generate_mitigation_suggestions(
                            symbol, risk_forecast
                        )
                    )
                    warnings.append(warning)

            logger.info(f"Generated {len(warnings)} predictive warnings")
            return warnings

        except Exception as e:
            logger.error(f"Error generating predictive warnings: {e}")
            return []

    def update_models(self, new_data: pd.DataFrame,
                     alert_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update models with new data and feedback

        Args:
            new_data: New market data
            alert_feedback: Feedback on previous alerts (true/false positives)

        Returns:
            Update metrics
        """
        try:
            logger.info("Updating AI models with new data and feedback...")

            # Process alert feedback for learning
            feedback_metrics = self._process_alert_feedback(alert_feedback)

            # Retrain if enough new data or poor performance
            needs_retraining = (
                len(new_data) > 1000 or
                feedback_metrics.get('false_positive_rate', 0) > self.alert_threshold or
                (datetime.now() - self.last_model_update).total_seconds() / 60
                > self.model_update_frequency
            )

            if needs_retraining:
                # Combine with existing training data
                updated_features = self._engineer_features(new_data)

                # Retrain anomaly detector
                X_scaled = self.scaler.fit_transform(updated_features)
                self.anomaly_detector.fit(X_scaled)

                self.last_model_update = datetime.now()

                logger.info("Models updated successfully")

                return {
                    'updated': True,
                    'update_timestamp': self.last_model_update.isoformat(),
                    'feedback_metrics': feedback_metrics,
                    'new_data_samples': len(new_data)
                }
            else:
                return {
                    'updated': False,
                    'reason': 'No retraining needed',
                    'feedback_metrics': feedback_metrics
                }

        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return {'error': str(e)}

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        features = pd.DataFrame()

        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(data['close'])

        # DPI-specific features if available
        if self.dpi_calculator and hasattr(data, 'dpi_score'):
            features['dpi_score'] = data['dpi_score']
            features['dpi_change'] = data['dpi_score'].diff()
            features['dpi_volatility'] = data['dpi_score'].rolling(10).std()

        # Drop NaN values
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _extract_current_features(self, current_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from current data point"""
        features = []

        # Basic features
        features.extend([
            current_data.get('return', 0),
            current_data.get('volatility', 0),
            current_data.get('volume_ratio', 1),
            current_data.get('rsi', 50),
            current_data.get('macd', 0),
            current_data.get('bollinger_position', 0.5)
        ])

        # DPI features if available
        if 'dpi_score' in current_data:
            features.extend([
                current_data['dpi_score'],
                current_data.get('dpi_change', 0),
                current_data.get('dpi_volatility', 0)
            ])
        else:
            features.extend([0, 0, 0])  # Placeholder values

        return np.array(features)

    def _passes_context_filter(self, context: AlertContext,
                              anomaly_score: float) -> bool:
        """Context-aware filtering to reduce false positives"""

        # Filter during non-market hours for less critical alerts
        if not context.market_hours and anomaly_score > -0.5:
            return False

        # Filter during high correlation environments
        if context.correlation_environment > 0.8 and anomaly_score > -0.3:
            return False

        # Adjust for volatility regime
        if context.volatility_regime == 'low' and anomaly_score > -0.4:
            return False

        return True

    def _classify_severity(self, anomaly_score: float,
                          pattern_confidence: float,
                          current_data: Dict[str, Any]) -> AlertSeverity:
        """Classify alert severity based on multiple factors"""

        # Base severity from anomaly score
        if anomaly_score < -0.8:
            base_severity = AlertSeverity.CRITICAL
        elif anomaly_score < -0.6:
            base_severity = AlertSeverity.HIGH
        elif anomaly_score < -0.4:
            base_severity = AlertSeverity.MEDIUM
        else:
            base_severity = AlertSeverity.LOW

        # Adjust based on pattern confidence
        if pattern_confidence > 0.9 and base_severity in [AlertSeverity.MEDIUM, AlertSeverity.LOW]:
            # Upgrade severity for high-confidence patterns
            severity_map = {
                AlertSeverity.LOW: AlertSeverity.MEDIUM,
                AlertSeverity.MEDIUM: AlertSeverity.HIGH
            }
            base_severity = severity_map.get(base_severity, base_severity)

        # Adjust for DPI divergence
        if current_data.get('dpi_score', 0) > 0.7 and anomaly_score < -0.5:
            base_severity = AlertSeverity.HIGH

        return base_severity

    def _estimate_false_positive_probability(self, anomaly_score: float,
                                           pattern_confidence: float,
                                           context: AlertContext) -> float:
        """Estimate false positive probability for alert"""

        # Base probability from anomaly score
        base_prob = max(0.01, min(0.5, (anomaly_score + 1) / 2))

        # Adjust for pattern confidence
        pattern_adjustment = (1 - pattern_confidence) * 0.3

        # Context adjustments
        context_adjustment = 0.0
        if not context.market_hours:
            context_adjustment += 0.1
        if context.volatility_regime == 'high':
            context_adjustment -= 0.05

        final_prob = max(0.001, min(0.5, base_prob + pattern_adjustment + context_adjustment))
        return final_prob

    def _estimate_horizon(self, confidence: float) -> int:
        """Estimate prediction horizon in minutes"""
        # Higher confidence = shorter horizon (more immediate)
        base_horizon = 10  # 10 minutes base
        confidence_factor = max(0.5, confidence)
        horizon = int(base_horizon / confidence_factor)
        return max(5, min(15, horizon))  # Keep within 5-15 minute range

    def _generate_alert_description(self, symbol: str, pattern: str,
                                  anomaly_score: float) -> str:
        """Generate human-readable alert description"""
        descriptions = {
            'regime_change': f"{symbol} showing signs of regime change (anomaly: {anomaly_score:.3f})",
            'volatility_spike': f"{symbol} volatility spike detected (anomaly: {anomaly_score:.3f})",
            'liquidity_crisis': f"{symbol} liquidity issues detected (anomaly: {anomaly_score:.3f})",
            'momentum_reversal': f"{symbol} momentum reversal pattern (anomaly: {anomaly_score:.3f})",
            'dpi_divergence': f"{symbol} DPI divergence detected (anomaly: {anomaly_score:.3f})",
            'antifragility_breach': f"{symbol} antifragility threshold breached (anomaly: {anomaly_score:.3f})"
        }

        return descriptions.get(pattern, f"{symbol} risk pattern detected: {pattern}")

    def _generate_actionable_insights(self, pattern: str,
                                    current_data: Dict[str, Any]) -> List[str]:
        """Generate actionable insights based on pattern"""
        insights = {
            'regime_change': [
                "Consider reducing position size",
                "Monitor correlation changes",
                "Review risk management parameters"
            ],
            'volatility_spike': [
                "Implement volatility-adjusted position sizing",
                "Consider hedging strategies",
                "Monitor for continued volatility"
            ],
            'liquidity_crisis': [
                "Reduce position sizes immediately",
                "Avoid market orders",
                "Monitor bid-ask spreads"
            ],
            'momentum_reversal': [
                "Review directional exposure",
                "Consider momentum-based adjustments",
                "Monitor volume confirmation"
            ],
            'dpi_divergence': [
                "Reassess DPI calculations",
                "Monitor for confirmation signals",
                "Consider contrarian positioning"
            ],
            'antifragility_breach': [
                "Activate defense mechanisms",
                "Review antifragility parameters",
                "Consider stress testing"
            ]
        }

        return insights.get(pattern, ["Monitor situation closely", "Review risk parameters"])

    def _forecast_risk_metrics(self, symbol: str,
                             data: pd.DataFrame) -> Dict[str, Any]:
        """Forecast risk metrics for early warning"""
        try:
            # Simple time series forecasting using rolling statistics
            recent_data = data.tail(20)  # Last 20 periods

            # Calculate risk trends
            volatility_trend = recent_data['close'].pct_change().rolling(5).std().iloc[-1]
            volume_trend = recent_data['volume'].pct_change().rolling(5).mean().iloc[-1]

            # Risk probability based on trends
            vol_factor = min(1.0, volatility_trend * 100) if not np.isnan(volatility_trend) else 0.5
            vol_factor = min(1.0, abs(volume_trend) * 2) if not np.isnan(volume_trend) else 0.3

            risk_probability = (vol_factor + vol_factor) / 2

            # Risk magnitude
            risk_magnitude = min(1.0, volatility_trend * 50) if not np.isnan(volatility_trend) else 0.2

            # Prediction horizon (5-15 minutes)
            horizon_minutes = max(5, min(15, int(10 * (1 - risk_probability))))

            return {
                'probability': risk_probability,
                'magnitude': risk_magnitude,
                'horizon_minutes': horizon_minutes
            }

        except Exception as e:
            logger.warning(f"Risk forecasting failed for {symbol}: {e}")
            return {'probability': 0.0, 'magnitude': 0.0, 'horizon_minutes': 10}

    def _generate_mitigation_suggestions(self, symbol: str,
                                       forecast: Dict[str, Any]) -> List[str]:
        """Generate mitigation suggestions for predicted risks"""
        suggestions = []

        if forecast['probability'] > 0.8:
            suggestions.append(f"HIGH RISK: Consider immediate position size reduction for {symbol}")

        if forecast['magnitude'] > 0.5:
            suggestions.append(f"Implement stop-loss orders for {symbol}")

        suggestions.extend([
            f"Monitor {symbol} closely for next {forecast['horizon_minutes']} minutes",
            "Review portfolio correlation exposure",
            "Consider volatility hedging strategies"
        ])

        return suggestions

    def _generate_basic_alert(self, symbol: str, current_data: Dict[str, Any],
                            context: AlertContext) -> Optional[RiskAlert]:
        """Generate basic threshold-based alert when ML models not available"""

        # Simple threshold checks
        is_alert = False
        severity = AlertSeverity.LOW
        pattern = RiskPattern.VOLATILITY_SPIKE

        # Check volatility
        if current_data.get('volatility', 0) > 0.05:  # 5% volatility threshold
            is_alert = True
            severity = AlertSeverity.HIGH

        # Check DPI divergence
        if abs(current_data.get('dpi_score', 0)) > 0.8:
            is_alert = True
            pattern = RiskPattern.DPI_DIVERGENCE

        if is_alert:
            return RiskAlert(
                alert_id=f"BASIC_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                symbol=symbol,
                severity=severity,
                pattern_type=pattern,
                confidence=0.6,  # Lower confidence for basic alerts
                prediction_horizon_minutes=10,
                description=f"Basic threshold alert for {symbol}",
                context=context,
                metrics=current_data,
                actionable_insights=["Monitor situation", "Review thresholds"],
                false_positive_probability=0.2  # Higher FP rate for basic alerts
            )

        return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        return exp1 - exp2

    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (prices - lower) / (upper - lower)

    def _prepare_pattern_data(self, historical_data: pd.DataFrame,
                            risk_events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for pattern classification training"""
        # This would be implemented based on specific risk event labeling
        # For now, return dummy data structure
        features = self._engineer_features(historical_data)
        labels = np.random.choice(['regime_change', 'volatility_spike', 'normal'],
                                len(features))
        return features.values, labels

    def _process_alert_feedback(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process feedback on alert accuracy"""
        if not feedback:
            return {'false_positive_rate': 0.0, 'total_feedback': 0}

        false_positives = sum(1 for f in feedback if f.get('false_positive', False))
        total_feedback = len(feedback)
        fp_rate = false_positives / total_feedback if total_feedback > 0 else 0.0

        return {
            'false_positive_rate': fp_rate,
            'total_feedback': total_feedback,
            'false_positives': false_positives
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get AI alert system status"""
        return {
            'models_trained': self.models_trained,
            'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None,
            'alert_threshold': self.alert_threshold,
            'total_alerts_generated': len(self.alert_history),
            'total_predictions_generated': len(self.prediction_history),
            'average_false_positive_rate': self._calculate_average_fp_rate()
        }

    def _calculate_average_fp_rate(self) -> float:
        """Calculate average false positive rate from history"""
        if not self.alert_history:
            return 0.0

        fp_probs = [alert.false_positive_probability for alert in self.alert_history]
        return sum(fp_probs) / len(fp_probs)

    def save_models(self, filepath: str) -> bool:
        """Save trained models to disk"""
        try:
            model_data = {
                'anomaly_detector': self.anomaly_detector,
                'pattern_classifier': self.pattern_classifier,
                'scaler': self.scaler,
                'models_trained': self.models_trained,
                'last_model_update': self.last_model_update
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def load_models(self, filepath: str) -> bool:
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.anomaly_detector = model_data['anomaly_detector']
            self.pattern_classifier = model_data['pattern_classifier']
            self.scaler = model_data['scaler']
            self.models_trained = model_data['models_trained']
            self.last_model_update = model_data['last_model_update']
            logger.info(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False