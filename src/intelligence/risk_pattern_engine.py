"""
Risk Pattern Recognition Engine

Advanced pattern detection system that identifies risk regime changes and market anomalies:
- Real-time pattern detection using ensemble methods
- Market regime classification and transition detection
- Multi-timeframe analysis for comprehensive risk assessment
- Integration with DPI system for enhanced signal validation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGE_BOUND = "range_bound"
    CRISIS = "crisis"
    TRANSITION = "transition"


class PatternType(Enum):
    """Risk pattern types"""
    REGIME_CHANGE = "regime_change"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    VOLUME_ANOMALY = "volume_anomaly"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_STRESS = "liquidity_stress"
    TAIL_RISK = "tail_risk"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PatternDetection:
    """Pattern detection result"""
    pattern_type: PatternType
    confidence: float
    strength: float  # Pattern strength 0-1
    timeframe: str  # 1m, 5m, 15m, 1h, 1d
    detection_time: datetime
    affected_symbols: List[str]
    description: str
    risk_metrics: Dict[str, float]
    suggested_actions: List[str]


@dataclass
class AlertMetadata:
    """Metadata for risk alerts"""
    alert_id: str
    severity: RiskLevel
    message: str
    timestamp: datetime
    source: str


@dataclass
class DetectionResult:
    """Risk detection result"""
    risk_level: RiskLevel
    confidence: float
    detected_patterns: List[PatternType]
    risk_score: float
    timestamp: datetime


@dataclass
class RegimeAnalysis:
    """Market regime analysis result"""
    current_regime: MarketRegime
    regime_confidence: float
    regime_duration: int  # days
    transition_probability: float
    next_likely_regime: MarketRegime
    regime_characteristics: Dict[str, float]


class PatternAnalyzer:
    """
    Pattern Analysis System for Risk Detection

    Analyzes market patterns for early risk signal detection
    and regime change identification.
    """

    def __init__(self):
        self.pattern_history = []
        self.regime_transitions = []

    def analyze_pattern(self, data: np.ndarray) -> PatternDetection:
        """Analyze pattern in market data"""
        pattern_type = PatternType.TREND_CONTINUATION  # Default
        confidence = 0.5
        timeframe = "1d"
        characteristics = {}

        return PatternDetection(
            pattern_type=pattern_type,
            confidence=confidence,
            timeframe=timeframe,
            characteristics=characteristics
        )

    def detect_regime_change(self, market_data: Dict[str, Any]) -> bool:
        """Detect if market regime is changing"""
        return False  # Simplified implementation

    def get_pattern_strength(self, pattern: PatternDetection) -> float:
        """Calculate pattern strength score"""
        return pattern.confidence


class RiskPatternEngine:
    """
    Advanced Risk Pattern Recognition Engine

    Uses ensemble methods to detect risk patterns across multiple timeframes
    and market conditions. Integrates with DPI system for enhanced accuracy.
    """

    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize Risk Pattern Engine

        Args:
            lookback_periods: Lookback periods for different timeframes
        """
        self.lookback_periods = lookback_periods or {
            '1m': 60,      # 1 hour of minute data
            '5m': 144,     # 12 hours of 5-minute data
            '15m': 96,     # 24 hours of 15-minute data
            '1h': 168,     # 1 week of hourly data
            '1d': 252      # 1 year of daily data
        }

        # Pattern detection parameters
        self.regime_lookback = 50  # days for regime analysis
        self.volatility_threshold = 2.0  # Z-score threshold
        self.correlation_threshold = 0.3  # Correlation breakdown threshold

        # Regime classifier
        self.regime_classifier = KMeans(n_clusters=7, random_state=42)
        self.scaler = MinMaxScaler()

        # Pattern history for learning
        self.detected_patterns = []
        self.regime_history = []

        logger.info("Risk Pattern Engine initialized")

    def detect_patterns(self, symbol: str, market_data: Dict[str, pd.DataFrame],
                       dpi_data: Optional[Dict[str, Any]] = None) -> List[PatternDetection]:
        """
        Detect risk patterns across multiple timeframes

        Args:
            symbol: Trading symbol to analyze
            market_data: Market data for different timeframes
            dpi_data: Optional DPI data for enhanced detection

        Returns:
            List of detected patterns
        """
        detected_patterns = []

        try:
            logger.info(f"Detecting risk patterns for {symbol}")

            # Analyze each timeframe
            for timeframe, data in market_data.items():
                if data.empty or len(data) < self.lookback_periods.get(timeframe, 20):
                    continue

                # Pattern detection methods
                patterns = []

                # 1. Regime change detection
                regime_pattern = self._detect_regime_change(symbol, data, timeframe)
                if regime_pattern:
                    patterns.append(regime_pattern)

                # 2. Volatility breakout detection
                volatility_pattern = self._detect_volatility_breakout(symbol, data, timeframe)
                if volatility_pattern:
                    patterns.append(volatility_pattern)

                # 3. Momentum divergence detection
                momentum_pattern = self._detect_momentum_divergence(symbol, data, timeframe)
                if momentum_pattern:
                    patterns.append(momentum_pattern)

                # 4. Volume anomaly detection
                volume_pattern = self._detect_volume_anomaly(symbol, data, timeframe)
                if volume_pattern:
                    patterns.append(volume_pattern)

                # 5. DPI-enhanced detection if available
                if dpi_data:
                    dpi_pattern = self._detect_dpi_divergence(symbol, data, timeframe, dpi_data)
                    if dpi_pattern:
                        patterns.append(dpi_pattern)

                detected_patterns.extend(patterns)

            # Filter and rank patterns
            filtered_patterns = self._filter_and_rank_patterns(detected_patterns)

            # Store for historical analysis
            self.detected_patterns.extend(filtered_patterns)

            logger.info(f"Detected {len(filtered_patterns)} patterns for {symbol}")
            return filtered_patterns

        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}")
            return []

    def analyze_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> RegimeAnalysis:
        """
        Analyze current market regime

        Args:
            market_data: Market data across symbols and timeframes

        Returns:
            Market regime analysis
        """
        try:
            # Aggregate market features across all symbols
            regime_features = self._extract_regime_features(market_data)

            # Classify current regime
            current_regime = self._classify_regime(regime_features)

            # Calculate regime confidence
            regime_confidence = self._calculate_regime_confidence(regime_features)

            # Estimate regime duration
            regime_duration = self._estimate_regime_duration(regime_features)

            # Predict regime transition probability
            transition_prob = self._calculate_transition_probability(regime_features)

            # Predict next likely regime
            next_regime = self._predict_next_regime(regime_features)

            # Extract regime characteristics
            characteristics = self._extract_regime_characteristics(regime_features)

            analysis = RegimeAnalysis(
                current_regime=current_regime,
                regime_confidence=regime_confidence,
                regime_duration=regime_duration,
                transition_probability=transition_prob,
                next_likely_regime=next_regime,
                regime_characteristics=characteristics
            )

            self.regime_history.append(analysis)
            logger.info(f"Market regime: {current_regime.value} (confidence: {regime_confidence:.2f})")

            return analysis

        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return RegimeAnalysis(
                current_regime=MarketRegime.TRANSITION,
                regime_confidence=0.5,
                regime_duration=1,
                transition_probability=0.5,
                next_likely_regime=MarketRegime.TRANSITION,
                regime_characteristics={}
            )

    def _detect_regime_change(self, symbol: str, data: pd.DataFrame,
                            timeframe: str) -> Optional[PatternDetection]:
        """Detect regime change patterns"""
        try:
            if len(data) < 30:
                return None

            # Calculate rolling statistics
            window = min(20, len(data) // 2)
            returns = data['close'].pct_change()
            volatility = returns.rolling(window).std()
            volume_trend = data['volume'].rolling(window).mean()

            # Detect structural breaks using rolling Z-scores
            vol_zscore = zscore(volatility.dropna())
            vol_breaks = np.where(np.abs(vol_zscore) > 2.5)[0]

            # Volume pattern changes
            vol_ratio = volume_trend / volume_trend.shift(window)
            vol_changes = np.where(vol_ratio > 1.5)[0]  # 50% volume increase

            # Regime change signal if both volatility and volume show breaks
            if len(vol_breaks) > 0 and len(vol_changes) > 0:
                # Check if breaks are recent
                recent_vol_break = vol_breaks[-1] > len(data) - 10
                recent_vol_change = vol_changes[-1] > len(data) - 10

                if recent_vol_break or recent_vol_change:
                    confidence = min(1.0, (len(vol_breaks) + len(vol_changes)) / 10.0)
                    strength = np.mean(np.abs(vol_zscore[vol_breaks])) / 5.0

                    return PatternDetection(
                        pattern_type=PatternType.REGIME_CHANGE,
                        confidence=confidence,
                        strength=min(1.0, strength),
                        timeframe=timeframe,
                        detection_time=datetime.now(),
                        affected_symbols=[symbol],
                        description=f"Regime change detected in {symbol} ({timeframe})",
                        risk_metrics={
                            'volatility_zscore': float(vol_zscore[-1]) if len(vol_zscore) > 0 else 0.0,
                            'volume_ratio': float(vol_ratio.iloc[-1]) if not vol_ratio.empty else 1.0,
                            'breaks_detected': len(vol_breaks) + len(vol_changes)
                        },
                        suggested_actions=[
                            "Reassess position sizes",
                            "Review correlation assumptions",
                            "Monitor for confirmation signals"
                        ]
                    )

            return None

        except Exception as e:
            logger.error(f"Regime change detection failed for {symbol}: {e}")
            return None

    def _detect_volatility_breakout(self, symbol: str, data: pd.DataFrame,
                                  timeframe: str) -> Optional[PatternDetection]:
        """Detect volatility breakout patterns"""
        try:
            if len(data) < 20:
                return None

            returns = data['close'].pct_change()
            current_vol = returns.rolling(5).std().iloc[-1]
            historical_vol = returns.rolling(20).std().mean()

            # Volatility breakout if current vol > 2x historical
            if current_vol > 2 * historical_vol and not np.isnan(current_vol):
                vol_ratio = current_vol / historical_vol
                confidence = min(1.0, vol_ratio / 3.0)

                return PatternDetection(
                    pattern_type=PatternType.VOLATILITY_BREAKOUT,
                    confidence=confidence,
                    strength=min(1.0, vol_ratio / 5.0),
                    timeframe=timeframe,
                    detection_time=datetime.now(),
                    affected_symbols=[symbol],
                    description=f"Volatility breakout in {symbol} ({timeframe})",
                    risk_metrics={
                        'current_volatility': float(current_vol),
                        'historical_volatility': float(historical_vol),
                        'volatility_ratio': float(vol_ratio)
                    },
                    suggested_actions=[
                        "Reduce position size",
                        "Implement volatility-adjusted stops",
                        "Consider volatility hedging"
                    ]
                )

            return None

        except Exception as e:
            logger.error(f"Volatility breakout detection failed for {symbol}: {e}")
            return None

    def _detect_momentum_divergence(self, symbol: str, data: pd.DataFrame,
                                  timeframe: str) -> Optional[PatternDetection]:
        """Detect momentum divergence patterns"""
        try:
            if len(data) < 20:
                return None

            # Price momentum
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

            # Volume momentum
            recent_volume = data['volume'].tail(5).mean()
            historical_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0

            # RSI for momentum confirmation
            rsi = self._calculate_rsi(data['close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # Divergence: price up but volume/momentum weak
            bullish_divergence = price_change > 0.02 and volume_ratio < 0.8 and current_rsi < 30
            bearish_divergence = price_change < -0.02 and volume_ratio < 0.8 and current_rsi > 70

            if bullish_divergence or bearish_divergence:
                divergence_type = "bullish" if bullish_divergence else "bearish"
                confidence = min(1.0, abs(price_change) * 10 * (1 / volume_ratio))

                return PatternDetection(
                    pattern_type=PatternType.MOMENTUM_DIVERGENCE,
                    confidence=confidence,
                    strength=min(1.0, abs(price_change) * 5),
                    timeframe=timeframe,
                    detection_time=datetime.now(),
                    affected_symbols=[symbol],
                    description=f"{divergence_type.title()} momentum divergence in {symbol} ({timeframe})",
                    risk_metrics={
                        'price_change': float(price_change),
                        'volume_ratio': float(volume_ratio),
                        'rsi': float(current_rsi),
                        'divergence_type': divergence_type
                    },
                    suggested_actions=[
                        f"Monitor for {divergence_type} reversal",
                        "Consider contrarian positioning",
                        "Validate with additional indicators"
                    ]
                )

            return None

        except Exception as e:
            logger.error(f"Momentum divergence detection failed for {symbol}: {e}")
            return None

    def _detect_volume_anomaly(self, symbol: str, data: pd.DataFrame,
                             timeframe: str) -> Optional[PatternDetection]:
        """Detect volume anomaly patterns"""
        try:
            if len(data) < 20:
                return None

            # Current vs historical volume
            current_volume = data['volume'].tail(3).mean()
            historical_volume = data['volume'].rolling(20).mean().iloc[-1]

            if historical_volume == 0:
                return None

            volume_ratio = current_volume / historical_volume
            volume_zscore = zscore(data['volume'])[-1] if len(data) > 10 else 0

            # Volume anomaly thresholds
            high_volume_anomaly = volume_ratio > 3.0 and volume_zscore > 2.0
            low_volume_anomaly = volume_ratio < 0.3 and volume_zscore < -2.0

            if high_volume_anomaly or low_volume_anomaly:
                anomaly_type = "high" if high_volume_anomaly else "low"
                confidence = min(1.0, abs(volume_zscore) / 3.0)

                return PatternDetection(
                    pattern_type=PatternType.VOLUME_ANOMALY,
                    confidence=confidence,
                    strength=min(1.0, abs(volume_zscore) / 3.0),
                    timeframe=timeframe,
                    detection_time=datetime.now(),
                    affected_symbols=[symbol],
                    description=f"{anomaly_type.title()} volume anomaly in {symbol} ({timeframe})",
                    risk_metrics={
                        'volume_ratio': float(volume_ratio),
                        'volume_zscore': float(volume_zscore),
                        'anomaly_type': anomaly_type
                    },
                    suggested_actions=[
                        f"Investigate {anomaly_type} volume cause",
                        "Monitor for price confirmation",
                        "Adjust execution strategy"
                    ]
                )

            return None

        except Exception as e:
            logger.error(f"Volume anomaly detection failed for {symbol}: {e}")
            return None

    def _detect_dpi_divergence(self, symbol: str, data: pd.DataFrame,
                             timeframe: str, dpi_data: Dict[str, Any]) -> Optional[PatternDetection]:
        """Detect DPI divergence patterns"""
        try:
            current_dpi = dpi_data.get('dpi_score', 0)
            dpi_confidence = dpi_data.get('confidence', 0.5)

            # Price action
            price_change = data['close'].pct_change(5).iloc[-1]

            # DPI-Price divergence
            dpi_bullish = current_dpi > 0.3
            dpi_bearish = current_dpi < -0.3
            price_bullish = price_change > 0.01
            price_bearish = price_change < -0.01

            # Divergence conditions
            bullish_divergence = dpi_bullish and price_bearish
            bearish_divergence = dpi_bearish and price_bullish

            if bullish_divergence or bearish_divergence and dpi_confidence > 0.6:
                divergence_type = "bullish" if bullish_divergence else "bearish"
                confidence = min(1.0, dpi_confidence * abs(current_dpi))

                return PatternDetection(
                    pattern_type=PatternType.REGIME_CHANGE,  # DPI divergence suggests regime change
                    confidence=confidence,
                    strength=min(1.0, abs(current_dpi)),
                    timeframe=timeframe,
                    detection_time=datetime.now(),
                    affected_symbols=[symbol],
                    description=f"DPI-Price divergence in {symbol} ({timeframe}): {divergence_type}",
                    risk_metrics={
                        'dpi_score': float(current_dpi),
                        'price_change': float(price_change),
                        'dpi_confidence': float(dpi_confidence),
                        'divergence_type': divergence_type
                    },
                    suggested_actions=[
                        "Consider contrarian positioning",
                        "Monitor DPI trend continuation",
                        "Validate with volume analysis"
                    ]
                )

            return None

        except Exception as e:
            logger.error(f"DPI divergence detection failed for {symbol}: {e}")
            return None

    def _filter_and_rank_patterns(self, patterns: List[PatternDetection]) -> List[PatternDetection]:
        """Filter and rank detected patterns by importance"""
        if not patterns:
            return []

        # Filter low confidence patterns
        filtered = [p for p in patterns if p.confidence > 0.3]

        # Rank by composite score (confidence * strength)
        ranked = sorted(filtered, key=lambda p: p.confidence * p.strength, reverse=True)

        # Take top patterns to avoid noise
        return ranked[:5]

    def _extract_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract features for regime classification"""
        features = {}

        try:
            # Aggregate features across all symbols and timeframes
            all_returns = []
            all_volumes = []
            volatilities = []

            for symbol, timeframe_data in market_data.items():
                for timeframe, data in timeframe_data.items():
                    if not data.empty:
                        returns = data['close'].pct_change().dropna()
                        volumes = data['volume']

                        all_returns.extend(returns.tolist())
                        all_volumes.extend(volumes.tolist())
                        volatilities.append(returns.std())

            if all_returns:
                features['mean_return'] = np.mean(all_returns)
                features['return_volatility'] = np.std(all_returns)
                features['return_skewness'] = pd.Series(all_returns).skew()
                features['return_kurtosis'] = pd.Series(all_returns).kurtosis()

            if all_volumes:
                features['mean_volume'] = np.mean(all_volumes)
                features['volume_volatility'] = np.std(all_volumes)

            if volatilities:
                features['cross_asset_volatility'] = np.mean(volatilities)
                features['volatility_dispersion'] = np.std(volatilities)

            # Default values if no data
            default_features = {
                'mean_return': 0.0,
                'return_volatility': 0.02,
                'return_skewness': 0.0,
                'return_kurtosis': 3.0,
                'mean_volume': 1000000.0,
                'volume_volatility': 500000.0,
                'cross_asset_volatility': 0.02,
                'volatility_dispersion': 0.01
            }

            # Fill missing features
            for key, default_value in default_features.items():
                if key not in features:
                    features[key] = default_value

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                'mean_return': 0.0,
                'return_volatility': 0.02,
                'return_skewness': 0.0,
                'return_kurtosis': 3.0,
                'mean_volume': 1000000.0,
                'volume_volatility': 500000.0,
                'cross_asset_volatility': 0.02,
                'volatility_dispersion': 0.01
            }

    def _classify_regime(self, features: Dict[str, float]) -> MarketRegime:
        """Classify market regime based on features"""
        try:
            # Simple rule-based classification
            vol = features['return_volatility']
            ret = features['mean_return']
            skew = features['return_skewness']

            # High volatility regimes
            if vol > 0.05:  # 5% daily volatility
                if skew < -0.5:
                    return MarketRegime.CRISIS
                else:
                    return MarketRegime.HIGH_VOLATILITY

            # Trend regimes
            if abs(ret) > 0.001:  # 0.1% daily return
                if ret > 0:
                    return MarketRegime.BULL_TREND
                else:
                    return MarketRegime.BEAR_TREND

            # Low volatility
            if vol < 0.01:
                return MarketRegime.LOW_VOLATILITY

            # Default
            return MarketRegime.RANGE_BOUND

        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return MarketRegime.TRANSITION

    def _calculate_regime_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in regime classification"""
        try:
            # Higher confidence for extreme values
            vol_extreme = min(1.0, abs(features['return_volatility'] - 0.02) / 0.03)
            ret_extreme = min(1.0, abs(features['mean_return']) / 0.002)
            skew_extreme = min(1.0, abs(features['return_skewness']) / 1.0)

            confidence = (vol_extreme + ret_extreme + skew_extreme) / 3.0
            return max(0.3, min(1.0, confidence))  # Keep between 0.3 and 1.0

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _estimate_regime_duration(self, features: Dict[str, float]) -> int:
        """Estimate regime duration in days"""
        try:
            # Simple heuristic based on volatility
            vol = features['return_volatility']

            if vol > 0.05:
                return 5  # Crisis regimes are short
            elif vol > 0.03:
                return 15  # High vol regimes
            else:
                return 30  # Stable regimes last longer

        except Exception as e:
            logger.error(f"Duration estimation failed: {e}")
            return 10

    def _calculate_transition_probability(self, features: Dict[str, float]) -> float:
        """Calculate probability of regime transition"""
        try:
            # Higher volatility = higher transition probability
            vol_factor = min(1.0, features['return_volatility'] / 0.04)

            # Extreme skewness indicates potential transition
            skew_factor = min(1.0, abs(features['return_skewness']) / 1.0)

            transition_prob = (vol_factor + skew_factor) / 2.0
            return max(0.1, min(0.9, transition_prob))

        except Exception as e:
            logger.error(f"Transition probability calculation failed: {e}")
            return 0.3

    def _predict_next_regime(self, features: Dict[str, float]) -> MarketRegime:
        """Predict next likely regime"""
        try:
            current_regime = self._classify_regime(features)

            # Transition patterns
            transitions = {
                MarketRegime.BULL_TREND: MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BEAR_TREND: MarketRegime.CRISIS,
                MarketRegime.HIGH_VOLATILITY: MarketRegime.RANGE_BOUND,
                MarketRegime.LOW_VOLATILITY: MarketRegime.BULL_TREND,
                MarketRegime.RANGE_BOUND: MarketRegime.LOW_VOLATILITY,
                MarketRegime.CRISIS: MarketRegime.BEAR_TREND,
                MarketRegime.TRANSITION: MarketRegime.RANGE_BOUND
            }

            return transitions.get(current_regime, MarketRegime.TRANSITION)

        except Exception as e:
            logger.error(f"Next regime prediction failed: {e}")
            return MarketRegime.TRANSITION

    def _extract_regime_characteristics(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract characteristics of current regime"""
        return {
            'volatility_level': features['return_volatility'],
            'return_bias': features['mean_return'],
            'skewness': features['return_skewness'],
            'tail_risk': features['return_kurtosis'],
            'volume_stress': features['volume_volatility'] / features['mean_volume']
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(prices), index=prices.index)

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns"""
        if not self.detected_patterns:
            return {'total_patterns': 0}

        pattern_types = {}
        total_confidence = 0
        high_risk_patterns = 0

        for pattern in self.detected_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            total_confidence += pattern.confidence

            if pattern.confidence > 0.8:
                high_risk_patterns += 1

        return {
            'total_patterns': len(self.detected_patterns),
            'pattern_types': pattern_types,
            'average_confidence': total_confidence / len(self.detected_patterns),
            'high_risk_patterns': high_risk_patterns,
            'pattern_distribution': pattern_types
        }