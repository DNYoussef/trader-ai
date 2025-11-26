"""
Enhanced HRM Feature Engineering Layer
Combines TimesFM forecasts + FinGPT sentiment + existing features → 32-dimensional feature vector
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .timesfm_forecaster import TimesFMForecaster
from .fingpt_sentiment import FinGPTSentimentAnalyzer
from .fingpt_forecaster import FinGPTForecaster

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeatureVector:
    """32-dimensional enhanced feature vector for HRM"""
    # Original 24 features (from enhanced_market_state)
    base_features: np.ndarray  # Shape: (24,)

    # New 8 features from TimesFM + FinGPT
    timesfm_vix_1h: float  # VIX forecast 1 hour ahead
    timesfm_vix_6h: float  # VIX forecast 6 hours ahead
    timesfm_vix_24h: float  # VIX forecast 24 hours ahead
    timesfm_price_forecast: float  # Price forecast (point estimate)
    timesfm_uncertainty: float  # Forecast uncertainty (quantile spread)
    fingpt_sentiment: float  # Average sentiment score (-1 to +1)
    fingpt_sentiment_vol: float  # Sentiment volatility
    fingpt_price_prob: float  # Price movement probability

    # Combined features
    combined_features: np.ndarray  # Shape: (32,)

    # Metadata
    timestamp: datetime
    symbol: str
    confidence: float
    metadata: Dict[str, Any]


class EnhancedHRMFeatureEngine:
    """
    Feature engineering layer that combines:
    - Original 24 market features
    - TimesFM volatility forecasts (3 horizons)
    - TimesFM price forecast + uncertainty
    - FinGPT sentiment analysis (3 features)
    - FinGPT price movement probability

    Output: 32-dimensional feature vector for enhanced HRM
    """

    def __init__(self,
                 timesfm_forecaster: Optional[TimesFMForecaster] = None,
                 fingpt_sentiment: Optional[FinGPTSentimentAnalyzer] = None,
                 fingpt_forecaster: Optional[FinGPTForecaster] = None):
        """
        Initialize enhanced feature engine

        Args:
            timesfm_forecaster: TimesFM forecaster instance
            fingpt_sentiment: FinGPT sentiment analyzer
            fingpt_forecaster: FinGPT price forecaster
        """
        # Initialize components (create if not provided)
        self.timesfm = timesfm_forecaster or TimesFMForecaster(use_fallback=True)
        self.fingpt_sentiment = fingpt_sentiment or FinGPTSentimentAnalyzer(use_fallback=True)
        self.fingpt_forecast = fingpt_forecaster or FinGPTForecaster(use_fallback=True)

        logger.info("Enhanced HRM Feature Engine initialized")
        logger.info("  - TimesFM: Multi-horizon forecasting")
        logger.info("  - FinGPT Sentiment: News/social sentiment")
        logger.info("  - FinGPT Forecaster: Price movement prediction")

    def create_enhanced_features(self,
                                 base_market_features: Dict[str, float],
                                 vix_history: np.ndarray,
                                 price_history: np.ndarray,
                                 news_articles: List[str],
                                 symbol: str = 'SPY') -> EnhancedFeatureVector:
        """
        Create 32-dimensional enhanced feature vector

        Args:
            base_market_features: Dict with 24 base features
            vix_history: Historical VIX data (100+ points)
            price_history: Historical price data (100+ points)
            news_articles: Recent news articles for sentiment
            symbol: Trading symbol

        Returns:
            EnhancedFeatureVector with all 32 features
        """
        # Step 1: Extract base 24 features
        base_features = self._extract_base_features(base_market_features)

        # Step 2: Generate TimesFM forecasts
        timesfm_features = self._generate_timesfm_features(vix_history, price_history)

        # Step 3: Generate FinGPT sentiment features
        fingpt_sent_features = self._generate_fingpt_sentiment_features(
            symbol, news_articles, price_history
        )

        # Step 4: Generate FinGPT price forecast
        fingpt_price_prob = self._generate_fingpt_price_features(
            symbol, news_articles, price_history
        )

        # Step 5: Combine all features (24 + 8 = 32)
        enhanced_features = np.concatenate([
            base_features,  # 24 features
            [
                timesfm_features['vix_1h'],
                timesfm_features['vix_6h'],
                timesfm_features['vix_24h'],
                timesfm_features['price_forecast'],
                timesfm_features['uncertainty'],
                fingpt_sent_features['sentiment'],
                fingpt_sent_features['sentiment_vol'],
                fingpt_price_prob
            ]  # 8 new features
        ])

        # Calculate overall confidence
        confidence = np.mean([
            timesfm_features['confidence'],
            fingpt_sent_features['confidence']
        ])

        return EnhancedFeatureVector(
            base_features=base_features,
            timesfm_vix_1h=float(timesfm_features['vix_1h']),
            timesfm_vix_6h=float(timesfm_features['vix_6h']),
            timesfm_vix_24h=float(timesfm_features['vix_24h']),
            timesfm_price_forecast=float(timesfm_features['price_forecast']),
            timesfm_uncertainty=float(timesfm_features['uncertainty']),
            fingpt_sentiment=float(fingpt_sent_features['sentiment']),
            fingpt_sentiment_vol=float(fingpt_sent_features['sentiment_vol']),
            fingpt_price_prob=float(fingpt_price_prob),
            combined_features=enhanced_features,
            timestamp=datetime.now(),
            symbol=symbol,
            confidence=float(confidence),
            metadata={
                'timesfm': timesfm_features,
                'fingpt_sentiment': fingpt_sent_features
            }
        )

    def _extract_base_features(self, market_features: Dict[str, float]) -> np.ndarray:
        """Extract and normalize base 24 features"""
        # Expected base features from enhanced_market_state (exclude 'name')
        feature_keys = sorted([k for k in market_features.keys() if k != 'name'])

        # Extract values in consistent order
        features = [market_features.get(key, 0.0) for key in feature_keys]

        # Ensure exactly 24 features
        if len(features) > 24:
            features = features[:24]
        elif len(features) < 24:
            features.extend([0.0] * (24 - len(features)))

        return np.array(features, dtype=np.float32)

    def _generate_timesfm_features(self,
                                   vix_history: np.ndarray,
                                   price_history: np.ndarray) -> Dict[str, float]:
        """Generate TimesFM forecast features"""
        try:
            # Multi-horizon VIX forecasts
            vix_1h = self.timesfm.forecast_volatility(vix_history, horizon_hours=1)
            vix_6h = self.timesfm.forecast_volatility(vix_history, horizon_hours=6)
            vix_24h = self.timesfm.forecast_volatility(vix_history, horizon_hours=24)

            # Price forecast
            price_forecast = self.timesfm.forecast_price(price_history, horizon=12)

            # Calculate uncertainty from quantile spread
            q95 = price_forecast.quantile_forecasts.get(0.95, price_forecast.point_forecast)
            q05 = price_forecast.quantile_forecasts.get(0.05, price_forecast.point_forecast)
            uncertainty = np.mean(q95 - q05) / np.mean(price_history[-10:])

            return {
                'vix_1h': float(vix_1h.vix_forecast[0]),  # First hour
                'vix_6h': float(vix_6h.vix_forecast[5]),  # 6th hour
                'vix_24h': float(vix_24h.vix_forecast[23]),  # 24th hour
                'price_forecast': float(price_forecast.point_forecast[0]) / price_history[-1] - 1.0,  # % change
                'uncertainty': float(uncertainty),
                'confidence': np.mean([vix_1h.confidence, vix_6h.confidence, vix_24h.confidence])
            }

        except Exception as e:
            logger.warning(f"TimesFM feature generation failed: {e}, using defaults")
            current_vix = float(vix_history[-1]) if len(vix_history) > 0 else 20.0
            return {
                'vix_1h': current_vix,
                'vix_6h': current_vix,
                'vix_24h': current_vix,
                'price_forecast': 0.0,
                'uncertainty': 0.02,
                'confidence': 0.5
            }

    def _generate_fingpt_sentiment_features(self,
                                           symbol: str,
                                           news_articles: List[str],
                                           price_history: np.ndarray) -> Dict[str, float]:
        """Generate FinGPT sentiment features"""
        try:
            if len(news_articles) > 0:
                # Analyze market sentiment
                market_sentiment = self.fingpt_sentiment.analyze_market_sentiment(
                    symbol=symbol,
                    news_articles=news_articles
                )

                return {
                    'sentiment': float(market_sentiment.average_sentiment),
                    'sentiment_vol': float(market_sentiment.sentiment_volatility),
                    'confidence': float(market_sentiment.confidence)
                }
            else:
                # No news available
                return {
                    'sentiment': 0.0,
                    'sentiment_vol': 0.0,
                    'confidence': 0.3
                }

        except Exception as e:
            logger.warning(f"FinGPT sentiment generation failed: {e}, using defaults")
            return {
                'sentiment': 0.0,
                'sentiment_vol': 0.0,
                'confidence': 0.3
            }

    def _generate_fingpt_price_features(self,
                                       symbol: str,
                                       news_articles: List[str],
                                       price_history: np.ndarray) -> float:
        """Generate FinGPT price movement probability"""
        try:
            if len(news_articles) > 0:
                forecast = self.fingpt_forecast.forecast_price_movement(
                    symbol=symbol,
                    news_articles=news_articles,
                    price_history=price_history,
                    forecast_days=5
                )

                # Convert direction + probability to single value
                if forecast.movement_direction == 'up':
                    prob = forecast.probability
                elif forecast.movement_direction == 'down':
                    prob = -forecast.probability
                else:
                    prob = 0.0

                return float(prob)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"FinGPT price feature generation failed: {e}")
            return 0.0

    def batch_create_features(self,
                             scenarios: List[Dict[str, Any]]) -> List[EnhancedFeatureVector]:
        """
        Create features for multiple scenarios (batch processing)

        Args:
            scenarios: List of scenario dicts with all required data

        Returns:
            List of EnhancedFeatureVector
        """
        feature_vectors = []

        for scenario in scenarios:
            try:
                features = self.create_enhanced_features(
                    base_market_features=scenario['base_features'],
                    vix_history=scenario['vix_history'],
                    price_history=scenario['price_history'],
                    news_articles=scenario.get('news_articles', []),
                    symbol=scenario.get('symbol', 'SPY')
                )
                feature_vectors.append(features)

            except Exception as e:
                logger.error(f"Failed to create features for scenario: {e}")

        return feature_vectors

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings (for analysis)"""
        # Base features (24)
        base_importance = {
            'vix_level': 0.95,
            'spy_returns_5d': 0.90,
            'spy_returns_20d': 0.85,
            'put_call_ratio': 0.88,
            'market_breadth': 0.82,
            'correlation': 0.80,
            'volume_ratio': 0.75,
            'gini_coefficient': 0.78,
            'sector_dispersion': 0.76,
            'signal_quality': 0.72
        }

        # New TimesFM features (5)
        timesfm_importance = {
            'timesfm_vix_24h': 0.92,  # High importance: long-range vol forecast
            'timesfm_vix_6h': 0.88,
            'timesfm_vix_1h': 0.85,
            'timesfm_uncertainty': 0.80,  # Uncertainty = risk
            'timesfm_price_forecast': 0.75
        }

        # New FinGPT features (3)
        fingpt_importance = {
            'fingpt_sentiment': 0.87,  # Sentiment breaks before price
            'fingpt_sentiment_vol': 0.82,  # Volatility = uncertainty
            'fingpt_price_prob': 0.78
        }

        # Combine all
        all_importance = {**base_importance, **timesfm_importance, **fingpt_importance}

        return dict(sorted(all_importance.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    # Test enhanced feature engine
    print("=== Testing Enhanced HRM Feature Engine ===")

    # Create synthetic data
    base_market_features = {
        'vix_level': 22.5,
        'spy_returns_5d': -0.02,
        'spy_returns_20d': 0.05,
        'put_call_ratio': 1.2,
        'market_breadth': 0.55,
        'correlation': 0.68,
        'volume_ratio': 1.3,
        'gini_coefficient': 0.48,
        'sector_dispersion': 0.012,
        'signal_quality': 0.65,
        # Add more to reach 24
        **{f'feature_{i}': np.random.rand() for i in range(10, 24)}
    }

    vix_history = np.random.normal(22, 5, 200)
    vix_history = np.clip(vix_history, 10, 50)

    price_history = np.random.normal(400, 10, 200)

    news_articles = [
        "Market shows signs of volatility as uncertainty grows",
        "Fed signals potential rate adjustments in coming months",
        "Tech sector leads market rally with strong earnings"
    ]

    # Initialize engine
    engine = EnhancedHRMFeatureEngine()

    # Create enhanced features
    print("\n1. Creating enhanced 32-dimensional features...")
    features = engine.create_enhanced_features(
        base_market_features=base_market_features,
        vix_history=vix_history,
        price_history=price_history,
        news_articles=news_articles,
        symbol='SPY'
    )

    print(f"   Base features (24): {features.base_features.shape}")
    print(f"   Combined features (32): {features.combined_features.shape}")
    print("\n   TimesFM Features:")
    print(f"     VIX 1h: {features.timesfm_vix_1h:.2f}")
    print(f"     VIX 6h: {features.timesfm_vix_6h:.2f}")
    print(f"     VIX 24h: {features.timesfm_vix_24h:.2f}")
    print(f"     Price forecast: {features.timesfm_price_forecast:+.2%}")
    print(f"     Uncertainty: {features.timesfm_uncertainty:.3f}")
    print("\n   FinGPT Features:")
    print(f"     Sentiment: {features.fingpt_sentiment:+.2f}")
    print(f"     Sentiment vol: {features.fingpt_sentiment_vol:.3f}")
    print(f"     Price probability: {features.fingpt_price_prob:+.2f}")
    print(f"\n   Overall confidence: {features.confidence:.2f}")

    # Show feature importance
    print("\n2. Feature importance rankings (top 10):")
    importance = engine.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
        print(f"   {i}. {feature}: {score:.2f}")

    print("\n=== Enhanced HRM Feature Engine Test Complete ===")