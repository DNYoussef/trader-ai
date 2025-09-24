"""
FinGPT Price Movement Forecaster
Uses FinGPT-Forecaster model for stock price movement prediction based on news and fundamentals
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import FinGPT forecaster components
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    FINGPT_FORECASTER_AVAILABLE = True
    logger.info("FinGPT Forecaster components imported")
except ImportError as e:
    FINGPT_FORECASTER_AVAILABLE = False
    logger.warning(f"FinGPT Forecaster not available: {e}. Using fallback.")


@dataclass
class PriceMovementForecast:
    """Price movement prediction result"""
    symbol: str
    timestamp: datetime
    movement_direction: str  # 'up', 'down', 'sideways'
    probability: float  # 0-1 confidence in direction
    expected_return: float  # Expected return magnitude
    forecast_horizon_days: int
    reasoning: str  # LLM-generated reasoning
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class FundamentalSignal:
    """Fundamental analysis signal from FinGPT"""
    symbol: str
    signal_type: str  # 'earnings_beat', 'guidance_raise', 'insider_buy', etc.
    strength: float  # 0-1
    description: str
    impact_estimate: float  # Expected price impact


class FinGPTForecaster:
    """
    FinGPT-based price movement forecaster
    Analyzes news, fundamentals, and market data for price predictions
    """

    def __init__(self,
                 model_name: str = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
                 use_fallback: bool = True):
        """
        Initialize FinGPT forecaster

        Args:
            model_name: HuggingFace model name for FinGPT-Forecaster
            use_fallback: Use statistical fallback if model unavailable
        """
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

        # Movement indicators for fallback
        self.bullish_indicators = {
            'earnings_beat': 0.03,  # +3% expected
            'revenue_beat': 0.02,
            'guidance_raise': 0.04,
            'analyst_upgrade': 0.02,
            'insider_buying': 0.015,
            'positive_momentum': 0.01
        }

        self.bearish_indicators = {
            'earnings_miss': -0.03,
            'revenue_miss': -0.02,
            'guidance_lower': -0.04,
            'analyst_downgrade': -0.02,
            'insider_selling': -0.015,
            'negative_momentum': -0.01
        }

        # Initialize
        if FINGPT_FORECASTER_AVAILABLE:
            self._initialize_fingpt_forecaster()
        elif use_fallback:
            logger.info("Using fallback forecasting (statistical)")
            self.is_initialized = True
        else:
            raise RuntimeError("FinGPT Forecaster not available and fallback disabled")

    def _initialize_fingpt_forecaster(self):
        """Initialize FinGPT-Forecaster model"""
        try:
            logger.info(f"Loading FinGPT-Forecaster: {self.model_name}")

            # Get HuggingFace token from environment
            hf_token = os.getenv('HF_TOKEN')

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            self.is_initialized = True
            logger.info("FinGPT-Forecaster loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load FinGPT-Forecaster: {e}")
            if self.use_fallback:
                logger.info("Falling back to statistical forecasting")
                self.is_initialized = True
            else:
                raise

    def forecast_price_movement(self,
                                symbol: str,
                                news_articles: List[str],
                                price_history: np.ndarray,
                                fundamental_data: Optional[Dict[str, Any]] = None,
                                forecast_days: int = 5) -> PriceMovementForecast:
        """
        Forecast price movement using FinGPT

        Args:
            symbol: Trading symbol
            news_articles: Recent news articles
            price_history: Historical price data
            fundamental_data: Optional fundamental metrics
            forecast_days: Forecast horizon in days

        Returns:
            PriceMovementForecast with direction and probability
        """
        if not self.is_initialized:
            raise RuntimeError("Forecaster not initialized")

        if FINGPT_FORECASTER_AVAILABLE and self.model is not None:
            # Use FinGPT model for prediction
            forecast = self._fingpt_forecast(
                symbol, news_articles, price_history, fundamental_data, forecast_days
            )
        else:
            # Fallback statistical forecast
            forecast = self._fallback_forecast(
                symbol, news_articles, price_history, fundamental_data, forecast_days
            )

        return forecast

    def _fingpt_forecast(self,
                        symbol: str,
                        news_articles: List[str],
                        price_history: np.ndarray,
                        fundamental_data: Optional[Dict],
                        forecast_days: int) -> PriceMovementForecast:
        """Generate forecast using FinGPT model"""
        # Prepare prompt for FinGPT-Forecaster
        news_summary = ' | '.join(news_articles[:5])  # Top 5 news items
        price_trend = self._calculate_price_trend(price_history)

        # Build FinGPT prompt
        prompt = f"""Analyze the following information and predict the price movement for {symbol} over the next {forecast_days} days:

Recent News: {news_summary}

Price Trend: {price_trend}

Fundamental Data: {fundamental_data if fundamental_data else 'Not available'}

Prediction (up/down/sideways with reasoning):"""

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Extract generated part

        # Parse response
        direction, probability, reasoning = self._parse_fingpt_response(response)

        # Estimate expected return
        expected_return = self._estimate_return_from_direction(
            direction, probability, price_history
        )

        return PriceMovementForecast(
            symbol=symbol,
            timestamp=datetime.now(),
            movement_direction=direction,
            probability=probability,
            expected_return=expected_return,
            forecast_horizon_days=forecast_days,
            reasoning=reasoning,
            confidence=probability,
            metadata={'model': 'FinGPT-Forecaster', 'response': response}
        )

    def _fallback_forecast(self,
                          symbol: str,
                          news_articles: List[str],
                          price_history: np.ndarray,
                          fundamental_data: Optional[Dict],
                          forecast_days: int) -> PriceMovementForecast:
        """Fallback statistical forecast"""
        # Calculate momentum
        returns = np.diff(price_history) / price_history[:-1]
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0

        # Extract signals from news (simple keyword matching)
        bullish_score = sum(
            1 for article in news_articles
            for keyword in ['beat', 'surge', 'rally', 'up', 'gain']
            if keyword in article.lower()
        )
        bearish_score = sum(
            1 for article in news_articles
            for keyword in ['miss', 'drop', 'fall', 'down', 'loss']
            if keyword in article.lower()
        )

        # Combine signals
        sentiment_signal = (bullish_score - bearish_score) / max(len(news_articles), 1)
        combined_signal = 0.6 * momentum + 0.4 * sentiment_signal

        # Determine direction and probability
        if combined_signal > 0.02:
            direction = 'up'
            probability = min(0.5 + abs(combined_signal) * 2, 0.85)
            expected_return = combined_signal * forecast_days
        elif combined_signal < -0.02:
            direction = 'down'
            probability = min(0.5 + abs(combined_signal) * 2, 0.85)
            expected_return = combined_signal * forecast_days
        else:
            direction = 'sideways'
            probability = 0.6
            expected_return = 0.0

        reasoning = f"Momentum: {momentum:.3f}, Sentiment: {sentiment_signal:.2f}, Combined signal: {combined_signal:.3f}"

        return PriceMovementForecast(
            symbol=symbol,
            timestamp=datetime.now(),
            movement_direction=direction,
            probability=float(probability),
            expected_return=float(expected_return),
            forecast_horizon_days=forecast_days,
            reasoning=reasoning,
            confidence=float(probability),
            metadata={'method': 'fallback_statistical'}
        )

    def _calculate_price_trend(self, price_history: np.ndarray) -> str:
        """Calculate price trend description"""
        if len(price_history) < 5:
            return "Insufficient data"

        returns = np.diff(price_history[-5:]) / price_history[-6:-1]
        avg_return = np.mean(returns)

        if avg_return > 0.01:
            return f"Strong uptrend (+{avg_return:.1%})"
        elif avg_return > 0.003:
            return f"Moderate uptrend (+{avg_return:.1%})"
        elif avg_return < -0.01:
            return f"Strong downtrend ({avg_return:.1%})"
        elif avg_return < -0.003:
            return f"Moderate downtrend ({avg_return:.1%})"
        else:
            return f"Sideways ({avg_return:.1%})"

    def _parse_fingpt_response(self, response: str) -> Tuple[str, float, str]:
        """Parse FinGPT-generated response"""
        response_lower = response.lower()

        # Extract direction
        if 'up' in response_lower or 'increase' in response_lower or 'bullish' in response_lower:
            direction = 'up'
        elif 'down' in response_lower or 'decrease' in response_lower or 'bearish' in response_lower:
            direction = 'down'
        else:
            direction = 'sideways'

        # Extract probability (look for percentages or confidence words)
        import re
        prob_match = re.search(r'(\d{1,3})%', response)
        if prob_match:
            probability = float(prob_match.group(1)) / 100
        elif 'high confidence' in response_lower or 'strong' in response_lower:
            probability = 0.75
        elif 'low confidence' in response_lower or 'weak' in response_lower:
            probability = 0.55
        else:
            probability = 0.65

        # Use response as reasoning (truncate if too long)
        reasoning = response[:200] + '...' if len(response) > 200 else response

        return direction, probability, reasoning

    def _estimate_return_from_direction(self,
                                       direction: str,
                                       probability: float,
                                       price_history: np.ndarray) -> float:
        """Estimate expected return from direction and probability"""
        # Calculate historical volatility
        returns = np.diff(price_history) / price_history[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02

        # Estimate return magnitude
        if direction == 'up':
            expected_return = probability * volatility * 2.0  # Bullish bias
        elif direction == 'down':
            expected_return = -probability * volatility * 2.0  # Bearish bias
        else:
            expected_return = 0.0

        return float(expected_return)


if __name__ == "__main__":
    # Test FinGPT forecaster
    print("=== Testing FinGPT Forecaster ===")

    # Sample data
    news_articles = [
        "Company beats earnings estimates, raises guidance for next quarter",
        "Analyst upgrades stock to buy with $500 price target",
        "Strong demand drives revenue growth above expectations"
    ]

    price_history = np.array([380, 385, 390, 392, 395, 400, 398, 402, 405])

    fundamental_data = {
        'pe_ratio': 22.5,
        'earnings_growth': 0.15,
        'revenue_growth': 0.12
    }

    # Initialize forecaster
    forecaster = FinGPTForecaster(use_fallback=True)

    # Generate forecast
    print("\n1. Forecasting price movement...")
    forecast = forecaster.forecast_price_movement(
        symbol='SPY',
        news_articles=news_articles,
        price_history=price_history,
        fundamental_data=fundamental_data,
        forecast_days=5
    )

    print(f"   Symbol: {forecast.symbol}")
    print(f"   Direction: {forecast.movement_direction}")
    print(f"   Probability: {forecast.probability:.1%}")
    print(f"   Expected return: {forecast.expected_return:+.2%}")
    print(f"   Forecast horizon: {forecast.forecast_horizon_days} days")
    print(f"   Reasoning: {forecast.reasoning}")
    print(f"   Confidence: {forecast.confidence:.2f}")

    print("\n=== FinGPT Forecaster Test Complete ===")