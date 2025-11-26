"""
FinGPT Sentiment Analysis Integration
Financial sentiment analysis using FinGPT models for news and social media
"""

import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import transformers for FinGPT
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    FINGPT_AVAILABLE = True
    logger.info("FinGPT/Transformers successfully imported")
except ImportError as e:
    FINGPT_AVAILABLE = False
    logger.warning(f"FinGPT/Transformers not available: {e}. Using fallback sentiment.")


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    label: str  # 'bullish', 'neutral', 'bearish'
    magnitude: float  # Strength of sentiment
    metadata: Dict[str, Any]


@dataclass
class MarketSentiment:
    """Aggregated market sentiment analysis"""
    symbol: str
    timestamp: datetime
    average_sentiment: float  # -1 to +1
    sentiment_volatility: float  # Standard deviation of sentiment
    bullish_ratio: float  # % of bullish signals
    bearish_ratio: float  # % of bearish signals
    neutral_ratio: float  # % of neutral signals
    confidence: float  # Overall confidence
    news_count: int
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    key_themes: List[str]  # Extracted themes/topics


@dataclass
class NarrativeGap:
    """Sentiment-price divergence (narrative gap)"""
    symbol: str
    sentiment_score: float  # Sentiment direction
    price_momentum: float  # Price direction
    gap_magnitude: float  # |sentiment - price_momentum|
    gap_direction: str  # 'bullish_divergence', 'bearish_divergence', 'aligned'
    signal_strength: float  # 0-1, strength of divergence signal
    description: str


class FinGPTSentimentAnalyzer:
    """
    Financial sentiment analysis using FinGPT models
    Processes news, social media, and financial text for sentiment signals
    """

    def __init__(self,
                 model_name: str = "ProsusAI/finbert",
                 use_fallback: bool = True):
        """
        Initialize FinGPT sentiment analyzer

        Args:
            model_name: HuggingFace model name (finbert, fingpt variants)
            use_fallback: Use simple fallback if models unavailable
        """
        self.model_name = model_name
        self.use_fallback = use_fallback
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_initialized = False

        # Sentiment keywords for fallback
        self.bullish_keywords = [
            'surge', 'rally', 'gains', 'up', 'rise', 'growth', 'strong', 'beat',
            'outperform', 'positive', 'bullish', 'buy', 'upgrade', 'momentum'
        ]

        self.bearish_keywords = [
            'crash', 'drop', 'fall', 'decline', 'down', 'weak', 'miss', 'underperform',
            'negative', 'bearish', 'sell', 'downgrade', 'risk', 'concern', 'loss'
        ]

        # Initialize model
        if FINGPT_AVAILABLE:
            self._initialize_fingpt()
        elif use_fallback:
            logger.info("Initializing fallback sentiment analysis")
            self.is_initialized = True
        else:
            raise RuntimeError("FinGPT not available and fallback disabled")

    def _initialize_fingpt(self):
        """Initialize FinGPT/FinBERT model"""
        try:
            logger.info(f"Loading FinGPT model: {self.model_name}")

            # Get HuggingFace token from environment
            hf_token = os.getenv('HF_TOKEN')

            # Load sentiment analysis model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, token=hf_token)

            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            self.is_initialized = True
            logger.info(f"FinGPT model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize FinGPT: {e}")
            if self.use_fallback:
                logger.info("Falling back to keyword-based sentiment")
                self.is_initialized = True
            else:
                raise

    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of single text

        Args:
            text: Financial text (news headline, tweet, etc.)

        Returns:
            SentimentScore with sentiment analysis
        """
        if not self.is_initialized:
            raise RuntimeError("Sentiment analyzer not initialized")

        if FINGPT_AVAILABLE and self.pipeline is not None:
            # Use FinGPT model
            result = self.pipeline(text[:512])[0]  # Truncate to 512 tokens

            # Convert to standardized format
            label = result['label'].lower()
            confidence = result['score']

            # Map to -1 to +1 scale
            if 'positive' in label or 'bullish' in label:
                score = confidence
                sentiment_label = 'bullish'
            elif 'negative' in label or 'bearish' in label:
                score = -confidence
                sentiment_label = 'bearish'
            else:
                score = 0.0
                sentiment_label = 'neutral'

            return SentimentScore(
                score=float(score),
                confidence=float(confidence),
                label=sentiment_label,
                magnitude=abs(score),
                metadata={'model': self.model_name, 'raw_result': result}
            )

        else:
            # Fallback: Keyword-based sentiment
            return self._fallback_sentiment_analysis(text)

    def analyze_market_sentiment(self,
                                 symbol: str,
                                 news_articles: List[str],
                                 lookback_hours: int = 24) -> MarketSentiment:
        """
        Analyze aggregate market sentiment for symbol

        Args:
            symbol: Trading symbol
            news_articles: List of news texts
            lookback_hours: Hours of data to analyze

        Returns:
            MarketSentiment with aggregated analysis
        """
        if not news_articles:
            # No news available
            return MarketSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                average_sentiment=0.0,
                sentiment_volatility=0.0,
                bullish_ratio=0.33,
                bearish_ratio=0.33,
                neutral_ratio=0.34,
                confidence=0.0,
                news_count=0,
                sentiment_trend='stable',
                key_themes=[]
            )

        # Analyze each article
        sentiments = []
        for article in news_articles:
            try:
                sent = self.analyze_text(article)
                sentiments.append(sent)
            except Exception as e:
                logger.warning(f"Failed to analyze article: {e}")

        if not sentiments:
            raise ValueError("No articles could be analyzed")

        # Calculate aggregate metrics
        scores = np.array([s.score for s in sentiments])
        avg_sentiment = float(np.mean(scores))
        sentiment_vol = float(np.std(scores))

        # Count sentiment categories
        bullish_count = sum(1 for s in sentiments if s.label == 'bullish')
        bearish_count = sum(1 for s in sentiments if s.label == 'bearish')
        neutral_count = len(sentiments) - bullish_count - bearish_count

        total = len(sentiments)
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        neutral_ratio = neutral_count / total

        # Detect trend (improving/declining/stable)
        if len(scores) >= 3:
            recent_avg = np.mean(scores[-3:])
            earlier_avg = np.mean(scores[:-3]) if len(scores) > 3 else avg_sentiment

            if recent_avg > earlier_avg + 0.1:
                trend = 'improving'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        # Extract key themes (simple keyword extraction)
        key_themes = self._extract_themes(news_articles)

        # Overall confidence (based on agreement)
        confidence = 1.0 - min(sentiment_vol, 0.8)  # Lower volatility = higher confidence

        return MarketSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            average_sentiment=avg_sentiment,
            sentiment_volatility=sentiment_vol,
            bullish_ratio=float(bullish_ratio),
            bearish_ratio=float(bearish_ratio),
            neutral_ratio=float(neutral_ratio),
            confidence=float(confidence),
            news_count=len(news_articles),
            sentiment_trend=trend,
            key_themes=key_themes
        )

    def detect_narrative_gap(self,
                            symbol: str,
                            sentiment: MarketSentiment,
                            price_returns_5d: float,
                            volume_ratio: float = 1.0) -> NarrativeGap:
        """
        Detect narrative gap (sentiment-price divergence)

        Args:
            symbol: Trading symbol
            sentiment: Market sentiment analysis
            price_returns_5d: 5-day price returns
            volume_ratio: Current volume / average volume

        Returns:
            NarrativeGap with divergence analysis
        """
        # Normalize price momentum to -1 to +1 scale
        price_momentum = np.tanh(price_returns_5d * 10)  # tanh scales to [-1, 1]

        # Calculate gap
        gap = sentiment.average_sentiment - price_momentum
        gap_magnitude = abs(gap)

        # Classify divergence type
        if gap > 0.3:  # Sentiment much more positive than price
            gap_direction = 'bullish_divergence'
            description = f"Sentiment ({sentiment.average_sentiment:.2f}) significantly more bullish than price action ({price_momentum:.2f})"
        elif gap < -0.3:  # Sentiment much more negative than price
            gap_direction = 'bearish_divergence'
            description = f"Sentiment ({sentiment.average_sentiment:.2f}) significantly more bearish than price action ({price_momentum:.2f})"
        else:
            gap_direction = 'aligned'
            description = f"Sentiment ({sentiment.average_sentiment:.2f}) aligned with price action ({price_momentum:.2f})"

        # Calculate signal strength (gap + confidence + volume confirmation)
        base_strength = gap_magnitude
        confidence_boost = sentiment.confidence * 0.3
        volume_boost = min((volume_ratio - 1.0) * 0.2, 0.2) if volume_ratio > 1.0 else 0

        signal_strength = min(base_strength + confidence_boost + volume_boost, 1.0)

        return NarrativeGap(
            symbol=symbol,
            sentiment_score=sentiment.average_sentiment,
            price_momentum=float(price_momentum),
            gap_magnitude=float(gap_magnitude),
            gap_direction=gap_direction,
            signal_strength=float(signal_strength),
            description=description
        )

    def _fallback_sentiment_analysis(self, text: str) -> SentimentScore:
        """Fallback keyword-based sentiment analysis"""
        text_lower = text.lower()

        # Count bullish/bearish keywords
        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)

        total_count = bullish_count + bearish_count

        if total_count == 0:
            return SentimentScore(
                score=0.0,
                confidence=0.3,
                label='neutral',
                magnitude=0.0,
                metadata={'method': 'fallback_keyword', 'keywords_found': 0}
            )

        # Calculate score
        score = (bullish_count - bearish_count) / total_count
        confidence = min(total_count / 5.0, 0.8)  # More keywords = higher confidence

        if score > 0.2:
            label = 'bullish'
        elif score < -0.2:
            label = 'bearish'
        else:
            label = 'neutral'

        return SentimentScore(
            score=float(score),
            confidence=float(confidence),
            label=label,
            magnitude=abs(score),
            metadata={
                'method': 'fallback_keyword',
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            }
        )

    def _extract_themes(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extract key themes from texts (simple approach)"""
        # Combine all texts
        combined = ' '.join(texts).lower()

        # Common financial themes/topics
        theme_keywords = {
            'earnings': ['earnings', 'eps', 'profit', 'revenue'],
            'fed': ['fed', 'federal reserve', 'powell', 'rate', 'inflation'],
            'economy': ['economy', 'gdp', 'unemployment', 'jobs'],
            'tech': ['tech', 'technology', 'ai', 'software'],
            'energy': ['oil', 'energy', 'gas', 'crude'],
            'china': ['china', 'chinese', 'beijing'],
            'crypto': ['crypto', 'bitcoin', 'blockchain'],
            'merger': ['merger', 'acquisition', 'm&a', 'deal'],
            'regulation': ['regulation', 'sec', 'compliance', 'lawsuit']
        }

        # Count theme occurrences
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            count = sum(combined.count(kw) for kw in keywords)
            if count > 0:
                theme_scores[theme] = count

        # Get top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:top_n]]


if __name__ == "__main__":
    # Test FinGPT sentiment analyzer
    print("=== Testing FinGPT Sentiment Analyzer ===")

    # Sample financial news
    news_articles = [
        "Stock market rallies to new highs as tech sector surges",
        "Fed signals rate cuts amid declining inflation concerns",
        "Major bank beats earnings estimates, stock jumps 5%",
        "Market crash fears grow as volatility spikes",
        "Investors cautious amid geopolitical tensions"
    ]

    # Initialize analyzer
    analyzer = FinGPTSentimentAnalyzer(use_fallback=True)

    # Test individual sentiment
    print("\n1. Analyzing individual texts...")
    for article in news_articles[:3]:
        sent = analyzer.analyze_text(article)
        print(f"   '{article[:50]}...'")
        print(f"   Sentiment: {sent.label} ({sent.score:+.2f}), Confidence: {sent.confidence:.2f}")

    # Test market sentiment
    print("\n2. Analyzing aggregate market sentiment...")
    market_sent = analyzer.analyze_market_sentiment('SPY', news_articles)
    print(f"   Symbol: {market_sent.symbol}")
    print(f"   Average sentiment: {market_sent.average_sentiment:+.2f}")
    print(f"   Bullish ratio: {market_sent.bullish_ratio:.1%}")
    print(f"   Bearish ratio: {market_sent.bearish_ratio:.1%}")
    print(f"   Trend: {market_sent.sentiment_trend}")
    print(f"   Key themes: {', '.join(market_sent.key_themes)}")

    # Test narrative gap
    print("\n3. Testing narrative gap detection...")
    gap = analyzer.detect_narrative_gap(
        symbol='SPY',
        sentiment=market_sent,
        price_returns_5d=-0.02,  # -2% price decline
        volume_ratio=1.5  # 50% above average volume
    )
    print(f"   Gap direction: {gap.gap_direction}")
    print(f"   Gap magnitude: {gap.gap_magnitude:.2f}")
    print(f"   Signal strength: {gap.signal_strength:.2f}")
    print(f"   {gap.description}")

    print("\n=== FinGPT Sentiment Analyzer Test Complete ===")