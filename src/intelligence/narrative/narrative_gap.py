"""
Narrative Gap (NG) Engine - Core Alpha Generation Component

This module implements the sophisticated narrative gap detection system that forms
the foundation of the alpha generation strategy. The NG engine identifies gaps
between market-implied paths and distribution-aware paths, generating signals
that can be leveraged for profitable trading.

Core Formula: Alpha = (market-implied path) - (distribution-aware path)

Key Components:
- Market consensus extraction from sell-side notes and media
- Distribution-aware pricing using DFL (Data-First Learning) data
- Time-to-diffusion clocks for narrative spread timing
- NG signal generation with normalized scoring (0-1 scale)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import json
import re
from textblob import TextBlob

logger = logging.getLogger(__name__)

@dataclass
class NarrativeData:
    """Container for narrative information from various sources"""
    source: str
    content: str
    timestamp: datetime
    sentiment_score: float
    confidence: float
    keywords: List[str] = field(default_factory=list)
    market_implications: Dict[str, float] = field(default_factory=dict)

@dataclass
class MarketConsensus:
    """Represents market consensus extracted from various sources"""
    symbol: str
    price_target: float
    time_horizon: int  # days
    confidence: float
    supporting_narratives: List[NarrativeData] = field(default_factory=list)
    consensus_strength: float = 0.0
    narrative_coherence: float = 0.0

@dataclass
class DFLDistribution:
    """Distribution-aware pricing data from DFL system"""
    symbol: str
    expected_path: np.ndarray
    confidence_bands: Tuple[np.ndarray, np.ndarray]  # (lower, upper)
    time_steps: np.ndarray
    distribution_params: Dict[str, float] = field(default_factory=dict)
    regime_probability: float = 1.0

@dataclass
class NGSignal:
    """Narrative Gap signal output"""
    symbol: str
    ng_score: float  # 0-1 scale
    market_path: np.ndarray
    dfl_path: np.ndarray
    gap_magnitude: float
    time_to_diffusion: float
    catalyst_proximity: float
    confidence: float
    timestamp: datetime
    supporting_data: Dict[str, Any] = field(default_factory=dict)

class ConsensusExtractor(ABC):
    """Abstract base class for market consensus extraction"""

    @abstractmethod
    async def extract_consensus(self, symbol: str) -> MarketConsensus:
        """Extract market consensus for given symbol"""
        pass

class SellSideNotesExtractor(ConsensusExtractor):
    """Extracts consensus from sell-side research notes"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.price_target_patterns = [
            r'price target[:\s]+\$?(\d+\.?\d*)',
            r'target price[:\s]+\$?(\d+\.?\d*)',
            r'fair value[:\s]+\$?(\d+\.?\d*)',
            r'valuation[:\s]+\$?(\d+\.?\d*)'
        ]

    async def extract_consensus(self, symbol: str) -> MarketConsensus:
        """Extract consensus from sell-side notes"""
        try:
            # Simulate research note retrieval (replace with actual API)
            notes = await self._fetch_research_notes(symbol)

            price_targets = []
            narratives = []

            for note in notes:
                # Extract price targets
                targets = self._extract_price_targets(note['content'])
                price_targets.extend(targets)

                # Create narrative data
                narrative = NarrativeData(
                    source=note['source'],
                    content=note['content'][:500],  # Truncate for storage
                    timestamp=datetime.fromisoformat(note['timestamp']),
                    sentiment_score=self._analyze_sentiment(note['content']),
                    confidence=note.get('analyst_confidence', 0.7),
                    keywords=self._extract_keywords(note['content'])
                )
                narratives.append(narrative)

            # Calculate consensus
            if price_targets:
                consensus_price = np.mean(price_targets)
                confidence = min(1.0, len(price_targets) / 5.0)  # More targets = higher confidence
            else:
                consensus_price = 0.0
                confidence = 0.0

            consensus = MarketConsensus(
                symbol=symbol,
                price_target=consensus_price,
                time_horizon=30,  # Default 30-day horizon
                confidence=confidence,
                supporting_narratives=narratives
            )

            # Calculate narrative coherence
            consensus.narrative_coherence = self._calculate_narrative_coherence(narratives)
            consensus.consensus_strength = self._calculate_consensus_strength(price_targets)

            return consensus

        except Exception as e:
            logger.error(f"Error extracting consensus for {symbol}: {e}")
            return MarketConsensus(symbol=symbol, price_target=0.0, time_horizon=30, confidence=0.0)

    async def _fetch_research_notes(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch research notes from data providers"""
        # Simulated data - replace with actual API calls
        return [
            {
                'source': 'Goldman Sachs',
                'content': f'Raising {symbol} price target to $120 based on strong fundamentals and market expansion',
                'timestamp': '2024-09-14T10:00:00',
                'analyst_confidence': 0.85
            },
            {
                'source': 'Morgan Stanley',
                'content': f'Maintaining {symbol} at $115 fair value with positive outlook on sector trends',
                'timestamp': '2024-09-13T14:30:00',
                'analyst_confidence': 0.75
            }
        ]

    def _extract_price_targets(self, text: str) -> List[float]:
        """Extract numerical price targets from text"""
        targets = []
        for pattern in self.price_target_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    targets.append(float(match))
                except ValueError:
                    continue
        return targets

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content"""
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Returns -1 to 1

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        blob = TextBlob(text)
        # Simple keyword extraction - can be enhanced with NLP libraries
        keywords = [word.lower() for word in blob.words if len(word) > 4]
        return list(set(keywords))[:10]  # Top 10 unique keywords

    def _calculate_narrative_coherence(self, narratives: List[NarrativeData]) -> float:
        """Calculate how coherent the narratives are"""
        if len(narratives) < 2:
            return 1.0

        sentiments = [n.sentiment_score for n in narratives]
        sentiment_std = np.std(sentiments)

        # Lower standard deviation = higher coherence
        coherence = max(0.0, 1.0 - sentiment_std)
        return coherence

    def _calculate_consensus_strength(self, targets: List[float]) -> float:
        """Calculate strength of price target consensus"""
        if len(targets) < 2:
            return 0.5

        cv = np.std(targets) / np.mean(targets) if np.mean(targets) > 0 else 1.0
        strength = max(0.0, 1.0 - cv)  # Lower coefficient of variation = higher strength
        return strength

class MediaSentimentExtractor(ConsensusExtractor):
    """Extracts consensus from media sentiment analysis"""

    def __init__(self):
        self.sentiment_keywords = {
            'bullish': ['bullish', 'optimistic', 'positive', 'growth', 'strong', 'buy'],
            'bearish': ['bearish', 'pessimistic', 'negative', 'decline', 'weak', 'sell'],
            'neutral': ['neutral', 'stable', 'maintain', 'hold', 'steady']
        }

    async def extract_consensus(self, symbol: str) -> MarketConsensus:
        """Extract consensus from media sentiment"""
        try:
            # Simulate media data retrieval
            articles = await self._fetch_media_articles(symbol)

            narratives = []
            sentiment_scores = []

            for article in articles:
                sentiment = self._analyze_article_sentiment(article)
                sentiment_scores.append(sentiment)

                narrative = NarrativeData(
                    source=article['source'],
                    content=article['title'] + ': ' + article['summary'][:300],
                    timestamp=datetime.fromisoformat(article['timestamp']),
                    sentiment_score=sentiment,
                    confidence=article.get('credibility_score', 0.6),
                    keywords=self._extract_media_keywords(article['content'])
                )
                narratives.append(narrative)

            # Convert sentiment to implied price movement
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            implied_price_change = avg_sentiment * 0.1  # 10% max movement from sentiment

            consensus = MarketConsensus(
                symbol=symbol,
                price_target=100 * (1 + implied_price_change),  # Assume $100 base
                time_horizon=7,  # Media sentiment is short-term
                confidence=min(1.0, len(sentiment_scores) / 10.0),
                supporting_narratives=narratives
            )

            return consensus

        except Exception as e:
            logger.error(f"Error extracting media consensus for {symbol}: {e}")
            return MarketConsensus(symbol=symbol, price_target=100.0, time_horizon=7, confidence=0.0)

    async def _fetch_media_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch recent media articles"""
        # Simulated data
        return [
            {
                'source': 'Financial Times',
                'title': f'{symbol} Shows Strong Q3 Performance',
                'summary': 'Company demonstrates resilient growth despite market headwinds',
                'content': 'Full article content here...',
                'timestamp': '2024-09-14T08:00:00',
                'credibility_score': 0.9
            }
        ]

    def _analyze_article_sentiment(self, article: Dict[str, Any]) -> float:
        """Analyze sentiment of media article"""
        text = article['title'] + ' ' + article.get('summary', '')
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def _extract_media_keywords(self, text: str) -> List[str]:
        """Extract keywords from media content"""
        # Enhanced keyword extraction for financial content
        financial_terms = ['earnings', 'revenue', 'profit', 'growth', 'margin', 'guidance']
        words = text.lower().split()
        keywords = [word for word in words if word in financial_terms or len(word) > 6]
        return list(set(keywords))[:8]

class DFLPricingEngine:
    """Distribution-First Learning pricing engine"""

    def __init__(self):
        self.model_params = {
            'volatility_decay': 0.94,
            'mean_reversion_speed': 0.15,
            'jump_intensity': 0.02
        }

    def generate_dfl_distribution(self, symbol: str, current_price: float,
                                horizon_days: int) -> DFLDistribution:
        """Generate distribution-aware price path"""
        try:
            # Time steps
            dt = 1.0 / 252  # Daily steps in trading year
            time_steps = np.linspace(0, horizon_days * dt, horizon_days + 1)

            # Generate expected path using enhanced stochastic model
            expected_path = self._generate_expected_path(current_price, time_steps)

            # Calculate confidence bands
            volatility = self._estimate_volatility(symbol)
            lower_band, upper_band = self._calculate_confidence_bands(
                expected_path, volatility, time_steps
            )

            # Distribution parameters
            dist_params = {
                'volatility': volatility,
                'skewness': self._estimate_skewness(symbol),
                'kurtosis': self._estimate_kurtosis(symbol),
                'mean_reversion': self.model_params['mean_reversion_speed']
            }

            return DFLDistribution(
                symbol=symbol,
                expected_path=expected_path,
                confidence_bands=(lower_band, upper_band),
                time_steps=time_steps,
                distribution_params=dist_params,
                regime_probability=self._estimate_regime_probability(symbol)
            )

        except Exception as e:
            logger.error(f"Error generating DFL distribution for {symbol}: {e}")
            # Return fallback distribution
            time_steps = np.linspace(0, horizon_days / 252, horizon_days + 1)
            flat_path = np.full_like(time_steps, current_price)
            return DFLDistribution(
                symbol=symbol,
                expected_path=flat_path,
                confidence_bands=(flat_path * 0.9, flat_path * 1.1),
                time_steps=time_steps
            )

    def _generate_expected_path(self, S0: float, time_steps: np.ndarray) -> np.ndarray:
        """Generate expected price path using sophisticated model"""
        n_steps = len(time_steps)
        path = np.zeros(n_steps)
        path[0] = S0

        # Model parameters
        mu = 0.08  # Expected annual return
        sigma = 0.2  # Base volatility
        dt = time_steps[1] - time_steps[0] if len(time_steps) > 1 else 1/252

        for i in range(1, n_steps):
            # Mean-reverting component
            mean_level = S0 * np.exp(mu * time_steps[i])
            reversion = self.model_params['mean_reversion_speed'] * (mean_level - path[i-1])

            # Volatility decay
            vol_t = sigma * (self.model_params['volatility_decay'] ** time_steps[i])

            # Random component
            dW = np.random.normal(0, np.sqrt(dt))

            # Jump component
            jump = 0
            if np.random.random() < self.model_params['jump_intensity'] * dt:
                jump = np.random.normal(0, 0.05) * path[i-1]

            # Path evolution
            path[i] = path[i-1] + reversion * dt + vol_t * path[i-1] * dW + jump

        return path

    def _calculate_confidence_bands(self, expected_path: np.ndarray,
                                  volatility: float, time_steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence bands around expected path"""
        # 95% confidence bands
        z_score = 1.96

        # Time-varying volatility
        vol_scaling = np.sqrt(time_steps)
        lower_band = expected_path * (1 - z_score * volatility * vol_scaling)
        upper_band = expected_path * (1 + z_score * volatility * vol_scaling)

        return lower_band, upper_band

    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate current volatility for symbol"""
        # Placeholder - would use historical data or volatility models
        base_vol = 0.25
        return base_vol * np.random.uniform(0.8, 1.2)

    def _estimate_skewness(self, symbol: str) -> float:
        """Estimate return distribution skewness"""
        return np.random.uniform(-0.5, 0.2)  # Typically negative for equities

    def _estimate_kurtosis(self, symbol: str) -> float:
        """Estimate return distribution kurtosis"""
        return np.random.uniform(3.5, 6.0)  # Excess kurtosis typical for financial data

    def _estimate_regime_probability(self, symbol: str) -> float:
        """Estimate probability of current market regime persistence"""
        return np.random.uniform(0.7, 0.95)

class NarrativeGapEngine:
    """Main engine for calculating narrative gaps and generating alpha signals"""

    def __init__(self):
        self.consensus_extractors = [
            SellSideNotesExtractor(),
            MediaSentimentExtractor()
        ]
        self.dfl_engine = DFLPricingEngine()
        self.scaler = MinMaxScaler()

        # Signal parameters
        self.signal_params = {
            'gap_weight': 0.4,
            'consensus_weight': 0.3,
            'catalyst_weight': 0.2,
            'coherence_weight': 0.1
        }

    async def calculate_narrative_gap(self, symbol: str, current_price: float,
                                    horizon_days: int = 30) -> NGSignal:
        """Calculate narrative gap signal for given symbol"""
        try:
            # Extract market consensus from multiple sources
            consensus_tasks = [
                extractor.extract_consensus(symbol)
                for extractor in self.consensus_extractors
            ]
            consensus_results = await asyncio.gather(*consensus_tasks)

            # Aggregate consensus
            aggregated_consensus = self._aggregate_consensus(consensus_results, symbol)

            # Generate DFL distribution
            dfl_distribution = self.dfl_engine.generate_dfl_distribution(
                symbol, current_price, horizon_days
            )

            # Calculate market-implied path
            market_path = self._calculate_market_implied_path(
                current_price, aggregated_consensus, horizon_days
            )

            # Calculate narrative gap
            gap_magnitude = self._calculate_gap_magnitude(market_path, dfl_distribution.expected_path)

            # Calculate time to diffusion
            time_to_diffusion = self._calculate_time_to_diffusion(aggregated_consensus)

            # Calculate catalyst proximity
            catalyst_proximity = self._calculate_catalyst_proximity(symbol, aggregated_consensus)

            # Generate final NG score
            ng_score = self._calculate_ng_score(
                gap_magnitude, aggregated_consensus, time_to_diffusion, catalyst_proximity
            )

            # Create signal
            signal = NGSignal(
                symbol=symbol,
                ng_score=ng_score,
                market_path=market_path,
                dfl_path=dfl_distribution.expected_path,
                gap_magnitude=gap_magnitude,
                time_to_diffusion=time_to_diffusion,
                catalyst_proximity=catalyst_proximity,
                confidence=aggregated_consensus.confidence,
                timestamp=datetime.now(),
                supporting_data={
                    'consensus': aggregated_consensus.__dict__,
                    'dfl_params': dfl_distribution.distribution_params,
                    'signal_components': {
                        'gap_magnitude': gap_magnitude,
                        'consensus_strength': aggregated_consensus.consensus_strength,
                        'narrative_coherence': aggregated_consensus.narrative_coherence
                    }
                }
            )

            logger.info(f"Generated NG signal for {symbol}: score={ng_score:.3f}")
            return signal

        except Exception as e:
            logger.error(f"Error calculating narrative gap for {symbol}: {e}")
            return self._create_fallback_signal(symbol, current_price)

    def _aggregate_consensus(self, consensus_list: List[MarketConsensus], symbol: str) -> MarketConsensus:
        """Aggregate consensus from multiple sources"""
        valid_consensus = [c for c in consensus_list if c.confidence > 0.1]

        if not valid_consensus:
            return MarketConsensus(symbol=symbol, price_target=0.0, time_horizon=30, confidence=0.0)

        # Weight by confidence
        weights = np.array([c.confidence for c in valid_consensus])
        weights = weights / weights.sum()

        # Weighted average price target
        price_targets = [c.price_target for c in valid_consensus]
        avg_price_target = np.average(price_targets, weights=weights)

        # Average time horizon
        avg_horizon = int(np.mean([c.time_horizon for c in valid_consensus]))

        # Aggregate confidence
        avg_confidence = np.mean([c.confidence for c in valid_consensus])

        # Collect all narratives
        all_narratives = []
        for consensus in valid_consensus:
            all_narratives.extend(consensus.supporting_narratives)

        aggregated = MarketConsensus(
            symbol=symbol,
            price_target=avg_price_target,
            time_horizon=avg_horizon,
            confidence=avg_confidence,
            supporting_narratives=all_narratives
        )

        # Calculate aggregated metrics
        consensus_strengths = [c.consensus_strength for c in valid_consensus if c.consensus_strength > 0]
        aggregated.consensus_strength = np.mean(consensus_strengths) if consensus_strengths else 0.5

        coherence_scores = [c.narrative_coherence for c in valid_consensus if c.narrative_coherence > 0]
        aggregated.narrative_coherence = np.mean(coherence_scores) if coherence_scores else 0.5

        return aggregated

    def _calculate_market_implied_path(self, current_price: float,
                                     consensus: MarketConsensus, horizon_days: int) -> np.ndarray:
        """Calculate market-implied price path from consensus"""
        time_steps = np.linspace(0, horizon_days, horizon_days + 1)

        if consensus.price_target <= 0 or consensus.confidence < 0.1:
            # No consensus - assume flat path
            return np.full_like(time_steps, current_price)

        # Calculate implied return
        target_return = (consensus.price_target - current_price) / current_price

        # Adjust for time horizon mismatch
        horizon_adjustment = horizon_days / consensus.time_horizon
        adjusted_return = target_return / horizon_adjustment

        # Generate smooth path to target
        path = np.zeros_like(time_steps)
        path[0] = current_price

        for i in range(1, len(time_steps)):
            progress = time_steps[i] / horizon_days
            # S-curve progression for more realistic path
            s_curve_progress = 3 * progress**2 - 2 * progress**3
            path[i] = current_price * (1 + adjusted_return * s_curve_progress)

        return path

    def _calculate_gap_magnitude(self, market_path: np.ndarray, dfl_path: np.ndarray) -> float:
        """Calculate magnitude of gap between market and DFL paths"""
        if len(market_path) != len(dfl_path):
            min_len = min(len(market_path), len(dfl_path))
            market_path = market_path[:min_len]
            dfl_path = dfl_path[:min_len]

        # Calculate relative differences
        relative_diffs = np.abs(market_path - dfl_path) / (dfl_path + 1e-8)

        # Weight later time steps more heavily
        weights = np.linspace(0.5, 1.0, len(relative_diffs))
        weighted_gap = np.average(relative_diffs, weights=weights)

        return min(weighted_gap, 1.0)  # Cap at 1.0

    def _calculate_time_to_diffusion(self, consensus: MarketConsensus) -> float:
        """Calculate expected time for narrative to diffuse through market"""
        base_diffusion_time = 5.0  # 5 days base

        # Adjust based on consensus strength and coherence
        strength_factor = 1.0 - consensus.consensus_strength
        coherence_factor = 1.0 - consensus.narrative_coherence

        diffusion_time = base_diffusion_time * (1 + strength_factor + coherence_factor)

        # Normalize to 0-1 scale (assuming max 30 days)
        return min(diffusion_time / 30.0, 1.0)

    def _calculate_catalyst_proximity(self, symbol: str, consensus: MarketConsensus) -> float:
        """Calculate proximity to potential catalysts"""
        # Check for upcoming events (earnings, announcements, etc.)
        # This would integrate with calendar data in production

        base_proximity = 0.5  # Default moderate proximity

        # Check narrative keywords for catalyst indicators
        catalyst_keywords = ['earnings', 'announcement', 'launch', 'merger', 'acquisition', 'fda']

        catalyst_count = 0
        for narrative in consensus.supporting_narratives:
            for keyword in catalyst_keywords:
                if keyword in ' '.join(narrative.keywords):
                    catalyst_count += 1

        # Higher catalyst count = higher proximity
        catalyst_factor = min(catalyst_count * 0.1, 0.4)

        return min(base_proximity + catalyst_factor, 1.0)

    def _calculate_ng_score(self, gap_magnitude: float, consensus: MarketConsensus,
                          time_to_diffusion: float, catalyst_proximity: float) -> float:
        """Calculate final NG score using weighted combination"""

        # Component scores
        gap_component = gap_magnitude * self.signal_params['gap_weight']
        consensus_component = consensus.consensus_strength * self.signal_params['consensus_weight']
        catalyst_component = catalyst_proximity * self.signal_params['catalyst_weight']
        coherence_component = consensus.narrative_coherence * self.signal_params['coherence_weight']

        # Time decay factor
        time_decay = max(0.1, 1.0 - time_to_diffusion)

        # Combine components
        raw_score = (gap_component + consensus_component +
                    catalyst_component + coherence_component) * time_decay

        # Apply confidence scaling
        confidence_adjusted_score = raw_score * consensus.confidence

        # Ensure 0-1 range
        return np.clip(confidence_adjusted_score, 0.0, 1.0)

    def _create_fallback_signal(self, symbol: str, current_price: float) -> NGSignal:
        """Create fallback signal when calculation fails"""
        time_steps = np.linspace(0, 30, 31)
        flat_path = np.full_like(time_steps, current_price)

        return NGSignal(
            symbol=symbol,
            ng_score=0.0,
            market_path=flat_path,
            dfl_path=flat_path,
            gap_magnitude=0.0,
            time_to_diffusion=1.0,
            catalyst_proximity=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            supporting_data={'error': 'Fallback signal due to calculation error'}
        )

# Example usage and testing
async def test_narrative_gap_engine():
    """Test the narrative gap engine"""
    engine = NarrativeGapEngine()

    # Test with sample data
    symbol = "AAPL"
    current_price = 150.0

    signal = await engine.calculate_narrative_gap(symbol, current_price)

    print(f"Narrative Gap Signal for {symbol}:")
    print(f"NG Score: {signal.ng_score:.3f}")
    print(f"Gap Magnitude: {signal.gap_magnitude:.3f}")
    print(f"Time to Diffusion: {signal.time_to_diffusion:.3f}")
    print(f"Catalyst Proximity: {signal.catalyst_proximity:.3f}")
    print(f"Confidence: {signal.confidence:.3f}")

    return signal

if __name__ == "__main__":
    # Run test
    import asyncio
    asyncio.run(test_narrative_gap_engine())