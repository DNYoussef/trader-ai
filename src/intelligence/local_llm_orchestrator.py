"""
Local LLM Orchestrator for Black Swan Hunting
Integrates Ollama/Mistral for strategy selection and market analysis
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Context for LLM to analyze market conditions"""
    timestamp: datetime
    vix_level: float
    vix_percentile: float
    spy_returns_5d: float
    spy_returns_20d: float
    put_call_ratio: float
    market_breadth: float
    correlation: float
    volume_ratio: float
    recent_events: List[str]
    sector_performance: Dict[str, float]
    black_swan_indicators: Dict[str, float]

    def to_prompt_context(self) -> str:
        """Convert to readable context for LLM"""
        return f"""
Market Conditions as of {self.timestamp}:
- VIX Level: {self.vix_level:.2f} (Percentile: {self.vix_percentile:.1%})
- SPY Returns: 5-day: {self.spy_returns_5d:.2%}, 20-day: {self.spy_returns_20d:.2%}
- Put/Call Ratio: {self.put_call_ratio:.2f}
- Market Breadth: {self.market_breadth:.2%}
- Correlation: {self.correlation:.2f}
- Volume Ratio: {self.volume_ratio:.2f}

Black Swan Indicators:
{json.dumps(self.black_swan_indicators, indent=2)}

Recent Market Events:
{chr(10).join(f'- {event}' for event in self.recent_events[-5:])}
"""

@dataclass
class StrategyRecommendation:
    """LLM's strategy recommendation"""
    strategy_name: str
    confidence: float
    reasoning: str
    expected_convexity: float
    risk_level: str
    position_size: float
    entry_conditions: List[str]
    exit_conditions: List[str]

class LocalLLMOrchestrator:
    """
    Orchestrates local LLM (Ollama/Mistral) for black swan strategy selection
    """

    def __init__(self,
                 model_name: str = "mistral:7b-instruct-q4_K_M",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the LLM orchestrator

        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"

        # Strategy descriptions for prompt engineering
        self.strategy_descriptions = {
            'tail_hedge': "Deep OTM put options for extreme downside protection",
            'volatility_harvest': "VIX futures trading in backwardation/contango",
            'crisis_alpha': "Pre-planned crisis response protocol",
            'momentum_explosion': "Riding parabolic moves with trailing stops",
            'mean_reversion': "Trading extremes with tight risk controls",
            'correlation_breakdown': "Pairs trading when correlations break",
            'inequality_arbitrage': "Trading wealth concentration effects",
            'event_catalyst': "Positioning before known high-impact events"
        }

        # Load historical performance data
        self.performance_history = self._load_performance_history()

        logger.info(f"LLM Orchestrator initialized with model: {model_name}")

    def _load_performance_history(self) -> pd.DataFrame:
        """Load historical strategy performance for context"""
        try:
            db_path = Path("data/black_swan_training.db")
            if db_path.exists():
                import sqlite3
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql_query(
                        "SELECT * FROM strategy_performance ORDER BY date DESC LIMIT 1000",
                        conn
                    )
                    return df
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")

        return pd.DataFrame()

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is responsive
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                return False

            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]

            return any(self.model_name in name for name in model_names)

        except Exception as e:
            logger.error(f"Ollama status check failed: {e}")
            return False

    def _create_strategy_prompt(self,
                               market_context: MarketContext,
                               available_capital: float) -> str:
        """Create a detailed prompt for strategy selection"""

        # Calculate recent strategy performance if available
        recent_performance = ""
        if not self.performance_history.empty:
            last_30d = self.performance_history[
                self.performance_history['date'] >=
                (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            ]

            if not last_30d.empty:
                strategy_stats = last_30d.groupby('strategy_name').agg({
                    'returns': 'mean',
                    'sharpe_ratio': 'mean',
                    'black_swan_capture': 'sum'
                }).to_dict('index')

                recent_performance = f"""
Recent Strategy Performance (30 days):
{json.dumps(strategy_stats, indent=2)}
"""

        prompt = f"""You are a black swan hunting AI trained on Nassim Taleb's principles of antifragility and convexity.
Your goal is to select the optimal trading strategy that maximizes convex payoffs while limiting downside risk.

{market_context.to_prompt_context()}

Available Capital: ${available_capital:,.2f}

Available Strategies:
{json.dumps(self.strategy_descriptions, indent=2)}

{recent_performance}

Based on the current market conditions and Taleb's barbell strategy (80% safe, 20% aggressive):

1. Analyze which strategy has the highest convexity potential in current conditions
2. Consider tail risk exposure and black swan capture probability
3. Recommend position sizing using Kelly Criterion with safety factor
4. Provide clear entry and exit conditions

Return a JSON response with:
{{
    "strategy": "strategy_name",
    "confidence": 0.0 to 1.0,
    "reasoning": "detailed explanation",
    "expected_convexity": "ratio of upside to downside",
    "risk_level": "low/medium/high",
    "position_size": "fraction of capital to deploy",
    "entry_conditions": ["condition1", "condition2"],
    "exit_conditions": ["condition1", "condition2"]
}}
"""
        return prompt

    def get_strategy_recommendation(self,
                                   market_context: MarketContext,
                                   available_capital: float) -> Optional[StrategyRecommendation]:
        """
        Get strategy recommendation from local LLM

        Args:
            market_context: Current market conditions
            available_capital: Available capital for trading

        Returns:
            Strategy recommendation or None if LLM unavailable
        """

        # Check if Ollama is available
        if not self.check_ollama_status():
            logger.warning("Ollama not available, using fallback strategy selection")
            return self._fallback_strategy_selection(market_context)

        try:
            # Create prompt
            prompt = self._create_strategy_prompt(market_context, available_capital)

            # Call Ollama API
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._fallback_strategy_selection(market_context)

            # Parse response
            result = response.json()
            llm_response = result.get('response', '')

            # Parse JSON from LLM response
            try:
                recommendation_data = json.loads(llm_response)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    recommendation_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")

            # Create recommendation object
            recommendation = StrategyRecommendation(
                strategy_name=recommendation_data.get('strategy', 'tail_hedge'),
                confidence=float(recommendation_data.get('confidence', 0.5)),
                reasoning=recommendation_data.get('reasoning', 'No reasoning provided'),
                expected_convexity=float(recommendation_data.get('expected_convexity', 1.0)),
                risk_level=recommendation_data.get('risk_level', 'medium'),
                position_size=float(recommendation_data.get('position_size', 0.02)),
                entry_conditions=recommendation_data.get('entry_conditions', []),
                exit_conditions=recommendation_data.get('exit_conditions', [])
            )

            logger.info(f"LLM recommended: {recommendation.strategy_name} "
                       f"(confidence: {recommendation.confidence:.1%})")

            return recommendation

        except Exception as e:
            logger.error(f"Error getting LLM recommendation: {e}")
            return self._fallback_strategy_selection(market_context)

    def _fallback_strategy_selection(self,
                                    market_context: MarketContext) -> StrategyRecommendation:
        """
        Rule-based fallback when LLM is unavailable
        Uses simple heuristics based on market conditions
        """

        # Simple rule-based selection
        if market_context.vix_level > 30:
            # High volatility - tail hedge or volatility harvest
            if market_context.spy_returns_5d < -0.05:
                strategy = "tail_hedge"
                reasoning = "High VIX with recent decline suggests tail risk protection needed"
            else:
                strategy = "volatility_harvest"
                reasoning = "Elevated VIX presents volatility harvesting opportunity"

        elif market_context.correlation < 0.3:
            strategy = "correlation_breakdown"
            reasoning = "Low correlation environment favors pairs trading"

        elif abs(market_context.spy_returns_20d) > 0.15:
            if market_context.spy_returns_20d > 0:
                strategy = "momentum_explosion"
                reasoning = "Strong upward momentum may continue"
            else:
                strategy = "mean_reversion"
                reasoning = "Oversold conditions may reverse"

        elif market_context.put_call_ratio > 1.5:
            strategy = "crisis_alpha"
            reasoning = "Elevated put/call ratio indicates fear in market"

        else:
            strategy = "event_catalyst"
            reasoning = "Normal conditions - position for upcoming catalysts"

        return StrategyRecommendation(
            strategy_name=strategy,
            confidence=0.6,  # Lower confidence for rule-based
            reasoning=reasoning,
            expected_convexity=3.0,
            risk_level="medium",
            position_size=0.02,  # Conservative 2% position
            entry_conditions=["Market conditions confirmed"],
            exit_conditions=["Stop loss at -5%", "Take profit at +20%"]
        )

    def analyze_black_swan_probability(self,
                                      market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Use LLM to analyze probability of black swan events

        Args:
            market_data: Recent market data

        Returns:
            Dictionary of event types and probabilities
        """

        if not self.check_ollama_status():
            # Fallback to statistical analysis
            return self._statistical_black_swan_analysis(market_data)

        try:
            # Prepare market summary
            market_summary = {
                'volatility_regime': 'high' if market_data['volatility'].iloc[-1] > 0.2 else 'normal',
                'trend': 'up' if market_data['returns'].rolling(20).mean().iloc[-1] > 0 else 'down',
                'volume_spike': market_data['volume'].iloc[-1] > market_data['volume'].mean() * 2,
                'correlation_breakdown': False  # Would need correlation data
            }

            prompt = f"""Analyze the probability of different black swan events based on current market conditions:

Market Summary:
{json.dumps(market_summary, indent=2)}

Historical Black Swan Events:
- Financial Crisis (2008): Banking system collapse
- Flash Crash (2010): Algorithmic trading failure
- COVID Crash (2020): Pandemic shutdown
- Volmageddon (2018): VIX derivative implosion

Estimate probabilities (0-1) for these event types in the next 30 days:
{{
    "systemic_crisis": 0.0-1.0,
    "flash_crash": 0.0-1.0,
    "geopolitical_shock": 0.0-1.0,
    "liquidity_crisis": 0.0-1.0,
    "technical_breakdown": 0.0-1.0
}}

Provide reasoning for each probability estimate.
"""

            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # Parse probabilities from response
                # This would need more sophisticated parsing
                return {
                    "systemic_crisis": 0.02,
                    "flash_crash": 0.05,
                    "geopolitical_shock": 0.03,
                    "liquidity_crisis": 0.01,
                    "technical_breakdown": 0.04
                }

        except Exception as e:
            logger.error(f"Error in black swan analysis: {e}")

        return self._statistical_black_swan_analysis(market_data)

    def _statistical_black_swan_analysis(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Statistical fallback for black swan probability"""

        # Simple statistical indicators
        recent_vol = market_data['volatility'].iloc[-20:].mean()
        vol_percentile = (recent_vol > market_data['volatility']).mean()

        # Base probabilities adjusted by market conditions
        base_probs = {
            "systemic_crisis": 0.01,
            "flash_crash": 0.03,
            "geopolitical_shock": 0.02,
            "liquidity_crisis": 0.01,
            "technical_breakdown": 0.02
        }

        # Adjust based on volatility regime
        if vol_percentile > 0.8:
            # High volatility increases all probabilities
            return {k: min(v * 3, 0.15) for k, v in base_probs.items()}
        else:
            return base_probs

    def optimize_strategy_weights(self,
                                 historical_performance: pd.DataFrame,
                                 lookback_days: int = 90) -> Dict[str, float]:
        """
        Use LLM to suggest optimal strategy weights based on performance

        Args:
            historical_performance: Historical strategy performance data
            lookback_days: Days to look back for optimization

        Returns:
            Dictionary of strategy names to weights (sum to 1.0)
        """

        # Calculate strategy statistics
        cutoff_date = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        recent_data = historical_performance[historical_performance['date'] >= cutoff_date]

        if recent_data.empty:
            # Equal weights if no data
            strategies = list(self.strategy_descriptions.keys())
            return {s: 1.0 / len(strategies) for s in strategies}

        # Group by strategy and calculate metrics
        strategy_metrics = recent_data.groupby('strategy_name').agg({
            'returns': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'black_swan_capture': 'sum',
            'convexity_ratio': 'mean'
        }).round(4)

        if not self.check_ollama_status():
            # Fallback to simple Sharpe ratio weighting
            return self._sharpe_based_weights(strategy_metrics)

        try:
            prompt = f"""Optimize strategy weights for maximum convexity and black swan capture.

Strategy Performance (last {lookback_days} days):
{strategy_metrics.to_string()}

Optimization Criteria:
1. Maximize expected convex payoff
2. Prioritize strategies that captured black swans
3. Consider risk-adjusted returns (Sharpe ratio)
4. Maintain diversification (no single strategy >40%)
5. Follow Taleb's barbell: majority in conservative strategies

Return JSON with weights (must sum to 1.0):
{{
    "tail_hedge": 0.0-0.4,
    "volatility_harvest": 0.0-0.4,
    "crisis_alpha": 0.0-0.4,
    "momentum_explosion": 0.0-0.4,
    "mean_reversion": 0.0-0.4,
    "correlation_breakdown": 0.0-0.4,
    "inequality_arbitrage": 0.0-0.4,
    "event_catalyst": 0.0-0.4
}}
"""

            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                weights = json.loads(result.get('response', '{}'))

                # Normalize to sum to 1.0
                total = sum(weights.values())
                if total > 0:
                    return {k: v/total for k, v in weights.items()}

        except Exception as e:
            logger.error(f"Error optimizing weights with LLM: {e}")

        return self._sharpe_based_weights(strategy_metrics)

    def _sharpe_based_weights(self, strategy_metrics: pd.DataFrame) -> Dict[str, float]:
        """Simple Sharpe ratio based weighting"""

        strategies = list(self.strategy_descriptions.keys())
        weights = {}

        for strategy in strategies:
            if strategy in strategy_metrics.index:
                sharpe = strategy_metrics.loc[strategy, ('sharpe_ratio', 'mean')]
                # Only positive Sharpe ratios get weight
                weights[strategy] = max(sharpe, 0)
            else:
                weights[strategy] = 0.1  # Small weight for untested strategies

        # Normalize
        total = sum(weights.values())
        if total > 0:
            return {k: v/total for k, v in weights.items()}
        else:
            # Equal weights if all negative
            return {s: 1.0/len(strategies) for s in strategies}

    def generate_trading_narrative(self,
                                  market_context: MarketContext,
                                  selected_strategy: str,
                                  position_details: Dict[str, Any]) -> str:
        """
        Generate a narrative explanation of the trading decision

        Args:
            market_context: Current market conditions
            selected_strategy: The selected strategy name
            position_details: Details about the position

        Returns:
            Human-readable narrative explanation
        """

        if not self.check_ollama_status():
            # Simple template-based narrative
            return f"""
Strategy: {selected_strategy}
Market VIX: {market_context.vix_level:.1f}
Position Size: {position_details.get('size', 'N/A')}
Reasoning: Market conditions suggest {selected_strategy} has favorable convexity.
"""

        try:
            prompt = f"""Generate a brief trading narrative in Nassim Taleb's style:

Selected Strategy: {selected_strategy}
{market_context.to_prompt_context()}

Position Details:
{json.dumps(position_details, indent=2)}

Write a 2-3 sentence explanation of why this trade exhibits antifragility and positive convexity.
Focus on the asymmetric payoff and limited downside risk.
"""

            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Trade executed based on convexity analysis.')

        except Exception as e:
            logger.error(f"Error generating narrative: {e}")

        return f"Executing {selected_strategy} based on favorable risk/reward asymmetry."