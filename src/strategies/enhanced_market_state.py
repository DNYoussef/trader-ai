"""
Enhanced Market State Integration
Combines traditional market data with AI dashboard data sources for better signal generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

# Import dashboard integration
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dashboard'))
    from ai_dashboard_integration import ai_dashboard_integrator
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMarketState:
    """Enhanced market state with AI dashboard data integration"""

    # Traditional market data
    timestamp: datetime
    vix_level: float
    vix_percentile: float
    spy_returns_5d: float
    spy_returns_20d: float
    put_call_ratio: float
    market_breadth: float
    correlation: float
    volume_ratio: float
    regime: str

    # Dashboard AI data integration
    inequality_metrics: Dict[str, float] = field(default_factory=dict)
    sector_performance: Dict[str, float] = field(default_factory=dict)
    ai_mispricing_signals: List[Dict[str, Any]] = field(default_factory=list)
    volatility_clustering: float = 0.0
    real_time_sentiment: float = 0.0
    cross_asset_correlations: Dict[str, float] = field(default_factory=dict)

    # Enhanced indicators
    indicators: Dict[str, float] = field(default_factory=dict)

    # AI confidence metrics
    ai_confidence_level: float = 0.0
    signal_quality_score: float = 0.0

    def __post_init__(self):
        """Initialize enhanced market state with dashboard data"""
        # Always use mock data for synchronous initialization
        # Dashboard data can be loaded separately with load_dashboard_data_async()
        self._load_mock_dashboard_data()

    async def load_dashboard_data_async(self):
        """Load real-time data from AI dashboard asynchronously"""
        if not DASHBOARD_AVAILABLE:
            logger.warning("Dashboard not available, using mock data")
            return

        try:
            # Get inequality panel data
            inequality_data = await ai_dashboard_integrator.get_inequality_panel_data()
            self.inequality_metrics = inequality_data.get('metrics', {})

            # Get contrarian signals
            contrarian_data = await ai_dashboard_integrator.get_contrarian_trades_data()
            self.ai_mispricing_signals = contrarian_data.get('opportunities', [])

            # Recalculate enhanced indicators with new data
            self._calculate_enhanced_indicators()

            logger.info("Successfully loaded dashboard data into enhanced market state")

        except Exception as e:
            logger.warning(f"Failed to load dashboard data, keeping mock data: {e}")

    def _load_mock_dashboard_data(self):
        """Load mock data when dashboard is not available"""

        # Mock inequality metrics
        self.inequality_metrics = {
            'giniCoefficient': 0.48,
            'top1PercentWealth': 32.0,
            'wageGrowthReal': -0.5,
            'luxuryVsDiscountSpend': 1.8,
            'wealthVelocity': 0.25,
            'consensusWrongScore': 0.7
        }

        # Mock sector performance
        self.sector_performance = {
            'XLF': 0.02,  # Financials
            'XLK': 0.01,  # Technology
            'XLE': -0.01, # Energy
            'XLU': 0.005, # Utilities
            'XLV': 0.015, # Healthcare
            'XLI': 0.008, # Industrials
            'XLY': 0.012, # Consumer Discretionary
            'XLP': 0.003, # Consumer Staples
            'XLB': -0.005 # Materials
        }

        # Mock cross-asset correlations
        self.cross_asset_correlations = {
            'spy_tlt': -0.3,  # Stocks vs bonds
            'spy_gld': -0.1,  # Stocks vs gold
            'spy_usd': 0.2,   # Stocks vs dollar
            'spy_oil': 0.4,   # Stocks vs oil
            'tlt_gld': 0.1    # Bonds vs gold
        }

        # Calculate indicators
        self._calculate_enhanced_indicators()

    def _calculate_enhanced_indicators(self):
        """Calculate enhanced market indicators from dashboard data"""

        # Wealth concentration indicator
        gini = self.inequality_metrics.get('giniCoefficient', 0.48)
        self.indicators['wealth_concentration'] = gini

        # Inequality acceleration (proxy for social tension)
        wage_growth = self.inequality_metrics.get('wageGrowthReal', -0.5)
        top1_wealth = self.inequality_metrics.get('top1PercentWealth', 32.0)
        self.indicators['inequality_acceleration'] = (top1_wealth - 30) / 10 + abs(wage_growth) / 2

        # Luxury vs discount ratio (inequality signal)
        luxury_ratio = self.inequality_metrics.get('luxuryVsDiscountSpend', 1.8)
        self.indicators['luxury_discount_spread'] = luxury_ratio - 1.0

        # Sector rotation signal
        sector_momentum = []
        for sector, performance in self.sector_performance.items():
            sector_momentum.append(performance)

        if sector_momentum:
            self.indicators['sector_dispersion'] = np.std(sector_momentum)
            self.indicators['risk_on_sentiment'] = np.mean([
                p for p in sector_momentum if p > 0
            ]) if any(p > 0 for p in sector_momentum) else 0.0

        # Cross-asset correlation breakdown signal
        correlations = list(self.cross_asset_correlations.values())
        if correlations:
            self.indicators['correlation_breakdown'] = 1.0 - np.mean(np.abs(correlations))

        # VIX term structure (estimated from current VIX)
        vix_contango = self._estimate_vix_term_structure()
        self.indicators['vix_term_structure'] = vix_contango

        # Economic calendar events (mock - would be real in production)
        self.indicators['days_to_fomc'] = self._days_to_next_fomc()
        self.indicators['days_to_cpi'] = self._days_to_next_cpi()
        self.indicators['days_to_earnings'] = self._days_to_earnings_season()

        # AI signal quality
        self._calculate_signal_quality()

    def _estimate_vix_term_structure(self) -> float:
        """Estimate VIX term structure from current level"""

        # Simplified estimation - in production would use actual VIX futures
        if self.vix_level < 15:
            # Low VIX typically in contango
            return 0.05
        elif self.vix_level > 25:
            # High VIX typically in backwardation
            return -0.03 - (self.vix_level - 25) * 0.01
        else:
            # Medium VIX - slight contango
            return 0.02

    def _days_to_next_fomc(self) -> int:
        """Calculate days to next FOMC meeting"""
        # Simplified - FOMC meets roughly every 6 weeks
        # In production would use actual calendar
        days_since_epoch = (self.timestamp - datetime(2024, 1, 1)).days
        cycle_position = days_since_epoch % 42  # 6 weeks
        return 42 - cycle_position if cycle_position > 21 else 21 - cycle_position

    def _days_to_next_cpi(self) -> int:
        """Calculate days to next CPI release"""
        # CPI typically released around 13th of each month
        current_day = self.timestamp.day
        if current_day < 13:
            return 13 - current_day
        else:
            # Next month
            next_month = self.timestamp.replace(day=1) + timedelta(days=32)
            next_cpi = next_month.replace(day=13)
            return (next_cpi - self.timestamp).days

    def _days_to_earnings_season(self) -> int:
        """Calculate days to earnings season"""
        # Earnings seasons: Jan, Apr, Jul, Oct
        earnings_months = [1, 4, 7, 10]
        current_month = self.timestamp.month

        # Find next earnings month
        next_earnings_month = None
        for month in earnings_months:
            if month > current_month:
                next_earnings_month = month
                break

        if next_earnings_month is None:
            next_earnings_month = earnings_months[0]  # Next year
            next_year = self.timestamp.year + 1
        else:
            next_year = self.timestamp.year

        next_earnings = datetime(next_year, next_earnings_month, 15)
        return (next_earnings - self.timestamp).days

    def _calculate_signal_quality(self):
        """Calculate overall signal quality score"""

        quality_factors = []

        # VIX level quality (extreme levels = higher quality)
        if self.vix_level < 12 or self.vix_level > 30:
            quality_factors.append(0.8)
        elif self.vix_level < 15 or self.vix_level > 25:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)

        # Volume confirmation
        if self.volume_ratio > 1.5:
            quality_factors.append(0.7)
        elif self.volume_ratio > 1.2:
            quality_factors.append(0.5)
        else:
            quality_factors.append(0.3)

        # Market breadth quality
        if self.market_breadth < 0.3 or self.market_breadth > 0.7:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.5)

        # Correlation breakdown quality
        correlation_breakdown = self.indicators.get('correlation_breakdown', 0)
        if correlation_breakdown > 0.3:
            quality_factors.append(0.8)
        elif correlation_breakdown > 0.2:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)

        # Inequality stress
        inequality_accel = self.indicators.get('inequality_acceleration', 0)
        if inequality_accel > 0.5:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.5)

        # Overall signal quality
        self.signal_quality_score = np.mean(quality_factors)

        # AI confidence from dashboard
        if self.ai_mispricing_signals:
            avg_conviction = np.mean([
                signal.get('conviction', 0.5)
                for signal in self.ai_mispricing_signals
            ])
            self.ai_confidence_level = avg_conviction
        else:
            self.ai_confidence_level = 0.5

    def get_regime_adjusted_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Adjust strategy thresholds based on market regime and signal quality"""

        adjusted = base_thresholds.copy()

        # Regime adjustments
        if self.regime == 'crisis':
            # Lower thresholds in crisis (more sensitive)
            for key in adjusted:
                if 'vix' in key.lower():
                    adjusted[key] *= 0.8
                elif 'momentum' in key.lower():
                    adjusted[key] *= 0.7
                elif 'correlation' in key.lower():
                    adjusted[key] *= 0.9

        elif self.regime == 'calm':
            # Raise thresholds in calm markets (less noise)
            for key in adjusted:
                if 'vix' in key.lower():
                    adjusted[key] *= 1.2
                elif 'momentum' in key.lower():
                    adjusted[key] *= 1.3

        # Signal quality adjustments
        if self.signal_quality_score > 0.7:
            # High quality environment - can use tighter thresholds
            for key in adjusted:
                adjusted[key] *= 0.9
        elif self.signal_quality_score < 0.4:
            # Low quality environment - need higher thresholds
            for key in adjusted:
                adjusted[key] *= 1.2

        return adjusted

    def get_enhanced_market_features(self) -> Dict[str, float]:
        """Get all market features for machine learning models"""

        features = {
            # Traditional features
            'vix_level': self.vix_level,
            'vix_percentile': self.vix_percentile,
            'spy_returns_5d': self.spy_returns_5d,
            'spy_returns_20d': self.spy_returns_20d,
            'put_call_ratio': self.put_call_ratio,
            'market_breadth': self.market_breadth,
            'correlation': self.correlation,
            'volume_ratio': self.volume_ratio,

            # Inequality features
            'gini_coefficient': self.inequality_metrics.get('giniCoefficient', 0.48),
            'top1_wealth': self.inequality_metrics.get('top1PercentWealth', 32.0),
            'wage_growth': self.inequality_metrics.get('wageGrowthReal', -0.5),
            'luxury_discount_ratio': self.inequality_metrics.get('luxuryVsDiscountSpend', 1.8),
            'wealth_velocity': self.inequality_metrics.get('wealthVelocity', 0.25),

            # Enhanced indicators
            'wealth_concentration': self.indicators.get('wealth_concentration', 0.48),
            'inequality_acceleration': self.indicators.get('inequality_acceleration', 0.0),
            'sector_dispersion': self.indicators.get('sector_dispersion', 0.0),
            'correlation_breakdown': self.indicators.get('correlation_breakdown', 0.0),
            'vix_term_structure': self.indicators.get('vix_term_structure', 0.0),
            'risk_on_sentiment': self.indicators.get('risk_on_sentiment', 0.0),

            # Calendar features
            'days_to_fomc': self.indicators.get('days_to_fomc', 999),
            'days_to_cpi': self.indicators.get('days_to_cpi', 999),
            'days_to_earnings': self.indicators.get('days_to_earnings', 999),

            # Quality metrics
            'signal_quality_score': self.signal_quality_score,
            'ai_confidence_level': self.ai_confidence_level
        }

        return features

def create_enhanced_market_state(
    timestamp: datetime,
    vix_level: float,
    spy_returns_5d: float,
    spy_returns_20d: float,
    put_call_ratio: float = 1.0,
    market_breadth: float = 0.5,
    correlation: float = 0.6,
    volume_ratio: float = 1.0,
    regime: str = 'normal'
) -> EnhancedMarketState:
    """Factory function to create enhanced market state"""

    # Calculate VIX percentile (simplified)
    vix_percentile = min(1.0, max(0.0, (vix_level - 10) / 30))

    return EnhancedMarketState(
        timestamp=timestamp,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        spy_returns_5d=spy_returns_5d,
        spy_returns_20d=spy_returns_20d,
        put_call_ratio=put_call_ratio,
        market_breadth=market_breadth,
        correlation=correlation,
        volume_ratio=volume_ratio,
        regime=regime
    )

if __name__ == "__main__":
    # Test enhanced market state
    import asyncio

    async def test_enhanced_market_state():
        market_state = create_enhanced_market_state(
            timestamp=datetime.now(),
            vix_level=22.5,
            spy_returns_5d=-0.02,
            spy_returns_20d=0.01,
            put_call_ratio=1.3,
            market_breadth=0.35,
            volume_ratio=1.8,
            regime='volatile'
        )

        print("Enhanced Market State Features:")
        features = market_state.get_enhanced_market_features()
        for key, value in features.items():
            print(f"  {key}: {value}")

        print(f"\nSignal Quality Score: {market_state.signal_quality_score:.2f}")
        print(f"AI Confidence Level: {market_state.ai_confidence_level:.2f}")

    asyncio.run(test_enhanced_market_state())