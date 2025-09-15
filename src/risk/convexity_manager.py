"""
Convexity Optimizer for Super-Gary Trading Framework

This module implements regime-aware convexity optimization to ensure survival-first trading
through gamma farming, options structuring, and intelligent exposure management around
regime boundaries and major market events.

Mathematical Foundation:
- Convexity Score: γ = ∂²P/∂S² (second derivative of position value w.r.t. underlying)
- Regime Detection: Hidden Markov Models for market state identification
- Gamma Farming: Long gamma positions delta-hedged for volatility harvesting
- Vanna/Charm Management: Greeks optimization around scheduled events

Key Features:
- HMM-based regime detection with uncertainty quantification
- Convexity requirements scaling with regime uncertainty
- Gamma farming strategies for volatility harvesting
- Event-aware positioning (CPI, FOMC, earnings)
- Options structure builder for optimal payoff profiles
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    LOW_VOL_CONSOLIDATION = "low_vol_consolidation"
    HIGH_VOL_CRISIS = "high_vol_crisis"
    TRANSITION = "transition"
    UNKNOWN = "unknown"

@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float  # Probability of current regime
    regime_probabilities: Dict[MarketRegime, float]
    uncertainty: float  # Entropy of regime distribution
    time_in_regime: int  # Days in current regime
    transition_probability: float  # Probability of regime change
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConvexityTarget:
    """Target convexity profile for position"""
    target_gamma: float  # Target gamma exposure
    current_gamma: float  # Current gamma exposure
    gamma_gap: float  # Difference (target - current)
    convexity_score: float  # Overall convexity adequacy [0,1]
    max_concave_exposure: float  # Maximum allowed concave exposure
    regime_adjusted_target: float  # Regime-adjusted target
    confidence: float  # Confidence in target

@dataclass
class OptionsStructure:
    """Options position structure"""
    structure_id: str
    underlying: str
    structure_type: str  # 'straddle', 'strangle', 'butterfly', etc.
    strikes: List[float]
    expiries: List[datetime]
    quantities: List[int]  # Positive for long, negative for short
    delta: float
    gamma: float
    theta: float
    vega: float
    cost: float
    max_profit: Optional[float]
    max_loss: Optional[float]
    breakeven_points: List[float]

class ConvexityManager:
    """
    Advanced convexity optimization system for regime-aware trading

    Implements survival-first convexity management through intelligent gamma farming,
    options structuring, and regime-aware exposure control.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Regime detection
        self.hmm_components = self.config.get('hmm_components', 4)
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_history: List[RegimeState] = []

        # Convexity tracking
        self.convexity_targets: Dict[str, ConvexityTarget] = {}
        self.gamma_positions: Dict[str, float] = {}
        self.options_structures: Dict[str, OptionsStructure] = {}

        # Event calendar
        self.major_events = self._initialize_event_calendar()
        self.event_horizon_days = self.config.get('event_horizon_days', 7)

        # Performance tracking
        self.convexity_performance: List[Dict] = []

        # Data persistence
        self.data_path = Path(self.config.get('data_path', './data/convexity'))
        self.data_path.mkdir(parents=True, exist_ok=True)

        self._initialize_regime_detection()

    def _default_config(self) -> Dict:
        """Default configuration for convexity manager"""
        return {
            'hmm_components': 4,  # Number of regime states
            'regime_lookback': 252,  # Days for regime detection
            'convexity_threshold': 0.3,  # Minimum convexity near regime boundaries
            'gamma_farming_threshold': 0.15,  # IV percentile for gamma farming
            'max_gamma_exposure': 0.1,  # Maximum gamma as % of portfolio
            'delta_hedge_frequency': 'hourly',  # Delta hedging frequency
            'event_horizon_days': 7,  # Days before event to adjust positioning
            'vanna_exposure_limit': 0.05,  # Maximum vanna exposure
            'charm_decay_threshold': 0.02,  # Daily theta decay limit
            'regime_confidence_threshold': 0.7,  # Minimum confidence for regime-based decisions
            'transition_probability_threshold': 0.3,  # Threshold for regime transition alerts
            'volatility_lookback': 21,  # Days for volatility calculations
            'convexity_update_frequency': 'daily'  # How often to update convexity targets
        }

    def _initialize_regime_detection(self):
        """Initialize regime detection model"""
        try:
            # Try to load existing model
            self._load_regime_model()
            self.logger.info("Loaded existing regime detection model")
        except Exception as e:
            self.logger.warning(f"Could not load regime model: {e}")
            self.logger.info("Will train new regime model when data is available")

    def _initialize_event_calendar(self) -> Dict[datetime, Dict]:
        """Initialize major events calendar"""
        # This would typically be loaded from external data source
        # For now, return sample high-impact events
        today = datetime.now()
        events = {}

        # Sample FOMC meetings (typically 8 per year)
        fomc_dates = [
            today + timedelta(days=30),
            today + timedelta(days=75),
            today + timedelta(days=120),
        ]

        for date in fomc_dates:
            events[date] = {
                'type': 'FOMC',
                'impact': 'high',
                'volatility_expected': 'high',
                'convexity_recommendation': 'increase'
            }

        # Sample CPI releases (monthly)
        for i in range(1, 13):
            cpi_date = today + timedelta(days=30*i - today.day + 12)  # Rough monthly schedule
            events[cpi_date] = {
                'type': 'CPI',
                'impact': 'medium-high',
                'volatility_expected': 'medium',
                'convexity_recommendation': 'moderate_increase'
            }

        return events

    def update_market_data(self,
                          price_data: pd.DataFrame,
                          volume_data: Optional[pd.DataFrame] = None,
                          volatility_data: Optional[pd.DataFrame] = None) -> RegimeState:
        """
        Update regime detection with new market data

        Args:
            price_data: DataFrame with price data (columns: open, high, low, close)
            volume_data: Optional volume data
            volatility_data: Optional volatility metrics

        Returns:
            Current regime state
        """
        try:
            # Prepare features for regime detection
            features = self._extract_regime_features(price_data, volume_data, volatility_data)

            if len(features) < self.config['regime_lookback']:
                self.logger.warning("Insufficient data for regime detection")
                return self._create_unknown_regime_state()

            # Train or update regime model
            if self.regime_model is None:
                self._train_regime_model(features)

            # Detect current regime
            current_regime = self._detect_current_regime(features)
            self.regime_history.append(current_regime)

            # Update convexity targets based on regime
            self._update_convexity_targets(current_regime)

            self.logger.info(f"Updated regime: {current_regime.regime.value} "
                           f"(confidence: {current_regime.confidence:.3f})")

            return current_regime

        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return self._create_unknown_regime_state()

    def get_convexity_requirements(self,
                                 asset: str,
                                 position_size: float,
                                 current_regime: Optional[RegimeState] = None) -> ConvexityTarget:
        """
        Calculate convexity requirements for given position

        Args:
            asset: Asset identifier
            position_size: Current position size
            current_regime: Current market regime (uses latest if None)

        Returns:
            Convexity target specification
        """
        if current_regime is None:
            current_regime = self.regime_history[-1] if self.regime_history else self._create_unknown_regime_state()

        # Base convexity requirement
        base_convexity = self.config['convexity_threshold']

        # Adjust for regime uncertainty
        uncertainty_multiplier = 1 + (current_regime.uncertainty * 2)  # Scale with entropy
        regime_adjusted_convexity = base_convexity * uncertainty_multiplier

        # Adjust for upcoming events
        event_adjustment = self._calculate_event_convexity_adjustment()
        final_target = regime_adjusted_convexity * event_adjustment

        # Calculate current gamma exposure
        current_gamma = self.gamma_positions.get(asset, 0.0)

        # Calculate gamma gap
        target_gamma = final_target * abs(position_size)  # Scale by position size
        gamma_gap = target_gamma - current_gamma

        # Calculate convexity score (how well current position meets requirements)
        if target_gamma > 0:
            convexity_score = min(1.0, current_gamma / target_gamma)
        else:
            convexity_score = 1.0  # No convexity needed

        target = ConvexityTarget(
            target_gamma=target_gamma,
            current_gamma=current_gamma,
            gamma_gap=gamma_gap,
            convexity_score=convexity_score,
            max_concave_exposure=final_target * -0.5,  # Allow some concave exposure
            regime_adjusted_target=regime_adjusted_convexity,
            confidence=current_regime.confidence
        )

        self.convexity_targets[asset] = target
        return target

    def optimize_gamma_farming(self,
                             underlying: str,
                             portfolio_value: float,
                             implied_vol_percentile: float,
                             current_volatility: float) -> Optional[OptionsStructure]:
        """
        Optimize gamma farming strategy for volatility harvesting

        Args:
            underlying: Underlying asset
            portfolio_value: Total portfolio value for sizing
            implied_vol_percentile: Current IV percentile [0,1]
            current_volatility: Current realized volatility

        Returns:
            Optimal options structure for gamma farming
        """
        # Only farm gamma when IV is relatively low
        if implied_vol_percentile > self.config['gamma_farming_threshold']:
            self.logger.info("IV too high for gamma farming")
            return None

        try:
            # Calculate optimal structure
            structure = self._design_gamma_farming_structure(
                underlying, portfolio_value, current_volatility
            )

            if structure and self._validate_gamma_structure(structure):
                self.options_structures[structure.structure_id] = structure
                self.logger.info(f"Created gamma farming structure: {structure.structure_id}")
                return structure
            else:
                self.logger.warning("Gamma farming structure failed validation")
                return None

        except Exception as e:
            self.logger.error(f"Error optimizing gamma farming: {e}")
            return None

    def manage_event_exposure(self,
                            upcoming_events: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Manage exposure around major events (FOMC, CPI, earnings)

        Args:
            upcoming_events: List of upcoming events (uses calendar if None)

        Returns:
            Event management recommendations
        """
        if upcoming_events is None:
            upcoming_events = self._get_upcoming_events()

        recommendations = {
            'actions': [],
            'risk_adjustments': {},
            'convexity_changes': {},
            'hedge_suggestions': []
        }

        for event in upcoming_events:
            days_until = (event['date'] - datetime.now()).days

            if days_until <= self.event_horizon_days:
                # Near-term event management
                if event['type'] in ['FOMC', 'CPI']:
                    # High-impact macro events
                    recommendations['actions'].append({
                        'action': 'increase_convexity',
                        'reason': f"{event['type']} in {days_until} days",
                        'target_increase': 0.5,  # 50% increase in convexity
                        'priority': 'high'
                    })

                    # Reduce vanna exposure (vol/spot correlation risk)
                    recommendations['risk_adjustments']['vanna'] = 'reduce'
                    recommendations['risk_adjustments']['charm'] = 'hedge'  # Time decay risk

                elif event['type'] == 'earnings':
                    # Earnings-specific management
                    recommendations['actions'].append({
                        'action': 'adjust_for_earnings',
                        'reason': f"Earnings in {days_until} days",
                        'target_adjustment': 'neutral_gamma',
                        'priority': 'medium'
                    })

                # Generate specific hedge suggestions
                hedge = self._generate_event_hedge(event, days_until)
                if hedge:
                    recommendations['hedge_suggestions'].append(hedge)

        return recommendations

    def calculate_optimal_strikes(self,
                                underlying_price: float,
                                volatility: float,
                                days_to_expiry: int,
                                structure_type: str = 'straddle') -> Dict[str, Any]:
        """
        Calculate optimal strikes for options structures

        Args:
            underlying_price: Current underlying price
            volatility: Implied volatility
            days_to_expiry: Days until expiration
            structure_type: Type of structure to optimize

        Returns:
            Optimal strike configuration
        """
        try:
            if structure_type == 'straddle':
                return self._optimize_straddle_strikes(underlying_price, volatility, days_to_expiry)
            elif structure_type == 'strangle':
                return self._optimize_strangle_strikes(underlying_price, volatility, days_to_expiry)
            elif structure_type == 'butterfly':
                return self._optimize_butterfly_strikes(underlying_price, volatility, days_to_expiry)
            else:
                raise ValueError(f"Unknown structure type: {structure_type}")

        except Exception as e:
            self.logger.error(f"Error calculating optimal strikes: {e}")
            return {'error': str(e)}

    def delta_hedge_gamma_positions(self) -> Dict[str, float]:
        """
        Calculate delta hedging requirements for gamma positions

        Returns:
            Dictionary of hedge quantities by asset
        """
        hedge_requirements = {}

        for structure_id, structure in self.options_structures.items():
            if abs(structure.delta) > 0.01:  # Only hedge if significant delta
                # Calculate hedge quantity (negative of current delta)
                hedge_quantity = -structure.delta

                # Apply hedge only if cost-effective
                hedge_cost = abs(hedge_quantity) * 0.001  # Assume 0.1% transaction cost
                gamma_benefit = structure.gamma * 0.01  # Benefit from gamma scalping

                if gamma_benefit > hedge_cost:
                    hedge_requirements[structure.underlying] = hedge_requirements.get(
                        structure.underlying, 0) + hedge_quantity

        return hedge_requirements

    def _extract_regime_features(self,
                               price_data: pd.DataFrame,
                               volume_data: Optional[pd.DataFrame],
                               volatility_data: Optional[pd.DataFrame]) -> np.ndarray:
        """Extract features for regime detection"""
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()

        # Volatility features
        vol_window = self.config['volatility_lookback']
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        vol_percentile = rolling_vol.rolling(252).rank(pct=True)

        # Trend features
        price_ma_short = price_data['close'].rolling(21).mean()
        price_ma_long = price_data['close'].rolling(63).mean()
        trend_strength = (price_ma_short / price_ma_long - 1) * 100

        # Momentum features
        momentum_short = returns.rolling(5).mean() * 252
        momentum_long = returns.rolling(21).mean() * 252

        # Volume features (if available)
        if volume_data is not None:
            volume_ma = volume_data['volume'].rolling(21).mean()
            volume_ratio = volume_data['volume'] / volume_ma
        else:
            volume_ratio = pd.Series(1.0, index=returns.index)

        # Market structure features
        high_low_ratio = (price_data['high'] / price_data['low'] - 1) * 100
        daily_range = high_low_ratio.rolling(21).mean()

        # Combine features
        features_df = pd.DataFrame({
            'returns': returns,
            'volatility': rolling_vol,
            'vol_percentile': vol_percentile,
            'trend_strength': trend_strength,
            'momentum_short': momentum_short,
            'momentum_long': momentum_long,
            'volume_ratio': volume_ratio,
            'daily_range': daily_range
        }).dropna()

        return features_df.values

    def _train_regime_model(self, features: np.ndarray):
        """Train HMM-based regime detection model"""
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)

            # Train Gaussian Mixture Model as proxy for HMM
            self.regime_model = GaussianMixture(
                n_components=self.hmm_components,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )

            self.regime_model.fit(features_scaled)
            self.logger.info(f"Trained regime model with {self.hmm_components} components")

        except Exception as e:
            self.logger.error(f"Error training regime model: {e}")
            raise

    def _detect_current_regime(self, features: np.ndarray) -> RegimeState:
        """Detect current market regime"""
        try:
            # Use recent data for current regime detection
            recent_features = features[-1:] if len(features) > 0 else features
            features_scaled = self.scaler.transform(recent_features)

            # Get regime probabilities
            regime_probs = self.regime_model.predict_proba(features_scaled)[0]
            current_regime_idx = np.argmax(regime_probs)

            # Map to regime types
            regime_mapping = {
                0: MarketRegime.BULL_TRENDING,
                1: MarketRegime.BEAR_TRENDING,
                2: MarketRegime.LOW_VOL_CONSOLIDATION,
                3: MarketRegime.HIGH_VOL_CRISIS
            }

            current_regime = regime_mapping.get(current_regime_idx, MarketRegime.UNKNOWN)
            confidence = regime_probs[current_regime_idx]

            # Calculate uncertainty (entropy)
            uncertainty = -np.sum(regime_probs * np.log(regime_probs + 1e-10))

            # Calculate transition probability
            if len(self.regime_history) > 0:
                prev_regime = self.regime_history[-1].regime
                transition_prob = 1 - confidence if prev_regime != current_regime else confidence
            else:
                transition_prob = 1 - confidence

            # Time in regime
            time_in_regime = 1
            if len(self.regime_history) > 0:
                for i in range(len(self.regime_history) - 1, -1, -1):
                    if self.regime_history[i].regime == current_regime:
                        time_in_regime += 1
                    else:
                        break

            return RegimeState(
                regime=current_regime,
                confidence=confidence,
                regime_probabilities={regime_mapping[i]: prob for i, prob in enumerate(regime_probs)},
                uncertainty=uncertainty,
                time_in_regime=time_in_regime,
                transition_probability=transition_prob
            )

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return self._create_unknown_regime_state()

    def _create_unknown_regime_state(self) -> RegimeState:
        """Create unknown regime state for error cases"""
        return RegimeState(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            regime_probabilities={regime: 0.0 for regime in MarketRegime},
            uncertainty=1.0,
            time_in_regime=0,
            transition_probability=1.0
        )

    def _update_convexity_targets(self, regime_state: RegimeState):
        """Update convexity targets based on regime"""
        # Higher convexity requirements during uncertain regimes
        base_multiplier = 1.0 + regime_state.uncertainty

        # Additional multiplier for transition periods
        if regime_state.transition_probability > self.config['transition_probability_threshold']:
            base_multiplier *= 1.5

        # Update all existing targets
        for asset, target in self.convexity_targets.items():
            target.regime_adjusted_target = target.target_gamma * base_multiplier

    def _calculate_event_convexity_adjustment(self) -> float:
        """Calculate convexity adjustment for upcoming events"""
        upcoming_events = self._get_upcoming_events()
        max_adjustment = 1.0

        for event in upcoming_events:
            days_until = (event['date'] - datetime.now()).days
            if days_until <= self.event_horizon_days:
                # Increase convexity as events approach
                event_impact = {
                    'FOMC': 0.5,
                    'CPI': 0.3,
                    'earnings': 0.2
                }.get(event['type'], 0.1)

                # Adjustment decreases linearly with time
                time_factor = 1 - (days_until / self.event_horizon_days)
                adjustment = 1 + (event_impact * time_factor)
                max_adjustment = max(max_adjustment, adjustment)

        return max_adjustment

    def _get_upcoming_events(self) -> List[Dict]:
        """Get upcoming events from calendar"""
        today = datetime.now()
        upcoming = []

        for date, event_info in self.major_events.items():
            if date >= today:
                upcoming.append({
                    'date': date,
                    **event_info
                })

        return sorted(upcoming, key=lambda x: x['date'])

    def _design_gamma_farming_structure(self,
                                      underlying: str,
                                      portfolio_value: float,
                                      current_volatility: float) -> Optional[OptionsStructure]:
        """Design optimal gamma farming structure"""
        # Simple long straddle for gamma farming
        structure_id = f"gamma_farm_{underlying}_{datetime.now().strftime('%Y%m%d')}"

        # Position sizing (limit gamma exposure)
        max_gamma_value = portfolio_value * self.config['max_gamma_exposure']

        # Estimate structure parameters (simplified)
        estimated_gamma = 0.02  # Approximate gamma for ATM straddle
        max_structures = int(max_gamma_value / (estimated_gamma * 100))  # 100 shares per contract

        if max_structures < 1:
            return None

        # Create structure
        structure = OptionsStructure(
            structure_id=structure_id,
            underlying=underlying,
            structure_type='straddle',
            strikes=[100.0],  # ATM (would be adjusted to actual price)
            expiries=[datetime.now() + timedelta(days=30)],  # 30-day expiry
            quantities=[max_structures],
            delta=0.0,  # ATM straddle starts delta-neutral
            gamma=estimated_gamma * max_structures,
            theta=-0.05 * max_structures,  # Approximate theta
            vega=0.15 * max_structures,   # Approximate vega
            cost=2.0 * max_structures,    # Approximate cost
            max_profit=None,  # Unlimited for straddle
            max_loss=2.0 * max_structures,  # Premium paid
            breakeven_points=[98.0, 102.0]  # Approximate breakevens
        )

        return structure

    def _validate_gamma_structure(self, structure: OptionsStructure) -> bool:
        """Validate gamma farming structure"""
        # Check position limits
        if abs(structure.gamma) > self.config['max_gamma_exposure']:
            return False

        # Check cost relative to potential profit
        max_daily_pnl = abs(structure.gamma) * 0.01  # 1% move benefit
        if structure.cost > max_daily_pnl * 30:  # 30-day payback period
            return False

        # Check vanna exposure
        if abs(structure.vega * structure.delta) > self.config['vanna_exposure_limit']:
            return False

        return True

    def _generate_event_hedge(self, event: Dict, days_until: int) -> Optional[Dict]:
        """Generate hedge recommendation for specific event"""
        if event['type'] == 'FOMC':
            return {
                'hedge_type': 'volatility_hedge',
                'recommendation': 'long_straddle',
                'reason': 'FOMC volatility spike protection',
                'sizing': 'light',  # Small hedge
                'duration': 'short_term'  # Close after event
            }
        elif event['type'] == 'CPI':
            return {
                'hedge_type': 'directional_hedge',
                'recommendation': 'collar',
                'reason': 'CPI directional protection',
                'sizing': 'moderate',
                'duration': 'event_based'
            }

        return None

    def _optimize_straddle_strikes(self, underlying_price: float, volatility: float, days_to_expiry: int) -> Dict:
        """Optimize straddle strike selection"""
        # ATM straddle is typically optimal for gamma farming
        atm_strike = round(underlying_price / 5) * 5  # Round to nearest $5

        return {
            'structure_type': 'straddle',
            'strikes': [atm_strike],
            'optimal_strike': atm_strike,
            'expected_gamma': 0.02,  # Approximate
            'expected_cost': underlying_price * 0.02  # 2% of underlying
        }

    def _optimize_strangle_strikes(self, underlying_price: float, volatility: float, days_to_expiry: int) -> Dict:
        """Optimize strangle strike selection"""
        # OTM strikes for strangle
        otm_distance = underlying_price * volatility * np.sqrt(days_to_expiry / 365) * 0.5

        call_strike = round((underlying_price + otm_distance) / 5) * 5
        put_strike = round((underlying_price - otm_distance) / 5) * 5

        return {
            'structure_type': 'strangle',
            'strikes': [put_strike, call_strike],
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expected_gamma': 0.015,  # Lower than straddle
            'expected_cost': underlying_price * 0.015
        }

    def _optimize_butterfly_strikes(self, underlying_price: float, volatility: float, days_to_expiry: int) -> Dict:
        """Optimize butterfly strike selection"""
        # ATM butterfly
        atm_strike = round(underlying_price / 5) * 5
        wing_width = max(5, round(underlying_price * 0.05 / 5) * 5)  # 5% wings

        return {
            'structure_type': 'butterfly',
            'strikes': [atm_strike - wing_width, atm_strike, atm_strike + wing_width],
            'center_strike': atm_strike,
            'wing_width': wing_width,
            'expected_gamma': 0.01,  # Limited gamma in butterfly
            'expected_cost': wing_width * 0.1  # Low cost structure
        }

    def _save_regime_model(self):
        """Save regime detection model"""
        try:
            import pickle
            model_path = self.data_path / 'regime_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.regime_model,
                    'scaler': self.scaler
                }, f)
            self.logger.info("Regime model saved")
        except Exception as e:
            self.logger.error(f"Error saving regime model: {e}")

    def _load_regime_model(self):
        """Load regime detection model"""
        import pickle
        model_path = self.data_path / 'regime_model.pkl'
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.regime_model = data['model']
            self.scaler = data['scaler']

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create convexity manager
    manager = ConvexityManager()

    # Simulate market data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Generate sample price data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    price_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, len(dates)))),
        'close': prices
    }, index=dates)

    # Update with market data
    regime_state = manager.update_market_data(price_data)
    print(f"Current Regime: {regime_state.regime.value}")
    print(f"Confidence: {regime_state.confidence:.3f}")
    print(f"Uncertainty: {regime_state.uncertainty:.3f}")

    # Get convexity requirements
    target = manager.get_convexity_requirements('SPY', 1000000, regime_state)
    print(f"Target Gamma: {target.target_gamma:.4f}")
    print(f"Convexity Score: {target.convexity_score:.3f}")

    # Optimize gamma farming
    structure = manager.optimize_gamma_farming('SPY', 1000000, 0.2, 0.15)
    if structure:
        print(f"Gamma Farming Structure: {structure.structure_type}")
        print(f"Expected Gamma: {structure.gamma:.4f}")
        print(f"Cost: ${structure.cost:.2f}")

    # Event management
    event_mgmt = manager.manage_event_exposure()
    print(f"Event Actions: {len(event_mgmt['actions'])}")
    for action in event_mgmt['actions']:
        print(f"- {action['action']}: {action['reason']}")