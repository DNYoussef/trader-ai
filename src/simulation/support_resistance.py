"""
Support/Resistance Calculator

Derives support and resistance levels mathematically from order history.

Key insight from Crayfer:
- S/R is NOT psychological - it emerges from order accumulation
- When price returns to a level with historical orders, it must "get through" them
- Dense order areas = strong S/R
- Gaps = areas where price moves quickly

This module provides mathematically-derived S/R that can be used as
predictive features for the AI model.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .orderbook import OrderBookSnapshot


@dataclass
class SRLevel:
    """Support or Resistance level."""
    price: float
    strength: float       # Volume/order density at this level
    level_type: str       # 'support' or 'resistance'
    touch_count: int      # How many times price has touched this level
    hold_count: int       # How many times level held
    break_count: int      # How many times level broke
    reliability: float    # hold_count / touch_count


class SupportResistanceCalculator:
    """
    Calculates support and resistance from order book history.

    The algorithm:
    1. Track cumulative order volume at each price level
    2. Levels with high historical volume = S/R zones
    3. Track touches, holds, and breaks to measure reliability
    4. Provide features for ML: distance to S/R, strength, reliability
    """

    def __init__(
        self,
        price_bucket_size: float = 0.50,  # Group prices into 50 cent buckets
        min_volume_percentile: float = 75,  # Top 25% volume = significant
    ):
        self.bucket_size = price_bucket_size
        self.min_volume_pct = min_volume_percentile

        # Track order volume at each price bucket
        self.volume_by_level: Dict[float, float] = defaultdict(float)

        # Track price touches
        self.touch_history: Dict[float, List[str]] = defaultdict(list)  # 'hold' or 'break'

        # Identified S/R levels
        self.support_levels: List[SRLevel] = []
        self.resistance_levels: List[SRLevel] = []

        # Price history for pattern detection
        self.price_history: List[float] = []

    def _bucket_price(self, price: float) -> float:
        """Round price to nearest bucket."""
        return round(price / self.bucket_size) * self.bucket_size

    def process_snapshot(self, snapshot: OrderBookSnapshot, current_price: float):
        """
        Process order book snapshot to accumulate volume data.

        Call this after each step of simulation.
        """
        self.price_history.append(current_price)

        # Accumulate bid volume
        for price, volume in snapshot.bid_depth.items():
            bucket = self._bucket_price(price)
            self.volume_by_level[bucket] += volume

        # Accumulate ask volume
        for price, volume in snapshot.ask_depth.items():
            bucket = self._bucket_price(price)
            self.volume_by_level[bucket] += volume

        # Check for touches of existing S/R levels
        self._check_touches(current_price)

    def _check_touches(self, current_price: float):
        """Check if price touched any S/R levels."""
        tolerance = self.bucket_size * 1.5

        for level in self.support_levels:
            if abs(current_price - level.price) < tolerance:
                level.touch_count += 1
                # Price came from above, touching support
                if len(self.price_history) > 1 and self.price_history[-2] > level.price:
                    if current_price >= level.price - tolerance:
                        level.hold_count += 1
                        self.touch_history[level.price].append('hold')
                    else:
                        level.break_count += 1
                        self.touch_history[level.price].append('break')
                level.reliability = level.hold_count / max(level.touch_count, 1)

        for level in self.resistance_levels:
            if abs(current_price - level.price) < tolerance:
                level.touch_count += 1
                # Price came from below, touching resistance
                if len(self.price_history) > 1 and self.price_history[-2] < level.price:
                    if current_price <= level.price + tolerance:
                        level.hold_count += 1
                        self.touch_history[level.price].append('hold')
                    else:
                        level.break_count += 1
                        self.touch_history[level.price].append('break')
                level.reliability = level.hold_count / max(level.touch_count, 1)

    def calculate_zones(self, current_price: float) -> Dict[str, List[SRLevel]]:
        """
        Calculate support and resistance zones from accumulated data.

        Returns dict with 'support' and 'resistance' level lists.
        """
        if not self.volume_by_level:
            return {'support': [], 'resistance': []}

        # Find significant volume levels
        volumes = list(self.volume_by_level.values())
        threshold = np.percentile(volumes, self.min_volume_pct)

        self.support_levels = []
        self.resistance_levels = []

        for price, volume in self.volume_by_level.items():
            if volume < threshold:
                continue

            # Determine if support or resistance based on position relative to current price
            if price < current_price:
                # Potential support (below current price)
                existing = next(
                    (l for l in self.support_levels if abs(l.price - price) < self.bucket_size * 2),
                    None
                )
                if existing:
                    existing.strength = max(existing.strength, volume)
                else:
                    level = SRLevel(
                        price=price,
                        strength=volume,
                        level_type='support',
                        touch_count=0,
                        hold_count=0,
                        break_count=0,
                        reliability=0.5  # Default 50%
                    )
                    self.support_levels.append(level)
            else:
                # Potential resistance (above current price)
                existing = next(
                    (l for l in self.resistance_levels if abs(l.price - price) < self.bucket_size * 2),
                    None
                )
                if existing:
                    existing.strength = max(existing.strength, volume)
                else:
                    level = SRLevel(
                        price=price,
                        strength=volume,
                        level_type='resistance',
                        touch_count=0,
                        hold_count=0,
                        break_count=0,
                        reliability=0.5
                    )
                    self.resistance_levels.append(level)

        # Sort by distance to current price
        self.support_levels.sort(key=lambda x: current_price - x.price)
        self.resistance_levels.sort(key=lambda x: x.price - current_price)

        return {
            'support': self.support_levels,
            'resistance': self.resistance_levels
        }

    def get_nearest_support(self, current_price: float) -> Optional[SRLevel]:
        """Get nearest support level below current price."""
        self.calculate_zones(current_price)
        if self.support_levels:
            return self.support_levels[0]
        return None

    def get_nearest_resistance(self, current_price: float) -> Optional[SRLevel]:
        """Get nearest resistance level above current price."""
        self.calculate_zones(current_price)
        if self.resistance_levels:
            return self.resistance_levels[0]
        return None

    def get_support_strength(self, current_price: float) -> float:
        """Get strength of nearest support (normalized)."""
        support = self.get_nearest_support(current_price)
        if support and self.volume_by_level:
            max_vol = max(self.volume_by_level.values())
            return support.strength / max_vol if max_vol > 0 else 0
        return 0.0

    def get_resistance_strength(self, current_price: float) -> float:
        """Get strength of nearest resistance (normalized)."""
        resistance = self.get_nearest_resistance(current_price)
        if resistance and self.volume_by_level:
            max_vol = max(self.volume_by_level.values())
            return resistance.strength / max_vol if max_vol > 0 else 0
        return 0.0

    def get_distance_to_support(self, current_price: float) -> float:
        """Get percentage distance to nearest support."""
        support = self.get_nearest_support(current_price)
        if support:
            return (current_price - support.price) / current_price
        return 0.1  # Default 10% if no support found

    def get_distance_to_resistance(self, current_price: float) -> float:
        """Get percentage distance to nearest resistance."""
        resistance = self.get_nearest_resistance(current_price)
        if resistance:
            return (resistance.price - current_price) / current_price
        return 0.1  # Default 10%

    def get_sr_features(self, current_price: float) -> np.ndarray:
        """
        Get S/R features for ML model.

        Returns 4 features:
        - Distance to nearest support (%)
        - Distance to nearest resistance (%)
        - Support strength (normalized)
        - Resistance strength (normalized)
        """
        features = np.zeros(4, dtype=np.float32)

        features[0] = self.get_distance_to_support(current_price)
        features[1] = self.get_distance_to_resistance(current_price)
        features[2] = self.get_support_strength(current_price)
        features[3] = self.get_resistance_strength(current_price)

        return features

    def predict_breakout_probability(self, current_price: float, direction: str) -> float:
        """
        Estimate probability of breaking nearest S/R level.

        Based on:
        - Historical reliability of the level
        - Current momentum (from price history)
        - Order book imbalance (would need to be passed in)
        """
        if direction == 'up':
            level = self.get_nearest_resistance(current_price)
        else:
            level = self.get_nearest_support(current_price)

        if not level:
            return 0.5  # No level = 50% probability

        # Base probability from historical reliability
        # If level has held 80% of time, break probability is 20%
        break_prob = 1.0 - level.reliability

        # Adjust for momentum
        if len(self.price_history) >= 5:
            recent_return = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]

            if direction == 'up' and recent_return > 0:
                # Positive momentum increases break probability
                break_prob += min(recent_return * 5, 0.2)  # Up to +20%
            elif direction == 'down' and recent_return < 0:
                break_prob += min(abs(recent_return) * 5, 0.2)

        return np.clip(break_prob, 0.05, 0.95)

    def get_pattern_analysis(self, current_price: float) -> Dict:
        """
        Analyze current price position relative to S/R structure.

        Returns analysis dict useful for decision making.
        """
        support = self.get_nearest_support(current_price)
        resistance = self.get_nearest_resistance(current_price)

        dist_support = self.get_distance_to_support(current_price)
        dist_resistance = self.get_distance_to_resistance(current_price)

        # Determine position
        if dist_support < 0.01:
            position = 'at_support'
        elif dist_resistance < 0.01:
            position = 'at_resistance'
        elif dist_support < dist_resistance:
            position = 'near_support'
        else:
            position = 'near_resistance'

        return {
            'position': position,
            'nearest_support': support.price if support else None,
            'nearest_resistance': resistance.price if resistance else None,
            'support_distance_pct': dist_support * 100,
            'resistance_distance_pct': dist_resistance * 100,
            'support_strength': self.get_support_strength(current_price),
            'resistance_strength': self.get_resistance_strength(current_price),
            'upside_break_prob': self.predict_breakout_probability(current_price, 'up'),
            'downside_break_prob': self.predict_breakout_probability(current_price, 'down'),
        }


def demo_sr_calculator():
    """Demo the support/resistance calculator."""
    from .orderbook_simulator import HistoricalOrderBookSimulator, SimulatorConfig

    print("Support/Resistance Calculator Demo")
    print("=" * 50)

    # Create simulator
    config = SimulatorConfig(initial_price=100.0, maker_ratio=0.20)
    sim = HistoricalOrderBookSimulator(config)

    # Create S/R calculator
    sr_calc = SupportResistanceCalculator(price_bucket_size=0.25)

    # Run simulation and feed snapshots to calculator
    for _ in range(200):
        result = sim.step(n_orders=15)
        sr_calc.process_snapshot(result['snapshot'], result['price'])

    current_price = sim.current_price

    print(f"\nCurrent Price: ${current_price:.2f}")

    # Calculate zones
    zones = sr_calc.calculate_zones(current_price)

    print(f"\nSupport Levels:")
    for level in zones['support'][:5]:
        print(f"  ${level.price:.2f} - Strength: {level.strength:.0f}, "
              f"Reliability: {level.reliability:.0%}")

    print(f"\nResistance Levels:")
    for level in zones['resistance'][:5]:
        print(f"  ${level.price:.2f} - Strength: {level.strength:.0f}, "
              f"Reliability: {level.reliability:.0%}")

    # Pattern analysis
    analysis = sr_calc.get_pattern_analysis(current_price)
    print(f"\nPattern Analysis:")
    print(f"  Position: {analysis['position']}")
    print(f"  Distance to Support: {analysis['support_distance_pct']:.1f}%")
    print(f"  Distance to Resistance: {analysis['resistance_distance_pct']:.1f}%")
    print(f"  Upside Break Probability: {analysis['upside_break_prob']:.0%}")
    print(f"  Downside Break Probability: {analysis['downside_break_prob']:.0%}")

    # Features
    features = sr_calc.get_sr_features(current_price)
    print(f"\nS/R Features for ML:")
    print(f"  [0] Distance to Support: {features[0]:.4f}")
    print(f"  [1] Distance to Resistance: {features[1]:.4f}")
    print(f"  [2] Support Strength: {features[2]:.4f}")
    print(f"  [3] Resistance Strength: {features[3]:.4f}")


if __name__ == '__main__':
    demo_sr_calculator()
