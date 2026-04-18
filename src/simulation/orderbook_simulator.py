"""
Order Book Market Simulator

Based on Crayfer's CAT market simulation insights:
- Random traders generate orders around current price
- Low maker ratio (20%) allows price movement
- Patterns emerge from order accumulation, not psychology
- Support/Resistance is mathematical, not psychological

Key insight: "The more makers, the less price moves.
             The fewer makers, the more price can gap."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .orderbook import (
    OrderBook, Order, Trade, OrderBookSnapshot,
    OrderSide, OrderType
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for order book simulator."""
    n_traders: int = 1000          # Number of random traders
    maker_ratio: float = 0.20      # 20% makers (Crayfer's key insight)
    price_variance: float = 0.02   # Orders +/- 2% of current price
    base_order_size: float = 100   # Base order size
    size_variance: float = 0.5     # Order size randomness
    tick_size: float = 0.01        # Price tick size
    initial_price: float = 100.0   # Starting price


class OrderBookSimulator:
    """
    Full market simulation with random traders.

    Implements Crayfer's insights:
    1. Low maker ratio allows price movement
    2. Patterns emerge from mathematics, not psychology
    3. Gaps in book predict rapid price movement
    4. Dense areas become support/resistance
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        self.orderbook = OrderBook(tick_size=self.config.tick_size)
        self.current_price = self.config.initial_price
        self.price_history: List[float] = [self.current_price]
        self.volume_history: List[float] = []
        self._step_count = 0

        # Initialize with some orders
        self._initialize_book()

    def _initialize_book(self, n_orders: int = 200):
        """Seed the order book with initial orders around starting price."""
        for _ in range(n_orders):
            side = np.random.choice([OrderSide.BID, OrderSide.ASK])

            if side == OrderSide.BID:
                # Bids below current price
                offset = np.random.uniform(0.001, 0.03) * self.current_price
                price = self.current_price - offset
            else:
                # Asks above current price
                offset = np.random.uniform(0.001, 0.03) * self.current_price
                price = self.current_price + offset

            price = round(price / self.config.tick_size) * self.config.tick_size
            size = self.config.base_order_size * np.random.uniform(0.5, 1.5)

            order = Order(
                price=price,
                size=size,
                side=side,
                timestamp=0,
                order_type=OrderType.MAKER
            )
            self.orderbook.add_order(order)

    def step(self, n_orders: int = 10) -> Dict:
        """
        Simulate one time step.

        1. Generate random orders from "traders"
        2. Match orders, update price
        3. Return market state

        Args:
            n_orders: Number of random orders to generate

        Returns:
            Dict with step results
        """
        self._step_count += 1
        self.orderbook.advance_time()

        trades = []
        step_volume = 0.0

        for _ in range(n_orders):
            # Determine order type based on maker ratio
            is_maker = np.random.random() < self.config.maker_ratio

            # Random side (buy or sell)
            side = np.random.choice([OrderSide.BID, OrderSide.ASK])

            # Random price around current
            if is_maker:
                # Maker: Place limit order away from current price
                if side == OrderSide.BID:
                    offset = np.random.uniform(0.001, self.config.price_variance)
                    price = self.current_price * (1 - offset)
                else:
                    offset = np.random.uniform(0.001, self.config.price_variance)
                    price = self.current_price * (1 + offset)
            else:
                # Taker: Try to execute near current price (may cross spread)
                if side == OrderSide.BID:
                    # Buy taker - willing to pay up to best ask + slippage
                    best_ask = self.orderbook.get_best_ask()
                    if best_ask:
                        price = best_ask * np.random.uniform(1.0, 1.005)
                    else:
                        price = self.current_price * 1.01
                else:
                    # Sell taker - willing to sell down to best bid - slippage
                    best_bid = self.orderbook.get_best_bid()
                    if best_bid:
                        price = best_bid * np.random.uniform(0.995, 1.0)
                    else:
                        price = self.current_price * 0.99

            # Quantize price
            price = round(price / self.config.tick_size) * self.config.tick_size

            # Random size
            size = self.config.base_order_size * np.random.uniform(
                1 - self.config.size_variance,
                1 + self.config.size_variance
            )

            order = Order(
                price=price,
                size=size,
                side=side,
                timestamp=self._step_count,
                order_type=OrderType.MAKER if is_maker else OrderType.TAKER
            )

            order_trades = self.orderbook.add_order(order)
            trades.extend(order_trades)
            step_volume += sum(t.size for t in order_trades)

        # Update current price based on trades
        if trades:
            self.current_price = trades[-1].price
        else:
            # No trades - use mid price
            mid = self.orderbook.get_mid_price()
            if mid:
                self.current_price = mid

        self.price_history.append(self.current_price)
        self.volume_history.append(step_volume)

        # Take snapshot
        snapshot = self.orderbook.take_snapshot()

        return {
            'step': self._step_count,
            'price': self.current_price,
            'trades': len(trades),
            'volume': step_volume,
            'spread': self.orderbook.get_spread(),
            'spread_pct': self.orderbook.get_spread_pct(),
            'imbalance': self.orderbook.get_bid_ask_imbalance(),
            'snapshot': snapshot,
        }

    def run(self, n_steps: int, orders_per_step: int = 10) -> List[Dict]:
        """Run simulation for n steps."""
        results = []
        for _ in range(n_steps):
            result = self.step(orders_per_step)
            results.append(result)
        return results

    def get_price_series(self) -> np.ndarray:
        """Get price history as numpy array."""
        return np.array(self.price_history)

    def get_returns(self) -> np.ndarray:
        """Get return series."""
        prices = self.get_price_series()
        return np.diff(prices) / prices[:-1]

    def get_volatility(self, window: int = 20) -> float:
        """Get rolling volatility (annualized)."""
        returns = self.get_returns()
        if len(returns) < window:
            return 0.0
        return np.std(returns[-window:]) * np.sqrt(252)

    def get_order_book_features(self) -> np.ndarray:
        """
        Extract order book features for ML.

        Returns 15 features used in training.
        """
        features = np.zeros(15, dtype=np.float32)

        # Imbalance features
        features[0] = self.orderbook.get_bid_ask_imbalance()
        features[1] = self.orderbook.get_spread_pct()

        # Depth features
        features[2] = self.orderbook.get_depth_at_level(0.01, 'bid') / 10000
        features[3] = self.orderbook.get_depth_at_level(0.01, 'ask') / 10000
        features[4] = self.orderbook.get_depth_at_level(0.02, 'bid') / 10000
        features[5] = self.orderbook.get_depth_at_level(0.02, 'ask') / 10000

        # Gap features
        gap_below = self.orderbook.find_largest_gap_below(self.current_price)
        gap_above = self.orderbook.find_largest_gap_above(self.current_price)
        features[6] = (gap_below[1] - gap_below[0]) / self.current_price if gap_below else 0
        features[7] = (gap_above[1] - gap_above[0]) / self.current_price if gap_above else 0

        # Support/Resistance features (from support_resistance module)
        features[8] = 0.0  # nearest_support (computed by SupportResistanceCalculator)
        features[9] = 0.0  # nearest_resistance
        features[10] = 0.0  # support_strength
        features[11] = 0.0  # resistance_strength

        # Flow features
        features[12] = self.orderbook.get_order_flow_imbalance()
        vwap = self.orderbook.get_vwap()
        features[13] = (self.current_price - vwap) / self.current_price if vwap > 0 else 0
        features[14] = self.orderbook.get_depth_slope() / 10000

        # Clamp to reasonable range
        features = np.clip(features, -10, 10)

        return features

    def inject_momentum(self, direction: str, strength: float = 0.5):
        """
        Inject directional momentum (for testing/scenario generation).

        Args:
            direction: 'up' or 'down'
            strength: 0-1, how strong the momentum
        """
        n_orders = int(strength * 50)

        for _ in range(n_orders):
            if direction == 'up':
                # Aggressive buys
                best_ask = self.orderbook.get_best_ask()
                if best_ask:
                    price = best_ask * 1.002
                    order = Order(
                        price=price,
                        size=self.config.base_order_size * 2,
                        side=OrderSide.BID,
                        timestamp=self._step_count,
                        order_type=OrderType.TAKER
                    )
                    self.orderbook.add_order(order)
            else:
                # Aggressive sells
                best_bid = self.orderbook.get_best_bid()
                if best_bid:
                    price = best_bid * 0.998
                    order = Order(
                        price=price,
                        size=self.config.base_order_size * 2,
                        side=OrderSide.ASK,
                        timestamp=self._step_count,
                        order_type=OrderType.TAKER
                    )
                    self.orderbook.add_order(order)

    def create_support_level(self, price: float, size: float = 5000):
        """Create artificial support level at price (for testing)."""
        for i in range(10):
            order = Order(
                price=price - i * self.config.tick_size,
                size=size / 10,
                side=OrderSide.BID,
                timestamp=self._step_count,
                order_type=OrderType.MAKER
            )
            self.orderbook.add_order(order)

    def create_resistance_level(self, price: float, size: float = 5000):
        """Create artificial resistance level at price (for testing)."""
        for i in range(10):
            order = Order(
                price=price + i * self.config.tick_size,
                size=size / 10,
                side=OrderSide.ASK,
                timestamp=self._step_count,
                order_type=OrderType.MAKER
            )
            self.orderbook.add_order(order)


class HistoricalOrderBookSimulator(OrderBookSimulator):
    """
    Simulator that tracks order history for S/R calculation.

    Accumulates snapshots that can be analyzed to find
    mathematically-derived support and resistance levels.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        super().__init__(config)
        self.level_history: Dict[float, float] = {}  # price -> cumulative volume

    def step(self, n_orders: int = 10) -> Dict:
        """Step and track order accumulation by price level."""
        result = super().step(n_orders)

        # Track volume at each price level
        snapshot = result['snapshot']
        for price, volume in snapshot.bid_depth.items():
            self.level_history[price] = self.level_history.get(price, 0) + volume
        for price, volume in snapshot.ask_depth.items():
            self.level_history[price] = self.level_history.get(price, 0) + volume

        return result

    def get_high_volume_levels(self, n_levels: int = 10) -> List[Tuple[float, float]]:
        """
        Get price levels with highest historical volume.

        These are mathematically-derived support/resistance levels.
        """
        sorted_levels = sorted(
            self.level_history.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_levels[:n_levels]

    def get_nearest_support(self, price: float) -> Optional[Tuple[float, float]]:
        """Find nearest high-volume level below current price."""
        levels = [
            (p, v) for p, v in self.level_history.items()
            if p < price and v > np.percentile(list(self.level_history.values()), 75)
        ]
        if not levels:
            return None
        return min(levels, key=lambda x: price - x[0])

    def get_nearest_resistance(self, price: float) -> Optional[Tuple[float, float]]:
        """Find nearest high-volume level above current price."""
        levels = [
            (p, v) for p, v in self.level_history.items()
            if p > price and v > np.percentile(list(self.level_history.values()), 75)
        ]
        if not levels:
            return None
        return min(levels, key=lambda x: x[0] - price)


def demo_simulation():
    """Demo the order book simulator."""
    print("Starting Order Book Simulation Demo")
    print("=" * 50)

    config = SimulatorConfig(
        n_traders=500,
        maker_ratio=0.20,  # Key Crayfer insight
        initial_price=100.0
    )

    sim = HistoricalOrderBookSimulator(config)

    # Run simulation
    results = sim.run(n_steps=100, orders_per_step=20)

    # Analyze results
    prices = sim.get_price_series()
    returns = sim.get_returns()

    print(f"\nSimulation Results:")
    print(f"  Initial Price: ${config.initial_price:.2f}")
    print(f"  Final Price:   ${prices[-1]:.2f}")
    print(f"  Total Return:  {(prices[-1]/prices[0] - 1)*100:.2f}%")
    print(f"  Volatility:    {np.std(returns) * np.sqrt(252) * 100:.1f}%")

    print(f"\nOrder Book Stats:")
    print(f"  Spread:      ${sim.orderbook.get_spread():.4f}")
    print(f"  Spread %:    {sim.orderbook.get_spread_pct()*100:.3f}%")
    print(f"  Imbalance:   {sim.orderbook.get_bid_ask_imbalance():.2f}")

    print(f"\nHigh Volume Levels (Support/Resistance):")
    for price, volume in sim.get_high_volume_levels(5):
        print(f"  ${price:.2f} - Volume: {volume:.0f}")

    support = sim.get_nearest_support(prices[-1])
    resistance = sim.get_nearest_resistance(prices[-1])

    if support:
        print(f"\nNearest Support: ${support[0]:.2f} (volume: {support[1]:.0f})")
    if resistance:
        print(f"Nearest Resistance: ${resistance[0]:.2f} (volume: {resistance[1]:.0f})")

    print(f"\nOrder Book Features:")
    features = sim.get_order_book_features()
    feature_names = [
        'bid_ask_imbalance', 'spread_pct',
        'depth_1pct_bid', 'depth_1pct_ask',
        'depth_2pct_bid', 'depth_2pct_ask',
        'largest_gap_below', 'largest_gap_above',
        'nearest_support', 'nearest_resistance',
        'support_strength', 'resistance_strength',
        'order_flow_imbalance', 'vwap_distance', 'depth_slope'
    ]
    for name, val in zip(feature_names, features):
        print(f"  {name}: {val:.4f}")


if __name__ == '__main__':
    demo_simulation()
