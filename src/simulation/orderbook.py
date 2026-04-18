"""
Order Book Data Structure

Core data structures for order book simulation based on Crayfer's insights:
- Orders are either Makers (place limit orders) or Takers (execute market orders)
- Price moves when gaps exist in the order book
- Support/Resistance emerges from order accumulation

This is NOT psychology-based - it's pure mathematics of supply/demand.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import bisect
import numpy as np


class OrderSide(Enum):
    BID = "bid"  # Buy order
    ASK = "ask"  # Sell order


class OrderType(Enum):
    MAKER = "maker"  # Limit order (sits in book)
    TAKER = "taker"  # Market order (executes immediately)


@dataclass
class Order:
    """Single order in the book."""
    price: float
    size: float
    side: OrderSide
    timestamp: int
    order_type: OrderType
    order_id: int = 0

    def __lt__(self, other):
        return self.price < other.price


@dataclass
class Trade:
    """Executed trade from order matching."""
    price: float
    size: float
    buyer_id: int
    seller_id: int
    timestamp: int
    aggressor: OrderSide  # Who initiated the trade


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state at a point in time."""
    timestamp: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bid_depth: Dict[float, float]  # price -> total size
    ask_depth: Dict[float, float]  # price -> total size
    total_bid_volume: float
    total_ask_volume: float


class OrderBook:
    """
    Order book implementation.

    Key insight from Crayfer:
    - Patterns emerge from the mathematics of order accumulation
    - Gaps in the book allow rapid price movement
    - Dense areas create support/resistance
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: List[Order] = []  # Sorted descending by price
        self.asks: List[Order] = []  # Sorted ascending by price
        self.order_history: List[Order] = []
        self.trade_history: List[Trade] = []
        self.snapshots: List[OrderBookSnapshot] = []
        self._next_order_id = 0
        self._current_timestamp = 0

    def _get_next_id(self) -> int:
        self._next_order_id += 1
        return self._next_order_id

    def add_order(self, order: Order) -> List[Trade]:
        """
        Add order to book, execute if crosses.

        Returns list of trades if order matches with existing orders.
        """
        order.order_id = self._get_next_id()
        order.timestamp = self._current_timestamp
        self.order_history.append(order)

        trades = []

        if order.order_type == OrderType.TAKER:
            # Market order - execute immediately against book
            trades = self._execute_market_order(order)
        else:
            # Limit order - check for cross, then add to book
            trades = self._execute_limit_order(order)

        return trades

    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against resting orders."""
        trades = []
        remaining_size = order.size

        if order.side == OrderSide.BID:
            # Buy order - hit asks
            while remaining_size > 0 and self.asks:
                best_ask = self.asks[0]
                trade_size = min(remaining_size, best_ask.size)

                trade = Trade(
                    price=best_ask.price,
                    size=trade_size,
                    buyer_id=order.order_id,
                    seller_id=best_ask.order_id,
                    timestamp=self._current_timestamp,
                    aggressor=OrderSide.BID
                )
                trades.append(trade)
                self.trade_history.append(trade)

                remaining_size -= trade_size
                best_ask.size -= trade_size

                if best_ask.size <= 0:
                    self.asks.pop(0)

        else:
            # Sell order - hit bids
            while remaining_size > 0 and self.bids:
                best_bid = self.bids[0]
                trade_size = min(remaining_size, best_bid.size)

                trade = Trade(
                    price=best_bid.price,
                    size=trade_size,
                    buyer_id=best_bid.order_id,
                    seller_id=order.order_id,
                    timestamp=self._current_timestamp,
                    aggressor=OrderSide.ASK
                )
                trades.append(trade)
                self.trade_history.append(trade)

                remaining_size -= trade_size
                best_bid.size -= trade_size

                if best_bid.size <= 0:
                    self.bids.pop(0)

        return trades

    def _execute_limit_order(self, order: Order) -> List[Trade]:
        """Execute limit order - match if crosses, else add to book."""
        trades = []
        remaining_size = order.size

        if order.side == OrderSide.BID:
            # Buy limit - check if crosses best ask
            while remaining_size > 0 and self.asks and order.price >= self.asks[0].price:
                best_ask = self.asks[0]
                trade_size = min(remaining_size, best_ask.size)

                trade = Trade(
                    price=best_ask.price,
                    size=trade_size,
                    buyer_id=order.order_id,
                    seller_id=best_ask.order_id,
                    timestamp=self._current_timestamp,
                    aggressor=OrderSide.BID
                )
                trades.append(trade)
                self.trade_history.append(trade)

                remaining_size -= trade_size
                best_ask.size -= trade_size

                if best_ask.size <= 0:
                    self.asks.pop(0)

            # Add remaining to book
            if remaining_size > 0:
                order.size = remaining_size
                self._insert_bid(order)

        else:
            # Sell limit - check if crosses best bid
            while remaining_size > 0 and self.bids and order.price <= self.bids[0].price:
                best_bid = self.bids[0]
                trade_size = min(remaining_size, best_bid.size)

                trade = Trade(
                    price=best_bid.price,
                    size=trade_size,
                    buyer_id=best_bid.order_id,
                    seller_id=order.order_id,
                    timestamp=self._current_timestamp,
                    aggressor=OrderSide.ASK
                )
                trades.append(trade)
                self.trade_history.append(trade)

                remaining_size -= trade_size
                best_bid.size -= trade_size

                if best_bid.size <= 0:
                    self.bids.pop(0)

            # Add remaining to book
            if remaining_size > 0:
                order.size = remaining_size
                self._insert_ask(order)

        return trades

    def _insert_bid(self, order: Order):
        """Insert bid order maintaining descending price sort."""
        # Binary search for insertion point (descending)
        prices = [-b.price for b in self.bids]
        idx = bisect.bisect_left(prices, -order.price)
        self.bids.insert(idx, order)

    def _insert_ask(self, order: Order):
        """Insert ask order maintaining ascending price sort."""
        prices = [a.price for a in self.asks]
        idx = bisect.bisect_left(prices, order.price)
        self.asks.insert(idx, order)

    def get_best_bid(self) -> Optional[float]:
        """Highest buy price."""
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Lowest sell price."""
        return self.asks[0].price if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        """Midpoint between best bid and ask."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    def get_spread(self) -> float:
        """Bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return float('inf')

    def get_spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.get_mid_price()
        if mid and mid > 0:
            return self.get_spread() / mid
        return 0.0

    def get_bid_ask_imbalance(self) -> float:
        """
        Imbalance between bid and ask volume.

        > 1: More buy pressure
        < 1: More sell pressure
        = 1: Balanced
        """
        total_bid = sum(o.size for o in self.bids)
        total_ask = sum(o.size for o in self.asks)
        if total_ask > 0:
            return total_bid / total_ask
        return 1.0

    def get_depth_at_level(self, pct: float, side: str) -> float:
        """
        Get order volume within pct% of best price.

        Args:
            pct: Percentage from best price (0.01 = 1%)
            side: 'bid' or 'ask'
        """
        if side == 'bid':
            best = self.get_best_bid()
            if best is None:
                return 0.0
            threshold = best * (1 - pct)
            return sum(o.size for o in self.bids if o.price >= threshold)
        else:
            best = self.get_best_ask()
            if best is None:
                return 0.0
            threshold = best * (1 + pct)
            return sum(o.size for o in self.asks if o.price <= threshold)

    def find_gaps(self, min_gap_pct: float = 0.005) -> List[Tuple[float, float]]:
        """
        Find price gaps in the order book.

        Gaps are regions with no orders - price can move quickly through gaps.
        This is key to Crayfer's insight about pattern emergence.
        """
        gaps = []

        # Find gaps in asks
        for i in range(len(self.asks) - 1):
            current = self.asks[i].price
            next_price = self.asks[i + 1].price
            gap = (next_price - current) / current
            if gap > min_gap_pct:
                gaps.append((current, next_price))

        # Find gaps in bids
        for i in range(len(self.bids) - 1):
            current = self.bids[i].price
            next_price = self.bids[i + 1].price
            gap = (current - next_price) / current
            if gap > min_gap_pct:
                gaps.append((next_price, current))

        return gaps

    def find_largest_gap_below(self, price: float) -> Optional[Tuple[float, float]]:
        """Find largest gap below current price."""
        gaps = [(g[0], g[1]) for g in self.find_gaps() if g[1] < price]
        if not gaps:
            return None
        return max(gaps, key=lambda g: g[1] - g[0])

    def find_largest_gap_above(self, price: float) -> Optional[Tuple[float, float]]:
        """Find largest gap above current price."""
        gaps = [(g[0], g[1]) for g in self.find_gaps() if g[0] > price]
        if not gaps:
            return None
        return max(gaps, key=lambda g: g[1] - g[0])

    def get_order_flow_imbalance(self, lookback: int = 100) -> float:
        """
        Net buy vs sell pressure from recent trades.

        Positive: More aggressive buyers
        Negative: More aggressive sellers
        """
        recent = self.trade_history[-lookback:] if len(self.trade_history) >= lookback else self.trade_history
        if not recent:
            return 0.0

        buy_volume = sum(t.size for t in recent if t.aggressor == OrderSide.BID)
        sell_volume = sum(t.size for t in recent if t.aggressor == OrderSide.ASK)
        total = buy_volume + sell_volume

        if total > 0:
            return (buy_volume - sell_volume) / total
        return 0.0

    def get_vwap(self, lookback: int = 100) -> float:
        """Volume-weighted average price from recent trades."""
        recent = self.trade_history[-lookback:] if len(self.trade_history) >= lookback else self.trade_history
        if not recent:
            mid = self.get_mid_price()
            return mid if mid else 0.0

        total_volume = sum(t.size for t in recent)
        if total_volume > 0:
            return sum(t.price * t.size for t in recent) / total_volume
        return self.get_mid_price() or 0.0

    def get_depth_slope(self, levels: int = 5) -> float:
        """
        Rate of depth change away from best price.

        High slope = Thin book (price moves easily)
        Low slope = Thick book (price is stable)
        """
        if not self.bids or not self.asks:
            return 0.0

        # Sample depth at multiple levels
        depths = []
        for i in range(1, levels + 1):
            pct = i * 0.005  # 0.5%, 1%, 1.5%, 2%, 2.5%
            bid_depth = self.get_depth_at_level(pct, 'bid')
            ask_depth = self.get_depth_at_level(pct, 'ask')
            depths.append(bid_depth + ask_depth)

        if len(depths) < 2:
            return 0.0

        # Calculate slope (how fast depth grows)
        x = np.arange(len(depths))
        slope = np.polyfit(x, depths, 1)[0]

        return slope

    def take_snapshot(self) -> OrderBookSnapshot:
        """Create snapshot of current book state."""
        bid_depth = {}
        for o in self.bids:
            bid_depth[o.price] = bid_depth.get(o.price, 0) + o.size

        ask_depth = {}
        for o in self.asks:
            ask_depth[o.price] = ask_depth.get(o.price, 0) + o.size

        snapshot = OrderBookSnapshot(
            timestamp=self._current_timestamp,
            best_bid=self.get_best_bid() or 0,
            best_ask=self.get_best_ask() or float('inf'),
            mid_price=self.get_mid_price() or 0,
            spread=self.get_spread(),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_bid_volume=sum(o.size for o in self.bids),
            total_ask_volume=sum(o.size for o in self.asks),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def advance_time(self, ticks: int = 1):
        """Advance simulation time."""
        self._current_timestamp += ticks

    def clear(self):
        """Clear the order book."""
        self.bids = []
        self.asks = []

    def get_price_levels(self, n_levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get top n price levels for visualization."""
        return {
            'bids': [(o.price, o.size) for o in self.bids[:n_levels]],
            'asks': [(o.price, o.size) for o in self.asks[:n_levels]],
        }
