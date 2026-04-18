"""
Order Book Feature Extractor

Extracts 15 order book features for the ML model.
Extends the existing 125 features to 140 total.

Features capture:
- Order flow imbalance
- Depth at various levels
- Gaps in the book (predict rapid movement)
- Support/Resistance distances and strengths
- VWAP deviation
"""

import numpy as np
from typing import Optional

from src.simulation.orderbook import OrderBook
from src.simulation.orderbook_simulator import OrderBookSimulator
from src.simulation.support_resistance import SupportResistanceCalculator


# Feature indices (125-139)
ORDERBOOK_FEATURE_START = 125
ORDERBOOK_FEATURE_COUNT = 15
TOTAL_FEATURES_WITH_ORDERBOOK = 140

# Feature names for documentation
ORDERBOOK_FEATURE_NAMES = [
    'bid_ask_imbalance',      # 125: sum(bids) / sum(asks)
    'spread_pct',             # 126: (ask - bid) / mid_price
    'depth_1pct_bid',         # 127: Order volume within 1% below price
    'depth_1pct_ask',         # 128: Order volume within 1% above price
    'depth_2pct_bid',         # 129: Order volume within 2% below price
    'depth_2pct_ask',         # 130: Order volume within 2% above price
    'largest_gap_below',      # 131: Size of biggest gap below price
    'largest_gap_above',      # 132: Size of biggest gap above price
    'nearest_support',        # 133: Distance to nearest support level
    'nearest_resistance',     # 134: Distance to nearest resistance level
    'support_strength',       # 135: Order volume at nearest support
    'resistance_strength',    # 136: Order volume at nearest resistance
    'order_flow_imbalance',   # 137: Net buy vs sell pressure
    'vwap_distance',          # 138: Distance from volume-weighted avg price
    'order_book_slope',       # 139: Rate of depth change away from price
]


class OrderBookFeatureExtractor:
    """
    Extracts 15 order book features for ML training.

    Can work with:
    1. Live order book data (from broker API)
    2. Simulated order book (from OrderBookSimulator)
    """

    def __init__(
        self,
        orderbook: Optional[OrderBook] = None,
        sr_calculator: Optional[SupportResistanceCalculator] = None,
        volume_normalization: float = 10000.0,  # Normalize volumes by this
    ):
        self.orderbook = orderbook
        self.sr_calculator = sr_calculator
        self.volume_norm = volume_normalization

    def set_orderbook(self, orderbook: OrderBook):
        """Set the order book to extract features from."""
        self.orderbook = orderbook

    def set_sr_calculator(self, sr_calculator: SupportResistanceCalculator):
        """Set the S/R calculator."""
        self.sr_calculator = sr_calculator

    def extract(self, current_price: float) -> np.ndarray:
        """
        Extract 15 order book features.

        Args:
            current_price: Current market price

        Returns:
            (15,) array of order book features
        """
        features = np.zeros(ORDERBOOK_FEATURE_COUNT, dtype=np.float32)

        if self.orderbook is None:
            return features

        ob = self.orderbook

        # === Imbalance Features ===
        features[0] = ob.get_bid_ask_imbalance() - 1.0  # Center around 0
        features[1] = ob.get_spread_pct() * 100  # Convert to percentage points

        # === Depth Features (normalized) ===
        features[2] = ob.get_depth_at_level(0.01, 'bid') / self.volume_norm
        features[3] = ob.get_depth_at_level(0.01, 'ask') / self.volume_norm
        features[4] = ob.get_depth_at_level(0.02, 'bid') / self.volume_norm
        features[5] = ob.get_depth_at_level(0.02, 'ask') / self.volume_norm

        # === Gap Features ===
        gap_below = ob.find_largest_gap_below(current_price)
        gap_above = ob.find_largest_gap_above(current_price)

        if gap_below:
            features[6] = (gap_below[1] - gap_below[0]) / current_price
        else:
            features[6] = 0.0

        if gap_above:
            features[7] = (gap_above[1] - gap_above[0]) / current_price
        else:
            features[7] = 0.0

        # === Support/Resistance Features ===
        if self.sr_calculator:
            sr_features = self.sr_calculator.get_sr_features(current_price)
            features[8] = sr_features[0]   # Distance to support
            features[9] = sr_features[1]   # Distance to resistance
            features[10] = sr_features[2]  # Support strength
            features[11] = sr_features[3]  # Resistance strength
        else:
            # Estimate from order book directly
            features[8] = 0.02  # Default 2% to support
            features[9] = 0.02  # Default 2% to resistance
            features[10] = 0.5  # Default mid strength
            features[11] = 0.5

        # === Flow Features ===
        features[12] = ob.get_order_flow_imbalance()

        vwap = ob.get_vwap()
        if vwap > 0:
            features[13] = (current_price - vwap) / vwap
        else:
            features[13] = 0.0

        features[14] = ob.get_depth_slope() / self.volume_norm

        # Clip to reasonable range
        features = np.clip(features, -10.0, 10.0)

        return features

    def extract_from_simulator(
        self,
        simulator: OrderBookSimulator,
        sr_calculator: Optional[SupportResistanceCalculator] = None
    ) -> np.ndarray:
        """
        Extract features directly from simulator.

        Args:
            simulator: OrderBookSimulator instance
            sr_calculator: Optional S/R calculator

        Returns:
            (15,) array of features
        """
        self.orderbook = simulator.orderbook
        self.sr_calculator = sr_calculator
        return self.extract(simulator.current_price)


class ExtendedOrderBookFeatureExtractor:
    """
    Combines market features (110) + portfolio context (15) + order book (15) = 140 total.

    This is the full feature extractor for the enhanced model.
    """

    def __init__(
        self,
        market_feature_extractor,  # Your existing market feature extractor
        portfolio_context_extractor,  # From portfolio_context_features.py
        orderbook_extractor: Optional[OrderBookFeatureExtractor] = None,
    ):
        self.market_extractor = market_feature_extractor
        self.portfolio_extractor = portfolio_context_extractor
        self.orderbook_extractor = orderbook_extractor or OrderBookFeatureExtractor()

    def extract_all(
        self,
        market_data: dict,
        portfolio_state: dict,
        orderbook: Optional[OrderBook] = None,
        sr_calculator: Optional[SupportResistanceCalculator] = None,
        current_price: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract all 140 features.

        Args:
            market_data: Dict with SPY, TLT, VIX data
            portfolio_state: Dict with capital, milestone info
            orderbook: Optional OrderBook for order book features
            sr_calculator: Optional S/R calculator
            current_price: Current price (for order book features)

        Returns:
            (140,) array of features
        """
        # Market features (110)
        market_features = self.market_extractor.extract(market_data)

        # Portfolio context features (15)
        portfolio_features = self.portfolio_extractor.extract(
            capital=portfolio_state.get('capital', 200),
            peak_capital=portfolio_state.get('peak_capital', 200),
            days_at_milestone=portfolio_state.get('days_at_milestone', 0),
            milestones_achieved=portfolio_state.get('milestones_achieved', 0),
        )

        # Order book features (15)
        if orderbook:
            self.orderbook_extractor.set_orderbook(orderbook)
            if sr_calculator:
                self.orderbook_extractor.set_sr_calculator(sr_calculator)
            price = current_price or market_data.get('spy_price', 100.0)
            orderbook_features = self.orderbook_extractor.extract(price)
        else:
            orderbook_features = np.zeros(ORDERBOOK_FEATURE_COUNT, dtype=np.float32)

        # Concatenate all features
        all_features = np.concatenate([
            market_features,      # 0-109
            portfolio_features,   # 110-124
            orderbook_features,   # 125-139
        ])

        return all_features


def generate_synthetic_orderbook_features(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic order book features for training when real data unavailable.

    Uses realistic distributions based on typical market conditions.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        (n_samples, 15) array of synthetic features
    """
    np.random.seed(seed)
    features = np.zeros((n_samples, ORDERBOOK_FEATURE_COUNT), dtype=np.float32)

    # bid_ask_imbalance: typically -1 to 1 (centered at 0)
    features[:, 0] = np.random.normal(0, 0.3, n_samples)

    # spread_pct: typically 0.01% to 0.1%
    features[:, 1] = np.abs(np.random.normal(0.03, 0.02, n_samples))

    # depth features: typically 0-2 (normalized)
    for i in range(2, 6):
        features[:, i] = np.abs(np.random.normal(0.5, 0.3, n_samples))

    # gap features: typically 0-0.05 (% of price)
    features[:, 6] = np.abs(np.random.exponential(0.01, n_samples))
    features[:, 7] = np.abs(np.random.exponential(0.01, n_samples))

    # S/R distance: typically 0.01-0.05 (1-5%)
    features[:, 8] = np.abs(np.random.normal(0.02, 0.01, n_samples))
    features[:, 9] = np.abs(np.random.normal(0.02, 0.01, n_samples))

    # S/R strength: 0-1
    features[:, 10] = np.random.beta(2, 2, n_samples)
    features[:, 11] = np.random.beta(2, 2, n_samples)

    # order flow imbalance: -1 to 1
    features[:, 12] = np.random.normal(0, 0.2, n_samples)

    # vwap distance: typically -0.02 to 0.02
    features[:, 13] = np.random.normal(0, 0.005, n_samples)

    # depth slope: normalized
    features[:, 14] = np.random.normal(0.1, 0.05, n_samples)

    # Clip to reasonable range
    features = np.clip(features, -10.0, 10.0)

    return features


def add_orderbook_correlation(
    market_features: np.ndarray,
    orderbook_features: np.ndarray,
    vix_idx: int = 0,
    spy_ret_idx: int = 2,
) -> np.ndarray:
    """
    Add correlation between order book features and market features.

    Makes synthetic order book features more realistic by correlating
    with market conditions.

    Args:
        market_features: (n_samples, 110) market features
        orderbook_features: (n_samples, 15) order book features
        vix_idx: Index of VIX feature in market features
        spy_ret_idx: Index of SPY return feature

    Returns:
        Correlated order book features
    """
    n_samples = len(market_features)

    # High VIX -> wider spreads, more imbalance
    vix_effect = market_features[:, vix_idx] - 0.2  # VIX normalized
    orderbook_features[:, 1] += vix_effect * 0.01  # Wider spread
    orderbook_features[:, 0] += np.sign(vix_effect) * 0.1  # More imbalance

    # Positive returns -> positive order flow
    ret_effect = market_features[:, spy_ret_idx]
    orderbook_features[:, 12] += ret_effect * 2.0  # Order flow follows price

    # Large moves -> bigger gaps
    abs_ret = np.abs(ret_effect)
    orderbook_features[:, 6] += abs_ret * 0.5
    orderbook_features[:, 7] += abs_ret * 0.5

    # Clip
    orderbook_features = np.clip(orderbook_features, -10.0, 10.0)

    return orderbook_features


# Feature documentation
def get_feature_documentation() -> str:
    """Return documentation string for order book features."""
    doc = "Order Book Features (indices 125-139):\n\n"
    for i, name in enumerate(ORDERBOOK_FEATURE_NAMES):
        doc += f"  [{125 + i}] {name}\n"
    return doc
