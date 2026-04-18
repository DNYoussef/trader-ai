# Crayfer-Inspired Improvements Plan

## Executive Summary

Based on analysis of Crayfer's market simulation and AI trading videos, we will implement 7 major improvements to our trading system. The core insight is that **patterns emerge from order book mathematics, not psychology**, and our current system doesn't model this mechanism.

## Current System Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No order book simulation | Can't model support/resistance emergence | Critical |
| No order book features | Missing predictive liquidity information | High |
| No price prediction | Only predict allocation, not price direction | High |
| No unrealized P&L tracking | Model could learn to hide losses | Medium |
| 5-day horizon too long | Too much psychology influence | High |
| Single-shot inference | No recursive prediction capability | Medium |

---

## Phase 1: Order Book Simulator (Foundation)

### 1.1 Core Architecture

```
src/simulation/orderbook_simulator.py

Classes:
- Order: Dataclass for individual orders (price, size, side, timestamp)
- OrderBook: Manages bids/asks, matching, gaps
- OrderBookSimulator: Full market simulation with random traders
- SupportResistanceCalculator: Derives S/R from order history
```

### 1.2 Key Components

#### Order Book Structure
```python
@dataclass
class Order:
    price: float
    size: float
    side: str  # 'bid' or 'ask'
    timestamp: int
    order_type: str  # 'maker' or 'taker'

class OrderBook:
    bids: List[Order]  # Buy orders (sorted descending)
    asks: List[Order]  # Sell orders (sorted ascending)

    def add_order(order)
    def match_orders() -> List[Trade]
    def get_best_bid() -> float
    def get_best_ask() -> float
    def get_spread() -> float
    def get_depth_at_level(pct) -> float
    def find_gaps() -> List[Tuple[float, float]]
```

#### Simulation Logic (from Crayfer)
```python
class OrderBookSimulator:
    def __init__(self):
        self.maker_ratio = 0.2  # KEY: Low maker ratio allows price movement
        self.price_variance = 0.01  # Orders +/- 1% of current price

    def step(self):
        # 1. Random traders generate orders around current price
        # 2. 20% makers (sit in book), 80% takers (execute immediately)
        # 3. Match orders, update price
        # 4. Track order history for S/R calculation
```

### 1.3 Support/Resistance Emergence

The key insight: Orders accumulate at price levels. When price returns to that level, it must "get through" accumulated orders, creating resistance.

```python
class SupportResistanceCalculator:
    def __init__(self, order_history: List[OrderBookSnapshot]):
        self.history = order_history

    def calculate_zones(self) -> Dict[str, List[float]]:
        """
        Find price levels where orders accumulated historically.
        These become predictive support/resistance zones.
        """
        # Count order volume at each price level
        # Levels with high historical volume = S/R zones

    def get_nearest_support(self, current_price) -> float
    def get_nearest_resistance(self, current_price) -> float
    def get_gap_above(self, current_price) -> Tuple[float, float]
    def get_gap_below(self, current_price) -> Tuple[float, float]
```

---

## Phase 2: Order Book Features (15 new features)

### 2.1 Feature List

Add to existing 125 features (now 140 total):

| Index | Feature | Description |
|-------|---------|-------------|
| 125 | bid_ask_imbalance | sum(bids) / sum(asks) |
| 126 | spread_pct | (ask - bid) / mid_price |
| 127 | depth_1pct_bid | Order volume within 1% below price |
| 128 | depth_1pct_ask | Order volume within 1% above price |
| 129 | depth_2pct_bid | Order volume within 2% below price |
| 130 | depth_2pct_ask | Order volume within 2% above price |
| 131 | largest_gap_below | Size of biggest gap below price |
| 132 | largest_gap_above | Size of biggest gap above price |
| 133 | nearest_support | Distance to nearest support level |
| 134 | nearest_resistance | Distance to nearest resistance level |
| 135 | support_strength | Order volume at nearest support |
| 136 | resistance_strength | Order volume at nearest resistance |
| 137 | order_flow_imbalance | Net buy vs sell pressure |
| 138 | vwap_distance | Distance from volume-weighted avg price |
| 139 | order_book_slope | Rate of depth change away from price |

### 2.2 Feature Extractor

```python
# src/data/orderbook_features.py

class OrderBookFeatureExtractor:
    def __init__(self, order_book: OrderBook, sr_calculator: SupportResistanceCalculator):
        self.ob = order_book
        self.sr = sr_calculator

    def extract(self, current_price: float) -> np.ndarray:
        features = np.zeros(15, dtype=np.float32)

        # Imbalance features
        features[0] = self.ob.get_bid_ask_imbalance()
        features[1] = self.ob.get_spread() / current_price

        # Depth features
        features[2] = self.ob.get_depth_at_level(0.01, 'bid')
        features[3] = self.ob.get_depth_at_level(0.01, 'ask')
        features[4] = self.ob.get_depth_at_level(0.02, 'bid')
        features[5] = self.ob.get_depth_at_level(0.02, 'ask')

        # Gap features
        gap_below = self.ob.find_largest_gap_below(current_price)
        gap_above = self.ob.find_largest_gap_above(current_price)
        features[6] = gap_below[1] - gap_below[0] if gap_below else 0
        features[7] = gap_above[1] - gap_above[0] if gap_above else 0

        # S/R features
        features[8] = (current_price - self.sr.get_nearest_support(current_price)) / current_price
        features[9] = (self.sr.get_nearest_resistance(current_price) - current_price) / current_price
        features[10] = self.sr.get_support_strength(current_price)
        features[11] = self.sr.get_resistance_strength(current_price)

        # Flow features
        features[12] = self.ob.get_order_flow_imbalance()
        features[13] = (current_price - self.ob.get_vwap()) / current_price
        features[14] = self.ob.get_depth_slope()

        return features
```

---

## Phase 3: Price Prediction Head

### 3.1 Model Architecture Change

Add a supervised price prediction head alongside the existing allocation head:

```python
# src/models/hybrid_strategy_model.py

class HybridStrategyModel(nn.Module):
    def __init__(self, input_dim=140, hidden_dim=128):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Existing allocation head (RL)
        self.allocation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # SPY, TLT, Cash
            nn.Softmax(dim=-1)
        )

        # NEW: Price prediction head (Supervised)
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict next price change (%)
        )

        # NEW: Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 confidence in prediction
        )

    def forward(self, x):
        features = self.encoder(x)

        return {
            'allocations': self.allocation_head(features),
            'price_prediction': self.price_head(features),  # % change
            'confidence': self.confidence_head(features),
        }
```

### 3.2 Training Strategy

```python
# Hybrid loss: RL for allocation + Supervised for price

def compute_loss(self, output, targets, rewards):
    # 1. RL loss for allocation (existing)
    allocation_loss = self.compute_policy_loss(output['allocations'], rewards)

    # 2. Supervised loss for price prediction (NEW)
    price_loss = F.mse_loss(
        output['price_prediction'],
        targets['actual_price_change']
    )

    # 3. Confidence calibration loss (NEW)
    # Confidence should correlate with prediction accuracy
    prediction_error = torch.abs(output['price_prediction'] - targets['actual_price_change'])
    confidence_loss = F.mse_loss(
        output['confidence'],
        1.0 - prediction_error.clamp(0, 1)
    )

    # Combined loss
    total_loss = allocation_loss + 0.5 * price_loss + 0.1 * confidence_loss

    return total_loss
```

---

## Phase 4: Unrealized P&L Tracking

### 4.1 Problem Statement

From Crayfer: His genetic algorithm bots learned to HIDE losses by never closing losing trades. We must track unrealized P&L to prevent this.

### 4.2 Implementation

```python
# src/simulation/pnl_tracker.py

class PnLTracker:
    def __init__(self):
        self.open_positions = []  # List of (entry_price, size, side)
        self.realized_pnl = 0.0
        self.peak_equity = 0.0

    def open_position(self, price: float, size: float, side: str):
        self.open_positions.append({
            'entry_price': price,
            'size': size,
            'side': side,
            'timestamp': time.time()
        })

    def close_position(self, idx: int, exit_price: float):
        pos = self.open_positions.pop(idx)
        if pos['side'] == 'long':
            pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['size']
        self.realized_pnl += pnl
        return pnl

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate P&L on positions that haven't been closed."""
        unrealized = 0.0
        for pos in self.open_positions:
            if pos['side'] == 'long':
                unrealized += (current_price - pos['entry_price']) / pos['entry_price'] * pos['size']
            else:
                unrealized += (pos['entry_price'] - current_price) / pos['entry_price'] * pos['size']
        return unrealized

    def get_total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.get_unrealized_pnl(current_price)

    def get_max_drawdown(self, current_price: float) -> float:
        current_equity = self.get_total_pnl(current_price)
        self.peak_equity = max(self.peak_equity, current_equity)
        return (self.peak_equity - current_equity) / (self.peak_equity + 1e-8)
```

### 4.3 Reward Modification

```python
def compute_reward_with_unrealized(self, allocations, current_price, pnl_tracker):
    realized = pnl_tracker.realized_pnl
    unrealized = pnl_tracker.get_unrealized_pnl(current_price)

    # Base reward from total P&L
    base_reward = realized + unrealized

    # PENALTY for large unrealized losses (prevent hiding)
    if unrealized < -0.02:  # More than 2% unrealized loss
        unrealized_penalty = unrealized * 2.0  # 2x penalty
    else:
        unrealized_penalty = 0.0

    # PENALTY for holding losing positions too long
    stale_loss_penalty = 0.0
    for pos in pnl_tracker.open_positions:
        pos_pnl = pnl_tracker.get_position_pnl(pos, current_price)
        pos_age = time.time() - pos['timestamp']
        if pos_pnl < 0 and pos_age > 86400:  # Losing position > 1 day
            stale_loss_penalty -= 0.01 * pos_age / 86400

    return base_reward + unrealized_penalty + stale_loss_penalty
```

---

## Phase 5: 1-Day Candlestick Timeframe

### 5.1 Rationale

From Crayfer's quant contact: "Firms use AI on shorter timeframes because the further you zoom out, the more psychology affects price."

Current: 5-day horizon (too much psychology)
New: 1-day horizon (more math, less psychology)

### 5.2 Data Changes

```python
# src/data/daily_data_loader.py

class DailyDataLoader:
    def __init__(self, ticker='SPY'):
        self.ticker = ticker

    def fetch_daily_data(self, start_date, end_date):
        """Fetch 1-day OHLCV data."""
        df = yf.download(self.ticker, start=start_date, end=end_date, interval='1d')
        return df

    def compute_daily_features(self, df, idx):
        """
        Compute features for 1-day prediction.
        Focus on intraday patterns that resolve within a day.
        """
        features = []

        # Price action features
        features.append(df['Close'].iloc[idx] / df['Open'].iloc[idx] - 1)  # Daily return
        features.append((df['High'].iloc[idx] - df['Low'].iloc[idx]) / df['Open'].iloc[idx])  # Range
        features.append((df['Close'].iloc[idx] - df['Low'].iloc[idx]) / (df['High'].iloc[idx] - df['Low'].iloc[idx] + 1e-8))  # Close position in range

        # Short-term momentum (1-5 days, not weeks)
        for period in [1, 2, 3, 5]:
            features.append(df['Close'].iloc[idx] / df['Close'].iloc[idx-period] - 1)

        # Volume features
        features.append(df['Volume'].iloc[idx] / df['Volume'].iloc[idx-5:idx].mean())

        # Volatility (short-term)
        features.append(df['Close'].pct_change().iloc[idx-5:idx].std() * np.sqrt(252))

        return np.array(features, dtype=np.float32)
```

### 5.3 Simulator Changes

```python
# Change horizon from 5 days to 1 day

class DailySimulator:
    def __init__(self, spy_returns, tlt_returns, horizon_days=1):  # Changed from 5
        self.horizon_days = horizon_days

        # Pre-compute 1-day returns (no cumulative needed)
        self.spy_daily = spy_returns
        self.tlt_daily = tlt_returns

    def get_reward(self, allocations, time_indices):
        # Simple 1-day return
        spy_ret = self.spy_daily[time_indices]
        tlt_ret = self.tlt_daily[time_indices]

        portfolio_return = (
            allocations[:, 0] * spy_ret +
            allocations[:, 1] * tlt_ret
        )

        return portfolio_return
```

### 5.4 Training Schedule Change

```python
# More frequent rebalancing for 1-day horizon

class DailyTrainer:
    def __init__(self):
        self.rebalance_freq = 1  # Changed from 5 days
        self.prediction_horizon = 1  # Predict next day only

    def train_step(self, features, time_idx):
        # Get prediction for next day
        output = self.model(features)

        # Get actual next-day return
        actual_return = self.get_next_day_return(time_idx)

        # Train both heads
        allocation_loss = self.train_allocation(output, actual_return)
        price_loss = self.train_price_prediction(output, actual_return)

        return allocation_loss + price_loss
```

---

## Phase 6: GPT-Style Prediction Chaining

### 6.1 Concept

Like ChatGPT predicts the next word and feeds it back, we predict the next price and feed it back for multi-step prediction.

### 6.2 Implementation

```python
# src/models/recursive_predictor.py

class RecursivePredictor:
    def __init__(self, model):
        self.model = model

    def predict_multi_step(self, initial_features: torch.Tensor, n_steps: int = 5):
        """
        Predict n_steps into the future by chaining predictions.
        Like GPT predicting word by word.
        """
        predictions = []
        confidences = []
        current_features = initial_features.clone()

        for step in range(n_steps):
            with torch.no_grad():
                output = self.model(current_features)

            pred_price_change = output['price_prediction']
            confidence = output['confidence']
            allocation = output['allocations']

            predictions.append({
                'step': step + 1,
                'price_change': pred_price_change.item(),
                'confidence': confidence.item(),
                'allocation': allocation.cpu().numpy(),
            })
            confidences.append(confidence.item())

            # Update features with prediction (feed back in)
            current_features = self._update_features(
                current_features,
                pred_price_change,
                allocation
            )

        # Aggregate multi-step prediction
        return {
            'predictions': predictions,
            'avg_confidence': np.mean(confidences),
            'cumulative_price_change': sum(p['price_change'] for p in predictions),
            'final_allocation': predictions[-1]['allocation'],
        }

    def _update_features(self, features, price_change, allocation):
        """
        Update feature vector with new prediction.
        Simulates what features would look like after predicted move.
        """
        new_features = features.clone()

        # Update return features based on predicted price change
        new_features[0, 1] = price_change  # 1-day return
        new_features[0, 2] = new_features[0, 1] + price_change  # 2-day return
        # ... etc

        # Update momentum features
        new_features[0, 10] = 1.0 if price_change > 0 else 0.0  # Up day

        # Update allocation context
        new_features[0, -3:] = torch.tensor(allocation)

        return new_features
```

### 6.3 Training for Recursive Prediction

```python
def train_recursive(self, features, future_returns, n_steps=5):
    """
    Train model to be accurate across multiple recursive steps.
    """
    total_loss = 0.0
    current_features = features.clone()

    for step in range(n_steps):
        output = self.model(current_features)

        # Get actual return for this step
        actual_return = future_returns[:, step]

        # Supervised loss on price prediction
        step_loss = F.mse_loss(output['price_prediction'].squeeze(), actual_return)

        # Discount future steps (closer predictions more important)
        discount = 0.9 ** step
        total_loss += step_loss * discount

        # Update features for next step
        current_features = self._update_features(
            current_features,
            output['price_prediction'],
            output['allocations']
        )

    return total_loss / n_steps
```

---

## Phase 7: Integration & Training Pipeline

### 7.1 New Training Script

```python
# scripts/train_crayfer_improved.py

def main():
    # 1. Initialize order book simulator
    ob_sim = OrderBookSimulator(n_traders=1000, maker_ratio=0.2)

    # 2. Load 1-day data
    data_loader = DailyDataLoader()
    spy_data = data_loader.fetch_daily_data('2020-01-01', '2024-12-31')

    # 3. Initialize model with new features
    model = HybridStrategyModel(
        input_dim=140,  # 125 original + 15 order book
        hidden_dim=128
    )

    # 4. Initialize P&L tracker
    pnl_tracker = PnLTracker()

    # 5. Training loop
    for epoch in range(200):
        for batch in data_loader:
            # Extract features including order book
            market_features = extract_market_features(batch)
            ob_features = ob_extractor.extract(batch['price'])
            features = np.concatenate([market_features, ob_features])

            # Forward pass
            output = model(torch.tensor(features))

            # Compute hybrid loss
            loss = compute_hybrid_loss(
                output,
                actual_returns=batch['next_day_return'],
                pnl_tracker=pnl_tracker
            )

            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate recursive prediction
        recursive_eval = recursive_predictor.predict_multi_step(test_features, n_steps=5)
        print(f"Epoch {epoch}: Recursive accuracy = {recursive_eval['avg_confidence']:.2%}")
```

### 7.2 File Structure

```
src/
  simulation/
    orderbook_simulator.py      # NEW: Order book simulation
    orderbook.py                # NEW: Order book data structure
    support_resistance.py       # NEW: S/R calculation
    pnl_tracker.py              # NEW: Unrealized P&L tracking
    daily_simulator.py          # NEW: 1-day horizon simulator
  data/
    orderbook_features.py       # NEW: 15 order book features
    daily_data_loader.py        # NEW: Daily data loading
    extended_features.py        # UPDATED: 140 features total
  models/
    hybrid_strategy_model.py    # NEW: Price + allocation heads
    recursive_predictor.py      # NEW: GPT-style chaining

scripts/
  train_crayfer_improved.py     # NEW: Main training script
  evaluate_recursive.py         # NEW: Multi-step evaluation
  backtest_daily.py             # NEW: 1-day backtest
```

---

## Implementation Order

| Phase | Task | Dependencies | Estimated Effort |
|-------|------|--------------|------------------|
| 1 | Order Book Simulator | None | 4-6 hours |
| 2 | Order Book Features | Phase 1 | 2-3 hours |
| 3 | Price Prediction Head | None | 2-3 hours |
| 4 | Unrealized P&L Tracking | None | 2-3 hours |
| 5 | 1-Day Timeframe | Phases 1-4 | 3-4 hours |
| 6 | GPT-Style Chaining | Phase 3 | 3-4 hours |
| 7 | Integration | All | 4-6 hours |

**Total Estimated: 20-29 hours**

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Prediction Horizon | 5 days | 1 day |
| Feature Count | 125 | 140 |
| Model Outputs | 1 (allocation) | 3 (allocation, price, confidence) |
| P&L Tracking | Realized only | Realized + Unrealized |
| Prediction Type | Single-shot | Recursive (5-step) |
| S/R Awareness | None | Mathematical calculation |

---

## Next Steps

1. **Implement Phase 1** (Order Book Simulator) - Foundation for everything
2. **Test on synthetic data** - Verify patterns emerge mathematically
3. **Add features incrementally** - Validate each improves performance
4. **Train hybrid model** - Both price prediction and allocation
5. **Evaluate recursive accuracy** - Test multi-step prediction
6. **Backtest on real data** - Compare to current system
