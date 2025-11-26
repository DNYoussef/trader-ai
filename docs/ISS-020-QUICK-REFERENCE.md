# ISS-020: Real-Time Data Feeds - Quick Reference

## Environment Setup

```bash
# Set Alpaca API credentials
export ALPACA_API_KEY="PKXXXXXXXXXXXXX"
export ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxx"

# Optional: Force paper trading (default)
export ALPACA_PAPER=true
```

## Quick Start

### 1. Basic Price Fetching

```python
from src.market.alpaca_data_provider import AlpacaDataProvider
import asyncio

async def get_prices():
    provider = AlpacaDataProvider()
    await provider.connect()

    price = await provider.get_latest_price('SPY')
    print(f"SPY: ${price}")

    await provider.disconnect()

asyncio.run(get_prices())
```

### 2. Real-Time Streaming

```python
async def stream_data():
    provider = AlpacaDataProvider(use_websocket=True)
    await provider.connect()

    # Add callback
    async def on_trade(trade):
        print(f"{trade.symbol}: ${trade.price}")

    provider.add_trade_callback(on_trade)

    # Start streaming
    await provider.start_streaming(['SPY', 'ULTY'])
    await asyncio.sleep(30)  # Stream for 30 seconds

    await provider.stop_streaming()
    await provider.disconnect()

asyncio.run(stream_data())
```

### 3. Terminal Integration

```python
from src.market.alpaca_data_provider import AlpacaDataProvider
from src.trading.terminal_data_provider import TradingTerminalDataProvider

async def run_terminal():
    # Create Alpaca provider
    alpaca = AlpacaDataProvider(use_websocket=True)
    await alpaca.connect()

    # Wire into terminal
    terminal = TradingTerminalDataProvider(
        symbols=['SPY', 'ULTY', 'AMDY'],
        enable_live_data=True,
        alpaca_data_provider=alpaca
    )

    await terminal.start()
    # Terminal now uses real Alpaca data!

    # ... your code ...

    await terminal.stop()
    await alpaca.disconnect()

asyncio.run(run_terminal())
```

## API Reference

### AlpacaDataProvider

#### Constructor
```python
AlpacaDataProvider(
    api_key: Optional[str] = None,        # Defaults to ALPACA_API_KEY env var
    secret_key: Optional[str] = None,     # Defaults to ALPACA_SECRET_KEY env var
    use_websocket: bool = True,           # Enable WebSocket streaming
    cache_ttl_seconds: int = 5            # Cache time-to-live
)
```

#### Methods

**Connection**
- `connect() -> bool` - Connect to Alpaca APIs
- `disconnect()` - Disconnect from Alpaca

**Market Data**
- `get_latest_price(symbol: str) -> Optional[float]` - Get latest price
- `get_latest_quote(symbol: str) -> Optional[RealTimeQuote]` - Get bid/ask quote
- `get_latest_trade(symbol: str) -> Optional[RealTimeTrade]` - Get last trade
- `get_historical_bars(symbol, timeframe, start, end, limit) -> List[BarData]` - Get bars

**Streaming**
- `start_streaming(symbols: List[str])` - Start WebSocket streaming
- `stop_streaming()` - Stop WebSocket streaming
- `add_quote_callback(callback: Callable)` - Register quote callback
- `add_trade_callback(callback: Callable)` - Register trade callback

**Monitoring**
- `get_data_quality_stats() -> Dict[str, Any]` - Get quality statistics

### TradingTerminalDataProvider

#### Updated Constructor
```python
TradingTerminalDataProvider(
    symbols: List[str] = None,
    update_interval: float = 1.0,
    enable_live_data: bool = False,
    alpaca_data_provider: Optional[AlpacaDataProvider] = None  # NEW!
)
```

If `alpaca_data_provider` is provided and `enable_live_data=True`:
- Uses real Alpaca data
- Logs: "Using: Real Alpaca API"

If not provided:
- Falls back to mock data (backward compatible)
- Logs: "Using: Mock/Cached Data"

## Data Classes

### RealTimeQuote
```python
@dataclass
class RealTimeQuote:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def midpoint(self) -> float  # (bid + ask) / 2

    @property
    def spread(self) -> float    # ask - bid
```

### RealTimeTrade
```python
@dataclass
class RealTimeTrade:
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: Optional[str]
```

### BarData
```python
@dataclass
class BarData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
```

## Common Patterns

### Pattern 1: Get Price with Fallback
```python
price = await provider.get_latest_price('SPY')
if price is None:
    # Handle missing data
    price = 440.0  # Default fallback
```

### Pattern 2: Real-Time Callbacks
```python
async def on_quote(quote: RealTimeQuote):
    if quote.spread / quote.midpoint < 0.001:
        print(f"Tight spread on {quote.symbol}: ${quote.spread:.4f}")

provider.add_quote_callback(on_quote)
await provider.start_streaming(['SPY'])
```

### Pattern 3: Technical Analysis
```python
bars = await provider.get_historical_bars('SPY', timeframe='5Min', limit=100)
closes = [bar.close for bar in bars]

# Calculate SMA
sma_20 = sum(closes[-20:]) / 20
sma_50 = sum(closes[-50:]) / 50

# Golden cross
if sma_20 > sma_50:
    print("Bullish signal!")
```

### Pattern 4: Multiple Symbols
```python
symbols = ['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU']
for symbol in symbols:
    price = await provider.get_latest_price(symbol)
    print(f"{symbol}: ${price}")
```

## Error Handling

### Handle Connection Failures
```python
provider = AlpacaDataProvider()
if not await provider.connect():
    print("WARNING: Using fallback data")
    # Continue execution - will use cached/fallback prices
```

### Handle Missing Data
```python
quote = await provider.get_latest_quote('XYZ')
if quote is None:
    logger.warning("No quote available for XYZ")
    # Use fallback or skip
```

### Handle API Rate Limits
```python
# Built-in caching reduces API calls by ~90%
# Cache TTL: 5 seconds by default
# No manual rate limiting needed
```

## Data Quality Features

### Staleness Detection
- Threshold: 60 seconds
- Automatically falls back to cached data

### Outlier Detection
- Rejects prices with >10% deviation from recent average
- Maintains 100-sample price history per symbol

### Quote Validation
- Ensures bid < ask
- Validates spread < 10% of midpoint
- Checks for positive prices

## Troubleshooting

### Issue: "Alpaca library not available"
**Solution**: `pip install alpaca-py`

### Issue: No real data received
**Check**:
1. Are credentials set? `echo $ALPACA_API_KEY`
2. Is `enable_live_data=True`?
3. Is `alpaca_data_provider` passed to terminal?
4. Check logs for connection errors

### Issue: Stale data warnings
**Causes**:
- Market closed (normal)
- Low liquidity symbol
- Network issues

**Solutions**:
- Check market hours: `provider.get_market_status()`
- Increase `cache_ttl_seconds`
- Use more liquid symbols

### Issue: "Using fallback data"
**Reasons**:
- No credentials provided (expected in dev)
- API unavailable (temporary)
- Network error (transient)

**Action**: Check logs for specific error, set credentials if needed

## Performance Tuning

### Reduce API Calls
```python
# Increase cache TTL (default: 5 seconds)
provider = AlpacaDataProvider(cache_ttl_seconds=10)
```

### Optimize Streaming
```python
# Subscribe to specific symbols only
await provider.start_streaming(['SPY', 'QQQ'])  # Not all 500+ symbols
```

### Batch Requests
```python
# Instead of:
for symbol in symbols:
    price = await provider.get_latest_price(symbol)

# Use parallel requests:
prices = await asyncio.gather(
    *[provider.get_latest_price(s) for s in symbols]
)
```

## Testing Without API Keys

```python
# Works without credentials - uses fallback data
provider = AlpacaDataProvider(api_key=None, secret_key=None)
await provider.connect()  # Returns False, logs warning

price = await provider.get_latest_price('SPY')  # Returns fallback: $440.00
# System continues to work for development/testing
```

## Production Checklist

- [ ] Set `ALPACA_API_KEY` environment variable
- [ ] Set `ALPACA_SECRET_KEY` environment variable
- [ ] Verify `ALPACA_PAPER=true` for paper trading
- [ ] Test connection: `provider.connect()` returns `True`
- [ ] Check data quality: `provider.get_data_quality_stats()`
- [ ] Monitor logs for errors
- [ ] Verify real data: Check prices match market
- [ ] Set up alerting for connection failures

## Support

- **Implementation Details**: See `ISS-020-IMPLEMENTATION-SUMMARY.md`
- **Code Examples**: See `examples/real_data_integration_example.py`
- **API Documentation**: See `src/market/alpaca_data_provider.py` docstrings
- **Alpaca Docs**: https://docs.alpaca.markets/
