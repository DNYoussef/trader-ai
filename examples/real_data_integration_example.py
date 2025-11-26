"""
ISS-020: Example - Real Alpaca Data Integration

Demonstrates how to use the new AlpacaDataProvider with the trading terminal.
Shows both REST API and WebSocket streaming usage patterns.
"""

import asyncio
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_basic_usage():
    """Example 1: Basic usage with REST API"""
    from src.market.alpaca_data_provider import AlpacaDataProvider

    print("\n=== Example 1: Basic REST API Usage ===\n")

    # Initialize provider (uses environment variables)
    provider = AlpacaDataProvider(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        use_websocket=False,  # Disable WebSocket for this example
        cache_ttl_seconds=5
    )

    # Connect to Alpaca
    connected = await provider.connect()
    if not connected:
        print("WARNING: Not connected to Alpaca - will use fallback data")
    else:
        print("SUCCESS: Connected to Alpaca API")

    # Get latest price
    symbols = ['SPY', 'ULTY', 'AMDY']
    for symbol in symbols:
        price = await provider.get_latest_price(symbol)
        print(f"{symbol}: ${price:.2f}")

    # Get detailed quote
    quote = await provider.get_latest_quote('SPY')
    if quote:
        print(f"\nSPY Quote Details:")
        print(f"  Bid: ${quote.bid:.2f} x {quote.bid_size}")
        print(f"  Ask: ${quote.ask:.2f} x {quote.ask_size}")
        print(f"  Midpoint: ${quote.midpoint:.2f}")
        print(f"  Spread: ${quote.spread:.4f}")
        print(f"  Timestamp: {quote.timestamp}")

    # Get latest trade
    trade = await provider.get_latest_trade('SPY')
    if trade:
        print(f"\nSPY Last Trade:")
        print(f"  Price: ${trade.price:.2f}")
        print(f"  Size: {trade.size}")
        print(f"  Exchange: {trade.exchange}")
        print(f"  Timestamp: {trade.timestamp}")

    # Cleanup
    await provider.disconnect()


async def example_2_historical_bars():
    """Example 2: Fetching historical bars for technical analysis"""
    from src.market.alpaca_data_provider import AlpacaDataProvider

    print("\n=== Example 2: Historical Bars ===\n")

    provider = AlpacaDataProvider()
    await provider.connect()

    # Get 1-minute bars for the last 100 minutes
    bars = await provider.get_historical_bars(
        symbol='SPY',
        timeframe='1Min',
        limit=100
    )

    if bars:
        print(f"Retrieved {len(bars)} bars for SPY")
        print("\nLast 5 bars:")
        for bar in bars[-5:]:
            print(f"  {bar.timestamp.strftime('%H:%M')} - "
                  f"O: ${bar.open:.2f}, "
                  f"H: ${bar.high:.2f}, "
                  f"L: ${bar.low:.2f}, "
                  f"C: ${bar.close:.2f}, "
                  f"V: {bar.volume:,}")

        # Calculate simple moving average
        if len(bars) >= 20:
            sma_20 = sum(b.close for b in bars[-20:]) / 20
            current_price = bars[-1].close
            print(f"\nSMA(20): ${sma_20:.2f}")
            print(f"Current: ${current_price:.2f}")
            print(f"Position: {'Above' if current_price > sma_20 else 'Below'} SMA")

    await provider.disconnect()


async def example_3_websocket_streaming():
    """Example 3: Real-time WebSocket streaming"""
    from src.market.alpaca_data_provider import AlpacaDataProvider

    print("\n=== Example 3: WebSocket Streaming ===\n")

    provider = AlpacaDataProvider(use_websocket=True)
    await provider.connect()

    # Track updates
    trade_count = 0
    quote_count = 0

    async def trade_handler(trade):
        nonlocal trade_count
        trade_count += 1
        print(f"[TRADE] {trade.symbol}: ${trade.price:.2f} x {trade.size} @ {trade.timestamp.strftime('%H:%M:%S')}")

    async def quote_handler(quote):
        nonlocal quote_count
        quote_count += 1
        if quote_count % 10 == 0:  # Print every 10th quote to avoid spam
            print(f"[QUOTE] {quote.symbol}: ${quote.bid:.2f} / ${quote.ask:.2f} @ {quote.timestamp.strftime('%H:%M:%S')}")

    # Register callbacks
    provider.add_trade_callback(trade_handler)
    provider.add_quote_callback(quote_handler)

    # Start streaming
    await provider.start_streaming(['SPY', 'ULTY'])
    print("Streaming started... (will run for 30 seconds)")

    # Stream for 30 seconds
    await asyncio.sleep(30)

    # Stop streaming
    await provider.stop_streaming()
    print(f"\nStreaming stopped. Received {trade_count} trades and {quote_count} quotes")

    await provider.disconnect()


async def example_4_terminal_integration():
    """Example 4: Integration with TradingTerminalDataProvider"""
    from src.market.alpaca_data_provider import AlpacaDataProvider
    from src.trading.terminal_data_provider import TradingTerminalDataProvider

    print("\n=== Example 4: Terminal Integration ===\n")

    # Create Alpaca provider
    alpaca_provider = AlpacaDataProvider(use_websocket=True)
    await alpaca_provider.connect()

    # Create terminal provider with real data
    terminal_provider = TradingTerminalDataProvider(
        symbols=['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU'],
        update_interval=5.0,  # Update every 5 seconds
        enable_live_data=True,
        alpaca_data_provider=alpaca_provider
    )

    # Register callbacks
    def market_data_callback(symbol, data):
        print(f"[MARKET] {symbol}: ${data.price:.2f} "
              f"({data.change_percent:+.2f}%) "
              f"Bid: ${data.bid:.2f} Ask: ${data.ask:.2f}")

    terminal_provider.add_market_data_callback(market_data_callback)

    # Start terminal
    await terminal_provider.start()
    print("Terminal started with real Alpaca data...")

    # Get terminal snapshot
    snapshot = terminal_provider.get_terminal_snapshot()
    print(f"\nTerminal Snapshot at {datetime.fromtimestamp(snapshot['timestamp']).strftime('%H:%M:%S')}:")
    for symbol, data in snapshot['market_data'].items():
        print(f"  {symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")

    # Run for 30 seconds
    await asyncio.sleep(30)

    # Stop terminal
    await terminal_provider.stop()
    await alpaca_provider.disconnect()
    print("\nTerminal stopped")


async def example_5_data_quality_monitoring():
    """Example 5: Monitoring data quality statistics"""
    from src.market.alpaca_data_provider import AlpacaDataProvider

    print("\n=== Example 5: Data Quality Monitoring ===\n")

    provider = AlpacaDataProvider()
    connected = await provider.connect()

    if connected:
        # Get some data to populate caches
        await provider.get_latest_price('SPY')
        await provider.get_latest_quote('ULTY')
        await provider.get_historical_bars('AMDY', limit=50)

    # Get data quality stats
    stats = provider.get_data_quality_stats()
    print("Data Quality Statistics:")
    print(f"  Streaming Active: {stats['streaming_active']}")
    print(f"  Subscribed Symbols: {stats['subscribed_symbols']}")
    print(f"  Cached Quotes: {stats['cached_quotes']}")
    print(f"  Cached Trades: {stats['cached_trades']}")
    print(f"  Cached Bars: {stats['cached_bars']}")
    print(f"  Latest Quotes: {stats['latest_quotes']}")
    print(f"  Latest Trades: {stats['latest_trades']}")
    print(f"  Alpaca Available: {stats['alpaca_available']}")
    print(f"  API Configured: {stats['api_configured']}")

    await provider.disconnect()


async def example_6_fallback_behavior():
    """Example 6: Demonstrating fallback behavior without API credentials"""
    from src.market.alpaca_data_provider import AlpacaDataProvider

    print("\n=== Example 6: Fallback Behavior (No Credentials) ===\n")

    # Create provider without credentials
    provider = AlpacaDataProvider(
        api_key=None,
        secret_key=None
    )

    # Try to connect (will fail gracefully)
    connected = await provider.connect()
    print(f"Connected: {connected}")

    # Still works with fallback data
    price = await provider.get_latest_price('SPY')
    print(f"SPY Price (fallback): ${price:.2f}")

    quote = await provider.get_latest_quote('ULTY')
    if quote:
        print(f"ULTY Quote (fallback): ${quote.bid:.2f} / ${quote.ask:.2f}")

    print("\nNote: Data is from fallback prices, not real Alpaca API")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ISS-020: Real Alpaca Data Integration Examples")
    print("="*60)

    # Check for API credentials
    has_credentials = bool(os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY'))
    if not has_credentials:
        print("\nWARNING: No Alpaca API credentials found")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("Examples will use fallback data\n")

    try:
        # Run examples
        await example_1_basic_usage()
        await example_2_historical_bars()

        # Skip WebSocket examples if no credentials
        if has_credentials:
            await example_3_websocket_streaming()
            await example_4_terminal_integration()
        else:
            print("\n(Skipping WebSocket examples - no credentials)")

        await example_5_data_quality_monitoring()
        await example_6_fallback_behavior()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

    print("\n" + "="*60)
    print("Examples completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
