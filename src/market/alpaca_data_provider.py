"""
ISS-020: Real-Time Alpaca Data Provider

Replaces mock/random data generation with real Alpaca API integration.
Provides real-time market data with WebSocket streaming, caching, and fallback mechanisms.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import deque
import os

logger = logging.getLogger(__name__)

# Import Alpaca libraries with graceful fallback
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import (
        StockLatestQuoteRequest,
        StockLatestTradeRequest,
        StockBarsRequest,
        StockQuotesRequest
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not available - will use cached/fallback data only")


@dataclass
class RealTimeQuote:
    """Real-time quote data from Alpaca"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class RealTimeTrade:
    """Real-time trade data from Alpaca"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: Optional[str] = None


@dataclass
class BarData:
    """Bar/candlestick data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class AlpacaDataProvider:
    """
    Production-ready Alpaca data provider for real-time market data.

    Features:
    - Real-time quotes and trades via WebSocket
    - Historical bars for technical analysis
    - Intelligent caching to reduce API calls
    - Graceful fallback when API unavailable
    - Data quality validation (staleness checks, outlier detection)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        use_websocket: bool = True,
        cache_ttl_seconds: int = 5
    ):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)
            use_websocket: Enable WebSocket streaming for real-time data
            cache_ttl_seconds: Time-to-live for cached data
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.use_websocket = use_websocket and ALPACA_AVAILABLE
        self.cache_ttl = cache_ttl_seconds

        # Clients
        self.historical_client: Optional[StockHistoricalDataClient] = None
        self.stream_client: Optional[StockDataStream] = None

        # Caches
        self.quote_cache: Dict[str, tuple[RealTimeQuote, float]] = {}
        self.trade_cache: Dict[str, tuple[RealTimeTrade, float]] = {}
        self.bar_cache: Dict[str, tuple[List[BarData], float]] = {}

        # WebSocket data storage
        self.latest_quotes: Dict[str, RealTimeQuote] = {}
        self.latest_trades: Dict[str, RealTimeTrade] = {}
        self.price_history: Dict[str, deque] = {}  # symbol -> deque of recent prices

        # Callbacks for real-time updates
        self.quote_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []

        # Streaming state
        self.is_streaming = False
        self.subscribed_symbols: set = set()

        # Fallback prices (updated from real data when available)
        self.fallback_prices = {
            'SPY': 440.00,
            'ULTY': 5.57,
            'AMDY': 7.72,
            'VTIP': 48.50,
            'IAU': 42.00,
            'VTI': 250.00,
            'QQQ': 390.00,
            'GLD': 190.00,
            'TIP': 105.00
        }

        # Data quality tracking
        self.stale_data_threshold = 60  # seconds
        self.outlier_threshold = 0.10  # 10% price change threshold

        logger.info(f"AlpacaDataProvider initialized (API available: {ALPACA_AVAILABLE})")

    async def connect(self) -> bool:
        """
        Connect to Alpaca data APIs.

        Returns:
            True if connected successfully, False otherwise
        """
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca library not available - using cached/fallback data only")
            return False

        if not (self.api_key and self.secret_key):
            logger.warning("Alpaca credentials not provided - using cached/fallback data only")
            return False

        try:
            # Initialize historical data client
            self.historical_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            # Initialize WebSocket stream if enabled
            if self.use_websocket:
                self.stream_client = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )

                # Register default handlers
                async def quote_handler(data):
                    await self._handle_quote_update(data)

                async def trade_handler(data):
                    await self._handle_trade_update(data)

                self.stream_client.subscribe_quotes(quote_handler, *[])
                self.stream_client.subscribe_trades(trade_handler, *[])

            logger.info("Connected to Alpaca data APIs")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Alpaca data APIs."""
        if self.is_streaming and self.stream_client:
            await self.stop_streaming()

        self.historical_client = None
        self.stream_client = None
        logger.info("Disconnected from Alpaca data APIs")

    async def get_latest_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            RealTimeQuote or None if unavailable
        """
        # Check WebSocket data first
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            if self._is_data_fresh(quote.timestamp):
                return quote

        # Check cache
        if symbol in self.quote_cache:
            cached_quote, cached_time = self.quote_cache[symbol]
            if time.time() - cached_time < self.cache_ttl:
                return cached_quote

        # Fetch from API
        if self.historical_client:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                quotes = await asyncio.to_thread(
                    self.historical_client.get_stock_latest_quote, request
                )

                if quotes and symbol in quotes:
                    alpaca_quote = quotes[symbol]
                    quote = RealTimeQuote(
                        symbol=symbol,
                        timestamp=alpaca_quote.timestamp,
                        bid=float(alpaca_quote.bid_price),
                        ask=float(alpaca_quote.ask_price),
                        bid_size=int(alpaca_quote.bid_size),
                        ask_size=int(alpaca_quote.ask_size)
                    )

                    # Validate data quality
                    if self._validate_quote_quality(quote):
                        self.quote_cache[symbol] = (quote, time.time())
                        self.latest_quotes[symbol] = quote
                        return quote
                    else:
                        logger.warning(f"Quote quality validation failed for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")

        # Fallback: construct from cached trade or fallback price
        return self._get_fallback_quote(symbol)

    async def get_latest_trade(self, symbol: str) -> Optional[RealTimeTrade]:
        """
        Get latest trade for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            RealTimeTrade or None if unavailable
        """
        # Check WebSocket data first
        if symbol in self.latest_trades:
            trade = self.latest_trades[symbol]
            if self._is_data_fresh(trade.timestamp):
                return trade

        # Check cache
        if symbol in self.trade_cache:
            cached_trade, cached_time = self.trade_cache[symbol]
            if time.time() - cached_time < self.cache_ttl:
                return cached_trade

        # Fetch from API
        if self.historical_client:
            try:
                request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
                trades = await asyncio.to_thread(
                    self.historical_client.get_stock_latest_trade, request
                )

                if trades and symbol in trades:
                    alpaca_trade = trades[symbol]
                    trade = RealTimeTrade(
                        symbol=symbol,
                        timestamp=alpaca_trade.timestamp,
                        price=float(alpaca_trade.price),
                        size=int(alpaca_trade.size),
                        exchange=getattr(alpaca_trade, 'exchange', None)
                    )

                    # Validate data quality
                    if self._validate_trade_quality(trade):
                        self.trade_cache[symbol] = (trade, time.time())
                        self.latest_trades[symbol] = trade
                        self._update_price_history(symbol, trade.price)
                        return trade
                    else:
                        logger.warning(f"Trade quality validation failed for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching trade for {symbol}: {e}")

        # Fallback
        return self._get_fallback_trade(symbol)

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price (from trade or quote midpoint).

        Args:
            symbol: Stock symbol

        Returns:
            Latest price or None if unavailable
        """
        # Try latest trade first
        trade = await self.get_latest_trade(symbol)
        if trade:
            return trade.price

        # Fallback to quote midpoint
        quote = await self.get_latest_quote(symbol)
        if quote:
            return quote.midpoint

        # Final fallback
        return self.fallback_prices.get(symbol)

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = '1Min',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[BarData]:
        """
        Get historical bars for technical analysis.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            start: Start datetime (defaults to 1 day ago)
            end: End datetime (defaults to now)
            limit: Maximum number of bars to return

        Returns:
            List of BarData
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self.bar_cache:
            cached_bars, cached_time = self.bar_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_bars

        if not self.historical_client:
            logger.warning(f"Historical client not available for {symbol}")
            return []

        try:
            # Convert timeframe string to Alpaca TimeFrame
            tf_map = {
                '1Min': TimeFrame(1, TimeFrameUnit.Minute),
                '5Min': TimeFrame(5, TimeFrameUnit.Minute),
                '15Min': TimeFrame(15, TimeFrameUnit.Minute),
                '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
                '1Day': TimeFrame(1, TimeFrameUnit.Day)
            }
            alpaca_timeframe = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

            # Default time range: last day
            if not start:
                start = datetime.now() - timedelta(days=1)
            if not end:
                end = datetime.now()

            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                limit=limit
            )

            bars_response = await asyncio.to_thread(
                self.historical_client.get_stock_bars, request
            )

            if bars_response and symbol in bars_response:
                bars = []
                for alpaca_bar in bars_response[symbol]:
                    bar = BarData(
                        symbol=symbol,
                        timestamp=alpaca_bar.timestamp,
                        open=float(alpaca_bar.open),
                        high=float(alpaca_bar.high),
                        low=float(alpaca_bar.low),
                        close=float(alpaca_bar.close),
                        volume=int(alpaca_bar.volume)
                    )
                    bars.append(bar)

                # Cache results
                self.bar_cache[cache_key] = (bars, time.time())
                return bars

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")

        return []

    async def start_streaming(self, symbols: List[str]):
        """
        Start WebSocket streaming for real-time data.

        Args:
            symbols: List of symbols to stream
        """
        if not self.stream_client:
            logger.warning("Stream client not initialized - cannot start streaming")
            return

        if self.is_streaming:
            logger.info("Already streaming - updating subscriptions")
            # Update subscriptions
            new_symbols = set(symbols) - self.subscribed_symbols
            if new_symbols:
                self.stream_client.subscribe_quotes(lambda x: None, *new_symbols)
                self.stream_client.subscribe_trades(lambda x: None, *new_symbols)
                self.subscribed_symbols.update(new_symbols)
            return

        try:
            # Subscribe to quotes and trades
            self.stream_client.subscribe_quotes(lambda x: None, *symbols)
            self.stream_client.subscribe_trades(lambda x: None, *symbols)
            self.subscribed_symbols = set(symbols)

            # Start streaming in background
            asyncio.create_task(self.stream_client.run())
            self.is_streaming = True

            logger.info(f"Started streaming for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Error starting streaming: {e}")

    async def stop_streaming(self):
        """Stop WebSocket streaming."""
        if not self.is_streaming:
            return

        try:
            if self.stream_client:
                await self.stream_client.stop_ws()

            self.is_streaming = False
            self.subscribed_symbols.clear()
            logger.info("Stopped streaming")

        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")

    def add_quote_callback(self, callback: Callable):
        """Add callback for real-time quote updates."""
        self.quote_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable):
        """Add callback for real-time trade updates."""
        self.trade_callbacks.append(callback)

    # Private helper methods

    async def _handle_quote_update(self, data):
        """Handle incoming quote from WebSocket."""
        try:
            quote = RealTimeQuote(
                symbol=data.symbol,
                timestamp=data.timestamp,
                bid=float(data.bid_price),
                ask=float(data.ask_price),
                bid_size=int(data.bid_size),
                ask_size=int(data.ask_size)
            )

            if self._validate_quote_quality(quote):
                self.latest_quotes[data.symbol] = quote

                # Notify callbacks
                for callback in self.quote_callbacks:
                    try:
                        await callback(quote)
                    except Exception as e:
                        logger.error(f"Error in quote callback: {e}")

        except Exception as e:
            logger.error(f"Error handling quote update: {e}")

    async def _handle_trade_update(self, data):
        """Handle incoming trade from WebSocket."""
        try:
            trade = RealTimeTrade(
                symbol=data.symbol,
                timestamp=data.timestamp,
                price=float(data.price),
                size=int(data.size),
                exchange=getattr(data, 'exchange', None)
            )

            if self._validate_trade_quality(trade):
                self.latest_trades[data.symbol] = trade
                self._update_price_history(data.symbol, trade.price)

                # Update fallback prices with real data
                self.fallback_prices[data.symbol] = trade.price

                # Notify callbacks
                for callback in self.trade_callbacks:
                    try:
                        await callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    def _is_data_fresh(self, timestamp: datetime) -> bool:
        """Check if data timestamp is recent enough."""
        age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
        return age_seconds < self.stale_data_threshold

    def _validate_quote_quality(self, quote: RealTimeQuote) -> bool:
        """Validate quote data quality."""
        # Check for staleness
        if not self._is_data_fresh(quote.timestamp):
            return False

        # Check for valid prices
        if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
            return False

        # Check spread reasonableness (< 10%)
        if quote.spread / quote.midpoint > 0.10:
            logger.warning(f"Large spread for {quote.symbol}: {quote.spread / quote.midpoint:.2%}")
            return False

        return True

    def _validate_trade_quality(self, trade: RealTimeTrade) -> bool:
        """Validate trade data quality with outlier detection."""
        # Check for staleness
        if not self._is_data_fresh(trade.timestamp):
            return False

        # Check for valid price
        if trade.price <= 0:
            return False

        # Outlier detection: compare to recent price history
        if trade.symbol in self.price_history:
            recent_prices = list(self.price_history[trade.symbol])
            if recent_prices:
                avg_price = sum(recent_prices) / len(recent_prices)
                price_change = abs(trade.price - avg_price) / avg_price

                if price_change > self.outlier_threshold:
                    logger.warning(
                        f"Potential outlier for {trade.symbol}: "
                        f"${trade.price} vs avg ${avg_price:.2f} "
                        f"({price_change:.2%} change)"
                    )
                    return False

        return True

    def _update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)

        self.price_history[symbol].append(price)

    def _get_fallback_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """Get fallback quote when real data unavailable."""
        # Try to construct from latest trade
        if symbol in self.latest_trades:
            trade = self.latest_trades[symbol]
            return RealTimeQuote(
                symbol=symbol,
                timestamp=trade.timestamp,
                bid=trade.price - 0.01,
                ask=trade.price + 0.01,
                bid_size=100,
                ask_size=100
            )

        # Use fallback price
        if symbol in self.fallback_prices:
            price = self.fallback_prices[symbol]
            return RealTimeQuote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=price - 0.01,
                ask=price + 0.01,
                bid_size=100,
                ask_size=100
            )

        return None

    def _get_fallback_trade(self, symbol: str) -> Optional[RealTimeTrade]:
        """Get fallback trade when real data unavailable."""
        if symbol in self.fallback_prices:
            return RealTimeTrade(
                symbol=symbol,
                timestamp=datetime.now(),
                price=self.fallback_prices[symbol],
                size=100,
                exchange='FALLBACK'
            )

        return None

    def get_data_quality_stats(self) -> Dict[str, Any]:
        """Get data quality statistics."""
        return {
            'streaming_active': self.is_streaming,
            'subscribed_symbols': len(self.subscribed_symbols),
            'cached_quotes': len(self.quote_cache),
            'cached_trades': len(self.trade_cache),
            'cached_bars': len(self.bar_cache),
            'latest_quotes': len(self.latest_quotes),
            'latest_trades': len(self.latest_trades),
            'alpaca_available': ALPACA_AVAILABLE,
            'api_configured': bool(self.api_key and self.secret_key)
        }
