"""
Production Alpaca broker adapter for the Gary×Taleb trading system.

This is the REAL production implementation with no mocks or simulation.
Requires valid Alpaca API credentials and handles real money transactions.
"""

import os
import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import json
import time

# Alpaca SDK imports (required for production)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        GetOrdersRequest, ClosePositionRequest
    )
    from alpaca.trading.enums import (
        OrderSide, TimeInForce as AlpacaTIF, OrderType as AlpacaOrderType,
        OrderStatus as AlpacaOrderStatus, AssetClass, QueryOrderStatus
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import (
        StockLatestQuoteRequest, StockBarsRequest,
        StockTradesRequest, StockSnapshotRequest
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.common.exceptions import APIError
    ALPACA_INSTALLED = True
except ImportError as e:
    raise ImportError(
        "PRODUCTION ERROR: alpaca-py is required for production trading.\n"
        "Install with: pip install alpaca-py\n"
        f"Error: {e}"
    )

from ..brokers.broker_interface import (
    BrokerInterface, Order, Position, Fill, OrderStatus, OrderType, TimeInForce,
    BrokerError, ConnectionError, AuthenticationError, InsufficientFundsError,
    InvalidOrderError, MarketClosedError, RateLimitError
)

logger = logging.getLogger(__name__)


class AlpacaProductionAdapter(BrokerInterface):
    """
    Production Alpaca broker adapter.

    This is the REAL implementation - no mocks, no simulation.
    Handles real money and requires valid API credentials.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize production Alpaca adapter.

        Args:
            config: Configuration containing:
                - api_key: Alpaca API key (REQUIRED)
                - secret_key: Alpaca secret key (REQUIRED)
                - paper_trading: Use paper account (default: True for safety)
                - base_url: Override base URL (optional)

        Raises:
            ValueError: If API credentials are missing
        """
        super().__init__(config)

        # Extract credentials (REQUIRED for production)
        self.api_key = config.get('api_key') or os.getenv('ALPACA_API_KEY')
        self.secret_key = config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "PRODUCTION ERROR: Alpaca API credentials are required.\n"
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables\n"
                "or provide them in the configuration."
            )

        # Safety: Default to paper trading unless explicitly set to live
        self.paper_trading = config.get('paper_trading', True)
        self.base_url = config.get('base_url')

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.stream_client = None

        # Connection state
        self.is_connected = False
        self.last_heartbeat = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Position and order caches
        self._position_cache = {}
        self._order_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 5  # 5 second cache TTL

        # Alpaca supports up to 6 decimal places
        self.qty_precision = 6

        logger.info(
            f"Initialized Alpaca PRODUCTION adapter - "
            f"Mode: {'PAPER' if self.paper_trading else 'LIVE'}"
        )

    async def connect(self) -> bool:
        """
        Connect to Alpaca API.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If credentials are invalid
        """
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_trading,
                url_override=self.base_url
            )

            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )

            # Test connection by getting account info
            account = self.trading_client.get_account()

            if account:
                self.is_connected = True
                self.last_heartbeat = time.time()

                logger.info(
                    f"Connected to Alpaca {'PAPER' if self.paper_trading else 'LIVE'} - "
                    f"Account: {account.account_number}, "
                    f"Buying Power: ${account.buying_power}, "
                    f"Portfolio Value: ${account.portfolio_value}"
                )

                # Log important account status
                if account.trading_blocked:
                    logger.warning("TRADING IS BLOCKED on this account")
                if account.pattern_day_trader:
                    logger.info("Account is flagged as Pattern Day Trader")

                return True

            return False

        except APIError as e:
            if 'authentication' in str(e).lower() or '401' in str(e):
                raise AuthenticationError(f"Invalid Alpaca credentials: {e}")
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to Alpaca: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""
        try:
            # Close any open websocket connections
            if self.stream_client:
                await self.stream_client.close()

            self.is_connected = False
            self.trading_client = None
            self.data_client = None
            self.stream_client = None

            logger.info("Disconnected from Alpaca")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {e}")
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Account information including balance, buying power, etc.
        """
        self._check_connection()
        self._rate_limit()

        try:
            account = self.trading_client.get_account()

            return {
                'account_id': account.account_number,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'daytrade_count': account.daytrade_count,
                'daytrading_buying_power': float(account.daytrading_buying_power or 0),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'maintenance_margin': float(account.maintenance_margin),
                'initial_margin': float(account.initial_margin),
                'sma': float(account.sma or 0)
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise BrokerError(f"Failed to get account info: {e}")

    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        self._check_connection()

        # Check cache
        if self._is_cache_valid():
            return list(self._position_cache.values())

        self._rate_limit()

        try:
            alpaca_positions = self.trading_client.get_all_positions()

            positions = []
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=Decimal(str(pos.qty)),
                    avg_price=Decimal(str(pos.avg_entry_price)),
                    current_price=Decimal(str(pos.current_price or 0)),
                    market_value=Decimal(str(pos.market_value or 0)),
                    unrealized_pnl=Decimal(str(pos.unrealized_pl or 0)),
                    realized_pnl=Decimal(str(pos.unrealized_pl or 0)),  # Alpaca doesn't separate
                    side='long' if float(pos.qty) > 0 else 'short'
                )
                positions.append(position)
                self._position_cache[pos.symbol] = position

            self._cache_timestamp = time.time()
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    async def submit_order(self, order: Order) -> str:
        """
        Submit an order to Alpaca.

        Args:
            order: Order object with details

        Returns:
            Order ID if successful

        Raises:
            InvalidOrderError: If order is invalid
            InsufficientFundsError: If insufficient buying power
            MarketClosedError: If market is closed
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Validate order
            if order.quantity <= 0:
                raise InvalidOrderError("Order quantity must be positive")

            # Round quantity to supported precision
            qty = float(Decimal(str(order.quantity)).quantize(
                Decimal(f"0.{'0' * self.qty_precision}"),
                rounding=ROUND_HALF_UP
            ))

            # Map order side
            side = OrderSide.BUY if order.side == 'buy' else OrderSide.SELL

            # Map time in force
            tif_map = {
                TimeInForce.DAY: AlpacaTIF.DAY,
                TimeInForce.GTC: AlpacaTIF.GTC,
                TimeInForce.IOC: AlpacaTIF.IOC,
                TimeInForce.FOK: AlpacaTIF.FOK
            }
            tif = tif_map.get(order.time_in_force, AlpacaTIF.DAY)

            # Create order request based on type
            if order.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif
                )
            elif order.order_type == OrderType.LIMIT:
                if not order.limit_price:
                    raise InvalidOrderError("Limit price required for limit orders")
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.limit_price)
                )
            elif order.order_type == OrderType.STOP:
                if not order.stop_price:
                    raise InvalidOrderError("Stop price required for stop orders")
                order_request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    stop_price=float(order.stop_price)
                )
            else:
                raise InvalidOrderError(f"Unsupported order type: {order.order_type}")

            # Submit order
            alpaca_order = self.trading_client.submit_order(order_request)

            # Cache order
            self._order_cache[alpaca_order.id] = alpaca_order

            logger.info(
                f"Submitted {order.order_type} {order.side} order: "
                f"{order.quantity} {order.symbol} @ "
                f"${order.limit_price or 'market'}"
            )

            return alpaca_order.id

        except APIError as e:
            error_msg = str(e).lower()
            if 'insufficient' in error_msg or 'buying power' in error_msg:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif 'market is closed' in error_msg:
                raise MarketClosedError(f"Market is closed: {e}")
            elif 'invalid' in error_msg:
                raise InvalidOrderError(f"Invalid order: {e}")
            else:
                raise BrokerError(f"Order submission failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error submitting order: {e}")
            raise BrokerError(f"Order submission failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation successful
        """
        self._check_connection()
        self._rate_limit()

        try:
            self.trading_client.cancel_order_by_id(order_id)

            # Remove from cache
            if order_id in self._order_cache:
                del self._order_cache[order_id]

            logger.info(f"Cancelled order: {order_id}")
            return True

        except APIError as e:
            if '404' in str(e) or 'not found' in str(e).lower():
                logger.warning(f"Order not found: {order_id}")
                return False
            logger.error(f"Failed to cancel order: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cancelling order: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get status of an order.

        Args:
            order_id: ID of order

        Returns:
            OrderStatus enum or None if not found
        """
        self._check_connection()
        self._rate_limit()

        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)

            status_map = {
                AlpacaOrderStatus.NEW: OrderStatus.PENDING,
                AlpacaOrderStatus.ACCEPTED: OrderStatus.PENDING,
                AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
                AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
                AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
                AlpacaOrderStatus.DONE_FOR_DAY: OrderStatus.CANCELLED,
                AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
                AlpacaOrderStatus.EXPIRED: OrderStatus.CANCELLED,
                AlpacaOrderStatus.REPLACED: OrderStatus.CANCELLED,
                AlpacaOrderStatus.PENDING_CANCEL: OrderStatus.PENDING,
                AlpacaOrderStatus.PENDING_REPLACE: OrderStatus.PENDING,
                AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
                AlpacaOrderStatus.SUSPENDED: OrderStatus.PENDING,
                AlpacaOrderStatus.CALCULATED: OrderStatus.PENDING
            }

            return status_map.get(alpaca_order.status, OrderStatus.PENDING)

        except APIError as e:
            if '404' in str(e) or 'not found' in str(e).lower():
                return None
            logger.error(f"Failed to get order status: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting order status: {e}")
            return None

    async def get_fills(self, symbol: Optional[str] = None,
                        start_time: Optional[datetime] = None) -> List[Fill]:
        """
        Get filled orders.

        Args:
            symbol: Filter by symbol (optional)
            start_time: Get fills after this time (optional)

        Returns:
            List of Fill objects
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Build request
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol] if symbol else None,
                after=start_time if start_time else datetime.now(timezone.utc) - timedelta(days=1)
            )

            orders = self.trading_client.get_orders(request)

            fills = []
            for order in orders:
                if order.filled_qty and float(order.filled_qty) > 0:
                    fill = Fill(
                        order_id=order.id,
                        symbol=order.symbol,
                        quantity=Decimal(str(order.filled_qty)),
                        price=Decimal(str(order.filled_avg_price or 0)),
                        side=order.side.value.lower(),
                        timestamp=order.filled_at or order.updated_at,
                        commission=Decimal('0')  # Alpaca doesn't provide commission in order
                    )
                    fills.append(fill)

            return fills

        except Exception as e:
            logger.error(f"Failed to get fills: {e}")
            raise BrokerError(f"Failed to get fills: {e}")

    async def close_position(self, symbol: str) -> bool:
        """
        Close a position completely.

        Args:
            symbol: Symbol to close

        Returns:
            True if position closed successfully
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Close position
            order = self.trading_client.close_position(symbol)

            logger.info(f"Closed position: {symbol}, Order ID: {order.id}")

            # Clear from cache
            if symbol in self._position_cache:
                del self._position_cache[symbol]

            return True

        except APIError as e:
            if '404' in str(e) or 'not found' in str(e).lower():
                logger.warning(f"No position found for {symbol}")
                return True  # Position doesn't exist, consider it closed
            logger.error(f"Failed to close position: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error closing position: {e}")
            return False

    async def close_all_positions(self) -> bool:
        """
        Close ALL open positions (EMERGENCY USE).

        Returns:
            True if all positions closed successfully
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Close all positions
            orders = self.trading_client.close_all_positions(cancel_orders=True)

            logger.warning(f"CLOSED ALL POSITIONS: {len(orders)} orders submitted")

            # Clear position cache
            self._position_cache.clear()

            return True

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    async def get_market_hours(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get market hours for a given date.

        Args:
            date: Date to check (default: today)

        Returns:
            Dict with market hours information
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Get clock
            clock = self.trading_client.get_clock()

            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if clock.next_open else None,
                'next_close': clock.next_close.isoformat() if clock.next_close else None,
                'timestamp': clock.timestamp.isoformat() if clock.timestamp else None
            }

        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {'is_open': False}

    async def get_current_price(self, symbol: str) -> Decimal:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current price as Decimal
        """
        self._check_connection()
        self._rate_limit()

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)

            if symbol in quote:
                return Decimal(str(quote[symbol].ask_price or quote[symbol].bid_price or 0))

            return Decimal('0')

        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise BrokerError(f"Failed to get price: {e}")

    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical bar data.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            limit: Number of bars to retrieve

        Returns:
            List of bar data dictionaries
        """
        self._check_connection()
        self._rate_limit()

        try:
            # Map timeframe
            tf_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, 'Min'),
                '15Min': TimeFrame(15, 'Min'),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            tf = tf_map.get(timeframe, TimeFrame.Day)

            # Get bars
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit
            )
            bars = self.data_client.get_stock_bars(request)

            result = []
            if symbol in bars:
                for bar in bars[symbol]:
                    result.append({
                        'timestamp': bar.timestamp.isoformat(),
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                        'vwap': float(bar.vwap) if hasattr(bar, 'vwap') else 0
                    })

            return result

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise BrokerError(f"Failed to get historical data: {e}")

    async def withdraw_funds(self, amount: Decimal) -> bool:
        """
        Withdraw funds from the account.

        NOTE: Alpaca doesn't support programmatic withdrawals.
        This logs the request for manual processing.

        Args:
            amount: Amount to withdraw

        Returns:
            False (not supported via API)
        """
        logger.warning(
            f"WITHDRAWAL REQUEST: ${amount} - "
            f"Alpaca does not support programmatic withdrawals. "
            f"Please process manually via Alpaca dashboard."
        )

        # Log to audit file for manual processing
        withdrawal_log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'amount': str(amount),
            'status': 'PENDING_MANUAL',
            'account': await self.get_account_info()
        }

        try:
            with open('withdrawals_pending.json', 'a') as f:
                json.dump(withdrawal_log, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log withdrawal request: {e}")

        return False

    def get_last_withdrawal_id(self) -> Optional[str]:
        """
        Get ID of last withdrawal.

        NOTE: Returns None as Alpaca doesn't support programmatic withdrawals.

        Returns:
            None
        """
        return None

    def _check_connection(self):
        """Check if connected to Alpaca."""
        if not self.is_connected or not self.trading_client:
            raise ConnectionError("Not connected to Alpaca")

    def _rate_limit(self):
        """Apply rate limiting to avoid hitting API limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_ttl

    async def health_check(self) -> bool:
        """
        Perform health check on the connection.

        Returns:
            True if healthy
        """
        try:
            # Get clock to test connection
            clock = self.trading_client.get_clock()
            self.last_heartbeat = time.time()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_connection_status(self) -> str:
        """
        Get detailed connection status.

        Returns:
            Status string
        """
        if not self.is_connected:
            return "DISCONNECTED"

        if self.last_heartbeat:
            elapsed = time.time() - self.last_heartbeat
            if elapsed > 60:
                return f"STALE (last heartbeat: {elapsed:.0f}s ago)"

        return f"CONNECTED ({'PAPER' if self.paper_trading else 'LIVE'})"