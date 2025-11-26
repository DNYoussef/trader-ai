"""
Alpaca broker adapter for the Gary×Taleb trading system.

Implements the BrokerInterface for Alpaca Trading API with support for both
paper and live trading modes, including fractional shares up to 6 decimal places.
"""

import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .broker_interface import (
    BrokerInterface, Order, Position, OrderStatus, OrderType, TimeInForce,
    BrokerError, ConnectionError, AuthenticationError, InsufficientFundsError,
    InvalidOrderError, RateLimitError
)

logger = logging.getLogger(__name__)

# Try to import alpaca-py, use mock mode if not available
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest,
        TrailingStopOrderRequest, OrderSide, TimeInForce as AlpacaTimeInForce
    )
    from alpaca.trading.models import Order as AlpacaOrder, Position as AlpacaPosition
    from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus, OrderType as AlpacaOrderType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
    logger.info("PRODUCTION MODE: Alpaca-py library successfully imported")
except ImportError as e:
    ALPACA_AVAILABLE = False
    TradingClient = None
    StockHistoricalDataClient = None
    StockDataStream = None
    APIError = Exception
    logger.warning(f"MOCK MODE: Alpaca-py not available - {e}")


# NO MOCK CLIENT - REAL ALPACA ONLY
# Use real_alpaca_adapter.py for production


class AlpacaAdapter(BrokerInterface):
    """Alpaca broker adapter implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca adapter.

        Args:
            config: Configuration dictionary containing:
                - api_key: Alpaca API key
                - secret_key: Alpaca secret key
                - paper_trading: Whether to use paper trading (default: True)
                - base_url: Optional custom base URL
                - mock_mode: Not supported in production (will raise error)
        """
        super().__init__(config)

        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.base_url = config.get('base_url')
        # Mock mode not supported in production
        if config.get('mock_mode', False):
            raise ValueError("Mock mode not supported in production environment")

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.stream_client = None

        # Alpaca supports up to 6 decimal places for fractional shares
        self.qty_precision = 6

        # ISS-012: Production features merged from alpaca_production.py
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Position and order caches
        self._position_cache: Dict[str, Position] = {}
        self._order_cache: Dict[str, Any] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 5  # 5 second cache TTL
        self.last_heartbeat = None

        # Validate credentials for production mode
        if not (self.api_key and self.secret_key):
            raise ValueError("PRODUCTION ERROR: API credentials required for trading")

        # Determine operating mode - no mock mode in production
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py library required for production trading")
        elif self.is_paper_trading:
            mode = "PAPER"
        else:
            mode = "LIVE"

        logger.info(f"Initialized Alpaca adapter - Mode: {mode}, Library Available: {ALPACA_AVAILABLE}")

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            if not ALPACA_AVAILABLE:
                raise ConnectionError("PRODUCTION ERROR: alpaca-py library required for trading. Install with: pip install alpaca-py")
            
            # Initialize real Alpaca clients
            client_config = {
                'api_key': self.api_key,
                'secret_key': self.secret_key,
                'paper': self.is_paper_trading
            }
            
            if self.base_url:
                client_config['url_override'] = self.base_url
            
            self.trading_client = TradingClient(**client_config)
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection by getting account info
            account = await self._safe_api_call(self.trading_client.get_account)
            if account:
                self.is_connected = True
                logger.info(f"Connected to Alpaca (Account: {account.account_number})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self.stream_client:
            try:
                await self.stream_client.stop_ws()
            except Exception as e:
                logger.warning(f"Error stopping stream client: {e}")
        
        self.trading_client = None
        self.data_client = None
        self.stream_client = None
        self.is_connected = False
        logger.info("Disconnected from Alpaca")

    async def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            if not ALPACA_AVAILABLE:
                raise ConnectionError("Alpaca library required for market status")
            
            clock = await self._safe_api_call(self.trading_client.get_clock)
            return clock.is_open if clock else False
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    async def get_account_value(self) -> Decimal:
        """Get total account value."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            account = await self._safe_api_call(self.trading_client.get_account)
            if account and hasattr(account, 'portfolio_value'):
                return Decimal(str(account.portfolio_value))
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            raise BrokerError(f"Failed to get account value: {e}")

    async def get_cash_balance(self) -> Decimal:
        """Get available cash balance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            account = await self._safe_api_call(self.trading_client.get_account)
            if account and hasattr(account, 'cash'):
                return Decimal(str(account.cash))
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Error getting cash balance: {e}")
            raise BrokerError(f"Failed to get cash balance: {e}")

    async def get_buying_power(self) -> Decimal:
        """Get buying power."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            account = await self._safe_api_call(self.trading_client.get_account)
            if account and hasattr(account, 'buying_power'):
                return Decimal(str(account.buying_power))
            return Decimal("0")
            
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            raise BrokerError(f"Failed to get buying power: {e}")

    async def get_positions(self) -> List[Position]:
        """Get all current positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_positions = await self._safe_api_call(self.trading_client.get_all_positions)
            positions = []
            
            if alpaca_positions:
                for pos in alpaca_positions:
                    position = await self._convert_alpaca_position(pos)
                    if position:
                        positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_position = await self._safe_api_call(
                self.trading_client.get_open_position, symbol
            )
            
            if alpaca_position:
                return await self._convert_alpaca_position(alpaca_position)
            return None
            
        except Exception as e:
            # Position not found is not an error
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Error getting position for {symbol}: {e}")
            raise BrokerError(f"Failed to get position for {symbol}: {e}")

    async def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            # Validate order
            await self._validate_order(order)
            
            # Convert to Alpaca order request
            alpaca_order_request = await self._convert_to_alpaca_order(order)
            
            # Submit order
            alpaca_order = await self._safe_api_call(
                self.trading_client.submit_order, alpaca_order_request
            )
            
            if alpaca_order:
                # Convert back to our Order format
                return await self._convert_from_alpaca_order(alpaca_order)
            else:
                raise BrokerError("Order submission returned None")
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient funds for order: {e}")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(f"Invalid order: {e}")
            else:
                raise BrokerError(f"Failed to submit order: {e}")

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_order = await self._safe_api_call(
                self.trading_client.get_order_by_id, order_id
            )
            
            if alpaca_order:
                return await self._convert_from_alpaca_order(alpaca_order)
            return None
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    async def get_orders(self, 
                        status: Optional[OrderStatus] = None,
                        limit: int = 100,
                        after: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        direction: str = "desc") -> List[Order]:
        """Get orders with optional filtering."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            params = {
                'limit': limit,
                'direction': direction
            }
            
            if status:
                params['status'] = self._convert_to_alpaca_status(status)
            if after:
                params['after'] = after
            if until:
                params['until'] = until
            
            alpaca_orders = await self._safe_api_call(
                self.trading_client.get_orders, **params
            )
            
            orders = []
            if alpaca_orders:
                for alpaca_order in alpaca_orders:
                    order = await self._convert_from_alpaca_order(alpaca_order)
                    if order:
                        orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise BrokerError(f"Failed to get orders: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            result = await self._safe_api_call(
                self.trading_client.cancel_order_by_id, order_id
            )
            return result is not None
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            result = await self._safe_api_call(self.trading_client.cancel_all_orders)
            if isinstance(result, int):
                return result
            elif hasattr(result, '__len__'):
                return len(result)
            return 0
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return 0

    async def get_market_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for a symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            if not ALPACA_AVAILABLE:
                raise ConnectionError("Alpaca library required for price data")
            
            # Try to get latest trade first
            trade_request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = await self._safe_api_call(
                self.data_client.get_stock_latest_trade, trade_request
            )
            
            if trades and symbol in trades:
                return Decimal(str(trades[symbol].price))
            
            # Fallback to quote
            quote = await self.get_quote(symbol)
            if quote and 'midpoint' in quote:
                return Decimal(str(quote['midpoint']))
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return None

    async def get_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get last trade information for a symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            if not ALPACA_AVAILABLE:
                raise ConnectionError("Alpaca library required for trade data")
            
            trade_request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = await self._safe_api_call(
                self.data_client.get_stock_latest_trade, trade_request
            )
            
            if trades and symbol in trades:
                trade = trades[symbol]
                return {
                    'symbol': symbol,
                    'price': float(trade.price),
                    'size': int(trade.size),
                    'timestamp': trade.timestamp,
                    'conditions': trade.conditions if hasattr(trade, 'conditions') else [],
                    'exchange': trade.exchange if hasattr(trade, 'exchange') else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting last trade for {symbol}: {e}")
            return None

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current quote (bid/ask) for a symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        try:
            if not ALPACA_AVAILABLE:
                raise ConnectionError("Alpaca library required for quote data")
            
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = await self._safe_api_call(
                self.data_client.get_stock_latest_quote, quote_request
            )
            
            if quotes and symbol in quotes:
                quote = quotes[symbol]
                bid = float(quote.bid_price)
                ask = float(quote.ask_price)
                
                return {
                    'symbol': symbol,
                    'bid': bid,
                    'ask': ask,
                    'bid_size': int(quote.bid_size),
                    'ask_size': int(quote.ask_size),
                    'midpoint': (bid + ask) / 2,
                    'timestamp': quote.timestamp,
                    'conditions': quote.conditions if hasattr(quote, 'conditions') else [],
                    'exchange': quote.exchange if hasattr(quote, 'exchange') else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    # Helper methods
    
    async def _safe_api_call(self, func, *args, **kwargs):
        """Safely call Alpaca API with error handling."""
        try:
            # Handle both sync and async functions
            result = func(*args, **kwargs)
            if hasattr(result, '__await__'):
                return await result
            return result
            
        except Exception as e:
            if ALPACA_AVAILABLE and isinstance(e, APIError):
                if e.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {e}")
                elif e.status_code == 403:
                    raise AuthenticationError(f"Forbidden: {e}")
                elif e.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {e}")
                else:
                    raise BrokerError(f"API error: {e}")
            raise

    async def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if not order.symbol:
            raise InvalidOrderError("Order must have a symbol")
        
        if not order.side or order.side not in ['buy', 'sell']:
            raise InvalidOrderError("Order must have valid side (buy/sell)")
        
        if order.qty <= 0 and (not order.notional or order.notional <= 0):
            raise InvalidOrderError("Order must have positive quantity or notional value")
        
        # Check precision for fractional shares
        if order.qty > 0:
            # Round to 6 decimal places (Alpaca's limit)
            order.qty = order.qty.quantize(
                Decimal('0.000001'), rounding=ROUND_HALF_UP
            )

    async def _convert_to_alpaca_order(self, order: Order):
        """Convert our Order to Alpaca order request."""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py library required for order conversion")
        
        # Convert side
        side = OrderSide.BUY if order.side.lower() == 'buy' else OrderSide.SELL
        
        # Convert time in force
        tif_map = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
            TimeInForce.OPG: AlpacaTimeInForce.OPG,
            TimeInForce.CLS: AlpacaTimeInForce.CLS,
        }
        time_in_force = tif_map.get(order.time_in_force, AlpacaTimeInForce.DAY)
        
        # Base parameters
        base_params = {
            'symbol': order.symbol,
            'side': side,
            'time_in_force': time_in_force,
            'extended_hours': order.extended_hours
        }
        
        if order.client_order_id:
            base_params['client_order_id'] = order.client_order_id
        
        # Create appropriate order request based on type
        if order.order_type == OrderType.MARKET:
            if order.notional and order.notional > 0:
                base_params['notional'] = float(order.notional)
                return MarketOrderRequest(**base_params)
            else:
                base_params['qty'] = float(order.qty)
                return MarketOrderRequest(**base_params)
                
        elif order.order_type == OrderType.LIMIT:
            base_params['qty'] = float(order.qty)
            base_params['limit_price'] = float(order.limit_price)
            return LimitOrderRequest(**base_params)
            
        elif order.order_type == OrderType.STOP:
            base_params['qty'] = float(order.qty)
            base_params['stop_price'] = float(order.stop_price)
            return StopOrderRequest(**base_params)
            
        elif order.order_type == OrderType.STOP_LIMIT:
            base_params['qty'] = float(order.qty)
            base_params['limit_price'] = float(order.limit_price)
            base_params['stop_price'] = float(order.stop_price)
            return StopLimitOrderRequest(**base_params)
            
        elif order.order_type == OrderType.TRAILING_STOP:
            base_params['qty'] = float(order.qty)
            if order.trail_price:
                base_params['trail_price'] = float(order.trail_price)
            elif order.trail_percent:
                base_params['trail_percent'] = float(order.trail_percent)
            return TrailingStopOrderRequest(**base_params)
        
        else:
            raise InvalidOrderError(f"Unsupported order type: {order.order_type}")

    async def _convert_from_alpaca_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to our Order format."""
        # Handle status conversion
        status = None
        if hasattr(alpaca_order, 'status'):
            status = self._convert_from_alpaca_status(alpaca_order.status)
        
        # Handle order type conversion
        order_type = OrderType.MARKET
        if hasattr(alpaca_order, 'order_type'):
            type_map = {
                'market': OrderType.MARKET,
                'limit': OrderType.LIMIT,
                'stop': OrderType.STOP,
                'stop_limit': OrderType.STOP_LIMIT,
                'trailing_stop': OrderType.TRAILING_STOP,
            }
            order_type = type_map.get(str(alpaca_order.order_type).lower(), OrderType.MARKET)
        
        # Handle time in force conversion
        time_in_force = TimeInForce.DAY
        if hasattr(alpaca_order, 'time_in_force'):
            tif_map = {
                'day': TimeInForce.DAY,
                'gtc': TimeInForce.GTC,
                'ioc': TimeInForce.IOC,
                'fok': TimeInForce.FOK,
                'opg': TimeInForce.OPG,
                'cls': TimeInForce.CLS,
            }
            time_in_force = tif_map.get(str(alpaca_order.time_in_force).lower(), TimeInForce.DAY)
        
        return Order(
            id=str(alpaca_order.id) if hasattr(alpaca_order, 'id') else None,
            client_order_id=str(alpaca_order.client_order_id) if hasattr(alpaca_order, 'client_order_id') and alpaca_order.client_order_id else None,
            symbol=str(alpaca_order.symbol) if hasattr(alpaca_order, 'symbol') else "",
            qty=Decimal(str(alpaca_order.qty)) if hasattr(alpaca_order, 'qty') else Decimal("0"),
            notional=Decimal(str(alpaca_order.notional)) if hasattr(alpaca_order, 'notional') and alpaca_order.notional else None,
            side=str(alpaca_order.side) if hasattr(alpaca_order, 'side') else "",
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=Decimal(str(alpaca_order.limit_price)) if hasattr(alpaca_order, 'limit_price') and alpaca_order.limit_price else None,
            stop_price=Decimal(str(alpaca_order.stop_price)) if hasattr(alpaca_order, 'stop_price') and alpaca_order.stop_price else None,
            trail_price=Decimal(str(alpaca_order.trail_price)) if hasattr(alpaca_order, 'trail_price') and alpaca_order.trail_price else None,
            trail_percent=Decimal(str(alpaca_order.trail_percent)) if hasattr(alpaca_order, 'trail_percent') and alpaca_order.trail_percent else None,
            extended_hours=bool(alpaca_order.extended_hours) if hasattr(alpaca_order, 'extended_hours') else False,
            status=status,
            filled_qty=Decimal(str(alpaca_order.filled_qty)) if hasattr(alpaca_order, 'filled_qty') else Decimal("0"),
            filled_avg_price=Decimal(str(alpaca_order.filled_avg_price)) if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price else None,
            created_at=alpaca_order.created_at if hasattr(alpaca_order, 'created_at') else None,
            updated_at=alpaca_order.updated_at if hasattr(alpaca_order, 'updated_at') else None,
            submitted_at=alpaca_order.submitted_at if hasattr(alpaca_order, 'submitted_at') else None,
            filled_at=alpaca_order.filled_at if hasattr(alpaca_order, 'filled_at') else None,
            expired_at=alpaca_order.expired_at if hasattr(alpaca_order, 'expired_at') else None,
            canceled_at=alpaca_order.canceled_at if hasattr(alpaca_order, 'canceled_at') else None,
            failed_at=alpaca_order.failed_at if hasattr(alpaca_order, 'failed_at') else None,
            replaced_at=alpaca_order.replaced_at if hasattr(alpaca_order, 'replaced_at') else None,
            replaced_by=str(alpaca_order.replaced_by) if hasattr(alpaca_order, 'replaced_by') and alpaca_order.replaced_by else None,
            replaces=str(alpaca_order.replaces) if hasattr(alpaca_order, 'replaces') and alpaca_order.replaces else None,
            asset_id=str(alpaca_order.asset_id) if hasattr(alpaca_order, 'asset_id') else None,
            asset_class=str(alpaca_order.asset_class) if hasattr(alpaca_order, 'asset_class') else "us_equity",
            commission=Decimal(str(alpaca_order.commission)) if hasattr(alpaca_order, 'commission') and alpaca_order.commission else None,
        )

    async def _convert_alpaca_position(self, alpaca_position) -> Optional[Position]:
        """Convert Alpaca position to our Position format."""
        if not alpaca_position:
            return None
        
        try:
            return Position(
                asset_id=str(alpaca_position.asset_id) if hasattr(alpaca_position, 'asset_id') else "",
                symbol=str(alpaca_position.symbol) if hasattr(alpaca_position, 'symbol') else "",
                exchange=str(alpaca_position.exchange) if hasattr(alpaca_position, 'exchange') else "",
                asset_class=str(alpaca_position.asset_class) if hasattr(alpaca_position, 'asset_class') else "us_equity",
                avg_entry_price=Decimal(str(alpaca_position.avg_entry_price)) if hasattr(alpaca_position, 'avg_entry_price') else Decimal("0"),
                qty=Decimal(str(alpaca_position.qty)) if hasattr(alpaca_position, 'qty') else Decimal("0"),
                side=str(alpaca_position.side) if hasattr(alpaca_position, 'side') else "long",
                market_value=Decimal(str(alpaca_position.market_value)) if hasattr(alpaca_position, 'market_value') and alpaca_position.market_value else None,
                cost_basis=Decimal(str(alpaca_position.cost_basis)) if hasattr(alpaca_position, 'cost_basis') and alpaca_position.cost_basis else None,
                unrealized_pl=Decimal(str(alpaca_position.unrealized_pl)) if hasattr(alpaca_position, 'unrealized_pl') and alpaca_position.unrealized_pl else None,
                unrealized_plpc=Decimal(str(alpaca_position.unrealized_plpc)) if hasattr(alpaca_position, 'unrealized_plpc') and alpaca_position.unrealized_plpc else None,
                unrealized_intraday_pl=Decimal(str(alpaca_position.unrealized_intraday_pl)) if hasattr(alpaca_position, 'unrealized_intraday_pl') and alpaca_position.unrealized_intraday_pl else None,
                unrealized_intraday_plpc=Decimal(str(alpaca_position.unrealized_intraday_plpc)) if hasattr(alpaca_position, 'unrealized_intraday_plpc') and alpaca_position.unrealized_intraday_plpc else None,
                current_price=Decimal(str(alpaca_position.current_price)) if hasattr(alpaca_position, 'current_price') and alpaca_position.current_price else None,
                lastday_price=Decimal(str(alpaca_position.lastday_price)) if hasattr(alpaca_position, 'lastday_price') and alpaca_position.lastday_price else None,
                change_today=Decimal(str(alpaca_position.change_today)) if hasattr(alpaca_position, 'change_today') and alpaca_position.change_today else None,
                qty_available=Decimal(str(alpaca_position.qty_available)) if hasattr(alpaca_position, 'qty_available') and alpaca_position.qty_available else None,
            )
        except Exception as e:
            logger.error(f"Error converting Alpaca position: {e}")
            return None

    def _convert_from_alpaca_status(self, alpaca_status) -> OrderStatus:
        """Convert Alpaca order status to our OrderStatus."""
        status_map = {
            'new': OrderStatus.NEW,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.DONE_FOR_DAY,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.REPLACED,
            'pending_cancel': OrderStatus.PENDING_CANCEL,
            'pending_replace': OrderStatus.PENDING_REPLACE,
            'accepted': OrderStatus.ACCEPTED,
            'pending_new': OrderStatus.PENDING_NEW,
            'accepted_for_bidding': OrderStatus.ACCEPTED_FOR_BIDDING,
            'stopped': OrderStatus.STOPPED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.SUSPENDED,
            'calculated': OrderStatus.CALCULATED,
        }
        return status_map.get(str(alpaca_status).lower(), OrderStatus.PENDING)

    def _convert_to_alpaca_status(self, status: OrderStatus) -> str:
        """Convert our OrderStatus to Alpaca status string."""
        status_map = {
            OrderStatus.PENDING: 'pending',
            OrderStatus.NEW: 'new',
            OrderStatus.PARTIALLY_FILLED: 'partially_filled',
            OrderStatus.FILLED: 'filled',
            OrderStatus.DONE_FOR_DAY: 'done_for_day',
            OrderStatus.CANCELED: 'canceled',
            OrderStatus.EXPIRED: 'expired',
            OrderStatus.REPLACED: 'replaced',
            OrderStatus.PENDING_CANCEL: 'pending_cancel',
            OrderStatus.PENDING_REPLACE: 'pending_replace',
            OrderStatus.ACCEPTED: 'accepted',
            OrderStatus.PENDING_NEW: 'pending_new',
            OrderStatus.ACCEPTED_FOR_BIDDING: 'accepted_for_bidding',
            OrderStatus.STOPPED: 'stopped',
            OrderStatus.REJECTED: 'rejected',
            OrderStatus.SUSPENDED: 'suspended',
            OrderStatus.CALCULATED: 'calculated',
        }
        return status_map.get(status, 'pending')

    async def withdraw_funds(self, amount: Decimal) -> bool:
        """
        Withdraw funds from the account.

        Note: Alpaca doesn't support direct withdrawals via API.
        This would need to be done through ACH transfer on the Alpaca website.

        Args:
            amount: Amount to withdraw

        Returns:
            bool: True if withdrawal initiated (always False for Alpaca)
        """
        logger.warning(f"Withdrawal of ${amount} requested but Alpaca doesn't support API withdrawals")
        return False

    async def get_last_withdrawal_id(self) -> Optional[str]:
        """
        Get the ID of the last withdrawal transaction.

        Note: Alpaca doesn't support withdrawal tracking via API.

        Returns:
            Optional[str]: None (Alpaca doesn't support this)
        """
        return None

    # ISS-012: Production features merged from alpaca_production.py

    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid hitting API limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_ttl

    async def close_position(self, symbol: str) -> bool:
        """
        Close a position completely.

        Args:
            symbol: Symbol to close

        Returns:
            True if position closed successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")

        self._rate_limit()

        try:
            order = await self._safe_api_call(
                self.trading_client.close_position, symbol
            )
            logger.info(f"Closed position: {symbol}, Order ID: {order.id if order else 'N/A'}")

            # Clear from cache
            if symbol in self._position_cache:
                del self._position_cache[symbol]

            return True

        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.warning(f"No position found for {symbol}")
                return True  # Position doesn't exist, consider it closed
            logger.error(f"Failed to close position: {e}")
            return False

    async def close_all_positions(self) -> bool:
        """
        Close ALL open positions (EMERGENCY USE).

        Returns:
            True if all positions closed successfully
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")

        self._rate_limit()

        try:
            orders = await self._safe_api_call(
                self.trading_client.close_all_positions, cancel_orders=True
            )
            logger.warning(f"CLOSED ALL POSITIONS: {len(orders) if orders else 0} orders submitted")

            # Clear position cache
            self._position_cache.clear()
            return True

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    async def get_current_price(self, symbol: str) -> Decimal:
        """
        Get current price for a symbol (alias for get_market_price).

        Args:
            symbol: Stock symbol

        Returns:
            Current price as Decimal
        """
        price = await self.get_market_price(symbol)
        return price if price is not None else Decimal("0")

    async def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours information.

        Returns:
            Dict with market hours information
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")

        self._rate_limit()

        try:
            clock = await self._safe_api_call(self.trading_client.get_clock)
            if clock:
                return {
                    'is_open': clock.is_open,
                    'next_open': clock.next_open.isoformat() if clock.next_open else None,
                    'next_close': clock.next_close.isoformat() if clock.next_close else None,
                    'timestamp': clock.timestamp.isoformat() if hasattr(clock, 'timestamp') and clock.timestamp else None
                }
            return {'is_open': False}

        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {'is_open': False}

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

        mode = "PAPER" if self.is_paper_trading else "LIVE"
        return f"CONNECTED ({mode})"

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on broker connection.

        Returns:
            Dict: Health status information
        """
        try:
            clock = await self._safe_api_call(self.trading_client.get_clock)
            self.last_heartbeat = time.time()
            return {
                "connected": self.is_connected,
                "paper_trading": self.is_paper_trading,
                "timestamp": datetime.now(timezone.utc),
                "broker": self.__class__.__name__,
                "market_open": clock.is_open if clock else False,
                "healthy": True
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "connected": False,
                "paper_trading": self.is_paper_trading,
                "timestamp": datetime.now(timezone.utc),
                "broker": self.__class__.__name__,
                "healthy": False,
                "error": str(e)
            }