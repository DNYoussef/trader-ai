"""
Alpaca broker adapter for the Gary×Taleb trading system.

Implements the BrokerInterface for Alpaca Trading API with support for both
paper and live trading modes, including fractional shares up to 6 decimal places.
"""

import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

from .broker_interface import (
    BrokerInterface, Order, Position, Fill, OrderStatus, OrderType, TimeInForce,
    BrokerError, ConnectionError, AuthenticationError, InsufficientFundsError,
    InvalidOrderError, MarketClosedError, RateLimitError
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


class MockAlpacaClient:
    """Mock Alpaca client for development without Alpaca library."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self._account_value = Decimal("100000.00")
        self._cash_balance = Decimal("50000.00")
        self._positions = {}
        self._orders = {}
        
    async def get_account(self):
        """Mock account data."""
        return type('MockAccount', (), {
            'portfolio_value': str(self._account_value),
            'cash': str(self._cash_balance),
            'buying_power': str(self._cash_balance * 2),  # Simulate 2:1 margin
            'regt_buying_power': str(self._cash_balance * 2),
            'daytrading_buying_power': str(self._cash_balance * 4),
            'non_marginable_buying_power': str(self._cash_balance),
        })()
    
    async def get_all_positions(self):
        """Mock positions data."""
        return []
    
    async def get_open_position(self, symbol: str):
        """Mock single position data."""
        return None
    
    async def submit_order(self, order_data):
        """Mock order submission."""
        order_id = str(uuid.uuid4())
        mock_order = type('MockOrder', (), {
            'id': order_id,
            'client_order_id': getattr(order_data, 'client_order_id', None),
            'symbol': order_data.symbol,
            'qty': str(getattr(order_data, 'qty', 0)) if hasattr(order_data, 'qty') else None,
            'notional': str(getattr(order_data, 'notional', 0)) if hasattr(order_data, 'notional') and order_data.notional else None,
            'side': order_data.side.value,
            'order_type': order_data.type.value if hasattr(order_data, 'type') else 'market',
            'time_in_force': order_data.time_in_force.value,
            'limit_price': str(getattr(order_data, 'limit_price', 0)) if hasattr(order_data, 'limit_price') and order_data.limit_price else None,
            'stop_price': str(getattr(order_data, 'stop_price', 0)) if hasattr(order_data, 'stop_price') and order_data.stop_price else None,
            'trail_price': str(getattr(order_data, 'trail_price', 0)) if hasattr(order_data, 'trail_price') and order_data.trail_price else None,
            'trail_percent': str(getattr(order_data, 'trail_percent', 0)) if hasattr(order_data, 'trail_percent') and order_data.trail_percent else None,
            'extended_hours': getattr(order_data, 'extended_hours', False),
            'status': 'filled',
            'filled_qty': str(getattr(order_data, 'qty', 0)),
            'filled_avg_price': '100.00',  # Mock price
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'submitted_at': datetime.now(timezone.utc),
            'filled_at': datetime.now(timezone.utc),
            'asset_id': str(uuid.uuid4()),
            'asset_class': 'us_equity',
        })()
        
        self._orders[order_id] = mock_order
        return mock_order
    
    async def get_order_by_id(self, order_id: str):
        """Mock get order by ID."""
        return self._orders.get(order_id)
    
    async def get_orders(self, **kwargs):
        """Mock get orders."""
        return list(self._orders.values())
    
    async def cancel_order_by_id(self, order_id: str):
        """Mock cancel order."""
        if order_id in self._orders:
            self._orders[order_id].status = 'canceled'
            return True
        return False
    
    async def cancel_all_orders(self):
        """Mock cancel all orders."""
        canceled = 0
        for order in self._orders.values():
            if order.status in ['new', 'partially_filled', 'pending_new']:
                order.status = 'canceled'
                canceled += 1
        return canceled


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
                - mock_mode: Force mock mode even if alpaca-py is available (default: False)
        """
        super().__init__(config)

        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.base_url = config.get('base_url')
        self.mock_mode = config.get('mock_mode', False)

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.stream_client = None

        # Alpaca supports up to 6 decimal places for fractional shares
        self.qty_precision = 6

        # Validate credentials for production mode (unless in mock mode)
        if ALPACA_AVAILABLE and not self.mock_mode and not (self.api_key and self.secret_key):
            raise ValueError("PRODUCTION ERROR: API credentials required when alpaca-py is available")

        # Determine operating mode
        if self.mock_mode or not ALPACA_AVAILABLE:
            mode = "MOCK"
        elif self.is_paper_trading:
            mode = "PAPER"
        else:
            mode = "LIVE"

        logger.info(f"Initialized Alpaca adapter - Mode: {mode}, Library Available: {ALPACA_AVAILABLE}, Mock Mode: {self.mock_mode}")

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            if self.mock_mode or not ALPACA_AVAILABLE:
                if not ALPACA_AVAILABLE:
                    logger.warning("Alpaca library not available. Install with: pip install alpaca-py")
                logger.info("Using mock mode for development/testing")
                self.trading_client = MockAlpacaClient(
                    api_key=self.api_key or "mock_key",
                    secret_key=self.secret_key or "mock_secret",
                    paper=self.is_paper_trading
                )
                self.is_connected = True
                return True
            
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
                logger.info(f"Connected to Alpaca (Account: {account.account_number if hasattr(account, 'account_number') else 'Mock'})")
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
                # Mock: assume market is open during business hours
                now = datetime.now()
                return 9 <= now.hour < 16  # Simplified market hours
            
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
                # Mock price
                return Decimal("100.00")
            
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
                # Mock trade data
                return {
                    'symbol': symbol,
                    'price': 100.00,
                    'size': 100,
                    'timestamp': datetime.now(timezone.utc)
                }
            
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
                # Mock quote data
                return {
                    'symbol': symbol,
                    'bid': 99.50,
                    'ask': 100.50,
                    'bid_size': 100,
                    'ask_size': 100,
                    'midpoint': 100.00,
                    'timestamp': datetime.now(timezone.utc)
                }
            
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
            # Return mock request object
            return type('MockOrderRequest', (), {
                'symbol': order.symbol,
                'qty': order.qty,
                'notional': order.notional,
                'side': type('Side', (), {'value': order.side})(),
                'type': type('Type', (), {'value': order.order_type.value})(),
                'time_in_force': type('TIF', (), {'value': order.time_in_force.value})(),
                'limit_price': order.limit_price,
                'stop_price': order.stop_price,
                'trail_price': order.trail_price,
                'trail_percent': order.trail_percent,
                'extended_hours': order.extended_hours,
                'client_order_id': order.client_order_id
            })()
        
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