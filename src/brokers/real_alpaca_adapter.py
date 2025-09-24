"""
REAL Alpaca broker adapter - NO MOCKS, PRODUCTION READY
Implements real trading with Alpaca API
"""

import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
import os

from .broker_interface import (
    BrokerInterface, Order, Position, Fill, OrderStatus, OrderType, TimeInForce,
    BrokerError, ConnectionError, AuthenticationError, InsufficientFundsError,
    InvalidOrderError, MarketClosedError, RateLimitError
)

# REAL Alpaca imports - no fallbacks
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

logger = logging.getLogger(__name__)


class RealAlpacaAdapter(BrokerInterface):
    """REAL Alpaca broker adapter - NO MOCKS"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize real Alpaca adapter.

        Args:
            config: Configuration dictionary containing:
                - api_key: Alpaca API key (REQUIRED)
                - secret_key: Alpaca secret key (REQUIRED)
                - paper_trading: Whether to use paper trading (default: True)
                - base_url: Optional custom base URL
        """
        super().__init__(config)

        # REQUIRE real credentials
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')

        if not self.api_key or not self.secret_key:
            # Try environment variables
            self.api_key = os.environ.get('ALPACA_API_KEY')
            self.secret_key = os.environ.get('ALPACA_SECRET_KEY')

            if not self.api_key or not self.secret_key:
                raise ValueError(
                    "REAL Alpaca credentials required. Set api_key/secret_key in config "
                    "or ALPACA_API_KEY/ALPACA_SECRET_KEY environment variables"
                )

        self.base_url = config.get('base_url')

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.stream_client = None

        # Alpaca supports up to 6 decimal places for fractional shares
        self.qty_precision = 6

        # Operating mode
        mode = "PAPER" if self.is_paper_trading else "LIVE"
        logger.info(f"Initialized REAL Alpaca adapter - Mode: {mode}")

    async def connect(self) -> bool:
        """Connect to real Alpaca API."""
        try:
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
                logger.info(f"Connected to REAL Alpaca (Account: {account.account_number})")
                logger.info(f"  Portfolio Value: ${account.portfolio_value}")
                logger.info(f"  Cash: ${account.cash}")
                logger.info(f"  Buying Power: ${account.buying_power}")
                return True

            return False

        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            if "authentication" in str(e).lower():
                raise AuthenticationError(f"Invalid Alpaca credentials: {e}")
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self.is_connected = False
        if self.stream_client:
            await self.stream_client.close()
        logger.info("Disconnected from Alpaca")

    async def get_account_balance(self) -> Decimal:
        """Get account cash balance."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            account = await self._safe_api_call(self.trading_client.get_account)
            if account:
                return Decimal(str(account.cash))
            return Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise BrokerError(f"Failed to get account balance: {e}")

    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            alpaca_positions = await self._safe_api_call(self.trading_client.get_all_positions)
            positions = []

            for pos in alpaca_positions or []:
                position = Position(
                    symbol=pos.symbol,
                    quantity=Decimal(str(pos.qty)),
                    avg_price=Decimal(str(pos.avg_entry_price)),
                    current_price=Decimal(str(pos.current_price)) if pos.current_price else Decimal("0"),
                    unrealized_pnl=Decimal(str(pos.unrealized_pl)) if pos.unrealized_pl else Decimal("0"),
                    realized_pnl=Decimal(str(pos.realized_pl)) if pos.realized_pl else Decimal("0")
                )
                positions.append(position)

            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            pos = await self._safe_api_call(
                lambda: self.trading_client.get_open_position(symbol)
            )

            if pos:
                return Position(
                    symbol=pos.symbol,
                    quantity=Decimal(str(pos.qty)),
                    avg_price=Decimal(str(pos.avg_entry_price)),
                    current_price=Decimal(str(pos.current_price)) if pos.current_price else Decimal("0"),
                    unrealized_pnl=Decimal(str(pos.unrealized_pl)) if pos.unrealized_pl else Decimal("0"),
                    realized_pnl=Decimal(str(pos.realized_pl)) if pos.realized_pl else Decimal("0")
                )

            return None
        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise BrokerError(f"Failed to get position: {e}")

    async def place_order(self, order: Order) -> str:
        """Place an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Create appropriate order request based on type
            order_data = self._create_order_request(order)

            # Submit order
            alpaca_order = await self._safe_api_call(
                lambda: self.trading_client.submit_order(order_data)
            )

            if alpaca_order:
                logger.info(f"Placed order {alpaca_order.id} for {order.symbol}")
                return str(alpaca_order.id)

            raise InvalidOrderError("Failed to place order - no response from Alpaca")

        except APIError as e:
            error_msg = str(e)
            if "insufficient" in error_msg.lower():
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif "market is closed" in error_msg.lower():
                raise MarketClosedError(f"Market is closed: {e}")
            elif "rate limit" in error_msg.lower():
                raise RateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise InvalidOrderError(f"Invalid order: {e}")
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise BrokerError(f"Failed to place order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            await self._safe_api_call(
                lambda: self.trading_client.cancel_order_by_id(order_id)
            )
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            if "order not found" in str(e).lower():
                return False
            raise BrokerError(f"Failed to cancel order: {e}")

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            alpaca_order = await self._safe_api_call(
                lambda: self.trading_client.get_order_by_id(order_id)
            )

            if alpaca_order:
                return self._convert_alpaca_order(alpaca_order)

            return None
        except Exception as e:
            if "order not found" in str(e).lower():
                return None
            logger.error(f"Failed to get order {order_id}: {e}")
            raise BrokerError(f"Failed to get order: {e}")

    async def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders with optional status filter."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            request = {}
            if status:
                request['status'] = status.lower()

            alpaca_orders = await self._safe_api_call(
                lambda: self.trading_client.get_orders(filter=request)
            )

            orders = []
            for alpaca_order in alpaca_orders or []:
                order = self._convert_alpaca_order(alpaca_order)
                if order:
                    orders.append(order)

            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise BrokerError(f"Failed to get orders: {e}")

    async def get_fills(self, symbol: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Fill]:
        """Get filled orders (trades)."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Get filled orders
            alpaca_orders = await self._safe_api_call(
                lambda: self.trading_client.get_orders(
                    filter={'status': 'filled', 'symbols': symbol} if symbol else {'status': 'filled'}
                )
            )

            fills = []
            for order in alpaca_orders or []:
                if order.filled_at:
                    filled_time = order.filled_at
                    if start_date and filled_time < start_date:
                        continue
                    if end_date and filled_time > end_date:
                        continue

                    fill = Fill(
                        order_id=str(order.id),
                        symbol=order.symbol,
                        quantity=Decimal(str(order.filled_qty)),
                        price=Decimal(str(order.filled_avg_price)),
                        timestamp=filled_time,
                        side=order.side,
                        commission=Decimal("0")  # Alpaca has zero commissions
                    )
                    fills.append(fill)

            return fills
        except Exception as e:
            logger.error(f"Failed to get fills: {e}")
            raise BrokerError(f"Failed to get fills: {e}")

    def _create_order_request(self, order: Order):
        """Create Alpaca order request from Order object."""
        # Convert side
        side = OrderSide.BUY if order.side.upper() == 'BUY' else OrderSide.SELL

        # Convert time in force
        tif_map = {
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK
        }
        time_in_force = tif_map.get(order.time_in_force, AlpacaTimeInForce.DAY)

        # Round quantity to 6 decimal places
        quantity = float(order.quantity.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

        # Create appropriate order request
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=quantity,
                side=side,
                time_in_force=time_in_force
            )
        elif order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=quantity,
                side=side,
                time_in_force=time_in_force,
                limit_price=float(order.limit_price)
            )
        elif order.order_type == OrderType.STOP:
            return StopOrderRequest(
                symbol=order.symbol,
                qty=quantity,
                side=side,
                time_in_force=time_in_force,
                stop_price=float(order.stop_price)
            )
        elif order.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=quantity,
                side=side,
                time_in_force=time_in_force,
                stop_price=float(order.stop_price),
                limit_price=float(order.limit_price)
            )
        elif order.order_type == OrderType.TRAILING_STOP:
            return TrailingStopOrderRequest(
                symbol=order.symbol,
                qty=quantity,
                side=side,
                time_in_force=time_in_force,
                trail_price=float(order.trail_amount) if order.trail_amount else None,
                trail_percent=float(order.trail_percent) if order.trail_percent else None
            )
        else:
            raise InvalidOrderError(f"Unsupported order type: {order.order_type}")

    def _convert_alpaca_order(self, alpaca_order: AlpacaOrder) -> Order:
        """Convert Alpaca order to Order object."""
        # Map Alpaca status to our status
        status_map = {
            AlpacaOrderStatus.NEW: OrderStatus.PENDING,
            AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
            AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
            AlpacaOrderStatus.EXPIRED: OrderStatus.EXPIRED,
            AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
            AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
            AlpacaOrderStatus.PENDING_CANCEL: OrderStatus.PENDING,
            AlpacaOrderStatus.ACCEPTED: OrderStatus.PENDING,
        }

        # Map Alpaca order type to our type
        type_map = {
            AlpacaOrderType.MARKET: OrderType.MARKET,
            AlpacaOrderType.LIMIT: OrderType.LIMIT,
            AlpacaOrderType.STOP: OrderType.STOP,
            AlpacaOrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
            AlpacaOrderType.TRAILING_STOP: OrderType.TRAILING_STOP
        }

        return Order(
            order_id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            quantity=Decimal(str(alpaca_order.qty)),
            side=alpaca_order.side,
            order_type=type_map.get(alpaca_order.order_type, OrderType.MARKET),
            time_in_force=TimeInForce.DAY,  # Default
            limit_price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
            stop_price=Decimal(str(alpaca_order.stop_price)) if alpaca_order.stop_price else None,
            status=status_map.get(alpaca_order.status, OrderStatus.PENDING),
            filled_quantity=Decimal(str(alpaca_order.filled_qty)) if alpaca_order.filled_qty else Decimal("0"),
            filled_price=Decimal(str(alpaca_order.filled_avg_price)) if alpaca_order.filled_avg_price else None,
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at
        )

    async def _safe_api_call(self, func, max_retries: int = 3):
        """Execute API call with retry logic."""
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func)
            except APIError as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(1)
                else:
                    raise

        return None