"""
Production Trade Executor for Gary×Taleb trading system.

Executes real market orders through Alpaca with proper risk management,
order validation, and gate-based position tracking.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: Optional[Decimal]
    notional: Optional[Decimal]  # Dollar amount for fractional shares
    status: str
    filled_quantity: Decimal
    filled_price: Optional[Decimal]
    gate: str
    timestamp: datetime
    broker_response: Dict[str, Any]


class TradeExecutor:
    """
    Production Trade Executor with real broker integration.

    Handles market orders, fractional shares, risk management,
    and position tracking across multiple gates.
    """

    def __init__(self, broker_adapter, portfolio_manager, market_data_provider):
        """
        Initialize trade executor.

        Args:
            broker_adapter: Connected broker adapter
            portfolio_manager: Portfolio manager instance
            market_data_provider: Market data provider
        """
        self.broker = broker_adapter
        self.portfolio = portfolio_manager
        self.market_data = market_data_provider

        # Risk management settings
        self.max_position_size_percent = Decimal("40.0")  # Max 40% in any single position
        self.min_order_amount = Decimal("1.00")  # Minimum $1 order
        self.max_order_amount = Decimal("1000.00")  # Maximum $1000 order
        self.market_impact_threshold = Decimal("0.5")  # 0.5% market impact tolerance

        # Order tracking
        self.pending_orders: Dict[str, OrderResult] = {}
        self.completed_orders: List[OrderResult] = []

        logger.info("Trade Executor initialized with production broker integration")

    async def buy_market_order(self, symbol: str, dollar_amount: Decimal, gate: str) -> OrderResult:
        """
        Execute a buy market order for a specific dollar amount.

        Args:
            symbol: Symbol to buy
            dollar_amount: Dollar amount to invest
            gate: Gate designation (SPY_HEDGE, MOMENTUM, etc.)

        Returns:
            OrderResult with execution details
        """
        try:
            # Validate inputs
            await self._validate_order_params(symbol, dollar_amount, "buy", gate)

            # Get current market price for validation
            current_price = await self.market_data.get_current_price(symbol)
            if current_price is None:
                raise ValueError(f"Unable to get current price for {symbol}")

            # Check buying power
            buying_power = await self.broker.get_buying_power()
            if dollar_amount > buying_power:
                raise ValueError(f"Insufficient buying power: ${buying_power} < ${dollar_amount}")

            # Risk management checks
            await self._validate_position_size(symbol, dollar_amount, "buy")

            # Create order
            order_id = str(uuid.uuid4())
            client_order_id = f"buy_{gate}_{symbol}_{int(datetime.now().timestamp())}"

            # Use notional (dollar amount) order for fractional shares
            from ..brokers.broker_interface import Order, OrderType, TimeInForce

            order = Order(
                id=None,
                client_order_id=client_order_id,
                symbol=symbol,
                qty=None,  # Use notional instead
                notional=dollar_amount,
                side="buy",
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                extended_hours=False
            )

            logger.info(f"Submitting BUY order: {symbol} ${dollar_amount} (Gate: {gate})")

            # Submit to broker
            submitted_order = await self.broker.submit_order(order)

            # Create order result
            result = OrderResult(
                order_id=submitted_order.id,
                client_order_id=client_order_id,
                symbol=symbol,
                side="buy",
                quantity=submitted_order.qty,
                notional=dollar_amount,
                status=submitted_order.status.value if submitted_order.status else "submitted",
                filled_quantity=submitted_order.filled_qty or Decimal("0"),
                filled_price=submitted_order.filled_avg_price,
                gate=gate,
                timestamp=datetime.now(timezone.utc),
                broker_response={
                    'order_id': submitted_order.id,
                    'status': submitted_order.status.value if submitted_order.status else "unknown",
                    'submitted_at': str(submitted_order.submitted_at) if submitted_order.submitted_at else None
                }
            )

            # Track order
            self.pending_orders[submitted_order.id] = result

            # Record transaction in portfolio
            await self.portfolio.record_transaction(
                transaction_type="buy",
                amount=dollar_amount,
                symbol=symbol,
                quantity=result.filled_quantity,
                price=result.filled_price,
                gate=gate
            )

            logger.info(f"BUY order submitted: {symbol} ${dollar_amount} - Order ID: {submitted_order.id}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute BUY order {symbol} ${dollar_amount}: {e}")
            # Return error result
            return OrderResult(
                order_id=f"error_{uuid.uuid4()}",
                client_order_id=f"buy_error_{symbol}",
                symbol=symbol,
                side="buy",
                quantity=None,
                notional=dollar_amount,
                status="error",
                filled_quantity=Decimal("0"),
                filled_price=None,
                gate=gate,
                timestamp=datetime.now(timezone.utc),
                broker_response={'error': str(e)}
            )

    async def sell_market_order(self, symbol: str, dollar_amount: Decimal, gate: str) -> OrderResult:
        """
        Execute a sell market order for a specific dollar amount.

        Args:
            symbol: Symbol to sell
            dollar_amount: Dollar amount to sell
            gate: Gate designation

        Returns:
            OrderResult with execution details
        """
        try:
            # Validate inputs
            await self._validate_order_params(symbol, dollar_amount, "sell", gate)

            # Get current position
            position = await self.broker.get_position(symbol)
            if not position or position.qty <= 0:
                raise ValueError(f"No position to sell for {symbol}")

            # Get current market price
            current_price = await self.market_data.get_current_price(symbol)
            if current_price is None:
                raise ValueError(f"Unable to get current price for {symbol}")

            # Calculate quantity to sell based on dollar amount
            current_position_value = position.market_value or (position.qty * Decimal(str(current_price)))

            if dollar_amount > current_position_value:
                # Sell entire position if requested amount exceeds position value
                sell_quantity = position.qty
                actual_dollar_amount = current_position_value
                logger.warning(f"Requested sell ${dollar_amount} > position value ${current_position_value}. Selling entire position.")
            else:
                # Calculate fractional quantity
                sell_quantity = (dollar_amount / Decimal(str(current_price))).quantize(
                    Decimal('0.000001'), rounding=ROUND_HALF_UP
                )
                actual_dollar_amount = dollar_amount

            # Create order
            order_id = str(uuid.uuid4())
            client_order_id = f"sell_{gate}_{symbol}_{int(datetime.now().timestamp())}"

            from ..brokers.broker_interface import Order, OrderType, TimeInForce

            order = Order(
                id=None,
                client_order_id=client_order_id,
                symbol=symbol,
                qty=sell_quantity,
                notional=None,
                side="sell",
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                extended_hours=False
            )

            logger.info(f"Submitting SELL order: {symbol} {sell_quantity} shares ~${actual_dollar_amount} (Gate: {gate})")

            # Submit to broker
            submitted_order = await self.broker.submit_order(order)

            # Create order result
            result = OrderResult(
                order_id=submitted_order.id,
                client_order_id=client_order_id,
                symbol=symbol,
                side="sell",
                quantity=sell_quantity,
                notional=actual_dollar_amount,
                status=submitted_order.status.value if submitted_order.status else "submitted",
                filled_quantity=submitted_order.filled_qty or Decimal("0"),
                filled_price=submitted_order.filled_avg_price,
                gate=gate,
                timestamp=datetime.now(timezone.utc),
                broker_response={
                    'order_id': submitted_order.id,
                    'status': submitted_order.status.value if submitted_order.status else "unknown",
                    'submitted_at': str(submitted_order.submitted_at) if submitted_order.submitted_at else None
                }
            )

            # Track order
            self.pending_orders[submitted_order.id] = result

            # Record transaction in portfolio
            await self.portfolio.record_transaction(
                transaction_type="sell",
                amount=actual_dollar_amount,
                symbol=symbol,
                quantity=result.filled_quantity,
                price=result.filled_price,
                gate=gate
            )

            logger.info(f"SELL order submitted: {symbol} {sell_quantity} shares - Order ID: {submitted_order.id}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute SELL order {symbol} ${dollar_amount}: {e}")
            # Return error result
            return OrderResult(
                order_id=f"error_{uuid.uuid4()}",
                client_order_id=f"sell_error_{symbol}",
                symbol=symbol,
                side="sell",
                quantity=None,
                notional=dollar_amount,
                status="error",
                filled_quantity=Decimal("0"),
                filled_price=None,
                gate=gate,
                timestamp=datetime.now(timezone.utc),
                broker_response={'error': str(e)}
            )

    async def rebalance_gate(self, gate: str, target_allocations: Dict[str, Decimal]) -> List[OrderResult]:
        """
        Rebalance positions within a gate to target allocations.

        Args:
            gate: Gate to rebalance
            target_allocations: Dict of symbol -> target dollar amount

        Returns:
            List of OrderResult from rebalancing orders
        """
        results = []

        try:
            # Get current gate positions
            current_positions = await self.portfolio.get_gate_positions(gate)

            # Calculate orders needed
            for symbol, target_amount in target_allocations.items():
                current_position = current_positions.get(symbol)
                current_value = current_position.market_value if current_position else Decimal("0")

                difference = target_amount - current_value

                if abs(difference) > self.min_order_amount:
                    if difference > 0:
                        # Need to buy more
                        result = await self.buy_market_order(symbol, difference, gate)
                        results.append(result)
                    else:
                        # Need to sell some
                        result = await self.sell_market_order(symbol, abs(difference), gate)
                        results.append(result)

            logger.info(f"Gate {gate} rebalancing completed - {len(results)} orders executed")
            return results

        except Exception as e:
            logger.error(f"Failed to rebalance gate {gate}: {e}")
            return results

    async def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get current status of an order."""
        try:
            # Check if we're tracking this order
            if order_id in self.pending_orders:
                # Update from broker
                broker_order = await self.broker.get_order(order_id)
                if broker_order:
                    # Update our tracking
                    result = self.pending_orders[order_id]
                    result.status = broker_order.status.value if broker_order.status else "unknown"
                    result.filled_quantity = broker_order.filled_qty or Decimal("0")
                    result.filled_price = broker_order.filled_avg_price

                    # Move to completed if filled
                    if result.status.lower() in ['filled', 'canceled', 'rejected', 'expired']:
                        self.completed_orders.append(result)
                        del self.pending_orders[order_id]

                    return result

            # Check completed orders
            for result in self.completed_orders:
                if result.order_id == order_id:
                    return result

            return None

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            success = await self.broker.cancel_order(order_id)
            if success and order_id in self.pending_orders:
                result = self.pending_orders[order_id]
                result.status = "canceled"
                self.completed_orders.append(result)
                del self.pending_orders[order_id]
                logger.info(f"Order {order_id} canceled successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def _validate_order_params(self, symbol: str, dollar_amount: Decimal, side: str, gate: str) -> None:
        """Validate order parameters."""
        if not symbol or len(symbol) < 1:
            raise ValueError("Invalid symbol")

        if dollar_amount < self.min_order_amount:
            raise ValueError(f"Order amount ${dollar_amount} below minimum ${self.min_order_amount}")

        if dollar_amount > self.max_order_amount:
            raise ValueError(f"Order amount ${dollar_amount} above maximum ${self.max_order_amount}")

        if side not in ["buy", "sell"]:
            raise ValueError("Side must be 'buy' or 'sell'")

        if not gate:
            raise ValueError("Gate designation required")

        # Check if broker is connected
        if not self.broker.is_connected:
            raise ValueError("Broker not connected")

        # Check if market is open (for market orders)
        is_market_open = await self.broker.is_market_open()
        if not is_market_open:
            logger.warning(f"Market is closed - order {symbol} may be queued until open")

    async def _validate_position_size(self, symbol: str, dollar_amount: Decimal, side: str) -> None:
        """Validate position sizing rules."""
        if side == "buy":
            # Check if this order would create oversized position
            portfolio_value = await self.portfolio.get_total_portfolio_value()
            current_position = await self.portfolio.positions.get(symbol)
            current_value = current_position.market_value if current_position else Decimal("0")

            new_position_value = current_value + dollar_amount
            position_percent = (new_position_value / portfolio_value * Decimal("100")) if portfolio_value > 0 else Decimal("0")

            if position_percent > self.max_position_size_percent:
                raise ValueError(f"Position would be {position_percent}% of portfolio, exceeds {self.max_position_size_percent}% limit")

    def get_order_history(self, gate: str = None, limit: int = 100) -> List[OrderResult]:
        """Get order history, optionally filtered by gate."""
        orders = self.completed_orders + list(self.pending_orders.values())

        if gate:
            orders = [order for order in orders if order.gate == gate]

        # Sort by timestamp, most recent first
        orders.sort(key=lambda x: x.timestamp, reverse=True)

        return orders[:limit]

    def get_pending_orders(self) -> List[OrderResult]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    async def cancel_all_pending_orders(self) -> int:
        """Cancel all pending orders."""
        canceled_count = 0
        pending_order_ids = list(self.pending_orders.keys())

        for order_id in pending_order_ids:
            success = await self.cancel_order(order_id)
            if success:
                canceled_count += 1

        logger.info(f"Canceled {canceled_count} pending orders")
        return canceled_count