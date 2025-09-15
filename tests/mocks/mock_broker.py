"""
Mock broker implementation for testing Foundation phase components.
Provides realistic simulation of broker interactions without external dependencies.
"""
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class MockPosition:
    """Mock position object for testing"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.avg_cost
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.avg_cost) * self.quantity


@dataclass
class MockOrder:
    """Mock order object for testing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    commission: float = 0.0


class MockBrokerError(Exception):
    """Mock broker-specific errors for testing error handling"""
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code


class MockBroker:
    """
    Comprehensive mock broker for testing Foundation phase components.
    Simulates realistic broker behavior including delays, errors, and state changes.
    """
    
    def __init__(self, 
                 connection_delay: float = 0.1,
                 order_fill_delay: float = 0.5,
                 error_rate: float = 0.0,
                 simulate_partial_fills: bool = False):
        """
        Initialize mock broker with configurable behavior
        
        Args:
            connection_delay: Simulated connection delay in seconds
            order_fill_delay: Simulated order processing delay
            error_rate: Probability of random errors (0.0-1.0)
            simulate_partial_fills: Whether to simulate partial order fills
        """
        self.connection_delay = connection_delay
        self.order_fill_delay = order_fill_delay
        self.error_rate = error_rate
        self.simulate_partial_fills = simulate_partial_fills
        
        # Connection state
        self.connected = False
        self.last_connection_time = None
        self.connection_count = 0
        
        # Market data
        self.market_prices = {}
        self.price_volatility = 0.01  # 1% default volatility
        
        # Orders and positions
        self.orders: Dict[str, MockOrder] = {}
        self.positions: Dict[str, MockPosition] = {}
        self.account_balance = 100000.0  # Default account balance
        self.buying_power = 100000.0
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks for testing
        self.on_order_update = None
        self.on_position_update = None
        self.on_connection_change = None

    def connect(self) -> bool:
        """Mock broker connection with realistic delay"""
        if self.connection_delay > 0:
            time.sleep(self.connection_delay)
            
        # Simulate occasional connection failures
        if self.error_rate > 0 and self._should_error():
            raise MockBrokerError("Connection failed", 1001)
            
        with self._lock:
            self.connected = True
            self.last_connection_time = datetime.now()
            self.connection_count += 1
            
        if self.on_connection_change:
            self.on_connection_change(True)
            
        return True

    def disconnect(self) -> bool:
        """Mock broker disconnection"""
        with self._lock:
            self.connected = False
            
        if self.on_connection_change:
            self.on_connection_change(False)
            
        return True

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected

    def place_order(self, symbol: str, quantity: float, 
                   order_type: OrderType = OrderType.MARKET,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> MockOrder:
        """
        Place a mock order with realistic behavior
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (positive for buy, negative for sell)
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            MockOrder object
            
        Raises:
            MockBrokerError: Various order-related errors
        """
        if not self.connected:
            raise MockBrokerError("Not connected to broker", 1002)
            
        if abs(quantity) == 0:
            raise MockBrokerError("Invalid quantity", 2001)
            
        if symbol not in self.market_prices:
            self.market_prices[symbol] = 100.0  # Default price
            
        # Check buying power for buy orders
        current_price = self.get_current_price(symbol)
        required_capital = abs(quantity) * current_price
        
        if quantity > 0 and required_capital > self.buying_power:
            raise MockBrokerError("Insufficient buying power", 2002)
            
        # Create order
        order = MockOrder(
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING
        )
        
        with self._lock:
            self.orders[order.id] = order
            
        # Simulate order processing in background
        if self.order_fill_delay > 0:
            threading.Thread(
                target=self._process_order,
                args=(order,),
                daemon=True
            ).start()
        else:
            self._fill_order(order)
            
        return order

    def get_order(self, order_id: str) -> Optional[MockOrder]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_orders(self, symbol: str = None) -> List[MockOrder]:
        """Get all orders, optionally filtered by symbol"""
        orders = list(self.orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if not self.connected:
            raise MockBrokerError("Not connected to broker", 1002)
            
        order = self.orders.get(order_id)
        if not order:
            raise MockBrokerError("Order not found", 2003)
            
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise MockBrokerError("Cannot cancel order in current status", 2004)
            
        with self._lock:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
        if self.on_order_update:
            self.on_order_update(order)
            
        return True

    def get_position(self, symbol: str) -> Optional[MockPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    def get_positions(self) -> List[MockPosition]:
        """Get all positions"""
        return list(self.positions.values())

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.account_balance

    def get_buying_power(self) -> float:
        """Get current buying power"""
        return self.buying_power

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        if symbol not in self.market_prices:
            self.market_prices[symbol] = 100.0
            
        # Simulate price movement
        base_price = self.market_prices[symbol]
        volatility_factor = 1 + (self.price_volatility * (0.5 - hash(str(datetime.now().microsecond)) % 100 / 100))
        
        return base_price * volatility_factor

    def set_market_price(self, symbol: str, price: float):
        """Set market price for testing (test helper method)"""
        self.market_prices[symbol] = price
        
        # Update position market values
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = price
            position.market_value = position.quantity * price
            position.unrealized_pnl = (price - position.avg_cost) * position.quantity
            position.last_updated = datetime.now()
            
            if self.on_position_update:
                self.on_position_update(position)

    def _should_error(self) -> bool:
        """Determine if an error should occur based on error rate"""
        import random
        return random.random() < self.error_rate

    def _process_order(self, order: MockOrder):
        """Process order in background thread"""
        time.sleep(self.order_fill_delay)
        
        if self._should_error():
            with self._lock:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()
                self.failed_trades += 1
        else:
            self._fill_order(order)

    def _fill_order(self, order: MockOrder):
        """Fill an order"""
        current_price = self.get_current_price(order.symbol)
        
        # Determine fill price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = current_price
        elif order.order_type == OrderType.LIMIT:
            if order.quantity > 0:  # Buy limit
                if current_price <= order.limit_price:
                    fill_price = order.limit_price
                else:
                    return  # Order not filled
            else:  # Sell limit
                if current_price >= order.limit_price:
                    fill_price = order.limit_price
                else:
                    return  # Order not filled
        else:
            fill_price = current_price  # Simplified for other order types
            
        # Simulate partial fills
        fill_quantity = order.quantity
        if self.simulate_partial_fills and abs(order.quantity) > 100:
            fill_quantity = order.quantity * 0.7  # Fill 70%
            
        with self._lock:
            # Update order
            order.filled_quantity += fill_quantity
            order.avg_fill_price = fill_price
            order.commission = abs(fill_quantity) * 0.01  # $0.01 per share
            
            if abs(order.filled_quantity) >= abs(order.quantity):
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIAL
                
            order.updated_at = datetime.now()
            
            # Update position
            self._update_position(order.symbol, fill_quantity, fill_price, order.commission)
            
            # Update account
            trade_cost = fill_quantity * fill_price + order.commission
            self.account_balance -= trade_cost
            self.buying_power = self.account_balance  # Simplified
            
            self.total_trades += 1
            self.successful_trades += 1
            
        # Trigger callbacks
        if self.on_order_update:
            self.on_order_update(order)
            
        if order.symbol in self.positions and self.on_position_update:
            self.on_position_update(self.positions[order.symbol])

    def _update_position(self, symbol: str, quantity: float, price: float, commission: float):
        """Update position after order fill"""
        if symbol not in self.positions:
            if quantity != 0:
                self.positions[symbol] = MockPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price
                )
        else:
            position = self.positions[symbol]
            
            # Calculate new average cost
            total_cost = (position.quantity * position.avg_cost) + (quantity * price)
            total_quantity = position.quantity + quantity
            
            if total_quantity != 0:
                position.avg_cost = total_cost / total_quantity
                position.quantity = total_quantity
            else:
                # Position closed
                position.realized_pnl += (price - position.avg_cost) * (-quantity)
                del self.positions[symbol]
                return
                
            position.current_price = self.get_current_price(symbol)
            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = (position.current_price - position.avg_cost) * position.quantity
            position.last_updated = datetime.now()

    def reset_for_testing(self):
        """Reset broker state for testing"""
        with self._lock:
            self.orders.clear()
            self.positions.clear()
            self.market_prices.clear()
            self.account_balance = 100000.0
            self.buying_power = 100000.0
            self.total_trades = 0
            self.successful_trades = 0
            self.failed_trades = 0
            self.connected = False
            self.connection_count = 0


def create_mock_broker(**kwargs) -> MockBroker:
    """Factory function to create mock broker with default settings"""
    return MockBroker(**kwargs)


def create_test_position(symbol: str = "AAPL", 
                        quantity: float = 100, 
                        avg_cost: float = 150.0) -> MockPosition:
    """Factory function to create test position"""
    return MockPosition(
        symbol=symbol,
        quantity=quantity,
        avg_cost=avg_cost,
        current_price=avg_cost
    )


def create_test_order(symbol: str = "AAPL",
                     quantity: float = 100,
                     order_type: OrderType = OrderType.MARKET,
                     status: OrderStatus = OrderStatus.PENDING) -> MockOrder:
    """Factory function to create test order"""
    return MockOrder(
        symbol=symbol,
        quantity=quantity,
        order_type=order_type,
        status=status
    )