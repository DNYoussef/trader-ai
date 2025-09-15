"""
Abstract broker interface for the Gary×Taleb trading system.

Provides standardized interfaces for orders, positions, fills, and broker operations
that can be implemented by different broker adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    OPG = "opg"  # At the Opening
    CLS = "cls"  # At the Close


@dataclass
class Order:
    """Represents a trading order."""
    id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str = ""
    qty: Decimal = Decimal("0")
    notional: Optional[Decimal] = None
    side: str = ""  # "buy" or "sell"
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trail_price: Optional[Decimal] = None
    trail_percent: Optional[Decimal] = None
    extended_hours: bool = False
    status: Optional[OrderStatus] = None
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    replaced_at: Optional[datetime] = None
    replaced_by: Optional[str] = None
    replaces: Optional[str] = None
    asset_id: Optional[str] = None
    asset_class: str = "us_equity"
    commission: Optional[Decimal] = None
    legs: Optional[List[Dict[str, Any]]] = None


@dataclass
class Position:
    """Represents a trading position."""
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    avg_entry_price: Decimal
    qty: Decimal
    side: str  # "long" or "short"
    market_value: Optional[Decimal] = None
    cost_basis: Optional[Decimal] = None
    unrealized_pl: Optional[Decimal] = None
    unrealized_plpc: Optional[Decimal] = None
    unrealized_intraday_pl: Optional[Decimal] = None
    unrealized_intraday_plpc: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    lastday_price: Optional[Decimal] = None
    change_today: Optional[Decimal] = None
    swap_rate: Optional[Decimal] = None
    avg_entry_swap_rate: Optional[Decimal] = None
    usd_rate: Optional[Decimal] = None
    qty_available: Optional[Decimal] = None


@dataclass
class Fill:
    """Represents a trade fill."""
    id: str
    order_id: str
    execution_id: str
    symbol: str
    qty: Decimal
    price: Decimal
    side: str  # "buy" or "sell"
    timestamp: datetime
    commission: Optional[Decimal] = None
    liquidity_flag: Optional[str] = None


class BrokerInterface(ABC):
    """Abstract interface for broker implementations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize broker with configuration."""
        self.config = config
        self.is_connected = False
        self.is_paper_trading = config.get('paper_trading', True)

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        pass

    @abstractmethod
    async def get_account_value(self) -> Decimal:
        """
        Get total account value including cash and positions.
        
        Returns:
            Decimal: Total account value in USD
        """
        pass

    @abstractmethod
    async def get_cash_balance(self) -> Decimal:
        """
        Get available cash balance.
        
        Returns:
            Decimal: Available cash balance in USD
        """
        pass

    @abstractmethod
    async def get_buying_power(self) -> Decimal:
        """
        Get buying power (may include margin).
        
        Returns:
            Decimal: Available buying power in USD
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List[Position]: List of current positions
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Optional[Position]: Position if exists, None otherwise
        """
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Order: Updated order with broker response data
            
        Raises:
            BrokerError: If order submission fails
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Order]: Order if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_orders(self, 
                        status: Optional[OrderStatus] = None,
                        limit: int = 100,
                        after: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        direction: str = "desc") -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            status: Filter by order status
            limit: Maximum number of orders to return
            after: Filter orders after this time
            until: Filter orders before this time
            direction: Sort direction ("asc" or "desc")
            
        Returns:
            List[Order]: List of orders matching criteria
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            int: Number of orders canceled
        """
        pass

    @abstractmethod
    async def get_market_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Optional[Decimal]: Current market price if available
        """
        pass

    @abstractmethod
    async def get_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get last trade information for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Optional[Dict]: Last trade data if available
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current quote (bid/ask) for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Optional[Dict]: Quote data if available
        """
        pass

    @abstractmethod
    async def withdraw_funds(self, amount: Decimal) -> bool:
        """
        Withdraw funds from the account.

        Args:
            amount: Amount to withdraw in USD

        Returns:
            bool: True if withdrawal successful, False otherwise

        Raises:
            InsufficientFundsError: If insufficient funds for withdrawal
            BrokerError: If withdrawal fails for other reasons
        """
        pass

    @abstractmethod
    async def get_last_withdrawal_id(self) -> Optional[str]:
        """
        Get the ID of the last withdrawal transaction.

        Returns:
            Optional[str]: Withdrawal transaction ID if available
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on broker connection.

        Returns:
            Dict: Health status information
        """
        return {
            "connected": self.is_connected,
            "paper_trading": self.is_paper_trading,
            "timestamp": datetime.utcnow(),
            "broker": self.__class__.__name__
        }


class BrokerError(Exception):
    """Base exception for broker-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ConnectionError(BrokerError):
    """Raised when broker connection fails."""
    pass


class AuthenticationError(BrokerError):
    """Raised when broker authentication fails."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when account has insufficient funds for an operation."""
    pass


class InvalidOrderError(BrokerError):
    """Raised when an order is invalid."""
    pass


class MarketClosedError(BrokerError):
    """Raised when attempting to trade while market is closed."""
    pass


class RateLimitError(BrokerError):
    """Raised when API rate limit is exceeded."""
    pass