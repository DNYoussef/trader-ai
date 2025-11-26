"""
Broker integration layer for Gary x Taleb trading system.

This module provides broker abstractions and implementations for executing trades
across different brokers with unified interfaces.
"""

from .broker_interface import (
    BrokerInterface, Order, Position, Fill, OrderStatus, OrderType, TimeInForce,
    BrokerError, ConnectionError, AuthenticationError, InsufficientFundsError,
    InvalidOrderError, MarketClosedError, RateLimitError
)
from .alpaca_adapter import AlpacaAdapter

# ISS-012: Backwards compatibility alias - AlpacaProductionAdapter merged into AlpacaAdapter
AlpacaProductionAdapter = AlpacaAdapter

__all__ = [
    'BrokerInterface',
    'Order',
    'Position',
    'Fill',
    'OrderStatus',
    'OrderType',
    'TimeInForce',
    'AlpacaAdapter',
    'AlpacaProductionAdapter',  # Deprecated: Use AlpacaAdapter
    # Exceptions
    'BrokerError',
    'ConnectionError',
    'AuthenticationError',
    'InsufficientFundsError',
    'InvalidOrderError',
    'MarketClosedError',
    'RateLimitError',
]