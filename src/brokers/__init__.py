"""
Broker integration layer for Gary×Taleb trading system.

This module provides broker abstractions and implementations for executing trades
across different brokers with unified interfaces.
"""

from .broker_interface import BrokerInterface, Order, Position, Fill, OrderStatus, OrderType, TimeInForce
from .alpaca_adapter import AlpacaAdapter

__all__ = [
    'BrokerInterface',
    'Order',
    'Position', 
    'Fill',
    'OrderStatus',
    'OrderType',
    'TimeInForce',
    'AlpacaAdapter',
]