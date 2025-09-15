"""
Production Portfolio Manager for Gary×Taleb trading system.

Manages real portfolio positions, tracks performance, calculates NAV,
and handles deposits/withdrawals with full broker integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: Decimal
    gate: str
    last_updated: datetime


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time."""
    date: datetime
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    total_pnl: Decimal
    total_pnl_percent: Decimal
    positions: Dict[str, Position]


class PortfolioManager:
    """
    Production Portfolio Manager with real broker integration.

    Tracks positions, performance, NAV calculation, and transaction history
    across multiple gates (SPY hedge, ULTY+AMDY momentum, etc.).
    """

    def __init__(self, broker_adapter, market_data_provider, initial_capital: Decimal = Decimal("200.00")):
        """
        Initialize portfolio manager.

        Args:
            broker_adapter: Connected broker adapter for position/account data
            market_data_provider: Real market data provider
            initial_capital: Starting capital amount
        """
        self.broker = broker_adapter
        self.market_data = market_data_provider
        self.initial_capital = initial_capital
        self.inception_date = datetime.now(timezone.utc).date()

        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash_balance = initial_capital
        self.total_deposits = initial_capital
        self.total_withdrawals = Decimal("0.00")

        # Performance tracking
        self.daily_snapshots: Dict[date, PortfolioSnapshot] = {}
        self.transaction_history: List[Dict[str, Any]] = []

        # Gates configuration
        self.gates = {
            'SPY_HEDGE': {'allocation': Decimal('0.40'), 'symbols': ['SPY']},
            'MOMENTUM': {'allocation': Decimal('0.35'), 'symbols': ['ULTY', 'AMDY']},
            'BOND_HEDGE': {'allocation': Decimal('0.15'), 'symbols': ['VTIP']},
            'GOLD_HEDGE': {'allocation': Decimal('0.10'), 'symbols': ['IAU']}
        }

        logger.info(f"Portfolio Manager initialized with ${initial_capital} capital")

    async def sync_with_broker(self) -> bool:
        """
        Synchronize portfolio state with actual broker positions.

        Returns:
            bool: True if sync successful
        """
        try:
            if not self.broker.is_connected:
                logger.error("Broker not connected - cannot sync portfolio")
                return False

            # Get real account info
            account_value = await self.broker.get_account_value()
            cash_balance = await self.broker.get_cash_balance()

            # Get all positions from broker
            broker_positions = await self.broker.get_positions()

            # Update our portfolio state
            self.cash_balance = cash_balance
            self.positions.clear()

            for pos in broker_positions:
                if pos.qty > 0:  # Only track long positions
                    current_price = await self.market_data.get_current_price(pos.symbol)
                    if current_price is None:
                        current_price = pos.current_price or Decimal("0.00")

                    gate = self._determine_gate(pos.symbol)

                    portfolio_pos = Position(
                        symbol=pos.symbol,
                        quantity=pos.qty,
                        avg_cost=pos.avg_entry_price,
                        current_price=Decimal(str(current_price)),
                        market_value=pos.market_value or (pos.qty * Decimal(str(current_price))),
                        unrealized_pnl=pos.unrealized_pl or Decimal("0.00"),
                        unrealized_pnl_percent=pos.unrealized_plpc or Decimal("0.00"),
                        gate=gate,
                        last_updated=datetime.now(timezone.utc)
                    )

                    self.positions[pos.symbol] = portfolio_pos

            logger.info(f"Portfolio synced - Value: ${account_value}, Cash: ${cash_balance}, Positions: {len(self.positions)}")
            return True

        except Exception as e:
            logger.error(f"Failed to sync with broker: {e}")
            return False

    async def get_gate_positions(self, gate: str) -> Dict[str, Position]:
        """
        Get positions for a specific gate.

        Args:
            gate: Gate name (SPY_HEDGE, MOMENTUM, etc.)

        Returns:
            Dict of symbol -> Position for the gate
        """
        gate_positions = {}

        if gate in self.gates:
            gate_symbols = self.gates[gate]['symbols']
            for symbol in gate_symbols:
                if symbol in self.positions:
                    gate_positions[symbol] = self.positions[symbol]
        else:
            # Return positions matching the gate name
            for symbol, position in self.positions.items():
                if position.gate == gate:
                    gate_positions[symbol] = position

        return gate_positions

    async def get_gate_allocation(self, gate: str) -> Decimal:
        """Get current allocation percentage for a gate."""
        gate_positions = await self.get_gate_positions(gate)
        gate_value = sum(pos.market_value for pos in gate_positions.values())
        total_value = await self.get_total_portfolio_value()

        if total_value > 0:
            return (gate_value / total_value) * Decimal("100")
        return Decimal("0.00")

    async def get_total_portfolio_value(self) -> Decimal:
        """Get total portfolio value (cash + positions)."""
        await self.sync_with_broker()  # Refresh from broker

        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value

    async def get_nav_at_date(self, target_date: date) -> Decimal:
        """
        Get NAV at specific date.

        Args:
            target_date: Date to get NAV for

        Returns:
            NAV value at that date
        """
        if target_date in self.daily_snapshots:
            return self.daily_snapshots[target_date].total_value

        # If exact date not found, find closest previous date
        closest_date = None
        for snapshot_date in sorted(self.daily_snapshots.keys(), reverse=True):
            if snapshot_date <= target_date:
                closest_date = snapshot_date
                break

        if closest_date:
            return self.daily_snapshots[closest_date].total_value

        # Fallback to current value if no historical data
        return await self.get_total_portfolio_value()

    def get_deposits_in_period(self, start_date: date, end_date: date) -> Decimal:
        """Get total deposits in the specified period."""
        total_deposits = Decimal("0.00")

        for transaction in self.transaction_history:
            transaction_date = transaction['date'].date() if isinstance(transaction['date'], datetime) else transaction['date']

            if (start_date <= transaction_date <= end_date and
                transaction['type'] == 'deposit'):
                total_deposits += Decimal(str(transaction['amount']))

        return total_deposits

    def get_withdrawals_in_period(self, start_date: date, end_date: date) -> Decimal:
        """Get total withdrawals in the specified period."""
        total_withdrawals = Decimal("0.00")

        for transaction in self.transaction_history:
            transaction_date = transaction['date'].date() if isinstance(transaction['date'], datetime) else transaction['date']

            if (start_date <= transaction_date <= end_date and
                transaction['type'] == 'withdrawal'):
                total_withdrawals += Decimal(str(transaction['amount']))

        return total_withdrawals

    async def calculate_performance_metrics(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            period_days: Period to calculate metrics for

        Returns:
            Dictionary with performance metrics
        """
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=period_days)

        current_value = await self.get_total_portfolio_value()
        start_value = await self.get_nav_at_date(start_date)

        deposits = self.get_deposits_in_period(start_date, end_date)
        withdrawals = self.get_withdrawals_in_period(start_date, end_date)

        # Calculate time-weighted return
        adjusted_start_value = start_value + deposits - withdrawals
        total_return = current_value - adjusted_start_value
        total_return_percent = (total_return / adjusted_start_value * Decimal("100")) if adjusted_start_value > 0 else Decimal("0.00")

        # Calculate annualized return
        years = Decimal(str(period_days)) / Decimal("365.25")
        annualized_return = (((current_value / adjusted_start_value) ** (Decimal("1") / years)) - Decimal("1")) * Decimal("100") if adjusted_start_value > 0 and years > 0 else Decimal("0.00")

        return {
            'period_days': period_days,
            'start_value': start_value,
            'current_value': current_value,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'annualized_return_percent': annualized_return,
            'deposits': deposits,
            'withdrawals': withdrawals,
            'net_deposits': deposits - withdrawals
        }

    async def create_daily_snapshot(self) -> PortfolioSnapshot:
        """Create and store daily portfolio snapshot."""
        await self.sync_with_broker()

        total_value = await self.get_total_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())

        # Calculate total P&L vs initial investment
        total_invested = self.total_deposits - self.total_withdrawals
        total_pnl = total_value - total_invested
        total_pnl_percent = (total_pnl / total_invested * Decimal("100")) if total_invested > 0 else Decimal("0.00")

        snapshot = PortfolioSnapshot(
            date=datetime.now(timezone.utc),
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            positions=self.positions.copy()
        )

        today = datetime.now(timezone.utc).date()
        self.daily_snapshots[today] = snapshot

        logger.info(f"Daily snapshot created - Value: ${total_value}, P&L: ${total_pnl} ({total_pnl_percent}%)")
        return snapshot

    async def record_transaction(self, transaction_type: str, amount: Decimal, symbol: str = None,
                                quantity: Decimal = None, price: Decimal = None, gate: str = None) -> None:
        """Record a transaction in the history."""
        transaction = {
            'date': datetime.now(timezone.utc),
            'type': transaction_type,  # deposit, withdrawal, buy, sell
            'amount': amount,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'gate': gate,
            'portfolio_value': await self.get_total_portfolio_value()
        }

        self.transaction_history.append(transaction)

        # Update totals for deposits/withdrawals
        if transaction_type == 'deposit':
            self.total_deposits += amount
        elif transaction_type == 'withdrawal':
            self.total_withdrawals += amount

        logger.info(f"Transaction recorded: {transaction_type} ${amount} {symbol or ''}")

    def _determine_gate(self, symbol: str) -> str:
        """Determine which gate a symbol belongs to."""
        for gate_name, gate_config in self.gates.items():
            if symbol in gate_config['symbols']:
                return gate_name

        # Default gate assignment logic
        if symbol == 'SPY':
            return 'SPY_HEDGE'
        elif symbol in ['ULTY', 'AMDY']:
            return 'MOMENTUM'
        elif symbol in ['VTIP', 'TIP']:
            return 'BOND_HEDGE'
        elif symbol in ['IAU', 'GLD']:
            return 'GOLD_HEDGE'
        else:
            return 'OTHER'

    async def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        await self.sync_with_broker()

        total_value = await self.get_total_portfolio_value()

        gate_summaries = {}
        for gate_name in self.gates.keys():
            gate_positions = await self.get_gate_positions(gate_name)
            gate_value = sum(pos.market_value for pos in gate_positions.values())
            gate_allocation = (gate_value / total_value * Decimal("100")) if total_value > 0 else Decimal("0.00")

            gate_summaries[gate_name] = {
                'positions': len(gate_positions),
                'value': gate_value,
                'allocation_percent': gate_allocation,
                'target_allocation_percent': self.gates[gate_name]['allocation'] * Decimal("100"),
                'symbols': list(gate_positions.keys())
            }

        return {
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'positions_count': len(self.positions),
            'gates': gate_summaries,
            'last_updated': datetime.now(timezone.utc)
        }