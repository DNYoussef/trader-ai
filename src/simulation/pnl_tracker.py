"""
P&L Tracker with Unrealized P&L

Based on Crayfer's insight about genetic algorithm bots:
- His GA bots learned to HIDE losses by never closing losing trades
- They showed amazing "realized" P&L but had massive hidden unrealized losses
- Solution: Track and penalize unrealized P&L in training

This module tracks:
- Realized P&L (closed positions)
- Unrealized P&L (open positions)
- Max drawdown (including unrealized)
- Stale loss detection (old losing positions)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Position:
    """Open trading position."""
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    entry_time: float
    position_id: int
    asset: str = 'SPY'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class ClosedTrade:
    """Closed/realized trade."""
    entry_price: float
    exit_price: float
    size: float
    side: str
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float
    hold_duration: float


class PnLTracker:
    """
    Tracks both realized and unrealized P&L.

    Critical for AI training to prevent the model from
    learning to hide losses in open positions.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.realized_pnl = 0.0
        self.peak_equity = initial_capital
        self._next_position_id = 0
        self._current_time = 0.0

        # Track equity curve
        self.equity_history: List[Tuple[float, float]] = []  # (time, equity)

    def _get_next_id(self) -> int:
        self._next_position_id += 1
        return self._next_position_id

    def set_time(self, t: float):
        """Set current simulation time."""
        self._current_time = t

    def advance_time(self, dt: float = 1.0):
        """Advance simulation time."""
        self._current_time += dt

    def open_position(
        self,
        price: float,
        size: float,
        side: str = 'long',
        asset: str = 'SPY',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            price: Entry price
            size: Position size (in dollars)
            side: 'long' or 'short'
            asset: Asset symbol
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            Position object
        """
        position = Position(
            entry_price=price,
            size=size,
            side=side,
            entry_time=self._current_time,
            position_id=self._get_next_id(),
            asset=asset,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.open_positions.append(position)
        self.cash -= size  # Allocate capital

        return position

    def close_position(self, position_id: int, exit_price: float) -> Optional[ClosedTrade]:
        """
        Close a position and realize P&L.

        Args:
            position_id: ID of position to close
            exit_price: Exit price

        Returns:
            ClosedTrade with P&L details
        """
        # Find position
        pos_idx = None
        for i, pos in enumerate(self.open_positions):
            if pos.position_id == position_id:
                pos_idx = i
                break

        if pos_idx is None:
            return None

        pos = self.open_positions.pop(pos_idx)

        # Calculate P&L
        if pos.side == 'long':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl = pnl_pct * pos.size
        hold_duration = self._current_time - pos.entry_time

        trade = ClosedTrade(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=self._current_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_duration=hold_duration
        )

        self.closed_trades.append(trade)
        self.realized_pnl += pnl
        self.cash += pos.size + pnl  # Return capital + P&L

        return trade

    def close_all_positions(self, prices: Dict[str, float]) -> List[ClosedTrade]:
        """Close all open positions at given prices."""
        trades = []
        position_ids = [p.position_id for p in self.open_positions]

        for pid in position_ids:
            pos = next((p for p in self.open_positions if p.position_id == pid), None)
            if pos:
                price = prices.get(pos.asset, pos.entry_price)
                trade = self.close_position(pid, price)
                if trade:
                    trades.append(trade)

        return trades

    def get_position_pnl(self, position: Position, current_price: float) -> float:
        """Calculate P&L for a single position."""
        if position.side == 'long':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price

        return pnl_pct * position.size

    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized P&L on open positions.

        This is the KEY metric that prevents hiding losses.
        """
        unrealized = 0.0
        for pos in self.open_positions:
            price = current_prices.get(pos.asset, pos.entry_price)
            unrealized += self.get_position_pnl(pos, price)
        return unrealized

    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.get_unrealized_pnl(current_prices)

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Get current total equity."""
        return self.initial_capital + self.get_total_pnl(current_prices)

    def get_max_drawdown(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate maximum drawdown including unrealized P&L.

        Returns percentage drawdown (0.10 = 10% drawdown).
        """
        current_equity = self.get_equity(current_prices)
        self.peak_equity = max(self.peak_equity, current_equity)

        if self.peak_equity <= 0:
            return 0.0

        return (self.peak_equity - current_equity) / self.peak_equity

    def get_stale_losing_positions(
        self,
        current_prices: Dict[str, float],
        max_age: float = 86400.0  # 1 day in seconds
    ) -> List[Tuple[Position, float]]:
        """
        Find positions that are losing AND old.

        These are the "hidden losses" that Crayfer's GA bots learned to accumulate.

        Returns:
            List of (position, unrealized_pnl) tuples
        """
        stale_losers = []

        for pos in self.open_positions:
            price = current_prices.get(pos.asset, pos.entry_price)
            pnl = self.get_position_pnl(pos, price)
            age = self._current_time - pos.entry_time

            if pnl < 0 and age > max_age:
                stale_losers.append((pos, pnl))

        return stale_losers

    def record_equity(self, current_prices: Dict[str, float]):
        """Record current equity for history."""
        equity = self.get_equity(current_prices)
        self.equity_history.append((self._current_time, equity))

    def get_trade_stats(self) -> Dict:
        """Get statistics on closed trades."""
        if not self.closed_trades:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_realized': 0.0,
            }

        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))

        return {
            'n_trades': len(self.closed_trades),
            'win_rate': len(wins) / len(self.closed_trades),
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': total_losses / len(losses) if losses else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'total_realized': self.realized_pnl,
        }

    def get_full_stats(self, current_prices: Dict[str, float]) -> Dict:
        """Get comprehensive P&L statistics."""
        trade_stats = self.get_trade_stats()

        unrealized = self.get_unrealized_pnl(current_prices)
        total_pnl = self.get_total_pnl(current_prices)
        equity = self.get_equity(current_prices)
        drawdown = self.get_max_drawdown(current_prices)
        stale_losers = self.get_stale_losing_positions(current_prices)

        return {
            **trade_stats,
            'unrealized_pnl': unrealized,
            'total_pnl': total_pnl,
            'equity': equity,
            'max_drawdown': drawdown,
            'n_open_positions': len(self.open_positions),
            'n_stale_losers': len(stale_losers),
            'stale_loss_amount': sum(pnl for _, pnl in stale_losers),
            'return_pct': (equity - self.initial_capital) / self.initial_capital,
        }


def compute_reward_with_pnl_tracking(
    tracker: PnLTracker,
    current_prices: Dict[str, float],
    base_reward: float
) -> float:
    """
    Compute reward that accounts for unrealized P&L.

    This prevents the model from hiding losses.

    Args:
        tracker: PnLTracker with current state
        current_prices: Dict of asset -> price
        base_reward: Base reward from returns

    Returns:
        Adjusted reward
    """
    unrealized = tracker.get_unrealized_pnl(current_prices)
    equity = tracker.get_equity(current_prices)

    # Base reward from total P&L (not just realized)
    total_pnl = tracker.get_total_pnl(current_prices)
    adjusted_reward = base_reward

    # PENALTY for large unrealized losses (prevent hiding)
    if unrealized < -0.02 * tracker.initial_capital:  # More than 2% unrealized loss
        unrealized_penalty = (unrealized / tracker.initial_capital) * 2.0  # 2x penalty
        adjusted_reward += unrealized_penalty

    # PENALTY for stale losing positions
    stale_losers = tracker.get_stale_losing_positions(current_prices)
    for pos, pnl in stale_losers:
        age_days = (tracker._current_time - pos.entry_time) / 86400
        stale_penalty = (pnl / tracker.initial_capital) * min(age_days, 5) * 0.01
        adjusted_reward += stale_penalty

    # BONUS for realized gains (encourages taking profits)
    if tracker.realized_pnl > 0:
        realized_bonus = (tracker.realized_pnl / tracker.initial_capital) * 0.1
        adjusted_reward += realized_bonus

    return adjusted_reward


def demo_pnl_tracker():
    """Demo the P&L tracker."""
    print("P&L Tracker Demo")
    print("=" * 50)

    tracker = PnLTracker(initial_capital=10000.0)

    # Simulate some trading
    tracker.set_time(0)

    # Open a winning position
    pos1 = tracker.open_position(price=100.0, size=2000.0, side='long', asset='SPY')
    print(f"Opened position 1: Long $2000 SPY at $100")

    # Open a losing position
    pos2 = tracker.open_position(price=100.0, size=1000.0, side='long', asset='SPY')
    print(f"Opened position 2: Long $1000 SPY at $100")

    # Price moves
    current_prices = {'SPY': 105.0}  # SPY up 5%
    tracker.advance_time(3600)  # 1 hour

    print(f"\nSPY price moved to ${current_prices['SPY']}")
    print(f"Unrealized P&L: ${tracker.get_unrealized_pnl(current_prices):.2f}")
    print(f"Equity: ${tracker.get_equity(current_prices):.2f}")

    # Close the winning position
    trade = tracker.close_position(pos1.position_id, exit_price=105.0)
    print(f"\nClosed position 1 at $105")
    print(f"Realized P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1%})")

    # Now price drops (position 2 becomes a loser)
    current_prices = {'SPY': 95.0}  # SPY down 5%
    tracker.advance_time(86400 * 2)  # 2 days

    print(f"\nSPY price dropped to ${current_prices['SPY']}")
    stats = tracker.get_full_stats(current_prices)

    print(f"\nFull Stats:")
    print(f"  Realized P&L: ${stats['total_realized']:.2f}")
    print(f"  Unrealized P&L: ${stats['unrealized_pnl']:.2f}")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Equity: ${stats['equity']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
    print(f"  Stale Losers: {stats['n_stale_losers']}")
    print(f"  Stale Loss Amount: ${stats['stale_loss_amount']:.2f}")

    # Compute adjusted reward
    base_reward = 0.05  # 5% return
    adjusted = compute_reward_with_pnl_tracking(tracker, current_prices, base_reward)
    print(f"\nReward Calculation:")
    print(f"  Base Reward: {base_reward:.2%}")
    print(f"  Adjusted Reward: {adjusted:.2%}")
    print(f"  (Penalized for unrealized loss and stale position)")


if __name__ == '__main__':
    demo_pnl_tracker()
