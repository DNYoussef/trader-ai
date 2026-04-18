"""
Milestone Portfolio Tracker

Track progress: $200 -> $300 -> $400 -> $500
Risk decreases at each milestone achieved.

Milestones:
  M0: $200-299  -> MAXIMUM AGGRESSION (90% SPY)
  M1: $300-399  -> HIGH AGGRESSION (80% SPY)
  M2: $400-499  -> MODERATE (70% SPY)
  M3: $500+     -> GOAL ACHIEVED - PROTECT (60% SPY)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Milestone(Enum):
    """Portfolio milestones."""
    M0_START = "M0_START"       # $200-299: Maximum aggression
    M1_TRACTION = "M1_TRACTION" # $300-399: High aggression
    M2_MOMENTUM = "M2_MOMENTUM" # $400-499: Moderate
    M3_GOAL = "M3_GOAL"         # $500+: Goal achieved


# Milestone configurations
MILESTONE_CONFIG = {
    Milestone.M0_START: {
        'capital_min': 200,
        'capital_max': 299.99,
        'name': 'START',
        'target': 300,
        'spy_pct': 0.90,      # MAXIMUM aggression
        'tlt_pct': 0.08,
        'cash_pct': 0.02,
        'description': 'Maximum aggression - grow fast!',
        'color': '#ef4444',   # Red - aggressive
    },
    Milestone.M1_TRACTION: {
        'capital_min': 300,
        'capital_max': 399.99,
        'name': 'TRACTION',
        'target': 400,
        'spy_pct': 0.80,      # High aggression
        'tlt_pct': 0.15,
        'cash_pct': 0.05,
        'description': 'High aggression - keep pushing',
        'color': '#f97316',   # Orange
    },
    Milestone.M2_MOMENTUM: {
        'capital_min': 400,
        'capital_max': 499.99,
        'name': 'MOMENTUM',
        'target': 500,
        'spy_pct': 0.70,      # Moderate
        'tlt_pct': 0.20,
        'cash_pct': 0.10,
        'description': 'Moderate risk - almost there!',
        'color': '#eab308',   # Yellow
    },
    Milestone.M3_GOAL: {
        'capital_min': 500,
        'capital_max': float('inf'),
        'name': 'GOAL ACHIEVED',
        'target': None,
        'spy_pct': 0.60,      # Protect gains
        'tlt_pct': 0.25,
        'cash_pct': 0.15,
        'description': 'Goal achieved - protect gains!',
        'color': '#22c55e',   # Green - success
    },
}


@dataclass
class MilestoneEvent:
    """Record of a milestone achievement."""
    milestone: str
    capital: float
    timestamp: str
    days_from_start: int
    previous_milestone: Optional[str] = None


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: str
    capital: float
    milestone: str
    allocation: Dict[str, float]
    daily_return: float
    total_return: float


@dataclass
class MilestoneState:
    """Current state of milestone tracking."""
    current_capital: float = 200.0
    current_milestone: str = "M0_START"
    start_date: str = ""
    start_capital: float = 200.0

    # Tracking
    milestone_history: List[Dict] = field(default_factory=list)
    daily_snapshots: List[Dict] = field(default_factory=list)

    # Stats
    peak_capital: float = 200.0
    max_drawdown: float = 0.0
    days_at_current_milestone: int = 0


class MilestoneTracker:
    """
    Track portfolio progress through milestones.

    Usage:
        tracker = MilestoneTracker()
        tracker.update_capital(225.50)
        allocation = tracker.get_current_allocation()
        tracker.print_status()
    """

    def __init__(self, data_dir: str = "D:/Projects/trader-ai/data/milestones"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state = MilestoneState()
        self._load_state()

        if not self.state.start_date:
            self.state.start_date = date.today().isoformat()

    def _get_milestone_for_capital(self, capital: float) -> Milestone:
        """Determine milestone based on capital."""
        for milestone, config in MILESTONE_CONFIG.items():
            if config['capital_min'] <= capital <= config['capital_max']:
                return milestone
        return Milestone.M3_GOAL if capital >= 500 else Milestone.M0_START

    def update_capital(self, new_capital: float, record_date: str = None) -> Dict:
        """
        Update current capital and check for milestone changes.

        Returns dict with status information.
        """
        old_capital = self.state.current_capital
        old_milestone = Milestone(self.state.current_milestone)

        self.state.current_capital = new_capital
        new_milestone = self._get_milestone_for_capital(new_capital)

        # Track peak and drawdown
        if new_capital > self.state.peak_capital:
            self.state.peak_capital = new_capital

        drawdown = (self.state.peak_capital - new_capital) / self.state.peak_capital
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

        # Calculate days from start
        start = date.fromisoformat(self.state.start_date)
        today = date.fromisoformat(record_date) if record_date else date.today()
        days_from_start = (today - start).days

        # Check for milestone change
        milestone_changed = new_milestone != old_milestone

        if milestone_changed:
            event = MilestoneEvent(
                milestone=new_milestone.value,
                capital=new_capital,
                timestamp=datetime.now().isoformat(),
                days_from_start=days_from_start,
                previous_milestone=old_milestone.value
            )
            self.state.milestone_history.append(asdict(event))
            self.state.current_milestone = new_milestone.value
            self.state.days_at_current_milestone = 0

            logger.info(f"MILESTONE ACHIEVED: {old_milestone.value} -> {new_milestone.value} "
                       f"(${new_capital:.2f}, day {days_from_start})")
        else:
            self.state.days_at_current_milestone += 1

        # Record daily snapshot
        config = MILESTONE_CONFIG[new_milestone]
        daily_return = (new_capital - old_capital) / old_capital if old_capital > 0 else 0
        total_return = (new_capital - self.state.start_capital) / self.state.start_capital

        snapshot = DailySnapshot(
            date=record_date or date.today().isoformat(),
            capital=new_capital,
            milestone=new_milestone.value,
            allocation={
                'spy': config['spy_pct'],
                'tlt': config['tlt_pct'],
                'cash': config['cash_pct']
            },
            daily_return=daily_return,
            total_return=total_return
        )
        self.state.daily_snapshots.append(asdict(snapshot))

        self._save_state()

        return {
            'capital': new_capital,
            'milestone': new_milestone.value,
            'milestone_changed': milestone_changed,
            'allocation': config,
            'days_from_start': days_from_start,
            'total_return': total_return * 100,
            'to_next_milestone': self._distance_to_next(new_capital, new_milestone),
        }

    def _distance_to_next(self, capital: float, milestone: Milestone) -> Optional[Dict]:
        """Calculate distance to next milestone."""
        config = MILESTONE_CONFIG[milestone]
        target = config.get('target')

        if target is None:
            return None

        return {
            'target': target,
            'remaining': target - capital,
            'progress_pct': (capital - config['capital_min']) / (target - config['capital_min']) * 100
        }

    def get_current_allocation(self) -> Tuple[float, float, float]:
        """Get current allocation based on milestone."""
        milestone = Milestone(self.state.current_milestone)
        config = MILESTONE_CONFIG[milestone]
        return (config['spy_pct'], config['tlt_pct'], config['cash_pct'])

    def get_status(self) -> Dict:
        """Get comprehensive status report."""
        milestone = Milestone(self.state.current_milestone)
        config = MILESTONE_CONFIG[milestone]

        start = date.fromisoformat(self.state.start_date)
        days = (date.today() - start).days

        return {
            'current_capital': self.state.current_capital,
            'start_capital': self.state.start_capital,
            'total_return_pct': (self.state.current_capital - self.state.start_capital) / self.state.start_capital * 100,
            'current_milestone': milestone.value,
            'milestone_name': config['name'],
            'milestone_description': config['description'],
            'allocation': {
                'spy': config['spy_pct'] * 100,
                'tlt': config['tlt_pct'] * 100,
                'cash': config['cash_pct'] * 100,
            },
            'days_trading': days,
            'peak_capital': self.state.peak_capital,
            'max_drawdown_pct': self.state.max_drawdown * 100,
            'milestones_achieved': len(self.state.milestone_history),
            'to_next': self._distance_to_next(self.state.current_capital, milestone),
            'to_goal': 500 - self.state.current_capital if self.state.current_capital < 500 else 0,
        }

    def print_status(self):
        """Print formatted status."""
        status = self.get_status()
        milestone = Milestone(self.state.current_milestone)
        config = MILESTONE_CONFIG[milestone]

        print()
        print("=" * 60)
        print("MILESTONE TRACKER")
        print("=" * 60)
        print()

        # Progress bar
        progress = min(100, (status['current_capital'] - 200) / 3 * 100)  # $200-$500 = 0-100%
        bar_filled = int(progress / 2)
        bar_empty = 50 - bar_filled
        print(f"  GOAL: $200 -----> $500")
        print(f"  [{'#' * bar_filled}{'-' * bar_empty}] {progress:.0f}%")
        print()

        # Current status
        print(f"  Current Capital:  ${status['current_capital']:.2f}")
        print(f"  Total Return:     {status['total_return_pct']:+.1f}%")
        print(f"  Days Trading:     {status['days_trading']}")
        print()

        # Milestone status
        print(f"  MILESTONE: {config['name']}")
        print(f"  {config['description']}")
        print()

        # Allocation
        print(f"  Current Allocation:")
        print(f"    SPY:  {status['allocation']['spy']:.0f}%")
        print(f"    TLT:  {status['allocation']['tlt']:.0f}%")
        print(f"    Cash: {status['allocation']['cash']:.0f}%")
        print()

        # Next milestone
        if status['to_next']:
            print(f"  Next Target: ${status['to_next']['target']}")
            print(f"  Remaining:   ${status['to_next']['remaining']:.2f}")
            print(f"  Progress:    {status['to_next']['progress_pct']:.0f}%")
        else:
            print(f"  GOAL ACHIEVED!")

        print()
        print("=" * 60)

        # Milestone journey
        print()
        print("MILESTONE JOURNEY:")
        print("-" * 40)

        milestones = [
            ("M0: $200-299", "START", 0.90, status['current_capital'] >= 200),
            ("M1: $300-399", "TRACTION", 0.80, status['current_capital'] >= 300),
            ("M2: $400-499", "MOMENTUM", 0.70, status['current_capital'] >= 400),
            ("M3: $500+", "GOAL!", 0.60, status['current_capital'] >= 500),
        ]

        for range_str, name, spy, achieved in milestones:
            marker = "[X]" if achieved else "[ ]"
            arrow = " <-- YOU" if (achieved and not (status['current_capital'] >=
                                   [200, 300, 400, 500][milestones.index((range_str, name, spy, achieved))+1]
                                   if milestones.index((range_str, name, spy, achieved)) < 3 else True)) else ""
            print(f"  {marker} {range_str} | {name:<10} | SPY: {spy*100:.0f}%{arrow}")

        print()

    def _save_state(self):
        """Save state to disk."""
        state_file = self.data_dir / "milestone_state.json"
        with open(state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)

    def _load_state(self):
        """Load state from disk."""
        state_file = self.data_dir / "milestone_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.state = MilestoneState(**data)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def reset(self, start_capital: float = 200.0):
        """Reset tracker to initial state."""
        self.state = MilestoneState(
            current_capital=start_capital,
            start_capital=start_capital,
            start_date=date.today().isoformat(),
            peak_capital=start_capital,
        )
        self._save_state()
        logger.info(f"Tracker reset to ${start_capital}")


def print_milestone_schedule():
    """Print the full milestone schedule."""
    print()
    print("=" * 70)
    print("MILESTONE SCHEDULE: $200 -> $500")
    print("=" * 70)
    print()
    print(f"{'Milestone':<12} {'Range':<15} {'SPY%':>8} {'TLT%':>8} {'Cash%':>8} {'Risk Level':<15}")
    print("-" * 70)

    for milestone, config in MILESTONE_CONFIG.items():
        range_str = f"${config['capital_min']:.0f}-${config['capital_max']:.0f}" if config['capital_max'] < 1000 else f"${config['capital_min']:.0f}+"
        print(f"{config['name']:<12} {range_str:<15} {config['spy_pct']*100:>7.0f}% {config['tlt_pct']*100:>7.0f}% {config['cash_pct']*100:>7.0f}%  {config['description']}")

    print()
    print("STRATEGY:")
    print("  - Start AGGRESSIVE (90% SPY) to grow fast")
    print("  - Reduce risk at each milestone achieved")
    print("  - Protect gains once goal is reached")
    print("=" * 70)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print_milestone_schedule()

    # Demo
    print()
    print("DEMO: Simulating progress...")
    print("-" * 40)

    tracker = MilestoneTracker()
    tracker.reset(200)

    # Simulate some updates
    for capital in [200, 225, 250, 280, 310, 350, 390, 420, 480, 510]:
        result = tracker.update_capital(capital)
        if result['milestone_changed']:
            print(f"  ${capital}: MILESTONE ACHIEVED -> {result['milestone']}")
        else:
            print(f"  ${capital}: {result['milestone']}")

    print()
    tracker.print_status()
