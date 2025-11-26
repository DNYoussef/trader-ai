"""
ISS-005: State Bridge for Dashboard-Engine Communication.

Simple file-based state store that allows TradingEngine to publish state
and Dashboard to read it. This is a lightweight alternative to Redis.
"""
import json
import logging
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import time

logger = logging.getLogger(__name__)

# Default state file location
DEFAULT_STATE_FILE = "data/trading_state.json"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class StateBridge:
    """
    File-based state bridge for dashboard-engine communication.

    The TradingEngine writes its state to a JSON file, and the dashboard
    reads from it. This provides a simple, dependency-free way to share
    state between processes.
    """

    def __init__(self, state_file: str = DEFAULT_STATE_FILE):
        """
        Initialize state bridge.

        Args:
            state_file: Path to the state file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_update = 0

    def publish_state(self, state: Dict[str, Any]) -> bool:
        """
        Publish trading engine state.

        Args:
            state: State dictionary to publish

        Returns:
            True if successful
        """
        try:
            with self._lock:
                # Add timestamp
                state['_timestamp'] = datetime.now().isoformat()
                state['_epoch'] = time.time()

                # Write atomically (write to temp, then rename)
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state, f, cls=DecimalEncoder, indent=2)

                # Rename to final file (atomic on most systems)
                temp_file.replace(self.state_file)
                self._last_update = time.time()

                return True

        except Exception as e:
            logger.error(f"Failed to publish state: {e}")
            return False

    def read_state(self) -> Optional[Dict[str, Any]]:
        """
        Read current trading engine state.

        Returns:
            State dictionary or None if not available
        """
        try:
            if not self.state_file.exists():
                return None

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Check if state is stale (older than 5 minutes)
            epoch = state.get('_epoch', 0)
            age_seconds = time.time() - epoch
            state['_age_seconds'] = age_seconds
            state['_is_stale'] = age_seconds > 300  # 5 minutes

            return state

        except Exception as e:
            logger.error(f"Failed to read state: {e}")
            return None

    def publish_positions(self, positions: List[Dict[str, Any]]) -> bool:
        """
        Publish current positions.

        Args:
            positions: List of position dictionaries

        Returns:
            True if successful
        """
        state = self.read_state() or {}
        state['positions'] = positions
        return self.publish_state(state)

    def publish_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Publish risk metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            True if successful
        """
        state = self.read_state() or {}
        state['metrics'] = metrics
        return self.publish_state(state)

    def publish_engine_status(self, status: Dict[str, Any]) -> bool:
        """
        Publish engine status (running, mode, kill_switch, etc).

        Args:
            status: Status dictionary

        Returns:
            True if successful
        """
        state = self.read_state() or {}
        state['engine_status'] = status
        return self.publish_state(state)

    def publish_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Publish a trade execution.

        Args:
            trade: Trade details

        Returns:
            True if successful
        """
        state = self.read_state() or {}

        # Keep last 100 trades
        trades = state.get('recent_trades', [])
        trades.insert(0, {
            **trade,
            'timestamp': datetime.now().isoformat()
        })
        state['recent_trades'] = trades[:100]

        return self.publish_state(state)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        state = self.read_state()
        return state.get('positions', []) if state else []

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics."""
        state = self.read_state()
        return state.get('metrics') if state else None

    def get_engine_status(self) -> Optional[Dict[str, Any]]:
        """Get engine status."""
        state = self.read_state()
        return state.get('engine_status') if state else None

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades."""
        state = self.read_state()
        trades = state.get('recent_trades', []) if state else []
        return trades[:limit]

    def is_engine_running(self) -> bool:
        """Check if trading engine is running (state not stale)."""
        state = self.read_state()
        if not state:
            return False
        return not state.get('_is_stale', True)


# Global instance for easy access
_bridge: Optional[StateBridge] = None


def get_state_bridge(state_file: str = DEFAULT_STATE_FILE) -> StateBridge:
    """
    Get the global state bridge instance.

    Args:
        state_file: Path to state file

    Returns:
        StateBridge instance
    """
    global _bridge
    if _bridge is None:
        _bridge = StateBridge(state_file)
    return _bridge
