"""
ISS-005: LiveDataProvider - Real-time data provider for dashboard.

Replaces mock data generation with real data from TradingEngine.
Supports both in-process connection and file-based state reading.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.integration.trading_state_provider import (
    TradingStateProvider,
    get_state_provider
)

logger = logging.getLogger(__name__)


class LiveDataProvider:
    """
    Live data provider that replaces mock data generation.

    This class provides the same interface as RealDataProvider but returns
    real data from the TradingEngine via TradingStateProvider.
    """

    def __init__(self, state_provider: Optional[TradingStateProvider] = None,
                 state_file: Optional[str] = None):
        """
        Initialize the live data provider.

        Args:
            state_provider: Optional TradingStateProvider for direct access
            state_file: Optional path to state file for file-based access
        """
        self._provider = state_provider or get_state_provider()
        self._state_file = Path(state_file) if state_file else Path("data/dashboard_state.json")
        self._cached_state: Dict[str, Any] = {}
        self._cache_time = 0
        self._cache_ttl = 0.5  # 500ms cache

        # Fallback to mock if no real data available
        self._use_mock_fallback = True

        logger.info("LiveDataProvider initialized")

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state from provider or file."""
        # Check cache
        if time.time() - self._cache_time < self._cache_ttl and self._cached_state:
            return self._cached_state

        # Try direct provider first
        if self._provider.is_connected:
            try:
                # Use sync wrapper for backwards compatibility
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - schedule coroutine
                    asyncio.ensure_future(self._provider.get_full_state())
                    # Can't await here in sync context, use cached or file
                    pass
                else:
                    state = loop.run_until_complete(self._provider.get_full_state())
                    self._cached_state = state
                    self._cache_time = time.time()
                    return state
            except Exception as e:
                logger.warning(f"Direct provider access failed: {e}")

        # Try file-based state
        state = self._provider.read_published_state()
        if state and not state.get('_is_stale', True):
            self._cached_state = state
            self._cache_time = time.time()
            return state

        # Return cached if available
        if self._cached_state:
            return self._cached_state

        # Return empty state
        return {}

    async def get_state_async(self) -> Dict[str, Any]:
        """Get current state asynchronously."""
        if self._provider.is_connected:
            try:
                state = await self._provider.get_full_state()
                self._cached_state = state
                self._cache_time = time.time()
                return state
            except Exception as e:
                logger.warning(f"Async state fetch failed: {e}")

        # Try file-based
        state = self._provider.read_published_state()
        if state:
            self._cached_state = state
            return state

        return self._cached_state or {}

    def generate_metrics(self) -> Dict[str, Any]:
        """
        Generate risk metrics for dashboard.

        Returns the same format as the original RealDataProvider.generate_metrics()
        """
        state = self._get_current_state()
        metrics = state.get('metrics', {})

        if not metrics:
            # Return default metrics
            return self._default_metrics()

        return {
            'timestamp': metrics.get('timestamp', time.time()),
            'portfolio_value': metrics.get('portfolio_value', 0),
            'p_ruin': metrics.get('p_ruin', 0),
            'var_95': metrics.get('var_95', 0),
            'var_99': metrics.get('var_99', 0),
            'expected_shortfall': metrics.get('expected_shortfall', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'volatility': metrics.get('volatility', 0),
            'beta': metrics.get('beta', 1.0),
            'positions_count': metrics.get('positions_count', 0),
            'cash_available': metrics.get('cash_available', 0),
            'margin_used': metrics.get('margin_used', 0),
            'unrealized_pnl': metrics.get('unrealized_pnl', 0),
            'daily_pnl': metrics.get('daily_pnl', 0),
            'buying_power': metrics.get('buying_power', 0),
            'source': state.get('source', 'unknown')
        }

    async def generate_metrics_async(self) -> Dict[str, Any]:
        """Async version of generate_metrics."""
        state = await self.get_state_async()
        metrics = state.get('metrics', {})

        if not metrics:
            return self._default_metrics()

        return {
            'timestamp': metrics.get('timestamp', time.time()),
            'portfolio_value': metrics.get('portfolio_value', 0),
            'p_ruin': metrics.get('p_ruin', 0),
            'var_95': metrics.get('var_95', 0),
            'var_99': metrics.get('var_99', 0),
            'expected_shortfall': metrics.get('expected_shortfall', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'volatility': metrics.get('volatility', 0),
            'beta': metrics.get('beta', 1.0),
            'positions_count': metrics.get('positions_count', 0),
            'cash_available': metrics.get('cash_available', 0),
            'margin_used': metrics.get('margin_used', 0),
            'unrealized_pnl': metrics.get('unrealized_pnl', 0),
            'daily_pnl': metrics.get('daily_pnl', 0),
            'buying_power': metrics.get('buying_power', 0),
            'source': state.get('source', 'unknown')
        }

    def generate_positions(self) -> List[Dict[str, Any]]:
        """
        Generate position data for dashboard.

        Returns the same format as the original RealDataProvider.generate_positions()
        """
        state = self._get_current_state()
        positions = state.get('positions', [])

        if not positions:
            return []

        return [
            {
                'symbol': p.get('symbol', 'UNKNOWN'),
                'quantity': p.get('quantity', 0),
                'market_value': p.get('market_value', 0),
                'entry_price': p.get('entry_price', 0),
                'current_price': p.get('current_price', 0),
                'unrealized_pnl': p.get('unrealized_pnl', 0),
                'unrealized_pnl_percent': p.get('unrealized_pnl_percent', 0),
                'weight': p.get('weight', 0),
                'gate': p.get('gate', 'OTHER'),
                'side': 'long' if p.get('quantity', 0) > 0 else 'short'
            }
            for p in positions
        ]

    async def generate_positions_async(self) -> List[Dict[str, Any]]:
        """Async version of generate_positions."""
        state = await self.get_state_async()
        positions = state.get('positions', [])

        return [
            {
                'symbol': p.get('symbol', 'UNKNOWN'),
                'quantity': p.get('quantity', 0),
                'market_value': p.get('market_value', 0),
                'entry_price': p.get('entry_price', 0),
                'current_price': p.get('current_price', 0),
                'unrealized_pnl': p.get('unrealized_pnl', 0),
                'unrealized_pnl_percent': p.get('unrealized_pnl_percent', 0),
                'weight': p.get('weight', 0),
                'gate': p.get('gate', 'OTHER'),
                'side': 'long' if p.get('quantity', 0) > 0 else 'short'
            }
            for p in positions
        ]

    def generate_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate alerts for dashboard.

        Returns the same format as the original RealDataProvider.generate_alerts()
        """
        state = self._get_current_state()
        alerts = state.get('alerts', [])

        # Add connection status alert if needed
        engine_status = state.get('engine_status', {})
        if not engine_status.get('connected', False):
            alerts.insert(0, {
                'id': 'engine_disconnected',
                'type': 'CONNECTION',
                'severity': 'warning',
                'message': 'Trading engine not connected - displaying cached data',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False
            })

        # Add stale data alert
        if state.get('_is_stale', False):
            alerts.insert(0, {
                'id': 'stale_data',
                'type': 'DATA_QUALITY',
                'severity': 'warning',
                'message': f"Data is stale ({state.get('_age_seconds', 0):.0f}s old)",
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False
            })

        return alerts

    async def generate_alerts_async(self) -> List[Dict[str, Any]]:
        """Async version of generate_alerts."""
        state = await self.get_state_async()
        return state.get('alerts', [])

    def generate_barbell_allocation(self) -> Dict[str, Any]:
        """
        Generate barbell allocation data for dashboard.
        """
        state = self._get_current_state()
        barbell = state.get('barbell', {})

        if not barbell:
            return {
                'safe_allocation': 65,
                'risky_allocation': 35,
                'safe_instruments': ['SPY', 'VTIP', 'IAU'],
                'risky_instruments': ['ULTY', 'AMDY'],
                'source': 'default'
            }

        return barbell

    async def generate_barbell_allocation_async(self) -> Dict[str, Any]:
        """Async version of generate_barbell_allocation."""
        state = await self.get_state_async()
        return state.get('barbell', {
            'safe_allocation': 65,
            'risky_allocation': 35,
            'safe_instruments': ['SPY', 'VTIP', 'IAU'],
            'risky_instruments': ['ULTY', 'AMDY'],
            'source': 'default'
        })

    def generate_engine_status(self) -> Dict[str, Any]:
        """Generate engine status for dashboard."""
        state = self._get_current_state()
        return state.get('engine_status', {
            'connected': False,
            'status': 'unknown',
            'mode': 'paper'
        })

    async def generate_engine_status_async(self) -> Dict[str, Any]:
        """Async version of generate_engine_status."""
        state = await self.get_state_async()
        return state.get('engine_status', {
            'connected': False,
            'status': 'unknown',
            'mode': 'paper'
        })

    def _default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when no data available."""
        return {
            'timestamp': time.time(),
            'portfolio_value': 0,
            'p_ruin': 0,
            'var_95': 0,
            'var_99': 0,
            'expected_shortfall': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'volatility': 0,
            'beta': 1.0,
            'positions_count': 0,
            'cash_available': 0,
            'margin_used': 0,
            'unrealized_pnl': 0,
            'daily_pnl': 0,
            'buying_power': 0,
            'source': 'no_data'
        }

    # Compatibility methods for RealDataProvider interface

    def generate_inequality_data(self) -> Dict[str, Any]:
        """Generate inequality data (placeholder for compatibility)."""
        return {
            'gini_coefficient': 0.45,
            'wealth_concentration': [],
            'flow_direction': 'neutral',
            'source': 'placeholder'
        }

    def generate_contrarian_data(self) -> Dict[str, Any]:
        """Generate contrarian data (placeholder for compatibility)."""
        return {
            'signals': [],
            'dpi_value': 0,
            'recommendation': 'hold',
            'source': 'placeholder'
        }

    def generate_ai_status(self) -> Dict[str, Any]:
        """Generate AI status data."""
        return {
            'hrm_active': False,
            'models_loaded': [],
            'last_prediction': None,
            'source': 'placeholder'
        }


# Factory function for easy instantiation
def create_live_data_provider(trading_engine=None) -> LiveDataProvider:
    """
    Create a LiveDataProvider connected to a trading engine.

    Args:
        trading_engine: Optional TradingEngine instance

    Returns:
        Configured LiveDataProvider
    """
    from src.integration.trading_state_provider import get_state_provider, set_trading_engine

    if trading_engine:
        set_trading_engine(trading_engine)

    provider = get_state_provider()
    return LiveDataProvider(state_provider=provider)
