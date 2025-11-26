"""
ISS-005: TradingStateProvider - Bridge between TradingEngine and Dashboard.

This module provides a clean interface for the dashboard to access real-time
trading engine state without tight coupling to internal implementation.
"""
import asyncio
import logging
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class DashboardMetrics:
    """Real-time risk metrics for dashboard display."""
    timestamp: float
    portfolio_value: float
    cash_available: float
    positions_count: int
    unrealized_pnl: float
    daily_pnl: float
    p_ruin: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    margin_used: float
    buying_power: float


@dataclass
class DashboardPosition:
    """Position data for dashboard display."""
    symbol: str
    quantity: float
    market_value: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    gate: str


@dataclass
class DashboardAlert:
    """Alert data for dashboard display."""
    id: str
    type: str
    severity: str
    message: str
    timestamp: str
    acknowledged: bool = False


class TradingStateProvider:
    """
    Provides dashboard-ready state from TradingEngine.

    This class wraps the TradingEngine and exposes methods that return
    data in the format expected by the dashboard frontend.
    """

    def __init__(self, trading_engine=None):
        """
        Initialize the state provider.

        Args:
            trading_engine: Optional TradingEngine instance. If None, operates in standalone mode.
        """
        self._engine = trading_engine
        self._subscribers: List[Callable] = []
        self._last_state: Dict[str, Any] = {}
        self._state_file = Path("data/dashboard_state.json")
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._update_interval = 1.0  # seconds
        self._running = False
        self._lock = threading.Lock()

        # Alert history
        self._alerts: List[DashboardAlert] = []
        self._alert_id_counter = 0

    def set_engine(self, engine):
        """Set or update the trading engine reference."""
        self._engine = engine
        logger.info("TradingStateProvider connected to TradingEngine")

    @property
    def is_connected(self) -> bool:
        """Check if connected to a trading engine."""
        return self._engine is not None

    async def get_metrics(self) -> DashboardMetrics:
        """
        Get current risk metrics from trading engine.

        Returns:
            DashboardMetrics with current values
        """
        if not self._engine:
            return self._get_default_metrics()

        try:
            # Get engine status
            status = await self._engine.get_status()

            # Get account values from broker
            nav = float(status.get('nav', 0))
            cash = float(status.get('cash', 0))
            positions_count = status.get('positions_count', 0)

            # Get positions for P&L calculation
            positions = await self._get_positions_internal()
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)

            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(positions, nav)

            return DashboardMetrics(
                timestamp=time.time(),
                portfolio_value=nav,
                cash_available=cash,
                positions_count=positions_count,
                unrealized_pnl=unrealized_pnl,
                daily_pnl=risk_metrics.get('daily_pnl', 0),
                p_ruin=risk_metrics.get('p_ruin', 0),
                var_95=risk_metrics.get('var_95', 0),
                var_99=risk_metrics.get('var_99', 0),
                expected_shortfall=risk_metrics.get('expected_shortfall', 0),
                max_drawdown=risk_metrics.get('max_drawdown', 0),
                sharpe_ratio=risk_metrics.get('sharpe_ratio', 0),
                volatility=risk_metrics.get('volatility', 0),
                beta=risk_metrics.get('beta', 1.0),
                margin_used=risk_metrics.get('margin_used', 0),
                buying_power=cash * 2  # 2x margin typical for paper
            )

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return self._get_default_metrics()

    async def get_positions(self) -> List[DashboardPosition]:
        """
        Get current positions from trading engine.

        Returns:
            List of DashboardPosition objects
        """
        return await self._get_positions_internal()

    async def _get_positions_internal(self) -> List[DashboardPosition]:
        """Internal method to get positions."""
        if not self._engine or not self._engine.broker:
            return []

        try:
            broker_positions = await self._engine.broker.get_positions()
            dashboard_positions = []

            for pos in broker_positions:
                qty = float(pos.qty) if hasattr(pos, 'qty') else float(pos.get('qty', 0))
                market_value = float(pos.market_value) if hasattr(pos, 'market_value') else float(pos.get('market_value', 0))
                entry_price = float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') else float(pos.get('avg_entry_price', 0))
                current_price = float(pos.current_price) if hasattr(pos, 'current_price') else market_value / qty if qty else 0
                unrealized_pnl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else float(pos.get('unrealized_pl', 0))
                symbol = pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', 'UNKNOWN')

                # Calculate unrealized P&L percent
                cost_basis = entry_price * qty
                pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

                # Get total portfolio value for weight calculation
                nav = await self._engine.broker.get_account_value()
                weight = (market_value / float(nav) * 100) if nav > 0 else 0

                # Determine gate based on symbol
                gate = self._get_gate_for_symbol(symbol)

                dashboard_positions.append(DashboardPosition(
                    symbol=symbol,
                    quantity=qty,
                    market_value=market_value,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=pnl_percent,
                    weight=weight,
                    gate=gate
                ))

            return dashboard_positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def _get_gate_for_symbol(self, symbol: str) -> str:
        """Determine which gate a symbol belongs to based on config."""
        if not self._engine or not hasattr(self._engine, 'config'):
            # Default mapping
            gate_map = {
                'SPY': 'SAFE_HEDGE',
                'VTIP': 'SAFE_HEDGE',
                'IAU': 'SAFE_HEDGE',
                'ULTY': 'MOMENTUM',
                'AMDY': 'MOMENTUM'
            }
            return gate_map.get(symbol, 'OTHER')

        # Use config-based mapping
        asset_universe = self._engine.config.get('asset_universe', {})
        safe_assets = asset_universe.get('safe_assets', ['SPY', 'VTIP', 'IAU'])
        momentum_assets = asset_universe.get('momentum_assets', ['ULTY', 'AMDY'])

        if symbol in safe_assets:
            return 'SAFE_HEDGE'
        elif symbol in momentum_assets:
            return 'MOMENTUM'
        else:
            return 'OTHER'

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get current alerts from safety systems.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        if self._engine and self._engine.safety_integration:
            try:
                health = self._engine.safety_integration.get_system_health()

                # Check for critical conditions
                if health.get('overall_status') == 'CRITICAL':
                    alerts.append(self._create_alert(
                        'SAFETY_CRITICAL',
                        'critical',
                        'Safety system reports CRITICAL status'
                    ))

                # Check circuit breakers
                circuit_breakers = health.get('safety_systems', {}).get('circuit_breakers', {})
                for breaker, state in circuit_breakers.items():
                    if state == 'OPEN':
                        alerts.append(self._create_alert(
                            'CIRCUIT_BREAKER',
                            'warning',
                            f'Circuit breaker {breaker} is OPEN'
                        ))

            except Exception as e:
                logger.error(f"Error getting safety alerts: {e}")

        # Add kill switch alert if activated
        if self._engine and self._engine.kill_switch_activated:
            alerts.append(self._create_alert(
                'KILL_SWITCH',
                'critical',
                'KILL SWITCH ACTIVATED - Trading halted'
            ))

        # Add stored alerts
        alerts.extend([asdict(a) for a in self._alerts[-50:]])  # Last 50 alerts

        return alerts

    def _create_alert(self, alert_type: str, severity: str, message: str) -> Dict[str, Any]:
        """Create an alert dictionary."""
        self._alert_id_counter += 1
        return {
            'id': f'alert_{self._alert_id_counter}',
            'type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }

    def add_alert(self, alert_type: str, severity: str, message: str):
        """Add a new alert to the system."""
        alert = DashboardAlert(
            id=f'alert_{self._alert_id_counter}',
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        self._alert_id_counter += 1
        self._alerts.append(alert)

        # Trim old alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    async def get_engine_status(self) -> Dict[str, Any]:
        """
        Get trading engine status.

        Returns:
            Status dictionary with engine health info
        """
        if not self._engine:
            return {
                'connected': False,
                'status': 'disconnected',
                'mode': 'unknown',
                'broker_connected': False,
                'safety_enabled': False
            }

        try:
            status = await self._engine.get_status()
            return {
                'connected': True,
                'status': status.get('status', 'unknown'),
                'mode': status.get('mode', 'paper'),
                'broker_connected': status.get('broker_connected', False),
                'market_open': status.get('market_open', False),
                'kill_switch': status.get('kill_switch', False),
                'safety_enabled': status.get('safety', {}).get('enabled', False),
                'safety_running': status.get('safety', {}).get('running', False),
                'nav': float(status.get('nav', 0)),
                'cash': float(status.get('cash', 0)),
                'positions_count': status.get('positions_count', 0)
            }
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return {
                'connected': True,
                'status': 'error',
                'error': str(e)
            }

    async def get_barbell_allocation(self) -> Dict[str, Any]:
        """
        Get barbell allocation from AntifragilityEngine.

        Returns:
            Barbell allocation data
        """
        if not self._engine or not self._engine.antifragility_engine:
            return {
                'safe_allocation': 65,
                'risky_allocation': 35,
                'safe_instruments': ['SPY', 'VTIP', 'IAU'],
                'risky_instruments': ['ULTY', 'AMDY'],
                'source': 'default'
            }

        try:
            nav = await self._engine.broker.get_account_value()
            barbell = self._engine.antifragility_engine.calculate_barbell_allocation(float(nav))

            return {
                'safe_allocation': barbell['safe_percentage'],
                'risky_allocation': barbell['risky_percentage'],
                'safe_amount': barbell['safe_amount'],
                'risky_amount': barbell['risky_amount'],
                'safe_instruments': barbell['safe_instruments'],
                'risky_instruments': barbell['risky_instruments'],
                'total_allocated': barbell['total_allocated'],
                'source': 'antifragility_engine'
            }
        except Exception as e:
            logger.error(f"Error getting barbell allocation: {e}")
            return {
                'safe_allocation': 65,
                'risky_allocation': 35,
                'error': str(e),
                'source': 'error_fallback'
            }

    async def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete dashboard state in one call.

        Returns:
            Complete state dictionary
        """
        metrics = await self.get_metrics()
        positions = await self.get_positions()
        alerts = await self.get_alerts()
        status = await self.get_engine_status()
        barbell = await self.get_barbell_allocation()

        return {
            'metrics': asdict(metrics),
            'positions': [asdict(p) for p in positions],
            'alerts': alerts,
            'engine_status': status,
            'barbell': barbell,
            'timestamp': datetime.now().isoformat(),
            'source': 'live' if self._engine else 'disconnected'
        }

    async def _calculate_risk_metrics(self, positions: List[DashboardPosition],
                                      nav: float) -> Dict[str, float]:
        """Calculate risk metrics from positions."""
        if not positions or nav <= 0:
            return {
                'daily_pnl': 0,
                'p_ruin': 0,
                'var_95': 0,
                'var_99': 0,
                'expected_shortfall': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'beta': 1.0,
                'margin_used': 0
            }

        try:
            # Calculate basic metrics
            total_pnl = sum(p.unrealized_pnl for p in positions)
            daily_pnl_pct = (total_pnl / nav) * 100 if nav > 0 else 0

            # Estimate volatility from position weights
            # This is simplified - real implementation would use historical returns
            weights = [p.weight / 100 for p in positions]
            avg_volatility = 0.20  # 20% annualized as baseline

            # Calculate VaR (simplified parametric)
            var_95 = nav * avg_volatility * 1.645 / (252 ** 0.5)  # Daily 95% VaR
            var_99 = nav * avg_volatility * 2.326 / (252 ** 0.5)  # Daily 99% VaR

            # Expected shortfall (simplified)
            expected_shortfall = var_99 * 1.2

            # P(ruin) based on Kelly criterion approximation
            # With proper sizing, P(ruin) should be low
            p_ruin = max(0, min(0.05, abs(daily_pnl_pct) / 100))

            return {
                'daily_pnl': total_pnl,
                'p_ruin': p_ruin,
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': expected_shortfall,
                'max_drawdown': abs(min(0, total_pnl)),
                'sharpe_ratio': 1.5,  # Would need historical data
                'volatility': avg_volatility,
                'beta': 1.0,
                'margin_used': 0
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'daily_pnl': 0,
                'p_ruin': 0,
                'var_95': 0,
                'var_99': 0,
                'expected_shortfall': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'beta': 1.0,
                'margin_used': 0
            }

    def _get_default_metrics(self) -> DashboardMetrics:
        """Return default metrics when engine not connected."""
        return DashboardMetrics(
            timestamp=time.time(),
            portfolio_value=0,
            cash_available=0,
            positions_count=0,
            unrealized_pnl=0,
            daily_pnl=0,
            p_ruin=0,
            var_95=0,
            var_99=0,
            expected_shortfall=0,
            max_drawdown=0,
            sharpe_ratio=0,
            volatility=0,
            beta=1.0,
            margin_used=0,
            buying_power=0
        )

    # State persistence for inter-process communication

    def publish_state(self, state: Dict[str, Any]) -> bool:
        """
        Publish state to file for dashboard to read.
        Used when engine and dashboard run in separate processes.
        """
        try:
            with self._lock:
                state['_timestamp'] = datetime.now().isoformat()
                state['_epoch'] = time.time()

                temp_file = self._state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state, f, cls=DecimalEncoder, indent=2)

                temp_file.replace(self._state_file)
                self._last_state = state
                return True

        except Exception as e:
            logger.error(f"Failed to publish state: {e}")
            return False

    def read_published_state(self) -> Optional[Dict[str, Any]]:
        """
        Read state from file published by engine.
        Used by dashboard when running in separate process.
        """
        try:
            if not self._state_file.exists():
                return None

            with open(self._state_file, 'r') as f:
                state = json.load(f)

            # Check staleness
            epoch = state.get('_epoch', 0)
            age = time.time() - epoch
            state['_age_seconds'] = age
            state['_is_stale'] = age > 60  # Stale after 60 seconds

            return state

        except Exception as e:
            logger.error(f"Failed to read state: {e}")
            return None

    async def start_publishing(self, interval: float = 1.0):
        """Start periodic state publishing."""
        self._running = True
        self._update_interval = interval

        while self._running:
            try:
                state = await self.get_full_state()
                self.publish_state(state)
            except Exception as e:
                logger.error(f"Error in state publishing: {e}")

            await asyncio.sleep(self._update_interval)

    def stop_publishing(self):
        """Stop periodic state publishing."""
        self._running = False


# Global instance
_provider: Optional[TradingStateProvider] = None


def get_state_provider() -> TradingStateProvider:
    """Get the global state provider instance."""
    global _provider
    if _provider is None:
        _provider = TradingStateProvider()
    return _provider


def set_trading_engine(engine):
    """Connect the global state provider to a trading engine."""
    provider = get_state_provider()
    provider.set_engine(engine)
