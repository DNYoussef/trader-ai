"""
Real-time WebSocket server for Gary×Taleb trading dashboard.
Provides live risk metrics, P(ruin) calculations, and position updates.
"""

import asyncio
import json
import logging
import time
import redis
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Real-time risk metrics data structure."""
    timestamp: float
    portfolio_value: float
    p_ruin: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    positions_count: int
    cash_available: float
    margin_used: float
    unrealized_pnl: float
    daily_pnl: float

@dataclass
class PositionUpdate:
    """Real-time position update data structure."""
    timestamp: float
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    entry_price: float
    current_price: float
    weight: float
    last_update: float

@dataclass
class AlertEvent:
    """Risk alert event data structure."""
    timestamp: float
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    acknowledged: bool = False

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            'connected_at': time.time(),
            'last_ping': time.time(),
            'subscriptions': set()
        }
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: Dict, subscription_type: Optional[str] = None):
        """Broadcast message to all connected clients or filtered by subscription."""
        disconnected_clients = []

        for client_id, websocket in self.active_connections.items():
            try:
                # Filter by subscription if specified
                if subscription_type:
                    client_subs = self.connection_metadata[client_id]['subscriptions']
                    if subscription_type not in client_subs:
                        continue

                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def add_subscription(self, client_id: str, subscription_type: str):
        """Add subscription for client."""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]['subscriptions'].add(subscription_type)

    def remove_subscription(self, client_id: str, subscription_type: str):
        """Remove subscription for client."""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]['subscriptions'].discard(subscription_type)

class RiskCalculator:
    """Real-time risk metrics calculator for Gary×Taleb system."""

    def __init__(self):
        self.lookback_period = 252  # Trading days for volatility calculation
        self.confidence_levels = [0.95, 0.99]

    def calculate_p_ruin(self,
                        current_capital: float,
                        historical_returns: List[float],
                        target_return: float = 0.15) -> float:
        """
        Calculate probability of ruin using Gary's DPI methodology.

        P(ruin) = probability that capital falls below critical threshold
        Uses Monte Carlo simulation with historical return distribution.
        """
        if not historical_returns or len(historical_returns) < 30:
            return 0.5  # Default conservative estimate

        returns_array = np.array(historical_returns)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)

        # Monte Carlo simulation for ruin probability
        n_simulations = 10000
        n_periods = 252  # One year simulation
        ruin_threshold = current_capital * 0.2  # 80% drawdown threshold

        ruin_count = 0
        for _ in range(n_simulations):
            capital = current_capital
            for _ in range(n_periods):
                daily_return = np.random.normal(mean_return, volatility)
                capital *= (1 + daily_return)
                if capital <= ruin_threshold:
                    ruin_count += 1
                    break

        p_ruin = ruin_count / n_simulations
        return min(max(p_ruin, 0.0), 1.0)  # Clamp between 0 and 1

    def calculate_var(self,
                     portfolio_value: float,
                     historical_returns: List[float],
                     confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) at specified confidence level."""
        if not historical_returns or len(historical_returns) < 30:
            return portfolio_value * 0.05  # Default 5% VaR

        returns_array = np.array(historical_returns)
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns_array, var_percentile)

        return abs(portfolio_value * var_return)

    def calculate_expected_shortfall(self,
                                   portfolio_value: float,
                                   historical_returns: List[float],
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if not historical_returns or len(historical_returns) < 30:
            return portfolio_value * 0.08  # Default 8% ES

        returns_array = np.array(historical_returns)
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns_array, var_percentile)

        tail_returns = returns_array[returns_array <= var_threshold]
        if len(tail_returns) == 0:
            return portfolio_value * 0.08

        expected_shortfall = abs(portfolio_value * np.mean(tail_returns))
        return expected_shortfall

    def calculate_sharpe_ratio(self,
                              historical_returns: List[float],
                              risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if not historical_returns or len(historical_returns) < 30:
            return 0.0

        returns_array = np.array(historical_returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe

class RiskDashboardServer:
    """Main WebSocket server for real-time risk dashboard."""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.app = FastAPI(title="Risk Dashboard API", version="1.0.0")
        self.connection_manager = ConnectionManager()
        self.risk_calculator = RiskCalculator()

        # Redis for real-time data pub/sub
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None

        # In-memory storage as fallback
        self.latest_metrics: Optional[RiskMetrics] = None
        self.positions: Dict[str, PositionUpdate] = {}
        self.alerts: List[AlertEvent] = []
        self.historical_data: Dict[str, List] = {
            'returns': [],
            'portfolio_values': [],
            'timestamps': []
        }

        # Alert thresholds
        self.alert_thresholds = {
            'p_ruin': {'high': 0.1, 'critical': 0.2},
            'var_95': {'high': 0.05, 'critical': 0.1},
            'max_drawdown': {'high': 0.1, 'critical': 0.2},
            'margin_used': {'high': 0.8, 'critical': 0.9}
        }

        self._setup_routes()
        self._setup_cors()

        # Background tasks
        self.update_task = None
        self.running = False

    def _setup_cors(self):
        """Setup CORS middleware for frontend access."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes and WebSocket endpoints."""

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.connection_manager.connect(websocket, client_id)
            try:
                # Send initial data
                if self.latest_metrics:
                    await self.connection_manager.send_personal_message(
                        {
                            'type': 'risk_metrics',
                            'data': asdict(self.latest_metrics)
                        },
                        client_id
                    )

                # Send current positions
                for position in self.positions.values():
                    await self.connection_manager.send_personal_message(
                        {
                            'type': 'position_update',
                            'data': asdict(position)
                        },
                        client_id
                    )

                # Handle incoming messages
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_client_message(message, client_id)

            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
                self.connection_manager.disconnect(client_id)

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {
                'status': 'healthy',
                'timestamp': time.time(),
                'connections': len(self.connection_manager.active_connections)
            }

        @self.app.get("/api/metrics/current")
        async def get_current_metrics():
            """Get current risk metrics."""
            if self.latest_metrics:
                return asdict(self.latest_metrics)
            return {'error': 'No metrics available'}

        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions."""
            return [asdict(pos) for pos in self.positions.values()]

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get current alerts."""
            return [asdict(alert) for alert in self.alerts[-100:]]  # Last 100 alerts

        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert."""
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True

                    # Broadcast acknowledgment
                    await self.connection_manager.broadcast({
                        'type': 'alert_acknowledged',
                        'data': {'alert_id': alert_id}
                    })
                    return {'status': 'acknowledged'}

            return {'error': 'Alert not found'}

    async def _handle_client_message(self, message: Dict, client_id: str):
        """Handle incoming WebSocket messages from clients."""
        msg_type = message.get('type')

        if msg_type == 'subscribe':
            subscription_type = message.get('subscription')
            if subscription_type:
                self.connection_manager.add_subscription(client_id, subscription_type)
                logger.info(f"Client {client_id} subscribed to {subscription_type}")

        elif msg_type == 'unsubscribe':
            subscription_type = message.get('subscription')
            if subscription_type:
                self.connection_manager.remove_subscription(client_id, subscription_type)
                logger.info(f"Client {client_id} unsubscribed from {subscription_type}")

        elif msg_type == 'ping':
            # Update last ping time
            if client_id in self.connection_manager.connection_metadata:
                self.connection_manager.connection_metadata[client_id]['last_ping'] = time.time()

            # Send pong response
            await self.connection_manager.send_personal_message({
                'type': 'pong',
                'timestamp': time.time()
            }, client_id)

    def update_risk_metrics(self, metrics_data: Dict):
        """Update risk metrics and broadcast to clients."""
        try:
            # Create RiskMetrics object
            metrics = RiskMetrics(
                timestamp=time.time(),
                portfolio_value=metrics_data.get('portfolio_value', 0.0),
                p_ruin=self.risk_calculator.calculate_p_ruin(
                    metrics_data.get('portfolio_value', 0.0),
                    self.historical_data['returns']
                ),
                var_95=self.risk_calculator.calculate_var(
                    metrics_data.get('portfolio_value', 0.0),
                    self.historical_data['returns'],
                    0.95
                ),
                var_99=self.risk_calculator.calculate_var(
                    metrics_data.get('portfolio_value', 0.0),
                    self.historical_data['returns'],
                    0.99
                ),
                expected_shortfall=self.risk_calculator.calculate_expected_shortfall(
                    metrics_data.get('portfolio_value', 0.0),
                    self.historical_data['returns']
                ),
                max_drawdown=metrics_data.get('max_drawdown', 0.0),
                sharpe_ratio=self.risk_calculator.calculate_sharpe_ratio(
                    self.historical_data['returns']
                ),
                volatility=metrics_data.get('volatility', 0.0),
                beta=metrics_data.get('beta', 1.0),
                positions_count=metrics_data.get('positions_count', 0),
                cash_available=metrics_data.get('cash_available', 0.0),
                margin_used=metrics_data.get('margin_used', 0.0),
                unrealized_pnl=metrics_data.get('unrealized_pnl', 0.0),
                daily_pnl=metrics_data.get('daily_pnl', 0.0)
            )

            self.latest_metrics = metrics

            # Store historical data
            self.historical_data['portfolio_values'].append(metrics.portfolio_value)
            self.historical_data['timestamps'].append(metrics.timestamp)

            # Keep only last 1000 data points
            if len(self.historical_data['portfolio_values']) > 1000:
                self.historical_data['portfolio_values'] = self.historical_data['portfolio_values'][-1000:]
                self.historical_data['timestamps'] = self.historical_data['timestamps'][-1000:]

            # Check for alerts
            self._check_alerts(metrics)

            # Broadcast to clients
            asyncio.create_task(self.connection_manager.broadcast({
                'type': 'risk_metrics',
                'data': asdict(metrics)
            }, 'risk_metrics'))

            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.set('latest_risk_metrics', json.dumps(asdict(metrics)))
                except Exception as e:
                    logger.error(f"Redis storage error: {e}")

        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    def update_position(self, symbol: str, position_data: Dict):
        """Update position data and broadcast to clients."""
        try:
            position = PositionUpdate(
                timestamp=time.time(),
                symbol=symbol,
                quantity=position_data.get('quantity', 0.0),
                market_value=position_data.get('market_value', 0.0),
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                entry_price=position_data.get('entry_price', 0.0),
                current_price=position_data.get('current_price', 0.0),
                weight=position_data.get('weight', 0.0),
                last_update=time.time()
            )

            self.positions[symbol] = position

            # Broadcast to clients
            asyncio.create_task(self.connection_manager.broadcast({
                'type': 'position_update',
                'data': asdict(position)
            }, 'positions'))

        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")

    def _check_alerts(self, metrics: RiskMetrics):
        """Check metrics against alert thresholds."""
        time.time()

        # P(ruin) alerts
        if metrics.p_ruin >= self.alert_thresholds['p_ruin']['critical']:
            self._create_alert('p_ruin', 'critical',
                             f"Critical P(ruin) level: {metrics.p_ruin:.1%}",
                             metrics.p_ruin, self.alert_thresholds['p_ruin']['critical'])
        elif metrics.p_ruin >= self.alert_thresholds['p_ruin']['high']:
            self._create_alert('p_ruin', 'high',
                             f"High P(ruin) level: {metrics.p_ruin:.1%}",
                             metrics.p_ruin, self.alert_thresholds['p_ruin']['high'])

        # VaR alerts
        var_ratio = metrics.var_95 / metrics.portfolio_value if metrics.portfolio_value > 0 else 0
        if var_ratio >= self.alert_thresholds['var_95']['critical']:
            self._create_alert('var_95', 'critical',
                             f"Critical VaR level: {var_ratio:.1%}",
                             var_ratio, self.alert_thresholds['var_95']['critical'])
        elif var_ratio >= self.alert_thresholds['var_95']['high']:
            self._create_alert('var_95', 'high',
                             f"High VaR level: {var_ratio:.1%}",
                             var_ratio, self.alert_thresholds['var_95']['high'])

        # Max drawdown alerts
        if metrics.max_drawdown >= self.alert_thresholds['max_drawdown']['critical']:
            self._create_alert('max_drawdown', 'critical',
                             f"Critical drawdown: {metrics.max_drawdown:.1%}",
                             metrics.max_drawdown, self.alert_thresholds['max_drawdown']['critical'])
        elif metrics.max_drawdown >= self.alert_thresholds['max_drawdown']['high']:
            self._create_alert('max_drawdown', 'high',
                             f"High drawdown: {metrics.max_drawdown:.1%}",
                             metrics.max_drawdown, self.alert_thresholds['max_drawdown']['high'])

    def _create_alert(self, metric_name: str, severity: str, message: str,
                     current_value: float, threshold_value: float):
        """Create and broadcast new alert."""
        alert = AlertEvent(
            timestamp=time.time(),
            alert_id=f"{metric_name}_{severity}_{int(time.time())}",
            alert_type='threshold',
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value
        )

        self.alerts.append(alert)

        # Broadcast alert
        asyncio.create_task(self.connection_manager.broadcast({
            'type': 'alert',
            'data': asdict(alert)
        }, 'alerts'))

        logger.warning(f"Alert created: {message}")

    async def start_background_tasks(self):
        """Start background update tasks."""
        self.running = True
        self.update_task = asyncio.create_task(self._background_update_loop())
        logger.info("Background tasks started")

    async def stop_background_tasks(self):
        """Stop background update tasks."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        logger.info("Background tasks stopped")

    async def _background_update_loop(self):
        """Background loop for periodic updates."""
        while self.running:
            try:
                # Clean up old alerts (keep last 24 hours)
                cutoff_time = time.time() - (24 * 3600)
                self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

                # Send heartbeat to maintain connections
                await self.connection_manager.broadcast({
                    'type': 'heartbeat',
                    'timestamp': time.time()
                })

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")
                await asyncio.sleep(5)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the WebSocket server."""
        logger.info(f"Starting Risk Dashboard Server on {host}:{port}")

        # Start background tasks before running server
        asyncio.create_task(self.start_background_tasks())

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run server
    server = RiskDashboardServer()

    # Example: simulate some data updates
    import threading
    import random

    def simulate_data():
        """Simulate trading data for testing."""
        time.sleep(2)  # Wait for server to start

        portfolio_value = 10000.0
        positions = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

        while True:
            try:
                # Simulate portfolio changes
                daily_return = random.gauss(0.001, 0.02)  # 0.1% mean, 2% volatility
                portfolio_value *= (1 + daily_return)

                # Update risk metrics
                server.update_risk_metrics({
                    'portfolio_value': portfolio_value,
                    'max_drawdown': random.uniform(0.0, 0.15),
                    'volatility': random.uniform(0.15, 0.25),
                    'beta': random.uniform(0.8, 1.2),
                    'positions_count': len(positions),
                    'cash_available': portfolio_value * 0.1,
                    'margin_used': random.uniform(0.0, 0.5),
                    'unrealized_pnl': random.gauss(0, portfolio_value * 0.02),
                    'daily_pnl': portfolio_value * daily_return
                })

                # Update positions
                for symbol in positions:
                    server.update_position(symbol, {
                        'quantity': random.randint(10, 100),
                        'market_value': portfolio_value / len(positions),
                        'unrealized_pnl': random.gauss(0, 500),
                        'entry_price': random.uniform(100, 300),
                        'current_price': random.uniform(100, 300),
                        'weight': 1.0 / len(positions)
                    })

                # Add return to historical data
                server.historical_data['returns'].append(daily_return)
                if len(server.historical_data['returns']) > 252:
                    server.historical_data['returns'] = server.historical_data['returns'][-252:]

                time.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Simulation error: {e}")
                time.sleep(1)

    # Start simulation in background thread for testing
    threading.Thread(target=simulate_data, daemon=True).start()

    # Run server
    server.run()