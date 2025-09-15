#!/usr/bin/env python3
"""
Simplified startup script for the Gary×Taleb Risk Dashboard Server.
This version runs without Redis dependency for development.
"""

import sys
import os
import asyncio
import logging
import json
import random
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict

# Add the parent directory to the path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import AI dashboard integration
try:
    # Add current directory to path for direct imports
    sys.path.insert(0, str(Path(__file__).parent))
    import ai_dashboard_integration

    # Import functions
    ai_dashboard_integrator = ai_dashboard_integration.ai_dashboard_integrator
    get_dashboard_inequality_data = getattr(ai_dashboard_integration, 'get_dashboard_inequality_data', None)
    get_dashboard_contrarian_data = getattr(ai_dashboard_integration, 'get_dashboard_contrarian_data', None)
    get_ai_status_data = getattr(ai_dashboard_integration, 'get_ai_status_data', None)
    execute_trade = getattr(ai_dashboard_integration, 'execute_trade', None)

    AI_AVAILABLE = True
    logging.info("AI dashboard integration loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    logging.warning(f"AI dashboard integration not available - running in mock mode: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    entry_price: float
    current_price: float
    weight: float
    last_updated: float

class SimpleDashboardServer:
    """Simplified dashboard server without Redis dependency."""

    def __init__(self):
        self.app = FastAPI(title="Gary×Taleb Risk Dashboard")
        self.active_connections: Set[WebSocket] = set()
        self.setup_cors()
        self.setup_routes()
        self.mock_data_generator = MockDataGenerator()

    def setup_cors(self):
        """Configure CORS for frontend access."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            return {"message": "Gary×Taleb Risk Dashboard API", "status": "running", "ai_enabled": AI_AVAILABLE}

        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "connections": len(self.active_connections),
                "ai_available": AI_AVAILABLE
            }

        @self.app.get("/api/metrics/current")
        async def get_current_metrics():
            """Get current risk metrics."""
            return self.mock_data_generator.generate_metrics()

        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions."""
            return self.mock_data_generator.generate_positions()

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get active alerts."""
            return self.mock_data_generator.generate_alerts()

        # AI-Enhanced endpoints
        @self.app.get("/api/inequality/data")
        async def get_inequality_data():
            """Get AI-enhanced inequality panel data."""
            if AI_AVAILABLE:
                return await get_dashboard_inequality_data()
            else:
                return self.mock_data_generator.generate_inequality_data()

        @self.app.get("/api/contrarian/opportunities")
        async def get_contrarian_opportunities():
            """Get AI-detected contrarian opportunities."""
            if AI_AVAILABLE:
                return await get_dashboard_contrarian_data()
            else:
                return self.mock_data_generator.generate_contrarian_data()

        @self.app.get("/api/ai/status")
        async def get_ai_status():
            """Get AI calibration and status data."""
            if AI_AVAILABLE:
                return await get_ai_status_data()
            else:
                return self.mock_data_generator.generate_ai_status()

        @self.app.post("/api/trade/execute/{asset}")
        async def execute_ai_trade(asset: str):
            """Execute AI-recommended trade."""
            if AI_AVAILABLE:
                return await execute_trade(asset)
            else:
                return {"success": False, "error": "AI not available in mock mode"}

        @self.app.get("/api/barbell/allocation")
        async def get_barbell_allocation():
            """Get current barbell allocation status."""
            if AI_AVAILABLE:
                # This would come from AI mispricing detector
                return {"safe_allocation": 0.8, "risky_allocation": 0.2, "ai_managed": True}
            else:
                return self.mock_data_generator.generate_barbell_allocation()

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.connect(websocket)

            # Connect to AI dashboard integrator if available
            if AI_AVAILABLE:
                ai_dashboard_integrator.add_websocket_connection(websocket)

            try:
                # Send initial data
                await self.send_initial_data(websocket)

                # Start sending updates
                update_task = asyncio.create_task(self.send_periodic_updates(websocket))

                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        await self.handle_client_message(websocket, message)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

                update_task.cancel()

            except WebSocketDisconnect:
                pass
            finally:
                if AI_AVAILABLE:
                    ai_dashboard_integrator.remove_websocket_connection(websocket)
                self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_initial_data(self, websocket: WebSocket):
        """Send initial data to newly connected client."""
        metrics = self.mock_data_generator.generate_metrics()
        positions = self.mock_data_generator.generate_positions()

        await websocket.send_json({
            "type": "initial_data",
            "data": {
                "metrics": metrics,
                "positions": positions
            },
            "timestamp": time.time()
        })

    async def send_periodic_updates(self, websocket: WebSocket):
        """Send periodic updates to connected client."""
        while True:
            try:
                await asyncio.sleep(1)  # Update every second

                # Generate and send risk metrics
                metrics = self.mock_data_generator.generate_metrics()
                await websocket.send_json({
                    "type": "risk_metrics",
                    "data": metrics,
                    "timestamp": time.time()
                })

                # Occasionally send position updates
                if random.random() < 0.3:  # 30% chance
                    positions = self.mock_data_generator.generate_positions()
                    await websocket.send_json({
                        "type": "position_update",
                        "data": positions,
                        "timestamp": time.time()
                    })

                # Occasionally send alerts
                if random.random() < 0.1:  # 10% chance
                    alert = self.mock_data_generator.generate_alert()
                    await websocket.send_json({
                        "type": "alert",
                        "data": alert,
                        "timestamp": time.time()
                    })

            except Exception as e:
                logger.error(f"Error sending update: {e}")
                break

    async def handle_client_message(self, websocket: WebSocket, message: Dict):
        """Handle incoming messages from client."""
        msg_type = message.get('type')

        if msg_type == 'ping':
            await websocket.send_json({"type": "pong", "timestamp": time.time()})
        elif msg_type == 'subscribe':
            # Handle subscription requests
            subscription = message.get('subscription')
            logger.info(f"Client subscribed to: {subscription}")
        elif msg_type == 'unsubscribe':
            # Handle unsubscribe requests
            subscription = message.get('subscription')
            logger.info(f"Client unsubscribed from: {subscription}")

    def run(self, host="0.0.0.0", port=8000):
        """Run the server."""
        logger.info(f"Starting dashboard server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

class MockDataGenerator:
    """Generate mock data for development."""

    def __init__(self):
        self.portfolio_value = 10000
        self.positions = self._initialize_positions()

    def _initialize_positions(self):
        """Initialize mock positions."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ", "VTI"]
        positions = []

        for symbol in symbols[:5]:  # Start with 5 positions
            positions.append({
                "symbol": symbol,
                "quantity": random.randint(10, 100),
                "entry_price": random.uniform(100, 500),
                "current_price": random.uniform(100, 500)
            })

        return positions

    def generate_metrics(self) -> Dict:
        """Generate mock risk metrics."""
        # Add some random walk to portfolio value
        self.portfolio_value *= (1 + random.uniform(-0.02, 0.02))

        return {
            "portfolio_value": round(self.portfolio_value, 2),
            "p_ruin": round(random.uniform(0.01, 0.15), 4),
            "var_95": round(self.portfolio_value * random.uniform(0.02, 0.05), 2),
            "var_99": round(self.portfolio_value * random.uniform(0.03, 0.08), 2),
            "expected_shortfall": round(self.portfolio_value * random.uniform(0.04, 0.10), 2),
            "max_drawdown": round(random.uniform(0.05, 0.25), 4),
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
            "volatility": round(random.uniform(0.1, 0.3), 4),
            "beta": round(random.uniform(0.8, 1.2), 2),
            "positions_count": len(self.positions),
            "cash_available": round(self.portfolio_value * 0.3, 2),
            "margin_used": round(self.portfolio_value * 0.7, 2),
            "unrealized_pnl": round(random.uniform(-500, 1000), 2),
            "daily_pnl": round(random.uniform(-200, 300), 2)
        }

    def generate_positions(self) -> List[Dict]:
        """Generate mock positions."""
        positions = []

        for pos in self.positions:
            # Update current price with random walk
            pos["current_price"] *= (1 + random.uniform(-0.03, 0.03))

            market_value = pos["quantity"] * pos["current_price"]
            unrealized_pnl = (pos["current_price"] - pos["entry_price"]) * pos["quantity"]

            positions.append({
                "symbol": pos["symbol"],
                "quantity": pos["quantity"],
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "entry_price": round(pos["entry_price"], 2),
                "current_price": round(pos["current_price"], 2),
                "weight": round(market_value / self.portfolio_value, 4),
                "last_updated": time.time()
            })

        return positions

    def generate_alerts(self) -> List[Dict]:
        """Generate mock alerts."""
        alerts = []

        # Generate some sample alerts
        if random.random() < 0.3:
            alerts.append({
                "id": f"alert_{int(time.time())}",
                "severity": random.choice(["low", "medium", "high", "critical"]),
                "type": "risk_threshold",
                "message": "P(ruin) approaching threshold",
                "value": round(random.uniform(0.08, 0.12), 4),
                "threshold": 0.10,
                "timestamp": time.time()
            })

        return alerts

    def generate_alert(self) -> Dict:
        """Generate a single alert."""
        alert_types = [
            ("P(ruin) exceeding threshold", "p_ruin"),
            ("Maximum drawdown alert", "drawdown"),
            ("Margin usage high", "margin"),
            ("Volatility spike detected", "volatility"),
            ("Gary Moment detected", "gary_moment"),
            ("AI confidence dropped", "ai_confidence"),
            ("Inequality signal spike", "inequality")
        ]

        msg, alert_type = random.choice(alert_types)

        return {
            "id": f"alert_{int(time.time() * 1000)}",
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "type": alert_type,
            "message": msg,
            "timestamp": time.time()
        }

    def generate_inequality_data(self) -> Dict:
        """Generate mock inequality panel data."""
        return {
            'metrics': {
                'giniCoefficient': 0.475 + np.random.normal(0, 0.005),
                'top1PercentWealth': 32.0 + np.random.normal(0, 0.5),
                'top10PercentWealth': 58.0 + np.random.normal(0, 1.0),
                'wageGrowthReal': -0.5 + np.random.normal(0, 0.2),
                'corporateProfitsToGdp': 12.5 + np.random.normal(0, 0.5),
                'householdDebtToIncome': 105.0 + np.random.normal(0, 2.0),
                'luxuryVsDiscountSpend': 1.8 + np.random.normal(0, 0.1),
                'wealthVelocity': 0.18 + np.random.normal(0, 0.02),
                'consensusWrongScore': 0.7 + np.random.normal(0, 0.05),
                'ai_confidence_level': 0.75 + np.random.normal(0, 0.05),
                'mathematical_signal_strength': abs(np.random.normal(0, 0.3)),
                'ai_prediction_accuracy': 0.65 + np.random.normal(0, 0.1)
            },
            'historicalData': self._generate_historical_inequality(),
            'wealthFlows': self._generate_wealth_flows(),
            'contrarianSignals': self._generate_contrarian_signals()
        }

    def _generate_historical_inequality(self) -> List[Dict]:
        """Generate historical inequality data."""
        data = []
        base_date = datetime.now()

        for i in range(90):
            date = base_date - timedelta(days=89-i)
            trend = i / 90.0

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'gini': 0.47 + trend * 0.008 + np.random.normal(0, 0.002),
                'top1': 31.0 + trend * 2.0 + np.random.normal(0, 0.2),
                'wageGrowth': -0.2 - trend * 0.6 + np.random.normal(0, 0.1)
            })

        return data

    def _generate_wealth_flows(self) -> List[Dict]:
        """Generate wealth flow data."""
        return [
            {'source': 'Working Class Wages', 'target': 'Corporate Profits', 'value': 25.0 + np.random.normal(0, 3), 'color': '#ef4444'},
            {'source': 'Middle Class Savings', 'target': 'Asset Prices', 'value': 35.0 + np.random.normal(0, 4), 'color': '#f59e0b'},
            {'source': 'Government Debt', 'target': 'Bond Holders', 'value': 18.0 + np.random.normal(0, 2), 'color': '#8b5cf6'},
            {'source': 'Rent Payments', 'target': 'Property Owners', 'value': 22.0 + np.random.normal(0, 3), 'color': '#10b981'}
        ]

    def _generate_contrarian_signals(self) -> List[Dict]:
        """Generate contrarian signals."""
        signals = [
            {'topic': 'Housing Market', 'consensusView': 'Rates will crash housing', 'realityView': 'Cash buyers support prices', 'conviction': 0.8, 'opportunity': 'Long REITs'},
            {'topic': 'Tech Stocks', 'consensusView': 'Overvalued growth', 'realityView': 'Wealth concentration flows to tech', 'conviction': 0.7, 'opportunity': 'Long QQQ'},
            {'topic': 'Treasury Bonds', 'consensusView': 'Rising rates hurt bonds', 'realityView': 'Safe haven demand from wealthy', 'conviction': 0.75, 'opportunity': 'Long TLT'}
        ]

        # Add some randomness to conviction scores
        for signal in signals:
            signal['conviction'] += np.random.normal(0, 0.05)
            signal['conviction'] = max(0.3, min(1.0, signal['conviction']))

        return signals

    def generate_contrarian_data(self) -> Dict:
        """Generate mock contrarian opportunities data."""
        opportunities = []
        symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX', 'IWM']

        for i, symbol in enumerate(symbols):
            gary_score = np.random.beta(2, 3)  # Skewed toward lower values, occasional high
            conviction = 0.6 + gary_score * 0.3

            opportunities.append({
                'id': f'mock_opp_{i}',
                'symbol': symbol,
                'thesis': f'Inequality analysis suggests {symbol} mispricing',
                'consensusView': 'Market efficiency holds',
                'contrarianView': 'Wealth concentration creates bias',
                'inequalityCorrelation': 0.7 + np.random.normal(0, 0.1),
                'convictionScore': conviction,
                'expectedPayoff': 1.2 + conviction * 1.5,
                'timeframeDays': random.randint(30, 180),
                'entryPrice': 100.0 + np.random.normal(0, 10),
                'targetPrice': 100.0 + (1.0 + conviction) * 15,
                'stopLoss': 100.0 - 12.0,
                'currentPrice': 100.0 + np.random.normal(0, 3),
                'historicalAccuracy': 0.6 + conviction * 0.2,
                'garyMomentScore': gary_score,
                'allocationBucket': 'risky_20' if gary_score < 0.7 else 'safe_80',
                'safetyScore': 0.3 + gary_score * 0.5,
                'positionSize': 0.02 + conviction * 0.08,
                'supportingData': [
                    {'metric': 'DPI Signal', 'value': gary_score * 100, 'trend': 'up'},
                    {'metric': 'AI Confidence', 'value': conviction * 100, 'trend': 'up'},
                    {'metric': 'Narrative Gap', 'value': abs(conviction - 0.5) * 100, 'trend': 'up'},
                    {'metric': 'Catalyst Timing', 'value': 75.0, 'trend': 'up'}
                ]
            })

        return {
            'opportunities': opportunities,
            'barbell_allocation': {
                'safe_allocation': 0.8,
                'risky_allocation': 0.2,
                'safe_assets': ['TLT', 'SHY'],
                'risky_assets': ['SPY', 'QQQ', 'VIX'],
                'total_mispricings': len(opportunities)
            }
        }

    def generate_ai_status(self) -> Dict:
        """Generate mock AI status data."""
        return {
            'utility_parameters': {
                'risk_aversion': 0.5 + np.random.normal(0, 0.05),
                'loss_aversion': 2.0 + np.random.normal(0, 0.1),
                'kelly_safety_factor': 0.25 + np.random.normal(0, 0.02),
                'confidence_threshold': 0.7 + np.random.normal(0, 0.03),
                'last_updated': datetime.now().isoformat()
            },
            'calibration_metrics': {
                'total_predictions': random.randint(50, 200),
                'resolved_predictions': random.randint(30, 150),
                'overall_accuracy': 0.65 + np.random.normal(0, 0.05),
                'brier_score': 0.25 + np.random.normal(0, 0.05),
                'log_loss': 0.6 + np.random.normal(0, 0.1),
                'calibration_error': 0.1 + np.random.normal(0, 0.02),
                'pit_p_value': np.random.uniform(0.05, 0.95)
            },
            'mathematical_framework': {
                'dpi_active': True,
                'narrative_gap_tracking': True,
                'repricing_potential_calculated': True,
                'kelly_optimization': True,
                'evt_risk_management': True,
                'barbell_constraints': True
            },
            'streaming_status': {
                'ai_processing': True,
                'mispricing_detection': True,
                'websocket_connections': len(self.active_connections),
                'last_update': datetime.now().isoformat()
            }
        }

    def generate_barbell_allocation(self) -> Dict:
        """Generate mock barbell allocation data."""
        return {
            'allocation_summary': {
                'safe_allocation': 0.8,
                'risky_allocation': 0.2,
                'transition_allocation': 0.0
            },
            'safe_assets': ['TLT', 'IEF', 'SHY'],
            'risky_assets': ['SPY', 'QQQ'],
            'transition_assets': ['GLD'],
            'last_rebalance': datetime.now().isoformat(),
            'rebalance_reason': 'Periodic rebalancing',
            'total_mispricings': 5,
            'safety_promotions_available': 1
        }

async def init_ai_services():
    """Initialize AI services if available."""
    if AI_AVAILABLE:
        try:
            logger.info("Starting AI dashboard integration...")
            await ai_dashboard_integrator.start_ai_dashboard_integration()
            logger.info("AI dashboard integration started successfully")
        except Exception as e:
            logger.error(f"Failed to start AI integration: {e}")
    else:
        logger.info("Running in mock mode - AI services not available")

def main():
    """Main entry point."""
    server = SimpleDashboardServer()

    # Initialize AI services
    if AI_AVAILABLE:
        # Start AI services in the background
        import threading
        def run_ai_init():
            asyncio.run(init_ai_services())

        ai_thread = threading.Thread(target=run_ai_init, daemon=True)
        ai_thread.start()

    # Start the server
    try:
        logger.info("Starting Gary×Taleb Risk Dashboard with AI integration")
        server.run(host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()