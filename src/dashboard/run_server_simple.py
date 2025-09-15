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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict

# Add the parent directory to the path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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
            return {"message": "Gary×Taleb Risk Dashboard API", "status": "running"}

        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "connections": len(self.active_connections)
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

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.connect(websocket)
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
            ("Volatility spike detected", "volatility")
        ]

        msg, alert_type = random.choice(alert_types)

        return {
            "id": f"alert_{int(time.time() * 1000)}",
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "type": alert_type,
            "message": msg,
            "timestamp": time.time()
        }

def main():
    """Main entry point."""
    server = SimpleDashboardServer()

    # Start the server
    try:
        server.run(host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()