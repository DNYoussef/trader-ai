#!/usr/bin/env python3
"""
Startup script for the Gary×Taleb Risk Dashboard Server.
Integrates with the existing trading system and provides real-time WebSocket updates.
"""

import sys
import os
import asyncio
import logging
import threading
from pathlib import Path

# Add the parent directory to the path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.server.websocket_server import RiskDashboardServer
from src.trading_engine import TradingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/dashboard_server.log')
    ]
)

logger = logging.getLogger(__name__)

class DashboardIntegration:
    """
    Integration layer between trading engine and dashboard server.
    Monitors trading system and pushes updates to dashboard.
    """

    def __init__(self, trading_engine: TradingEngine, dashboard_server: RiskDashboardServer):
        self.trading_engine = trading_engine
        self.dashboard_server = dashboard_server
        self.running = False
        self.update_interval = 1.0  # Update every second

    async def start_monitoring(self):
        """Start monitoring the trading engine and pushing updates."""
        self.running = True
        logger.info("Starting dashboard integration monitoring")

        while self.running:
            try:
                await self.update_dashboard()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in dashboard update: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def update_dashboard(self):
        """Update dashboard with latest trading data."""
        try:
            # Get portfolio data from trading engine
            portfolio_data = await self.get_portfolio_data()
            if portfolio_data:
                self.dashboard_server.update_risk_metrics(portfolio_data)

            # Get positions data
            positions_data = await self.get_positions_data()
            for symbol, position_data in positions_data.items():
                self.dashboard_server.update_position(symbol, position_data)

        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")

    async def get_portfolio_data(self) -> dict:
        """Extract portfolio metrics from trading engine."""
        try:
            # This would integrate with the actual trading engine
            # For now, we'll use simulated data
            return {
                'portfolio_value': 10000.0,  # Replace with actual portfolio value
                'max_drawdown': 0.05,        # Replace with actual max drawdown
                'volatility': 0.15,          # Replace with actual volatility
                'beta': 1.0,                 # Replace with actual beta
                'positions_count': 5,        # Replace with actual position count
                'cash_available': 2000.0,    # Replace with actual cash
                'margin_used': 0.3,          # Replace with actual margin usage
                'unrealized_pnl': 150.0,     # Replace with actual unrealized P&L
                'daily_pnl': 75.0,           # Replace with actual daily P&L
            }
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return {}

    async def get_positions_data(self) -> dict:
        """Extract positions data from trading engine."""
        try:
            # This would integrate with the actual trading engine
            # For now, we'll use simulated data
            positions = {}
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

            for symbol in symbols:
                positions[symbol] = {
                    'quantity': 100,          # Replace with actual quantity
                    'market_value': 2000.0,   # Replace with actual market value
                    'unrealized_pnl': 50.0,   # Replace with actual unrealized P&L
                    'entry_price': 150.0,     # Replace with actual entry price
                    'current_price': 155.0,   # Replace with actual current price
                    'weight': 0.2,            # Replace with actual position weight
                }

            return positions
        except Exception as e:
            logger.error(f"Error getting positions data: {e}")
            return {}

    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        logger.info("Stopped dashboard integration monitoring")

def main():
    """Main entry point for the dashboard server."""
    try:
        # Ensure log directory exists
        os.makedirs('logs', exist_ok=True)

        logger.info("Starting Gary×Taleb Risk Dashboard Server")

        # Initialize trading engine (if available)
        trading_engine = None
        try:
            trading_engine = TradingEngine()
            logger.info("Trading engine connected")
        except Exception as e:
            logger.warning(f"Trading engine not available: {e}")
            logger.info("Dashboard will run in standalone mode")

        # Create dashboard server
        dashboard_server = RiskDashboardServer()

        # Create integration layer if trading engine is available
        integration = None
        if trading_engine:
            integration = DashboardIntegration(trading_engine, dashboard_server)

        # Start background tasks
        async def start_services():
            # Start dashboard background tasks
            await dashboard_server.start_background_tasks()

            # Start integration monitoring if available
            if integration:
                asyncio.create_task(integration.start_monitoring())
                logger.info("Integration monitoring started")

        # Run services in background thread
        def run_background_services():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_services())
            loop.run_forever()

        background_thread = threading.Thread(target=run_background_services, daemon=True)
        background_thread.start()

        # Start the server
        dashboard_server.run(host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        if integration:
            integration.stop_monitoring()
        logger.info("Dashboard server stopped")

if __name__ == "__main__":
    main()