#!/usr/bin/env python3
"""
Production deployment script for Gary×Taleb trading system.

This script deploys the trading system to production after validation.
DANGER: This handles REAL MONEY when configured for live trading.
"""

import sys
import os
import asyncio
import logging
import signal
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """Production deployment for the trading system."""

    def __init__(self):
        self.factory = None
        self.systems = None
        self.running = False
        self.broker = None

    async def deploy(self):
        """
        Deploy the trading system to production.

        This is the main entry point for production deployment.
        """
        print("\n" + "="*60)
        print("GARY×TALEB TRADING SYSTEM - PRODUCTION DEPLOYMENT")
        print("="*60 + "\n")

        # Validate first
        print("[1/5] Running production validation...")
        from validate_production import ProductionValidator
        validator = ProductionValidator()

        if not await validator.validate_all():
            logger.error("Production validation failed - deployment aborted")
            return False

        # Initialize systems
        print("\n[2/5] Initializing production systems...")
        if not await self.initialize_systems():
            logger.error("System initialization failed")
            return False

        # Connect to broker
        print("\n[3/5] Connecting to broker...")
        if not await self.connect_broker():
            logger.error("Broker connection failed")
            return False

        # Start monitoring
        print("\n[4/5] Starting monitoring and safety systems...")
        if not await self.start_monitoring():
            logger.error("Monitoring initialization failed")
            return False

        # Start trading loop
        print("\n[5/5] Starting trading loop...")
        await self.run_trading_loop()

        return True

    async def initialize_systems(self) -> bool:
        """Initialize all production systems."""
        try:
            from src.integration.phase2_factory import Phase2SystemFactory
            from config.production_config import ProductionConfig

            # Create production instance
            self.factory = Phase2SystemFactory.create_production_instance()

            # Initialize all systems
            phase1 = self.factory.initialize_phase1_systems()
            phase2 = self.factory.initialize_phase2_systems()

            # Get integrated system
            self.systems = self.factory.get_integrated_system()

            logger.info(
                f"Systems initialized - "
                f"Phase 1: {len(phase1)}, Phase 2: {len(phase2)}"
            )

            # Log configuration
            logger.info(f"Paper Trading: {ProductionConfig.PAPER_TRADING}")
            logger.info(f"Max Position Size: {ProductionConfig.MAX_POSITION_SIZE}")
            logger.info(f"Daily Loss Limit: ${ProductionConfig.DAILY_LOSS_LIMIT}")

            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    async def connect_broker(self) -> bool:
        """Connect to the broker."""
        try:
            self.broker = self.systems['broker']

            # Connect
            connected = await self.broker.connect()
            if not connected:
                return False

            # Get account info
            account = await self.broker.get_account_info()

            logger.info(
                f"Connected to broker - "
                f"Account: {account['account_id']}, "
                f"Portfolio: ${account['portfolio_value']:,.2f}, "
                f"Buying Power: ${account['buying_power']:,.2f}"
            )

            # Check market hours
            market = await self.broker.get_market_hours()
            if market['is_open']:
                logger.info("Market is OPEN")
            else:
                logger.info(
                    f"Market is CLOSED - "
                    f"Next open: {market.get('next_open', 'unknown')}"
                )

            return True

        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return False

    async def start_monitoring(self) -> bool:
        """Start monitoring and safety systems."""
        try:
            # Activate kill switch
            kill_switch = self.systems['kill_switch']
            kill_switch.armed = True
            logger.info("Kill switch ARMED")

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            # Start heartbeat
            asyncio.create_task(self.heartbeat_loop())

            # Start performance monitoring
            asyncio.create_task(self.monitor_performance())

            logger.info("Monitoring systems activated")
            return True

        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
            return False

    async def run_trading_loop(self):
        """Main trading loop."""
        self.running = True

        logger.info("="*60)
        logger.info("TRADING SYSTEM ACTIVE - PRODUCTION MODE")
        logger.info("="*60)

        try:
            while self.running:
                # Check if market is open
                market = await self.broker.get_market_hours()

                if not market['is_open']:
                    logger.info("Market closed - waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue

                # Get current positions
                positions = await self.broker.get_positions()
                logger.info(f"Current positions: {len(positions)}")

                # Run trading logic
                await self.execute_trading_cycle()

                # Check for weekly siphon
                await self.check_weekly_siphon()

                # Sleep before next cycle
                await asyncio.sleep(60)  # Run every minute

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await self.emergency_shutdown()

    async def execute_trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # Get systems
            dpi = self.systems['dpi_calculator']
            antifragility = self.systems['antifragility_engine']
            kelly = self.systems['kelly_calculator']
            gate_manager = self.systems['gate_manager']
            trade_executor = self.systems['trade_executor']

            # Get market data
            symbols = ['SPY', 'QQQ', 'IWM']  # Start with major ETFs

            for symbol in symbols:
                try:
                    # Get current price
                    price = await self.broker.get_current_price(symbol)

                    # Get historical data
                    bars = await self.broker.get_historical_data(symbol, '1Day', 100)

                    if not bars:
                        continue

                    # Calculate DPI
                    dpi_value = dpi.calculate_dpi(bars)

                    # Check if we should trade
                    if abs(dpi_value) > 0.7:  # Strong signal
                        # Calculate position size with Kelly
                        kelly_size = kelly.calculate_kelly_fraction(
                            win_probability=0.55,  # Conservative estimate
                            win_return=0.02,       # 2% target
                            loss_return=0.01       # 1% stop loss
                        )

                        # Apply gate constraints
                        max_size = gate_manager.get_max_position_size()
                        position_size = min(kelly_size, max_size)

                        logger.info(
                            f"{symbol}: DPI={dpi_value:.2f}, "
                            f"Kelly={kelly_size:.2%}, "
                            f"Position={position_size:.2%}"
                        )

                        # ISS-045: Trade execution disabled by design
                        # Enable only after: (1) paper trading validation, (2) risk review
                        # Call: await self.broker.submit_order(...) when ready

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    async def check_weekly_siphon(self):
        """Check if weekly siphon should execute."""
        try:
            siphon = self.systems['siphon_automator']

            # Check if it's time for siphon
            if siphon.should_execute_now():
                logger.info("Weekly siphon time detected")

                # Calculate profit
                account = await self.broker.get_account_info()
                current_value = account['portfolio_value']

                profit_calc = self.systems['profit_calculator']
                profit = profit_calc.calculate_weekly_profit(current_value)

                if profit > 100:  # Minimum $100 profit
                    siphon_amount = profit * 0.5  # 50/50 split
                    logger.info(
                        f"Siphon eligible - "
                        f"Profit: ${profit:.2f}, "
                        f"Siphon: ${siphon_amount:.2f}"
                    )

                    # For safety, require manual approval in production
                    logger.info("Manual siphon approval required - check logs")

        except Exception as e:
            logger.error(f"Siphon check error: {e}")

    async def heartbeat_loop(self):
        """Send heartbeats to monitor system health."""
        while self.running:
            try:
                # Check broker health
                if self.broker:
                    health = await self.broker.health_check()
                    if not health:
                        logger.warning("Broker health check failed")

                # Update kill switch heartbeat
                kill_switch = self.systems.get('kill_switch')
                if kill_switch:
                    kill_switch.heartbeat()

                await asyncio.sleep(30)  # Every 30 seconds

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def monitor_performance(self):
        """Monitor system performance metrics."""
        while self.running:
            try:
                # Get account metrics
                if self.broker:
                    account = await self.broker.get_account_info()

                    # Check for risk breaches
                    daily_pnl = account['equity'] - account['last_equity']

                    if daily_pnl < -10:  # $10 daily loss limit
                        logger.warning(f"Daily loss limit approaching: ${daily_pnl:.2f}")

                        if daily_pnl < -20:  # Emergency threshold
                            logger.error("DAILY LOSS LIMIT BREACHED - TRIGGERING KILL SWITCH")
                            await self.emergency_shutdown()

                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.running = False
        asyncio.create_task(self.graceful_shutdown())

    async def graceful_shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")

        try:
            # Cancel all pending orders
            if self.broker:
                logger.info("Cancelling pending orders...")
                # ISS-045: Order cancellation handled by broker.disconnect()
                # For explicit cancellation: await self.broker.cancel_all_orders()

                # Disconnect from broker
                await self.broker.disconnect()

            logger.info("Graceful shutdown complete")

        except Exception as e:
            logger.error(f"Graceful shutdown error: {e}")

    async def emergency_shutdown(self):
        """Emergency shutdown - close all positions immediately."""
        logger.error("EMERGENCY SHUTDOWN INITIATED")

        try:
            # Trigger kill switch
            kill_switch = self.systems.get('kill_switch')
            if kill_switch:
                kill_switch.trigger('EMERGENCY', "Emergency shutdown")

            # Close all positions
            if self.broker:
                logger.error("CLOSING ALL POSITIONS")
                await self.broker.close_all_positions()

            # Stop trading
            self.running = False

            logger.error("EMERGENCY SHUTDOWN COMPLETE - SYSTEM HALTED")

        except Exception as e:
            logger.error(f"CRITICAL: Emergency shutdown failed: {e}")


async def main():
    """Main entry point."""
    print("\n⚠️  WARNING: This is PRODUCTION deployment ⚠️")
    print("This system will trade with REAL MONEY if configured for live trading.")
    print("\nAre you sure you want to continue? (yes/no): ", end="")

    confirmation = input().strip().lower()
    if confirmation != 'yes':
        print("Deployment cancelled.")
        return

    deployment = ProductionDeployment()
    await deployment.deploy()


if __name__ == "__main__":
    asyncio.run(main())