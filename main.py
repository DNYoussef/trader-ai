#!/usr/bin/env python3
"""Main entry point for Gary×Taleb Autonomous Trading System"""
import sys
import signal
import logging
import argparse
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_engine import TradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global engine instance for signal handling
engine = None

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info(f"Received signal {signum}")
    if engine:
        if signum == signal.SIGTERM:
            engine.stop()
        elif signum == signal.SIGINT:
            logger.warning("Kill switch activated via Ctrl+C")
            engine.activate_kill_switch()
    sys.exit(0)

def main():
    """Main entry point"""
    global engine

    parser = argparse.ArgumentParser(description='Gary×Taleb Trading System')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--config', default='config/config.json',
                       help='Configuration file path')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (one cycle then exit)')

    args = parser.parse_args()

    # Safety check for live mode
    if args.mode == 'live':
        print("\n" + "="*50)
        print("WARNING: LIVE TRADING MODE")
        print("="*50)
        print("You are about to start LIVE TRADING with real money.")
        print("This requires:")
        print("  1. Valid Alpaca API credentials")
        print("  2. Funded account with minimum $200")
        print("  3. Understanding of all risks")
        print("\nType 'CONFIRM' to proceed or anything else to cancel:")

        confirmation = input().strip()
        if confirmation != 'CONFIRM':
            print("Live trading cancelled.")
            sys.exit(0)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Initialize engine
    engine = TradingEngine(config_path=args.config)

    # Override mode if specified
    if args.mode:
        engine.mode = args.mode

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting Gary×Taleb Trading System in {engine.mode} mode")

    if args.test:
        # Test mode - initialize and get status
        if engine.initialize():
            status = engine.get_status()
            logger.info(f"Engine status: {status}")
            logger.info("Test mode complete")
        else:
            logger.error("Failed to initialize engine")
            sys.exit(1)
    else:
        # Normal operation
        try:
            asyncio.run(engine.start())
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if engine:
                engine.activate_kill_switch()
            sys.exit(1)

if __name__ == '__main__':
    main()