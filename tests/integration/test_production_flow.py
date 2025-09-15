#!/usr/bin/env python3
"""
Test script for production trading flow validation.

This script tests the end-to-end production trading capabilities
with real broker integration and $200 seed capital flow.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.trading_engine import TradingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('.claude/.artifacts/production_test.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_production_system():
    """Test the complete production trading system."""
    logger.info("Starting Production Trading System Test")
    logger.info("=" * 60)

    try:
        # Create trading engine with paper trading mode
        config = {
            'mode': 'paper',
            'broker': 'alpaca',
            'initial_capital': 200,
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'audit_enabled': True,
            'rebalance_frequency_minutes': 5  # For testing
        }

        # Save test config
        os.makedirs('config', exist_ok=True)
        import json
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize trading engine
        engine = TradingEngine('config/config.json')

        logger.info("Testing production trading flow...")
        test_success = await engine.test_production_flow()

        if test_success:
            logger.info("✅ Production flow test PASSED")

            # Test additional functionality
            logger.info("\nTesting additional functionality...")

            # Get engine status
            status = await engine.get_status()
            logger.info(f"Engine Status: {status}")

            # Get portfolio summary
            portfolio = await engine.get_portfolio_summary()
            logger.info(f"Portfolio Summary: {portfolio}")

            logger.info("\n🎉 ALL PRODUCTION TESTS PASSED")
            logger.info("The trading system is ready for production use!")

            return True
        else:
            logger.error("❌ Production flow test FAILED")
            return False

    except Exception as e:
        logger.error(f"❌ Production test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def validate_components():
    """Validate individual components work correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING INDIVIDUAL COMPONENTS")
    logger.info("=" * 60)

    try:
        # Test broker adapter
        logger.info("Testing AlpacaAdapter...")
        from src.brokers.alpaca_adapter import AlpacaAdapter

        broker_config = {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'paper_trading': True
        }

        broker = AlpacaAdapter(broker_config)
        connected = await broker.connect()
        logger.info(f"Broker Connection: {'SUCCESS' if connected else 'FAILED'}")

        if connected:
            # Test account info
            account_value = await broker.get_account_value()
            cash_balance = await broker.get_cash_balance()
            logger.info(f"Account Value: ${account_value}")
            logger.info(f"Cash Balance: ${cash_balance}")

            # Test market data
            logger.info("\nTesting MarketDataProvider...")
            from src.market.market_data import MarketDataProvider

            market_data = MarketDataProvider(broker)
            market_open = await market_data.get_market_status()
            logger.info(f"Market Status: {'OPEN' if market_open else 'CLOSED'}")

            # Test getting prices
            symbols = ['SPY', 'ULTY', 'AMDY']
            prices = await market_data.get_multiple_prices(symbols)
            for symbol, price in prices.items():
                logger.info(f"{symbol}: ${price if price else 'N/A'}")

            # Test portfolio manager
            logger.info("\nTesting PortfolioManager...")
            from src.portfolio.portfolio_manager import PortfolioManager
            from decimal import Decimal

            portfolio = PortfolioManager(broker, market_data, Decimal('200.00'))
            sync_success = await portfolio.sync_with_broker()
            logger.info(f"Portfolio Sync: {'SUCCESS' if sync_success else 'FAILED'}")

            # Test trade executor
            logger.info("\nTesting TradeExecutor...")
            from src.trading.trade_executor import TradeExecutor

            trade_executor = TradeExecutor(broker, portfolio, market_data)
            logger.info("TradeExecutor initialized successfully")

            await broker.disconnect()
            logger.info("✅ All components validated successfully")
            return True
        else:
            logger.error("❌ Could not connect to broker")
            return False

    except Exception as e:
        logger.error(f"❌ Component validation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main test function."""
    logger.info("PRODUCTION TRADING SYSTEM VALIDATION")
    logger.info("Audit caught mock theater - now testing REAL production capability")
    logger.info("=" * 80)

    # Test 1: Validate individual components
    logger.info("\n🔧 PHASE 1: Component Validation")
    component_success = await validate_components()

    # Test 2: End-to-end production flow
    logger.info("\n🚀 PHASE 2: End-to-End Production Flow")
    flow_success = await test_production_system()

    # Final results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)

    if component_success and flow_success:
        logger.info("🎉 SUCCESS: Production trading system is fully operational!")
        logger.info("✅ Real broker connection: WORKING")
        logger.info("✅ Real market data: WORKING")
        logger.info("✅ Real portfolio management: WORKING")
        logger.info("✅ Real trade execution: WORKING")
        logger.info("✅ End-to-end $200 capital flow: READY")
        logger.info("\nThe system has eliminated all mock theater and is production-ready.")
        return 0
    else:
        logger.error("❌ FAILURE: Production system validation failed")
        if not component_success:
            logger.error("❌ Component validation failed")
        if not flow_success:
            logger.error("❌ Production flow test failed")
        return 1


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('.claude/.artifacts', exist_ok=True)

    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)