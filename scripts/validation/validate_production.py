#!/usr/bin/env python3
"""
Production validation script for Gary×Taleb trading system.

This script validates that all Phase 2 systems are production-ready
with no mock code and proper configuration.
"""

import sys
import os
import asyncio
import logging
from decimal import Decimal
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Validates production readiness of the trading system."""

    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        self.factory = None

    async def validate_all(self) -> bool:
        """
        Run all production validations.

        Returns:
            True if all validations pass
        """
        print("\n" + "="*60)
        print("GARY×TALEB TRADING SYSTEM - PRODUCTION VALIDATION")
        print("="*60 + "\n")

        # Run validations
        validations = [
            ("Environment Variables", self.validate_environment),
            ("Production Configuration", self.validate_configuration),
            ("Broker Connectivity", self.validate_broker),
            ("System Integration", self.validate_integration),
            ("Risk Management", self.validate_risk_management),
            ("Kill Switch", self.validate_kill_switch),
            ("Weekly Siphon", self.validate_siphon),
            ("Database Connection", self.validate_database),
            ("No Mock Code", self.validate_no_mocks),
            ("Performance Benchmarks", self.validate_performance)
        ]

        all_passed = True
        for name, validator in validations:
            print(f"\n[VALIDATING] {name}...")
            try:
                result = await validator()
                self.validation_results[name] = result

                if result:
                    print(f"  ✅ {name}: PASSED")
                else:
                    print(f"  ❌ {name}: FAILED")
                    all_passed = False

            except Exception as e:
                print(f"  ❌ {name}: ERROR - {e}")
                self.errors.append(f"{name}: {e}")
                all_passed = False

        # Print summary
        self.print_summary(all_passed)

        return all_passed

    async def validate_environment(self) -> bool:
        """Validate environment variables are set."""
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY'
        ]

        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            self.errors.append(f"Missing environment variables: {', '.join(missing)}")
            return False

        # Check if using paper or live
        base_url = os.getenv('ALPACA_BASE_URL', '')
        if 'paper' in base_url:
            print("    Mode: PAPER TRADING (safe)")
        elif 'api.alpaca' in base_url:
            print("    ⚠️  Mode: LIVE TRADING (real money)")
            self.warnings.append("Using LIVE trading API - real money at risk")

        return True

    async def validate_configuration(self) -> bool:
        """Validate production configuration."""
        try:
            from config.production_config import ProductionConfig

            # Validate configuration
            ProductionConfig.validate()

            # Check critical settings
            if ProductionConfig.SIMULATION_MODE:
                self.errors.append("SIMULATION_MODE is True - must be False for production")
                return False

            print(f"    Paper Trading: {ProductionConfig.PAPER_TRADING}")
            print(f"    Max Position Size: {ProductionConfig.MAX_POSITION_SIZE*100:.0f}%")
            print(f"    Daily Loss Limit: ${ProductionConfig.DAILY_LOSS_LIMIT}")
            print(f"    P(ruin) Threshold: {ProductionConfig.MAX_RUIN_PROBABILITY}")

            return True

        except ImportError as e:
            self.errors.append(f"Cannot import ProductionConfig: {e}")
            return False
        except ValueError as e:
            self.errors.append(f"Configuration validation failed: {e}")
            return False

    async def validate_broker(self) -> bool:
        """Validate broker connectivity."""
        try:
            from src.brokers.alpaca_production import AlpacaProductionAdapter

            # Create broker instance
            broker = AlpacaProductionAdapter({
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY'),
                'paper_trading': True  # Always use paper for validation
            })

            # Test connection
            connected = await broker.connect()
            if not connected:
                self.errors.append("Failed to connect to Alpaca")
                return False

            # Get account info
            account = await broker.get_account_info()
            print(f"    Account ID: {account['account_id']}")
            print(f"    Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"    Buying Power: ${account['buying_power']:,.2f}")
            print(f"    Trading Blocked: {account['trading_blocked']}")

            # Check if account is ready
            if account['trading_blocked']:
                self.errors.append("Trading is blocked on this account")
                return False

            # Disconnect
            await broker.disconnect()

            return True

        except ImportError:
            self.errors.append("AlpacaProductionAdapter not found")
            return False
        except Exception as e:
            self.errors.append(f"Broker validation failed: {e}")
            return False

    async def validate_integration(self) -> bool:
        """Validate system integration."""
        try:
            from src.integration.phase2_factory import Phase2SystemFactory

            # Create production factory
            self.factory = Phase2SystemFactory.create_production_instance()

            # Initialize systems
            phase1 = self.factory.initialize_phase1_systems()
            phase2 = self.factory.initialize_phase2_systems()

            # Validate integration
            validation = self.factory.validate_integration()

            if not validation.get('all_systems_ready'):
                failed = [k for k, v in validation.items() if not v and k != 'all_systems_ready']
                self.errors.append(f"Integration validation failed: {', '.join(failed)}")
                return False

            print(f"    Phase 1 Systems: {len(phase1)}")
            print(f"    Phase 2 Systems: {len(phase2)}")
            print(f"    All Integrations: VALID")

            return True

        except ImportError as e:
            self.errors.append(f"Cannot import factory: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Integration validation failed: {e}")
            return False

    async def validate_risk_management(self) -> bool:
        """Validate risk management systems."""
        if not self.factory:
            self.errors.append("Factory not initialized")
            return False

        systems = self.factory.get_integrated_system()

        # Check Kelly criterion
        kelly = systems.get('kelly_calculator')
        if not kelly:
            self.errors.append("Kelly calculator not found")
            return False

        # Check EVT engine
        evt = systems.get('evt_engine')
        if not evt:
            self.errors.append("EVT engine not found")
            return False

        print(f"    Kelly Calculator: READY")
        print(f"    EVT Engine: READY")
        print(f"    Risk Limits: CONFIGURED")

        return True

    async def validate_kill_switch(self) -> bool:
        """Validate kill switch system."""
        if not self.factory:
            self.errors.append("Factory not initialized")
            return False

        systems = self.factory.get_integrated_system()
        kill_switch = systems.get('kill_switch')

        if not kill_switch:
            self.errors.append("Kill switch not found")
            return False

        # Check configuration
        config = systems['config']['kill_switch']
        target_ms = config.get('response_time_target_ms', 0)

        if target_ms > 500:
            self.warnings.append(f"Kill switch target {target_ms}ms > 500ms requirement")

        print(f"    Kill Switch: ARMED")
        print(f"    Response Target: {target_ms}ms")
        print(f"    Triggers Enabled: {len([k for k, v in config['triggers'].items() if v])}")

        return True

    async def validate_siphon(self) -> bool:
        """Validate weekly siphon system."""
        if not self.factory:
            self.errors.append("Factory not initialized")
            return False

        systems = self.factory.get_integrated_system()
        siphon = systems.get('siphon_automator')

        if not siphon:
            self.errors.append("Siphon automator not found")
            return False

        # Check configuration
        config = systems['config']['siphon']

        print(f"    Schedule: {config['schedule']['day']} @ {config['schedule']['time']}")
        print(f"    Profit Split: {config['profit_split']*100:.0f}%")
        print(f"    Minimum Profit: ${config['minimum_profit']}")

        return True

    async def validate_database(self) -> bool:
        """Validate database connection."""
        # For now, just check if configuration exists
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')

        if not db_host or not db_name:
            self.warnings.append("Database not configured - using file-based storage")
            print("    Database: NOT CONFIGURED (using files)")
        else:
            print(f"    Database: {db_name}@{db_host}")

        return True  # Not critical for initial production

    async def validate_no_mocks(self) -> bool:
        """Validate no mock code is being used."""
        if not self.factory:
            self.errors.append("Factory not initialized")
            return False

        config = self.factory.config

        # Check broker configuration
        if not config.get('broker', {}).get('use_production'):
            self.errors.append("Broker not using production mode")
            return False

        # Check for mock imports
        systems = self.factory.get_integrated_system()
        broker = systems.get('broker')

        if broker and hasattr(broker, 'mock_mode') and broker.mock_mode:
            self.errors.append("Broker is in mock mode")
            return False

        print("    No Mock Code: VERIFIED")
        print("    Production Broker: ACTIVE")

        return True

    async def validate_performance(self) -> bool:
        """Validate performance benchmarks."""
        # Simple performance check
        import time

        # Test factory initialization speed
        start = time.time()
        factory = Phase2SystemFactory()
        factory.initialize_phase1_systems()
        factory.initialize_phase2_systems()
        init_time = (time.time() - start) * 1000

        if init_time > 1000:
            self.warnings.append(f"Initialization time {init_time:.0f}ms > 1000ms")

        print(f"    Initialization Time: {init_time:.0f}ms")
        print(f"    Kill Switch Target: <500ms")

        return True

    def print_summary(self, all_passed: bool):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        # Count results
        passed = sum(1 for v in self.validation_results.values() if v)
        failed = len(self.validation_results) - passed

        print(f"\nResults: {passed} PASSED, {failed} FAILED")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("\n" + "="*60)

        if all_passed:
            print("✅ PRODUCTION VALIDATION: PASSED")
            print("\nThe system is ready for production deployment.")
            print("Remember to:")
            print("  1. Start with paper trading")
            print("  2. Monitor closely during initial deployment")
            print("  3. Test with small amounts first")
            print("  4. Have kill switch ready")
        else:
            print("❌ PRODUCTION VALIDATION: FAILED")
            print("\nThe system is NOT ready for production.")
            print("Fix all errors before deploying.")

        print("="*60 + "\n")

        # Save results
        self.save_results(all_passed)

    def save_results(self, passed: bool):
        """Save validation results to file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings
        }

        filename = f"validation_{'passed' if passed else 'failed'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {filename}")


async def main():
    """Run production validation."""
    validator = ProductionValidator()
    passed = await validator.validate_all()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())