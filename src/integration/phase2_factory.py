"""
Phase 2 System Factory and Dependency Injection
Provides centralized initialization for all Phase 2 components with proper dependency wiring
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from decimal import Decimal

# Phase 1 imports
from ..strategies.dpi_calculator import DistributionalPressureIndex
from ..strategies.antifragility_engine import AntifragilityEngine
from ..gates.gate_manager import GateManager
from ..brokers.alpaca_adapter import AlpacaAdapter
from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.trade_executor import TradeExecutor
from ..market.market_data import MarketDataProvider

# Phase 2 imports
from ..safety.kill_switch_system import KillSwitchSystem
from ..cycles.weekly_siphon_automator import WeeklySiphonAutomator
from ..cycles.profit_calculator import ProfitCalculator
from ..risk.kelly_criterion import KellyCriterionCalculator
from ..risk.enhanced_evt_models import EnhancedEVTEngine

logger = logging.getLogger(__name__)


class Phase2SystemFactory:
    """Factory for initializing and wiring Phase 2 systems"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory with configuration

        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_path)
        self.phase1_systems = {}
        self.phase2_systems = {}
        self._initialize_logging()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        return {
            "broker": {
                "provider": "alpaca",
                "api_key": os.getenv("ALPACA_API_KEY", ""),
                "api_secret": os.getenv("ALPACA_API_SECRET", ""),
                "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                "paper_trading": True
            },
            "risk": {
                "max_position_size": 0.25,  # 25% max per position
                "max_kelly": 0.25,  # 25% max Kelly
                "cash_floor": 0.5,  # 50% minimum cash
                "loss_limit_daily": 0.05,  # 5% daily loss limit
                "loss_limit_weekly": 0.10  # 10% weekly loss limit
            },
            "kill_switch": {
                "response_time_target_ms": 500,
                "triggers": {
                    "api_failure": True,
                    "loss_limit": True,
                    "position_limit": True,
                    "heartbeat_timeout": True
                },
                "hardware_auth": {
                    "enabled": False,  # Disabled for initial testing
                    "yubikey": False,
                    "biometric": False
                }
            },
            "siphon": {
                "schedule": {
                    "day": "friday",
                    "time": "18:00",  # 6:00 PM
                    "timezone": "US/Eastern"
                },
                "profit_split": 0.5,  # 50/50 split
                "minimum_profit": 100,  # $100 minimum
                "safety_buffer": 50  # $50 safety buffer
            },
            "dashboard": {
                "websocket_port": 8080,
                "http_port": 3000,
                "update_interval_ms": 1000
            }
        }

    def _initialize_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def initialize_phase1_systems(self) -> Dict[str, Any]:
        """Initialize all Phase 1 foundation systems"""
        logger.info("Initializing Phase 1 foundation systems...")

        # Initialize broker
        broker_config = self.config["broker"]

        # Check if we should use production or mock
        use_production = (
            broker_config.get("use_production", False) and
            broker_config.get("api_key") and
            broker_config.get("api_secret")
        )

        if use_production:
            # Use REAL production broker
            from ..brokers.alpaca_production import AlpacaProductionAdapter
            broker = AlpacaProductionAdapter({
                "api_key": broker_config.get("api_key"),
                "secret_key": broker_config.get("api_secret"),
                "paper_trading": broker_config.get("paper_trading", True),
                "base_url": broker_config.get("base_url")
            })
            logger.info("Using PRODUCTION Alpaca broker adapter")
        else:
            # ISS-004 FIX: Require real credentials, no mock defaults
            api_key = broker_config.get("api_key")
            api_secret = broker_config.get("api_secret")
            if not api_key or not api_secret:
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY required. "
                    "Set in .env file or config. See .env.example for setup."
                )
            broker = AlpacaAdapter({
                "api_key": api_key,
                "secret_key": api_secret,
                "base_url": broker_config.get("base_url", "https://paper-api.alpaca.markets"),
                "paper_trading": broker_config.get("paper_trading", True)
            })
            logger.info("Using Alpaca broker adapter (credentials required)")

        # Initialize market data provider
        market_data = MarketDataProvider(broker)

        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(broker, market_data)

        # Initialize DPI calculator
        dpi_calculator = DistributionalPressureIndex()

        # Initialize antifragility engine
        antifragility_engine = AntifragilityEngine(
            portfolio_value=200.0,  # $200 seed capital
            risk_tolerance=0.02  # 2% risk tolerance
        )

        # Initialize gate manager
        gate_manager = GateManager()

        # Initialize trade executor
        trade_executor = TradeExecutor(broker, portfolio_manager, market_data)

        self.phase1_systems = {
            "broker": broker,
            "market_data": market_data,
            "portfolio_manager": portfolio_manager,
            "dpi_calculator": dpi_calculator,
            "antifragility_engine": antifragility_engine,
            "gate_manager": gate_manager,
            "trade_executor": trade_executor
        }

        logger.info("Phase 1 systems initialized successfully")
        return self.phase1_systems

    def initialize_phase2_systems(self) -> Dict[str, Any]:
        """Initialize all Phase 2 risk & quality systems"""
        if not self.phase1_systems:
            self.initialize_phase1_systems()

        logger.info("Initializing Phase 2 risk & quality systems...")

        # Initialize kill switch system
        kill_switch = KillSwitchSystem(
            broker_interface=self.phase1_systems["broker"],
            config=self.config["kill_switch"]
        )

        # Initialize Kelly criterion calculator
        kelly_calculator = KellyCriterionCalculator(
            dpi_calculator=self.phase1_systems["dpi_calculator"],
            gate_manager=self.phase1_systems["gate_manager"]
        )

        # Initialize enhanced EVT engine
        evt_engine = EnhancedEVTEngine()

        # Initialize profit calculator
        profit_calculator = ProfitCalculator(
            initial_capital=Decimal("200")  # $200 seed capital
        )

        # Initialize weekly siphon automator
        siphon_automator = WeeklySiphonAutomator(
            portfolio_manager=self.phase1_systems["portfolio_manager"],
            broker_adapter=self.phase1_systems["broker"],
            profit_calculator=profit_calculator,
            enable_auto_execution=False  # Disabled for safety in testing
        )

        self.phase2_systems = {
            "kill_switch": kill_switch,
            "kelly_calculator": kelly_calculator,
            "evt_engine": evt_engine,
            "profit_calculator": profit_calculator,
            "siphon_automator": siphon_automator
        }

        logger.info("Phase 2 systems initialized successfully")
        return self.phase2_systems

    def get_integrated_system(self) -> Dict[str, Any]:
        """Get complete integrated Phase 1 + Phase 2 system"""
        if not self.phase1_systems:
            self.initialize_phase1_systems()
        if not self.phase2_systems:
            self.initialize_phase2_systems()

        return {
            **self.phase1_systems,
            **self.phase2_systems,
            "config": self.config
        }

    def validate_integration(self) -> Dict[str, bool]:
        """Validate that all systems are properly integrated"""
        validation_results = {}

        try:
            # Test Phase 1 systems
            validation_results["broker_connection"] = self.phase1_systems["broker"] is not None
            validation_results["dpi_calculator"] = self.phase1_systems["dpi_calculator"] is not None
            validation_results["gate_manager"] = self.phase1_systems["gate_manager"] is not None

            # Test Phase 2 systems
            validation_results["kill_switch"] = self.phase2_systems["kill_switch"] is not None
            validation_results["kelly_calculator"] = self.phase2_systems["kelly_calculator"] is not None
            validation_results["evt_engine"] = self.phase2_systems["evt_engine"] is not None
            validation_results["siphon_automator"] = self.phase2_systems["siphon_automator"] is not None

            # Test integration points
            validation_results["kill_switch_broker"] = (
                hasattr(self.phase2_systems["kill_switch"], "broker") and
                self.phase2_systems["kill_switch"].broker is not None
            )
            validation_results["kelly_dpi_integration"] = (
                hasattr(self.phase2_systems["kelly_calculator"], "dpi_calculator") and
                self.phase2_systems["kelly_calculator"].dpi_calculator is not None
            )

            # Overall status
            validation_results["all_systems_ready"] = all(validation_results.values())

        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_results["error"] = str(e)
            validation_results["all_systems_ready"] = False

        return validation_results

    def create_test_instance(self) -> 'Phase2SystemFactory':
        """Create a test instance with mock data for development"""
        test_config = self.config.copy()
        test_config["broker"]["paper_trading"] = True
        test_config["broker"]["use_production"] = False
        test_config["kill_switch"]["hardware_auth"]["enabled"] = False

        test_factory = Phase2SystemFactory()
        test_factory.config = test_config
        return test_factory

    @classmethod
    def create_production_instance(cls) -> 'Phase2SystemFactory':
        """
        Create a PRODUCTION instance with real broker and configuration.

        This uses the ProductionConfig and requires real API credentials.

        Returns:
            Production-configured factory instance

        Raises:
            ValueError: If production requirements not met
        """
        # Import production configuration
        from ...config.production_config import ProductionConfig

        # Validate production config
        ProductionConfig.validate()

        # Create production configuration
        production_config = {
            "broker": {
                "provider": "alpaca",
                "use_production": True,  # USE REAL BROKER
                "api_key": ProductionConfig.ALPACA_API_KEY,
                "api_secret": ProductionConfig.ALPACA_SECRET_KEY,
                "base_url": ProductionConfig.ALPACA_BASE_URL,
                "paper_trading": ProductionConfig.PAPER_TRADING
            },
            "risk": {
                "max_position_size": ProductionConfig.MAX_POSITION_SIZE,
                "max_kelly": ProductionConfig.MAX_KELLY_FRACTION,
                "cash_floor": ProductionConfig.CASH_FLOOR,
                "loss_limit_daily": float(ProductionConfig.DAILY_LOSS_LIMIT),
                "loss_limit_weekly": float(ProductionConfig.WEEKLY_LOSS_LIMIT)
            },
            "kill_switch": ProductionConfig.KILL_SWITCH_CONFIG,
            "siphon": ProductionConfig.SIPHON_CONFIG,
            "dashboard": ProductionConfig.DASHBOARD_CONFIG,
            "monitoring": ProductionConfig.MONITORING_CONFIG,
            "audit": ProductionConfig.AUDIT_CONFIG
        }

        # Create factory with production config
        factory = cls()
        factory.config = production_config

        logger.warning(
            "PRODUCTION INSTANCE CREATED - "
            f"Mode: {'PAPER' if ProductionConfig.PAPER_TRADING else 'LIVE'}, "
            f"Simulation: {ProductionConfig.SIMULATION_MODE}"
        )

        return factory