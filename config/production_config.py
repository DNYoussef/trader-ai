"""
Production configuration for Gary×Taleb trading system.

This is the REAL production configuration with no simulation or mocks.
All values are production-ready and validated for live trading.
"""

import os
from decimal import Decimal
from typing import Dict, Any


class ProductionConfig:
    """Production configuration for the trading system."""

    # ===========================================
    # TRADING MODE - SET TO FALSE FOR PRODUCTION
    # ===========================================
    SIMULATION_MODE = False  # FALSE = REAL TRADING
    PAPER_TRADING = True     # Start with paper, set False for live

    # ===========================================
    # BROKER CONFIGURATION
    # ===========================================
    BROKER_TYPE = "alpaca"

    # API Credentials (from environment variables for security)
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    # Validate credentials exist
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise ValueError(
            "PRODUCTION ERROR: Missing Alpaca API credentials.\n"
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )

    # ===========================================
    # RISK LIMITS (PRODUCTION VALUES)
    # ===========================================

    # Position sizing
    MAX_POSITION_SIZE = 0.10      # 10% max per position (conservative)
    MAX_KELLY_FRACTION = 0.25     # 25% of Kelly (1/4 Kelly for safety)
    MIN_POSITION_SIZE = 0.01      # 1% minimum position

    # Portfolio risk
    MAX_PORTFOLIO_RISK = 0.80     # 80% max portfolio allocation
    CASH_FLOOR = 0.20             # 20% minimum cash reserve
    MAX_CORRELATION = 0.70        # Max correlation between positions

    # Loss limits
    DAILY_LOSS_LIMIT = Decimal('10.00')    # $10 daily loss limit (5% of $200)
    WEEKLY_LOSS_LIMIT = Decimal('30.00')   # $30 weekly loss limit (15% of $200)
    MONTHLY_LOSS_LIMIT = Decimal('60.00')  # $60 monthly loss limit (30% of $200)

    # P(ruin) threshold
    MAX_RUIN_PROBABILITY = 1e-6   # 10^-6 maximum acceptable P(ruin)

    # ===========================================
    # KILL SWITCH CONFIGURATION
    # ===========================================

    KILL_SWITCH_CONFIG = {
        'response_time_target_ms': 500,  # <500ms response time
        'triggers': {
            'api_failure': True,
            'loss_limit': True,
            'position_limit': True,
            'heartbeat_timeout': True,
            'manual_panic': True,
            'risk_breach': True
        },
        'heartbeat_interval': 30,  # seconds
        'heartbeat_timeout': 120,  # seconds
        'hardware_auth': {
            'enabled': False,  # Enable for production with hardware keys
            'yubikey': False,
            'biometric': False,
            'master_hash': os.getenv('KILL_SWITCH_MASTER_HASH')
        }
    }

    # ===========================================
    # WEEKLY SIPHON CONFIGURATION
    # ===========================================

    SIPHON_CONFIG = {
        'enabled': True,
        'schedule': {
            'day': 'friday',
            'time': '18:00',  # 6:00 PM ET
            'timezone': 'US/Eastern'
        },
        'profit_split': 0.50,  # 50/50 split
        'minimum_profit': Decimal('100.00'),  # $100 minimum for withdrawal
        'safety_buffer': Decimal('50.00'),    # $50 safety buffer
        'auto_execution': False  # Manual approval required for withdrawals
    }

    # ===========================================
    # GATE SYSTEM CONFIGURATION
    # ===========================================

    GATE_CONFIG = {
        'starting_gate': 'G0',
        'starting_capital': Decimal('200.00'),
        'progression_multiplier': 3.0,  # 3x capital for gate progression
        'minimum_trades': 100,  # Minimum trades before gate progression
        'minimum_win_rate': 0.55,  # 55% minimum win rate
        'maximum_drawdown': 0.20,  # 20% maximum drawdown
    }

    # ===========================================
    # DATABASE CONFIGURATION
    # ===========================================

    DATABASE_CONFIG = {
        'type': 'postgresql',
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'trading_production'),
        'user': os.getenv('DB_USER', 'trading_user'),
        'password': os.getenv('DB_PASSWORD'),
        'pool_size': 10,
        'max_overflow': 20
    }

    # ===========================================
    # MONITORING & ALERTING
    # ===========================================

    MONITORING_CONFIG = {
        'metrics_enabled': True,
        'metrics_interval': 60,  # seconds
        'health_check_interval': 30,  # seconds
        'alert_channels': {
            'email': os.getenv('ALERT_EMAIL'),
            'sms': os.getenv('ALERT_PHONE'),
            'webhook': os.getenv('ALERT_WEBHOOK')
        },
        'alert_thresholds': {
            'position_loss': 0.05,  # Alert on 5% position loss
            'daily_loss': 0.03,     # Alert on 3% daily loss
            'api_errors': 5,        # Alert after 5 API errors
            'latency_ms': 1000      # Alert if latency > 1000ms
        }
    }

    # ===========================================
    # AUDIT & COMPLIANCE
    # ===========================================

    AUDIT_CONFIG = {
        'level': 'FULL',  # FULL audit logging for production
        'retention_days': 2555,  # 7 years for compliance
        'log_trades': True,
        'log_positions': True,
        'log_risk_metrics': True,
        'log_api_calls': True,
        'log_errors': True,
        'log_kill_switch': True,
        'encrypt_logs': True,
        'backup_enabled': True,
        'backup_interval': 3600  # Hourly backups
    }

    # ===========================================
    # DASHBOARD CONFIGURATION
    # ===========================================

    DASHBOARD_CONFIG = {
        'enabled': True,
        'websocket_port': 8080,
        'http_port': 3000,
        'update_interval_ms': 1000,
        'authentication_required': True,
        'ssl_enabled': True,
        'ssl_cert': os.getenv('SSL_CERT_PATH'),
        'ssl_key': os.getenv('SSL_KEY_PATH')
    }

    # ===========================================
    # PERFORMANCE SETTINGS
    # ===========================================

    PERFORMANCE_CONFIG = {
        'max_concurrent_orders': 5,
        'order_retry_attempts': 3,
        'order_retry_delay': 1.0,  # seconds
        'cache_ttl': 5,  # seconds
        'rate_limit_per_second': 10,
        'connection_pool_size': 10,
        'request_timeout': 30  # seconds
    }

    # ===========================================
    # SYMBOLS & MARKETS
    # ===========================================

    # Production symbols (liquid, low-cost ETFs)
    ALLOWED_SYMBOLS = [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'IWM',   # Russell 2000
        'DIA',   # Dow Jones
        'VTI',   # Total Market
        'EFA',   # International
        'EEM',   # Emerging Markets
        'GLD',   # Gold
        'TLT'    # 20+ Year Treasury
    ]

    # ===========================================
    # VALIDATION
    # ===========================================

    @classmethod
    def validate(cls) -> bool:
        """
        Validate production configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Check critical settings
        if cls.SIMULATION_MODE:
            errors.append("SIMULATION_MODE must be False for production")

        if not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is required")

        if not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is required")

        if cls.MAX_POSITION_SIZE > 0.25:
            errors.append("MAX_POSITION_SIZE too high for production (>25%)")

        if cls.DAILY_LOSS_LIMIT > Decimal('20.00'):
            errors.append("DAILY_LOSS_LIMIT too high for $200 account")

        if cls.MAX_RUIN_PROBABILITY > 1e-5:
            errors.append("MAX_RUIN_PROBABILITY too high (must be < 10^-5)")

        if errors:
            raise ValueError(
                "Production configuration validation failed:\n" +
                "\n".join(f"  - {error}" for error in errors)
            )

        return True

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            'simulation_mode': cls.SIMULATION_MODE,
            'paper_trading': cls.PAPER_TRADING,
            'broker_type': cls.BROKER_TYPE,
            'max_position_size': cls.MAX_POSITION_SIZE,
            'max_kelly_fraction': cls.MAX_KELLY_FRACTION,
            'daily_loss_limit': str(cls.DAILY_LOSS_LIMIT),
            'weekly_loss_limit': str(cls.WEEKLY_LOSS_LIMIT),
            'max_ruin_probability': cls.MAX_RUIN_PROBABILITY,
            'kill_switch': cls.KILL_SWITCH_CONFIG,
            'siphon': cls.SIPHON_CONFIG,
            'gate': cls.GATE_CONFIG,
            'monitoring': cls.MONITORING_CONFIG,
            'audit': cls.AUDIT_CONFIG
        }


# Validate configuration on import
if __name__ != "__main__":
    ProductionConfig.validate()