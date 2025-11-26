"""
Custom exceptions for the Trader-AI system.

ISS-013 FIX: Provides explicit exceptions instead of silent fallback values
to ensure trading decisions are based on real data, not defaults.
"""


class TraderAIError(Exception):
    """Base exception for all Trader-AI errors."""
    pass


class MarketDataUnavailable(TraderAIError):
    """
    Raised when market data cannot be retrieved.

    This replaces silent fallback to default prices (e.g., $100)
    which could lead to incorrect trading decisions.

    Usage:
        if not self.broker.is_connected:
            raise MarketDataUnavailable(
                f"Cannot get price for {symbol}: broker disconnected"
            )
    """
    pass


class BrokerDisconnected(TraderAIError):
    """Raised when broker connection is lost during an operation."""
    pass


class InsufficientCapital(TraderAIError):
    """Raised when account has insufficient capital for a trade."""
    pass


class SafetyLimitReached(TraderAIError):
    """Raised when a safety limit (loss, position size, etc.) is reached."""
    pass


class KillSwitchTriggered(TraderAIError):
    """Raised when the kill switch has been activated."""
    pass


class ConfigurationError(TraderAIError):
    """Raised when configuration is invalid or missing required values."""
    pass


class CredentialsError(TraderAIError):
    """Raised when API credentials are missing or invalid."""
    pass
