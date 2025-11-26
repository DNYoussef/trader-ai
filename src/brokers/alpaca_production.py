"""
ISS-012: DEPRECATED - Production features merged into alpaca_adapter.py

This file exists for backwards compatibility only.
Use AlpacaAdapter from alpaca_adapter.py instead.

Migration:
    # Old (deprecated)
    from src.brokers.alpaca_production import AlpacaProductionAdapter

    # New (recommended)
    from src.brokers.alpaca_adapter import AlpacaAdapter
"""

import warnings
from .alpaca_adapter import AlpacaAdapter

# Re-export with deprecation warning
class AlpacaProductionAdapter(AlpacaAdapter):
    """
    DEPRECATED: Use AlpacaAdapter directly.

    This class exists for backwards compatibility only.
    All production features have been merged into AlpacaAdapter.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AlpacaProductionAdapter is deprecated. "
            "Use AlpacaAdapter from alpaca_adapter.py instead. "
            "All production features have been merged.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


__all__ = ['AlpacaProductionAdapter']
