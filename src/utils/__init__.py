"""
ISS-023: Trader-AI utility modules with Decimal-based financial calculations.

IMPORTANT: All financial calculations should use Decimal or Money class to avoid
floating-point precision errors. Never use float for:
- Prices
- Quantities
- Portfolio values
- P&L calculations
- Percentages (use Decimal('0.05') not 0.05)

Usage:
    from src.utils import Money, as_money, ZERO

    # Create money values (prefer str or Decimal, not float)
    price = Money("123.45")           # From string (preferred)
    qty = Money(Decimal("100.5"))     # From Decimal (also good)
    legacy = as_money(123.45)         # From float (use sparingly)

    # Arithmetic
    total = price * qty
    profit = total - Money("10000")

    # Comparisons
    if total > ZERO:
        print(f"Profit: {total}")

    # Convert to Decimal for Alpaca API
    api_value = total.to_decimal()
"""

from .money import Money, as_money, round_money, ZERO

__all__ = [
    'Money',
    'as_money',
    'round_money',
    'ZERO',
]
