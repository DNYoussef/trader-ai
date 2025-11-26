"""
Financial Math Utilities using Decimal for precision.

ISS-009 FIX: Provides Decimal-based utilities for all financial calculations
to prevent floating-point precision errors in trading operations.

Usage:
    from src.utils.money import Money, as_money, round_money

    # Create money values
    price = Money("123.45")
    quantity = Money(100)

    # Arithmetic (always returns Money)
    total = price * quantity  # Money("12345.00")

    # Convert from float (use sparingly)
    legacy_value = as_money(123.456789)  # Money("123.46")

    # Round to specific places
    rounded = round_money(Money("123.456"), places=2)  # Money("123.46")
"""

from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation, getcontext
from typing import Union
import logging

logger = logging.getLogger(__name__)

# Set precision high enough for financial calculations
getcontext().prec = 28

# Type alias for values that can become Money
MoneyLike = Union['Money', Decimal, str, int, float]


class Money:
    """
    Immutable Decimal-based money value with safe arithmetic.

    Uses banker's rounding (ROUND_HALF_EVEN) which is the standard
    for financial calculations to minimize cumulative rounding errors.

    Attributes:
        value: The underlying Decimal value
        currency: Currency code (default: USD)
    """

    __slots__ = ('_value', '_currency')

    # Default number of decimal places for different operations
    DEFAULT_PRICE_PLACES = 2  # Prices typically 2 decimal places
    DEFAULT_QTY_PLACES = 6    # Alpaca supports 6 decimal places for fractional shares
    DEFAULT_PCT_PLACES = 4    # Percentages with 4 decimal places (0.01%)

    def __init__(self, value: MoneyLike, currency: str = 'USD'):
        """
        Create a Money value.

        Args:
            value: Numeric value (Decimal, str, int preferred; float accepted with warning)
            currency: Currency code (default: USD)

        Raises:
            ValueError: If value cannot be converted to Decimal
        """
        if isinstance(value, Money):
            self._value = value._value
            self._currency = value._currency
            return

        if isinstance(value, float):
            # Log warning for float usage in production
            logger.debug(f"Converting float {value} to Money - prefer str or Decimal")
            # Convert via string to avoid float representation issues
            value = str(value)

        try:
            self._value = Decimal(value)
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Cannot convert {value!r} to Money: {e}") from e

        self._currency = currency.upper()

    @property
    def value(self) -> Decimal:
        """Get the underlying Decimal value."""
        return self._value

    @property
    def currency(self) -> str:
        """Get the currency code."""
        return self._currency

    def round(self, places: int = 2) -> 'Money':
        """
        Round to specified decimal places using banker's rounding.

        Args:
            places: Number of decimal places (default: 2)

        Returns:
            New Money with rounded value
        """
        quantize_str = '0.' + '0' * places if places > 0 else '1'
        rounded = self._value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_EVEN)
        return Money(rounded, self._currency)

    def round_price(self) -> 'Money':
        """Round to standard price precision (2 decimal places)."""
        return self.round(self.DEFAULT_PRICE_PLACES)

    def round_qty(self) -> 'Money':
        """Round to quantity precision (6 decimal places for fractional shares)."""
        return self.round(self.DEFAULT_QTY_PLACES)

    def round_pct(self) -> 'Money':
        """Round to percentage precision (4 decimal places)."""
        return self.round(self.DEFAULT_PCT_PLACES)

    def __add__(self, other: MoneyLike) -> 'Money':
        """Add two Money values."""
        other_money = self._ensure_money(other)
        self._check_currency(other_money)
        return Money(self._value + other_money._value, self._currency)

    def __radd__(self, other: MoneyLike) -> 'Money':
        """Support sum() and other right-side additions."""
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: MoneyLike) -> 'Money':
        """Subtract two Money values."""
        other_money = self._ensure_money(other)
        self._check_currency(other_money)
        return Money(self._value - other_money._value, self._currency)

    def __mul__(self, other: MoneyLike) -> 'Money':
        """Multiply Money by a scalar or another Money."""
        if isinstance(other, Money):
            return Money(self._value * other._value, self._currency)
        other_decimal = self._to_decimal(other)
        return Money(self._value * other_decimal, self._currency)

    def __rmul__(self, other: MoneyLike) -> 'Money':
        """Support scalar * Money."""
        return self.__mul__(other)

    def __truediv__(self, other: MoneyLike) -> 'Money':
        """Divide Money by a scalar."""
        if isinstance(other, Money):
            if other._value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Money(self._value / other._value, self._currency)
        other_decimal = self._to_decimal(other)
        if other_decimal == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Money(self._value / other_decimal, self._currency)

    def __neg__(self) -> 'Money':
        """Negate the value."""
        return Money(-self._value, self._currency)

    def __abs__(self) -> 'Money':
        """Absolute value."""
        return Money(abs(self._value), self._currency)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, Money):
            return self._value == other._value and self._currency == other._currency
        if isinstance(other, (Decimal, int, str)):
            return self._value == Decimal(str(other))
        if isinstance(other, float):
            return self._value == Decimal(str(other))
        return False

    def __lt__(self, other: MoneyLike) -> bool:
        """Less than comparison."""
        other_money = self._ensure_money(other)
        return self._value < other_money._value

    def __le__(self, other: MoneyLike) -> bool:
        """Less than or equal comparison."""
        other_money = self._ensure_money(other)
        return self._value <= other_money._value

    def __gt__(self, other: MoneyLike) -> bool:
        """Greater than comparison."""
        other_money = self._ensure_money(other)
        return self._value > other_money._value

    def __ge__(self, other: MoneyLike) -> bool:
        """Greater than or equal comparison."""
        other_money = self._ensure_money(other)
        return self._value >= other_money._value

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash((self._value, self._currency))

    def __repr__(self) -> str:
        """Debug representation."""
        return f"Money({self._value!r}, {self._currency!r})"

    def __str__(self) -> str:
        """String representation with currency symbol."""
        if self._currency == 'USD':
            return f"${self._value:,.2f}"
        return f"{self._value:,.2f} {self._currency}"

    def __float__(self) -> float:
        """Convert to float (use sparingly, logs warning)."""
        logger.debug("Converting Money to float - precision may be lost")
        return float(self._value)

    def __int__(self) -> int:
        """Convert to int (truncates)."""
        return int(self._value)

    def to_decimal(self) -> Decimal:
        """Get the Decimal value (preferred over float)."""
        return self._value

    def to_float(self) -> float:
        """Explicitly convert to float when needed for external APIs."""
        return float(self._value)

    def _ensure_money(self, other: MoneyLike) -> 'Money':
        """Convert value to Money if needed."""
        if isinstance(other, Money):
            return other
        return Money(other, self._currency)

    def _to_decimal(self, value: MoneyLike) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Money):
            return value._value
        if isinstance(value, Decimal):
            return value
        if isinstance(value, float):
            return Decimal(str(value))
        return Decimal(value)

    def _check_currency(self, other: 'Money') -> None:
        """Check currency compatibility."""
        if self._currency != other._currency:
            raise ValueError(
                f"Currency mismatch: {self._currency} vs {other._currency}"
            )

    @classmethod
    def zero(cls, currency: str = 'USD') -> 'Money':
        """Create a zero Money value."""
        return cls(0, currency)

    @classmethod
    def from_cents(cls, cents: int, currency: str = 'USD') -> 'Money':
        """Create Money from cents (integer)."""
        return cls(Decimal(cents) / 100, currency)


def as_money(value: MoneyLike, currency: str = 'USD') -> Money:
    """
    Convert any numeric value to Money.

    Convenience function for converting legacy float values.

    Args:
        value: Value to convert
        currency: Currency code (default: USD)

    Returns:
        Money instance
    """
    return Money(value, currency)


def round_money(value: Money, places: int = 2) -> Money:
    """
    Round a Money value to specified decimal places.

    Args:
        value: Money value to round
        places: Number of decimal places

    Returns:
        Rounded Money value
    """
    return value.round(places)


# Convenience constants
ZERO = Money.zero()
