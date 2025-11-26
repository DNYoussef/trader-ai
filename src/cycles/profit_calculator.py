"""
Profit Calculator for Weekly Siphon Automation

Calculates profits vs base capital and determines 50/50 split amounts
with strict capital protection safeguards.
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProfitStatus(Enum):
    """Profit calculation status"""
    PROFIT_AVAILABLE = "profit_available"
    NO_PROFIT = "no_profit"
    LOSS = "loss"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ProfitCalculation:
    """Profit calculation result"""
    base_capital: Decimal
    current_value: Decimal
    total_profit: Decimal
    profit_status: ProfitStatus
    reinvestment_amount: Decimal
    withdrawal_amount: Decimal
    remaining_capital: Decimal
    calculation_timestamp: datetime
    weekly_return_percent: Decimal
    cumulative_return_percent: Decimal


class ProfitCalculator:
    """
    Calculates profits and determines 50/50 split for weekly siphon.

    Key Features:
    - Strict base capital protection (never withdraw initial investment)
    - Accurate profit calculation excluding deposits/withdrawals
    - 50/50 split: reinvest 50%, withdraw 50% of profits only
    - Historical tracking for audit trail
    """

    def __init__(self, initial_capital: Decimal):
        """
        Initialize profit calculator.

        Args:
            initial_capital: The base capital that must never be withdrawn
        """
        self.base_capital = initial_capital
        self.inception_date = datetime.now(timezone.utc).date()
        self.profit_history: List[ProfitCalculation] = []
        self.total_withdrawals = Decimal("0.00")
        self.total_deposits_beyond_base = Decimal("0.00")

        logger.info(f"ProfitCalculator initialized with base capital: ${initial_capital}")

    def calculate_weekly_profit(self,
                              portfolio_manager,
                              week_start: Optional[date] = None) -> ProfitCalculation:
        """
        Calculate profit for current week and determine 50/50 split.

        Args:
            portfolio_manager: Portfolio manager with current positions
            week_start: Week start date (defaults to current week)

        Returns:
            ProfitCalculation with split amounts
        """
        try:
            if week_start is None:
                week_start = self._get_current_week_start()

            # Get current portfolio value
            current_value = portfolio_manager.get_total_portfolio_value()

            # Calculate total profit vs base capital and additional deposits
            adjusted_base = self.base_capital + self.total_deposits_beyond_base
            total_profit = current_value - adjusted_base

            # Calculate return percentages
            weekly_return = self._calculate_weekly_return(portfolio_manager, week_start)
            cumulative_return = (total_profit / adjusted_base * Decimal("100")) if adjusted_base > 0 else Decimal("0.00")

            # Determine profit status
            if total_profit <= Decimal("0.00"):
                profit_status = ProfitStatus.NO_PROFIT if total_profit == Decimal("0.00") else ProfitStatus.LOSS
                reinvestment_amount = Decimal("0.00")
                withdrawal_amount = Decimal("0.00")
                remaining_capital = current_value

                logger.info(f"No profit available - Total profit: ${total_profit}")

            else:
                profit_status = ProfitStatus.PROFIT_AVAILABLE

                # 50/50 split of profits only
                reinvestment_amount = (total_profit / Decimal("2.0")).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP
                )
                withdrawal_amount = total_profit - reinvestment_amount  # Ensures exact split
                remaining_capital = current_value - withdrawal_amount

                # Validate that we never withdraw base capital
                if remaining_capital < self.base_capital:
                    # Adjust withdrawal to protect base capital
                    max_safe_withdrawal = current_value - self.base_capital
                    withdrawal_amount = max_safe_withdrawal
                    reinvestment_amount = total_profit - withdrawal_amount
                    remaining_capital = current_value - withdrawal_amount

                    logger.warning(f"Adjusted withdrawal to protect base capital: ${withdrawal_amount}")

                logger.info(f"Profit calculation - Total: ${total_profit}, Reinvest: ${reinvestment_amount}, Withdraw: ${withdrawal_amount}")

            # Create calculation result
            calculation = ProfitCalculation(
                base_capital=self.base_capital,
                current_value=current_value,
                total_profit=total_profit,
                profit_status=profit_status,
                reinvestment_amount=reinvestment_amount,
                withdrawal_amount=withdrawal_amount,
                remaining_capital=remaining_capital,
                calculation_timestamp=datetime.now(timezone.utc),
                weekly_return_percent=weekly_return,
                cumulative_return_percent=cumulative_return
            )

            # Store in history
            self.profit_history.append(calculation)

            return calculation

        except Exception as e:
            logger.error(f"Failed to calculate weekly profit: {e}")
            return ProfitCalculation(
                base_capital=self.base_capital,
                current_value=Decimal("0.00"),
                total_profit=Decimal("0.00"),
                profit_status=ProfitStatus.INSUFFICIENT_DATA,
                reinvestment_amount=Decimal("0.00"),
                withdrawal_amount=Decimal("0.00"),
                remaining_capital=Decimal("0.00"),
                calculation_timestamp=datetime.now(timezone.utc),
                weekly_return_percent=Decimal("0.00"),
                cumulative_return_percent=Decimal("0.00")
            )

    def validate_withdrawal_safety(self, withdrawal_amount: Decimal, current_value: Decimal) -> Tuple[bool, str]:
        """
        Validate that a withdrawal amount is safe and won't touch base capital.

        Args:
            withdrawal_amount: Amount to withdraw
            current_value: Current portfolio value

        Returns:
            Tuple of (is_safe, reason)
        """
        if withdrawal_amount <= Decimal("0.00"):
            return True, "No withdrawal requested"

        remaining_after_withdrawal = current_value - withdrawal_amount

        if remaining_after_withdrawal < self.base_capital:
            shortage = self.base_capital - remaining_after_withdrawal
            return False, f"Withdrawal would breach base capital by ${shortage}"

        # Additional safety margin (5% of base capital)
        safety_margin = self.base_capital * Decimal("0.05")
        if remaining_after_withdrawal < (self.base_capital + safety_margin):
            return False, f"Withdrawal would leave insufficient safety margin (need ${safety_margin})"

        return True, "Withdrawal is safe"

    def record_withdrawal(self, amount: Decimal) -> None:
        """Record a completed withdrawal."""
        self.total_withdrawals += amount
        logger.info(f"Recorded withdrawal: ${amount} (Total withdrawals: ${self.total_withdrawals})")

    def record_additional_deposit(self, amount: Decimal) -> None:
        """Record additional deposits beyond base capital."""
        self.total_deposits_beyond_base += amount
        logger.info(f"Recorded additional deposit: ${amount} (Total beyond base: ${self.total_deposits_beyond_base})")

    def get_profit_summary(self, weeks: int = 4) -> Dict[str, any]:
        """
        Get profit summary for recent weeks.

        Args:
            weeks: Number of recent weeks to summarize

        Returns:
            Dictionary with profit summary
        """
        recent_calculations = self.profit_history[-weeks:] if self.profit_history else []

        if not recent_calculations:
            return {
                'weeks_analyzed': 0,
                'total_profit': Decimal("0.00"),
                'total_withdrawals': self.total_withdrawals,
                'average_weekly_return': Decimal("0.00"),
                'successful_siphons': 0
            }

        total_profit = sum(calc.total_profit for calc in recent_calculations if calc.total_profit > 0)
        total_withdrawals = sum(calc.withdrawal_amount for calc in recent_calculations)
        average_return = sum(calc.weekly_return_percent for calc in recent_calculations) / len(recent_calculations)
        successful_siphons = sum(1 for calc in recent_calculations if calc.withdrawal_amount > 0)

        return {
            'weeks_analyzed': len(recent_calculations),
            'total_profit': total_profit,
            'total_withdrawals': total_withdrawals,
            'average_weekly_return': average_return,
            'successful_siphons': successful_siphons,
            'base_capital_protection': self.base_capital,
            'cumulative_withdrawals': self.total_withdrawals
        }

    def _calculate_weekly_return(self, portfolio_manager, week_start: date) -> Decimal:
        """Calculate weekly return percentage."""
        try:
            week_end = week_start + timedelta(days=7)

            start_nav = portfolio_manager.get_nav_at_date(week_start)
            current_nav = portfolio_manager.get_total_portfolio_value()

            # Get cash flows during the week
            deposits = portfolio_manager.get_deposits_in_period(week_start, week_end)
            withdrawals = portfolio_manager.get_withdrawals_in_period(week_start, week_end)

            # Time-weighted return calculation
            net_cash_flow = deposits - withdrawals
            adjusted_start_nav = start_nav + net_cash_flow

            if adjusted_start_nav > 0:
                return ((current_nav - adjusted_start_nav) / adjusted_start_nav * Decimal("100"))

            return Decimal("0.00")

        except Exception as e:
            logger.warning(f"Could not calculate weekly return: {e}")
            return Decimal("0.00")

    def _get_current_week_start(self) -> date:
        """Get Monday of current week."""
        today = datetime.now(timezone.utc).date()
        days_since_monday = today.weekday()
        return today - timedelta(days=days_since_monday)

    def get_capital_protection_status(self, current_value: Decimal) -> Dict[str, any]:
        """Get current capital protection status."""
        return {
            'base_capital': self.base_capital,
            'current_value': current_value,
            'buffer_amount': current_value - self.base_capital if current_value > self.base_capital else Decimal("0.00"),
            'buffer_percent': ((current_value - self.base_capital) / self.base_capital * Decimal("100")) if self.base_capital > 0 and current_value > self.base_capital else Decimal("0.00"),
            'is_protected': current_value >= self.base_capital,
            'total_withdrawals_to_date': self.total_withdrawals
        }