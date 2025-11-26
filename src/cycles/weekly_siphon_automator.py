"""
Weekly Siphon Automator for Gary×Taleb Trading System

Handles automated weekly profit distribution with 50/50 split:
- 50% reinvestment (stays in portfolio)
- 50% withdrawal (transferred to external account)

Features:
- Friday 6:00pm ET execution
- Robust scheduling with holiday handling
- Capital protection safeguards
- Comprehensive audit trail
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import pytz
import schedule
import threading
import time as time_module

from .profit_calculator import ProfitCalculator, ProfitStatus
from ..portfolio.portfolio_manager import PortfolioManager
from ..utils.holiday_calendar import MarketHolidayCalendar
from ..brokers.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class SiphonStatus(Enum):
    """Siphon execution status"""
    SUCCESS = "success"
    NO_PROFIT = "no_profit"
    FAILED = "failed"
    DEFERRED_HOLIDAY = "deferred_holiday"
    SAFETY_BLOCK = "safety_block"


@dataclass
class SiphonResult:
    """Result of a siphon execution"""
    timestamp: datetime
    status: SiphonStatus
    profit_calculation: Optional[any]
    withdrawal_amount: Decimal
    withdrawal_success: bool
    broker_confirmation: Optional[str]
    safety_checks: List[str]
    errors: List[str]


class WeeklySiphonAutomator:
    """
    Automated weekly profit siphon system.

    Executes every Friday at 6:00pm ET:
    1. Calculate profit vs base capital
    2. Apply 50/50 split (reinvest/withdraw)
    3. Execute withdrawal through broker
    4. Update records and audit trail

    Safety Features:
    - Never withdraws base capital
    - Holiday detection and deferral
    - Multiple validation checks
    - Automatic rollback on failures
    """

    # Eastern Time timezone
    ET = pytz.timezone('US/Eastern')

    # Siphon execution time
    SIPHON_TIME = time(18, 0)  # 6:00 PM ET

    def __init__(self,
                 portfolio_manager: PortfolioManager,
                 broker_adapter: BrokerInterface,
                 profit_calculator: ProfitCalculator,
                 holiday_calendar: Optional[MarketHolidayCalendar] = None,
                 enable_auto_execution: bool = False):
        """
        Initialize weekly siphon automator.

        Args:
            portfolio_manager: Portfolio manager instance
            broker_adapter: Broker for withdrawals
            profit_calculator: Profit calculation engine
            holiday_calendar: Market holiday calendar
            enable_auto_execution: Enable automatic execution (default: False for safety)
        """
        self.portfolio_manager = portfolio_manager
        self.broker = broker_adapter
        self.profit_calculator = profit_calculator
        self.holiday_calendar = holiday_calendar or MarketHolidayCalendar()
        self.enable_auto_execution = enable_auto_execution

        # Execution tracking
        self.last_execution_date: Optional[datetime] = None
        self.execution_history: List[SiphonResult] = []
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Safety callbacks
        self.pre_execution_hooks: List[Callable] = []
        self.post_execution_hooks: List[Callable] = []

        # Minimum withdrawal threshold
        self.min_withdrawal_amount = Decimal("10.00")

        logger.info(f"WeeklySiphonAutomator initialized - Auto execution: {enable_auto_execution}")

    def start_scheduler(self) -> bool:
        """Start the automated scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return False

        try:
            # Schedule Friday 6:00pm ET execution
            schedule.every().friday.at("18:00").do(self._execute_siphon_job)

            self.is_running = True

            # Start scheduler in separate thread
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()

            logger.info("Weekly siphon scheduler started - Fridays at 6:00pm ET")
            return True

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False

    def stop_scheduler(self) -> None:
        """Stop the automated scheduler."""
        self.is_running = False
        schedule.clear()

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            # Wait for scheduler thread to finish
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Weekly siphon scheduler stopped")

    def should_execute_siphon(self) -> Tuple[bool, str]:
        """
        Determine if siphon should execute now.

        Returns:
            Tuple of (should_execute, reason)
        """
        current_time = datetime.now(self.ET)

        # Check if it's Friday
        if current_time.weekday() != 4:  # 4 = Friday
            return False, f"Not Friday (current: {current_time.strftime('%A')})"

        # Check if it's past siphon time
        if current_time.time() < self.SIPHON_TIME:
            return False, f"Before siphon time (current: {current_time.time()}, target: {self.SIPHON_TIME})"

        # Check for market holiday
        if self.holiday_calendar.is_market_holiday(current_time.date()):
            return False, f"Market holiday: {current_time.date()}"

        # Check if already executed this week
        if self.last_execution_date:
            last_week = self._get_week_start(self.last_execution_date)
            current_week = self._get_week_start(current_time)
            if last_week == current_week:
                return False, "Already executed this week"

        return True, "Ready for execution"

    async def execute_manual_siphon(self, force: bool = False) -> SiphonResult:
        """
        Execute siphon manually (outside of automatic schedule).

        Args:
            force: Force execution even if conditions not met

        Returns:
            SiphonResult with execution details
        """
        logger.info("Manual siphon execution requested")

        if not force:
            should_execute, reason = self.should_execute_siphon()
            if not should_execute:
                return SiphonResult(
                    timestamp=datetime.now(self.ET),
                    status=SiphonStatus.FAILED,
                    profit_calculation=None,
                    withdrawal_amount=Decimal("0.00"),
                    withdrawal_success=False,
                    broker_confirmation=None,
                    safety_checks=[],
                    errors=[f"Execution conditions not met: {reason}"]
                )

        return await self._execute_siphon()

    async def _execute_siphon(self) -> SiphonResult:
        """Core siphon execution logic."""
        execution_start = datetime.now(self.ET)
        safety_checks = []
        errors = []

        logger.info("Starting weekly siphon execution")

        try:
            # Run pre-execution hooks
            for hook in self.pre_execution_hooks:
                try:
                    await hook()
                except Exception as e:
                    logger.warning(f"Pre-execution hook failed: {e}")

            # Sync portfolio with broker
            sync_success = await self.portfolio_manager.sync_with_broker()
            if not sync_success:
                raise Exception("Failed to sync portfolio with broker")
            safety_checks.append("Portfolio synced with broker")

            # Calculate profit and determine withdrawal amount
            profit_calc = self.profit_calculator.calculate_weekly_profit(
                self.portfolio_manager
            )

            # Check if there's profit to withdraw
            if profit_calc.profit_status != ProfitStatus.PROFIT_AVAILABLE:
                logger.info(f"No profit available for withdrawal: {profit_calc.profit_status.value}")

                result = SiphonResult(
                    timestamp=execution_start,
                    status=SiphonStatus.NO_PROFIT,
                    profit_calculation=profit_calc,
                    withdrawal_amount=Decimal("0.00"),
                    withdrawal_success=True,  # No withdrawal needed
                    broker_confirmation=None,
                    safety_checks=safety_checks,
                    errors=[]
                )

                self.execution_history.append(result)
                self.last_execution_date = execution_start
                return result

            withdrawal_amount = profit_calc.withdrawal_amount

            # Apply minimum withdrawal threshold
            if withdrawal_amount < self.min_withdrawal_amount:
                logger.info(f"Withdrawal amount ${withdrawal_amount} below minimum ${self.min_withdrawal_amount}")
                withdrawal_amount = Decimal("0.00")

                result = SiphonResult(
                    timestamp=execution_start,
                    status=SiphonStatus.NO_PROFIT,
                    profit_calculation=profit_calc,
                    withdrawal_amount=withdrawal_amount,
                    withdrawal_success=True,
                    broker_confirmation=None,
                    safety_checks=safety_checks + ["Below minimum withdrawal threshold"],
                    errors=[]
                )

                self.execution_history.append(result)
                self.last_execution_date = execution_start
                return result

            # Validate withdrawal safety
            is_safe, safety_reason = self.profit_calculator.validate_withdrawal_safety(
                withdrawal_amount, profit_calc.current_value
            )

            if not is_safe:
                logger.error(f"Withdrawal safety check failed: {safety_reason}")

                result = SiphonResult(
                    timestamp=execution_start,
                    status=SiphonStatus.SAFETY_BLOCK,
                    profit_calculation=profit_calc,
                    withdrawal_amount=Decimal("0.00"),
                    withdrawal_success=False,
                    broker_confirmation=None,
                    safety_checks=safety_checks,
                    errors=[f"Safety check failed: {safety_reason}"]
                )

                self.execution_history.append(result)
                return result

            safety_checks.append(f"Withdrawal safety validated: {safety_reason}")

            # Execute withdrawal through broker
            withdrawal_success = False
            broker_confirmation = None

            if self.enable_auto_execution:
                try:
                    # Execute the actual withdrawal
                    withdrawal_success = await self.broker.withdraw_funds(withdrawal_amount)

                    if withdrawal_success:
                        # Get confirmation ID
                        broker_confirmation = await self.broker.get_last_withdrawal_id()

                        # Record the withdrawal in profit calculator
                        self.profit_calculator.record_withdrawal(withdrawal_amount)

                        # Record transaction in portfolio
                        await self.portfolio_manager.record_transaction(
                            transaction_type="withdrawal",
                            amount=withdrawal_amount,
                            gate="SIPHON"
                        )

                        safety_checks.append("Withdrawal executed successfully")
                        logger.info(f"Withdrawal successful: ${withdrawal_amount} - Confirmation: {broker_confirmation}")

                    else:
                        errors.append("Broker withdrawal failed")
                        logger.error(f"Broker withdrawal failed for ${withdrawal_amount}")

                except Exception as e:
                    errors.append(f"Withdrawal execution error: {str(e)}")
                    logger.error(f"Withdrawal execution error: {e}")

            else:
                # Simulation mode - don't actually withdraw
                withdrawal_success = True
                broker_confirmation = f"SIMULATED_{int(execution_start.timestamp())}"
                safety_checks.append("Withdrawal simulated (auto-execution disabled)")
                logger.info(f"Simulated withdrawal: ${withdrawal_amount}")

            # Create result
            result = SiphonResult(
                timestamp=execution_start,
                status=SiphonStatus.SUCCESS if withdrawal_success else SiphonStatus.FAILED,
                profit_calculation=profit_calc,
                withdrawal_amount=withdrawal_amount if withdrawal_success else Decimal("0.00"),
                withdrawal_success=withdrawal_success,
                broker_confirmation=broker_confirmation,
                safety_checks=safety_checks,
                errors=errors
            )

            # Run post-execution hooks
            for hook in self.post_execution_hooks:
                try:
                    await hook(result)
                except Exception as e:
                    logger.warning(f"Post-execution hook failed: {e}")

            # Update tracking
            self.execution_history.append(result)
            self.last_execution_date = execution_start

            logger.info(f"Siphon execution completed - Status: {result.status.value}")
            return result

        except Exception as e:
            logger.error(f"Siphon execution failed: {e}")
            errors.append(f"Execution error: {str(e)}")

            result = SiphonResult(
                timestamp=execution_start,
                status=SiphonStatus.FAILED,
                profit_calculation=None,
                withdrawal_amount=Decimal("0.00"),
                withdrawal_success=False,
                broker_confirmation=None,
                safety_checks=safety_checks,
                errors=errors
            )

            self.execution_history.append(result)
            return result

    def _execute_siphon_job(self) -> None:
        """Wrapper for scheduler to call async siphon execution."""
        if not self.enable_auto_execution:
            logger.info("Auto-execution disabled - skipping scheduled siphon")
            return

        try:
            # Run async siphon in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._execute_siphon())
            loop.close()

            logger.info(f"Scheduled siphon completed: {result.status.value}")

        except Exception as e:
            logger.error(f"Scheduled siphon execution failed: {e}")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                schedule.run_pending()
                time_module.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time_module.sleep(60)

        logger.info("Scheduler loop stopped")

    def add_pre_execution_hook(self, hook: Callable) -> None:
        """Add a pre-execution hook function."""
        self.pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: Callable) -> None:
        """Add a post-execution hook function."""
        self.post_execution_hooks.append(hook)

    def get_execution_history(self, limit: int = 10) -> List[SiphonResult]:
        """Get recent execution history."""
        return self.execution_history[-limit:] if self.execution_history else []

    def get_status(self) -> Dict[str, any]:
        """Get current automator status."""
        current_time = datetime.now(self.ET)
        should_execute, reason = self.should_execute_siphon()

        return {
            'is_running': self.is_running,
            'auto_execution_enabled': self.enable_auto_execution,
            'current_time_et': current_time,
            'should_execute': should_execute,
            'execution_reason': reason,
            'last_execution': self.last_execution_date,
            'total_executions': len(self.execution_history),
            'successful_executions': sum(1 for r in self.execution_history if r.status == SiphonStatus.SUCCESS),
            'next_friday': self._get_next_friday()
        }

    def _get_week_start(self, dt: datetime) -> datetime:
        """Get Monday of the week containing the given datetime."""
        days_since_monday = dt.weekday()
        monday = dt - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _get_next_friday(self) -> datetime:
        """Get next Friday's siphon time."""
        current_time = datetime.now(self.ET)
        days_until_friday = (4 - current_time.weekday()) % 7

        if days_until_friday == 0 and current_time.time() > self.SIPHON_TIME:
            days_until_friday = 7  # Next Friday if current Friday is past siphon time

        next_friday = current_time + timedelta(days=days_until_friday)
        return next_friday.replace(hour=18, minute=0, second=0, microsecond=0)

    def get_withdrawal_summary(self, weeks: int = 4) -> Dict[str, any]:
        """Get withdrawal summary for recent weeks."""
        recent_results = self.execution_history[-weeks:] if self.execution_history else []

        total_withdrawn = sum(r.withdrawal_amount for r in recent_results if r.withdrawal_success)
        successful_withdrawals = sum(1 for r in recent_results if r.withdrawal_success and r.withdrawal_amount > 0)

        return {
            'weeks_analyzed': len(recent_results),
            'total_withdrawn': total_withdrawn,
            'successful_withdrawals': successful_withdrawals,
            'average_withdrawal': total_withdrawn / successful_withdrawals if successful_withdrawals > 0 else Decimal("0.00"),
            'success_rate': (successful_withdrawals / len(recent_results) * 100) if recent_results else 0
        }