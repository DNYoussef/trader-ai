"""
Weekly Buy/Siphon Cycle System

Handles weekly trading cycles with:
- Friday 4:10pm ET: Buy phase execution
- Friday 6:00pm ET: Siphon phase (50/50 split)
- Timezone conversion and market holiday handling
- Gate-specific allocation logic
- Weekly delta tracking and performance monitoring
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, Optional, List
from enum import Enum
import pytz
from dataclasses import dataclass

from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.trade_executor import TradeExecutor
from ..market.market_data import MarketDataProvider
from ..utils.holiday_calendar import MarketHolidayCalendar
from ..strategies.dpi_calculator import DistributionalPressureIndex, DPIWeeklyCycleIntegrator

logger = logging.getLogger(__name__)


class CyclePhase(Enum):
    """Weekly cycle phases"""
    BUY = "buy"
    SIPHON = "siphon"
    IDLE = "idle"


@dataclass
class GateAllocation:
    """Gate-specific allocation configuration"""
    ulty_pct: float
    amdy_pct: float
    iau_pct: float = 0.0
    vtip_pct: float = 0.0
    
    def validate(self):
        """Ensure allocations sum to 100%"""
        total = self.ulty_pct + self.amdy_pct + self.iau_pct + self.vtip_pct
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Allocation percentages must sum to 100%, got {total}%")


@dataclass
class WeeklyDelta:
    """Weekly performance tracking"""
    week_start: datetime
    week_end: datetime
    nav_start: float
    nav_end: float
    deposits: float
    withdrawals: float
    delta: float  # NAV change minus deposits/withdrawals
    delta_pct: float


class WeeklyCycle:
    """
    Manages weekly buy/siphon trading cycles
    
    Schedule:
    - Friday 4:10pm ET: Buy phase (purchase securities)
    - Friday 6:00pm ET: Siphon phase (50/50 split management)
    """
    
    # Eastern Time timezone
    ET = pytz.timezone('US/Eastern')
    
    # Weekly schedule times (in ET)
    BUY_TIME = time(16, 10)      # 4:10 PM ET
    SIPHON_TIME = time(18, 0)    # 6:00 PM ET
    
    # Gate allocations
    GATE_ALLOCATIONS = {
        'G0': GateAllocation(ulty_pct=70.0, amdy_pct=30.0),
        'G1': GateAllocation(ulty_pct=50.0, amdy_pct=20.0, iau_pct=15.0, vtip_pct=15.0)
    }
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        trade_executor: TradeExecutor,
        market_data: MarketDataProvider,
        holiday_calendar: Optional[MarketHolidayCalendar] = None,
        enable_dpi: bool = True
    ):
        self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor
        self.market_data = market_data
        self.holiday_calendar = holiday_calendar or MarketHolidayCalendar()

        # Initialize Gary's DPI system
        self.enable_dpi = enable_dpi
        if self.enable_dpi:
            self.dpi_calculator = DistributionalPressureIndex()
            self.dpi_integrator = DPIWeeklyCycleIntegrator(self.dpi_calculator)
            logger.info("DPI system enabled for weekly cycle")
        else:
            self.dpi_calculator = None
            self.dpi_integrator = None
            logger.info("DPI system disabled - using base allocations only")

        # Validate gate allocations
        for gate, allocation in self.GATE_ALLOCATIONS.items():
            allocation.validate()

        self._last_buy_execution = None
        self._last_siphon_execution = None
        self._weekly_deltas: List[WeeklyDelta] = []
    
    def get_current_et_time(self) -> datetime:
        """Get current time in Eastern Time"""
        return datetime.now(self.ET)
    
    def should_execute_buy(self) -> bool:
        """
        Determine if buy phase should execute
        
        Returns True if:
        - Current time is Friday 4:10pm ET or later
        - Market is open (not a holiday)
        - Haven't executed buy phase this week
        """
        current_time = self.get_current_et_time()
        
        # Check if it's Friday
        if current_time.weekday() != 4:  # 4 = Friday
            return False
        
        # Check if it's past buy time
        if current_time.time() < self.BUY_TIME:
            return False
        
        # Check for market holiday
        if self.holiday_calendar.is_market_holiday(current_time.date()):
            logger.info(f"Market holiday detected: {current_time.date()}")
            return False
        
        # Check if already executed this week
        if self._last_buy_execution:
            last_week_start = self._get_week_start(self._last_buy_execution)
            current_week_start = self._get_week_start(current_time)
            if last_week_start == current_week_start:
                return False
        
        return True
    
    def should_execute_siphon(self) -> bool:
        """
        Determine if siphon phase should execute
        
        Returns True if:
        - Current time is Friday 6:00pm ET or later
        - Market is open (not a holiday)
        - Haven't executed siphon phase this week
        - Buy phase has been executed this week
        """
        current_time = self.get_current_et_time()
        
        # Check if it's Friday
        if current_time.weekday() != 4:  # 4 = Friday
            return False
        
        # Check if it's past siphon time
        if current_time.time() < self.SIPHON_TIME:
            return False
        
        # Check for market holiday
        if self.holiday_calendar.is_market_holiday(current_time.date()):
            return False
        
        # Check if already executed this week
        if self._last_siphon_execution:
            last_week_start = self._get_week_start(self._last_siphon_execution)
            current_week_start = self._get_week_start(current_time)
            if last_week_start == current_week_start:
                return False
        
        # Check if buy phase executed this week
        if not self._last_buy_execution:
            return False
        
        buy_week_start = self._get_week_start(self._last_buy_execution)
        current_week_start = self._get_week_start(current_time)
        if buy_week_start != current_week_start:
            return False
        
        return True
    
    def execute_buy_phase(self, gate: str, available_cash: float) -> Dict:
        """
        Execute buy phase for specified gate
        
        Args:
            gate: Gate identifier ('G0' or 'G1')
            available_cash: Available cash for purchases
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing buy phase for {gate} with ${available_cash:,.2f}")
        
        if gate not in self.GATE_ALLOCATIONS:
            raise ValueError(f"Unknown gate: {gate}")
        
        allocation = self.GATE_ALLOCATIONS[gate]
        execution_results = {
            'gate': gate,
            'phase': CyclePhase.BUY.value,
            'timestamp': self.get_current_et_time(),
            'total_cash': available_cash,
            'trades': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Base allocation percentages
            base_allocations = {
                'ULTY': allocation.ulty_pct,
                'AMDY': allocation.amdy_pct,
            }

            if allocation.iau_pct > 0:
                base_allocations['IAU'] = allocation.iau_pct

            if allocation.vtip_pct > 0:
                base_allocations['VTIP'] = allocation.vtip_pct

            # Apply Gary's DPI enhancement if enabled
            if self.enable_dpi and self.dpi_integrator:
                try:
                    symbols = list(base_allocations.keys())
                    enhanced_allocations = self.dpi_integrator.get_dpi_enhanced_allocations(
                        symbols, available_cash, base_allocations
                    )

                    # Log DPI analysis
                    dpi_summary = self.dpi_calculator.get_dpi_summary(symbols)
                    execution_results['dpi_analysis'] = dpi_summary

                    logger.info(f"DPI enhanced allocations: {enhanced_allocations}")

                    # Convert to dollar amounts using DPI-enhanced percentages
                    allocations = {
                        symbol: available_cash * (pct / 100.0)
                        for symbol, pct in enhanced_allocations.items()
                    }

                except Exception as e:
                    logger.warning(f"DPI enhancement failed, using base allocations: {e}")
                    # Fallback to base allocations
                    allocations = {
                        symbol: available_cash * (pct / 100.0)
                        for symbol, pct in base_allocations.items()
                    }
            else:
                # Use base allocations without DPI enhancement
                allocations = {
                    symbol: available_cash * (pct / 100.0)
                    for symbol, pct in base_allocations.items()
                }
            
            # Execute trades for each allocation
            for symbol, amount in allocations.items():
                if amount < 1.0:  # Skip if less than $1
                    continue
                
                try:
                    trade_result = self.trade_executor.buy_market_order(
                        symbol=symbol,
                        dollar_amount=amount,
                        gate=gate
                    )
                    
                    execution_results['trades'].append({
                        'symbol': symbol,
                        'amount': amount,
                        'result': trade_result
                    })
                    
                    logger.info(f"Buy order executed: {symbol} ${amount:,.2f}")
                    
                except Exception as e:
                    error_msg = f"Failed to execute buy order for {symbol}: {str(e)}"
                    logger.error(error_msg)
                    execution_results['errors'].append(error_msg)
                    execution_results['success'] = False
            
            # Update execution tracking
            self._last_buy_execution = execution_results['timestamp']
            
        except Exception as e:
            error_msg = f"Buy phase execution failed: {str(e)}"
            logger.error(error_msg)
            execution_results['success'] = False
            execution_results['errors'].append(error_msg)
        
        return execution_results
    
    def execute_siphon_phase(self, gate: str) -> Dict:
        """
        Execute siphon phase (50/50 split management)
        
        Args:
            gate: Gate identifier ('G0' or 'G1')
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing siphon phase for {gate}")
        
        execution_results = {
            'gate': gate,
            'phase': CyclePhase.SIPHON.value,
            'timestamp': self.get_current_et_time(),
            'operations': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Get current portfolio positions for the gate
            positions = self.portfolio_manager.get_gate_positions(gate)
            
            # Calculate total portfolio value
            total_value = sum(pos.market_value for pos in positions.values())
            target_per_position = total_value / len(positions)
            
            logger.info(f"Total portfolio value: ${total_value:,.2f}")
            logger.info(f"Target per position: ${target_per_position:,.2f}")
            
            # Execute rebalancing trades
            for symbol, position in positions.items():
                current_value = position.market_value
                value_diff = current_value - target_per_position
                
                # Only rebalance if difference is significant (>1% of target)
                if abs(value_diff) < target_per_position * 0.01:
                    continue
                
                try:
                    if value_diff > 0:
                        # Sell excess
                        sell_amount = abs(value_diff)
                        trade_result = self.trade_executor.sell_market_order(
                            symbol=symbol,
                            dollar_amount=sell_amount,
                            gate=gate
                        )
                        operation = f"Sell ${sell_amount:,.2f} of {symbol}"
                        
                    else:
                        # Buy to reach target
                        buy_amount = abs(value_diff)
                        trade_result = self.trade_executor.buy_market_order(
                            symbol=symbol,
                            dollar_amount=buy_amount,
                            gate=gate
                        )
                        operation = f"Buy ${buy_amount:,.2f} of {symbol}"
                    
                    execution_results['operations'].append({
                        'symbol': symbol,
                        'operation': operation,
                        'amount': abs(value_diff),
                        'result': trade_result
                    })
                    
                    logger.info(f"Siphon operation: {operation}")
                    
                except Exception as e:
                    error_msg = f"Failed siphon operation for {symbol}: {str(e)}"
                    logger.error(error_msg)
                    execution_results['errors'].append(error_msg)
                    execution_results['success'] = False
            
            # Update execution tracking
            self._last_siphon_execution = execution_results['timestamp']
            
        except Exception as e:
            error_msg = f"Siphon phase execution failed: {str(e)}"
            logger.error(error_msg)
            execution_results['success'] = False
            execution_results['errors'].append(error_msg)
        
        return execution_results
    
    def calculate_weekly_delta(self, week_start: datetime) -> WeeklyDelta:
        """
        Calculate weekly delta (NAV change minus deposits/withdrawals)
        
        Args:
            week_start: Start of the week to calculate
            
        Returns:
            WeeklyDelta object with performance metrics
        """
        week_end = week_start + timedelta(days=7)
        
        # Get NAV values
        nav_start = self.portfolio_manager.get_nav_at_date(week_start.date())
        nav_end = self.portfolio_manager.get_nav_at_date(week_end.date())
        
        # Get cash flows for the week
        deposits = self.portfolio_manager.get_deposits_in_period(
            week_start.date(), week_end.date()
        )
        withdrawals = self.portfolio_manager.get_withdrawals_in_period(
            week_start.date(), week_end.date()
        )
        
        # Calculate delta (performance excluding cash flows)
        delta = (nav_end - nav_start) - (deposits - withdrawals)
        delta_pct = (delta / nav_start) * 100.0 if nav_start > 0 else 0.0
        
        weekly_delta = WeeklyDelta(
            week_start=week_start,
            week_end=week_end,
            nav_start=nav_start,
            nav_end=nav_end,
            deposits=deposits,
            withdrawals=withdrawals,
            delta=delta,
            delta_pct=delta_pct
        )
        
        self._weekly_deltas.append(weekly_delta)
        
        logger.info(f"Weekly delta calculated: {delta_pct:.2f}% (${delta:,.2f})")
        
        return weekly_delta
    
    def handle_market_holiday(self, holiday_date: datetime.date) -> Dict:
        """
        Handle market holiday by deferring execution
        
        Args:
            holiday_date: Date of the market holiday
            
        Returns:
            Dictionary with holiday handling information
        """
        logger.info(f"Handling market holiday: {holiday_date}")
        
        # Find next trading day
        next_trading_day = self.holiday_calendar.get_next_trading_day(holiday_date)
        
        return {
            'holiday_date': holiday_date,
            'next_trading_day': next_trading_day,
            'action': 'defer_execution',
            'message': f"Execution deferred from {holiday_date} to {next_trading_day}"
        }
    
    def get_cycle_status(self) -> Dict:
        """
        Get current cycle status
        
        Returns:
            Dictionary with current cycle information
        """
        current_time = self.get_current_et_time()
        
        return {
            'current_time_et': current_time,
            'current_phase': self._determine_current_phase(),
            'should_execute_buy': self.should_execute_buy(),
            'should_execute_siphon': self.should_execute_siphon(),
            'last_buy_execution': self._last_buy_execution,
            'last_siphon_execution': self._last_siphon_execution,
            'weekly_deltas_count': len(self._weekly_deltas),
            'next_friday': self._get_next_friday()
        }
    
    def get_weekly_performance(self, weeks: int = 4) -> List[WeeklyDelta]:
        """
        Get recent weekly performance data
        
        Args:
            weeks: Number of recent weeks to return
            
        Returns:
            List of WeeklyDelta objects
        """
        return self._weekly_deltas[-weeks:] if self._weekly_deltas else []
    
    def _determine_current_phase(self) -> CyclePhase:
        """Determine current cycle phase based on time and execution status"""
        current_time = self.get_current_et_time()
        
        if current_time.weekday() != 4:  # Not Friday
            return CyclePhase.IDLE
        
        if current_time.time() >= self.SIPHON_TIME:
            if self.should_execute_siphon():
                return CyclePhase.SIPHON
        elif current_time.time() >= self.BUY_TIME:
            if self.should_execute_buy():
                return CyclePhase.BUY
        
        return CyclePhase.IDLE
    
    def _get_week_start(self, dt: datetime) -> datetime:
        """Get Monday of the week containing the given datetime"""
        days_since_monday = dt.weekday()
        monday = dt - timedelta(days=days_since_monday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_next_friday(self) -> datetime:
        """Get next Friday's date"""
        current_time = self.get_current_et_time()
        days_until_friday = (4 - current_time.weekday()) % 7
        if days_until_friday == 0 and current_time.time() > self.SIPHON_TIME:
            days_until_friday = 7  # Next Friday if current Friday is past siphon time
        
        next_friday = current_time + timedelta(days=days_until_friday)
        return next_friday.replace(hour=16, minute=10, second=0, microsecond=0)