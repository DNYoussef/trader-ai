"""
Mock weekly cycle implementation for testing Foundation phase components.
Simulates weekly trading cycles, allocation management, and delta calculations.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import threading
import uuid
import time as time_module


class CyclePhase(Enum):
    PREPARATION = "preparation"
    ANALYSIS = "analysis"
    ALLOCATION = "allocation"
    EXECUTION = "execution"
    REVIEW = "review"
    CLOSED = "closed"


class AllocationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class Allocation:
    """Represents a portfolio allocation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    target_weight: float = 0.0
    current_weight: float = 0.0
    target_value: float = 0.0
    current_value: float = 0.0
    delta: float = 0.0  # target_value - current_value
    delta_percentage: float = 0.0
    priority: int = 100
    status: AllocationStatus = AllocationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleMetrics:
    """Metrics for a weekly cycle"""
    total_portfolio_value: float = 0.0
    target_portfolio_value: float = 0.0
    total_delta: float = 0.0
    absolute_delta: float = 0.0
    delta_percentage: float = 0.0
    number_of_allocations: int = 0
    approved_allocations: int = 0
    executed_allocations: int = 0
    largest_delta: float = 0.0
    smallest_delta: float = 0.0
    concentration_risk: float = 0.0
    diversification_score: float = 0.0


@dataclass
class WeeklyCycle:
    """Represents a complete weekly trading cycle"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    week_number: int = 0
    year: int = 0
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    phase: CyclePhase = CyclePhase.PREPARATION
    allocations: Dict[str, Allocation] = field(default_factory=dict)
    metrics: CycleMetrics = field(default_factory=CycleMetrics)
    phase_timestamps: Dict[str, datetime] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockWeeklyCycleError(Exception):
    """Mock weekly cycle specific errors"""
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code


class MockWeeklyCycleManager:
    """
    Mock weekly cycle manager for testing Foundation phase cycle management.
    Handles cycle timing, allocation management, and delta calculations.
    """
    
    def __init__(self,
                 cycle_duration: timedelta = timedelta(days=7),
                 phase_delays: Dict[str, float] = None,
                 auto_advance_phases: bool = False,
                 error_rate: float = 0.0):
        """
        Initialize mock weekly cycle manager
        
        Args:
            cycle_duration: Duration of each cycle
            phase_delays: Simulated delays for each phase
            auto_advance_phases: Whether to automatically advance phases
            error_rate: Probability of random errors (0.0-1.0)
        """
        self.cycle_duration = cycle_duration
        self.phase_delays = phase_delays or {}
        self.auto_advance_phases = auto_advance_phases
        self.error_rate = error_rate
        
        # Cycle management
        self.current_cycle: Optional[WeeklyCycle] = None
        self.cycle_history: List[WeeklyCycle] = []
        self.cycle_counter = 0
        
        # Market data for calculations
        self.portfolio_value = 1000000.0  # Default $1M portfolio
        self.market_prices: Dict[str, float] = {}
        self.position_values: Dict[str, float] = {}
        
        # Timing configuration
        self.market_open_time = time(9, 30)  # 9:30 AM
        self.market_close_time = time(16, 0)  # 4:00 PM
        self.cycle_start_day = 0  # Monday = 0
        
        # Performance tracking
        self.total_cycles = 0
        self.completed_cycles = 0
        self.failed_cycles = 0
        self.avg_cycle_duration = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks for testing
        self.on_cycle_start = None
        self.on_cycle_end = None
        self.on_phase_change = None
        self.on_allocation_update = None
        
        # Auto-advance thread
        self._auto_advance_thread = None
        self._stop_auto_advance = threading.Event()

    def start_new_cycle(self, start_date: Optional[datetime] = None) -> WeeklyCycle:
        """
        Start a new weekly cycle
        
        Args:
            start_date: Optional start date, defaults to now
            
        Returns:
            WeeklyCycle object
        """
        if self.current_cycle and self.current_cycle.phase != CyclePhase.CLOSED:
            raise MockWeeklyCycleError("Cannot start new cycle while current cycle is active", 4001)
            
        if start_date is None:
            start_date = datetime.now()
            
        # Calculate week number and year
        week_number = start_date.isocalendar()[1]
        year = start_date.year
        
        cycle = WeeklyCycle(
            week_number=week_number,
            year=year,
            start_date=start_date,
            end_date=start_date + self.cycle_duration,
            phase=CyclePhase.PREPARATION
        )
        
        with self._lock:
            self.current_cycle = cycle
            self.cycle_counter += 1
            self.total_cycles += 1
            
            # Record phase timestamp
            cycle.phase_timestamps[CyclePhase.PREPARATION.value] = datetime.now()
            
        # Trigger callback
        if self.on_cycle_start:
            self.on_cycle_start(cycle)
            
        # Start auto-advance if configured
        if self.auto_advance_phases:
            self._start_auto_advance()
            
        return cycle

    def advance_phase(self) -> bool:
        """
        Advance to next phase in current cycle
        
        Returns:
            bool: Success status
        """
        if not self.current_cycle:
            raise MockWeeklyCycleError("No active cycle to advance", 4002)
            
        current_phase = self.current_cycle.phase
        
        # Define phase progression
        phase_progression = {
            CyclePhase.PREPARATION: CyclePhase.ANALYSIS,
            CyclePhase.ANALYSIS: CyclePhase.ALLOCATION,
            CyclePhase.ALLOCATION: CyclePhase.EXECUTION,
            CyclePhase.EXECUTION: CyclePhase.REVIEW,
            CyclePhase.REVIEW: CyclePhase.CLOSED
        }
        
        next_phase = phase_progression.get(current_phase)
        if not next_phase:
            return False  # Already at final phase
            
        # Simulate phase delay
        delay = self.phase_delays.get(current_phase.value, 0.0)
        if delay > 0:
            time_module.sleep(delay)
            
        # Check for random errors
        if self.error_rate > 0 and self._should_error():
            raise MockWeeklyCycleError(f"Phase advancement failed: {current_phase.value}", 4003)
            
        with self._lock:
            self.current_cycle.phase = next_phase
            self.current_cycle.updated_at = datetime.now()
            self.current_cycle.phase_timestamps[next_phase.value] = datetime.now()
            
        # Handle cycle completion
        if next_phase == CyclePhase.CLOSED:
            self._complete_cycle()
            
        # Trigger callback
        if self.on_phase_change:
            self.on_phase_change(self.current_cycle, current_phase, next_phase)
            
        return True

    def add_allocation(self, symbol: str, target_weight: float, 
                      priority: int = 100) -> Allocation:
        """
        Add allocation to current cycle
        
        Args:
            symbol: Trading symbol
            target_weight: Target portfolio weight (0.0-1.0)
            priority: Allocation priority
            
        Returns:
            Allocation object
        """
        if not self.current_cycle:
            raise MockWeeklyCycleError("No active cycle for allocation", 4004)
            
        if self.current_cycle.phase not in [CyclePhase.PREPARATION, CyclePhase.ANALYSIS, CyclePhase.ALLOCATION]:
            raise MockWeeklyCycleError("Cannot add allocations in current phase", 4005)
            
        if not 0.0 <= target_weight <= 1.0:
            raise MockWeeklyCycleError("Target weight must be between 0.0 and 1.0", 4006)
            
        # Get current position value
        current_value = self.position_values.get(symbol, 0.0)
        current_weight = current_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        # Calculate target value and delta
        target_value = self.portfolio_value * target_weight
        delta = target_value - current_value
        delta_percentage = (delta / current_value) if current_value > 0 else 0.0
        
        allocation = Allocation(
            symbol=symbol,
            target_weight=target_weight,
            current_weight=current_weight,
            target_value=target_value,
            current_value=current_value,
            delta=delta,
            delta_percentage=delta_percentage,
            priority=priority
        )
        
        with self._lock:
            self.current_cycle.allocations[allocation.id] = allocation
            self._update_cycle_metrics()
            
        # Trigger callback
        if self.on_allocation_update:
            self.on_allocation_update(allocation)
            
        return allocation

    def update_allocation(self, allocation_id: str, 
                         target_weight: Optional[float] = None,
                         priority: Optional[int] = None,
                         status: Optional[AllocationStatus] = None) -> bool:
        """
        Update existing allocation
        
        Args:
            allocation_id: ID of allocation to update
            target_weight: New target weight
            priority: New priority
            status: New status
            
        Returns:
            bool: Success status
        """
        if not self.current_cycle:
            raise MockWeeklyCycleError("No active cycle", 4007)
            
        allocation = self.current_cycle.allocations.get(allocation_id)
        if not allocation:
            raise MockWeeklyCycleError("Allocation not found", 4008)
            
        with self._lock:
            if target_weight is not None:
                if not 0.0 <= target_weight <= 1.0:
                    raise MockWeeklyCycleError("Target weight must be between 0.0 and 1.0", 4006)
                    
                allocation.target_weight = target_weight
                allocation.target_value = self.portfolio_value * target_weight
                allocation.delta = allocation.target_value - allocation.current_value
                allocation.delta_percentage = (
                    allocation.delta / allocation.current_value 
                    if allocation.current_value > 0 else 0.0
                )
                
            if priority is not None:
                allocation.priority = priority
                
            if status is not None:
                allocation.status = status
                
            allocation.updated_at = datetime.now()
            self._update_cycle_metrics()
            
        # Trigger callback
        if self.on_allocation_update:
            self.on_allocation_update(allocation)
            
        return True

    def remove_allocation(self, allocation_id: str) -> bool:
        """Remove allocation from current cycle"""
        if not self.current_cycle:
            return False
            
        with self._lock:
            if allocation_id in self.current_cycle.allocations:
                del self.current_cycle.allocations[allocation_id]
                self._update_cycle_metrics()
                return True
                
        return False

    def get_allocation(self, allocation_id: str) -> Optional[Allocation]:
        """Get allocation by ID"""
        if not self.current_cycle:
            return None
        return self.current_cycle.allocations.get(allocation_id)

    def get_allocations(self, status: Optional[AllocationStatus] = None) -> List[Allocation]:
        """
        Get all allocations, optionally filtered by status
        
        Args:
            status: Optional status filter
            
        Returns:
            List of allocations
        """
        if not self.current_cycle:
            return []
            
        allocations = list(self.current_cycle.allocations.values())
        
        if status:
            allocations = [a for a in allocations if a.status == status]
            
        return allocations

    def calculate_deltas(self) -> Dict[str, float]:
        """
        Calculate deltas for all allocations
        
        Returns:
            Dictionary mapping allocation IDs to delta values
        """
        if not self.current_cycle:
            return {}
            
        deltas = {}
        
        with self._lock:
            for allocation_id, allocation in self.current_cycle.allocations.items():
                # Recalculate with current market data
                current_value = self.position_values.get(allocation.symbol, 0.0)
                target_value = self.portfolio_value * allocation.target_weight
                delta = target_value - current_value
                
                allocation.current_value = current_value
                allocation.current_weight = current_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
                allocation.delta = delta
                allocation.delta_percentage = delta / current_value if current_value > 0 else 0.0
                allocation.updated_at = datetime.now()
                
                deltas[allocation_id] = delta
                
            self._update_cycle_metrics()
            
        return deltas

    def approve_allocations(self, allocation_ids: List[str]) -> int:
        """
        Approve specified allocations
        
        Args:
            allocation_ids: List of allocation IDs to approve
            
        Returns:
            Number of successfully approved allocations
        """
        if not self.current_cycle:
            return 0
            
        approved_count = 0
        
        with self._lock:
            for allocation_id in allocation_ids:
                allocation = self.current_cycle.allocations.get(allocation_id)
                if allocation and allocation.status == AllocationStatus.PENDING:
                    allocation.status = AllocationStatus.APPROVED
                    allocation.updated_at = datetime.now()
                    approved_count += 1
                    
            self._update_cycle_metrics()
            
        return approved_count

    def execute_allocations(self, allocation_ids: List[str] = None) -> int:
        """
        Execute approved allocations
        
        Args:
            allocation_ids: Optional list of specific allocations to execute
            
        Returns:
            Number of successfully executed allocations
        """
        if not self.current_cycle:
            return 0
            
        if self.current_cycle.phase != CyclePhase.EXECUTION:
            raise MockWeeklyCycleError("Cannot execute allocations outside execution phase", 4009)
            
        executed_count = 0
        
        with self._lock:
            allocations_to_execute = (
                [self.current_cycle.allocations[aid] for aid in allocation_ids 
                 if aid in self.current_cycle.allocations]
                if allocation_ids 
                else [a for a in self.current_cycle.allocations.values() 
                      if a.status == AllocationStatus.APPROVED]
            )
            
            for allocation in allocations_to_execute:
                if allocation.status == AllocationStatus.APPROVED:
                    # Simulate execution
                    if self._should_error():
                        allocation.status = AllocationStatus.REJECTED
                    else:
                        allocation.status = AllocationStatus.EXECUTED
                        # Update position value to reflect execution
                        self.position_values[allocation.symbol] = allocation.target_value
                        executed_count += 1
                        
                    allocation.updated_at = datetime.now()
                    
            self._update_cycle_metrics()
            
        return executed_count

    def get_current_cycle(self) -> Optional[WeeklyCycle]:
        """Get current active cycle"""
        return self.current_cycle

    def get_cycle_history(self) -> List[WeeklyCycle]:
        """Get all historical cycles"""
        return self.cycle_history[:]

    def get_cycle_metrics(self) -> Optional[CycleMetrics]:
        """Get metrics for current cycle"""
        if not self.current_cycle:
            return None
        return self.current_cycle.metrics

    def set_portfolio_value(self, value: float):
        """Set total portfolio value for calculations"""
        self.portfolio_value = value
        
        # Recalculate allocations if cycle is active
        if self.current_cycle:
            self.calculate_deltas()

    def set_position_value(self, symbol: str, value: float):
        """Set current position value for symbol"""
        self.position_values[symbol] = value
        
        # Recalculate affected allocations
        if self.current_cycle:
            for allocation in self.current_cycle.allocations.values():
                if allocation.symbol == symbol:
                    allocation.current_value = value
                    allocation.current_weight = value / self.portfolio_value if self.portfolio_value > 0 else 0.0
                    allocation.delta = allocation.target_value - value
                    allocation.delta_percentage = allocation.delta / value if value > 0 else 0.0
                    allocation.updated_at = datetime.now()
                    
            self._update_cycle_metrics()

    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if market is open at given time"""
        if check_time is None:
            check_time = datetime.now()
            
        # Simplified market hours check (ignores holidays)
        weekday = check_time.weekday()
        if weekday >= 5:  # Saturday or Sunday
            return False
            
        current_time = check_time.time()
        return self.market_open_time <= current_time <= self.market_close_time

    def time_to_next_cycle(self) -> Optional[timedelta]:
        """Calculate time until next cycle should start"""
        if not self.current_cycle:
            return timedelta(0)
            
        if self.current_cycle.phase != CyclePhase.CLOSED:
            return None  # Current cycle still active
            
        # Calculate next cycle start (next Monday)
        now = datetime.now()
        days_ahead = self.cycle_start_day - now.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
            
        next_start = now + timedelta(days=days_ahead)
        next_start = next_start.replace(hour=self.market_open_time.hour, 
                                      minute=self.market_open_time.minute,
                                      second=0, microsecond=0)
        
        return next_start - now

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        success_rate = self.completed_cycles / max(self.total_cycles, 1)
        
        return {
            "total_cycles": self.total_cycles,
            "completed_cycles": self.completed_cycles,
            "failed_cycles": self.failed_cycles,
            "success_rate": success_rate,
            "avg_cycle_duration": self.avg_cycle_duration,
            "current_cycle_phase": self.current_cycle.phase.value if self.current_cycle else None,
            "portfolio_value": self.portfolio_value,
            "active_positions": len(self.position_values),
            "auto_advance_enabled": self.auto_advance_phases
        }

    def reset_for_testing(self):
        """Reset manager state for testing"""
        self._stop_auto_advance.set()
        if self._auto_advance_thread:
            self._auto_advance_thread.join()
            
        with self._lock:
            self.current_cycle = None
            self.cycle_history.clear()
            self.cycle_counter = 0
            self.portfolio_value = 1000000.0
            self.market_prices.clear()
            self.position_values.clear()
            self.total_cycles = 0
            self.completed_cycles = 0
            self.failed_cycles = 0
            self.avg_cycle_duration = 0.0
            
        self._stop_auto_advance.clear()

    def _complete_cycle(self):
        """Complete current cycle and move to history"""
        if self.current_cycle:
            with self._lock:
                # Calculate final metrics
                self._update_cycle_metrics()
                
                # Calculate cycle duration
                start_time = self.current_cycle.phase_timestamps.get(CyclePhase.PREPARATION.value)
                end_time = datetime.now()
                if start_time:
                    duration = (end_time - start_time).total_seconds()
                    self.current_cycle.metadata["duration_seconds"] = duration
                    
                    # Update average duration
                    if self.completed_cycles > 0:
                        self.avg_cycle_duration = (
                            (self.avg_cycle_duration * self.completed_cycles + duration) 
                            / (self.completed_cycles + 1)
                        )
                    else:
                        self.avg_cycle_duration = duration
                        
                self.cycle_history.append(self.current_cycle)
                self.completed_cycles += 1
                
                # Trigger callback
                if self.on_cycle_end:
                    self.on_cycle_end(self.current_cycle)
                    
                self.current_cycle = None

    def _update_cycle_metrics(self):
        """Update metrics for current cycle"""
        if not self.current_cycle:
            return
            
        allocations = list(self.current_cycle.allocations.values())
        
        if not allocations:
            return
            
        metrics = self.current_cycle.metrics
        
        # Basic counts
        metrics.number_of_allocations = len(allocations)
        metrics.approved_allocations = len([a for a in allocations if a.status == AllocationStatus.APPROVED])
        metrics.executed_allocations = len([a for a in allocations if a.status == AllocationStatus.EXECUTED])
        
        # Portfolio values
        metrics.total_portfolio_value = self.portfolio_value
        metrics.target_portfolio_value = sum(a.target_value for a in allocations)
        
        # Delta calculations
        deltas = [a.delta for a in allocations]
        metrics.total_delta = sum(deltas)
        metrics.absolute_delta = sum(abs(d) for d in deltas)
        metrics.delta_percentage = metrics.total_delta / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        if deltas:
            metrics.largest_delta = max(deltas)
            metrics.smallest_delta = min(deltas)
            
        # Risk metrics
        weights = [a.target_weight for a in allocations]
        if weights:
            # Concentration risk (largest weight)
            metrics.concentration_risk = max(weights)
            
            # Simple diversification score (inverse of concentration)
            metrics.diversification_score = 1.0 - metrics.concentration_risk

    def _should_error(self) -> bool:
        """Determine if an error should occur based on error rate"""
        import random
        return random.random() < self.error_rate

    def _start_auto_advance(self):
        """Start auto-advance thread"""
        if self._auto_advance_thread and self._auto_advance_thread.is_alive():
            return
            
        self._stop_auto_advance.clear()
        self._auto_advance_thread = threading.Thread(target=self._auto_advance_worker, daemon=True)
        self._auto_advance_thread.start()

    def _auto_advance_worker(self):
        """Worker thread for auto-advancing phases"""
        while not self._stop_auto_advance.wait(1.0):  # Check every second
            if self.current_cycle and self.current_cycle.phase != CyclePhase.CLOSED:
                try:
                    # Check if it's time to advance based on phase delays
                    phase = self.current_cycle.phase
                    phase_start = self.current_cycle.phase_timestamps.get(phase.value)
                    
                    if phase_start:
                        delay = self.phase_delays.get(phase.value, 5.0)  # Default 5 second delay
                        elapsed = (datetime.now() - phase_start).total_seconds()
                        
                        if elapsed >= delay:
                            self.advance_phase()
                            
                except Exception:
                    pass  # Ignore errors in auto-advance


def create_mock_weekly_cycle_manager(**kwargs) -> MockWeeklyCycleManager:
    """Factory function to create mock weekly cycle manager"""
    return MockWeeklyCycleManager(**kwargs)


def create_test_allocation(symbol: str = "AAPL",
                          target_weight: float = 0.1,
                          current_value: float = 50000.0) -> Allocation:
    """Factory function to create test allocation"""
    portfolio_value = 1000000.0
    target_value = portfolio_value * target_weight
    
    return Allocation(
        symbol=symbol,
        target_weight=target_weight,
        current_weight=current_value / portfolio_value,
        target_value=target_value,
        current_value=current_value,
        delta=target_value - current_value,
        delta_percentage=(target_value - current_value) / current_value if current_value > 0 else 0.0
    )


def create_test_cycle(phase: CyclePhase = CyclePhase.PREPARATION) -> WeeklyCycle:
    """Factory function to create test cycle"""
    now = datetime.now()
    return WeeklyCycle(
        week_number=now.isocalendar()[1],
        year=now.year,
        start_date=now,
        end_date=now + timedelta(days=7),
        phase=phase
    )