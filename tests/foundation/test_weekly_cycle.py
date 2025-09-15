"""
Comprehensive tests for weekly cycle management in Foundation phase.
Tests cycle timing, allocation management, delta calculations, and phase progression.
"""
import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any

# Import mock objects
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mocks.mock_weekly_cycle import (
    MockWeeklyCycleManager, WeeklyCycle, Allocation, CycleMetrics,
    CyclePhase, AllocationStatus, MockWeeklyCycleError,
    create_mock_weekly_cycle_manager, create_test_allocation, create_test_cycle
)


class TestCycleCreation:
    """Test weekly cycle creation and initialization"""
    
    def test_start_new_cycle(self):
        """Test starting a new weekly cycle"""
        manager = create_mock_weekly_cycle_manager()
        
        cycle = manager.start_new_cycle()
        
        assert cycle is not None
        assert cycle.phase == CyclePhase.PREPARATION
        assert cycle.week_number > 0
        assert cycle.year == datetime.now().year
        assert manager.current_cycle == cycle
        assert manager.total_cycles == 1

    def test_start_cycle_with_specific_date(self):
        """Test starting cycle with specific start date"""
        manager = create_mock_weekly_cycle_manager()
        
        start_date = datetime(2024, 6, 3, 9, 30)  # Monday 9:30 AM
        cycle = manager.start_new_cycle(start_date)
        
        assert cycle.start_date == start_date
        assert cycle.end_date == start_date + timedelta(days=7)
        assert cycle.week_number == start_date.isocalendar()[1]

    def test_cannot_start_cycle_while_active(self):
        """Test that starting a new cycle while one is active fails"""
        manager = create_mock_weekly_cycle_manager()
        
        # Start first cycle
        manager.start_new_cycle()
        
        # Try to start second cycle
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.start_new_cycle()
            
        assert exc_info.value.error_code == 4001

    def test_cycle_start_callback(self):
        """Test cycle start callback"""
        manager = create_mock_weekly_cycle_manager()
        callback_results = []
        
        def start_callback(cycle):
            callback_results.append(cycle.id)
            
        manager.on_cycle_start = start_callback
        
        cycle = manager.start_new_cycle()
        
        assert cycle.id in callback_results

    def test_cycle_counter_increment(self):
        """Test that cycle counter increments properly"""
        manager = create_mock_weekly_cycle_manager()
        
        initial_counter = manager.cycle_counter
        
        cycle1 = manager.start_new_cycle()
        assert manager.cycle_counter == initial_counter + 1
        
        # Complete first cycle
        manager.current_cycle.phase = CyclePhase.CLOSED
        manager._complete_cycle()
        
        cycle2 = manager.start_new_cycle()
        assert manager.cycle_counter == initial_counter + 2


class TestPhaseManagement:
    """Test cycle phase progression"""
    
    @pytest.fixture
    def cycle_with_manager(self):
        """Fixture providing manager with active cycle"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={"preparation": 0.01, "analysis": 0.01, "allocation": 0.01}
        )
        cycle = manager.start_new_cycle()
        return manager, cycle

    def test_advance_phase_sequence(self, cycle_with_manager):
        """Test advancing through all phases in sequence"""
        manager, cycle = cycle_with_manager
        
        expected_phases = [
            CyclePhase.PREPARATION,
            CyclePhase.ANALYSIS,
            CyclePhase.ALLOCATION,
            CyclePhase.EXECUTION,
            CyclePhase.REVIEW,
            CyclePhase.CLOSED
        ]
        
        for i, expected_phase in enumerate(expected_phases):
            assert cycle.phase == expected_phase
            
            if i < len(expected_phases) - 1:  # Not the last phase
                result = manager.advance_phase()
                assert result is True

    def test_advance_phase_with_delays(self):
        """Test phase advancement with configured delays"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={"preparation": 0.1}
        )
        cycle = manager.start_new_cycle()
        
        start_time = time.time()
        manager.advance_phase()
        end_time = time.time()
        
        assert (end_time - start_time) >= 0.1
        assert cycle.phase == CyclePhase.ANALYSIS

    def test_advance_phase_without_cycle(self):
        """Test advancing phase without active cycle"""
        manager = create_mock_weekly_cycle_manager()
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.advance_phase()
            
        assert exc_info.value.error_code == 4002

    def test_advance_phase_timestamps(self, cycle_with_manager):
        """Test that phase timestamps are recorded"""
        manager, cycle = cycle_with_manager
        
        initial_timestamps = len(cycle.phase_timestamps)
        
        manager.advance_phase()
        
        assert len(cycle.phase_timestamps) == initial_timestamps + 1
        assert CyclePhase.ANALYSIS.value in cycle.phase_timestamps

    def test_phase_change_callback(self, cycle_with_manager):
        """Test phase change callback"""
        manager, cycle = cycle_with_manager
        callback_results = []
        
        def phase_callback(cycle, old_phase, new_phase):
            callback_results.append((old_phase, new_phase))
            
        manager.on_phase_change = phase_callback
        
        manager.advance_phase()
        
        assert len(callback_results) == 1
        assert callback_results[0] == (CyclePhase.PREPARATION, CyclePhase.ANALYSIS)

    def test_advance_phase_error_handling(self):
        """Test phase advancement error handling"""
        manager = create_mock_weekly_cycle_manager(error_rate=1.0)  # Force errors
        manager.start_new_cycle()
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.advance_phase()
            
        assert exc_info.value.error_code == 4003

    def test_cycle_completion_on_final_phase(self, cycle_with_manager):
        """Test that cycle is completed when reaching final phase"""
        manager, cycle = cycle_with_manager
        
        # Advance to final phase
        while cycle.phase != CyclePhase.CLOSED:
            manager.advance_phase()
            
        # Cycle should be completed
        assert manager.current_cycle is None
        assert len(manager.cycle_history) == 1
        assert manager.completed_cycles == 1

    def test_auto_advance_phases(self):
        """Test automatic phase advancement"""
        manager = create_mock_weekly_cycle_manager(
            auto_advance_phases=True,
            phase_delays={"preparation": 0.05, "analysis": 0.05}
        )
        
        cycle = manager.start_new_cycle()
        initial_phase = cycle.phase
        
        # Wait for auto-advancement
        time.sleep(0.15)
        
        # Phase should have advanced
        assert cycle.phase != initial_phase


class TestAllocationManagement:
    """Test allocation creation and management"""
    
    @pytest.fixture
    def manager_with_cycle(self):
        """Fixture providing manager with active cycle"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)  # $1M portfolio
        manager.set_position_value("AAPL", 50000.0)  # Current $50K in AAPL
        
        cycle = manager.start_new_cycle()
        return manager, cycle

    def test_add_allocation(self, manager_with_cycle):
        """Test adding allocation to cycle"""
        manager, cycle = manager_with_cycle
        
        allocation = manager.add_allocation("AAPL", 0.10, priority=100)  # 10% target
        
        assert allocation.symbol == "AAPL"
        assert allocation.target_weight == 0.10
        assert allocation.target_value == 100000.0  # 10% of $1M
        assert allocation.current_value == 50000.0
        assert allocation.delta == 50000.0  # Need to buy $50K more
        assert allocation.status == AllocationStatus.PENDING
        assert allocation.id in cycle.allocations

    def test_add_allocation_without_cycle(self):
        """Test adding allocation without active cycle"""
        manager = create_mock_weekly_cycle_manager()
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.add_allocation("AAPL", 0.10)
            
        assert exc_info.value.error_code == 4004

    def test_add_allocation_in_wrong_phase(self, manager_with_cycle):
        """Test adding allocation in wrong phase"""
        manager, cycle = manager_with_cycle
        
        # Move to execution phase
        cycle.phase = CyclePhase.EXECUTION
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.add_allocation("AAPL", 0.10)
            
        assert exc_info.value.error_code == 4005

    def test_add_allocation_invalid_weight(self, manager_with_cycle):
        """Test adding allocation with invalid weight"""
        manager, cycle = manager_with_cycle
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.add_allocation("AAPL", 1.5)  # > 1.0
            
        assert exc_info.value.error_code == 4006
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.add_allocation("AAPL", -0.1)  # < 0.0
            
        assert exc_info.value.error_code == 4006

    def test_update_allocation(self, manager_with_cycle):
        """Test updating existing allocation"""
        manager, cycle = manager_with_cycle
        
        allocation = manager.add_allocation("AAPL", 0.10)
        original_target = allocation.target_value
        
        # Update target weight
        result = manager.update_allocation(allocation.id, target_weight=0.15)
        
        assert result is True
        assert allocation.target_weight == 0.15
        assert allocation.target_value == 150000.0  # 15% of $1M
        assert allocation.target_value != original_target

    def test_update_allocation_priority(self, manager_with_cycle):
        """Test updating allocation priority"""
        manager, cycle = manager_with_cycle
        
        allocation = manager.add_allocation("AAPL", 0.10, priority=100)
        
        result = manager.update_allocation(allocation.id, priority=200)
        
        assert result is True
        assert allocation.priority == 200

    def test_update_allocation_status(self, manager_with_cycle):
        """Test updating allocation status"""
        manager, cycle = manager_with_cycle
        
        allocation = manager.add_allocation("AAPL", 0.10)
        
        result = manager.update_allocation(allocation.id, status=AllocationStatus.APPROVED)
        
        assert result is True
        assert allocation.status == AllocationStatus.APPROVED

    def test_update_nonexistent_allocation(self, manager_with_cycle):
        """Test updating non-existent allocation"""
        manager, cycle = manager_with_cycle
        
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.update_allocation("fake-id", target_weight=0.10)
            
        assert exc_info.value.error_code == 4008

    def test_remove_allocation(self, manager_with_cycle):
        """Test removing allocation"""
        manager, cycle = manager_with_cycle
        
        allocation = manager.add_allocation("AAPL", 0.10)
        allocation_id = allocation.id
        
        result = manager.remove_allocation(allocation_id)
        
        assert result is True
        assert allocation_id not in cycle.allocations

    def test_get_allocations_by_status(self, manager_with_cycle):
        """Test filtering allocations by status"""
        manager, cycle = manager_with_cycle
        
        # Add allocations with different statuses
        alloc1 = manager.add_allocation("AAPL", 0.10)
        alloc2 = manager.add_allocation("GOOGL", 0.15)
        alloc3 = manager.add_allocation("MSFT", 0.12)
        
        # Approve some
        manager.update_allocation(alloc1.id, status=AllocationStatus.APPROVED)
        manager.update_allocation(alloc2.id, status=AllocationStatus.APPROVED)
        
        # Get approved allocations
        approved = manager.get_allocations(status=AllocationStatus.APPROVED)
        assert len(approved) == 2
        
        # Get pending allocations
        pending = manager.get_allocations(status=AllocationStatus.PENDING)
        assert len(pending) == 1

    def test_allocation_callback(self, manager_with_cycle):
        """Test allocation update callback"""
        manager, cycle = manager_with_cycle
        callback_results = []
        
        def allocation_callback(allocation):
            callback_results.append(allocation.symbol)
            
        manager.on_allocation_update = allocation_callback
        
        manager.add_allocation("AAPL", 0.10)
        
        assert "AAPL" in callback_results


class TestDeltaCalculations:
    """Test delta calculation functionality"""
    
    @pytest.fixture
    def manager_with_allocations(self):
        """Fixture providing manager with test allocations"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        
        # Set current positions
        manager.set_position_value("AAPL", 80000.0)   # Currently $80K
        manager.set_position_value("GOOGL", 120000.0) # Currently $120K
        manager.set_position_value("MSFT", 60000.0)   # Currently $60K
        
        cycle = manager.start_new_cycle()
        
        # Add target allocations
        manager.add_allocation("AAPL", 0.10)   # Target $100K (need +$20K)
        manager.add_allocation("GOOGL", 0.15)  # Target $150K (need +$30K)
        manager.add_allocation("MSFT", 0.08)   # Target $80K (need +$20K)
        
        return manager, cycle

    def test_calculate_basic_deltas(self, manager_with_allocations):
        """Test basic delta calculations"""
        manager, cycle = manager_with_allocations
        
        deltas = manager.calculate_deltas()
        
        # Verify delta calculations
        allocations = list(cycle.allocations.values())
        aapl_alloc = next(a for a in allocations if a.symbol == "AAPL")
        googl_alloc = next(a for a in allocations if a.symbol == "GOOGL")
        msft_alloc = next(a for a in allocations if a.symbol == "MSFT")
        
        assert aapl_alloc.delta == 20000.0   # $100K target - $80K current
        assert googl_alloc.delta == 30000.0  # $150K target - $120K current
        assert msft_alloc.delta == 20000.0   # $80K target - $60K current

    def test_calculate_negative_deltas(self):
        """Test calculations with negative deltas (reductions)"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        manager.set_position_value("AAPL", 200000.0)  # Over-allocated
        
        manager.start_new_cycle()
        manager.add_allocation("AAPL", 0.10)  # Target only $100K
        
        deltas = manager.calculate_deltas()
        
        allocation = next(iter(manager.current_cycle.allocations.values()))
        assert allocation.delta == -100000.0  # Need to reduce by $100K

    def test_delta_percentage_calculations(self, manager_with_allocations):
        """Test delta percentage calculations"""
        manager, cycle = manager_with_allocations
        
        manager.calculate_deltas()
        
        allocations = list(cycle.allocations.values())
        aapl_alloc = next(a for a in allocations if a.symbol == "AAPL")
        
        # Delta percentage = delta / current_value
        expected_percentage = 20000.0 / 80000.0  # 25%
        assert abs(aapl_alloc.delta_percentage - expected_percentage) < 0.001

    def test_recalculate_on_position_update(self, manager_with_allocations):
        """Test that deltas recalculate when positions update"""
        manager, cycle = manager_with_allocations
        
        # Get initial delta
        aapl_alloc = next(a for a in cycle.allocations.values() if a.symbol == "AAPL")
        initial_delta = aapl_alloc.delta
        
        # Update position value
        manager.set_position_value("AAPL", 90000.0)  # Changed from $80K to $90K
        
        # Delta should have changed
        updated_delta = aapl_alloc.delta
        assert updated_delta != initial_delta
        assert updated_delta == 10000.0  # $100K target - $90K current

    def test_recalculate_on_portfolio_value_change(self, manager_with_allocations):
        """Test that deltas recalculate when portfolio value changes"""
        manager, cycle = manager_with_allocations
        
        # Get initial delta
        aapl_alloc = next(a for a in cycle.allocations.values() if a.symbol == "AAPL")
        initial_delta = aapl_alloc.delta
        
        # Change portfolio value
        manager.set_portfolio_value(2000000.0)  # Double the portfolio
        
        # Target values and deltas should have changed
        assert aapl_alloc.target_value == 200000.0  # 10% of $2M
        assert aapl_alloc.delta != initial_delta

    def test_zero_position_delta_calculation(self):
        """Test delta calculations for zero current positions"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        # Don't set any position value (defaults to 0)
        
        manager.start_new_cycle()
        manager.add_allocation("NEW_STOCK", 0.05)  # 5% target
        
        manager.calculate_deltas()
        
        allocation = next(iter(manager.current_cycle.allocations.values()))
        assert allocation.current_value == 0.0
        assert allocation.delta == 50000.0  # Full target amount needed
        assert allocation.delta_percentage == 0.0  # Can't divide by zero


class TestAllocationApprovalAndExecution:
    """Test allocation approval and execution workflow"""
    
    @pytest.fixture
    def manager_with_pending_allocations(self):
        """Fixture providing manager with pending allocations"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        
        cycle = manager.start_new_cycle()
        
        # Add multiple allocations
        alloc1 = manager.add_allocation("AAPL", 0.10)
        alloc2 = manager.add_allocation("GOOGL", 0.15) 
        alloc3 = manager.add_allocation("MSFT", 0.12)
        
        return manager, cycle, [alloc1, alloc2, alloc3]

    def test_approve_allocations(self, manager_with_pending_allocations):
        """Test approving allocations"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        # Approve first two allocations
        allocation_ids = [allocations[0].id, allocations[1].id]
        approved_count = manager.approve_allocations(allocation_ids)
        
        assert approved_count == 2
        assert allocations[0].status == AllocationStatus.APPROVED
        assert allocations[1].status == AllocationStatus.APPROVED
        assert allocations[2].status == AllocationStatus.PENDING

    def test_approve_nonexistent_allocations(self, manager_with_pending_allocations):
        """Test approving non-existent allocations"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        fake_ids = ["fake-id-1", "fake-id-2"]
        approved_count = manager.approve_allocations(fake_ids)
        
        assert approved_count == 0

    def test_execute_approved_allocations(self, manager_with_pending_allocations):
        """Test executing approved allocations"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        # Move to execution phase
        while cycle.phase != CyclePhase.EXECUTION:
            manager.advance_phase()
            
        # Approve all allocations
        allocation_ids = [a.id for a in allocations]
        manager.approve_allocations(allocation_ids)
        
        # Execute all approved
        executed_count = manager.execute_allocations()
        
        assert executed_count == 3
        assert all(a.status == AllocationStatus.EXECUTED for a in allocations)

    def test_execute_specific_allocations(self, manager_with_pending_allocations):
        """Test executing specific allocations"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        # Move to execution phase
        while cycle.phase != CyclePhase.EXECUTION:
            manager.advance_phase()
            
        # Approve all but execute only first two
        manager.approve_allocations([a.id for a in allocations])
        
        executed_count = manager.execute_allocations([allocations[0].id, allocations[1].id])
        
        assert executed_count == 2
        assert allocations[0].status == AllocationStatus.EXECUTED
        assert allocations[1].status == AllocationStatus.EXECUTED
        assert allocations[2].status == AllocationStatus.APPROVED  # Not executed

    def test_execute_outside_execution_phase(self, manager_with_pending_allocations):
        """Test executing allocations outside execution phase"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        # Try to execute in preparation phase
        with pytest.raises(MockWeeklyCycleError) as exc_info:
            manager.execute_allocations()
            
        assert exc_info.value.error_code == 4009

    def test_execution_with_errors(self):
        """Test allocation execution with simulated errors"""
        manager = create_mock_weekly_cycle_manager(error_rate=0.5)  # 50% error rate
        manager.set_portfolio_value(1000000.0)
        
        cycle = manager.start_new_cycle()
        
        # Add and approve multiple allocations
        allocations = []
        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]:
            alloc = manager.add_allocation(symbol, 0.1)
            allocations.append(alloc)
            
        # Move to execution phase
        while cycle.phase != CyclePhase.EXECUTION:
            manager.advance_phase()
            
        manager.approve_allocations([a.id for a in allocations])
        
        # Execute and expect some failures
        executed_count = manager.execute_allocations()
        rejected_count = len([a for a in allocations if a.status == AllocationStatus.REJECTED])
        
        assert executed_count + rejected_count == len(allocations)
        assert rejected_count > 0  # Some should fail with 50% error rate

    def test_position_updates_after_execution(self, manager_with_pending_allocations):
        """Test that position values update after successful execution"""
        manager, cycle, allocations = manager_with_pending_allocations
        
        # Move to execution phase and approve
        while cycle.phase != CyclePhase.EXECUTION:
            manager.advance_phase()
            
        manager.approve_allocations([a.id for a in allocations])
        
        # Get initial position values
        initial_positions = dict(manager.position_values)
        
        # Execute
        manager.execute_allocations()
        
        # Position values should be updated to target values
        for allocation in allocations:
            if allocation.status == AllocationStatus.EXECUTED:
                assert manager.position_values[allocation.symbol] == allocation.target_value


class TestCycleMetrics:
    """Test cycle metrics calculation and tracking"""
    
    @pytest.fixture
    def manager_with_metrics_data(self):
        """Fixture providing manager with data for metrics testing"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        
        cycle = manager.start_new_cycle()
        
        # Add allocations with various properties
        manager.add_allocation("AAPL", 0.20)   # Large allocation
        manager.add_allocation("GOOGL", 0.15)  # Medium allocation
        manager.add_allocation("MSFT", 0.10)   # Medium allocation
        manager.add_allocation("SMALL", 0.02)  # Small allocation
        
        return manager, cycle

    def test_basic_metrics_calculation(self, manager_with_metrics_data):
        """Test basic metrics calculations"""
        manager, cycle = manager_with_metrics_data
        
        metrics = manager.get_cycle_metrics()
        
        assert metrics.number_of_allocations == 4
        assert metrics.total_portfolio_value == 1000000.0
        assert metrics.target_portfolio_value == 470000.0  # Sum of target values
        assert metrics.approved_allocations == 0  # None approved yet
        assert metrics.executed_allocations == 0  # None executed yet

    def test_delta_metrics(self, manager_with_metrics_data):
        """Test delta-related metrics"""
        manager, cycle = manager_with_metrics_data
        
        # Set some current positions to create meaningful deltas
        manager.set_position_value("AAPL", 150000.0)   # Over-allocated
        manager.set_position_value("GOOGL", 100000.0)  # Under-allocated
        
        metrics = manager.get_cycle_metrics()
        
        # Total delta = sum of all deltas
        expected_total_delta = (
            (200000.0 - 150000.0) +  # AAPL: +50K
            (150000.0 - 100000.0) +  # GOOGL: +50K
            (100000.0 - 0.0) +       # MSFT: +100K
            (20000.0 - 0.0)          # SMALL: +20K
        )  # = 220K
        
        assert abs(metrics.total_delta - expected_total_delta) < 1.0

    def test_risk_metrics(self, manager_with_metrics_data):
        """Test risk-related metrics"""
        manager, cycle = manager_with_metrics_data
        
        metrics = manager.get_cycle_metrics()
        
        # Concentration risk should be the largest weight
        assert metrics.concentration_risk == 0.20  # AAPL allocation
        
        # Diversification score should be inverse of concentration
        assert abs(metrics.diversification_score - 0.80) < 0.001

    def test_metrics_update_on_allocation_changes(self, manager_with_metrics_data):
        """Test that metrics update when allocations change"""
        manager, cycle = manager_with_metrics_data
        
        initial_metrics = manager.get_cycle_metrics()
        initial_count = initial_metrics.number_of_allocations
        
        # Add new allocation
        manager.add_allocation("NEW_STOCK", 0.05)
        
        updated_metrics = manager.get_cycle_metrics()
        assert updated_metrics.number_of_allocations == initial_count + 1

    def test_metrics_after_approval_and_execution(self, manager_with_metrics_data):
        """Test metrics after allocation approval and execution"""
        manager, cycle = manager_with_metrics_data
        
        # Approve some allocations
        allocations = list(cycle.allocations.values())
        manager.approve_allocations([allocations[0].id, allocations[1].id])
        
        metrics = manager.get_cycle_metrics()
        assert metrics.approved_allocations == 2
        
        # Move to execution phase and execute
        while cycle.phase != CyclePhase.EXECUTION:
            manager.advance_phase()
            
        manager.execute_allocations([allocations[0].id])
        
        metrics = manager.get_cycle_metrics()
        assert metrics.executed_allocations == 1


class TestMarketTiming:
    """Test market timing and scheduling functionality"""
    
    def test_market_open_check(self):
        """Test market open time checking"""
        manager = create_mock_weekly_cycle_manager()
        
        # Test during market hours (Tuesday 2 PM)
        market_time = datetime(2024, 6, 4, 14, 0)  # Tuesday 2:00 PM
        assert manager.is_market_open(market_time) is True
        
        # Test outside market hours (early morning)
        early_time = datetime(2024, 6, 4, 6, 0)  # Tuesday 6:00 AM
        assert manager.is_market_open(early_time) is False
        
        # Test on weekend
        weekend_time = datetime(2024, 6, 8, 14, 0)  # Saturday 2:00 PM
        assert manager.is_market_open(weekend_time) is False

    def test_time_to_next_cycle(self):
        """Test calculating time until next cycle"""
        manager = create_mock_weekly_cycle_manager()
        
        # Start and complete a cycle
        cycle = manager.start_new_cycle()
        cycle.phase = CyclePhase.CLOSED
        manager._complete_cycle()
        
        # Calculate time to next cycle
        time_to_next = manager.time_to_next_cycle()
        
        assert time_to_next is not None
        assert isinstance(time_to_next, timedelta)
        assert time_to_next.total_seconds() > 0

    def test_time_to_next_cycle_with_active_cycle(self):
        """Test time to next cycle when current cycle is active"""
        manager = create_mock_weekly_cycle_manager()
        
        # Start cycle (still active)
        manager.start_new_cycle()
        
        # Should return None since cycle is still active
        time_to_next = manager.time_to_next_cycle()
        assert time_to_next is None

    def test_cycle_duration_tracking(self):
        """Test that cycle duration is tracked"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={"preparation": 0.01}
        )
        
        cycle = manager.start_new_cycle()
        
        # Advance through all phases quickly
        while cycle.phase != CyclePhase.CLOSED:
            manager.advance_phase()
            
        # Check that duration was recorded
        completed_cycle = manager.cycle_history[0]
        assert "duration_seconds" in completed_cycle.metadata
        assert completed_cycle.metadata["duration_seconds"] > 0


class TestPerformanceTracking:
    """Test performance metrics and tracking"""
    
    def test_cycle_completion_tracking(self):
        """Test tracking of completed vs failed cycles"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={"preparation": 0.01}
        )
        
        # Complete successful cycle
        cycle1 = manager.start_new_cycle()
        while cycle1.phase != CyclePhase.CLOSED:
            manager.advance_phase()
            
        assert manager.completed_cycles == 1
        assert manager.failed_cycles == 0
        
        # Start another cycle
        cycle2 = manager.start_new_cycle()
        
        assert manager.total_cycles == 2

    def test_average_cycle_duration_calculation(self):
        """Test average cycle duration calculation"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={"preparation": 0.01}
        )
        
        # Complete multiple cycles
        for _ in range(3):
            cycle = manager.start_new_cycle()
            while cycle.phase != CyclePhase.CLOSED:
                manager.advance_phase()
                
        assert manager.avg_cycle_duration > 0
        assert len(manager.cycle_history) == 3

    def test_performance_summary(self):
        """Test performance summary generation"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(500000.0)
        manager.set_position_value("AAPL", 100000.0)
        
        cycle = manager.start_new_cycle()
        
        summary = manager.get_performance_summary()
        
        assert summary["total_cycles"] >= 1
        assert summary["portfolio_value"] == 500000.0
        assert summary["active_positions"] == 1
        assert summary["current_cycle_phase"] == CyclePhase.PREPARATION.value
        assert "success_rate" in summary


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent operations"""
    
    def test_concurrent_allocation_operations(self):
        """Test concurrent allocation operations"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        manager.start_new_cycle()
        
        def add_allocations(prefix):
            for i in range(5):
                try:
                    symbol = f"{prefix}_{i}"
                    manager.add_allocation(symbol, 0.02)
                except MockWeeklyCycleError:
                    pass  # Expected with concurrent access
                    
        # Start multiple threads
        threads = [threading.Thread(target=add_allocations, args=(f"THREAD{i}",)) for i in range(3)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Manager should be in consistent state
        assert len(manager.current_cycle.allocations) <= 15  # Maximum possible

    def test_concurrent_phase_advancement(self):
        """Test concurrent phase advancement attempts"""
        manager = create_mock_weekly_cycle_manager(phase_delays={"preparation": 0.05})
        manager.start_new_cycle()
        
        results = []
        
        def advance_phase():
            try:
                result = manager.advance_phase()
                results.append(result)
            except MockWeeklyCycleError:
                results.append(False)
                
        # Start multiple threads trying to advance phase
        threads = [threading.Thread(target=advance_phase) for _ in range(5)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Only one should succeed
        successful_advances = sum(1 for r in results if r is True)
        assert successful_advances == 1

    def test_concurrent_delta_calculations(self):
        """Test concurrent delta calculations"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        manager.start_new_cycle()
        
        # Add allocations
        for i in range(5):
            manager.add_allocation(f"STOCK_{i}", 0.1)
            
        calculation_results = []
        
        def calculate_deltas():
            deltas = manager.calculate_deltas()
            calculation_results.append(len(deltas))
            
        # Start multiple calculation threads
        threads = [threading.Thread(target=calculate_deltas) for _ in range(3)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All should get consistent results
        assert all(count == 5 for count in calculation_results)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_operations_without_cycle(self):
        """Test operations that require active cycle"""
        manager = create_mock_weekly_cycle_manager()
        
        # All these should fail without active cycle
        with pytest.raises(MockWeeklyCycleError):
            manager.add_allocation("AAPL", 0.10)
            
        assert manager.get_allocations() == []
        assert manager.get_cycle_metrics() is None
        assert manager.calculate_deltas() == {}

    def test_reset_functionality(self):
        """Test manager reset for testing"""
        manager = create_mock_weekly_cycle_manager(auto_advance_phases=True)
        manager.set_portfolio_value(500000.0)
        manager.set_position_value("AAPL", 100000.0)
        
        # Create some state
        cycle = manager.start_new_cycle()
        manager.add_allocation("AAPL", 0.10)
        
        # Verify state exists
        assert manager.current_cycle is not None
        assert len(manager.position_values) > 0
        assert manager.total_cycles > 0
        
        # Reset
        manager.reset_for_testing()
        
        # Verify clean state
        assert manager.current_cycle is None
        assert len(manager.position_values) == 0
        assert manager.total_cycles == 0
        assert manager.portfolio_value == 1000000.0  # Reset to default

    def test_edge_case_zero_portfolio_value(self):
        """Test handling zero portfolio value"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(0.0)
        
        manager.start_new_cycle()
        
        # Should handle gracefully
        allocation = manager.add_allocation("AAPL", 0.10)
        assert allocation.target_value == 0.0
        assert allocation.current_weight == 0.0

    def test_edge_case_very_small_amounts(self):
        """Test handling very small amounts"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(100.0)  # Very small portfolio
        
        manager.start_new_cycle()
        allocation = manager.add_allocation("AAPL", 0.01)  # 1%
        
        assert allocation.target_value == 1.0
        assert allocation.target_weight == 0.01


@pytest.mark.integration
class TestWeeklyCycleIntegrationScenarios:
    """Integration test scenarios combining multiple weekly cycle features"""
    
    def test_complete_weekly_cycle_workflow(self):
        """Test complete weekly cycle workflow from start to finish"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={phase.value: 0.01 for phase in CyclePhase}
        )
        manager.set_portfolio_value(1000000.0)
        
        # Set current positions
        manager.set_position_value("AAPL", 50000.0)
        manager.set_position_value("GOOGL", 80000.0)
        
        # 1. Start new cycle
        cycle = manager.start_new_cycle()
        assert cycle.phase == CyclePhase.PREPARATION
        
        # 2. Add allocations during early phases
        aapl_alloc = manager.add_allocation("AAPL", 0.15)  # Target $150K
        googl_alloc = manager.add_allocation("GOOGL", 0.20) # Target $200K
        msft_alloc = manager.add_allocation("MSFT", 0.10)   # Target $100K
        
        # 3. Advance through analysis phase
        manager.advance_phase()
        assert cycle.phase == CyclePhase.ANALYSIS
        
        # 4. Move to allocation phase and approve allocations
        manager.advance_phase()
        assert cycle.phase == CyclePhase.ALLOCATION
        
        # Approve high-priority allocations
        approved_count = manager.approve_allocations([aapl_alloc.id, googl_alloc.id])
        assert approved_count == 2
        
        # 5. Execute allocations
        manager.advance_phase()
        assert cycle.phase == CyclePhase.EXECUTION
        
        executed_count = manager.execute_allocations()
        assert executed_count >= 1  # At least some should execute
        
        # 6. Review phase
        manager.advance_phase()
        assert cycle.phase == CyclePhase.REVIEW
        
        # 7. Complete cycle
        manager.advance_phase()
        assert cycle.phase == CyclePhase.CLOSED
        
        # Verify cycle completion
        assert manager.current_cycle is None
        assert len(manager.cycle_history) == 1
        assert manager.completed_cycles == 1
        
        # Verify metrics were calculated
        completed_cycle = manager.cycle_history[0]
        assert completed_cycle.metrics.number_of_allocations == 3
        assert "duration_seconds" in completed_cycle.metadata

    def test_risk_management_scenario(self):
        """Test risk management through allocation limits"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        
        cycle = manager.start_new_cycle()
        
        # Try to create over-concentrated portfolio
        large_allocations = []
        for i, weight in enumerate([0.30, 0.25, 0.20, 0.15, 0.10]):  # Total = 100%
            symbol = f"STOCK_{i}"
            allocation = manager.add_allocation(symbol, weight)
            large_allocations.append(allocation)
            
        # Check concentration risk
        metrics = manager.get_cycle_metrics()
        assert metrics.concentration_risk == 0.30  # Largest allocation
        assert metrics.diversification_score == 0.70  # 1 - concentration
        
        # Verify total allocation doesn't exceed 100%
        total_weight = sum(a.target_weight for a in large_allocations)
        assert total_weight == 1.0

    def test_multi_cycle_scenario(self):
        """Test running multiple cycles in sequence"""
        manager = create_mock_weekly_cycle_manager(
            phase_delays={phase.value: 0.01 for phase in CyclePhase}
        )
        manager.set_portfolio_value(1000000.0)
        
        cycles_to_run = 3
        
        for cycle_num in range(cycles_to_run):
            # Start new cycle
            cycle = manager.start_new_cycle()
            
            # Add different allocations each cycle
            symbols = [f"CYCLE{cycle_num}_STOCK{i}" for i in range(3)]
            for i, symbol in enumerate(symbols):
                weight = 0.05 + (i * 0.02)  # Varying weights
                manager.add_allocation(symbol, weight)
                
            # Complete cycle quickly
            while cycle.phase != CyclePhase.CLOSED:
                if cycle.phase == CyclePhase.ALLOCATION:
                    # Approve all allocations
                    allocations = manager.get_allocations()
                    manager.approve_allocations([a.id for a in allocations])
                    
                manager.advance_phase()
                
        # Verify all cycles completed
        assert len(manager.cycle_history) == cycles_to_run
        assert manager.completed_cycles == cycles_to_run
        assert manager.total_cycles == cycles_to_run

    def test_high_frequency_allocation_updates(self):
        """Test rapid allocation updates during cycle"""
        manager = create_mock_weekly_cycle_manager()
        manager.set_portfolio_value(1000000.0)
        
        cycle = manager.start_new_cycle()
        
        # Add initial allocation
        allocation = manager.add_allocation("DYNAMIC_STOCK", 0.10)
        
        # Rapidly update allocation multiple times
        target_weights = [0.12, 0.08, 0.15, 0.05, 0.20, 0.10]
        
        for weight in target_weights:
            manager.update_allocation(allocation.id, target_weight=weight)
            
            # Verify update took effect
            assert allocation.target_weight == weight
            assert allocation.target_value == manager.portfolio_value * weight
            
        # Final verification
        final_metrics = manager.get_cycle_metrics()
        assert final_metrics.target_portfolio_value == manager.portfolio_value * 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])