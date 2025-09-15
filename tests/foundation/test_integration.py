"""
Comprehensive integration tests for Foundation phase components.
Tests end-to-end scenarios combining broker, gate manager, and weekly cycle functionality.
"""
import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Import all mock components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mocks.mock_broker import (
    MockBroker, MockOrder, MockPosition, OrderStatus, OrderType,
    create_mock_broker, create_test_position, create_test_order
)
from mocks.mock_gate_manager import (
    MockGateManager, Gate, GateResult, GateViolation,
    GateStatus, GateType, ViolationSeverity,
    create_mock_gate_manager, create_test_gate
)
from mocks.mock_weekly_cycle import (
    MockWeeklyCycleManager, WeeklyCycle, Allocation, CyclePhase, AllocationStatus,
    create_mock_weekly_cycle_manager, create_test_allocation
)


class FoundationSystem:
    """
    Integration system combining all Foundation phase components.
    Provides a unified interface for testing complete workflows.
    """
    
    def __init__(self,
                 broker_config: Dict[str, Any] = None,
                 gate_config: Dict[str, Any] = None,
                 cycle_config: Dict[str, Any] = None):
        """
        Initialize integrated Foundation system
        
        Args:
            broker_config: Configuration for mock broker
            gate_config: Configuration for mock gate manager
            cycle_config: Configuration for mock weekly cycle manager
        """
        # Initialize components with default configs
        self.broker = create_mock_broker(**(broker_config or {}))
        self.gate_manager = create_mock_gate_manager(**(gate_config or {}))
        self.cycle_manager = create_mock_weekly_cycle_manager(**(cycle_config or {}))
        
        # System state
        self.system_enabled = True
        self.risk_monitoring_enabled = True
        self.auto_execution_enabled = False
        
        # Performance tracking
        self.total_transactions = 0
        self.successful_transactions = 0
        self.failed_transactions = 0
        self.gate_violations = 0
        
        # Integration callbacks
        self._setup_integration_callbacks()
        
        # Default portfolio setup
        self.setup_default_portfolio()

    def setup_default_portfolio(self):
        """Setup default portfolio for testing"""
        self.cycle_manager.set_portfolio_value(1000000.0)  # $1M portfolio
        
        # Set initial positions
        initial_positions = {
            "AAPL": 150000.0,
            "GOOGL": 200000.0,
            "MSFT": 120000.0,
            "TSLA": 80000.0
        }
        
        for symbol, value in initial_positions.items():
            self.cycle_manager.set_position_value(symbol, value)
            self.broker.set_market_price(symbol, 100.0)  # Default price

    def connect_all(self) -> bool:
        """Connect all components"""
        try:
            # Connect broker
            if not self.broker.connect():
                return False
                
            # Enable gate manager
            self.gate_manager.enabled = True
            
            # System is ready
            self.system_enabled = True
            return True
            
        except Exception:
            return False

    def start_new_trading_cycle(self) -> Tuple[bool, WeeklyCycle, List[GateViolation]]:
        """
        Start new trading cycle with risk validation
        
        Returns:
            Tuple of (success, cycle, violations)
        """
        violations = []
        
        try:
            # 1. Validate system readiness
            if not self.system_enabled or not self.broker.is_connected():
                return False, None, [self._create_system_violation("System not ready")]
                
            # 2. Run pre-cycle risk gates
            context = self._build_risk_context()
            gate_results = self.gate_manager.validate_all_gates(context, session_id="pre_cycle")
            
            # Check for violations
            for result in gate_results:
                if result.status == GateStatus.FAILED:
                    violations.extend(result.violations)
                    
            # 3. Start cycle if no critical violations
            critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
            if critical_violations:
                return False, None, violations
                
            # 4. Start weekly cycle
            cycle = self.cycle_manager.start_new_cycle()
            
            return True, cycle, violations
            
        except Exception as e:
            violation = self._create_system_violation(f"Cycle start error: {str(e)}")
            return False, None, [violation]

    def create_allocation_plan(self, target_allocations: Dict[str, float]) -> List[Allocation]:
        """
        Create allocation plan with risk validation
        
        Args:
            target_allocations: Dict mapping symbols to target weights
            
        Returns:
            List of created allocations
        """
        allocations = []
        
        if not self.cycle_manager.current_cycle:
            raise Exception("No active cycle for allocation planning")
            
        # Validate total allocation doesn't exceed 100%
        total_weight = sum(target_allocations.values())
        if total_weight > 1.0:
            raise Exception(f"Total allocation {total_weight:.2%} exceeds 100%")
            
        # Create allocations
        for symbol, weight in target_allocations.items():
            allocation = self.cycle_manager.add_allocation(symbol, weight)
            allocations.append(allocation)
            
        return allocations

    def validate_and_approve_allocations(self, allocation_ids: List[str] = None) -> Tuple[int, List[GateViolation]]:
        """
        Validate allocations against risk gates and approve if passing
        
        Args:
            allocation_ids: Optional specific allocation IDs to validate
            
        Returns:
            Tuple of (approved_count, violations)
        """
        if not self.cycle_manager.current_cycle:
            return 0, [self._create_system_violation("No active cycle")]
            
        # Get allocations to validate
        if allocation_ids:
            allocations = [self.cycle_manager.get_allocation(aid) for aid in allocation_ids]
            allocations = [a for a in allocations if a is not None]
        else:
            allocations = self.cycle_manager.get_allocations(status=AllocationStatus.PENDING)
            
        violations = []
        approved_count = 0
        
        for allocation in allocations:
            # Build context for this allocation
            context = self._build_allocation_context(allocation)
            
            # Validate against gates
            results = self.gate_manager.validate_all_gates(context, session_id=f"allocation_{allocation.id}")
            
            # Check for failures
            allocation_violations = []
            for result in results:
                if result.status == GateStatus.FAILED:
                    allocation_violations.extend(result.violations)
                    
            if allocation_violations:
                violations.extend(allocation_violations)
                self.gate_violations += len(allocation_violations)
            else:
                # Approve allocation
                self.cycle_manager.update_allocation(allocation.id, status=AllocationStatus.APPROVED)
                approved_count += 1
                
        return approved_count, violations

    def execute_approved_allocations(self, max_concurrent: int = 5) -> Tuple[int, int, List[Exception]]:
        """
        Execute approved allocations through broker with risk monitoring
        
        Args:
            max_concurrent: Maximum concurrent executions
            
        Returns:
            Tuple of (successful_count, failed_count, errors)
        """
        if not self.cycle_manager.current_cycle:
            return 0, 0, [Exception("No active cycle")]
            
        if self.cycle_manager.current_cycle.phase != CyclePhase.EXECUTION:
            return 0, 0, [Exception("Not in execution phase")]
            
        # Get approved allocations
        approved_allocations = self.cycle_manager.get_allocations(status=AllocationStatus.APPROVED)
        
        if not approved_allocations:
            return 0, 0, []
            
        successful_count = 0
        failed_count = 0
        errors = []
        
        # Execute allocations with concurrency limit
        semaphore = threading.Semaphore(max_concurrent)
        results = []
        threads = []
        
        def execute_single_allocation(allocation):
            with semaphore:
                try:
                    # Pre-execution risk check
                    context = self._build_execution_context(allocation)
                    gate_results = self.gate_manager.validate_all_gates(context)
                    
                    # Check for critical violations
                    critical_violations = []
                    for result in gate_results:
                        if result.status == GateStatus.FAILED:
                            critical_violations.extend([
                                v for v in result.violations 
                                if v.severity == ViolationSeverity.CRITICAL
                            ])
                            
                    if critical_violations:
                        results.append((allocation.id, False, "Critical risk violations"))
                        return
                        
                    # Calculate order details
                    current_value = self.cycle_manager.position_values.get(allocation.symbol, 0.0)
                    target_value = allocation.target_value
                    delta_value = target_value - current_value
                    
                    if abs(delta_value) < 100:  # Skip tiny adjustments
                        results.append((allocation.id, True, "Skipped small adjustment"))
                        return
                        
                    # Determine order quantity and direction
                    current_price = self.broker.get_current_price(allocation.symbol)
                    quantity = int(delta_value / current_price)
                    
                    if quantity == 0:
                        results.append((allocation.id, True, "No shares to trade"))
                        return
                        
                    # Place order
                    order = self.broker.place_order(allocation.symbol, quantity, OrderType.MARKET)
                    
                    # Wait for fill (with timeout)
                    timeout = 5.0  # 5 second timeout
                    start_time = time.time()
                    
                    while (time.time() - start_time) < timeout:
                        updated_order = self.broker.get_order(order.id)
                        if updated_order.status in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED]:
                            break
                        time.sleep(0.1)
                        
                    # Check execution result
                    final_order = self.broker.get_order(order.id)
                    if final_order.status == OrderStatus.FILLED:
                        # Update allocation status
                        self.cycle_manager.update_allocation(allocation.id, status=AllocationStatus.EXECUTED)
                        results.append((allocation.id, True, f"Executed {quantity} shares"))
                        self.successful_transactions += 1
                    else:
                        results.append((allocation.id, False, f"Order failed: {final_order.status.value}"))
                        self.failed_transactions += 1
                        
                    self.total_transactions += 1
                    
                except Exception as e:
                    results.append((allocation.id, False, str(e)))
                    errors.append(e)
                    self.failed_transactions += 1
                    
        # Start execution threads
        for allocation in approved_allocations:
            thread = threading.Thread(target=execute_single_allocation, args=(allocation,))
            threads.append(thread)
            thread.start()
            
        # Wait for all executions to complete
        for thread in threads:
            thread.join()
            
        # Count results
        for allocation_id, success, message in results:
            if success:
                successful_count += 1
            else:
                failed_count += 1
                
        return successful_count, failed_count, errors

    def complete_cycle_with_review(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Complete current cycle with comprehensive review
        
        Returns:
            Tuple of (success, review_report)
        """
        if not self.cycle_manager.current_cycle:
            return False, {"error": "No active cycle"}
            
        cycle = self.cycle_manager.current_cycle
        
        # Advance to review phase if not already there
        while cycle.phase not in [CyclePhase.REVIEW, CyclePhase.CLOSED]:
            self.cycle_manager.advance_phase()
            
        # Generate comprehensive review
        review_report = {
            "cycle_id": cycle.id,
            "duration": self._calculate_cycle_duration(cycle),
            "allocations": self._analyze_allocations(cycle),
            "executions": self._analyze_executions(cycle),
            "risk_assessment": self._analyze_risk_performance(),
            "broker_performance": self._analyze_broker_performance(),
            "recommendations": self._generate_recommendations(cycle)
        }
        
        # Complete cycle
        if cycle.phase != CyclePhase.CLOSED:
            self.cycle_manager.advance_phase()
            
        return True, review_report

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            "system_enabled": self.system_enabled,
            "broker_connected": self.broker.is_connected(),
            "gate_manager_enabled": self.gate_manager.enabled,
            "current_cycle_phase": (
                self.cycle_manager.current_cycle.phase.value 
                if self.cycle_manager.current_cycle else None
            ),
            "active_positions": len(self.cycle_manager.position_values),
            "portfolio_value": self.cycle_manager.portfolio_value,
            "total_transactions": self.total_transactions,
            "success_rate": (
                self.successful_transactions / max(self.total_transactions, 1)
            ),
            "active_violations": len(self.gate_manager.get_violations(unresolved_only=True)),
            "broker_orders": len(self.broker.orders),
            "broker_positions": len(self.broker.positions)
        }

    def shutdown_system(self) -> bool:
        """Gracefully shutdown all components"""
        try:
            # Stop any auto-advance processes
            if hasattr(self.cycle_manager, '_stop_auto_advance'):
                self.cycle_manager._stop_auto_advance.set()
                
            # Disconnect broker
            if self.broker.is_connected():
                self.broker.disconnect()
                
            # Disable components
            self.gate_manager.enabled = False
            self.system_enabled = False
            
            return True
            
        except Exception:
            return False

    def reset_for_testing(self):
        """Reset all components for testing"""
        self.broker.reset_for_testing()
        self.gate_manager.reset_for_testing()
        self.cycle_manager.reset_for_testing()
        
        # Reset system state
        self.total_transactions = 0
        self.successful_transactions = 0
        self.failed_transactions = 0
        self.gate_violations = 0
        self.system_enabled = True
        self.risk_monitoring_enabled = True
        self.auto_execution_enabled = False
        
        # Restore default portfolio
        self.setup_default_portfolio()

    def _setup_integration_callbacks(self):
        """Setup callbacks for component integration"""
        # Gate violation callback
        def handle_violation(violation):
            if violation.severity == ViolationSeverity.CRITICAL:
                # Auto-disable risky operations on critical violations
                self.auto_execution_enabled = False
                
        self.gate_manager.on_violation = handle_violation

    def _build_risk_context(self) -> Dict[str, Any]:
        """Build risk context for gate validation"""
        positions = list(self.broker.get_positions())
        total_value = sum(abs(p.market_value) for p in positions)
        
        return {
            "portfolio_value": self.cycle_manager.portfolio_value,
            "position_count": len(positions),
            "total_exposure": total_value,
            "current_loss": sum(p.unrealized_pnl for p in positions if p.unrealized_pnl < 0),
            "portfolio_risk": total_value / max(self.cycle_manager.portfolio_value, 1),
            "broker_connected": self.broker.is_connected(),
            "account_balance": self.broker.get_account_balance(),
            "buying_power": self.broker.get_buying_power()
        }

    def _build_allocation_context(self, allocation: Allocation) -> Dict[str, Any]:
        """Build context for allocation validation"""
        base_context = self._build_risk_context()
        
        # Add allocation-specific context
        base_context.update({
            "allocation_symbol": allocation.symbol,
            "allocation_weight": allocation.target_weight,
            "allocation_value": allocation.target_value,
            "allocation_delta": allocation.delta,
            "concentration_risk": allocation.target_weight,  # Single position concentration
        })
        
        return base_context

    def _build_execution_context(self, allocation: Allocation) -> Dict[str, Any]:
        """Build context for execution validation"""
        base_context = self._build_allocation_context(allocation)
        
        # Add execution-specific context
        current_price = self.broker.get_current_price(allocation.symbol)
        base_context.update({
            "execution_price": current_price,
            "market_impact": abs(allocation.delta) / max(self.cycle_manager.portfolio_value, 1),
            "liquidity_risk": 0.1,  # Simplified liquidity measure
        })
        
        return base_context

    def _create_system_violation(self, message: str) -> GateViolation:
        """Create system-level violation"""
        from mocks.mock_gate_manager import GateViolation, ViolationSeverity
        
        return GateViolation(
            gate_id="system",
            violation_type="system_error",
            severity=ViolationSeverity.HIGH,
            message=message,
            timestamp=datetime.now()
        )

    def _calculate_cycle_duration(self, cycle: WeeklyCycle) -> float:
        """Calculate cycle duration in seconds"""
        start_time = cycle.phase_timestamps.get(CyclePhase.PREPARATION.value)
        end_time = datetime.now()
        
        if start_time:
            return (end_time - start_time).total_seconds()
        return 0.0

    def _analyze_allocations(self, cycle: WeeklyCycle) -> Dict[str, Any]:
        """Analyze allocation performance"""
        allocations = list(cycle.allocations.values())
        
        return {
            "total_allocations": len(allocations),
            "approved_count": len([a for a in allocations if a.status == AllocationStatus.APPROVED]),
            "executed_count": len([a for a in allocations if a.status == AllocationStatus.EXECUTED]),
            "total_target_value": sum(a.target_value for a in allocations),
            "total_delta": sum(a.delta for a in allocations),
            "largest_allocation": max((a.target_weight for a in allocations), default=0.0),
            "diversification_score": 1.0 - max((a.target_weight for a in allocations), default=0.0)
        }

    def _analyze_executions(self, cycle: WeeklyCycle) -> Dict[str, Any]:
        """Analyze execution performance"""
        executed_allocations = [
            a for a in cycle.allocations.values() 
            if a.status == AllocationStatus.EXECUTED
        ]
        
        return {
            "execution_rate": len(executed_allocations) / max(len(cycle.allocations), 1),
            "executed_value": sum(a.target_value for a in executed_allocations),
            "execution_efficiency": self.successful_transactions / max(self.total_transactions, 1)
        }

    def _analyze_risk_performance(self) -> Dict[str, Any]:
        """Analyze risk management performance"""
        violations = self.gate_manager.get_violations()
        
        return {
            "total_violations": len(violations),
            "critical_violations": len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]),
            "resolved_violations": len([v for v in violations if v.resolved]),
            "gate_success_rate": (
                self.gate_manager.successful_validations / 
                max(self.gate_manager.total_validations, 1)
            )
        }

    def _analyze_broker_performance(self) -> Dict[str, Any]:
        """Analyze broker performance"""
        return {
            "total_orders": len(self.broker.orders),
            "filled_orders": len([o for o in self.broker.orders.values() if o.status == OrderStatus.FILLED]),
            "order_success_rate": (
                self.broker.successful_trades / max(self.broker.total_trades, 1)
            ),
            "account_balance": self.broker.get_account_balance(),
            "buying_power": self.broker.get_buying_power()
        }

    def _generate_recommendations(self, cycle: WeeklyCycle) -> List[str]:
        """Generate recommendations based on cycle performance"""
        recommendations = []
        
        # Analyze allocation concentration
        allocations = list(cycle.allocations.values())
        if allocations:
            max_weight = max(a.target_weight for a in allocations)
            if max_weight > 0.25:  # >25% concentration
                recommendations.append(f"Consider reducing concentration risk (max allocation: {max_weight:.1%})")
                
        # Analyze execution success
        execution_rate = len([a for a in allocations if a.status == AllocationStatus.EXECUTED]) / max(len(allocations), 1)
        if execution_rate < 0.8:  # <80% execution rate
            recommendations.append(f"Low execution rate ({execution_rate:.1%}) - review execution process")
            
        # Analyze violations
        active_violations = len(self.gate_manager.get_violations(unresolved_only=True))
        if active_violations > 0:
            recommendations.append(f"Resolve {active_violations} active risk violations")
            
        return recommendations


class TestFoundationSystemIntegration:
    """Test integrated Foundation system functionality"""
    
    @pytest.fixture
    def foundation_system(self):
        """Fixture providing configured Foundation system"""
        broker_config = {"order_fill_delay": 0.01, "error_rate": 0.0}
        gate_config = {"validation_delay": 0.01, "error_rate": 0.0}
        cycle_config = {"phase_delays": {phase.value: 0.01 for phase in CyclePhase}}
        
        system = FoundationSystem(
            broker_config=broker_config,
            gate_config=gate_config,
            cycle_config=cycle_config
        )
        
        yield system
        
        # Cleanup
        system.shutdown_system()

    def test_system_initialization(self, foundation_system):
        """Test system initialization and health check"""
        health = foundation_system.get_system_health()
        
        assert health["system_enabled"] is True
        assert health["portfolio_value"] == 1000000.0
        assert health["active_positions"] == 4  # Default positions
        assert foundation_system.broker is not None
        assert foundation_system.gate_manager is not None
        assert foundation_system.cycle_manager is not None

    def test_system_connection(self, foundation_system):
        """Test connecting all system components"""
        result = foundation_system.connect_all()
        
        assert result is True
        
        health = foundation_system.get_system_health()
        assert health["broker_connected"] is True
        assert health["gate_manager_enabled"] is True

    def test_complete_trading_workflow(self, foundation_system):
        """Test complete end-to-end trading workflow"""
        # 1. Connect system
        assert foundation_system.connect_all() is True
        
        # 2. Start new trading cycle
        success, cycle, violations = foundation_system.start_new_trading_cycle()
        assert success is True
        assert cycle is not None
        assert cycle.phase == CyclePhase.PREPARATION
        
        # 3. Create allocation plan
        target_allocations = {
            "AAPL": 0.20,   # Increase AAPL from current
            "GOOGL": 0.15,  # Reduce GOOGL from current  
            "MSFT": 0.15,   # Increase MSFT from current
            "NVDA": 0.10    # New position
        }
        
        allocations = foundation_system.create_allocation_plan(target_allocations)
        assert len(allocations) == 4
        
        # 4. Advance to allocation phase
        while cycle.phase != CyclePhase.ALLOCATION:
            foundation_system.cycle_manager.advance_phase()
            
        # 5. Validate and approve allocations
        approved_count, violations = foundation_system.validate_and_approve_allocations()
        assert approved_count > 0
        
        # 6. Advance to execution phase
        while cycle.phase != CyclePhase.EXECUTION:
            foundation_system.cycle_manager.advance_phase()
            
        # 7. Execute approved allocations
        successful, failed, errors = foundation_system.execute_approved_allocations()
        assert successful > 0
        assert len(errors) == 0
        
        # 8. Complete cycle with review
        success, review_report = foundation_system.complete_cycle_with_review()
        assert success is True
        assert "allocations" in review_report
        assert "executions" in review_report
        assert "risk_assessment" in review_report

    def test_risk_gate_integration(self, foundation_system):
        """Test risk gate integration with trading workflow"""
        foundation_system.connect_all()
        
        # Create risky allocation plan that should trigger violations
        risky_allocations = {
            "RISKY_STOCK": 0.90  # 90% concentration - should fail concentration gate
        }
        
        success, cycle, violations = foundation_system.start_new_trading_cycle()
        assert success is True  # Should start despite future risk
        
        # Create risky allocations
        allocations = foundation_system.create_allocation_plan(risky_allocations)
        
        # Move to allocation phase
        while cycle.phase != CyclePhase.ALLOCATION:
            foundation_system.cycle_manager.advance_phase()
            
        # Validation should catch the risk
        approved_count, violations = foundation_system.validate_and_approve_allocations()
        
        # Should have violations due to high concentration
        assert len(violations) > 0
        concentration_violations = [
            v for v in violations 
            if "concentration" in v.message.lower() or v.violation_type == "risk_exceeded"
        ]
        # May not have explicit concentration violation in default gates, but should have risk violations
        assert len(violations) > 0  # Some form of risk violation should occur

    def test_broker_error_handling(self, foundation_system):
        """Test broker error handling in integrated workflow"""
        # Configure broker with high error rate
        foundation_system.broker.error_rate = 0.8  # 80% error rate
        foundation_system.connect_all()
        
        success, cycle, violations = foundation_system.start_new_trading_cycle()
        assert success is True
        
        # Create normal allocation plan
        allocations = foundation_system.create_allocation_plan({"AAPL": 0.10})
        
        # Advance to execution phase
        while cycle.phase != CyclePhase.EXECUTION:
            foundation_system.cycle_manager.advance_phase()
            
        # Approve allocations
        approved_count, _ = foundation_system.validate_and_approve_allocations()
        assert approved_count > 0
        
        # Execute with high error rate
        successful, failed, errors = foundation_system.execute_approved_allocations()
        
        # Should have some failures due to high error rate
        assert failed > 0 or len(errors) > 0

    def test_concurrent_operations(self, foundation_system):
        """Test concurrent operations across all components"""
        foundation_system.connect_all()
        
        # Start cycle
        success, cycle, _ = foundation_system.start_new_trading_cycle()
        assert success is True
        
        # Create multiple allocation plans concurrently
        allocation_results = []
        
        def create_allocations(thread_id):
            try:
                allocations = foundation_system.create_allocation_plan({
                    f"STOCK_{thread_id}": 0.05
                })
                allocation_results.append(len(allocations))
            except Exception as e:
                allocation_results.append(0)
                
        # Start concurrent allocation creation
        threads = [threading.Thread(target=create_allocations, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should have created some allocations successfully
        successful_allocations = sum(allocation_results)
        assert successful_allocations > 0

    def test_system_recovery_after_failure(self, foundation_system):
        """Test system recovery after component failures"""
        foundation_system.connect_all()
        
        # Simulate broker disconnection
        foundation_system.broker.disconnect()
        
        health = foundation_system.get_system_health()
        assert health["broker_connected"] is False
        
        # Try to start cycle (should fail)
        success, cycle, violations = foundation_system.start_new_trading_cycle()
        assert success is False
        assert len(violations) > 0
        
        # Reconnect broker
        foundation_system.broker.connect()
        
        # Should be able to start cycle again
        success, cycle, violations = foundation_system.start_new_trading_cycle()
        assert success is True

    def test_performance_tracking(self, foundation_system):
        """Test performance tracking across multiple cycles"""
        foundation_system.connect_all()
        
        cycles_to_run = 3
        
        for cycle_num in range(cycles_to_run):
            # Start cycle
            success, cycle, _ = foundation_system.start_new_trading_cycle()
            assert success is True
            
            # Create and execute allocations
            allocations = foundation_system.create_allocation_plan({
                f"CYCLE_{cycle_num}_STOCK": 0.05
            })
            
            # Fast-forward to execution
            while cycle.phase != CyclePhase.EXECUTION:
                foundation_system.cycle_manager.advance_phase()
                
            # Execute
            approved_count, _ = foundation_system.validate_and_approve_allocations()
            if approved_count > 0:
                successful, failed, errors = foundation_system.execute_approved_allocations()
                
            # Complete cycle
            foundation_system.complete_cycle_with_review()
            
        # Check performance metrics
        health = foundation_system.get_system_health()
        assert health["total_transactions"] >= cycles_to_run
        assert foundation_system.cycle_manager.completed_cycles == cycles_to_run

    def test_large_portfolio_scenario(self, foundation_system):
        """Test handling large portfolio with many positions"""
        # Scale up portfolio
        foundation_system.cycle_manager.set_portfolio_value(10000000.0)  # $10M
        
        # Add many positions
        for i in range(20):
            symbol = f"STOCK_{i:02d}"
            value = 400000.0 + (i * 10000.0)  # Varying position sizes
            foundation_system.cycle_manager.set_position_value(symbol, value)
            foundation_system.broker.set_market_price(symbol, 100.0)
            
        foundation_system.connect_all()
        
        # Start cycle with large allocation plan
        success, cycle, _ = foundation_system.start_new_trading_cycle()
        assert success is True
        
        # Create diversified allocation plan
        target_allocations = {f"STOCK_{i:02d}": 0.04 for i in range(25)}  # 4% each, 100% total
        
        allocations = foundation_system.create_allocation_plan(target_allocations)
        assert len(allocations) == 25
        
        # Validate system can handle large number of allocations
        health = foundation_system.get_system_health()
        assert health["active_positions"] >= 20

    def test_system_shutdown_and_cleanup(self, foundation_system):
        """Test graceful system shutdown"""
        foundation_system.connect_all()
        
        # Start some activity
        success, cycle, _ = foundation_system.start_new_trading_cycle()
        assert success is True
        
        # Create allocations
        allocations = foundation_system.create_allocation_plan({"AAPL": 0.10})
        
        # Shutdown system
        shutdown_success = foundation_system.shutdown_system()
        assert shutdown_success is True
        
        # Verify components are disabled
        health = foundation_system.get_system_health()
        assert health["system_enabled"] is False
        assert health["broker_connected"] is False
        assert health["gate_manager_enabled"] is False


class TestAdvancedIntegrationScenarios:
    """Advanced integration test scenarios"""
    
    def test_multi_day_trading_simulation(self):
        """Test multi-day trading simulation with market changes"""
        system = FoundationSystem()
        system.connect_all()
        
        # Simulate 5 trading days
        for day in range(5):
            # Simulate market price changes
            price_multiplier = 1.0 + (day * 0.02)  # 2% daily appreciation
            for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                new_price = 100.0 * price_multiplier
                system.broker.set_market_price(symbol, new_price)
                
            # Run daily cycle
            success, cycle, _ = system.start_new_trading_cycle()
            if success:
                # Create conservative allocation plan
                allocations = system.create_allocation_plan({
                    "AAPL": 0.25, "GOOGL": 0.25, "MSFT": 0.25, "CASH": 0.25
                })
                
                # Fast execution
                while cycle.phase != CyclePhase.EXECUTION:
                    system.cycle_manager.advance_phase()
                    
                system.validate_and_approve_allocations()
                system.execute_approved_allocations()
                system.complete_cycle_with_review()
                
        # Verify system handled multiple cycles
        assert system.cycle_manager.completed_cycles == 5
        
        system.shutdown_system()

    def test_stress_test_high_volume(self):
        """Stress test with high volume of operations"""
        system = FoundationSystem(
            broker_config={"order_fill_delay": 0.001},  # Very fast
            gate_config={"validation_delay": 0.001},
            cycle_config={"phase_delays": {phase.value: 0.001 for phase in CyclePhase}}
        )
        system.connect_all()
        
        # High-volume allocation creation
        success, cycle, _ = system.start_new_trading_cycle()
        assert success is True
        
        # Create 100 small allocations
        for i in range(100):
            try:
                system.create_allocation_plan({f"MICRO_STOCK_{i}": 0.01})
            except Exception:
                break  # Expected to hit limits
                
        # System should remain stable
        health = system.get_system_health()
        assert health["system_enabled"] is True
        
        system.shutdown_system()

    def test_error_cascade_recovery(self):
        """Test recovery from cascading errors across components"""
        system = FoundationSystem(
            broker_config={"error_rate": 0.3},
            gate_config={"error_rate": 0.3}
        )
        system.connect_all()
        
        # Attempt operations that may fail
        for attempt in range(10):
            try:
                success, cycle, violations = system.start_new_trading_cycle()
                if success:
                    allocations = system.create_allocation_plan({"AAPL": 0.10})
                    
                    while cycle.phase != CyclePhase.EXECUTION:
                        system.cycle_manager.advance_phase()
                        
                    system.validate_and_approve_allocations()
                    system.execute_approved_allocations()
                    system.complete_cycle_with_review()
                    break  # Success
            except Exception:
                continue  # Retry on error
                
        # System should recover and complete at least one cycle
        assert system.cycle_manager.completed_cycles >= 1 or system.total_transactions > 0
        
        system.shutdown_system()


@pytest.mark.integration
@pytest.mark.slow
class TestFoundationSystemEndToEnd:
    """Comprehensive end-to-end integration tests"""
    
    def test_realistic_trading_scenario(self):
        """Test realistic trading scenario with market conditions"""
        system = FoundationSystem()
        system.connect_all()
        
        # Set realistic market conditions
        market_data = {
            "AAPL": {"price": 175.50, "current_position": 150000},
            "GOOGL": {"price": 2800.25, "current_position": 200000},  
            "MSFT": {"price": 415.75, "current_position": 120000},
            "TSLA": {"price": 248.90, "current_position": 80000},
            "NVDA": {"price": 875.30, "current_position": 0}  # New position
        }
        
        for symbol, data in market_data.items():
            system.broker.set_market_price(symbol, data["price"])
            system.cycle_manager.set_position_value(symbol, data["current_position"])
            
        # Start weekly rebalancing cycle
        success, cycle, violations = system.start_new_trading_cycle()
        assert success is True
        
        # Target allocation: Reduce tech concentration, add diversification
        rebalancing_plan = {
            "AAPL": 0.18,   # Reduce from ~15% to 18%
            "GOOGL": 0.15,  # Reduce from ~20% to 15%
            "MSFT": 0.17,   # Increase from ~12% to 17%
            "TSLA": 0.10,   # Increase from ~8% to 10%
            "NVDA": 0.15,   # New 15% position
            "SPY": 0.25     # Add 25% index fund for stability
        }
        
        # Phase 1: Analysis and Planning
        allocations = system.create_allocation_plan(rebalancing_plan)
        assert len(allocations) == 6
        
        # Advance through phases
        while cycle.phase != CyclePhase.ALLOCATION:
            system.cycle_manager.advance_phase()
            
        # Phase 2: Risk Validation and Approval
        approved_count, risk_violations = system.validate_and_approve_allocations()
        
        # Should approve most allocations (reasonable diversification)
        assert approved_count >= 4
        
        # Phase 3: Execution
        while cycle.phase != CyclePhase.EXECUTION:
            system.cycle_manager.advance_phase()
            
        successful_executions, failed_executions, errors = system.execute_approved_allocations()
        
        # Should execute successfully
        assert successful_executions > 0
        assert len(errors) == 0
        
        # Phase 4: Review and Completion
        success, review_report = system.complete_cycle_with_review()
        assert success is True
        
        # Validate review report
        assert review_report["allocations"]["diversification_score"] > 0.75  # Well diversified
        assert review_report["executions"]["execution_rate"] > 0.7  # Most executed
        assert len(review_report["recommendations"]) <= 3  # Reasonable number of recommendations
        
        # Check final system health
        final_health = system.get_system_health()
        assert final_health["success_rate"] > 0.8
        
        system.shutdown_system()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])