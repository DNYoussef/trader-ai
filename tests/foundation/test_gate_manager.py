"""
Comprehensive tests for gate manager in Foundation phase.
Tests gate validation, progression tracking, violation management, and error handling.
"""
import pytest
import time
import threading
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable

# Import mock objects
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mocks.mock_gate_manager import (
    MockGateManager, Gate, GateResult, GateViolation,
    GateStatus, GateType, ViolationSeverity, MockGateManagerError,
    create_mock_gate_manager, create_test_gate, create_test_violation
)


class TestGateRegistration:
    """Test gate registration and management"""
    
    def test_register_basic_gate(self):
        """Test registering a basic gate"""
        manager = create_mock_gate_manager()
        gate = create_test_gate("Test Gate", GateType.RISK_CHECK)
        
        result = manager.register_gate(gate)
        assert result is True
        assert gate.id in manager.gates
        assert gate.id in manager.gate_execution_order

    def test_register_gate_without_name(self):
        """Test registering gate without name fails"""
        manager = create_mock_gate_manager()
        gate = Gate(name="")
        
        with pytest.raises(MockGateManagerError) as exc_info:
            manager.register_gate(gate)
            
        assert exc_info.value.error_code == 3001

    def test_register_gates_with_priority_ordering(self):
        """Test gates are ordered by priority"""
        manager = create_mock_gate_manager()
        
        # Clear default gates for clean test
        manager.reset_for_testing()
        
        # Register gates with different priorities
        gate1 = create_test_gate("Low Priority", priority=50)
        gate2 = create_test_gate("High Priority", priority=100)
        gate3 = create_test_gate("Medium Priority", priority=75)
        
        manager.register_gate(gate1)
        manager.register_gate(gate2)
        manager.register_gate(gate3)
        
        # Should be ordered by priority (highest first)
        expected_order = [gate2.id, gate3.id, gate1.id]
        assert manager.gate_execution_order == expected_order

    def test_unregister_gate(self):
        """Test unregistering a gate"""
        manager = create_mock_gate_manager()
        gate = create_test_gate()
        
        manager.register_gate(gate)
        assert gate.id in manager.gates
        
        result = manager.unregister_gate(gate.id)
        assert result is True
        assert gate.id not in manager.gates
        assert gate.id not in manager.gate_execution_order

    def test_unregister_nonexistent_gate(self):
        """Test unregistering non-existent gate"""
        manager = create_mock_gate_manager()
        
        result = manager.unregister_gate("fake-gate-id")
        assert result is False

    def test_get_gate_by_id(self):
        """Test retrieving gate by ID"""
        manager = create_mock_gate_manager()
        gate = create_test_gate("Findable Gate")
        
        manager.register_gate(gate)
        retrieved_gate = manager.get_gate(gate.id)
        
        assert retrieved_gate is not None
        assert retrieved_gate.name == "Findable Gate"

    def test_get_gates_filtered_by_type(self):
        """Test filtering gates by type"""
        manager = create_mock_gate_manager()
        manager.reset_for_testing()  # Clear defaults
        
        risk_gate = create_test_gate("Risk Gate", GateType.RISK_CHECK)
        position_gate = create_test_gate("Position Gate", GateType.POSITION_LIMIT)
        loss_gate = create_test_gate("Loss Gate", GateType.LOSS_LIMIT)
        
        manager.register_gate(risk_gate)
        manager.register_gate(position_gate)
        manager.register_gate(loss_gate)
        
        risk_gates = manager.get_gates(gate_type=GateType.RISK_CHECK)
        assert len(risk_gates) == 1
        assert risk_gates[0].name == "Risk Gate"

    def test_get_enabled_gates_only(self):
        """Test filtering for enabled gates only"""
        manager = create_mock_gate_manager()
        manager.reset_for_testing()
        
        enabled_gate = create_test_gate("Enabled Gate", enabled=True)
        disabled_gate = create_test_gate("Disabled Gate", enabled=False)
        
        manager.register_gate(enabled_gate)
        manager.register_gate(disabled_gate)
        
        enabled_gates = manager.get_gates(enabled_only=True)
        assert len(enabled_gates) == 1
        assert enabled_gates[0].name == "Enabled Gate"


class TestGateEnableDisable:
    """Test gate enable/disable functionality"""
    
    def test_enable_gate(self):
        """Test enabling a gate"""
        manager = create_mock_gate_manager()
        gate = create_test_gate(enabled=False)
        
        manager.register_gate(gate)
        assert not gate.enabled
        
        result = manager.enable_gate(gate.id)
        assert result is True
        assert gate.enabled

    def test_disable_gate(self):
        """Test disabling a gate"""
        manager = create_mock_gate_manager()
        gate = create_test_gate(enabled=True)
        
        manager.register_gate(gate)
        assert gate.enabled
        
        result = manager.disable_gate(gate.id)
        assert result is True
        assert not gate.enabled

    def test_enable_nonexistent_gate(self):
        """Test enabling non-existent gate"""
        manager = create_mock_gate_manager()
        
        result = manager.enable_gate("fake-gate-id")
        assert result is False

    def test_disable_nonexistent_gate(self):
        """Test disabling non-existent gate"""
        manager = create_mock_gate_manager()
        
        result = manager.disable_gate("fake-gate-id")
        assert result is False


class TestGateValidation:
    """Test individual gate validation"""
    
    @pytest.fixture
    def manager_with_gates(self):
        """Fixture providing manager with test gates"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        # Add custom validation gate
        def custom_validator(context, threshold_config):
            risk_level = context.get("risk_level", 0.0)
            max_risk = threshold_config.get("max_risk", 1.0)
            return risk_level <= max_risk
            
        gate = Gate(
            name="Custom Risk Gate",
            gate_type=GateType.RISK_CHECK,
            validator_func=custom_validator,
            threshold_config={"max_risk": 0.8}
        )
        
        manager.register_gate(gate)
        return manager, gate

    def test_successful_gate_validation(self, manager_with_gates):
        """Test successful gate validation"""
        manager, gate = manager_with_gates
        
        context = {"risk_level": 0.5}  # Below threshold
        result = manager.validate_gate(gate.id, context)
        
        assert result.status == GateStatus.PASSED
        assert result.gate_id == gate.id
        assert len(result.violations) == 0

    def test_failed_gate_validation(self, manager_with_gates):
        """Test failed gate validation"""
        manager, gate = manager_with_gates
        
        context = {"risk_level": 0.9}  # Above threshold
        result = manager.validate_gate(gate.id, context)
        
        assert result.status == GateStatus.FAILED
        assert result.gate_id == gate.id

    def test_validate_disabled_gate(self):
        """Test validating disabled gate returns skipped"""
        manager = create_mock_gate_manager()
        gate = create_test_gate(enabled=False)
        
        manager.register_gate(gate)
        result = manager.validate_gate(gate.id)
        
        assert result.status == GateStatus.SKIPPED
        assert "disabled" in result.message.lower()

    def test_validate_with_manager_disabled(self):
        """Test validating when manager is disabled"""
        manager = create_mock_gate_manager()
        manager.enabled = False
        
        gate = create_test_gate()
        manager.register_gate(gate)
        
        result = manager.validate_gate(gate.id)
        
        assert result.status == GateStatus.SKIPPED
        assert "manager disabled" in result.message.lower()

    def test_validate_nonexistent_gate(self):
        """Test validating non-existent gate"""
        manager = create_mock_gate_manager()
        
        with pytest.raises(MockGateManagerError) as exc_info:
            manager.validate_gate("fake-gate-id")
            
        assert exc_info.value.error_code == 3002

    def test_validation_with_delay(self):
        """Test validation with realistic delay"""
        manager = create_mock_gate_manager(validation_delay=0.1)
        gate = create_test_gate()
        
        manager.register_gate(gate)
        
        start_time = time.time()
        result = manager.validate_gate(gate.id)
        end_time = time.time()
        
        assert (end_time - start_time) >= 0.1
        assert result.execution_time >= 0.1

    def test_validation_error_handling(self):
        """Test validation with simulated errors"""
        manager = create_mock_gate_manager(error_rate=1.0)  # Force errors
        gate = create_test_gate()
        
        manager.register_gate(gate)
        result = manager.validate_gate(gate.id)
        
        assert result.status == GateStatus.FAILED
        assert "error" in result.message.lower()

    def test_custom_validator_exception(self):
        """Test handling exceptions in custom validators"""
        manager = create_mock_gate_manager()
        
        def failing_validator(context, threshold_config):
            raise Exception("Validator failed")
            
        gate = Gate(
            name="Failing Gate",
            validator_func=failing_validator
        )
        
        manager.register_gate(gate)
        result = manager.validate_gate(gate.id)
        
        assert result.status == GateStatus.FAILED
        assert "validator error" in result.message.lower()

    def test_validation_result_storage(self):
        """Test that validation results are stored"""
        manager = create_mock_gate_manager()
        gate = create_test_gate()
        
        manager.register_gate(gate)
        result = manager.validate_gate(gate.id)
        
        # Check result stored in gate
        assert gate.last_result is not None
        assert gate.last_result.status == result.status
        
        # Check result stored in manager
        stored_result = manager.validation_results.get(gate.id)
        assert stored_result is not None
        assert stored_result.status == result.status

    def test_validation_callbacks(self):
        """Test validation result callbacks"""
        manager = create_mock_gate_manager()
        gate = create_test_gate()
        callback_results = []
        
        def result_callback(result):
            callback_results.append(result.gate_id)
            
        manager.on_gate_result = result_callback
        manager.register_gate(gate)
        
        manager.validate_gate(gate.id)
        
        assert gate.id in callback_results


class TestDefaultGateValidation:
    """Test default gate validation logic"""
    
    def test_risk_check_gate_pass(self):
        """Test risk check gate passing validation"""
        manager = create_mock_gate_manager()
        
        # Get the default risk gate
        risk_gates = manager.get_gates(gate_type=GateType.RISK_CHECK)
        assert len(risk_gates) > 0
        
        risk_gate = risk_gates[0]
        context = {"portfolio_risk": 0.5}  # Below default threshold of 0.8
        
        result = manager.validate_gate(risk_gate.id, context)
        assert result.status == GateStatus.PASSED

    def test_risk_check_gate_fail(self):
        """Test risk check gate failing validation"""
        manager = create_mock_gate_manager()
        
        risk_gates = manager.get_gates(gate_type=GateType.RISK_CHECK)
        risk_gate = risk_gates[0]
        
        context = {"portfolio_risk": 0.9}  # Above default threshold
        result = manager.validate_gate(risk_gate.id, context)
        
        assert result.status == GateStatus.FAILED
        assert len(result.violations) > 0
        assert result.violations[0].severity == ViolationSeverity.HIGH

    def test_position_limit_gate(self):
        """Test position limit gate validation"""
        manager = create_mock_gate_manager()
        
        position_gates = manager.get_gates(gate_type=GateType.POSITION_LIMIT)
        position_gate = position_gates[0]
        
        # Test pass
        context = {"position_count": 5}  # Below default limit of 20
        result = manager.validate_gate(position_gate.id, context)
        assert result.status == GateStatus.PASSED
        
        # Test fail
        context = {"position_count": 25}  # Above limit
        result = manager.validate_gate(position_gate.id, context)
        assert result.status == GateStatus.FAILED

    def test_loss_limit_gate(self):
        """Test loss limit gate validation"""
        manager = create_mock_gate_manager()
        
        loss_gates = manager.get_gates(gate_type=GateType.LOSS_LIMIT)
        loss_gate = loss_gates[0]
        
        # Test pass (loss within limit)
        context = {"current_loss": -1000.0}  # Above default limit of -5000
        result = manager.validate_gate(loss_gate.id, context)
        assert result.status == GateStatus.PASSED
        
        # Test fail (loss exceeds limit)
        context = {"current_loss": -8000.0}  # Below limit
        result = manager.validate_gate(loss_gate.id, context)
        assert result.status == GateStatus.FAILED
        assert result.violations[0].severity == ViolationSeverity.CRITICAL


class TestBulkValidation:
    """Test validating multiple gates"""
    
    def test_validate_all_enabled_gates(self):
        """Test validating all enabled gates"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        context = {
            "portfolio_risk": 0.5,
            "position_count": 10,
            "current_loss": -1000.0
        }
        
        results = manager.validate_all_gates(context)
        
        # Should have results for all default enabled gates
        enabled_gates = manager.get_gates(enabled_only=True)
        assert len(results) == len(enabled_gates)
        
        # All should pass with good context
        passed_results = [r for r in results if r.status == GateStatus.PASSED]
        assert len(passed_results) == len(results)

    def test_validate_all_with_failures(self):
        """Test validation with some failures"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        context = {
            "portfolio_risk": 0.9,  # Will fail risk check
            "position_count": 5,    # Will pass position check
            "current_loss": -1000.0 # Will pass loss check
        }
        
        results = manager.validate_all_gates(context)
        
        # Should have mixed results
        passed_results = [r for r in results if r.status == GateStatus.PASSED]
        failed_results = [r for r in results if r.status == GateStatus.FAILED]
        
        assert len(failed_results) > 0
        assert len(passed_results) > 0

    def test_fail_fast_behavior(self):
        """Test fail-fast behavior in bulk validation"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        manager.fail_fast = True
        
        # Reset and add gates with known order
        manager.reset_for_testing()
        
        gate1 = Gate(name="Gate 1", priority=100)  # Will be first
        gate2 = Gate(name="Gate 2", priority=90)   # Will be second
        gate3 = Gate(name="Gate 3", priority=80)   # Will be third
        
        # Make first gate fail
        def failing_validator(context, threshold_config):
            return False
            
        gate1.validator_func = failing_validator
        
        manager.register_gate(gate1)
        manager.register_gate(gate2)
        manager.register_gate(gate3)
        
        results = manager.validate_all_gates()
        
        # Should stop after first failure
        assert len(results) == 1
        assert results[0].status == GateStatus.FAILED

    def test_validation_session_tracking(self):
        """Test validation session tracking"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        session_id = "test-session"
        results = manager.validate_all_gates(session_id=session_id)
        
        # Session should be stored
        stored_results = manager.validation_sessions.get(session_id)
        assert stored_results is not None
        assert len(stored_results) == len(results)

    def test_validation_completion_callback(self):
        """Test bulk validation completion callback"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        callback_data = []
        
        def completion_callback(session_id, results):
            callback_data.append((session_id, len(results)))
            
        manager.on_validation_complete = completion_callback
        
        session_id = "test-callback-session"
        results = manager.validate_all_gates(session_id=session_id)
        
        assert len(callback_data) == 1
        assert callback_data[0][0] == session_id
        assert callback_data[0][1] == len(results)


class TestViolationManagement:
    """Test violation tracking and management"""
    
    def test_violation_creation_and_storage(self):
        """Test that violations are created and stored"""
        manager = create_mock_gate_manager()
        
        # Trigger a violation
        risk_gates = manager.get_gates(gate_type=GateType.RISK_CHECK)
        risk_gate = risk_gates[0]
        
        context = {"portfolio_risk": 0.95}  # Above threshold
        result = manager.validate_gate(risk_gate.id, context)
        
        assert len(result.violations) > 0
        assert len(manager.violations) > 0
        assert len(manager.violation_history) > 0
        
        violation = result.violations[0]
        assert violation.gate_id == risk_gate.id
        assert violation.severity == ViolationSeverity.HIGH

    def test_get_violations_by_severity(self):
        """Test filtering violations by severity"""
        manager = create_mock_gate_manager()
        
        # Create violations of different severities
        high_violation = create_test_violation("gate1", ViolationSeverity.HIGH)
        medium_violation = create_test_violation("gate2", ViolationSeverity.MEDIUM)
        
        manager.violations.extend([high_violation, medium_violation])
        
        high_violations = manager.get_violations(severity=ViolationSeverity.HIGH)
        assert len(high_violations) == 1
        assert high_violations[0].severity == ViolationSeverity.HIGH

    def test_get_unresolved_violations_only(self):
        """Test filtering for unresolved violations"""
        manager = create_mock_gate_manager()
        
        resolved_violation = create_test_violation("gate1", resolved=True)
        unresolved_violation = create_test_violation("gate2", resolved=False)
        
        manager.violations.extend([resolved_violation, unresolved_violation])
        
        unresolved = manager.get_violations(unresolved_only=True)
        assert len(unresolved) == 1
        assert not unresolved[0].resolved

    def test_get_violations_since_timestamp(self):
        """Test filtering violations by timestamp"""
        manager = create_mock_gate_manager()
        
        # Create violation in the past
        old_violation = create_test_violation()
        old_violation.timestamp = datetime.now() - timedelta(hours=1)
        
        # Create recent violation
        recent_violation = create_test_violation()
        recent_violation.timestamp = datetime.now()
        
        manager.violations.extend([old_violation, recent_violation])
        
        cutoff_time = datetime.now() - timedelta(minutes=30)
        recent_violations = manager.get_violations(since=cutoff_time)
        
        assert len(recent_violations) == 1
        assert recent_violations[0].timestamp >= cutoff_time

    def test_resolve_violation(self):
        """Test resolving violations"""
        manager = create_mock_gate_manager()
        
        violation = create_test_violation(resolved=False)
        manager.violations.append(violation)
        
        result = manager.resolve_violation(violation.id, "Fixed by user")
        
        assert result is True
        assert violation.resolved is True
        assert violation.resolution_time is not None
        assert violation.metadata.get("resolution_note") == "Fixed by user"

    def test_resolve_nonexistent_violation(self):
        """Test resolving non-existent violation"""
        manager = create_mock_gate_manager()
        
        result = manager.resolve_violation("fake-violation-id")
        assert result is False

    def test_clear_resolved_violations(self):
        """Test clearing resolved violations"""
        manager = create_mock_gate_manager()
        
        resolved_violation = create_test_violation(resolved=True)
        unresolved_violation = create_test_violation(resolved=False)
        
        manager.violations.extend([resolved_violation, unresolved_violation])
        
        manager.clear_violations(resolved_only=True)
        
        # Only unresolved should remain
        assert len(manager.violations) == 1
        assert not manager.violations[0].resolved

    def test_clear_violations_by_severity(self):
        """Test clearing violations by severity"""
        manager = create_mock_gate_manager()
        
        high_violation = create_test_violation(severity=ViolationSeverity.HIGH, resolved=True)
        medium_violation = create_test_violation(severity=ViolationSeverity.MEDIUM, resolved=True)
        
        manager.violations.extend([high_violation, medium_violation])
        
        manager.clear_violations(severity=ViolationSeverity.HIGH, resolved_only=True)
        
        # Only medium violation should remain
        assert len(manager.violations) == 1
        assert manager.violations[0].severity == ViolationSeverity.MEDIUM

    def test_violation_callbacks(self):
        """Test violation callbacks"""
        manager = create_mock_gate_manager()
        violation_callbacks = []
        
        def violation_callback(violation):
            violation_callbacks.append(violation.gate_id)
            
        manager.on_violation = violation_callback
        
        # Trigger violation
        risk_gates = manager.get_gates(gate_type=GateType.RISK_CHECK)
        risk_gate = risk_gates[0]
        
        context = {"portfolio_risk": 0.95}
        manager.validate_gate(risk_gate.id, context)
        
        assert risk_gate.id in violation_callbacks


class TestPerformanceMetrics:
    """Test performance tracking and metrics"""
    
    def test_validation_counting(self):
        """Test validation performance counting"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        gate = create_test_gate()
        
        manager.register_gate(gate)
        
        initial_total = manager.total_validations
        initial_success = manager.successful_validations
        
        # Successful validation
        manager.validate_gate(gate.id)
        
        assert manager.total_validations == initial_total + 1
        assert manager.successful_validations == initial_success + 1

    def test_failed_validation_counting(self):
        """Test failed validation counting"""
        manager = create_mock_gate_manager(error_rate=1.0)  # Force failures
        gate = create_test_gate()
        
        manager.register_gate(gate)
        
        initial_failed = manager.failed_validations
        
        result = manager.validate_gate(gate.id)
        
        assert result.status == GateStatus.FAILED
        assert manager.failed_validations == initial_failed + 1

    def test_average_validation_time_calculation(self):
        """Test average validation time calculation"""
        manager = create_mock_gate_manager(validation_delay=0.05)
        gate = create_test_gate()
        
        manager.register_gate(gate)
        
        # Perform multiple validations
        for _ in range(3):
            manager.validate_gate(gate.id)
            
        assert manager.avg_validation_time > 0.04  # Should be around 0.05

    def test_validation_summary(self):
        """Test getting validation summary"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        # Perform some validations
        enabled_gates = manager.get_gates(enabled_only=True)
        for gate in enabled_gates[:2]:
            manager.validate_gate(gate.id)
            
        summary = manager.get_validation_summary()
        
        assert "total_gates" in summary
        assert "enabled_gates" in summary
        assert "total_validations" in summary
        assert "success_rate" in summary
        assert summary["total_validations"] >= 2


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent operations"""
    
    def test_concurrent_validation(self):
        """Test concurrent gate validation"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        gate = create_test_gate()
        manager.register_gate(gate)
        
        results = []
        
        def validate_gate():
            result = manager.validate_gate(gate.id)
            results.append(result)
            
        # Start multiple threads
        threads = [threading.Thread(target=validate_gate) for _ in range(5)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All validations should complete
        assert len(results) == 5
        
        # Manager should be in consistent state
        assert manager.total_validations >= 5

    def test_concurrent_gate_registration(self):
        """Test concurrent gate registration"""
        manager = create_mock_gate_manager()
        manager.reset_for_testing()  # Clear defaults
        
        def register_gates():
            for i in range(5):
                gate = create_test_gate(f"Gate {threading.current_thread().ident}_{i}")
                manager.register_gate(gate)
                
        # Start multiple threads
        threads = [threading.Thread(target=register_gates) for _ in range(3)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All gates should be registered
        assert len(manager.gates) == 15  # 3 threads * 5 gates each

    def test_concurrent_violation_management(self):
        """Test concurrent violation operations"""
        manager = create_mock_gate_manager()
        
        # Add initial violations
        for i in range(10):
            violation = create_test_violation(f"gate_{i}")
            manager.violations.append(violation)
            
        def resolve_violations():
            for violation in manager.violations[:5]:
                manager.resolve_violation(violation.id)
                
        def clear_violations():
            manager.clear_violations(resolved_only=True)
            
        # Start concurrent operations
        thread1 = threading.Thread(target=resolve_violations)
        thread2 = threading.Thread(target=clear_violations)
        
        thread1.start()
        time.sleep(0.01)  # Small delay to let resolutions start
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Manager should be in consistent state
        assert len(manager.violations) <= 10


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_validation_with_error_rate(self):
        """Test validation with configured error rate"""
        manager = create_mock_gate_manager(error_rate=0.5, validation_delay=0.01)
        gate = create_test_gate()
        
        manager.register_gate(gate)
        
        # Run multiple validations
        failed_count = 0
        total_validations = 20
        
        for _ in range(total_validations):
            result = manager.validate_gate(gate.id)
            if result.status == GateStatus.FAILED:
                failed_count += 1
                
        # Some should fail due to error rate
        assert failed_count > 0
        assert failed_count < total_validations  # Not all should fail

    def test_reset_functionality(self):
        """Test manager reset for testing"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        # Create some state
        gate = create_test_gate()
        manager.register_gate(gate)
        manager.validate_gate(gate.id)
        
        # Add violation
        violation = create_test_violation()
        manager.violations.append(violation)
        
        # Verify state exists
        assert len(manager.gates) > 4  # Default gates + test gate
        assert manager.total_validations > 0
        assert len(manager.violations) > 0
        
        # Reset
        manager.reset_for_testing()
        
        # Verify clean state (but with default gates restored)
        default_gate_count = 4  # Default gates should be restored
        assert len(manager.gates) == default_gate_count
        assert manager.total_validations == 0
        assert len(manager.violations) == 0

    def test_manager_disabled_state(self):
        """Test manager behavior when disabled"""
        manager = create_mock_gate_manager()
        manager.enabled = False
        
        gate = create_test_gate()
        manager.register_gate(gate)
        
        # Single validation should be skipped
        result = manager.validate_gate(gate.id)
        assert result.status == GateStatus.SKIPPED
        
        # Bulk validation should be skipped
        results = manager.validate_all_gates()
        assert all(r.status == GateStatus.SKIPPED for r in results)


@pytest.mark.integration
class TestGateManagerIntegrationScenarios:
    """Integration test scenarios combining multiple gate manager features"""
    
    def test_complete_risk_management_workflow(self):
        """Test complete risk management workflow"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        
        # 1. Setup custom gates for comprehensive risk management
        def leverage_validator(context, config):
            leverage = context.get("leverage", 1.0)
            max_leverage = config.get("max_leverage", 3.0)
            return leverage <= max_leverage
            
        def correlation_validator(context, config):
            correlation = context.get("max_correlation", 0.0)
            max_correlation = config.get("max_correlation", 0.7)
            return correlation <= max_correlation
            
        leverage_gate = Gate(
            name="Leverage Check",
            gate_type=GateType.CUSTOM,
            priority=110,
            validator_func=leverage_validator,
            threshold_config={"max_leverage": 2.5}
        )
        
        correlation_gate = Gate(
            name="Correlation Check", 
            gate_type=GateType.CORRELATION,
            priority=85,
            validator_func=correlation_validator,
            threshold_config={"max_correlation": 0.6}
        )
        
        manager.register_gate(leverage_gate)
        manager.register_gate(correlation_gate)
        
        # 2. Test with good risk parameters
        good_context = {
            "portfolio_risk": 0.4,
            "position_count": 8,
            "current_loss": -500.0,
            "leverage": 2.0,
            "max_correlation": 0.5
        }
        
        results = manager.validate_all_gates(good_context, session_id="good_scenario")
        
        # All should pass
        passed_results = [r for r in results if r.status == GateStatus.PASSED]
        assert len(passed_results) == len(results)
        
        # 3. Test with risky parameters
        risky_context = {
            "portfolio_risk": 0.85,  # High risk
            "position_count": 25,    # Too many positions
            "current_loss": -6000.0, # Excessive loss
            "leverage": 3.0,         # High leverage
            "max_correlation": 0.8   # High correlation
        }
        
        results = manager.validate_all_gates(risky_context, session_id="risky_scenario")
        
        # Multiple gates should fail
        failed_results = [r for r in results if r.status == GateStatus.FAILED]
        assert len(failed_results) >= 3
        
        # 4. Verify violations were recorded
        violations = manager.get_violations(unresolved_only=True)
        assert len(violations) >= 3
        
        # 5. Resolve some violations
        critical_violations = manager.get_violations(severity=ViolationSeverity.CRITICAL)
        for violation in critical_violations:
            manager.resolve_violation(violation.id, "Risk reduced")
            
        # 6. Check summary
        summary = manager.get_validation_summary()
        assert summary["total_validations"] >= 12  # 6+ gates * 2 scenarios
        assert summary["active_violations"] < len(violations)  # Some resolved

    def test_gate_progression_scenario(self):
        """Test progressive gate validation scenario"""
        manager = create_mock_gate_manager(validation_delay=0.01)
        manager.reset_for_testing()
        
        # Setup progressive gates (each depends on previous)
        def progressive_validator_1(context, config):
            return context.get("stage_1_complete", False)
            
        def progressive_validator_2(context, config):
            return context.get("stage_2_complete", False)
            
        def progressive_validator_3(context, config):
            return context.get("stage_3_complete", False)
            
        stage1_gate = Gate(
            name="Stage 1",
            priority=100,
            validator_func=progressive_validator_1
        )
        
        stage2_gate = Gate(
            name="Stage 2", 
            priority=90,
            validator_func=progressive_validator_2
        )
        
        stage3_gate = Gate(
            name="Stage 3",
            priority=80,
            validator_func=progressive_validator_3
        )
        
        manager.register_gate(stage1_gate)
        manager.register_gate(stage2_gate)
        manager.register_gate(stage3_gate)
        
        # Test progressive completion
        contexts = [
            {},  # Nothing complete
            {"stage_1_complete": True},  # Stage 1 done
            {"stage_1_complete": True, "stage_2_complete": True},  # Stages 1-2 done
            {"stage_1_complete": True, "stage_2_complete": True, "stage_3_complete": True}  # All done
        ]
        
        for i, context in enumerate(contexts):
            session_id = f"progression_step_{i}"
            results = manager.validate_all_gates(context, session_id=session_id)
            
            passed_count = len([r for r in results if r.status == GateStatus.PASSED])
            expected_passed = i  # Should match the progression step
            
            assert passed_count == expected_passed

    def test_high_throughput_scenario(self):
        """Test high-throughput validation scenario"""
        manager = create_mock_gate_manager(validation_delay=0.001)  # Fast validation
        
        # Setup multiple validation sessions concurrently
        def validation_worker(worker_id):
            for i in range(10):
                context = {
                    "portfolio_risk": 0.3 + (worker_id * 0.1),
                    "position_count": 5 + worker_id,
                    "current_loss": -100 * worker_id
                }
                session_id = f"worker_{worker_id}_session_{i}"
                manager.validate_all_gates(context, session_id=session_id)
                
        # Start multiple workers
        threads = [threading.Thread(target=validation_worker, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify high throughput was handled
        summary = manager.get_validation_summary()
        assert summary["total_validations"] >= 200  # 5 workers * 10 sessions * 4+ gates
        assert len(manager.validation_sessions) == 50  # 5 workers * 10 sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])