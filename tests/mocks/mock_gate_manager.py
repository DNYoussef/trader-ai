"""
Mock gate manager implementation for testing Foundation phase components.
Simulates gate validation, progression, and violation tracking.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading


class GateStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GateType(Enum):
    RISK_CHECK = "risk_check"
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    DRAWDOWN = "drawdown"
    CUSTOM = "custom"


@dataclass
class GateViolation:
    """Represents a gate violation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gate_id: str = ""
    violation_type: str = ""
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    message: str = ""
    actual_value: Any = None
    threshold_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class GateResult:
    """Result of gate validation"""
    gate_id: str
    status: GateStatus
    message: str = ""
    violations: List[GateViolation] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Gate:
    """Represents a validation gate"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    gate_type: GateType = GateType.CUSTOM
    enabled: bool = True
    priority: int = 100
    timeout_seconds: int = 30
    validator_func: Optional[Callable] = None
    threshold_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_result: Optional[GateResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class MockGateManagerError(Exception):
    """Mock gate manager specific errors"""
    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.error_code = error_code


class MockGateManager:
    """
    Mock gate manager for testing Foundation phase gate validation system.
    Provides comprehensive gate management, validation, and violation tracking.
    """
    
    def __init__(self,
                 validation_delay: float = 0.1,
                 error_rate: float = 0.0,
                 enable_async_validation: bool = True):
        """
        Initialize mock gate manager
        
        Args:
            validation_delay: Simulated validation delay in seconds
            error_rate: Probability of random validation errors (0.0-1.0)
            enable_async_validation: Whether to support async validation
        """
        self.validation_delay = validation_delay
        self.error_rate = error_rate
        self.enable_async_validation = enable_async_validation
        
        # Gate registry
        self.gates: Dict[str, Gate] = {}
        self.gate_execution_order: List[str] = []
        
        # Violation tracking
        self.violations: List[GateViolation] = []
        self.violation_history: List[GateViolation] = []
        
        # Validation results
        self.validation_results: Dict[str, GateResult] = {}
        self.validation_sessions: Dict[str, List[GateResult]] = {}
        
        # State management
        self.enabled = True
        self.global_timeout = 300  # 5 minutes default
        self.fail_fast = False
        
        # Performance tracking
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.avg_validation_time = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Callbacks for testing
        self.on_gate_result = None
        self.on_violation = None
        self.on_validation_complete = None
        
        # Default gate configurations
        self._setup_default_gates()

    def register_gate(self, gate: Gate) -> bool:
        """
        Register a new validation gate
        
        Args:
            gate: Gate object to register
            
        Returns:
            bool: Success status
        """
        if not gate.id:
            gate.id = str(uuid.uuid4())
            
        if not gate.name:
            raise MockGateManagerError("Gate name is required", 3001)
            
        with self._lock:
            self.gates[gate.id] = gate
            if gate.id not in self.gate_execution_order:
                # Insert based on priority (higher priority first)
                inserted = False
                for i, existing_id in enumerate(self.gate_execution_order):
                    existing_gate = self.gates[existing_id]
                    if gate.priority > existing_gate.priority:
                        self.gate_execution_order.insert(i, gate.id)
                        inserted = True
                        break
                        
                if not inserted:
                    self.gate_execution_order.append(gate.id)
                    
        return True

    def unregister_gate(self, gate_id: str) -> bool:
        """Unregister a gate"""
        with self._lock:
            if gate_id in self.gates:
                del self.gates[gate_id]
                if gate_id in self.gate_execution_order:
                    self.gate_execution_order.remove(gate_id)
                return True
        return False

    def get_gate(self, gate_id: str) -> Optional[Gate]:
        """Get gate by ID"""
        return self.gates.get(gate_id)

    def get_gates(self, gate_type: Optional[GateType] = None, 
                  enabled_only: bool = False) -> List[Gate]:
        """
        Get all gates, optionally filtered
        
        Args:
            gate_type: Filter by gate type
            enabled_only: Only return enabled gates
            
        Returns:
            List of gates
        """
        gates = list(self.gates.values())
        
        if gate_type:
            gates = [g for g in gates if g.gate_type == gate_type]
            
        if enabled_only:
            gates = [g for g in gates if g.enabled]
            
        return gates

    def enable_gate(self, gate_id: str) -> bool:
        """Enable a gate"""
        gate = self.gates.get(gate_id)
        if gate:
            gate.enabled = True
            gate.updated_at = datetime.now()
            return True
        return False

    def disable_gate(self, gate_id: str) -> bool:
        """Disable a gate"""
        gate = self.gates.get(gate_id)
        if gate:
            gate.enabled = False
            gate.updated_at = datetime.now()
            return True
        return False

    def validate_gate(self, gate_id: str, context: Dict[str, Any] = None) -> GateResult:
        """
        Validate a specific gate
        
        Args:
            gate_id: ID of gate to validate
            context: Validation context data
            
        Returns:
            GateResult object
        """
        if not self.enabled:
            return GateResult(
                gate_id=gate_id,
                status=GateStatus.SKIPPED,
                message="Gate manager disabled"
            )
            
        gate = self.gates.get(gate_id)
        if not gate:
            raise MockGateManagerError(f"Gate not found: {gate_id}", 3002)
            
        if not gate.enabled:
            return GateResult(
                gate_id=gate_id,
                status=GateStatus.SKIPPED,
                message="Gate disabled"
            )
            
        start_time = datetime.now()
        
        try:
            # Simulate validation delay
            if self.validation_delay > 0:
                import time
                time.sleep(self.validation_delay)
                
            # Perform validation
            result = self._execute_gate_validation(gate, context or {})
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update gate result
            gate.last_result = result
            gate.updated_at = datetime.now()
            
            # Store result
            with self._lock:
                self.validation_results[gate_id] = result
                self.total_validations += 1
                
                if result.status == GateStatus.PASSED:
                    self.successful_validations += 1
                else:
                    self.failed_validations += 1
                    
                # Update average validation time
                self.avg_validation_time = (
                    (self.avg_validation_time * (self.total_validations - 1) + execution_time) 
                    / self.total_validations
                )
                
            # Handle violations
            if result.violations:
                with self._lock:
                    self.violations.extend(result.violations)
                    self.violation_history.extend(result.violations)
                    
                for violation in result.violations:
                    if self.on_violation:
                        self.on_violation(violation)
                        
            # Trigger callback
            if self.on_gate_result:
                self.on_gate_result(result)
                
            return result
            
        except Exception as e:
            result = GateResult(
                gate_id=gate_id,
                status=GateStatus.FAILED,
                message=f"Validation error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            with self._lock:
                self.failed_validations += 1
                self.total_validations += 1
                
            return result

    def validate_all_gates(self, context: Dict[str, Any] = None, 
                          session_id: str = None) -> List[GateResult]:
        """
        Validate all enabled gates in priority order
        
        Args:
            context: Validation context data
            session_id: Optional session ID for tracking
            
        Returns:
            List of GateResult objects
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        results = []
        context = context or {}
        
        # Get enabled gates in execution order
        enabled_gate_ids = [
            gate_id for gate_id in self.gate_execution_order
            if self.gates[gate_id].enabled
        ]
        
        for gate_id in enabled_gate_ids:
            try:
                result = self.validate_gate(gate_id, context)
                results.append(result)
                
                # Check for fail-fast behavior
                if self.fail_fast and result.status == GateStatus.FAILED:
                    break
                    
            except Exception as e:
                result = GateResult(
                    gate_id=gate_id,
                    status=GateStatus.FAILED,
                    message=f"Unexpected error: {str(e)}"
                )
                results.append(result)
                
                if self.fail_fast:
                    break
                    
        # Store session results
        with self._lock:
            self.validation_sessions[session_id] = results
            
        # Trigger completion callback
        if self.on_validation_complete:
            self.on_validation_complete(session_id, results)
            
        return results

    def get_violations(self, 
                      severity: Optional[ViolationSeverity] = None,
                      unresolved_only: bool = False,
                      since: Optional[datetime] = None) -> List[GateViolation]:
        """
        Get violations with optional filtering
        
        Args:
            severity: Filter by severity
            unresolved_only: Only unresolved violations
            since: Only violations after this time
            
        Returns:
            List of violations
        """
        violations = self.violations[:]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
            
        if unresolved_only:
            violations = [v for v in violations if not v.resolved]
            
        if since:
            violations = [v for v in violations if v.timestamp >= since]
            
        return violations

    def resolve_violation(self, violation_id: str, 
                         resolution_note: str = "") -> bool:
        """
        Mark a violation as resolved
        
        Args:
            violation_id: ID of violation to resolve
            resolution_note: Optional resolution note
            
        Returns:
            bool: Success status
        """
        with self._lock:
            for violation in self.violations:
                if violation.id == violation_id:
                    violation.resolved = True
                    violation.resolution_time = datetime.now()
                    if resolution_note:
                        violation.metadata["resolution_note"] = resolution_note
                    return True
        return False

    def clear_violations(self, 
                        severity: Optional[ViolationSeverity] = None,
                        resolved_only: bool = True):
        """
        Clear violations from active list
        
        Args:
            severity: Only clear violations of this severity
            resolved_only: Only clear resolved violations
        """
        with self._lock:
            violations_to_remove = []
            
            for violation in self.violations:
                should_remove = True
                
                if severity and violation.severity != severity:
                    should_remove = False
                    
                if resolved_only and not violation.resolved:
                    should_remove = False
                    
                if should_remove:
                    violations_to_remove.append(violation)
                    
            for violation in violations_to_remove:
                self.violations.remove(violation)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation performance summary"""
        active_violations = len([v for v in self.violations if not v.resolved])
        
        return {
            "total_gates": len(self.gates),
            "enabled_gates": len([g for g in self.gates.values() if g.enabled]),
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "success_rate": (
                self.successful_validations / max(self.total_validations, 1)
            ),
            "avg_validation_time": self.avg_validation_time,
            "active_violations": active_violations,
            "total_violations": len(self.violation_history),
            "enabled": self.enabled,
            "fail_fast": self.fail_fast
        }

    def reset_for_testing(self):
        """Reset manager state for testing"""
        with self._lock:
            self.gates.clear()
            self.gate_execution_order.clear()
            self.violations.clear()
            self.violation_history.clear()
            self.validation_results.clear()
            self.validation_sessions.clear()
            self.total_validations = 0
            self.successful_validations = 0
            self.failed_validations = 0
            self.avg_validation_time = 0.0
            self.enabled = True
            self.fail_fast = False
            
        # Re-setup default gates
        self._setup_default_gates()

    def _execute_gate_validation(self, gate: Gate, context: Dict[str, Any]) -> GateResult:
        """Execute validation for a specific gate"""
        # Check for random errors
        if self.error_rate > 0 and self._should_error():
            raise MockGateManagerError("Simulated validation error", 3003)
            
        # Use custom validator if provided
        if gate.validator_func:
            try:
                validation_result = gate.validator_func(context, gate.threshold_config)
                if isinstance(validation_result, bool):
                    status = GateStatus.PASSED if validation_result else GateStatus.FAILED
                    message = f"Gate {'passed' if validation_result else 'failed'}"
                    violations = []
                elif isinstance(validation_result, tuple):
                    status, message, violations = validation_result
                else:
                    status = GateStatus.FAILED
                    message = "Invalid validator result"
                    violations = []
            except Exception as e:
                status = GateStatus.FAILED
                message = f"Validator error: {str(e)}"
                violations = []
        else:
            # Default validation based on gate type
            status, message, violations = self._default_gate_validation(gate, context)
            
        return GateResult(
            gate_id=gate.id,
            status=status,
            message=message,
            violations=violations,
            metadata={"context_keys": list(context.keys())}
        )

    def _default_gate_validation(self, gate: Gate, context: Dict[str, Any]):
        """Default validation logic based on gate type"""
        violations = []
        
        if gate.gate_type == GateType.RISK_CHECK:
            portfolio_risk = context.get("portfolio_risk", 0.5)
            max_risk = gate.threshold_config.get("max_risk", 1.0)
            
            if portfolio_risk > max_risk:
                violation = GateViolation(
                    gate_id=gate.id,
                    violation_type="risk_exceeded",
                    severity=ViolationSeverity.HIGH,
                    message=f"Portfolio risk {portfolio_risk} exceeds limit {max_risk}",
                    actual_value=portfolio_risk,
                    threshold_value=max_risk
                )
                violations.append(violation)
                return GateStatus.FAILED, "Risk limit exceeded", violations
                
        elif gate.gate_type == GateType.POSITION_LIMIT:
            position_count = context.get("position_count", 0)
            max_positions = gate.threshold_config.get("max_positions", 10)
            
            if position_count > max_positions:
                violation = GateViolation(
                    gate_id=gate.id,
                    violation_type="position_limit_exceeded",
                    severity=ViolationSeverity.MEDIUM,
                    message=f"Position count {position_count} exceeds limit {max_positions}",
                    actual_value=position_count,
                    threshold_value=max_positions
                )
                violations.append(violation)
                return GateStatus.FAILED, "Position limit exceeded", violations
                
        elif gate.gate_type == GateType.LOSS_LIMIT:
            current_loss = context.get("current_loss", 0.0)
            max_loss = gate.threshold_config.get("max_loss", -10000.0)
            
            if current_loss < max_loss:
                violation = GateViolation(
                    gate_id=gate.id,
                    violation_type="loss_limit_exceeded",
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Current loss {current_loss} exceeds limit {max_loss}",
                    actual_value=current_loss,
                    threshold_value=max_loss
                )
                violations.append(violation)
                return GateStatus.FAILED, "Loss limit exceeded", violations
                
        # Default pass for other gate types or when thresholds not exceeded
        return GateStatus.PASSED, "Gate validation passed", violations

    def _should_error(self) -> bool:
        """Determine if an error should occur based on error rate"""
        import random
        return random.random() < self.error_rate

    def _setup_default_gates(self):
        """Setup default gates for testing"""
        default_gates = [
            Gate(
                name="Portfolio Risk Check",
                gate_type=GateType.RISK_CHECK,
                priority=100,
                threshold_config={"max_risk": 0.8}
            ),
            Gate(
                name="Position Limit Check",
                gate_type=GateType.POSITION_LIMIT,
                priority=90,
                threshold_config={"max_positions": 20}
            ),
            Gate(
                name="Loss Limit Check",
                gate_type=GateType.LOSS_LIMIT,
                priority=95,
                threshold_config={"max_loss": -5000.0}
            ),
            Gate(
                name="Concentration Check",
                gate_type=GateType.CONCENTRATION,
                priority=80,
                threshold_config={"max_concentration": 0.25}
            )
        ]
        
        for gate in default_gates:
            self.register_gate(gate)


def create_mock_gate_manager(**kwargs) -> MockGateManager:
    """Factory function to create mock gate manager"""
    return MockGateManager(**kwargs)


def create_test_gate(name: str = "Test Gate",
                    gate_type: GateType = GateType.CUSTOM,
                    enabled: bool = True,
                    priority: int = 100) -> Gate:
    """Factory function to create test gate"""
    return Gate(
        name=name,
        gate_type=gate_type,
        enabled=enabled,
        priority=priority
    )


def create_test_violation(gate_id: str = "test_gate",
                         severity: ViolationSeverity = ViolationSeverity.MEDIUM,
                         resolved: bool = False) -> GateViolation:
    """Factory function to create test violation"""
    return GateViolation(
        gate_id=gate_id,
        violation_type="test_violation",
        severity=severity,
        message="Test violation message",
        resolved=resolved
    )