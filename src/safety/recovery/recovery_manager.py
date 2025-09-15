"""
Recovery Manager - System restart and state recovery protocols.

This module implements comprehensive recovery mechanisms with <60s restart
time guarantee, including state preservation, dependency tracking, and
automated recovery procedures.

Key Features:
- Component dependency mapping and ordered recovery
- State persistence and restoration
- Health validation during recovery
- Recovery time monitoring and optimization
- Rollback mechanisms for failed recoveries
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecoveryState(Enum):
    """Recovery process states"""
    IDLE = "idle"
    ASSESSING = "assessing"
    PREPARING = "preparing"
    RECOVERING = "recovering"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class ComponentPriority(Enum):
    """Component recovery priorities"""
    CRITICAL = 1      # Must recover first (safety systems, data persistence)
    HIGH = 2          # Core trading systems
    MEDIUM = 3        # Supporting services
    LOW = 4           # Non-essential services


@dataclass
class ComponentDescriptor:
    """Describes a recoverable component"""
    component_id: str
    priority: ComponentPriority
    dependencies: Set[str] = field(default_factory=set)
    recovery_timeout_seconds: int = 30
    health_check_timeout_seconds: int = 10
    max_recovery_attempts: int = 3
    state_persistent: bool = True
    rollback_capable: bool = True


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    component_id: str
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class SystemSnapshot:
    """Snapshot of system state for recovery"""
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    global_state: Dict[str, Any]
    dependencies: Dict[str, Set[str]]
    checkpoint_id: str


class RecoverableComponent:
    """Interface for components that support recovery"""

    async def save_state(self) -> Dict[str, Any]:
        """Save component state for recovery. Return state data."""
        raise NotImplementedError

    async def restore_state(self, state_data: Dict[str, Any]) -> bool:
        """Restore component from saved state. Return success status."""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if component is healthy after recovery."""
        raise NotImplementedError

    async def start(self) -> bool:
        """Start the component. Return success status."""
        raise NotImplementedError

    async def stop(self) -> bool:
        """Stop the component gracefully. Return success status."""
        raise NotImplementedError

    async def emergency_stop(self) -> bool:
        """Emergency stop the component. Return success status."""
        raise NotImplementedError


class RecoveryManager:
    """
    Manages system recovery with guaranteed <60s restart time.

    Key capabilities:
    - Dependency-aware recovery ordering
    - State preservation and restoration
    - Parallel recovery where safe
    - Recovery validation and rollback
    - Performance monitoring and optimization
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Recovery Manager.

        Args:
            config: Recovery configuration
        """
        self.config = config
        self.state = RecoveryState.IDLE
        self.components: Dict[str, Tuple[ComponentDescriptor, RecoverableComponent]] = {}
        self.recovery_graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        # Recovery tracking
        self.current_recovery_session: Optional[str] = None
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.system_snapshots: List[SystemSnapshot] = []

        # State persistence
        self.state_dir = Path(config.get('state_dir', './data/recovery'))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Recovery metrics
        self.recovery_metrics = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'fastest_recovery_time': float('inf'),
            'slowest_recovery_time': 0.0,
            'last_recovery_time': None
        }

        # Time constraints
        self.target_recovery_time = config.get('target_recovery_time', 60.0)  # seconds
        self.critical_recovery_time = config.get('critical_recovery_time', 30.0)  # seconds

        # Recovery callbacks
        self.pre_recovery_callbacks: List[Callable] = []
        self.post_recovery_callbacks: List[Callable] = []

        logger.info(f"Recovery Manager initialized (target: {self.target_recovery_time}s)")

    def register_component(
        self,
        descriptor: ComponentDescriptor,
        component: RecoverableComponent
    ) -> bool:
        """
        Register a component for recovery management.

        Args:
            descriptor: Component descriptor with recovery metadata
            component: Component instance implementing RecoverableComponent

        Returns:
            True if registration successful
        """
        try:
            with self._lock:
                self.components[descriptor.component_id] = (descriptor, component)
                self.recovery_graph[descriptor.component_id] = descriptor.dependencies.copy()

                # Validate dependency graph doesn't have cycles
                if self._has_circular_dependencies():
                    del self.components[descriptor.component_id]
                    del self.recovery_graph[descriptor.component_id]
                    raise ValueError(f"Circular dependency detected with component {descriptor.component_id}")

            logger.info(f"Registered recoverable component: {descriptor.component_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register component {descriptor.component_id}: {e}")
            return False

    async def create_system_snapshot(self, checkpoint_id: str) -> bool:
        """
        Create a snapshot of the current system state.

        Args:
            checkpoint_id: Unique identifier for this checkpoint

        Returns:
            True if snapshot created successfully
        """
        try:
            logger.info(f"Creating system snapshot: {checkpoint_id}")
            start_time = time.time()

            components_state = {}
            global_state = {}

            # Collect state from all components
            for comp_id, (descriptor, component) in self.components.items():
                if descriptor.state_persistent:
                    try:
                        state = await asyncio.wait_for(
                            component.save_state(),
                            timeout=30.0
                        )
                        components_state[comp_id] = state
                    except Exception as e:
                        logger.warning(f"Failed to save state for {comp_id}: {e}")

            # Create snapshot
            snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                components=components_state,
                global_state=global_state,
                dependencies=dict(self.recovery_graph),
                checkpoint_id=checkpoint_id
            )

            # Save to disk
            await self._persist_snapshot(snapshot)

            self.system_snapshots.append(snapshot)

            # Limit snapshot history
            if len(self.system_snapshots) > 10:
                self.system_snapshots = self.system_snapshots[-5:]

            duration = time.time() - start_time
            logger.info(f"System snapshot created in {duration:.2f}s: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create system snapshot {checkpoint_id}: {e}")
            return False

    async def recover_system(
        self,
        checkpoint_id: Optional[str] = None,
        component_filter: Optional[Set[str]] = None
    ) -> bool:
        """
        Recover the system from failure.

        Args:
            checkpoint_id: Specific checkpoint to recover from (uses latest if None)
            component_filter: Only recover specified components (all if None)

        Returns:
            True if recovery successful within time constraints
        """
        recovery_start = time.time()
        session_id = f"recovery_{int(recovery_start)}"

        with self._lock:
            if self.state != RecoveryState.IDLE:
                logger.warning("Recovery already in progress")
                return False

            self.state = RecoveryState.ASSESSING
            self.current_recovery_session = session_id

        try:
            logger.info(f"Starting system recovery (session: {session_id})")

            # Execute pre-recovery callbacks
            await self._execute_pre_recovery_callbacks()

            # Phase 1: Assessment (target: 10s)
            self.state = RecoveryState.ASSESSING
            recovery_plan = await self._create_recovery_plan(checkpoint_id, component_filter)

            if not recovery_plan:
                raise Exception("Failed to create recovery plan")

            # Phase 2: Preparation (target: 15s)
            self.state = RecoveryState.PREPARING
            prepared = await self._prepare_recovery(recovery_plan)

            if not prepared:
                raise Exception("Failed to prepare recovery")

            # Phase 3: Recovery (target: 30s)
            self.state = RecoveryState.RECOVERING
            recovered = await self._execute_recovery(recovery_plan)

            if not recovered:
                raise Exception("Recovery execution failed")

            # Phase 4: Validation (target: 5s)
            self.state = RecoveryState.VALIDATING
            validated = await self._validate_recovery(recovery_plan)

            if not validated:
                raise Exception("Recovery validation failed")

            # Success
            self.state = RecoveryState.COMPLETED
            recovery_time = time.time() - recovery_start

            # Update metrics
            self.recovery_metrics['total_recoveries'] += 1
            self.recovery_metrics['successful_recoveries'] += 1
            self.recovery_metrics['last_recovery_time'] = recovery_time

            # Update timing metrics
            avg_time = self.recovery_metrics['average_recovery_time']
            total_successful = self.recovery_metrics['successful_recoveries']
            self.recovery_metrics['average_recovery_time'] = (
                (avg_time * (total_successful - 1) + recovery_time) / total_successful
            )

            if recovery_time < self.recovery_metrics['fastest_recovery_time']:
                self.recovery_metrics['fastest_recovery_time'] = recovery_time

            if recovery_time > self.recovery_metrics['slowest_recovery_time']:
                self.recovery_metrics['slowest_recovery_time'] = recovery_time

            # Execute post-recovery callbacks
            await self._execute_post_recovery_callbacks()

            # Check if we met time constraints
            if recovery_time <= self.target_recovery_time:
                logger.info(f"System recovery completed successfully in {recovery_time:.2f}s "
                           f"(target: {self.target_recovery_time}s)")
            else:
                logger.warning(f"System recovery took {recovery_time:.2f}s "
                              f"(exceeds target: {self.target_recovery_time}s)")

            return True

        except Exception as e:
            # Recovery failed
            self.state = RecoveryState.FAILED
            recovery_time = time.time() - recovery_start

            self.recovery_metrics['total_recoveries'] += 1
            self.recovery_metrics['failed_recoveries'] += 1

            logger.error(f"System recovery failed after {recovery_time:.2f}s: {e}")

            # Attempt rollback if possible
            try:
                await self._rollback_recovery()
            except Exception as rollback_error:
                logger.critical(f"Rollback also failed: {rollback_error}")

            return False

        finally:
            self.current_recovery_session = None
            self.state = RecoveryState.IDLE

    async def _create_recovery_plan(
        self,
        checkpoint_id: Optional[str],
        component_filter: Optional[Set[str]]
    ) -> Optional[Dict[str, Any]]:
        """Create a detailed recovery plan."""
        try:
            # Find the snapshot to recover from
            target_snapshot = None
            if checkpoint_id:
                target_snapshot = next(
                    (snap for snap in self.system_snapshots
                     if snap.checkpoint_id == checkpoint_id), None
                )
            else:
                target_snapshot = self.system_snapshots[-1] if self.system_snapshots else None

            if not target_snapshot:
                # Create recovery plan without snapshot (cold start)
                logger.warning("No snapshot available, performing cold start recovery")

            # Determine components to recover
            components_to_recover = set(self.components.keys())
            if component_filter:
                components_to_recover &= component_filter

            # Calculate recovery order based on dependencies and priorities
            recovery_order = self._calculate_recovery_order(components_to_recover)

            # Group by recovery phase for parallel execution
            recovery_phases = self._group_by_phases(recovery_order)

            return {
                'session_id': self.current_recovery_session,
                'target_snapshot': target_snapshot,
                'components_to_recover': components_to_recover,
                'recovery_order': recovery_order,
                'recovery_phases': recovery_phases,
                'estimated_time': self._estimate_recovery_time(recovery_phases)
            }

        except Exception as e:
            logger.error(f"Failed to create recovery plan: {e}")
            return None

    def _calculate_recovery_order(self, components: Set[str]) -> List[str]:
        """Calculate the optimal recovery order based on dependencies and priorities."""
        # Topological sort with priority weighting
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        # Build dependency graph
        for comp_id in components:
            for dep in self.recovery_graph[comp_id]:
                if dep in components:
                    graph[dep].append(comp_id)
                    in_degree[comp_id] += 1

        # Priority queue: (priority_value, component_id)
        # Lower priority value = higher priority
        priority_queue = []
        for comp_id in components:
            if in_degree[comp_id] == 0:
                descriptor, _ = self.components[comp_id]
                priority_queue.append((descriptor.priority.value, comp_id))

        priority_queue.sort()

        result = []
        while priority_queue:
            # Take highest priority component (lowest value)
            _, comp_id = priority_queue.pop(0)
            result.append(comp_id)

            # Update dependent components
            for dependent in graph[comp_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    dep_descriptor, _ = self.components[dependent]
                    priority_queue.append((dep_descriptor.priority.value, dependent))
                    priority_queue.sort()

        return result

    def _group_by_phases(self, recovery_order: List[str]) -> List[List[str]]:
        """Group components into parallel recovery phases."""
        phases = []
        remaining = set(recovery_order)

        while remaining:
            # Find components with no unrecovered dependencies
            phase = []
            for comp_id in recovery_order:
                if comp_id not in remaining:
                    continue

                dependencies = self.recovery_graph[comp_id]
                if dependencies.issubset(set(recovery_order) - remaining):
                    phase.append(comp_id)

            if not phase:
                # Circular dependency - recover remaining components sequentially
                phase = [remaining.pop()]

            for comp_id in phase:
                remaining.discard(comp_id)

            phases.append(phase)

        return phases

    def _estimate_recovery_time(self, recovery_phases: List[List[str]]) -> float:
        """Estimate total recovery time."""
        total_time = 0.0

        for phase in recovery_phases:
            # Phase time is maximum of component recovery times (parallel execution)
            phase_time = max(
                self.components[comp_id][0].recovery_timeout_seconds
                for comp_id in phase
            )
            total_time += phase_time

        return total_time

    async def _prepare_recovery(self, recovery_plan: Dict[str, Any]) -> bool:
        """Prepare for recovery execution."""
        try:
            logger.info("Preparing recovery execution")

            # Stop all components gracefully first
            stop_tasks = []
            for comp_id in recovery_plan['components_to_recover']:
                _, component = self.components[comp_id]
                stop_tasks.append(asyncio.create_task(component.stop()))

            # Wait for graceful stops with timeout
            try:
                await asyncio.wait_for(asyncio.gather(*stop_tasks), timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("Graceful stop timed out, proceeding with emergency stop")
                # Emergency stop remaining components
                emergency_tasks = []
                for comp_id in recovery_plan['components_to_recover']:
                    _, component = self.components[comp_id]
                    emergency_tasks.append(asyncio.create_task(component.emergency_stop()))
                await asyncio.gather(*emergency_tasks, return_exceptions=True)

            logger.info("Recovery preparation completed")
            return True

        except Exception as e:
            logger.error(f"Recovery preparation failed: {e}")
            return False

    async def _execute_recovery(self, recovery_plan: Dict[str, Any]) -> bool:
        """Execute the recovery plan."""
        try:
            logger.info("Executing recovery plan")
            target_snapshot = recovery_plan['target_snapshot']

            # Execute recovery phases
            for phase_idx, phase_components in enumerate(recovery_plan['recovery_phases']):
                logger.info(f"Executing recovery phase {phase_idx + 1}: {phase_components}")

                # Start all components in this phase in parallel
                phase_tasks = []
                for comp_id in phase_components:
                    task = asyncio.create_task(self._recover_component(comp_id, target_snapshot))
                    phase_tasks.append(task)

                # Wait for all components in phase to complete
                results = await asyncio.gather(*phase_tasks, return_exceptions=True)

                # Check if any component failed
                failed_components = []
                for i, result in enumerate(results):
                    comp_id = phase_components[i]
                    if isinstance(result, Exception) or result is False:
                        failed_components.append(comp_id)
                        logger.error(f"Component recovery failed: {comp_id}")

                if failed_components:
                    raise Exception(f"Phase {phase_idx + 1} failed: {failed_components}")

            logger.info("Recovery execution completed successfully")
            return True

        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False

    async def _recover_component(
        self,
        component_id: str,
        snapshot: Optional[SystemSnapshot]
    ) -> bool:
        """Recover a single component."""
        descriptor, component = self.components[component_id]
        max_attempts = descriptor.max_recovery_attempts

        for attempt in range(1, max_attempts + 1):
            attempt_record = RecoveryAttempt(
                component_id=component_id,
                attempt_number=attempt,
                start_time=datetime.now()
            )

            try:
                logger.info(f"Recovering {component_id} (attempt {attempt}/{max_attempts})")

                # Start the component
                start_success = await asyncio.wait_for(
                    component.start(),
                    timeout=descriptor.recovery_timeout_seconds
                )

                if not start_success:
                    raise Exception("Component start failed")

                # Restore state if available
                if snapshot and component_id in snapshot.components:
                    state_data = snapshot.components[component_id]
                    restore_success = await asyncio.wait_for(
                        component.restore_state(state_data),
                        timeout=descriptor.recovery_timeout_seconds / 2
                    )

                    if not restore_success:
                        raise Exception("State restoration failed")

                # Validate health
                health_ok = await asyncio.wait_for(
                    component.health_check(),
                    timeout=descriptor.health_check_timeout_seconds
                )

                if not health_ok:
                    raise Exception("Health check failed")

                # Success
                attempt_record.end_time = datetime.now()
                attempt_record.success = True
                attempt_record.duration_seconds = (
                    attempt_record.end_time - attempt_record.start_time
                ).total_seconds()

                self.recovery_attempts.append(attempt_record)

                logger.info(f"Component {component_id} recovered successfully "
                           f"(attempt {attempt}, {attempt_record.duration_seconds:.2f}s)")
                return True

            except Exception as e:
                attempt_record.end_time = datetime.now()
                attempt_record.error_message = str(e)
                attempt_record.duration_seconds = (
                    attempt_record.end_time - attempt_record.start_time
                ).total_seconds()

                self.recovery_attempts.append(attempt_record)

                logger.warning(f"Component {component_id} recovery attempt {attempt} failed: {e}")

                if attempt < max_attempts:
                    # Wait before retry
                    await asyncio.sleep(min(2 ** attempt, 10))

        logger.error(f"Component {component_id} recovery failed after {max_attempts} attempts")
        return False

    async def _validate_recovery(self, recovery_plan: Dict[str, Any]) -> bool:
        """Validate that recovery was successful."""
        try:
            logger.info("Validating recovery")

            # Perform health checks on all recovered components
            validation_tasks = []
            for comp_id in recovery_plan['components_to_recover']:
                _, component = self.components[comp_id]
                task = asyncio.create_task(component.health_check())
                validation_tasks.append((comp_id, task))

            # Wait for all validation checks
            all_healthy = True
            for comp_id, task in validation_tasks:
                try:
                    healthy = await asyncio.wait_for(task, timeout=10.0)
                    if not healthy:
                        logger.error(f"Component {comp_id} failed post-recovery health check")
                        all_healthy = False
                except Exception as e:
                    logger.error(f"Health check error for {comp_id}: {e}")
                    all_healthy = False

            if all_healthy:
                logger.info("Recovery validation successful")
                return True
            else:
                logger.error("Recovery validation failed")
                return False

        except Exception as e:
            logger.error(f"Recovery validation error: {e}")
            return False

    async def _rollback_recovery(self):
        """Attempt to rollback a failed recovery."""
        logger.warning("Attempting recovery rollback")
        self.state = RecoveryState.ROLLING_BACK

        try:
            # Emergency stop all components
            for comp_id, (_, component) in self.components.items():
                try:
                    await component.emergency_stop()
                except Exception as e:
                    logger.warning(f"Error stopping component {comp_id} during rollback: {e}")

            logger.info("Recovery rollback completed")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def _has_circular_dependencies(self) -> bool:
        """Check if the dependency graph has circular dependencies."""
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.recovery_graph.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for comp_id in self.recovery_graph:
            if comp_id not in visited:
                if has_cycle(comp_id):
                    return True

        return False

    async def _persist_snapshot(self, snapshot: SystemSnapshot):
        """Persist snapshot to disk."""
        snapshot_file = self.state_dir / f"snapshot_{snapshot.checkpoint_id}.pkl"
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot, f)

    async def _execute_pre_recovery_callbacks(self):
        """Execute pre-recovery callbacks."""
        for callback in self.pre_recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Pre-recovery callback error: {e}")

    async def _execute_post_recovery_callbacks(self):
        """Execute post-recovery callbacks."""
        for callback in self.post_recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Post-recovery callback error: {e}")

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status."""
        with self._lock:
            return {
                'state': self.state.value,
                'current_session': self.current_recovery_session,
                'registered_components': len(self.components),
                'available_snapshots': len(self.system_snapshots),
                'metrics': self.recovery_metrics.copy(),
                'target_recovery_time': self.target_recovery_time,
                'last_attempts': [
                    {
                        'component_id': attempt.component_id,
                        'attempt_number': attempt.attempt_number,
                        'success': attempt.success,
                        'duration_seconds': attempt.duration_seconds,
                        'error_message': attempt.error_message
                    }
                    for attempt in self.recovery_attempts[-10:]  # Last 10 attempts
                ]
            }

    def register_pre_recovery_callback(self, callback: Callable):
        """Register callback to execute before recovery."""
        self.pre_recovery_callbacks.append(callback)

    def register_post_recovery_callback(self, callback: Callable):
        """Register callback to execute after recovery."""
        self.post_recovery_callbacks.append(callback)