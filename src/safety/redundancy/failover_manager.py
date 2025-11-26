"""
Failover Manager - Redundancy and failover mechanisms for critical components.

This module implements redundant system components and automatic failover
to eliminate single points of failure in the trading system.

Key Features:
- Primary/backup component pairs
- Automatic failover detection and switching
- Health monitoring of redundant components
- Seamless state synchronization
- Configurable failover policies
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FailoverState(Enum):
    """Failover system states"""
    PRIMARY_ACTIVE = "primary_active"
    BACKUP_ACTIVE = "backup_active"
    BOTH_FAILED = "both_failed"
    SWITCHING = "switching"
    SYNCHRONIZING = "synchronizing"


class ComponentRole(Enum):
    """Component role in redundancy pair"""
    PRIMARY = "primary"
    BACKUP = "backup"


@dataclass
class FailoverPolicy:
    """Failover policy configuration"""
    max_failure_threshold: int = 3  # Max consecutive failures before failover
    failure_window_seconds: int = 60  # Time window for failure counting
    health_check_interval: int = 5  # Health check frequency
    switch_timeout_seconds: int = 30  # Max time allowed for failover
    auto_recovery_enabled: bool = True  # Auto-switch back to primary when recovered
    sync_required: bool = True  # Require state sync before failover


@dataclass
class RedundantComponent:
    """Represents a redundant component pair"""
    component_id: str
    primary_instance: Any
    backup_instance: Any
    current_active: ComponentRole
    policy: FailoverPolicy
    last_health_check: datetime
    failure_count: Dict[ComponentRole, int] = field(default_factory=lambda: {
        ComponentRole.PRIMARY: 0, ComponentRole.BACKUP: 0
    })
    last_failures: Dict[ComponentRole, List[datetime]] = field(default_factory=lambda: {
        ComponentRole.PRIMARY: [], ComponentRole.BACKUP: []
    })
    state_data: Dict[str, Any] = field(default_factory=dict)
    switching_in_progress: bool = False


class RedundantComponentInterface(ABC):
    """Interface that redundant components must implement"""

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check. Return True if healthy."""
        pass

    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """Get current state for synchronization."""
        pass

    @abstractmethod
    async def sync_state(self, state_data: Dict[str, Any]) -> bool:
        """Synchronize state from another instance."""
        pass

    @abstractmethod
    async def activate(self) -> bool:
        """Activate this instance (become primary)."""
        pass

    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate this instance (become backup)."""
        pass

    @abstractmethod
    async def emergency_stop(self) -> bool:
        """Emergency stop for this instance."""
        pass


class FailoverManager:
    """
    Manages failover and redundancy for critical system components.

    Implements automatic failover with the following guarantees:
    - Zero single points of failure
    - Seamless failover with minimal downtime
    - State synchronization between primary/backup
    - Automatic recovery when primary comes back online
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Failover Manager.

        Args:
            config: Failover configuration
        """
        self.config = config
        self.redundant_components: Dict[str, RedundantComponent] = {}
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._monitoring_tasks: List[asyncio.Task] = []

        # Failover callbacks
        self._failover_callbacks: Dict[str, List[Callable]] = {}

        # Metrics
        self.metrics = {
            'failover_events': 0,
            'successful_failovers': 0,
            'failed_failovers': 0,
            'recovery_events': 0,
            'total_downtime_seconds': 0.0
        }

        logger.info("Failover Manager initialized")

    async def register_redundant_component(
        self,
        component_id: str,
        primary_instance: RedundantComponentInterface,
        backup_instance: RedundantComponentInterface,
        policy: Optional[FailoverPolicy] = None
    ) -> bool:
        """
        Register a redundant component pair.

        Args:
            component_id: Unique identifier for the component
            primary_instance: Primary component instance
            backup_instance: Backup component instance
            policy: Failover policy (uses default if None)

        Returns:
            True if registration successful
        """
        try:
            if policy is None:
                policy = FailoverPolicy()

            with self._lock:
                component = RedundantComponent(
                    component_id=component_id,
                    primary_instance=primary_instance,
                    backup_instance=backup_instance,
                    current_active=ComponentRole.PRIMARY,
                    policy=policy,
                    last_health_check=datetime.now()
                )

                self.redundant_components[component_id] = component

                # Initialize both instances
                await primary_instance.activate()
                await backup_instance.deactivate()

                # Start monitoring task
                task = asyncio.create_task(self._monitor_component(component_id))
                self._monitoring_tasks.append(task)

            logger.info(f"Registered redundant component: {component_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register redundant component {component_id}: {e}")
            return False

    def register_failover_callback(self, component_id: str, callback: Callable):
        """Register callback for failover events."""
        if component_id not in self._failover_callbacks:
            self._failover_callbacks[component_id] = []
        self._failover_callbacks[component_id].append(callback)

    async def force_failover(self, component_id: str) -> bool:
        """
        Force failover for a component.

        Args:
            component_id: Component to failover

        Returns:
            True if failover successful
        """
        with self._lock:
            if component_id not in self.redundant_components:
                logger.error(f"Component not found for failover: {component_id}")
                return False

            component = self.redundant_components[component_id]

        return await self._perform_failover(component, "manual_force")

    async def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a redundant component."""
        with self._lock:
            if component_id not in self.redundant_components:
                return None

            component = self.redundant_components[component_id]

            return {
                'component_id': component_id,
                'current_active': component.current_active.value,
                'switching_in_progress': component.switching_in_progress,
                'failure_counts': {
                    'primary': component.failure_count[ComponentRole.PRIMARY],
                    'backup': component.failure_count[ComponentRole.BACKUP]
                },
                'last_health_check': component.last_health_check.isoformat(),
                'policy': {
                    'max_failure_threshold': component.policy.max_failure_threshold,
                    'failure_window_seconds': component.policy.failure_window_seconds,
                    'auto_recovery_enabled': component.policy.auto_recovery_enabled
                }
            }

    async def shutdown(self):
        """Shutdown failover manager."""
        logger.info("Shutting down Failover Manager...")

        self._shutdown_event.set()

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        # Emergency stop all components
        with self._lock:
            for component in self.redundant_components.values():
                try:
                    await component.primary_instance.emergency_stop()
                    await component.backup_instance.emergency_stop()
                except Exception as e:
                    logger.error(f"Error stopping component {component.component_id}: {e}")

        logger.info("Failover Manager shutdown complete")

    async def _monitor_component(self, component_id: str):
        """Monitor a specific redundant component."""
        logger.info(f"Starting monitoring for component: {component_id}")

        try:
            while not self._shutdown_event.is_set():
                with self._lock:
                    if component_id not in self.redundant_components:
                        break
                    component = self.redundant_components[component_id]

                # Perform health checks
                await self._perform_health_checks(component)

                # Check if failover is needed
                if not component.switching_in_progress:
                    await self._evaluate_failover_need(component)

                # Sleep until next check
                await asyncio.sleep(component.policy.health_check_interval)

        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for component: {component_id}")
        except Exception as e:
            logger.error(f"Error monitoring component {component_id}: {e}")

    async def _perform_health_checks(self, component: RedundantComponent):
        """Perform health checks on both instances."""
        current_time = datetime.now()

        # Check primary
        primary_healthy = await self._check_instance_health(
            component.primary_instance, ComponentRole.PRIMARY, component
        )

        # Check backup
        backup_healthy = await self._check_instance_health(
            component.backup_instance, ComponentRole.BACKUP, component
        )

        # Update last health check time
        component.last_health_check = current_time

        # Log health status
        logger.debug(f"Health check {component.component_id}: "
                    f"Primary={primary_healthy}, Backup={backup_healthy}")

    async def _check_instance_health(
        self,
        instance: RedundantComponentInterface,
        role: ComponentRole,
        component: RedundantComponent
    ) -> bool:
        """Check health of a specific instance."""
        try:
            is_healthy = await asyncio.wait_for(
                instance.health_check(),
                timeout=10.0  # 10-second timeout
            )

            if is_healthy:
                # Reset failure count on successful health check
                component.failure_count[role] = 0
                # Clean old failures outside window
                self._clean_old_failures(component, role)
                return True
            else:
                self._record_failure(component, role)
                return False

        except Exception as e:
            logger.warning(f"Health check failed for {component.component_id} {role.value}: {e}")
            self._record_failure(component, role)
            return False

    def _record_failure(self, component: RedundantComponent, role: ComponentRole):
        """Record a failure for a component instance."""
        current_time = datetime.now()

        component.failure_count[role] += 1
        component.last_failures[role].append(current_time)

        # Clean old failures outside window
        self._clean_old_failures(component, role)

    def _clean_old_failures(self, component: RedundantComponent, role: ComponentRole):
        """Clean failures outside the failure window."""
        current_time = datetime.now()
        window = timedelta(seconds=component.policy.failure_window_seconds)
        cutoff_time = current_time - window

        component.last_failures[role] = [
            failure_time for failure_time in component.last_failures[role]
            if failure_time > cutoff_time
        ]

    async def _evaluate_failover_need(self, component: RedundantComponent):
        """Evaluate if failover is needed for a component."""
        active_role = component.current_active
        inactive_role = ComponentRole.BACKUP if active_role == ComponentRole.PRIMARY else ComponentRole.PRIMARY

        # Count failures in window
        active_failures = len(component.last_failures[active_role])
        inactive_failures = len(component.last_failures[inactive_role])

        # Check if active instance has too many failures
        if active_failures >= component.policy.max_failure_threshold:
            if inactive_failures < component.policy.max_failure_threshold:
                logger.warning(f"Failover needed for {component.component_id}: "
                              f"Active ({active_role.value}) has {active_failures} failures")
                await self._perform_failover(component, "health_failure")
            else:
                logger.critical(f"Both instances failed for {component.component_id}")
                await self._handle_both_failed(component)

        # Check for auto-recovery (switch back to primary if it's healthy)
        elif (component.policy.auto_recovery_enabled and
              active_role == ComponentRole.BACKUP and
              len(component.last_failures[ComponentRole.PRIMARY]) == 0):
            logger.info(f"Auto-recovery possible for {component.component_id}")
            await self._perform_failover(component, "auto_recovery")

    async def _perform_failover(self, component: RedundantComponent, reason: str) -> bool:
        """
        Perform failover from current active to backup instance.

        Args:
            component: Component to failover
            reason: Reason for failover

        Returns:
            True if failover successful
        """
        if component.switching_in_progress:
            logger.warning(f"Failover already in progress for {component.component_id}")
            return False

        start_time = time.time()
        component.switching_in_progress = True
        old_active = component.current_active
        new_active = ComponentRole.BACKUP if old_active == ComponentRole.PRIMARY else ComponentRole.PRIMARY

        try:
            logger.info(f"Starting failover for {component.component_id}: "
                       f"{old_active.value} -> {new_active.value} (reason: {reason})")

            old_instance = (component.primary_instance if old_active == ComponentRole.PRIMARY
                          else component.backup_instance)
            new_instance = (component.backup_instance if new_active == ComponentRole.BACKUP
                          else component.primary_instance)

            # Step 1: Sync state if required and possible
            if component.policy.sync_required:
                try:
                    state_data = await asyncio.wait_for(
                        old_instance.get_state(),
                        timeout=10.0
                    )

                    await asyncio.wait_for(
                        new_instance.sync_state(state_data),
                        timeout=10.0
                    )
                    logger.info(f"State synchronized for {component.component_id}")

                except Exception as e:
                    logger.warning(f"State sync failed for {component.component_id}: {e}")
                    # Continue with failover even if sync fails

            # Step 2: Activate new instance
            success = await asyncio.wait_for(
                new_instance.activate(),
                timeout=component.policy.switch_timeout_seconds
            )

            if not success:
                raise Exception("Failed to activate new instance")

            # Step 3: Deactivate old instance
            try:
                await asyncio.wait_for(
                    old_instance.deactivate(),
                    timeout=10.0
                )
            except Exception as e:
                logger.warning(f"Failed to deactivate old instance: {e}")
                # Continue - new instance is already active

            # Step 4: Update component state
            component.current_active = new_active

            # Step 5: Execute failover callbacks
            await self._execute_failover_callbacks(component.component_id, old_active, new_active)

            # Update metrics
            self.metrics['failover_events'] += 1
            self.metrics['successful_failovers'] += 1

            end_time = time.time()
            downtime = end_time - start_time
            self.metrics['total_downtime_seconds'] += downtime

            logger.info(f"Failover completed for {component.component_id}: "
                       f"{old_active.value} -> {new_active.value} "
                       f"(downtime: {downtime:.2f}s)")

            return True

        except Exception as e:
            logger.error(f"Failover failed for {component.component_id}: {e}")
            self.metrics['failed_failovers'] += 1
            return False

        finally:
            component.switching_in_progress = False

    async def _handle_both_failed(self, component: RedundantComponent):
        """Handle situation where both instances have failed."""
        logger.critical(f"CRITICAL: Both instances failed for {component.component_id}")

        # Try emergency recovery
        try:
            # Try to restart primary
            await component.primary_instance.emergency_stop()
            await component.primary_instance.activate()

            # If successful, switch to primary
            component.current_active = ComponentRole.PRIMARY
            logger.info(f"Emergency recovery successful for {component.component_id}")

        except Exception as e:
            logger.critical(f"Emergency recovery failed for {component.component_id}: {e}")
            # This is a critical system failure - alert operators

    async def _execute_failover_callbacks(
        self,
        component_id: str,
        old_active: ComponentRole,
        new_active: ComponentRole
    ):
        """Execute registered failover callbacks."""
        if component_id not in self._failover_callbacks:
            return

        for callback in self._failover_callbacks[component_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, old_active, new_active)
                else:
                    callback(component_id, old_active, new_active)
            except Exception as e:
                logger.error(f"Error executing failover callback: {e}")

    def get_failover_metrics(self) -> Dict[str, Any]:
        """Get failover system metrics."""
        total_failovers = self.metrics['successful_failovers'] + self.metrics['failed_failovers']
        success_rate = (self.metrics['successful_failovers'] / total_failovers * 100
                       if total_failovers > 0 else 100.0)

        avg_downtime = (self.metrics['total_downtime_seconds'] / self.metrics['successful_failovers']
                       if self.metrics['successful_failovers'] > 0 else 0.0)

        return {
            'total_components': len(self.redundant_components),
            'failover_events': self.metrics['failover_events'],
            'successful_failovers': self.metrics['successful_failovers'],
            'failed_failovers': self.metrics['failed_failovers'],
            'success_rate_percentage': success_rate,
            'total_downtime_seconds': self.metrics['total_downtime_seconds'],
            'average_downtime_seconds': avg_downtime,
            'recovery_events': self.metrics['recovery_events']
        }