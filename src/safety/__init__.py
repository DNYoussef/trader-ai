"""
Safety module for comprehensive kill switch system
"""

from .kill_switch_system import (
    KillSwitchSystem,
    KillSwitchEvent,
    TriggerType,
    KillSwitchIntegration,
    EmergencyPositionFlattener
)

from .hardware_auth_manager import (
    HardwareAuthManager,
    AuthMethod,
    AuthResult,
    YubiKeyAuthenticator,
    BiometricAuthenticator,
    MasterKeyAuthenticator
)

from .multi_trigger_system import (
    MultiTriggerSystem,
    TriggerCondition,
    TriggerStatus,
    TriggerSeverity,
    APIHealthMonitor,
    PositionMonitor,
    LossLimitMonitor,
    NetworkConnectivityMonitor
)

from .audit_trail_system import (
    AuditTrailSystem,
    AuditEvent,
    EventType,
    EventSeverity,
    AuditStorage
)

__all__ = [
    # Core kill switch
    'KillSwitchSystem',
    'KillSwitchEvent',
    'TriggerType',
    'KillSwitchIntegration',
    'EmergencyPositionFlattener',

    # Hardware authentication
    'HardwareAuthManager',
    'AuthMethod',
    'AuthResult',
    'YubiKeyAuthenticator',
    'BiometricAuthenticator',
    'MasterKeyAuthenticator',

    # Multi-trigger monitoring
    'MultiTriggerSystem',
    'TriggerCondition',
    'TriggerStatus',
    'TriggerSeverity',
    'APIHealthMonitor',
    'PositionMonitor',
    'LossLimitMonitor',
    'NetworkConnectivityMonitor',

    # Audit trail
    'AuditTrailSystem',
    'AuditEvent',
    'EventType',
    'EventSeverity',
    'AuditStorage'
]