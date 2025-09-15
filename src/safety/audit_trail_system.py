"""
Comprehensive Audit Trail System for Kill Switch
Complete logging of all kill switch events with forensic capabilities
"""

import asyncio
import time
import logging
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import queue
from pathlib import Path
import sqlite3
import gzip
import csv

from .kill_switch_system import TriggerType, KillSwitchEvent
from .hardware_auth_manager import AuthMethod, AuthResult

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Audit event types"""
    KILL_SWITCH_ACTIVATION = "kill_switch_activation"
    AUTHENTICATION_ATTEMPT = "authentication_attempt"
    POSITION_FLATTEN = "position_flatten"
    TRIGGER_ACTIVATION = "trigger_activation"
    SYSTEM_ERROR = "system_error"
    MONITORING_START = "monitoring_start"
    MONITORING_STOP = "monitoring_stop"
    HEARTBEAT = "heartbeat"
    CONFIG_CHANGE = "config_change"
    MANUAL_OVERRIDE = "manual_override"

class EventSeverity(Enum):
    """Event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.SYSTEM_ERROR
    severity: EventSeverity = EventSeverity.INFO

    # Core event data
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None

    # System state
    system_state: Dict[str, Any] = field(default_factory=dict)

    # Authentication context
    auth_method: Optional[AuthMethod] = None
    auth_success: Optional[bool] = None

    # Trading context
    positions_before: Optional[List[Dict]] = None
    positions_after: Optional[List[Dict]] = None
    account_equity: Optional[float] = None

    # Performance metrics
    response_time_ms: Optional[float] = None

    # Integrity
    checksum: str = field(init=False)

    def __post_init__(self):
        """Calculate checksum after initialization"""
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        # Create deterministic string representation
        data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'event_type': self.event_type.value if self.event_type else None,
            'severity': self.severity.value if self.severity else None,
            'message': self.message,
            'details': json.dumps(self.details, sort_keys=True),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'system_state': json.dumps(self.system_state, sort_keys=True)
        }

        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        return self.checksum == self._calculate_checksum()

class AuditStorage:
    """Secure storage for audit events"""

    def __init__(self, storage_path: str = '.claude/.artifacts/audit'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Primary storage files
        self.json_log_path = self.storage_path / 'audit_log.jsonl'
        self.db_path = self.storage_path / 'audit.db'
        self.csv_path = self.storage_path / 'audit_export.csv'

        # Initialize database
        self._init_database()

        # Event queue for async processing
        self.event_queue = queue.Queue(maxsize=1000)
        self.storage_worker = None
        self.storage_active = False

        logger.info(f"Audit storage initialized at {self.storage_path}")

    def _init_database(self):
        """Initialize SQLite database for audit events"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    system_state TEXT,
                    auth_method TEXT,
                    auth_success BOOLEAN,
                    positions_before TEXT,
                    positions_after TEXT,
                    account_equity REAL,
                    response_time_ms REAL,
                    checksum TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def start_storage_worker(self):
        """Start background storage worker"""
        if self.storage_active:
            return

        self.storage_active = True
        self.storage_worker = threading.Thread(target=self._storage_worker_loop)
        self.storage_worker.daemon = True
        self.storage_worker.start()

        logger.info("Audit storage worker started")

    def stop_storage_worker(self):
        """Stop background storage worker"""
        if not self.storage_active:
            return

        self.storage_active = False

        # Process remaining events
        self._flush_event_queue()

        if self.storage_worker:
            self.storage_worker.join(timeout=5.0)

        logger.info("Audit storage worker stopped")

    def _storage_worker_loop(self):
        """Background worker for storing audit events"""
        while self.storage_active:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._store_event_sync(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Storage worker error: {e}")

    def _flush_event_queue(self):
        """Process all remaining events in queue"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._store_event_sync(event)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error flushing event queue: {e}")

    async def store_event(self, event: AuditEvent):
        """Store audit event (async)"""
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logger.error("Audit event queue is full! Event may be lost.")

    def _store_event_sync(self, event: AuditEvent):
        """Store audit event synchronously"""
        try:
            # Verify integrity before storage
            if not event.verify_integrity():
                logger.error(f"Event integrity check failed: {event.event_id}")
                return

            # Store to JSONL file
            self._store_to_jsonl(event)

            # Store to database
            self._store_to_database(event)

        except Exception as e:
            logger.error(f"Event storage failed: {e}")

    def _store_to_jsonl(self, event: AuditEvent):
        """Store event to JSONL file"""
        with open(self.json_log_path, 'a', encoding='utf-8') as f:
            event_dict = asdict(event)
            # Convert enums to strings
            if event_dict['event_type']:
                event_dict['event_type'] = event_dict['event_type'].value
            if event_dict['severity']:
                event_dict['severity'] = event_dict['severity'].value
            if event_dict['auth_method']:
                event_dict['auth_method'] = event_dict['auth_method'].value

            json.dump(event_dict, f, ensure_ascii=False)
            f.write('\n')

    def _store_to_database(self, event: AuditEvent):
        """Store event to SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_events (
                event_id, timestamp, event_type, severity, message, details,
                user_id, session_id, ip_address, system_state, auth_method,
                auth_success, positions_before, positions_after, account_equity,
                response_time_ms, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp,
            event.event_type.value if event.event_type else None,
            event.severity.value if event.severity else None,
            event.message,
            json.dumps(event.details),
            event.user_id,
            event.session_id,
            event.ip_address,
            json.dumps(event.system_state),
            event.auth_method.value if event.auth_method else None,
            event.auth_success,
            json.dumps(event.positions_before) if event.positions_before else None,
            json.dumps(event.positions_after) if event.positions_after else None,
            event.account_equity,
            event.response_time_ms,
            event.checksum
        ))

        conn.commit()
        conn.close()

    def query_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        severity: Optional[EventSeverity] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events with filters"""

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            # Convert row back to AuditEvent
            event_data = dict(row)

            # Convert string enums back
            if event_data['event_type']:
                event_data['event_type'] = EventType(event_data['event_type'])
            if event_data['severity']:
                event_data['severity'] = EventSeverity(event_data['severity'])
            if event_data['auth_method']:
                event_data['auth_method'] = AuthMethod(event_data['auth_method'])

            # Parse JSON fields
            if event_data['details']:
                event_data['details'] = json.loads(event_data['details'])
            if event_data['system_state']:
                event_data['system_state'] = json.loads(event_data['system_state'])
            if event_data['positions_before']:
                event_data['positions_before'] = json.loads(event_data['positions_before'])
            if event_data['positions_after']:
                event_data['positions_after'] = json.loads(event_data['positions_after'])

            # Remove database-specific fields
            event_data.pop('created_at', None)

            # Recreate event (checksum will be recalculated)
            event = AuditEvent(**{k: v for k, v in event_data.items() if v is not None})
            events.append(event)

        return events

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """Export all events to CSV format"""
        if not output_path:
            output_path = str(self.csv_path)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM audit_events ORDER BY timestamp")

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow([desc[0] for desc in cursor.description])

            # Write data
            writer.writerows(cursor.fetchall())

        conn.close()

        logger.info(f"Audit events exported to {output_path}")
        return output_path

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Total events
        cursor.execute("SELECT COUNT(*) FROM audit_events")
        total_events = cursor.fetchone()[0]

        # Events by type
        cursor.execute("""
            SELECT event_type, COUNT(*)
            FROM audit_events
            GROUP BY event_type
            ORDER BY COUNT(*) DESC
        """)
        events_by_type = dict(cursor.fetchall())

        # Events by severity
        cursor.execute("""
            SELECT severity, COUNT(*)
            FROM audit_events
            GROUP BY severity
            ORDER BY COUNT(*) DESC
        """)
        events_by_severity = dict(cursor.fetchall())

        # Recent activity (last 24 hours)
        day_ago = time.time() - 86400
        cursor.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp >= ?", (day_ago,))
        recent_events = cursor.fetchone()[0]

        # Critical events count
        cursor.execute("SELECT COUNT(*) FROM audit_events WHERE severity = 'critical'")
        critical_events = cursor.fetchone()[0]

        conn.close()

        return {
            'total_events': total_events,
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'recent_events_24h': recent_events,
            'critical_events': critical_events,
            'storage_path': str(self.storage_path),
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'jsonl_size': self.json_log_path.stat().st_size if self.json_log_path.exists() else 0
        }

class AuditTrailSystem:
    """Comprehensive audit trail system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize storage
        storage_path = config.get('storage_path', '.claude/.artifacts/audit')
        self.storage = AuditStorage(storage_path)

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()

        # System context
        self.system_context = {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'version': config.get('version', '1.0.0'),
            'environment': config.get('environment', 'production')
        }

        logger.info(f"Audit trail system initialized (session: {self.session_id})")

    def start(self):
        """Start audit trail system"""
        self.storage.start_storage_worker()

        # Log system start
        asyncio.run(self.log_event(
            EventType.MONITORING_START,
            EventSeverity.INFO,
            "Audit trail system started",
            details={'session_id': self.session_id}
        ))

    def stop(self):
        """Stop audit trail system"""
        # Log system stop
        asyncio.run(self.log_event(
            EventType.MONITORING_STOP,
            EventSeverity.INFO,
            "Audit trail system stopped",
            details={'session_id': self.session_id, 'uptime_seconds': time.time() - self.start_time}
        ))

        self.storage.stop_storage_worker()

    async def log_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        auth_result: Optional[AuthResult] = None,
        positions_before: Optional[List[Dict]] = None,
        positions_after: Optional[List[Dict]] = None,
        response_time_ms: Optional[float] = None,
        system_state: Optional[Dict[str, Any]] = None
    ):
        """Log audit event"""

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            details=details or {},
            user_id=user_id,
            session_id=self.session_id,
            system_state=system_state or self.system_context,
            auth_method=auth_result.method if auth_result else None,
            auth_success=auth_result.success if auth_result else None,
            positions_before=positions_before,
            positions_after=positions_after,
            response_time_ms=response_time_ms
        )

        await self.storage.store_event(event)

        # Log critical events immediately
        if severity == EventSeverity.CRITICAL:
            logger.critical(f"AUDIT: {message} - {details}")

    async def log_kill_switch_activation(
        self,
        kill_switch_event: KillSwitchEvent,
        auth_result: Optional[AuthResult] = None,
        positions_before: Optional[List[Dict]] = None,
        positions_after: Optional[List[Dict]] = None
    ):
        """Log kill switch activation event"""

        await self.log_event(
            EventType.KILL_SWITCH_ACTIVATION,
            EventSeverity.CRITICAL,
            f"Kill switch activated: {kill_switch_event.trigger_type.value}",
            details={
                'trigger_type': kill_switch_event.trigger_type.value,
                'trigger_data': kill_switch_event.trigger_data,
                'response_time_ms': kill_switch_event.response_time_ms,
                'positions_flattened': kill_switch_event.positions_flattened,
                'success': kill_switch_event.success,
                'error': kill_switch_event.error
            },
            auth_result=auth_result,
            positions_before=positions_before,
            positions_after=positions_after,
            response_time_ms=kill_switch_event.response_time_ms
        )

    async def log_authentication(self, auth_result: AuthResult, user_id: Optional[str] = None):
        """Log authentication attempt"""

        severity = EventSeverity.INFO if auth_result.success else EventSeverity.WARNING
        message = f"Authentication {'successful' if auth_result.success else 'failed'}: {auth_result.method.value}"

        await self.log_event(
            EventType.AUTHENTICATION_ATTEMPT,
            severity,
            message,
            details={
                'method': auth_result.method.value,
                'duration_ms': auth_result.duration_ms,
                'confidence_score': auth_result.confidence_score,
                'error': auth_result.error
            },
            user_id=user_id,
            auth_result=auth_result,
            response_time_ms=auth_result.duration_ms
        )

    async def log_position_flatten(
        self,
        positions_before: List[Dict],
        positions_after: List[Dict],
        response_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Log position flattening event"""

        severity = EventSeverity.CRITICAL if success else EventSeverity.ERROR
        message = f"Position flatten {'completed' if success else 'failed'}"

        await self.log_event(
            EventType.POSITION_FLATTEN,
            severity,
            message,
            details={
                'positions_before_count': len(positions_before),
                'positions_after_count': len(positions_after),
                'positions_closed': len(positions_before) - len(positions_after),
                'success': success,
                'error': error
            },
            positions_before=positions_before,
            positions_after=positions_after,
            response_time_ms=response_time_ms
        )

    async def log_trigger_activation(
        self,
        trigger_name: str,
        trigger_type: TriggerType,
        current_value: float,
        threshold_value: float,
        consecutive_failures: int
    ):
        """Log trigger activation"""

        await self.log_event(
            EventType.TRIGGER_ACTIVATION,
            EventSeverity.HIGH,
            f"Trigger activated: {trigger_name}",
            details={
                'trigger_name': trigger_name,
                'trigger_type': trigger_type.value,
                'current_value': current_value,
                'threshold_value': threshold_value,
                'consecutive_failures': consecutive_failures
            }
        )

    async def log_heartbeat(self, system_status: Dict[str, Any]):
        """Log system heartbeat"""

        await self.log_event(
            EventType.HEARTBEAT,
            EventSeverity.INFO,
            "System heartbeat",
            details=system_status,
            system_state=system_status
        )

    async def log_system_error(self, error: Exception, context: Dict[str, Any]):
        """Log system error"""

        await self.log_event(
            EventType.SYSTEM_ERROR,
            EventSeverity.ERROR,
            f"System error: {str(error)}",
            details={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            }
        )

    def query_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        severity: Optional[EventSeverity] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events"""
        return self.storage.query_events(start_time, end_time, event_types, severity, user_id, limit)

    def export_events(self, output_path: Optional[str] = None) -> str:
        """Export events to CSV"""
        return self.storage.export_to_csv(output_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        return self.storage.get_statistics()

    def generate_report(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        if not end_time:
            end_time = time.time()
        if not start_time:
            start_time = end_time - 86400  # Last 24 hours

        # Get events in time range
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)

        # Analyze events
        kill_switch_activations = [e for e in events if e.event_type == EventType.KILL_SWITCH_ACTIVATION]
        auth_failures = [e for e in events if e.event_type == EventType.AUTHENTICATION_ATTEMPT and not e.auth_success]
        system_errors = [e for e in events if e.event_type == EventType.SYSTEM_ERROR]

        # Calculate metrics
        total_response_times = [e.response_time_ms for e in events if e.response_time_ms is not None]
        avg_response_time = sum(total_response_times) / len(total_response_times) if total_response_times else 0

        return {
            'report_period': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': (end_time - start_time) / 3600
            },
            'event_summary': {
                'total_events': len(events),
                'kill_switch_activations': len(kill_switch_activations),
                'authentication_failures': len(auth_failures),
                'system_errors': len(system_errors),
                'average_response_time_ms': avg_response_time
            },
            'kill_switch_details': [
                {
                    'timestamp': e.timestamp,
                    'trigger_type': e.details.get('trigger_type'),
                    'response_time_ms': e.response_time_ms,
                    'positions_flattened': e.details.get('positions_flattened'),
                    'success': e.details.get('success')
                }
                for e in kill_switch_activations
            ],
            'auth_failures': [
                {
                    'timestamp': e.timestamp,
                    'method': e.details.get('method'),
                    'user_id': e.user_id,
                    'error': e.details.get('error')
                }
                for e in auth_failures
            ],
            'system_errors': [
                {
                    'timestamp': e.timestamp,
                    'error_type': e.details.get('error_type'),
                    'error_message': e.details.get('error_message')
                }
                for e in system_errors[-10:]  # Last 10 errors
            ]
        }

if __name__ == '__main__':
    # Test audit trail system
    async def test_system():
        config = {
            'storage_path': '.claude/.artifacts/test_audit',
            'version': '1.0.0',
            'environment': 'test'
        }

        audit = AuditTrailSystem(config)
        audit.start()

        # Log some test events
        await audit.log_event(
            EventType.SYSTEM_ERROR,
            EventSeverity.ERROR,
            "Test error event",
            details={'test': True}
        )

        # Simulate kill switch activation
        from .kill_switch_system import KillSwitchEvent
        kill_event = KillSwitchEvent(
            timestamp=time.time(),
            trigger_type=TriggerType.MANUAL_PANIC,
            trigger_data={'test': True},
            response_time_ms=250.5,
            positions_flattened=3,
            authentication_method='master_key',
            success=True
        )

        await audit.log_kill_switch_activation(kill_event)

        # Wait for storage
        await asyncio.sleep(1)

        # Query events
        events = audit.query_events(limit=10)
        print(f"Retrieved {len(events)} events")

        # Generate report
        report = audit.generate_report()
        print(f"Report: {json.dumps(report, indent=2)}")

        # Get statistics
        stats = audit.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2)}")

        audit.stop()

    asyncio.run(test_system())