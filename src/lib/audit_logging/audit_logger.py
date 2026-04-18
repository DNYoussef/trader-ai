"""
Audit Logging Library Component

A reusable, configurable audit logging system for tracking database operations
(CREATE, UPDATE, DELETE) with comprehensive metadata capture.

This module provides:
- AuditLogBase: A SQLAlchemy mixin/base class for audit log tables
- AuditLogger: An async service for creating audit log entries
- Configurable model injection for different ORMs/databases
- Field-level diff tracking for UPDATE operations

Features:
- User tracking (who made the change)
- Timestamp tracking (when the change occurred)
- Field-level change diffs (what was changed)
- Operation type tracking (CREATE, UPDATE, DELETE)
- Client metadata capture (IP address, user agent)
- Async-first design with SQLAlchemy async session support
- Pluggable model architecture for custom implementations

Usage:
    # Option 1: Use the default SQLAlchemy model
    from sqlalchemy.ext.asyncio import AsyncSession
    from audit_logger import AuditLogBase, AuditLogger, create_audit_log_model

    # Create a model bound to your Base
    AuditLog = create_audit_log_model(YourBase)

    # Use the logger
    logger = AuditLogger(session, AuditLog)
    await logger.log_create("users", user.id, user_id="admin")

    # Option 2: Inject your own custom model
    logger = AuditLogger(session, YourCustomAuditModel)
    await logger.log_update("orders", order.id, old_data, new_data)

Author: Library Extraction from life-os-dashboard
License: MIT
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Protocol, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager, asynccontextmanager


class AuditOperation(str, Enum):
    """Enumeration of audit operation types."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class AuditEntry:
    """
    Data class representing an audit log entry.

    Used for returning audit data in a ORM-agnostic format.

    Attributes:
        id: Unique identifier for the audit entry
        user_id: Identifier of the user who performed the operation
        operation: Type of operation (CREATE, UPDATE, DELETE)
        table_name: Name of the affected database table
        record_id: ID of the affected record
        changed_fields: JSON diff of field changes (for UPDATE operations)
        timestamp: When the operation occurred
        ip_address: Client IP address (IPv4 or IPv6)
        user_agent: Client user agent string
    """
    id: Optional[int] = None
    user_id: Optional[str] = None
    operation: str = ""
    table_name: str = ""
    record_id: int = 0
    changed_fields: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogModelProtocol(Protocol):
    """
    Protocol defining the interface that audit log models must implement.

    This allows the AuditLogger to work with any ORM model that conforms
    to this interface, not just SQLAlchemy models.

    Required attributes:
        id: Primary key
        user_id: User who performed the action
        operation: CREATE, UPDATE, or DELETE
        table_name: Affected table name
        record_id: ID of affected record
        changed_fields: JSON dict of changes
        timestamp: When the action occurred
        ip_address: Client IP
        user_agent: Client user agent
    """
    id: int
    user_id: Optional[str]
    operation: str
    table_name: str
    record_id: int
    changed_fields: Optional[Dict[str, Any]]
    timestamp: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]


# Type variable for the audit log model
AuditLogModel = TypeVar("AuditLogModel", bound=AuditLogModelProtocol)


# =============================================================================
# LOW-AUDIT-01 FIX: Shared utility functions extracted to eliminate duplication
# =============================================================================

def calculate_diff(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive field-level diff between old and new data.

    Compares fields and categorizes changes into three types:
    - "added": Fields present in new_data but not in old_data
    - "removed": Fields present in old_data but not in new_data
    - "changed": Fields present in both with different values

    Args:
        old_data: Previous state of the record
        new_data: New state of the record

    Returns:
        Dictionary with three possible keys:
        - "added": Dict of field_name -> new_value for added fields
        - "removed": Dict of field_name -> old_value for removed fields
        - "changed": Dict of field_name -> {"old": ..., "new": ...} for changed fields

    Example:
        old = {"name": "John", "email": "john@old.com", "age": 30}
        new = {"name": "John", "email": "john@new.com", "role": "admin"}
        diff = calculate_diff(old, new)
        # Result:
        # {
        #     "changed": {"email": {"old": "john@old.com", "new": "john@new.com"}},
        #     "added": {"role": "admin"},
        #     "removed": {"age": 30}
        # }
    """
    result: Dict[str, Dict[str, Any]] = {}

    old_keys = set(old_data.keys())
    new_keys = set(new_data.keys())

    # Fields that exist in both - check for changes
    common_keys = old_keys & new_keys
    changed_fields = {}
    for key in common_keys:
        if old_data[key] == new_data[key]:
            continue
        changed_fields[key] = {
            "old": serialize_value(old_data[key]),
            "new": serialize_value(new_data[key]),
        }
    if changed_fields:
        result["changed"] = changed_fields

    # Fields only in new_data - additions
    added_keys = new_keys - old_keys
    if added_keys:
        result["added"] = {key: serialize_value(new_data[key]) for key in added_keys}

    # Fields only in old_data - removals
    removed_keys = old_keys - new_keys
    if removed_keys:
        result["removed"] = {key: serialize_value(old_data[key]) for key in removed_keys}

    return result


def serialize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize all values in a dictionary for JSON storage.

    Args:
        data: Dictionary to serialize

    Returns:
        Dictionary with all values serialized
    """
    return {key: serialize_value(value) for key, value in data.items()}


def serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON storage.

    Handles common non-JSON-serializable types:
    - datetime: Converted to ISO 8601 string
    - bytes: Decoded to UTF-8 string
    - objects with __dict__: Converted to string representation

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable version of the value
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "__dict__"):
        return str(value)
    return value


def create_audit_log_model(base_class: Any, table_name: str = "audit_logs") -> Type:
    """
    Factory function to create an AuditLog SQLAlchemy model.

    Creates a new SQLAlchemy model class bound to the provided declarative base.
    This allows the audit log table to be part of your application's database
    schema without hard-coding dependencies.

    Args:
        base_class: SQLAlchemy declarative base class (e.g., from declarative_base())
        table_name: Name for the audit log table (default: "audit_logs")

    Returns:
        A new SQLAlchemy model class for audit logs

    Example:
        from sqlalchemy.orm import declarative_base
        Base = declarative_base()
        AuditLog = create_audit_log_model(Base, table_name="app_audit_logs")

    Note:
        This function imports SQLAlchemy locally to avoid hard dependency
        at module level. If you're not using SQLAlchemy, you can implement
        your own model that conforms to AuditLogModelProtocol.
    """
    from sqlalchemy import Column, Integer, String, DateTime, JSON

    class AuditLog(base_class):
        """
        SQLAlchemy model for audit log entries.

        Stores comprehensive audit trail for database operations including
        user identification, operation type, affected records, and field-level
        change tracking.
        """
        __tablename__ = table_name

        id = Column(Integer, primary_key=True, index=True, autoincrement=True)
        user_id = Column(String(255), index=True, nullable=True)
        operation = Column(String(10), nullable=False, index=True)
        table_name = Column(String(100), nullable=False, index=True)
        record_id = Column(Integer, nullable=False)
        changed_fields = Column(JSON, nullable=True)
        timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
        ip_address = Column(String(45), nullable=True)  # Supports IPv6
        user_agent = Column(String(500), nullable=True)

        def __repr__(self) -> str:
            return (
                f"<AuditLog(id={self.id}, user={self.user_id}, "
                f"op={self.operation}, table={self.table_name}, "
                f"record={self.record_id}, ts={self.timestamp})>"
            )

        def to_entry(self) -> AuditEntry:
            """Convert SQLAlchemy model to AuditEntry dataclass."""
            return AuditEntry(
                id=self.id,
                user_id=self.user_id,
                operation=self.operation,
                table_name=self.table_name,
                record_id=self.record_id,
                changed_fields=self.changed_fields,
                timestamp=self.timestamp,
                ip_address=self.ip_address,
                user_agent=self.user_agent,
            )

    return AuditLog


class AuditLogger:
    """
    Async service for creating and querying audit log entries.

    Provides a clean interface for logging database operations with
    automatic field-level diff calculation for updates.

    This class is designed to work with SQLAlchemy async sessions but
    can be adapted for other async database libraries by subclassing.

    Attributes:
        session: SQLAlchemy AsyncSession for database operations
        model_class: The audit log model class to use for entries
        session_factory: Optional async session factory for separate transactions

    Transaction Isolation Warning:
        By default, audit log operations use the provided session and participate
        in the same transaction as your business logic. This means:
        - If the business transaction rolls back, audit logs are also lost
        - Audit logs are only visible after the transaction commits

        For audit logs that must persist regardless of business transaction outcome,
        use `separate_transaction=True` (requires session_factory to be set) or
        use the `log_batch()` method with a dedicated session.

    Example:
        # Setup
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        AuditLog = create_audit_log_model(Base)

        # Option 1: Simple usage (audit in same transaction)
        logger = AuditLogger(session, AuditLog)
        await logger.log_create("users", user.id, user_id="admin@example.com")

        # Option 2: With separate transaction support
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        logger = AuditLogger(session, AuditLog, session_factory=session_factory)
        await logger.log_create("users", user.id, user_id="admin", separate_transaction=True)

        # Log operations
        old_user = {"name": "John", "email": "john@old.com"}
        new_user = {"name": "John", "email": "john@new.com"}
        await logger.log_update("users", user.id, old_user, new_user, user_id="admin")

        await logger.log_delete("users", user.id, user_id="admin")

        # Batch logging for performance
        entries = [
            {"operation": "CREATE", "table_name": "orders", "record_id": 1, "user_id": "admin"},
            {"operation": "UPDATE", "table_name": "orders", "record_id": 2, "user_id": "admin",
             "old_data": {"status": "pending"}, "new_data": {"status": "shipped"}},
        ]
        await logger.log_batch(entries)

        # Query audit trail
        entries = await logger.get_audit_trail(table_name="users", limit=50)
    """

    def __init__(
        self,
        session: Any,
        model_class: Type[AuditLogModel],
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize the AuditLogger.

        Args:
            session: SQLAlchemy AsyncSession (or compatible async session)
            model_class: The audit log model class to instantiate for entries.
                         Must have attributes matching AuditLogModelProtocol.
            session_factory: Optional async session factory (e.g., async_sessionmaker)
                            for creating separate transactions. Required if you want
                            to use `separate_transaction=True` in log methods.
        """
        self.session = session
        self.model_class = model_class
        self.session_factory = session_factory

    @asynccontextmanager
    async def _get_session(self, separate_transaction: bool = False):
        """
        Get the appropriate session based on transaction isolation needs.

        Args:
            separate_transaction: If True, creates a new session from factory

        Yields:
            Session to use for the operation

        Raises:
            ValueError: If separate_transaction=True but no session_factory configured
        """
        if not separate_transaction:
            yield self.session
            return

        if self.session_factory is None:
            raise ValueError(
                "Cannot use separate_transaction=True without a session_factory. "
                "Initialize AuditLogger with session_factory parameter."
            )
        async with self.session_factory() as new_session:
            try:
                yield new_session
                await new_session.commit()
            except Exception:
                await new_session.rollback()
                raise

    async def log_create(
        self,
        table_name: str,
        record_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        created_data: Optional[Dict[str, Any]] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """
        Log a CREATE operation.

        Records that a new record was created in the specified table.

        Args:
            table_name: Name of the database table where record was created
            record_id: Primary key ID of the newly created record
            user_id: Identifier of the user who created the record (optional)
            ip_address: IP address of the client (optional, supports IPv4/IPv6)
            user_agent: User agent string from the client (optional)
            created_data: Optional dict of the created record's initial data
                         (stored in changed_fields for reference)
            separate_transaction: If True, creates the audit log in a separate
                                 transaction that commits independently. Requires
                                 session_factory to be configured. Default: False.

        Returns:
            The created audit log entry (model instance)

        Warning:
            By default (separate_transaction=False), the audit log participates in
            the same transaction as your business logic. If the business transaction
            rolls back, the audit log will also be lost. Set separate_transaction=True
            if audit logs must persist regardless of business transaction outcome.

        Example:
            entry = await logger.log_create(
                "orders",
                order.id,
                user_id="user123",
                ip_address="192.168.1.1",
                created_data={"total": 99.99, "status": "pending"}
            )

            # For critical auditing that must survive rollbacks:
            entry = await logger.log_create(
                "orders", order.id, user_id="admin",
                separate_transaction=True
            )
        """
        async with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.CREATE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=serialize_dict(created_data) if created_data else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            await session.flush()
            return audit_entry

    async def log_update(
        self,
        table_name: str,
        record_id: int,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """
        Log an UPDATE operation with comprehensive field-level diff.

        Automatically calculates which fields changed between old and new data,
        tracking additions, removals, and modifications separately.

        Args:
            table_name: Name of the database table where record was updated
            record_id: Primary key ID of the updated record
            old_data: Dictionary of the record's values before the update
            new_data: Dictionary of the record's values after the update
            user_id: Identifier of the user who made the update (optional)
            ip_address: IP address of the client (optional)
            user_agent: User agent string from the client (optional)
            separate_transaction: If True, creates the audit log in a separate
                                 transaction that commits independently. Requires
                                 session_factory to be configured. Default: False.

        Returns:
            The created audit log entry (model instance)

        Warning:
            By default (separate_transaction=False), the audit log participates in
            the same transaction as your business logic. If the business transaction
            rolls back, the audit log will also be lost.

        Example:
            old = {"status": "pending", "total": 99.99, "notes": "rush order"}
            new = {"status": "shipped", "total": 99.99, "tracking": "ABC123"}

            entry = await logger.log_update(
                "orders",
                order.id,
                old,
                new,
                user_id="admin"
            )
            # entry.changed_fields = {
            #     "changed": {"status": {"old": "pending", "new": "shipped"}},
            #     "added": {"tracking": "ABC123"},
            #     "removed": {"notes": "rush order"}
            # }
        """
        changed_fields = calculate_diff(old_data, new_data)

        async with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.UPDATE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=changed_fields if changed_fields else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            await session.flush()
            return audit_entry

    async def log_delete(
        self,
        table_name: str,
        record_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        deleted_data: Optional[Dict[str, Any]] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """
        Log a DELETE operation.

        Records that a record was deleted from the specified table.
        Optionally stores the deleted record's data for recovery purposes.

        Args:
            table_name: Name of the database table where record was deleted
            record_id: Primary key ID of the deleted record
            user_id: Identifier of the user who deleted the record (optional)
            ip_address: IP address of the client (optional)
            user_agent: User agent string from the client (optional)
            deleted_data: Optional dict of the record's data before deletion
                         (stored in changed_fields for potential recovery)
            separate_transaction: If True, creates the audit log in a separate
                                 transaction that commits independently. Requires
                                 session_factory to be configured. Default: False.

        Returns:
            The created audit log entry (model instance)

        Warning:
            By default (separate_transaction=False), the audit log participates in
            the same transaction as your business logic. If the business transaction
            rolls back, the audit log will also be lost.

        Example:
            entry = await logger.log_delete(
                "orders",
                order.id,
                user_id="admin",
                deleted_data={"total": 99.99, "status": "cancelled"}
            )
        """
        async with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.DELETE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=serialize_dict(deleted_data) if deleted_data else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            await session.flush()
            return audit_entry

    async def log_operation(
        self,
        operation: Union[AuditOperation, str],
        table_name: str,
        record_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_data: Optional[Dict[str, Any]] = None,
        new_data: Optional[Dict[str, Any]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """
        Generic method to log any operation type.

        Provides flexibility for logging custom operation types or when
        the operation type is determined at runtime.

        Args:
            operation: Operation type (AuditOperation enum or string)
            table_name: Name of the affected table
            record_id: ID of the affected record
            user_id: User who performed the operation
            ip_address: Client IP address
            user_agent: Client user agent
            old_data: Previous data (for UPDATE operations)
            new_data: New data (for UPDATE operations)
            extra_data: Additional metadata to store in changed_fields
            separate_transaction: If True, creates the audit log in a separate
                                 transaction that commits independently. Requires
                                 session_factory to be configured. Default: False.

        Returns:
            The created audit log entry

        Warning:
            By default (separate_transaction=False), the audit log participates in
            the same transaction as your business logic. If the business transaction
            rolls back, the audit log will also be lost.

        Example:
            # Log a custom operation
            await logger.log_operation(
                "ARCHIVE",
                "documents",
                doc.id,
                user_id="system",
                extra_data={"archive_reason": "retention_policy"}
            )
        """
        op_value = operation.value if isinstance(operation, AuditOperation) else operation

        # Calculate changed_fields based on what's provided
        changed_fields = None
        if old_data and new_data:
            changed_fields = calculate_diff(old_data, new_data)
        elif extra_data:
            changed_fields = serialize_dict(extra_data)

        async with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=op_value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=changed_fields,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            await session.flush()
            return audit_entry

    async def log_batch(
        self,
        entries: List[Dict[str, Any]],
        separate_transaction: bool = False,
    ) -> List[AuditLogModel]:
        """
        Log multiple audit entries in a single transaction for performance.

        Batch logging is significantly more efficient when you need to create
        multiple audit entries at once, as it reduces database round-trips.

        Args:
            entries: List of dictionaries, each containing:
                - operation: str or AuditOperation (required)
                - table_name: str (required)
                - record_id: int (required)
                - user_id: Optional[str]
                - ip_address: Optional[str]
                - user_agent: Optional[str]
                - old_data: Optional[Dict] (for UPDATE, triggers diff calculation)
                - new_data: Optional[Dict] (for UPDATE, triggers diff calculation)
                - created_data: Optional[Dict] (for CREATE)
                - deleted_data: Optional[Dict] (for DELETE)
                - extra_data: Optional[Dict] (fallback for changed_fields)
            separate_transaction: If True, creates audit logs in a separate
                                 transaction that commits independently. Requires
                                 session_factory to be configured. Default: False.

        Returns:
            List of created audit log entries (model instances)

        Warning:
            By default (separate_transaction=False), all audit logs participate in
            the same transaction as your business logic. If the business transaction
            rolls back, all audit logs will also be lost.

        Example:
            entries = [
                {
                    "operation": AuditOperation.CREATE,
                    "table_name": "orders",
                    "record_id": 1,
                    "user_id": "admin",
                    "created_data": {"total": 99.99, "status": "pending"}
                },
                {
                    "operation": AuditOperation.UPDATE,
                    "table_name": "orders",
                    "record_id": 2,
                    "user_id": "admin",
                    "old_data": {"status": "pending"},
                    "new_data": {"status": "shipped"}
                },
                {
                    "operation": "ARCHIVE",
                    "table_name": "documents",
                    "record_id": 100,
                    "user_id": "system",
                    "extra_data": {"archive_reason": "retention_policy"}
                },
            ]
            created_entries = await logger.log_batch(entries)
        """
        async with self._get_session(separate_transaction) as session:
            created_entries = []
            timestamp = datetime.now(timezone.utc)

            for entry_data in entries:
                operation = entry_data.get("operation")
                if operation is None:
                    raise ValueError("Each entry must have an 'operation' field")

                op_value = operation.value if isinstance(operation, AuditOperation) else operation
                table_name = entry_data.get("table_name")
                record_id = entry_data.get("record_id")

                if table_name is None or record_id is None:
                    raise ValueError("Each entry must have 'table_name' and 'record_id' fields")

                # Calculate changed_fields based on what's provided
                changed_fields = None
                old_data = entry_data.get("old_data")
                new_data = entry_data.get("new_data")
                created_data = entry_data.get("created_data")
                deleted_data = entry_data.get("deleted_data")
                extra_data = entry_data.get("extra_data")

                if old_data and new_data:
                    changed_fields = calculate_diff(old_data, new_data)
                elif created_data:
                    changed_fields = serialize_dict(created_data)
                elif deleted_data:
                    changed_fields = serialize_dict(deleted_data)
                elif extra_data:
                    changed_fields = serialize_dict(extra_data)

                audit_entry = self.model_class(
                    user_id=entry_data.get("user_id"),
                    operation=op_value,
                    table_name=table_name,
                    record_id=record_id,
                    changed_fields=changed_fields,
                    timestamp=timestamp,
                    ip_address=entry_data.get("ip_address"),
                    user_agent=entry_data.get("user_agent"),
                )
                session.add(audit_entry)
                created_entries.append(audit_entry)

            await session.flush()
            return created_entries

    async def get_audit_trail(
        self,
        table_name: Optional[str] = None,
        record_id: Optional[int] = None,
        user_id: Optional[str] = None,
        operation: Optional[Union[AuditOperation, str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogModel]:
        """
        Retrieve audit trail entries with flexible filtering.

        Query the audit log with various filters to retrieve relevant entries.
        Results are ordered by timestamp descending (most recent first).

        Args:
            table_name: Filter by affected table name
            record_id: Filter by specific record ID
            user_id: Filter by user who performed the operation
            operation: Filter by operation type (CREATE, UPDATE, DELETE)
            start_date: Filter entries after this datetime (inclusive)
            end_date: Filter entries before this datetime (inclusive)
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip for pagination (default: 0)

        Returns:
            List of audit log entries matching the filters

        Example:
            # Get all changes to a specific record
            entries = await logger.get_audit_trail(
                table_name="orders",
                record_id=123
            )

            # Get all deletions by a specific user in the last 7 days
            from datetime import timedelta
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            entries = await logger.get_audit_trail(
                user_id="admin",
                operation=AuditOperation.DELETE,
                start_date=week_ago,
                limit=50
            )
        """
        from sqlalchemy import select

        query = select(self.model_class).order_by(
            self.model_class.timestamp.desc()
        )

        if table_name:
            query = query.where(self.model_class.table_name == table_name)
        if record_id is not None:
            query = query.where(self.model_class.record_id == record_id)
        if user_id:
            query = query.where(self.model_class.user_id == user_id)
        if operation:
            op_value = operation.value if isinstance(operation, AuditOperation) else operation
            query = query.where(self.model_class.operation == op_value)
        if start_date:
            query = query.where(self.model_class.timestamp >= start_date)
        if end_date:
            query = query.where(self.model_class.timestamp <= end_date)

        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_record_history(
        self,
        table_name: str,
        record_id: int,
        limit: int = 100,
    ) -> List[AuditLogModel]:
        """
        Get the complete audit history for a specific record.

        Convenience method for retrieving all audit entries for a single
        database record, ordered chronologically (oldest first).

        Args:
            table_name: Name of the table containing the record
            record_id: Primary key of the record
            limit: Maximum entries to return (default: 100)

        Returns:
            List of audit entries for the record, oldest first

        Example:
            history = await logger.get_record_history("users", user.id)
            for entry in history:
                print(f"{entry.timestamp}: {entry.operation} by {entry.user_id}")
        """
        from sqlalchemy import select

        query = (
            select(self.model_class)
            .where(self.model_class.table_name == table_name)
            .where(self.model_class.record_id == record_id)
            .order_by(self.model_class.timestamp.asc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # Note: _calculate_diff, _serialize_dict, and _serialize_value have been
    # extracted to module-level functions (calculate_diff, serialize_dict,
    # serialize_value) to eliminate code duplication between AuditLogger and
    # SyncAuditLogger. See LOW-AUDIT-01 fix.


class SyncAuditLogger:
    """
    Synchronous version of AuditLogger for non-async applications.

    Provides the same interface as AuditLogger but uses synchronous
    database operations. Useful for CLI tools, scripts, or applications
    that don't use async/await.

    Transaction Isolation Warning:
        By default, audit log operations use the provided session and participate
        in the same transaction as your business logic. This means:
        - If the business transaction rolls back, audit logs are also lost
        - Audit logs are only visible after the transaction commits

        For audit logs that must persist regardless of business transaction outcome,
        use `separate_transaction=True` (requires session_factory to be set) or
        use the `log_batch()` method with a dedicated session.

    Example:
        from sqlalchemy.orm import Session, sessionmaker
        logger = SyncAuditLogger(session, AuditLog)
        logger.log_create("users", user.id, user_id="admin")

        # With separate transaction support
        session_factory = sessionmaker(bind=engine)
        logger = SyncAuditLogger(session, AuditLog, session_factory=session_factory)
        logger.log_create("users", user.id, user_id="admin", separate_transaction=True)
    """

    def __init__(
        self,
        session: Any,
        model_class: Type[AuditLogModel],
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize the SyncAuditLogger.

        Args:
            session: SQLAlchemy Session (synchronous)
            model_class: The audit log model class
            session_factory: Optional session factory (e.g., sessionmaker)
                            for creating separate transactions.
        """
        self.session = session
        self.model_class = model_class
        self.session_factory = session_factory

    @contextmanager
    def _get_session(self, separate_transaction: bool = False):
        """
        Get the appropriate session based on transaction isolation needs.

        Args:
            separate_transaction: If True, creates a new session from factory

        Yields:
            Session to use for the operation

        Raises:
            ValueError: If separate_transaction=True but no session_factory configured
        """
        if not separate_transaction:
            yield self.session
            return

        if self.session_factory is None:
            raise ValueError(
                "Cannot use separate_transaction=True without a session_factory. "
                "Initialize SyncAuditLogger with session_factory parameter."
            )
        new_session = self.session_factory()
        try:
            yield new_session
            new_session.commit()
        except Exception:
            new_session.rollback()
            raise
        finally:
            new_session.close()

    def log_create(
        self,
        table_name: str,
        record_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        created_data: Optional[Dict[str, Any]] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """Log a CREATE operation synchronously."""
        with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.CREATE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=serialize_dict(created_data) if created_data else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            session.flush()
            return audit_entry

    def log_update(
        self,
        table_name: str,
        record_id: int,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """Log an UPDATE operation synchronously with comprehensive field-level diff."""
        changed_fields = calculate_diff(old_data, new_data)

        with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.UPDATE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=changed_fields if changed_fields else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            session.flush()
            return audit_entry

    def log_delete(
        self,
        table_name: str,
        record_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        deleted_data: Optional[Dict[str, Any]] = None,
        separate_transaction: bool = False,
    ) -> AuditLogModel:
        """Log a DELETE operation synchronously."""
        with self._get_session(separate_transaction) as session:
            audit_entry = self.model_class(
                user_id=user_id,
                operation=AuditOperation.DELETE.value,
                table_name=table_name,
                record_id=record_id,
                changed_fields=serialize_dict(deleted_data) if deleted_data else None,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(audit_entry)
            session.flush()
            return audit_entry

    def log_batch(
        self,
        entries: List[Dict[str, Any]],
        separate_transaction: bool = False,
    ) -> List[AuditLogModel]:
        """
        Log multiple audit entries in a single transaction for performance.

        Args:
            entries: List of dictionaries, each containing operation details.
                    See AuditLogger.log_batch for full documentation.
            separate_transaction: If True, creates audit logs in a separate transaction.

        Returns:
            List of created audit log entries (model instances)
        """
        with self._get_session(separate_transaction) as session:
            created_entries = []
            timestamp = datetime.now(timezone.utc)

            for entry_data in entries:
                operation = entry_data.get("operation")
                if operation is None:
                    raise ValueError("Each entry must have an 'operation' field")

                op_value = operation.value if isinstance(operation, AuditOperation) else operation
                table_name = entry_data.get("table_name")
                record_id = entry_data.get("record_id")

                if table_name is None or record_id is None:
                    raise ValueError("Each entry must have 'table_name' and 'record_id' fields")

                # Calculate changed_fields based on what's provided
                changed_fields = None
                old_data = entry_data.get("old_data")
                new_data = entry_data.get("new_data")
                created_data = entry_data.get("created_data")
                deleted_data = entry_data.get("deleted_data")
                extra_data = entry_data.get("extra_data")

                if old_data and new_data:
                    changed_fields = calculate_diff(old_data, new_data)
                elif created_data:
                    changed_fields = serialize_dict(created_data)
                elif deleted_data:
                    changed_fields = serialize_dict(deleted_data)
                elif extra_data:
                    changed_fields = serialize_dict(extra_data)

                audit_entry = self.model_class(
                    user_id=entry_data.get("user_id"),
                    operation=op_value,
                    table_name=table_name,
                    record_id=record_id,
                    changed_fields=changed_fields,
                    timestamp=timestamp,
                    ip_address=entry_data.get("ip_address"),
                    user_agent=entry_data.get("user_agent"),
                )
                session.add(audit_entry)
                created_entries.append(audit_entry)

            session.flush()
            return created_entries

    def get_audit_trail(
        self,
        table_name: Optional[str] = None,
        record_id: Optional[int] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogModel]:
        """Retrieve audit trail entries synchronously."""
        from sqlalchemy import select

        query = select(self.model_class).order_by(
            self.model_class.timestamp.desc()
        )

        if table_name:
            query = query.where(self.model_class.table_name == table_name)
        if record_id is not None:
            query = query.where(self.model_class.record_id == record_id)
        if user_id:
            query = query.where(self.model_class.user_id == user_id)

        query = query.limit(limit)

        result = self.session.execute(query)
        return list(result.scalars().all())

    # Note: _calculate_diff, _serialize_dict, and _serialize_value have been
    # extracted to module-level functions (calculate_diff, serialize_dict,
    # serialize_value) to eliminate code duplication. See LOW-AUDIT-01 fix.
