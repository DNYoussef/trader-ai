"""
Audit Logging Library Component

A reusable, database-agnostic audit logging system for tracking
CREATE, UPDATE, and DELETE operations with comprehensive metadata.

Quick Start:
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.ext.asyncio import AsyncSession
    from audit_logging import AuditLogger, create_audit_log_model

    # 1. Create audit model bound to your SQLAlchemy Base
    Base = declarative_base()
    AuditLog = create_audit_log_model(Base)

    # 2. Initialize logger with your session
    async def log_user_creation(session: AsyncSession, user):
        logger = AuditLogger(session, AuditLog)
        await logger.log_create("users", user.id, user_id="admin")

    # 3. Log updates with automatic diff calculation
    async def log_user_update(session: AsyncSession, user, old_data, new_data):
        logger = AuditLogger(session, AuditLog)
        await logger.log_update("users", user.id, old_data, new_data)

For synchronous applications:
    from audit_logging import SyncAuditLogger
    logger = SyncAuditLogger(session, AuditLog)
    logger.log_create("users", user.id)

Components:
    - AuditLogger: Async audit logging service
    - SyncAuditLogger: Synchronous audit logging service
    - AuditOperation: Enum of operation types (CREATE, UPDATE, DELETE)
    - AuditEntry: Dataclass for ORM-agnostic audit data
    - AuditLogModelProtocol: Protocol for custom model implementations
    - create_audit_log_model: Factory for SQLAlchemy audit models

Version: 1.0.0
"""

from .audit_logger import (
    # Core service classes
    AuditLogger,
    SyncAuditLogger,
    # Model factory
    create_audit_log_model,
    # Data types
    AuditOperation,
    AuditEntry,
    # Protocol for custom implementations
    AuditLogModelProtocol,
)

__all__ = [
    # Primary exports
    "AuditLogger",
    "SyncAuditLogger",
    "create_audit_log_model",
    # Supporting types
    "AuditOperation",
    "AuditEntry",
    "AuditLogModelProtocol",
]

__version__ = "1.0.0"
