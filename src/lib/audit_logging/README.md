# Audit Logging Component

A reusable, database-agnostic audit logging system for tracking CREATE, UPDATE, and DELETE operations with comprehensive metadata capture.

## Features

- **Operation Tracking**: Log CREATE, UPDATE, DELETE operations
- **Field-Level Diffs**: Automatic calculation of changed fields for updates
- **User Attribution**: Track who made each change
- **Client Metadata**: Capture IP address and user agent
- **Async-First**: Built for async/await with synchronous fallback
- **Pluggable Models**: Works with any SQLAlchemy-compatible ORM
- **Query Interface**: Flexible filtering for audit trail retrieval

## Installation

This component requires SQLAlchemy. Install it if not already present:

```bash
pip install sqlalchemy
# For async support:
pip install sqlalchemy[asyncio] aiosqlite  # SQLite
# or
pip install sqlalchemy[asyncio] asyncpg    # PostgreSQL
```

## Quick Start

### 1. Create the Audit Model

```python
from sqlalchemy.orm import declarative_base
from audit_logging import create_audit_log_model

Base = declarative_base()
AuditLog = create_audit_log_model(Base, table_name="audit_logs")
```

### 2. Create Tables

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("sqlite+aiosqlite:///app.db")

async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
```

### 3. Use the Logger

```python
from sqlalchemy.ext.asyncio import AsyncSession
from audit_logging import AuditLogger

async def create_user(session: AsyncSession, user_data: dict):
    # Create user in your users table
    user = User(**user_data)
    session.add(user)
    await session.flush()

    # Log the creation
    logger = AuditLogger(session, AuditLog)
    await logger.log_create(
        table_name="users",
        record_id=user.id,
        user_id="admin@example.com",
        ip_address=request.client.host,
        created_data=user_data
    )

    await session.commit()
    return user
```

## Usage Examples

### Logging Updates with Diff

```python
async def update_user(session: AsyncSession, user_id: int, updates: dict):
    # Get current data
    user = await session.get(User, user_id)
    old_data = {"name": user.name, "email": user.email, "status": user.status}

    # Apply updates
    for key, value in updates.items():
        setattr(user, key, value)

    new_data = {"name": user.name, "email": user.email, "status": user.status}

    # Log the update (automatically calculates diff)
    logger = AuditLogger(session, AuditLog)
    await logger.log_update(
        table_name="users",
        record_id=user_id,
        old_data=old_data,
        new_data=new_data,
        user_id=current_user.id
    )
    # Result: changed_fields = {"status": {"old": "active", "new": "suspended"}}

    await session.commit()
```

### Logging Deletions

```python
async def delete_user(session: AsyncSession, user_id: int):
    user = await session.get(User, user_id)

    # Capture data before deletion for audit trail
    deleted_data = {"name": user.name, "email": user.email}

    # Log first, then delete
    logger = AuditLogger(session, AuditLog)
    await logger.log_delete(
        table_name="users",
        record_id=user_id,
        user_id=current_user.id,
        deleted_data=deleted_data
    )

    await session.delete(user)
    await session.commit()
```

### Querying Audit Trail

```python
async def get_user_audit_history(session: AsyncSession, user_id: int):
    logger = AuditLogger(session, AuditLog)

    # Get all changes to a specific user record
    entries = await logger.get_record_history("users", user_id)

    for entry in entries:
        print(f"{entry.timestamp}: {entry.operation} by {entry.user_id}")
        if entry.changed_fields:
            print(f"  Changes: {entry.changed_fields}")
```

### Filtering Audit Entries

```python
from datetime import datetime, timedelta
from audit_logging import AuditOperation

async def audit_report(session: AsyncSession):
    logger = AuditLogger(session, AuditLog)

    # Get all deletions in the last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)

    entries = await logger.get_audit_trail(
        operation=AuditOperation.DELETE,
        start_date=week_ago,
        limit=100
    )

    # Get all changes by a specific admin
    admin_changes = await logger.get_audit_trail(
        user_id="admin@example.com",
        limit=50
    )
```

## Synchronous Usage

For non-async applications:

```python
from sqlalchemy.orm import Session
from audit_logging import SyncAuditLogger

def create_user(session: Session, user_data: dict):
    user = User(**user_data)
    session.add(user)
    session.flush()

    logger = SyncAuditLogger(session, AuditLog)
    logger.log_create("users", user.id, user_id="admin")

    session.commit()
```

## Custom Model Implementation

If you need a custom audit model (e.g., for a different ORM):

```python
from audit_logging import AuditLogModelProtocol, AuditLogger

class MyCustomAuditLog:
    """Must implement AuditLogModelProtocol attributes."""
    id: int
    user_id: str | None
    operation: str
    table_name: str
    record_id: int
    changed_fields: dict | None
    timestamp: datetime
    ip_address: str | None
    user_agent: str | None

# Use with the standard logger
logger = AuditLogger(session, MyCustomAuditLog)
```

## API Reference

### AuditLogger

| Method | Description |
|--------|-------------|
| `log_create()` | Log a CREATE operation |
| `log_update()` | Log an UPDATE with field diff |
| `log_delete()` | Log a DELETE operation |
| `log_operation()` | Generic operation logging |
| `get_audit_trail()` | Query with filters |
| `get_record_history()` | Get history for one record |

### AuditOperation Enum

- `AuditOperation.CREATE`
- `AuditOperation.UPDATE`
- `AuditOperation.DELETE`

### create_audit_log_model()

Factory function to create SQLAlchemy model:

```python
AuditLog = create_audit_log_model(
    base_class=Base,           # SQLAlchemy declarative base
    table_name="audit_logs"    # Optional, defaults to "audit_logs"
)
```

## Schema

The default audit log table schema:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| user_id | String(255) | User who made the change |
| operation | String(10) | CREATE, UPDATE, DELETE |
| table_name | String(100) | Affected table |
| record_id | Integer | Affected record ID |
| changed_fields | JSON | Field-level diff |
| timestamp | DateTime | When it occurred |
| ip_address | String(45) | Client IP (IPv4/IPv6) |
| user_agent | String(500) | Client user agent |

## Source

Extracted from: `D:\Projects\life-os-dashboard\backend\app\core\audit_logging.py`

## Version

1.0.0
