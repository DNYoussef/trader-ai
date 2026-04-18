# Pydantic Base Models Component

Common Pydantic models for FastAPI applications with standard patterns.

## LEGO Compatibility

This component imports from `library.common.types`:
- `ValidationResult` - For error response creation
- `Severity` - For error severity levels

## Installation

```python
from library.components.api.pydantic_base import (
    PaginatedResponse,
    ErrorResponse,
    TimestampMixin,
    StatusEnum,
)
```

## Usage

### Response Models

#### PaginatedResponse

```python
from library.components.api.pydantic_base import PaginatedResponse

# In your endpoint
@router.get("/projects", response_model=PaginatedResponse[ProjectResponse])
async def list_projects(limit: int = 20, offset: int = 0):
    items = await get_projects(limit=limit, offset=offset)
    total = await count_projects()

    return PaginatedResponse.create(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )
```

#### ErrorResponse

```python
from library.components.api.pydantic_base import ErrorResponse, ErrorCode

# Custom error handler
@app.exception_handler(ValidationError)
async def validation_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorCode.VALIDATION_ERROR.value,
            message="Validation failed",
            details=[ErrorDetail(field=e["loc"], message=e["msg"]) for e in exc.errors()],
        ).model_dump()
    )
```

### Mixins

#### TimestampMixin

```python
from library.components.api.pydantic_base import TimestampMixin

class ProjectResponse(TimestampMixin):
    id: int
    name: str
    description: Optional[str]
    # Automatically includes created_at and updated_at

# Output:
# {
#     "id": 1,
#     "name": "My Project",
#     "description": null,
#     "created_at": "2026-01-10T12:00:00Z",
#     "updated_at": "2026-01-10T12:30:00Z"
# }
```

#### AuditMixin

```python
from library.components.api.pydantic_base import AuditMixin

class AuditedResponse(AuditMixin):
    id: int
    # Includes: created_at, updated_at, created_by, updated_by,
    #           deleted_at, deleted_by, is_deleted
```

### Enums

#### StatusEnum

```python
from library.components.api.pydantic_base import StatusEnum

class ProjectCreate(BaseModel):
    name: str
    status: StatusEnum = StatusEnum.DRAFT

# Valid values: active, inactive, pending, archived, deleted, draft, published
```

### Request Models

#### PaginationRequest

```python
from library.components.api.pydantic_base import PaginationRequest

@router.get("/items")
async def list_items(pagination: PaginationRequest = Depends()):
    return await get_items(
        limit=pagination.limit,
        offset=pagination.offset,
        sort_by=pagination.sort_by,
        sort_order=pagination.sort_order.value,
    )
```

#### FilterRequest

```python
from library.components.api.pydantic_base import FilterRequest

class ProjectFilter(FilterRequest):
    category: Optional[str] = None
    priority: Optional[int] = None

@router.get("/projects")
async def search_projects(filters: ProjectFilter = Depends()):
    # filters.search, filters.status, filters.created_after, etc.
    pass
```

### Health Check

```python
from library.components.api.pydantic_base import (
    HealthResponse,
    ComponentHealth,
    HealthStatus,
)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    db_health = await check_database()
    redis_health = await check_redis()

    components = [
        ComponentHealth(name="database", status=db_health.status, latency_ms=db_health.latency),
        ComponentHealth(name="redis", status=redis_health.status, latency_ms=redis_health.latency),
    ]

    overall = HealthStatus.HEALTHY
    if any(c.status == HealthStatus.UNHEALTHY for c in components):
        overall = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in components):
        overall = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall,
        version="1.0.0",
        components=components,
    )
```

## API Reference

### Response Models

| Model | Purpose |
|-------|---------|
| `PaginatedResponse[T]` | Generic paginated list response |
| `ErrorResponse` | Standard error format |
| `ErrorDetail` | Individual error details |
| `MessageResponse` | Simple success/message response |
| `IDResponse` | Single ID wrapper |
| `IDListResponse` | List of IDs wrapper |
| `HealthResponse` | Health check response |
| `ComponentHealth` | Component health details |

### Mixins

| Mixin | Fields |
|-------|--------|
| `TimestampMixin` | created_at, updated_at |
| `AuditMixin` | TimestampMixin + created_by, updated_by, deleted_at, deleted_by, is_deleted |
| `VersionMixin` | version (for optimistic locking) |

### Enums

| Enum | Values |
|------|--------|
| `StatusEnum` | active, inactive, pending, archived, deleted, draft, published |
| `SortOrder` | asc, desc |
| `ErrorCode` | validation_error, not_found, unauthorized, forbidden, conflict, internal_error, bad_request, rate_limited |
| `HealthStatus` | healthy, degraded, unhealthy |

## Source

Extracted from: `D:\Projects\life-os-dashboard\consolidated-ui\backend\app\schemas\`

Patterns from:
- `project_schemas.py` - Response models with validation
- `user_schemas.py` - Authentication schemas
- `__init__.py` - Common patterns
