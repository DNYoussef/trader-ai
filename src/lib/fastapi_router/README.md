# FastAPI CRUD Router Component

Standard FastAPI router template with CRUD operations, pagination, and validation.

## LEGO Compatibility

This component imports from `library.common.types`:
- `ValidationResult` - For custom validation in create/update operations

## Installation

```python
# Copy to your project or import from library
from library.components.api.fastapi_router import CRUDRouter, PaginationParams
```

## Usage

### Basic Usage

```python
from fastapi import FastAPI
from library.components.api.fastapi_router import create_crud_router
from your_app.models import Project
from your_app.schemas import ProjectCreate, ProjectUpdate, ProjectResponse
from your_app.database import get_db

app = FastAPI()

# Create CRUD router for Project model
projects_router = create_crud_router(
    model=Project,
    schema_create=ProjectCreate,
    schema_update=ProjectUpdate,
    schema_response=ProjectResponse,
    prefix="/api/projects",
    tags=["Projects"],
    get_db=get_db,
)

app.include_router(projects_router)
```

This creates endpoints:
- `GET /api/projects/` - List with pagination
- `GET /api/projects/{id}` - Get by ID
- `POST /api/projects/` - Create
- `PUT /api/projects/{id}` - Update
- `DELETE /api/projects/{id}` - Delete

### Custom Validation

```python
from library.components.api.fastapi_router import CRUDRouter
from library.common.types import ValidationResult

class ProjectRouter(CRUDRouter):
    def validate_create(self, data):
        errors = []
        if len(data.name) < 3:
            errors.append("Name must be at least 3 characters")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_update(self, data, existing):
        # Custom update validation
        return ValidationResult(valid=True)

router = ProjectRouter(
    model=Project,
    schema_create=ProjectCreate,
    schema_update=ProjectUpdate,
    schema_response=ProjectResponse,
    prefix="/projects",
    tags=["Projects"],
    get_db=get_db,
)
```

### Pagination Parameters

```python
from library.components.api.fastapi_router import PaginationParams

@router.get("/custom")
async def custom_list(pagination: PaginationParams = Depends()):
    # Access pagination.limit, pagination.offset, pagination.sort_by
    pass
```

### Selective Endpoints

```python
# Only create GET endpoints (no create/update/delete)
router = CRUDRouter(
    model=Project,
    # ... schemas ...
    include_create=False,
    include_update=False,
    include_delete=False,
)
```

## API Reference

### CRUDRouter

Generic CRUD router factory.

**Constructor Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Type | SQLAlchemy ORM model class |
| `schema_create` | Type | Pydantic schema for creation |
| `schema_update` | Type | Pydantic schema for updates |
| `schema_response` | Type | Pydantic schema for responses |
| `prefix` | str | URL prefix for routes |
| `tags` | List[str] | OpenAPI documentation tags |
| `get_db` | Callable | Dependency returning AsyncSession |
| `id_field` | str | Primary key field name (default: "id") |
| `include_*` | bool | Enable/disable specific endpoints |

### PaginationParams

Standard pagination dependency.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Items per page (1-100) |
| `offset` | int | 0 | Skip N items |
| `sort_by` | str | None | Sort field (- prefix for desc) |

### PaginatedResponse

Standard paginated response model.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `total` | int | Total items matching query |
| `limit` | int | Items per page |
| `offset` | int | Current offset |
| `items` | List | List of response items |

## Source

Extracted from: `D:\Projects\life-os-dashboard\backend\app\routers\`

Patterns from:
- `registry.py` - Agent listing with pagination
- `auth.py` - Standard CRUD operations
- `crud/project.py` - Async CRUD patterns
