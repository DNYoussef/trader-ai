"""
FastAPI CRUD Router Component

Standard FastAPI router template with CRUD operations and pagination.
LEGO-compatible: imports from library.common.types

Exports:
    - CRUDRouter: Generic CRUD router class
    - PaginationParams: Standard pagination dependency
    - PaginatedResponse: Standard paginated response model
    - create_crud_router: Factory function for quick setup
"""

from .router_template import (
    CRUDRouter,
    PaginationParams,
    PaginatedResponse,
    create_crud_router,
)

__all__ = [
    "CRUDRouter",
    "PaginationParams",
    "PaginatedResponse",
    "create_crud_router",
]

__version__ = "1.0.0"
__component__ = "api/fastapi-router"
