"""
FastAPI CRUD Router Template

Standard FastAPI router with CRUD operations, pagination, and validation.
LEGO-compatible: imports from library.common.types

Usage:
    from library.components.api.fastapi_router import CRUDRouter, PaginationParams

    router = CRUDRouter(
        model=MyModel,
        schema_create=MyCreate,
        schema_update=MyUpdate,
        schema_response=MyResponse,
        prefix="/items",
        tags=["Items"]
    )
    app.include_router(router.router)
"""

from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

# LEGO Import: Use shared types from library
try:
    from library.common.types import ValidationResult, Violation, Severity
except ImportError:
    try:
        from common.types import ValidationResult, Violation, Severity
    except ImportError:
        # Fallback for standalone use
        from dataclasses import dataclass, field
        from enum import Enum

        @dataclass
        class ValidationResult:
            """Fallback ValidationResult for standalone use."""
            valid: bool
            errors: List[str] = field(default_factory=list)
            warnings: List[str] = field(default_factory=list)
            metadata: Dict[str, Any] = field(default_factory=dict)

            def __bool__(self) -> bool:
                return self.valid

        @dataclass
        class Violation:
            """Fallback Violation for standalone use."""
            severity: str
            message: str
            file_path: Optional[str] = None
            line: Optional[int] = None

        class Severity(Enum):
            """Fallback Severity enum for standalone use."""
            CRITICAL = "critical"
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"
            INFO = "info"


# Type variables for generics
ModelT = TypeVar("ModelT", bound=DeclarativeBase)
CreateSchemaT = TypeVar("CreateSchemaT", bound=BaseModel)
UpdateSchemaT = TypeVar("UpdateSchemaT", bound=BaseModel)
ResponseSchemaT = TypeVar("ResponseSchemaT", bound=BaseModel)


class PaginationParams:
    """
    Standard pagination parameters for list endpoints.

    Usage:
        @router.get("/items")
        async def list_items(pagination: PaginationParams = Depends()):
            # pagination.limit, pagination.offset, pagination.sort_by
    """

    def __init__(
        self,
        limit: int = Query(20, ge=1, le=100, description="Items per page"),
        offset: int = Query(0, ge=0, description="Number of items to skip"),
        sort_by: Optional[str] = Query(None, description="Sort field (prefix with - for desc)"),
    ):
        self.limit = limit
        self.offset = offset
        self.sort_by = sort_by
        self.descending = False

        if sort_by and sort_by.startswith("-"):
            self.sort_by = sort_by[1:]
            self.descending = True


class PaginatedResponse(BaseModel, Generic[ResponseSchemaT]):
    """
    Standard paginated response wrapper.

    Example:
        {
            "total": 100,
            "limit": 20,
            "offset": 0,
            "items": [...]
        }
    """

    total: int
    limit: int
    offset: int
    items: List[Any]  # Generic type in class definition

    model_config = ConfigDict(from_attributes=True)


class CRUDRouter(Generic[ModelT, CreateSchemaT, UpdateSchemaT, ResponseSchemaT]):
    """
    Generic CRUD router factory for FastAPI.

    Creates standard REST endpoints:
        GET    /{prefix}/       - List with pagination
        GET    /{prefix}/{id}   - Get by ID
        POST   /{prefix}/       - Create
        PUT    /{prefix}/{id}   - Update
        DELETE /{prefix}/{id}   - Delete

    Usage:
        router = CRUDRouter(
            model=Project,
            schema_create=ProjectCreate,
            schema_update=ProjectUpdate,
            schema_response=ProjectResponse,
            prefix="/projects",
            tags=["Projects"],
            get_db=get_db_session,  # Your DB dependency
        )
        app.include_router(router.router)

    Attributes:
        router: The FastAPI APIRouter instance
        model: SQLAlchemy model class
        schema_create: Pydantic model for creation
        schema_update: Pydantic model for updates
        schema_response: Pydantic model for responses
    """

    def __init__(
        self,
        model: Type[ModelT],
        schema_create: Type[CreateSchemaT],
        schema_update: Type[UpdateSchemaT],
        schema_response: Type[ResponseSchemaT],
        prefix: str,
        tags: List[str],
        get_db: Callable,
        id_field: str = "id",
        include_create: bool = True,
        include_read: bool = True,
        include_update: bool = True,
        include_delete: bool = True,
        include_list: bool = True,
    ):
        """
        Initialize CRUD router.

        Args:
            model: SQLAlchemy ORM model class
            schema_create: Pydantic schema for create operations
            schema_update: Pydantic schema for update operations
            schema_response: Pydantic schema for response serialization
            prefix: URL prefix for all routes
            tags: OpenAPI tags for documentation
            get_db: Dependency function returning AsyncSession
            id_field: Primary key field name (default: "id")
            include_create: Generate POST endpoint
            include_read: Generate GET by ID endpoint
            include_update: Generate PUT endpoint
            include_delete: Generate DELETE endpoint
            include_list: Generate GET list endpoint
        """
        self.model = model
        self.schema_create = schema_create
        self.schema_update = schema_update
        self.schema_response = schema_response
        self.id_field = id_field
        self.get_db = get_db

        self.router = APIRouter(prefix=prefix, tags=tags)

        # Register enabled endpoints
        if include_list:
            self._register_list()
        if include_read:
            self._register_get()
        if include_create:
            self._register_create()
        if include_update:
            self._register_update()
        if include_delete:
            self._register_delete()

    def _register_list(self) -> None:
        """Register GET list endpoint with pagination."""

        @self.router.get("/", response_model=PaginatedResponse)
        async def list_items(
            pagination: PaginationParams = Depends(),
            db: AsyncSession = Depends(self.get_db),
        ) -> PaginatedResponse:
            """
            List items with pagination.

            Query params:
                - limit: Items per page (1-100, default 20)
                - offset: Skip N items (default 0)
                - sort_by: Sort field (prefix with - for desc)
            """
            # Count total
            count_query = select(func.count(getattr(self.model, self.id_field)))
            total_result = await db.execute(count_query)
            total = total_result.scalar_one()

            # Build query
            query = select(self.model)

            # Apply sorting
            if pagination.sort_by and hasattr(self.model, pagination.sort_by):
                sort_col = getattr(self.model, pagination.sort_by)
                if pagination.descending:
                    query = query.order_by(sort_col.desc())
                else:
                    query = query.order_by(sort_col.asc())

            # Apply pagination
            query = query.limit(pagination.limit).offset(pagination.offset)

            result = await db.execute(query)
            items = result.scalars().all()

            return PaginatedResponse(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                items=[self.schema_response.model_validate(item) for item in items],
            )

    def _register_get(self) -> None:
        """Register GET by ID endpoint."""

        @self.router.get("/{item_id}", response_model=self.schema_response)
        async def get_item(
            item_id: int,
            db: AsyncSession = Depends(self.get_db),
        ):
            """Get item by ID."""
            query = select(self.model).where(
                getattr(self.model, self.id_field) == item_id
            )
            result = await db.execute(query)
            item = result.scalar_one_or_none()

            if not item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{self.model.__name__} not found",
                )

            return self.schema_response.model_validate(item)

    def _register_create(self) -> None:
        """Register POST create endpoint."""

        @self.router.post(
            "/",
            response_model=self.schema_response,
            status_code=status.HTTP_201_CREATED,
        )
        async def create_item(
            data: self.schema_create,
            db: AsyncSession = Depends(self.get_db),
        ):
            """Create new item."""
            # Validate input
            validation = self.validate_create(data)
            if not validation.valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"errors": validation.errors},
                )

            item = self.model(**data.model_dump())
            db.add(item)
            await db.commit()
            await db.refresh(item)

            return self.schema_response.model_validate(item)

    def _register_update(self) -> None:
        """Register PUT update endpoint."""

        @self.router.put("/{item_id}", response_model=self.schema_response)
        async def update_item(
            item_id: int,
            data: self.schema_update,
            db: AsyncSession = Depends(self.get_db),
        ):
            """Update existing item."""
            query = select(self.model).where(
                getattr(self.model, self.id_field) == item_id
            )
            result = await db.execute(query)
            item = result.scalar_one_or_none()

            if not item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{self.model.__name__} not found",
                )

            # Validate update
            validation = self.validate_update(data, item)
            if not validation.valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"errors": validation.errors},
                )

            # Apply updates (only non-None fields)
            update_data = data.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                if hasattr(item, key):
                    setattr(item, key, value)

            # Update timestamp if available
            if hasattr(item, "updated_at"):
                setattr(item, "updated_at", datetime.utcnow())

            await db.commit()
            await db.refresh(item)

            return self.schema_response.model_validate(item)

    def _register_delete(self) -> None:
        """Register DELETE endpoint."""

        @self.router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
        async def delete_item(
            item_id: int,
            db: AsyncSession = Depends(self.get_db),
        ):
            """Delete item by ID."""
            query = select(self.model).where(
                getattr(self.model, self.id_field) == item_id
            )
            result = await db.execute(query)
            item = result.scalar_one_or_none()

            if not item:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{self.model.__name__} not found",
                )

            await db.delete(item)
            await db.commit()

    def validate_create(self, data: CreateSchemaT) -> ValidationResult:
        """
        Override to add custom create validation.

        Args:
            data: Create schema data

        Returns:
            ValidationResult with valid=True/False and errors list
        """
        return ValidationResult(valid=True)

    def validate_update(self, data: UpdateSchemaT, existing: ModelT) -> ValidationResult:
        """
        Override to add custom update validation.

        Args:
            data: Update schema data
            existing: Existing model instance

        Returns:
            ValidationResult with valid=True/False and errors list
        """
        return ValidationResult(valid=True)


def create_crud_router(
    model: Type[ModelT],
    schema_create: Type[CreateSchemaT],
    schema_update: Type[UpdateSchemaT],
    schema_response: Type[ResponseSchemaT],
    prefix: str,
    tags: List[str],
    get_db: Callable,
    **kwargs: Any,
) -> APIRouter:
    """
    Factory function to create a CRUD router.

    This is a convenience wrapper around CRUDRouter.

    Args:
        model: SQLAlchemy ORM model
        schema_create: Pydantic create schema
        schema_update: Pydantic update schema
        schema_response: Pydantic response schema
        prefix: URL prefix
        tags: OpenAPI tags
        get_db: DB session dependency
        **kwargs: Additional CRUDRouter options

    Returns:
        Configured FastAPI APIRouter
    """
    crud = CRUDRouter(
        model=model,
        schema_create=schema_create,
        schema_update=schema_update,
        schema_response=schema_response,
        prefix=prefix,
        tags=tags,
        get_db=get_db,
        **kwargs,
    )
    return crud.router
