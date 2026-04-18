"""
Pydantic Base Models Component

Common Pydantic models for FastAPI applications.
LEGO-compatible: imports from library.common.types

Provides:
    - PaginatedResponse: Standard paginated list response
    - ErrorResponse: Standard error response format
    - MessageResponse: Simple message response
    - TimestampMixin: Created/updated timestamps
    - AuditMixin: Full audit trail fields
    - StatusEnum: Common status values
    - SortOrder: Sort direction enum

Usage:
    from library.components.api.pydantic_base import (
        PaginatedResponse,
        ErrorResponse,
        TimestampMixin,
    )

    class ProjectResponse(TimestampMixin):
        id: int
        name: str
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LEGO Import: Use shared types from library
try:
    from library.common.types import ValidationResult, Violation, Severity
except ImportError:
    try:
        from common.types import ValidationResult, Violation, Severity
    except ImportError:
        # Fallback for standalone use
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class ValidationResult:
            """Fallback ValidationResult for standalone use."""
            valid: bool
            errors: List[str] = dc_field(default_factory=list)
            warnings: List[str] = dc_field(default_factory=list)
            metadata: Dict[str, Any] = dc_field(default_factory=dict)

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


# Type variable for generic responses
T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================

class StatusEnum(str, Enum):
    """Common status values for entities."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"
    PUBLISHED = "published"


class SortOrder(str, Enum):
    """Sort direction options."""
    ASC = "asc"
    DESC = "desc"


class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    CONFLICT = "conflict"
    INTERNAL_ERROR = "internal_error"
    BAD_REQUEST = "bad_request"
    RATE_LIMITED = "rate_limited"


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """Individual error detail."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Used for all API error responses to maintain consistency.

    Example:
        {
            "error": "validation_error",
            "message": "Validation failed",
            "details": [
                {"field": "name", "message": "Name is required"}
            ]
        }
    """
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request correlation ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Validation failed",
                "details": [
                    {"field": "email", "message": "Invalid email format"}
                ],
                "timestamp": "2026-01-10T12:00:00Z"
            }
        }
    )

    @classmethod
    def from_validation_result(cls, result: ValidationResult, request_id: Optional[str] = None) -> "ErrorResponse":
        """Create ErrorResponse from ValidationResult."""
        return cls(
            error=ErrorCode.VALIDATION_ERROR.value,
            message="Validation failed",
            details=[ErrorDetail(message=err) for err in result.errors],
            request_id=request_id,
        )


class MessageResponse(BaseModel):
    """
    Simple message response.

    Used for operations that return only a status message.

    Example:
        {"message": "Operation completed successfully"}
    """
    message: str = Field(..., description="Response message")
    success: bool = Field(True, description="Operation success status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Resource created successfully",
                "success": True
            }
        }
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated response wrapper.

    Use for all list endpoints that support pagination.

    Example:
        {
            "total": 100,
            "limit": 20,
            "offset": 0,
            "has_more": true,
            "items": [...]
        }
    """
    total: int = Field(..., ge=0, description="Total items matching query")
    limit: int = Field(..., ge=1, le=1000, description="Items per page")
    offset: int = Field(..., ge=0, description="Current offset")
    has_more: bool = Field(..., description="More items available")
    items: List[T] = Field(default_factory=list, description="Page items")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 100,
                "limit": 20,
                "offset": 0,
                "has_more": True,
                "items": []
            }
        }
    )

    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        limit: int,
        offset: int,
    ) -> "PaginatedResponse[T]":
        """
        Factory method to create paginated response.

        Args:
            items: List of items for current page
            total: Total count of all items
            limit: Items per page
            offset: Current offset

        Returns:
            PaginatedResponse with computed has_more
        """
        has_more = (offset + len(items)) < total
        return cls(
            total=total,
            limit=limit,
            offset=offset,
            has_more=has_more,
            items=items,
        )


# =============================================================================
# MIXINS
# =============================================================================

class TimestampMixin(BaseModel):
    """
    Mixin for created_at/updated_at timestamps.

    Add to response models that need timestamp fields.

    Usage:
        class ProjectResponse(TimestampMixin):
            id: int
            name: str
    """
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class AuditMixin(TimestampMixin):
    """
    Extended mixin with full audit trail fields.

    Includes user tracking and soft delete support.
    """
    created_by: Optional[str] = Field(None, description="User who created")
    updated_by: Optional[str] = Field(None, description="User who last updated")
    deleted_at: Optional[datetime] = Field(None, description="Soft delete timestamp")
    deleted_by: Optional[str] = Field(None, description="User who deleted")
    is_deleted: bool = Field(False, description="Soft delete flag")


class VersionMixin(BaseModel):
    """
    Mixin for optimistic locking version field.

    Use with entities that support concurrent updates.
    """
    version: int = Field(1, ge=1, description="Version for optimistic locking")


# =============================================================================
# REQUEST MODELS
# =============================================================================

class PaginationRequest(BaseModel):
    """
    Standard pagination request parameters.

    Use as query params in list endpoints.
    """
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    offset: int = Field(0, ge=0, description="Skip N items")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort direction")

    @field_validator("sort_by")
    @classmethod
    def validate_sort_field(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace and handle empty string."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class FilterRequest(BaseModel):
    """
    Base filter request for search/list endpoints.

    Extend with entity-specific filters.
    """
    search: Optional[str] = Field(None, max_length=100, description="Search query")
    status: Optional[StatusEnum] = Field(None, description="Filter by status")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")

    @field_validator("search")
    @classmethod
    def validate_search(cls, v: Optional[str]) -> Optional[str]:
        """Strip and sanitize search query."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Basic sanitization - remove control characters
            v = "".join(c for c in v if c.isprintable())
        return v


# =============================================================================
# ID WRAPPERS
# =============================================================================

class IDResponse(BaseModel):
    """Response containing only an ID."""
    id: int = Field(..., description="Resource ID")


class IDListResponse(BaseModel):
    """Response containing a list of IDs."""
    ids: List[int] = Field(default_factory=list, description="List of IDs")
    count: int = Field(0, ge=0, description="Number of IDs")


# =============================================================================
# HEALTH CHECK
# =============================================================================

class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")


class HealthResponse(BaseModel):
    """
    Standard health check response.

    Use for /health endpoints.
    """
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Check timestamp"
    )
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Component health details"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2026-01-10T12:00:00Z",
                "components": [
                    {"name": "database", "status": "healthy", "latency_ms": 5.2},
                    {"name": "redis", "status": "healthy", "latency_ms": 1.1}
                ]
            }
        }
    )
