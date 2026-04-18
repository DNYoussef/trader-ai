"""
Pydantic Base Models Component

Common Pydantic models for FastAPI applications.
LEGO-compatible: imports from library.common.types

Exports:
    Response Models:
        - PaginatedResponse: Standard paginated list response
        - ErrorResponse: Standard error response format
        - ErrorDetail: Individual error detail
        - MessageResponse: Simple message response
        - IDResponse: Single ID response
        - IDListResponse: List of IDs response
        - HealthResponse: Health check response
        - ComponentHealth: Component health status

    Mixins:
        - TimestampMixin: Created/updated timestamps
        - AuditMixin: Full audit trail fields
        - VersionMixin: Optimistic locking version

    Request Models:
        - PaginationRequest: Pagination query params
        - FilterRequest: Base filter/search params

    Enums:
        - StatusEnum: Common status values
        - SortOrder: Sort direction
        - ErrorCode: Standard error codes
        - HealthStatus: Health check statuses
"""

from .base_models import (
    # Response models
    PaginatedResponse,
    ErrorResponse,
    ErrorDetail,
    MessageResponse,
    IDResponse,
    IDListResponse,
    HealthResponse,
    ComponentHealth,
    # Mixins
    TimestampMixin,
    AuditMixin,
    VersionMixin,
    # Request models
    PaginationRequest,
    FilterRequest,
    # Enums
    StatusEnum,
    SortOrder,
    ErrorCode,
    HealthStatus,
)

__all__ = [
    # Response models
    "PaginatedResponse",
    "ErrorResponse",
    "ErrorDetail",
    "MessageResponse",
    "IDResponse",
    "IDListResponse",
    "HealthResponse",
    "ComponentHealth",
    # Mixins
    "TimestampMixin",
    "AuditMixin",
    "VersionMixin",
    # Request models
    "PaginationRequest",
    "FilterRequest",
    # Enums
    "StatusEnum",
    "SortOrder",
    "ErrorCode",
    "HealthStatus",
]

__version__ = "1.0.0"
__component__ = "api/pydantic-base"
