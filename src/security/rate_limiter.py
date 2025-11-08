"""
Rate limiting middleware using SlowAPI.

Provides IP-based rate limiting for API endpoints to prevent abuse
and protect against DDoS attacks.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, FastAPI
import logging

logger = logging.getLogger(__name__)

# Initialize limiter with IP-based key function
limiter = Limiter(key_func=get_remote_address)


def configure_rate_limiting(app: FastAPI) -> None:
    """
    Configure rate limiting for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Attach limiter to app state
    app.state.limiter = limiter

    # Register exception handler for rate limit exceeded
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Rate limiting configured successfully")


# Default rate limit decorators for common use cases
def rate_limit_strict():
    """Strict rate limit: 10 requests per minute per IP"""
    return limiter.limit("10/minute")


def rate_limit_moderate():
    """Moderate rate limit: 30 requests per minute per IP"""
    return limiter.limit("30/minute")


def rate_limit_relaxed():
    """Relaxed rate limit: 60 requests per minute per IP"""
    return limiter.limit("60/minute")


def rate_limit_websocket():
    """WebSocket rate limit: 100 connections per minute per IP"""
    return limiter.limit("100/minute")
