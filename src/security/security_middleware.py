"""
Security middleware for FastAPI application.

Provides comprehensive security features including:
- Rate limiting with SlowAPI
- Security headers (HSTS, CSP, etc.)
- HTTPS enforcement in production
"""

import logging
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)


def configure_security_middleware(app: FastAPI, project_root: Path) -> Optional[object]:
    """
    Configure all security middleware for the FastAPI application.

    This includes:
    1. Rate limiting (SlowAPI)
    2. Security headers (HSTS, CSP, X-Frame-Options, etc.)
    3. HTTPS enforcement (production only)

    Args:
        app: FastAPI application instance
        project_root: Path to project root directory

    Returns:
        Limiter object if rate limiting is configured, None otherwise
    """
    limiter = None

    # ==================
    # 1. Rate Limiting
    # ==================
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded

        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        logger.info("✓ Rate limiting configured successfully")
    except ImportError as e:
        logger.warning(f"⚠ Rate limiting not available: {e}")
        logger.warning("  Install with: pip install slowapi")

    # Load security configuration
    config_path = project_root / 'config' / 'config.json'
    security_config = {}

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                security_config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load security config: {e}")

    environment = security_config.get('environment', 'development')
    enforce_https = security_config.get('enforce_https', False)
    security_headers_enabled = security_config.get('security_headers_enabled', True)

    # ========================
    # 2. Security Headers
    # ========================
    if security_headers_enabled:
        @app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            """Add comprehensive security headers to all HTTP responses."""
            response = await call_next(request)

            # HTTP Strict Transport Security (HSTS)
            # Forces browsers to use HTTPS for 1 year
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            # Prevent MIME type sniffing
            # Stops browsers from interpreting files as different MIME type
            response.headers["X-Content-Type-Options"] = "nosniff"

            # Prevent clickjacking attacks
            # Prevents site from being embedded in iframe
            response.headers["X-Frame-Options"] = "DENY"

            # Enable XSS filter in browsers
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Content Security Policy
            # Controls which resources can be loaded
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "                      # Default: same origin only
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Allow inline scripts (needed for React)
                "style-src 'self' 'unsafe-inline'; "        # Allow inline styles
                "img-src 'self' data: https:; "             # Images: self, data URIs, HTTPS
                "connect-src 'self' ws: wss:; "             # WebSocket connections allowed
                "font-src 'self' data:; "                   # Fonts: self and data URIs
                "object-src 'none'; "                       # No Flash/Java applets
                "base-uri 'self'; "                         # Restrict <base> tag
                "form-action 'self';"                       # Forms can only submit to same origin
            )

            # Referrer Policy
            # Controls how much referrer info is sent
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Permissions Policy (formerly Feature-Policy)
            # Disable unnecessary browser features
            response.headers["Permissions-Policy"] = (
                "geolocation=(), "      # Disable geolocation
                "microphone=(), "       # Disable microphone
                "camera=(), "           # Disable camera
                "payment=(), "          # Disable payment API
                "usb=(), "              # Disable USB
                "magnetometer=(), "     # Disable magnetometer
                "gyroscope=(), "        # Disable gyroscope
                "accelerometer=()"      # Disable accelerometer
            )

            return response

        logger.info("✓ Security headers middleware configured")
    else:
        logger.info("ℹ Security headers disabled in config")

    # =======================
    # 3. HTTPS Enforcement
    # =======================
    if environment == "production" and enforce_https:
        @app.middleware("http")
        async def enforce_https_redirect(request: Request, call_next):
            """
            Redirect all HTTP requests to HTTPS in production.

            Returns 301 (Permanent Redirect) to HTTPS version of URL.
            """
            if request.url.scheme != "https":
                # Build HTTPS URL
                https_url = str(request.url).replace("http://", "https://", 1)
                logger.info(f"Redirecting HTTP → HTTPS: {request.url.path}")
                return RedirectResponse(url=https_url, status_code=301)

            return await call_next(request)

        logger.info("✓ HTTPS enforcement enabled (production mode)")
    else:
        logger.info(f"ℹ HTTPS enforcement disabled (environment: {environment})")

    logger.info("=" * 50)
    logger.info("Security middleware configuration complete")
    logger.info(f"  Environment: {environment}")
    logger.info(f"  Rate limiting: {'✓ Enabled' if limiter else '✗ Disabled'}")
    logger.info(f"  Security headers: {'✓ Enabled' if security_headers_enabled else '✗ Disabled'}")
    logger.info(f"  HTTPS enforcement: {'✓ Enabled' if (environment == 'production' and enforce_https) else '✗ Disabled'}")
    logger.info("=" * 50)

    return limiter


# Convenience decorators for rate limiting
def rate_limit_strict(limiter):
    """Strict rate limit: 10 requests per minute per IP"""
    if limiter:
        return limiter.limit("10/minute")
    return lambda f: f  # No-op decorator if limiter not available


def rate_limit_moderate(limiter):
    """Moderate rate limit: 30 requests per minute per IP"""
    if limiter:
        return limiter.limit("30/minute")
    return lambda f: f


def rate_limit_relaxed(limiter):
    """Relaxed rate limit: 60 requests per minute per IP"""
    if limiter:
        return limiter.limit("60/minute")
    return lambda f: f


def rate_limit_websocket(limiter):
    """WebSocket rate limit: 100 connections per minute per IP"""
    if limiter:
        return limiter.limit("100/minute")
    return lambda f: f
