"""
JWT Authentication Middleware for FastAPI

Enforces JWT authentication on all /api/* endpoints except public paths.
Integrates with existing JWT infrastructure in auth.py.
"""

import logging
from typing import Set
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Public endpoints that don't require authentication
PUBLIC_PATHS: Set[str] = {
    "/",
    "/app",
    "/health",
    "/api/health",
    "/api/v1/health",
    "/api/auth/login",
    "/api/auth/register",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce JWT authentication on API endpoints.

    All /api/* endpoints require valid JWT Bearer token except:
    - Public paths (health checks, auth endpoints, docs)
    - Static files
    - WebSocket connections (handled separately)
    """

    def __init__(self, app, verify_token_func=None):
        """
        Initialize JWT authentication middleware.

        Args:
            app: FastAPI application instance
            verify_token_func: Function to verify JWT tokens (from auth.py)
        """
        super().__init__(app)
        self.verify_token_func = verify_token_func
        logger.info("JWT Authentication Middleware initialized")
        logger.info(f"Public paths: {PUBLIC_PATHS}")

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and enforce JWT authentication.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler

        Raises:
            HTTPException: 401 if authentication is required but missing/invalid
        """
        path = request.url.path

        # Allow public paths without authentication
        if self._is_public_path(path):
            return await call_next(request)

        # Allow static files and assets
        if self._is_static_path(path):
            return await call_next(request)

        # Allow WebSocket connections (auth handled in WebSocket handler)
        if path.startswith("/ws"):
            return await call_next(request)

        # Require authentication for all /api/* endpoints
        if path.startswith("/api/"):
            # Check for Authorization header
            auth_header = request.headers.get("Authorization", "")

            if not auth_header:
                logger.warning(f"Missing Authorization header for {path}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authorization header",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            if not auth_header.startswith("Bearer "):
                logger.warning(f"Invalid Authorization header format for {path}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization header format. Expected: Bearer <token>",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Extract token
            token = auth_header[7:]  # Remove "Bearer " prefix

            # Verify token if verification function is available
            if self.verify_token_func:
                try:
                    # Import verify_token from auth.py
                    from jose import JWTError, jwt
                    import os

                    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
                    if not SECRET_KEY:
                        logger.error("JWT_SECRET_KEY not configured")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Authentication not configured"
                        )

                    ALGORITHM = "HS256"

                    # Verify and decode token
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    user_id = payload.get("sub")

                    if not user_id:
                        logger.warning(f"Token missing 'sub' claim for {path}")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid token: missing user identifier",
                            headers={"WWW-Authenticate": "Bearer"}
                        )

                    # Add user_id to request state for use in handlers
                    request.state.user_id = user_id

                    logger.debug(f"Authenticated user {user_id} for {path}")

                except JWTError as e:
                    logger.warning(f"JWT verification failed for {path}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error verifying token for {path}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Authentication error"
                    )
            else:
                logger.warning("Token verification function not available - authentication not enforced")

        # Continue to next handler
        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """
        Check if path is public (no authentication required).

        Args:
            path: Request path

        Returns:
            True if path is public, False otherwise
        """
        # Exact match
        if path in PUBLIC_PATHS:
            return True

        # Prefix match for public paths
        for public_path in PUBLIC_PATHS:
            if path.startswith(public_path + "/"):
                return True

        return False

    def _is_static_path(self, path: str) -> bool:
        """
        Check if path is for static files.

        Args:
            path: Request path

        Returns:
            True if path is for static files, False otherwise
        """
        static_prefixes = ["/assets", "/static", "/favicon.ico"]
        return any(path.startswith(prefix) for prefix in static_prefixes)


def configure_jwt_auth_middleware(app, verify_token_func=None):
    """
    Configure JWT authentication middleware for FastAPI app.

    Args:
        app: FastAPI application instance
        verify_token_func: Optional token verification function

    Returns:
        None
    """
    try:
        # Import verification function if not provided
        if verify_token_func is None:
            try:
                from src.security.auth import verify_token
                verify_token_func = verify_token
                logger.info("Using auth.verify_token for JWT verification")
            except ImportError:
                logger.warning("Could not import verify_token - tokens will be checked for format only")

        # Add middleware
        app.add_middleware(JWTAuthMiddleware, verify_token_func=verify_token_func)

        logger.info("=" * 60)
        logger.info("JWT Authentication Middleware configured successfully")
        logger.info("Protected paths: /api/* (except public paths)")
        logger.info(f"Public paths: {PUBLIC_PATHS}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to configure JWT authentication middleware: {e}")
        raise
