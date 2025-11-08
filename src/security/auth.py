"""
JWT Session Authentication Module

Provides JWT token creation and verification for API endpoints.
Replaces insecure query parameter access tokens with session-based authentication.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
import os


# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# HTTPBearer security scheme for Authorization header
security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Dictionary of claims to encode in the token (e.g., {"sub": user_id})
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string

    Example:
        token = create_access_token({"sub": "user123"})
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify JWT token from Authorization header.

    This dependency can be used in FastAPI route handlers to require authentication:

        @app.get("/protected")
        def protected_route(user_id: str = Depends(verify_token)):
            return {"user_id": user_id}

    Args:
        credentials: HTTPAuthCredentials extracted from Authorization: Bearer <token>

    Returns:
        user_id (subject) from the token

    Raises:
        HTTPException: 401 if token is invalid, expired, or missing required claims
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        return user_id

    except JWTError:
        raise credentials_exception


def create_session_token(user_id: str, plaid_item_id: Optional[str] = None) -> dict:
    """
    Create session token with optional Plaid item association.

    Args:
        user_id: User identifier
        plaid_item_id: Optional Plaid item ID to associate with session

    Returns:
        Dictionary containing:
            - access_token: JWT token string
            - token_type: "bearer"
            - expires_in: Seconds until expiration
            - expires_at: ISO timestamp of expiration
    """
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.utcnow() + expires_delta

    token_data = {"sub": user_id}
    if plaid_item_id:
        token_data["plaid_item_id"] = plaid_item_id

    access_token = create_access_token(token_data, expires_delta)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "expires_at": expires_at.isoformat()
    }


def decode_token(token: str) -> dict:
    """
    Decode JWT token without verification (for internal use).

    Args:
        token: JWT token string

    Returns:
        Dictionary of token claims

    Raises:
        JWTError: If token is malformed
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}")
