"""
Unit tests for JWT authentication module.

Tests JWT token creation, verification, expiration, and invalid token handling.
"""

import pytest
import os
from datetime import datetime, timedelta
from jose import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

# Set required environment variable for testing
os.environ['JWT_SECRET_KEY'] = 'test-secret-key-do-not-use-in-production-12345'

from src.security.auth import (
    create_access_token,
    verify_token,
    create_session_token,
    decode_token,
    SECRET_KEY,
    ALGORITHM
)


class TestJWTCreation:
    """Test JWT token creation."""

    def test_create_access_token_basic(self):
        """Test creating a basic access token."""
        data = {"sub": "user123"}
        token = create_access_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_custom_expiry(self):
        """Test creating token with custom expiration."""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)

        # Decode to verify expiration
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_timestamp = payload['exp']
        exp_datetime = datetime.utcfromtimestamp(exp_timestamp)

        # Should expire approximately 30 minutes from now
        expected_expiry = datetime.utcnow() + expires_delta
        time_diff = abs((exp_datetime - expected_expiry).total_seconds())

        assert time_diff < 5  # Within 5 seconds tolerance

    def test_create_access_token_includes_claims(self):
        """Test that token includes all provided claims."""
        data = {"sub": "user123", "plaid_item_id": "item_abc"}
        token = create_access_token(data)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload['sub'] == "user123"
        assert payload['plaid_item_id'] == "item_abc"
        assert 'exp' in payload


class TestJWTVerification:
    """Test JWT token verification."""

    def test_verify_valid_token(self):
        """Test verifying a valid token."""
        data = {"sub": "user123"}
        token = create_access_token(data)

        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        user_id = verify_token(credentials)

        assert user_id == "user123"

    def test_verify_expired_token(self):
        """Test verifying an expired token."""
        data = {"sub": "user123"}
        # Create token that expired 1 hour ago
        expires_delta = timedelta(hours=-1)
        token = create_access_token(data, expires_delta)

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)

        assert exc_info.value.status_code == 401

    def test_verify_invalid_token(self):
        """Test verifying a malformed token."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token.string")

        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)

        assert exc_info.value.status_code == 401

    def test_verify_token_missing_subject(self):
        """Test verifying token without 'sub' claim."""
        data = {"user": "user123"}  # Missing 'sub' claim
        token = create_access_token(data)

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)

        assert exc_info.value.status_code == 401

    def test_verify_token_with_wrong_secret(self):
        """Test verifying token signed with different secret."""
        data = {"sub": "user123"}
        # Sign with different secret
        wrong_token = jwt.encode(data, "wrong-secret", algorithm=ALGORITHM)

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=wrong_token)

        with pytest.raises(HTTPException) as exc_info:
            verify_token(credentials)

        assert exc_info.value.status_code == 401


class TestSessionToken:
    """Test session token creation."""

    def test_create_session_token_basic(self):
        """Test creating a session token."""
        result = create_session_token("user123")

        assert result['access_token'] is not None
        assert result['token_type'] == "bearer"
        assert result['expires_in'] == 3600  # 60 minutes in seconds
        assert 'expires_at' in result

    def test_create_session_token_with_plaid_item(self):
        """Test creating session token with Plaid item ID."""
        result = create_session_token("user123", "item_abc")

        token = result['access_token']
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload['sub'] == "user123"
        assert payload['plaid_item_id'] == "item_abc"

    def test_session_token_expiration_format(self):
        """Test that expires_at is in ISO format."""
        result = create_session_token("user123")

        expires_at = result['expires_at']
        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(expires_at)

        assert isinstance(parsed, datetime)


class TestDecodeToken:
    """Test token decoding utility."""

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        data = {"sub": "user123", "custom": "data"}
        token = create_access_token(data)

        payload = decode_token(token)

        assert payload['sub'] == "user123"
        assert payload['custom'] == "data"
        assert 'exp' in payload

    def test_decode_invalid_token(self):
        """Test decoding an invalid token raises error."""
        with pytest.raises(ValueError) as exc_info:
            decode_token("invalid.token.string")

        assert "Invalid token" in str(exc_info.value)


class TestSecurityConfiguration:
    """Test security configuration."""

    def test_secret_key_set(self):
        """Test that SECRET_KEY is configured."""
        assert SECRET_KEY is not None
        assert len(SECRET_KEY) > 0

    def test_algorithm_is_hs256(self):
        """Test that algorithm is HS256."""
        assert ALGORITHM == "HS256"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
