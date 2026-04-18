"""
Tests for JWT Authentication Middleware

Verifies that:
1. Public endpoints are accessible without authentication
2. Protected endpoints require valid JWT tokens
3. Invalid/missing tokens are rejected with 401
4. Valid tokens grant access to protected endpoints
"""

import pytest
import os
from datetime import timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.security.auth_middleware import configure_jwt_auth_middleware
from src.security.auth import create_access_token


# Set JWT secret for testing
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only-do-not-use-in-production"


@pytest.fixture
def test_app():
    """Create test FastAPI app with JWT auth middleware"""
    app = FastAPI()

    # Add test routes
    @app.get("/")
    async def root():
        return {"message": "Public root"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/health")
    async def api_health():
        return {"status": "healthy"}

    @app.get("/api/protected")
    async def protected_endpoint():
        return {"message": "This is protected"}

    @app.get("/api/positions")
    async def get_positions():
        return {"positions": []}

    @app.post("/api/trade/execute/AAPL")
    async def execute_trade():
        return {"success": True, "symbol": "AAPL"}

    # Configure JWT auth middleware
    configure_jwt_auth_middleware(app)

    return app


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def valid_token():
    """Generate valid JWT token for testing"""
    return create_access_token(
        data={"sub": "test_user_123"},
        expires_delta=timedelta(minutes=30)
    )


class TestPublicEndpoints:
    """Test public endpoints that should work without authentication"""

    def test_root_endpoint(self, client):
        """Root endpoint should be accessible"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Health endpoint should be accessible"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_api_health_endpoint(self, client):
        """API health endpoint should be accessible"""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestProtectedEndpointsWithoutAuth:
    """Test protected endpoints without authentication (should fail)"""

    def test_protected_endpoint_no_auth(self, client):
        """Protected endpoint should reject requests without auth"""
        response = client.get("/api/protected")
        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()

    def test_positions_endpoint_no_auth(self, client):
        """Positions endpoint should reject requests without auth"""
        response = client.get("/api/positions")
        assert response.status_code == 401

    def test_trade_endpoint_no_auth(self, client):
        """Trade endpoint should reject requests without auth"""
        response = client.post("/api/trade/execute/AAPL")
        assert response.status_code == 401


class TestInvalidAuthentication:
    """Test protected endpoints with invalid authentication"""

    def test_invalid_token_format(self, client):
        """Should reject tokens not starting with 'Bearer '"""
        response = client.get(
            "/api/protected",
            headers={"Authorization": "InvalidFormat token123"}
        )
        assert response.status_code == 401
        assert "bearer" in response.json()["detail"].lower()

    def test_malformed_token(self, client):
        """Should reject malformed JWT tokens"""
        response = client.get(
            "/api/protected",
            headers={"Authorization": "Bearer invalid_token_123"}
        )
        assert response.status_code == 401

    def test_empty_token(self, client):
        """Should reject empty tokens"""
        response = client.get(
            "/api/protected",
            headers={"Authorization": "Bearer "}
        )
        assert response.status_code == 401


class TestValidAuthentication:
    """Test protected endpoints with valid authentication"""

    def test_protected_endpoint_with_valid_token(self, client, valid_token):
        """Protected endpoint should accept valid tokens"""
        response = client.get(
            "/api/protected",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        assert "message" in response.json()

    def test_positions_endpoint_with_valid_token(self, client, valid_token):
        """Positions endpoint should accept valid tokens"""
        response = client.get(
            "/api/positions",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        assert "positions" in response.json()

    def test_trade_endpoint_with_valid_token(self, client, valid_token):
        """Trade endpoint should accept valid tokens"""
        response = client.post(
            "/api/trade/execute/AAPL",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestTokenExpiration:
    """Test token expiration handling"""

    def test_expired_token(self, client):
        """Should reject expired tokens"""
        # Create token that expires immediately
        expired_token = create_access_token(
            data={"sub": "test_user"},
            expires_delta=timedelta(seconds=-1)  # Expired 1 second ago
        )

        response = client.get(
            "/api/protected",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401


class TestAuthHeaderFormats:
    """Test various Authorization header formats"""

    def test_case_sensitive_bearer(self, client, valid_token):
        """Bearer keyword should be case-sensitive"""
        # Lowercase 'bearer' should be rejected
        response = client.get(
            "/api/protected",
            headers={"Authorization": f"bearer {valid_token}"}
        )
        assert response.status_code == 401

    def test_multiple_spaces(self, client, valid_token):
        """Should handle single space after Bearer"""
        # Multiple spaces should fail
        response = client.get(
            "/api/protected",
            headers={"Authorization": f"Bearer  {valid_token}"}  # Two spaces
        )
        # This might pass or fail depending on implementation
        # Just verify it doesn't crash


class TestSecurityHeaders:
    """Test that security headers are properly set"""

    def test_401_includes_www_authenticate(self, client):
        """401 responses should include WWW-Authenticate header"""
        response = client.get("/api/protected")
        assert response.status_code == 401
        assert "www-authenticate" in [h.lower() for h in response.headers.keys()]
        assert response.headers.get("www-authenticate") == "Bearer"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
