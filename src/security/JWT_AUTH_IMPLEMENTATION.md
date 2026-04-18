# JWT Authentication Middleware Implementation

## Overview

This security implementation adds JWT authentication to all trader-ai API endpoints, protecting sensitive trading operations from unauthorized access.

## Files Modified/Created

### Created Files

1. **src/security/auth_middleware.py**
   - JWT authentication middleware for FastAPI
   - Enforces Bearer token authentication on all /api/* endpoints
   - Integrates with existing auth.py JWT infrastructure

### Modified Files

1. **src/dashboard/run_server_simple.py**
   - Added JWT authentication middleware import (lines 52-59)
   - Integrated middleware in setup_cors() method (lines 153-156)

## Architecture

### Authentication Flow

```
Client Request
    |
    v
[JWT Auth Middleware]
    |
    |--> Public path? --> Allow
    |
    |--> Static file? --> Allow
    |
    |--> WebSocket? --> Allow (auth in WS handler)
    |
    |--> /api/* path?
           |
           |--> Authorization header present?
           |      |
           |      |--> NO --> 401 Unauthorized
           |      |
           |      |--> YES --> Bearer token format?
           |                    |
           |                    |--> NO --> 401 Invalid format
           |                    |
           |                    |--> YES --> Verify JWT signature
           |                                  |
           |                                  |--> Invalid --> 401 Could not validate
           |                                  |
           |                                  |--> Valid --> Add user_id to request.state
           |                                                 |
           |                                                 v
                                                        [Route Handler]
```

### Public Endpoints (No Auth Required)

The following endpoints are accessible without authentication:

- `/` - Root endpoint
- `/app/*` - Frontend SPA routes
- `/health` - Health check
- `/api/health` - API health check
- `/api/v1/health` - Versioned health check
- `/api/auth/login` - Login endpoint
- `/api/auth/register` - Registration endpoint
- `/docs` - API documentation
- `/redoc` - Alternative API docs
- `/openapi.json` - OpenAPI schema
- `/assets/*` - Static assets
- `/ws/*` - WebSocket connections (auth handled separately)

### Protected Endpoints (Auth Required)

All other `/api/*` endpoints now require valid JWT authentication:

#### Trading Operations
- `POST /api/trade/execute/{asset}` - Execute AI-recommended trade
- `POST /api/trading/execute` - Execute real trades

#### Market Data
- `GET /api/metrics/current` - Current risk metrics
- `GET /api/positions` - Portfolio positions
- `GET /api/alerts` - Active alerts
- `GET /api/engine/status` - Trading engine status
- `GET /api/signals/recent` - Recent scan signals
- `GET /api/signals/stats` - Signal statistics

#### AI Analysis
- `GET /api/inequality/data` - Inequality analysis
- `GET /api/contrarian/opportunities` - Contrarian opportunities
- `GET /api/ai/status` - AI calibration status
- `GET /api/ai/timesfm/volatility` - TimesFM volatility forecast
- `GET /api/ai/timesfm/risk` - TimesFM risk predictions
- `GET /api/ai/fingpt/sentiment` - FinGPT sentiment analysis
- `GET /api/ai/fingpt/forecast` - FinGPT price forecast
- `GET /api/ai/features/32d` - Enhanced 32D features

#### Portfolio Management
- `GET /api/barbell/allocation` - Barbell allocation status
- `GET /api/gates/status` - Gate progression status

## Configuration

### Environment Variables

```bash
# Required: JWT secret key for token signing/verification
JWT_SECRET_KEY=your-secret-key-here

# Optional: Token expiration (defaults to 60 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### Setting JWT_SECRET_KEY

**IMPORTANT**: The JWT_SECRET_KEY must be set or the application will fail to start.

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set environment variable (Windows)
set JWT_SECRET_KEY=your-generated-key

# Set environment variable (Linux/Mac)
export JWT_SECRET_KEY=your-generated-key

# Or add to .env file
echo "JWT_SECRET_KEY=your-generated-key" >> .env
```

## Usage

### Client-Side Authentication

#### 1. Obtain JWT Token

```python
import requests

# Login to get JWT token
response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "user", "password": "pass"}
)

token_data = response.json()
access_token = token_data["access_token"]
```

#### 2. Make Authenticated Requests

```python
# Use token in Authorization header
headers = {
    "Authorization": f"Bearer {access_token}"
}

# Request protected endpoint
response = requests.get(
    "http://localhost:8000/api/positions",
    headers=headers
)

positions = response.json()
```

### Server-Side Access to User ID

Route handlers can access the authenticated user ID:

```python
from fastapi import Request

@app.get("/api/my-endpoint")
async def my_endpoint(request: Request):
    # Get authenticated user ID
    user_id = getattr(request.state, 'user_id', None)

    if user_id:
        # User is authenticated
        return {"user_id": user_id, "data": "..."}
    else:
        # Should not happen (middleware ensures auth)
        return {"error": "Not authenticated"}
```

## Error Responses

### 401 Unauthorized - Missing Token

```json
{
    "detail": "Missing authorization header"
}
```

**Response Headers:**
```
WWW-Authenticate: Bearer
```

### 401 Unauthorized - Invalid Format

```json
{
    "detail": "Invalid authorization header format. Expected: Bearer <token>"
}
```

### 401 Unauthorized - Invalid Token

```json
{
    "detail": "Could not validate credentials"
}
```

### 500 Internal Server Error - Configuration Error

```json
{
    "detail": "Authentication not configured"
}
```

## Security Features

### Token Verification

- **Algorithm**: HS256 (HMAC with SHA-256)
- **Signature Verification**: Every token is cryptographically verified
- **Expiration Check**: Tokens expire after configured time (default 60 minutes)
- **Claims Validation**: Token must contain valid "sub" (subject/user_id) claim

### Security Headers

The middleware works in conjunction with existing security middleware:

- **Rate Limiting**: Prevents brute force attacks
- **CORS**: Controls cross-origin access
- **HSTS**: Forces HTTPS in production
- **CSP**: Content Security Policy

### Best Practices Implemented

1. **No Unicode**: All strings use ASCII (per project requirements)
2. **Fail Secure**: Missing/invalid tokens are rejected (not bypassed)
3. **Minimal Attack Surface**: Only necessary endpoints are public
4. **Secure Logging**: Token values are never logged
5. **Clear Error Messages**: Without revealing sensitive details

## Testing

### Manual Testing

```bash
# 1. Test public endpoint (should work without auth)
curl http://localhost:8000/health

# 2. Test protected endpoint without auth (should fail)
curl http://localhost:8000/api/positions

# 3. Test protected endpoint with auth (should work)
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/positions
```

### Automated Testing

Create test file: `tests/test_jwt_auth.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.dashboard.run_server_simple import create_dashboard_server

client = TestClient(create_dashboard_server().app)

def test_public_endpoint_no_auth():
    """Public endpoints should work without auth"""
    response = client.get("/health")
    assert response.status_code == 200

def test_protected_endpoint_no_auth():
    """Protected endpoints should reject requests without auth"""
    response = client.get("/api/positions")
    assert response.status_code == 401
    assert "authorization" in response.json()["detail"].lower()

def test_protected_endpoint_invalid_token():
    """Protected endpoints should reject invalid tokens"""
    response = client.get(
        "/api/positions",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

def test_protected_endpoint_valid_token():
    """Protected endpoints should accept valid tokens"""
    # First login to get valid token
    login_response = client.post(
        "/api/auth/login",
        json={"username": "test", "password": "test"}
    )
    token = login_response.json()["access_token"]

    # Use token to access protected endpoint
    response = client.get(
        "/api/positions",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

## Deployment Checklist

- [ ] JWT_SECRET_KEY environment variable is set
- [ ] JWT_SECRET_KEY is strong (32+ random characters)
- [ ] JWT_SECRET_KEY is different in each environment (dev/staging/prod)
- [ ] JWT_SECRET_KEY is stored securely (not in code)
- [ ] HTTPS is enabled in production
- [ ] Rate limiting is configured
- [ ] Logs do not contain tokens or secrets
- [ ] Token expiration is appropriate for your use case

## Troubleshooting

### Issue: "JWT_SECRET_KEY environment variable must be set"

**Solution**: Set the JWT_SECRET_KEY environment variable before starting the server.

### Issue: 401 Unauthorized on all requests

**Possible causes:**
1. Token not included in Authorization header
2. Token expired
3. Token signed with different secret key
4. Wrong authorization format (must be "Bearer <token>")

### Issue: Middleware not loading

**Check logs for:**
- "JWT authentication middleware loaded" (success)
- "JWT authentication middleware not available" (import error)

**Solution**: Ensure all dependencies are installed:
```bash
pip install python-jose[cryptography]
pip install fastapi
pip install starlette
```

## Future Enhancements

Potential improvements for future iterations:

1. **Token Refresh**: Implement refresh tokens for long-lived sessions
2. **Token Revocation**: Add token blacklist for immediate revocation
3. **Role-Based Access Control**: Add user roles and permissions
4. **API Key Support**: Alternative authentication for service-to-service calls
5. **OAuth2 Integration**: Support for third-party authentication providers
6. **Audit Logging**: Track all authentication events
7. **Rate Limiting by User**: Per-user rate limits instead of just IP-based

## References

- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
- JWT Best Practices: https://tools.ietf.org/html/rfc8725
- OWASP Authentication: https://owasp.org/www-project-authentication/
