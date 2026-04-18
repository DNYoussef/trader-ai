# JWT Authentication Component

Production-ready JWT authentication with access and refresh token support.

## Features

- Access token creation and verification
- Refresh token support with rotation
- Configurable expiry times
- Token revocation via JTI tracking
- Secure key generation utilities

## Usage

```python
from jwt_auth import JWTAuth, JWTConfig, generate_secure_token

# Generate a secure key
secret_key = generate_secure_token(32)

# Configure auth
config = JWTConfig(
    secret_key=secret_key,
    access_token_expire_minutes=15,
    refresh_token_expire_days=7
)

auth = JWTAuth(config)

# Create tokens
access_token = auth.create_access_token({"sub": "user-123", "role": "admin"})
refresh_token = auth.create_refresh_token({"sub": "user-123"})

# Verify tokens
payload = auth.verify_token(access_token)
print(f"User: {payload['sub']}")

# Refresh access token
new_access = auth.refresh_access_token(refresh_token)

# Rotate refresh token (new access + new refresh)
new_access, new_refresh = auth.rotate_refresh_token(refresh_token)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| secret_key | str | required | HMAC signing key (min 32 chars) |
| algorithm | str | "HS256" | JWT algorithm |
| access_token_expire_minutes | int | 15 | Access token lifetime |
| refresh_token_expire_days | int | 7 | Refresh token lifetime |
| issuer | str | None | Token issuer claim |
| audience | str | None | Token audience claim |

## Security Notes

- Use `generate_secure_token(32)` for production keys
- Store keys in environment variables, never in code
- Rotate keys every 90 days
- Use HTTPS for all token transmission
