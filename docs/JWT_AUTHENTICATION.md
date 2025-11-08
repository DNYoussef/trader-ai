# JWT Session Authentication Implementation

## Overview

This implementation replaces insecure query parameter access tokens with JWT (JSON Web Token) session-based authentication for all bank API endpoints.

## Changes Made

### 1. Dependencies Added (`requirements.txt`)
```
python-jose[cryptography]>=3.3.0,<4.0.0
passlib[bcrypt]>=1.7.4,<2.0.0
```

### 2. New Security Module (`src/security/auth.py`)

**Functions:**
- `create_access_token(data, expires_delta)` - Create JWT tokens
- `verify_token(credentials)` - Verify JWT from Authorization header
- `create_session_token(user_id, plaid_item_id)` - Create session with metadata
- `decode_token(token)` - Decode JWT payload

**Configuration:**
- Algorithm: HS256
- Default expiration: 60 minutes
- Secret key from `JWT_SECRET_KEY` environment variable

### 3. Database Schema Updates (`src/finances/bank_database.py`)

**New Table: `user_sessions`**
```sql
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    plaid_item_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY(plaid_item_id) REFERENCES plaid_items(item_id)
);
```

**New Methods:**
- `create_session()` - Store JWT session in database
- `get_session()` - Retrieve session by ID
- `get_access_token_by_user()` - Get Plaid access token for user
- `delete_expired_sessions()` - Clean up expired sessions

### 4. Modified API Endpoints (`src/dashboard/run_server_simple.py`)

#### `POST /api/auth/login`
**New endpoint** for creating JWT sessions.

**Request:**
```json
{
  "user_id": "user123",
  "plaid_item_id": "item_abc"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "expires_at": "2025-11-07T18:00:00"
}
```

#### `POST /api/plaid/exchange_public_token`
**Modified** to return JWT session token instead of raw access_token.

**Before:**
```json
{
  "success": true,
  "access_token": "access-sandbox-xxx"  // INSECURE
}
```

**After:**
```json
{
  "success": true,
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### `GET /api/bank/accounts`, `/api/bank/balances`, `/api/bank/transactions`
**Modified** to use user_id instead of access_token query parameter.

**Before:**
```
GET /api/bank/accounts?access_token=access-sandbox-xxx  // INSECURE
```

**After:**
```
GET /api/bank/accounts?user_id=user123
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

The backend now:
1. Extracts `user_id` from JWT token (or query param for transition)
2. Looks up Plaid `access_token` from database using `user_id`
3. Uses `access_token` to call Plaid API
4. Returns bank data to user

### 5. Tests (`tests/unit/test_auth.py`)

Comprehensive test suite covering:
- ✅ JWT token creation
- ✅ Token verification
- ✅ Expired token handling
- ✅ Invalid token rejection
- ✅ Missing claims detection
- ✅ Session token creation
- ✅ Token decoding utilities

**Run tests:**
```bash
# Set test environment variable
export JWT_SECRET_KEY=test-secret-key

# Run authentication tests
pytest tests/unit/test_auth.py -v
```

### 6. Environment Configuration (`.env.example`)

Template for required environment variables:
```env
JWT_SECRET_KEY=your-secret-key-here
PLAID_CLIENT_ID=your-plaid-client-id
PLAID_SECRET=your-plaid-secret-sandbox
PLAID_ENV=sandbox
```

**Generate secure key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Security Improvements

### Before (Insecure)
❌ Access tokens exposed in URL query parameters
❌ Tokens logged in browser history and server logs
❌ No expiration mechanism
❌ Vulnerable to CSRF attacks
❌ Tokens leaked via Referer headers

### After (Secure)
✅ JWT tokens in Authorization header (not URL)
✅ Automatic expiration (60 minutes)
✅ Tokens cryptographically signed (HS256)
✅ Session management in database
✅ No sensitive data in URLs

## Usage Examples

### 1. User Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

**Response:**
```json
{
  "success": true,
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

### 2. Connect Bank Account
```bash
curl -X POST http://localhost:8000/api/plaid/exchange_public_token \
  -H "Content-Type: application/json" \
  -d '{
    "public_token": "public-sandbox-xxx",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "success": true,
  "session_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 3. Get Bank Accounts (Authenticated)
```bash
curl -X GET "http://localhost:8000/api/bank/accounts?user_id=user123" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Migration Path

For smooth transition, endpoints currently support **both methods**:

1. **Legacy (temporary):** `?user_id=user123` query parameter
2. **New (production):** JWT token in `Authorization: Bearer <token>` header

**Production deployment steps:**
1. Deploy JWT implementation
2. Update frontend to use JWT tokens
3. Monitor for legacy usage
4. Remove query parameter support after migration complete

## Database Maintenance

Clean up expired sessions periodically:

```python
from src.finances.bank_database import init_bank_database

db = init_bank_database()
deleted_count = db.delete_expired_sessions()
print(f"Deleted {deleted_count} expired sessions")
```

**Recommended:** Run daily via cron job or background task.

## Production Checklist

- [ ] Set strong `JWT_SECRET_KEY` (32+ characters random)
- [ ] Use environment variables (never hardcode secrets)
- [ ] Enable HTTPS for all API endpoints
- [ ] Implement token refresh mechanism (optional)
- [ ] Add rate limiting on auth endpoints
- [ ] Monitor for suspicious authentication patterns
- [ ] Set up session cleanup cron job
- [ ] Remove legacy query parameter support
- [ ] Update frontend to use Authorization headers
- [ ] Add audit logging for authentication events

## Architecture Diagram

```
┌─────────────┐                ┌──────────────┐
│   Frontend  │                │   Backend    │
│             │                │   API        │
└─────┬───────┘                └──────┬───────┘
      │                               │
      │ 1. POST /api/auth/login       │
      │    {user_id: "user123"}       │
      ├──────────────────────────────>│
      │                               │
      │ 2. {session_token: "eyJ..."}  │
      │<──────────────────────────────┤
      │                               │
      │ Store token in memory         │
      │                               │
      │ 3. GET /api/bank/accounts     │
      │    Authorization: Bearer eyJ..│
      ├──────────────────────────────>│
      │                               │
      │                               │ 4. Verify JWT
      │                               │
      │                               │ 5. Lookup access_token
      │                               │    in database by user_id
      │                               │
      │                               │ 6. Call Plaid API
      │                               │
      │ 7. {accounts: [...]}          │
      │<──────────────────────────────┤
      │                               │
```

## Token Structure

**JWT Payload:**
```json
{
  "sub": "user123",              // User ID (required)
  "plaid_item_id": "item_abc",   // Plaid item (optional)
  "exp": 1699371600              // Expiration timestamp
}
```

**Signed with:** HS256 algorithm using `JWT_SECRET_KEY`

## Error Handling

### Invalid Token
```json
{
  "detail": "Could not validate credentials",
  "status_code": 401
}
```

### Expired Token
```json
{
  "detail": "Could not validate credentials",
  "status_code": 401
}
```

### Missing User
```json
{
  "success": false,
  "error": "No linked bank accounts found for user"
}
```

## Future Enhancements

1. **Token Refresh:** Add refresh token mechanism for long-lived sessions
2. **Role-Based Access:** Add user roles/permissions in JWT claims
3. **Multi-Factor Auth:** Integrate MFA before issuing tokens
4. **Token Revocation:** Add blacklist for invalidated tokens
5. **OAuth 2.0:** Support OAuth flows for third-party integrations

## References

- [JWT.io](https://jwt.io/) - JWT specification and debugger
- [python-jose](https://github.com/mpdavis/python-jose) - JWT library
- [OWASP JWT Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
