# Frontend JWT Authentication Upgrade

**Date**: 2025-11-07
**Location**: `C:\Users\17175\Desktop\trader-ai\src\dashboard\frontend`
**Status**: COMPLETED

## Overview

Updated the frontend to use secure Authorization headers with JWT tokens instead of passing access tokens in query parameters for Plaid bank integration.

## Changes Made

### 1. Created `src/services/authService.ts`

**Purpose**: Centralized JWT token management with localStorage persistence.

**Key Functions**:
- `setAuthToken(token)` - Store JWT token in memory and localStorage
- `getAuthToken()` - Retrieve JWT token from memory or localStorage
- `clearAuthToken()` - Remove JWT token from storage
- `exchangePublicToken(publicToken)` - Exchange Plaid public token for JWT
- `isAuthenticated()` - Check if user has valid token

**Security Features**:
- In-memory token storage for session
- localStorage fallback for persistence
- Automatic token cleanup on logout

### 2. Updated `src/services/api.ts`

**Purpose**: Enhanced DashboardAPI client with automatic JWT injection.

**Key Changes**:
- Added JWT token to all requests via Authorization header
- Request interceptor automatically retrieves token from authService
- Response interceptor handles 401 errors (clears token, redirects to login)
- Exposed `axiosClient` getter for direct axios access when needed

**Request Flow**:
```javascript
Request → Get JWT from authService → Add Authorization: Bearer <token> → Send to backend
Response → Check status → Handle 401 (clear token + redirect) → Return data
```

### 3. Updated `src/store/dashboardSlice.ts`

**Purpose**: Refactored Redux thunks to use authenticated API client.

**Key Changes**:
- Replaced `axios` imports with `dashboardAPI` client
- All async thunks now use `dashboardAPI.axiosClient.get/post()`
- `linkPlaidAccount` thunk now calls `exchangePublicToken()` from authService
- Removed manual access_token handling from all API calls

**Updated Thunks**:
- `fetchBankAccounts` - Uses JWT auth via api interceptor
- `fetchTransactions` - Uses JWT auth via api interceptor
- `linkPlaidAccount` - Exchanges public token for JWT, stores in authService
- `refreshBankData` - Uses JWT auth via api interceptor
- `fetchUnifiedNetWorth` - Uses JWT auth via api interceptor

### 4. Updated `src/components/PlaidLinkButton.tsx`

**Purpose**: Integrated JWT token exchange into Plaid Link flow.

**Key Changes**:
- Replaced `axios` import with `dashboardAPI` client
- Added `exchangePublicToken` from authService
- `handleExchangePublicToken()` now calls authService to store JWT
- `generateLinkToken()` uses `dashboardAPI.axiosClient.post()`
- Removed manual access_token handling

**Plaid Flow**:
```
1. User clicks "Connect Bank Account"
2. Generate link token via backend
3. Open Plaid Link modal
4. User authenticates with bank
5. Plaid returns public token
6. Exchange public token for JWT via authService
7. JWT stored in memory + localStorage
8. All subsequent API calls use JWT from Authorization header
```

## Security Improvements

### Before (Insecure)
```javascript
// Access token in query params (visible in logs, browser history)
axios.get(`/api/bank/accounts?access_token=${token}`);
```

### After (Secure)
```javascript
// JWT in Authorization header (not logged, not cached)
// Token automatically added by api interceptor
dashboardAPI.axiosClient.get('/api/bank/accounts');
// Header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## API Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ PlaidLink    │────────▶│ authService  │                     │
│  │ Button       │         │ - JWT storage│                     │
│  └──────────────┘         │ - Token mgmt │                     │
│         │                 └──────┬───────┘                     │
│         │                        │                              │
│         ▼                        ▼                              │
│  ┌──────────────────────────────────────┐                     │
│  │       dashboardAPI Client            │                     │
│  │  ┌────────────────────────────────┐  │                     │
│  │  │  Request Interceptor:          │  │                     │
│  │  │  - Get JWT from authService    │  │                     │
│  │  │  - Add Authorization header    │  │                     │
│  │  └────────────────────────────────┘  │                     │
│  │                                       │                     │
│  │  ┌────────────────────────────────┐  │                     │
│  │  │  Response Interceptor:         │  │                     │
│  │  │  - Handle 401 (clear + redir)  │  │                     │
│  │  │  - Error handling              │  │                     │
│  │  └────────────────────────────────┘  │                     │
│  └──────────────┬───────────────────────┘                     │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────┐                     │
│  │   Redux dashboardSlice Thunks        │                     │
│  │  - fetchBankAccounts()               │                     │
│  │  - fetchTransactions()               │                     │
│  │  - linkPlaidAccount()                │                     │
│  │  - refreshBankData()                 │                     │
│  │  - fetchUnifiedNetWorth()            │                     │
│  └──────────────┬───────────────────────┘                     │
│                 │                                               │
└─────────────────┼───────────────────────────────────────────────┘
                  │
                  ▼ Authorization: Bearer <JWT>
         ┌─────────────────────────┐
         │   Backend API Server    │
         │  - Validate JWT         │
         │  - Process request      │
         │  - Return data          │
         └─────────────────────────┘
```

## Testing Recommendations

### Manual Testing
1. **Link Bank Account**:
   - Click "Connect Bank Account" button
   - Complete Plaid Link flow
   - Verify JWT token stored in localStorage
   - Check browser network tab: No access_token in URL params

2. **Fetch Bank Data**:
   - After linking, fetch bank accounts
   - Verify Authorization header present in network tab
   - Check header value: `Authorization: Bearer eyJhbGci...`

3. **Token Expiry**:
   - Wait for JWT to expire (or manually clear)
   - Make API request
   - Verify 401 response redirects to login

### Automated Testing
Update `src/components/__tests__/PlaidLinkButton.test.tsx`:
- Mock `dashboardAPI.axiosClient` instead of `axios`
- Mock `exchangePublicToken` from authService
- Verify JWT token stored after public token exchange
- Verify Authorization header added to requests

## Backend Requirements

**IMPORTANT**: Backend must be updated to:

1. **Accept Authorization headers**:
```python
from fastapi import Header, HTTPException

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    # Validate JWT token
    payload = verify_jwt_token(token)
    return payload
```

2. **Return JWT on token exchange**:
```python
@router.post("/api/plaid/exchange_public_token")
async def exchange_token(request: ExchangeTokenRequest):
    # Exchange Plaid public token for access token
    access_token = exchange_plaid_token(request.public_token)

    # Generate JWT for this user
    jwt_token = create_jwt_token(user_id=request.user_id, access_token=access_token)

    return {
        "jwt_token": jwt_token,
        "message": "Bank account linked successfully"
    }
```

3. **Protect all Plaid endpoints**:
```python
@router.get("/api/bank/accounts")
async def get_accounts(current_user = Depends(get_current_user)):
    # Use current_user.access_token to fetch from Plaid
    accounts = fetch_plaid_accounts(current_user.access_token)
    return {"accounts": accounts}
```

## Migration Checklist

- [x] Create `authService.ts` for token management
- [x] Update `api.ts` with JWT interceptors
- [x] Refactor `dashboardSlice.ts` to use authenticated client
- [x] Update `PlaidLinkButton.tsx` for JWT exchange
- [x] Remove all `access_token` query parameters
- [x] Verify Authorization headers in all Plaid API calls
- [ ] Update backend to accept Authorization headers
- [ ] Update backend to return JWT on token exchange
- [ ] Add JWT validation middleware to backend
- [ ] Update unit tests for new authentication flow
- [ ] Test end-to-end flow with real Plaid integration

## Files Modified

1. `src/services/authService.ts` - **NEW FILE** (JWT token management)
2. `src/services/api.ts` - Enhanced with JWT interceptors
3. `src/store/dashboardSlice.ts` - Refactored to use authenticated API client
4. `src/components/PlaidLinkButton.tsx` - Integrated JWT token exchange

## Files Not Modified (No Changes Needed)

The following files contain `axios` imports but are not Plaid-related:
- `src/components/BankAccountCard.tsx` - Generic bank UI component
- `src/components/AIStrategyPanel.tsx` - AI strategy predictions
- `src/components/TradingControls.tsx` - Trading order execution
- `src/components/__tests__/PlaidLinkButton.test.tsx` - Unit tests (needs update)
- `src/docs/PlaidReduxIntegration.md` - Documentation
- `src/docs/PlaidReduxIntegrationSummary.md` - Documentation

These files will continue using `axios` for non-Plaid API calls. If they need authentication in the future, they should be updated to use `dashboardAPI.axiosClient` for automatic JWT injection.

## Next Steps

1. **Backend Implementation**: Update backend to support JWT authentication
2. **Testing**: Update unit tests and add integration tests
3. **Documentation**: Update API documentation with new authentication flow
4. **Deployment**: Deploy frontend and backend changes together
5. **Monitoring**: Monitor for authentication errors in production

## Security Notes

- JWT tokens stored in localStorage (consider httpOnly cookies for production)
- Token automatically cleared on 401 responses
- All Plaid API calls now use Authorization header
- No access tokens in URL params (prevents logging/caching vulnerabilities)

## References

- Plaid Documentation: https://plaid.com/docs/
- JWT Best Practices: https://jwt.io/introduction
- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
