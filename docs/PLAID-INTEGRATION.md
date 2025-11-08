# Plaid Banking Integration - Backend Complete

## Overview

Complete backend Plaid API integration for trader-ai with 6 FastAPI endpoints enabling bank account connections, balance tracking, and unified net worth calculations.

## Components Implemented

### 1. Plaid Client (`src/finances/plaid_client.py`)

**PlaidClient Class Features:**
- Multi-environment support (sandbox/development/production)
- Comprehensive error handling for rate limits and network errors
- Dataclass structures for accounts and transactions
- Automatic credential loading from environment variables

**Core Methods:**
- `create_link_token(user_id)` → Returns link_token for frontend Plaid Link
- `exchange_public_token(public_token)` → Stores access_token
- `get_accounts(access_token)` → Fetches all bank accounts
- `get_balances(access_token)` → Real-time balance updates
- `get_transactions(access_token, start_date, end_date)` → Transaction history

**Error Handling:**
- `ITEM_LOGIN_REQUIRED` → User re-authentication needed
- `RATE_LIMIT_EXCEEDED` → Automatic retry guidance
- `INVALID_REQUEST` → Detailed validation errors
- Network errors with automatic fallback

### 2. FastAPI Endpoints (added to `src/dashboard/run_server_simple.py`)

#### `POST /api/plaid/create_link_token`
Creates Plaid Link token for frontend initialization.

**Request:**
```json
{
  "user_id": "trader-ai-user"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "link_token": "link-sandbox-xxx",
  "expiration": "2025-11-07T20:00:00Z",
  "request_id": "req-xxx"
}
```

#### `POST /api/plaid/exchange_public_token`
Exchanges public token from Plaid Link for persistent access token.

**Request:**
```json
{
  "public_token": "public-sandbox-xxx"
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "access-sandbox-xxx",
  "message": "Bank account connected successfully"
}
```

#### `GET /api/bank/accounts?access_token=xxx`
Fetches all linked bank accounts with details.

**Response:**
```json
{
  "success": true,
  "accounts": [
    {
      "account_id": "acc-123",
      "name": "Plaid Checking",
      "official_name": "Plaid Gold Standard 0% Interest Checking",
      "type": "depository",
      "subtype": "checking",
      "mask": "0000",
      "current_balance": 1250.50,
      "available_balance": 1200.00,
      "currency": "USD"
    }
  ],
  "count": 1
}
```

#### `GET /api/bank/balances?access_token=xxx`
Real-time balance updates for all accounts.

**Response:**
```json
{
  "success": true,
  "balances": [
    {
      "account_id": "acc-123",
      "name": "Plaid Checking",
      "current": 1250.50,
      "available": 1200.00,
      "currency": "USD",
      "type": "depository",
      "subtype": "checking"
    }
  ],
  "count": 1,
  "timestamp": "2025-11-07T16:30:00Z"
}
```

#### `GET /api/bank/transactions?access_token=xxx&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
Fetches transaction history for date range (defaults to last 30 days).

**Response:**
```json
{
  "success": true,
  "transactions": [
    {
      "transaction_id": "txn-123",
      "account_id": "acc-123",
      "amount": -45.67,
      "date": "2025-11-06",
      "name": "Amazon.com",
      "merchant_name": "Amazon",
      "category": ["Shops", "Digital Purchase"],
      "pending": false
    }
  ],
  "count": 1,
  "start_date": "2025-10-08",
  "end_date": "2025-11-07"
}
```

#### `GET /api/networth?access_token=xxx`
Unified net worth calculation combining trader-ai NAV and bank balances.

**Response:**
```json
{
  "success": true,
  "trader_ai_nav": 10250.75,
  "bank_total": 1250.50,
  "total_networth": 11501.25,
  "bank_accounts_count": 1,
  "bank_accounts": [...],
  "timestamp": "2025-11-07T16:30:00Z",
  "currency": "USD"
}
```

### 3. Configuration (`config/config.json`)

Added Plaid credentials with environment variable support:

```json
{
  "plaid_client_id": "env:PLAID_CLIENT_ID",
  "plaid_secret": "env:PLAID_SECRET",
  "plaid_env": "sandbox"
}
```

### 4. Dependencies (`requirements.txt`)

Added `plaid-python>=14.0.0,<15.0.0` for official Plaid SDK.

## Setup Instructions

### 1. Install Dependencies
```bash
cd C:\Users\17175\Desktop\trader-ai
pip install -r requirements.txt
```

### 2. Get Plaid Credentials

**Sandbox (Free Testing):**
1. Sign up at https://dashboard.plaid.com/signup
2. Get sandbox credentials from dashboard
3. No account linking required - use sandbox data

**Development/Production:**
1. Complete Plaid onboarding
2. Request production access
3. Configure webhook URLs

### 3. Set Environment Variables

**Windows (PowerShell):**
```powershell
$env:PLAID_CLIENT_ID = "your_client_id"
$env:PLAID_SECRET = "your_secret"
```

**Linux/Mac:**
```bash
export PLAID_CLIENT_ID="your_client_id"
export PLAID_SECRET="your_secret"
```

**Permanent (add to .env file):**
```bash
PLAID_CLIENT_ID=your_client_id
PLAID_SECRET=your_secret
```

### 4. Start Backend Server
```bash
cd src/dashboard
python run_server_simple.py
```

Server runs on `http://localhost:8000`

## Testing Workflow

### 1. Create Link Token
```bash
curl -X POST http://localhost:8000/api/plaid/create_link_token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user"}'
```

### 2. Initialize Plaid Link (Frontend)
Use returned `link_token` to initialize Plaid Link UI in frontend.

### 3. Exchange Public Token
After user completes Link flow:
```bash
curl -X POST http://localhost:8000/api/plaid/exchange_public_token \
  -H "Content-Type: application/json" \
  -d '{"public_token": "public-sandbox-xxx"}'
```

Store returned `access_token` securely.

### 4. Fetch Bank Data
```bash
# Get accounts
curl "http://localhost:8000/api/bank/accounts?access_token=access-sandbox-xxx"

# Get balances
curl "http://localhost:8000/api/bank/balances?access_token=access-sandbox-xxx"

# Get transactions
curl "http://localhost:8000/api/bank/transactions?access_token=access-sandbox-xxx&start_date=2025-10-01&end_date=2025-11-07"

# Get unified net worth
curl "http://localhost:8000/api/networth?access_token=access-sandbox-xxx"
```

## Sandbox Testing Data

Plaid sandbox provides test credentials:
- **Username:** `user_good`
- **Password:** `pass_good`
- **MFA:** `1234` (if prompted)

Sandbox institutions include:
- First Platypus Bank
- Tattersall Federal Credit Union
- Tartan Bank

## Security Considerations

### Current Implementation
- Access tokens passed as query parameters
- Tokens returned in API responses
- No persistent storage

### Production Requirements (TODO)
1. **Secure Token Storage:**
   - Encrypt access tokens at rest
   - Store in secure database (SQLite with encryption or PostgreSQL)
   - Never return access tokens in responses

2. **Authentication:**
   - Implement user authentication (JWT/OAuth)
   - Associate access tokens with user accounts
   - Validate user identity before Plaid operations

3. **HTTPS Only:**
   - Enforce TLS for all API endpoints
   - Use secure cookies for session management

4. **Token Rotation:**
   - Implement token refresh mechanism
   - Handle `ITEM_LOGIN_REQUIRED` gracefully
   - Prompt users to re-authenticate when needed

5. **Audit Logging:**
   - Log all Plaid API calls with user_id
   - Track token creation/exchange events
   - Monitor for suspicious activity

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "error": "Detailed error message",
  "error_type": "plaid_accounts_error"
}
```

**Error Types:**
- `plaid_link_token_error` - Failed to create link token
- `plaid_token_exchange_error` - Failed to exchange public token
- `plaid_accounts_error` - Failed to fetch accounts
- `plaid_balances_error` - Failed to fetch balances
- `plaid_transactions_error` - Failed to fetch transactions
- `networth_calculation_error` - Failed to calculate net worth

## API Rate Limits

**Sandbox:**
- No rate limits
- Unlimited requests

**Development:**
- 100 requests/minute per IP
- Burst: 500 requests

**Production:**
- 200 requests/minute per IP
- Burst: 1000 requests

## Next Steps

### Frontend Integration (TODO)
1. **Install Plaid Link:**
   ```bash
   npm install react-plaid-link
   ```

2. **Implement Link Component:**
   ```typescript
   import { usePlaidLink } from 'react-plaid-link';

   const { open, ready } = usePlaidLink({
     token: linkToken,
     onSuccess: (public_token, metadata) => {
       // Call /api/plaid/exchange_public_token
     }
   });
   ```

3. **Add Bank Connection UI:**
   - "Connect Bank Account" button
   - Account list display
   - Real-time balance updates
   - Transaction history table

4. **Net Worth Dashboard:**
   - Unified net worth display
   - Breakdown: Trading + Banking
   - Historical net worth chart
   - Asset allocation visualization

### Backend Enhancements (TODO)
1. **Persistent Storage:**
   - Database schema for access tokens
   - User-token association
   - Token encryption

2. **Webhooks:**
   - Handle Plaid webhook events
   - Update balances automatically
   - Notify on failed transactions

3. **Batch Operations:**
   - Bulk account updates
   - Scheduled balance refreshes
   - Transaction sync jobs

4. **Analytics:**
   - Spending categorization
   - Cash flow analysis
   - Budget tracking

## Support

**Plaid Documentation:** https://plaid.com/docs/
**API Reference:** https://plaid.com/docs/api/
**Sandbox Guide:** https://plaid.com/docs/sandbox/

**Trader-AI Issues:** Contact development team

---

**Status:** ✅ Backend Complete
**Version:** 1.0.0
**Last Updated:** 2025-11-07
