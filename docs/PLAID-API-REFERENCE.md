# Plaid API Quick Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Create Link Token
**POST** `/api/plaid/create_link_token`

Initialize Plaid Link for frontend.

**Request Body (Optional):**
```json
{
  "user_id": "trader-ai-user"
}
```

**Response:**
```json
{
  "success": true,
  "link_token": "link-sandbox-abc123",
  "expiration": "2025-11-07T20:00:00Z",
  "request_id": "req-xyz"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/plaid/create_link_token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user"}'
```

---

### 2. Exchange Public Token
**POST** `/api/plaid/exchange_public_token`

Convert public token to persistent access token.

**Request Body:**
```json
{
  "public_token": "public-sandbox-abc123"
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "access-sandbox-xyz789",
  "message": "Bank account connected successfully"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/plaid/exchange_public_token \
  -H "Content-Type: application/json" \
  -d '{"public_token": "public-sandbox-abc123"}'
```

---

### 3. Get Bank Accounts
**GET** `/api/bank/accounts?access_token={token}`

Fetch all linked bank accounts.

**Query Parameters:**
- `access_token` (required): Access token from exchange

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

**cURL Example:**
```bash
curl "http://localhost:8000/api/bank/accounts?access_token=access-sandbox-xyz789"
```

---

### 4. Get Bank Balances
**GET** `/api/bank/balances?access_token={token}`

Real-time balance updates.

**Query Parameters:**
- `access_token` (required): Access token from exchange

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

**cURL Example:**
```bash
curl "http://localhost:8000/api/bank/balances?access_token=access-sandbox-xyz789"
```

---

### 5. Get Bank Transactions
**GET** `/api/bank/transactions?access_token={token}&start_date={date}&end_date={date}`

Fetch transaction history.

**Query Parameters:**
- `access_token` (required): Access token from exchange
- `start_date` (optional): YYYY-MM-DD format (default: 30 days ago)
- `end_date` (optional): YYYY-MM-DD format (default: today)
- `count` (optional): Max transactions (default: 100)

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

**cURL Example:**
```bash
curl "http://localhost:8000/api/bank/transactions?access_token=access-sandbox-xyz789&start_date=2025-10-01&end_date=2025-11-07"
```

---

### 6. Get Unified Net Worth
**GET** `/api/networth?access_token={token}`

Combined trader-ai portfolio + bank balances.

**Query Parameters:**
- `access_token` (optional): If provided, includes bank balances

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

**cURL Example:**
```bash
# With bank balances
curl "http://localhost:8000/api/networth?access_token=access-sandbox-xyz789"

# Trader-AI portfolio only
curl "http://localhost:8000/api/networth"
```

---

## Error Responses

All endpoints return standardized error format:

```json
{
  "success": false,
  "error": "Detailed error message",
  "error_type": "plaid_accounts_error"
}
```

**Error Types:**
- `plaid_link_token_error`
- `plaid_token_exchange_error`
- `plaid_accounts_error`
- `plaid_balances_error`
- `plaid_transactions_error`
- `networth_calculation_error`

---

## Sandbox Testing

**Plaid Sandbox Credentials:**
- Username: `user_good`
- Password: `pass_good`
- MFA: `1234` (if prompted)

**Test Workflow:**
1. Create link token → Get `link_token`
2. Open Plaid Link with `link_token` (frontend)
3. User selects bank and logs in (sandbox credentials)
4. Plaid Link returns `public_token`
5. Exchange `public_token` → Get `access_token`
6. Use `access_token` for all subsequent API calls

---

## Environment Setup

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

**Get Credentials:**
1. Sign up: https://dashboard.plaid.com/signup
2. Navigate to Keys → Sandbox
3. Copy Client ID and Secret

---

## Rate Limits

**Sandbox:** Unlimited
**Development:** 100 req/min per IP
**Production:** 200 req/min per IP

---

## Security Notes

⚠️ **Current Implementation:** Development only
- Access tokens passed as query parameters
- Tokens returned in responses
- No persistent storage

✅ **Production Requirements:**
- Store access tokens encrypted
- Use POST body for tokens (not query params)
- Implement user authentication
- HTTPS only
- Token rotation and refresh

---

## Support

**Plaid Docs:** https://plaid.com/docs/
**API Reference:** https://plaid.com/docs/api/
**Dashboard:** https://dashboard.plaid.com/
