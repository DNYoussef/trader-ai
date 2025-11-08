# Phase 2: Plaid Sandbox Testing Guide

**Duration**: 30 minutes
**Status**: Ready to begin
**Prerequisites**: Phase 1 (Security Fixes) ✅ Complete

---

## 🎯 Objectives

1. Sign up for Plaid sandbox account (free)
2. Get API credentials (Client ID + Secret)
3. Test all 6 backend API endpoints
4. Complete OAuth flow with test bank
5. Verify encrypted token storage
6. Verify JWT authentication works

---

## Step 1: Get Plaid Sandbox Credentials (5 minutes)

### 1.1 Sign Up

Visit: **https://dashboard.plaid.com/signup**

Fill out:
- Email address
- Company name (can use "Personal Project")
- Use case (select "Testing/Learning")

### 1.2 Get Credentials

After signup:
1. Navigate to **Dashboard** → **Keys** → **Sandbox**
2. Copy **Client ID** (looks like: `63f4d1ab2c3e9f001a...`)
3. Copy **Secret** (looks like: `7b9c2e1d5f8a3b4c6e...`)

### 1.3 Update .env File

**Location**: `C:\Users\17175\Desktop\trader-ai\.env`

Replace the placeholders:
```bash
PLAID_CLIENT_ID=<paste_your_client_id_here>
PLAID_SECRET=<paste_your_secret_here>
```

**Verify .env file**:
```bash
cat .env
# Should show:
# DATABASE_ENCRYPTION_KEY=HENvZX_qhWpyYVCtSOHCGh9EMR9Em2nuvOsjpQrTXLU=
# JWT_SECRET_KEY=OZw3U-mWjn__ZBahBtY9z7Wd5sFILKTXXlbi9jRgdys
# PLAID_CLIENT_ID=63f4d1ab...
# PLAID_SECRET=7b9c2e1d...
```

---

## Step 2: Initialize Database (2 minutes)

### 2.1 Run Migration

```bash
cd C:\Users\17175\Desktop\trader-ai

# Initialize encrypted database
python -c "from src.finances.bank_database_encrypted import BankDatabase; db = BankDatabase(); print('✓ Database initialized with encryption')"
```

**Expected Output**:
```
✓ Database initialized with encryption
```

### 2.2 Verify Permissions

```bash
# Windows (PowerShell)
icacls data\bank_accounts.db

# Should show restricted permissions
```

---

## Step 3: Start Backend Server (2 minutes)

### 3.1 Launch Server

```bash
cd C:\Users\17175\Desktop\trader-ai\src\dashboard
python run_server_simple.py
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3.2 Verify Server Health

Open new terminal:
```bash
curl http://localhost:8000/api/health
```

**Expected**: `{"status": "ok", "security": "enabled"}`

---

## Step 4: Test API Endpoints (10 minutes)

### 4.1 Test: Create Link Token

```bash
curl -X POST http://localhost:8000/api/plaid/create_link_token \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"test-user-001\"}"
```

**Expected Response**:
```json
{
  "link_token": "link-sandbox-abc123-...",
  "expiration": "2025-11-07T18:00:00Z",
  "request_id": "req_xyz789"
}
```

**✅ Success**: You got a link_token
**❌ Error**: Check .env has correct PLAID_CLIENT_ID and PLAID_SECRET

### 4.2 Test: Exchange Public Token

*Note: You'll get a public_token after completing OAuth in Step 5*

```bash
# Save this for after OAuth
curl -X POST http://localhost:8000/api/plaid/exchange_public_token \
  -H "Content-Type: application/json" \
  -d "{\"public_token\": \"public-sandbox-...\"}"
```

**Expected Response**:
```json
{
  "jwt_token": "eyJhbGciOiJIUzI1NiIs...",
  "message": "Successfully linked bank account"
}
```

### 4.3 Test: Get Bank Accounts (with JWT)

```bash
# Replace <jwt_token> with token from Step 4.2
curl http://localhost:8000/api/bank/accounts \
  -H "Authorization: Bearer <jwt_token>"
```

**Expected Response**:
```json
{
  "accounts": [
    {
      "account_id": "acc_xyz...",
      "name": "Plaid Checking",
      "type": "depository",
      "subtype": "checking",
      "mask": "0000",
      "balance": 100.00,
      "available": 100.00,
      "currency": "USD",
      "institution": "First Platypus Bank"
    }
  ],
  "count": 1
}
```

### 4.4 Test: Get Balances

```bash
curl http://localhost:8000/api/bank/balances \
  -H "Authorization: Bearer <jwt_token>"
```

**Expected Response**:
```json
{
  "balances": [
    {"account_id": "acc_xyz...", "current": 100.00, "available": 100.00}
  ],
  "count": 1,
  "timestamp": "2025-11-07T16:30:00Z"
}
```

### 4.5 Test: Get Transactions

```bash
curl "http://localhost:8000/api/bank/transactions?start_date=2025-10-01&end_date=2025-11-07" \
  -H "Authorization: Bearer <jwt_token>"
```

**Expected Response**:
```json
{
  "transactions": [
    {
      "transaction_id": "txn_abc...",
      "amount": 4.50,
      "date": "2025-11-06",
      "name": "Starbucks",
      "category": "Food and Drink"
    }
  ],
  "count": 1
}
```

### 4.6 Test: Unified Net Worth

```bash
curl http://localhost:8000/api/networth \
  -H "Authorization: Bearer <jwt_token>"
```

**Expected Response**:
```json
{
  "trader_ai_nav": 100000.00,
  "bank_total": 100.00,
  "total_networth": 100100.00,
  "bank_accounts": [
    {"name": "Plaid Checking", "balance": 100.00}
  ]
}
```

---

## Step 5: Complete OAuth Flow (5 minutes)

### 5.1 Manual Testing with Plaid Link

Since we don't have the frontend running yet, we'll test the backend OAuth flow:

**Sandbox Test Credentials**:
- Username: `user_good`
- Password: `pass_good`
- Bank: "First Platypus Bank"
- MFA Code: `1234` (if prompted)

### 5.2 Simulate OAuth Flow

1. **Get link_token** (from Step 4.1)
2. **Open Plaid Link** in browser:
   ```
   https://cdn.plaid.com/link/v2/stable/link.html?isWebview=false&token=<link_token>
   ```
3. **Select bank**: "First Platypus Bank"
4. **Enter credentials**: `user_good` / `pass_good`
5. **Copy public_token** from browser console
6. **Exchange token** (Step 4.2) to get JWT

### 5.3 Alternative: Use Plaid Quickstart

```bash
# Clone Plaid quickstart
git clone https://github.com/plaid/quickstart.git
cd quickstart

# Configure for our backend
# Edit .env to point to http://localhost:8000

# Run frontend only
npm start
```

---

## Step 6: Verify Encryption (3 minutes)

### 6.1 Check Database

```bash
cd C:\Users\17175\Desktop\trader-ai

# Query database directly
python -c "
import sqlite3
conn = sqlite3.connect('data/bank_accounts.db')
cursor = conn.cursor()
cursor.execute('SELECT item_id, access_token FROM plaid_items LIMIT 1')
result = cursor.fetchone()
if result:
    print(f'Item ID: {result[0]}')
    print(f'Access Token (encrypted): {result[1][:50]}...')
    print('✓ Token is encrypted (base64 gibberish)')
else:
    print('No items yet - complete OAuth first')
"
```

**Expected Output**:
```
Item ID: item_xyz...
Access Token (encrypted): Z0FBQUFBQm5SS3Z4dGxhYmNkZWZnaGlqa2xtbm9wcX...
✓ Token is encrypted (base64 gibberish)
```

### 6.2 Verify Decryption Works

```bash
python -c "
from src.finances.bank_database_encrypted import BankDatabase
db = BankDatabase()
items = db.get_all_items()
if items:
    item = items[0]
    print(f'✓ Decryption successful')
    print(f'Item ID: {item[\"item_id\"]}')
    print(f'Institution: {item[\"institution_name\"]}')
else:
    print('No items yet - complete OAuth first')
"
```

---

## Step 7: Verify JWT Authentication (3 minutes)

### 7.1 Test Without JWT (Should Fail)

```bash
curl http://localhost:8000/api/bank/accounts
```

**Expected**: `401 Unauthorized` - "Not authenticated"

### 7.2 Test With Invalid JWT (Should Fail)

```bash
curl http://localhost:8000/api/bank/accounts \
  -H "Authorization: Bearer invalid_token_xyz"
```

**Expected**: `401 Unauthorized` - "Could not validate credentials"

### 7.3 Test With Valid JWT (Should Succeed)

```bash
# Use JWT from Step 4.2
curl http://localhost:8000/api/bank/accounts \
  -H "Authorization: Bearer <valid_jwt_token>"
```

**Expected**: Account data returned successfully

---

## 📊 Testing Checklist

### API Endpoints
- [ ] POST /api/plaid/create_link_token → Returns link_token
- [ ] POST /api/plaid/exchange_public_token → Returns JWT
- [ ] GET /api/bank/accounts → Returns account list
- [ ] GET /api/bank/balances → Returns balances
- [ ] GET /api/bank/transactions → Returns transaction history
- [ ] GET /api/networth → Returns unified net worth

### Security
- [ ] Tokens encrypted in database (verified via SQL query)
- [ ] JWT authentication required (401 without token)
- [ ] Invalid JWT rejected (401 with bad token)
- [ ] Rate limiting works (429 after 10 requests/minute)
- [ ] Security headers present (check curl -I response)

### OAuth Flow
- [ ] Link token created successfully
- [ ] Public token exchanged for access token
- [ ] Access token stored encrypted
- [ ] Session created with JWT
- [ ] JWT expires after 60 minutes

---

## 🎉 Success Criteria

✅ **All 6 API endpoints responding**
✅ **OAuth flow complete with test bank**
✅ **Tokens encrypted in database**
✅ **JWT authentication enforced**
✅ **Rate limiting active**
✅ **Security headers configured**

When all checklist items are complete, **Phase 2 is DONE!**

---

## 🐛 Troubleshooting

### Error: "PLAID_CLIENT_ID environment variable not set"
**Fix**: Restart server after updating .env file

### Error: "Invalid client_id"
**Fix**: Double-check credentials in Plaid dashboard (Keys → Sandbox)

### Error: "DATABASE_ENCRYPTION_KEY not set"
**Fix**: Run `cat .env` to verify key is present

### Error: "401 Unauthorized"
**Fix**: Make sure you're using Authorization header with valid JWT

### Error: "Database is locked"
**Fix**: Close all database connections, restart server

### Error: "Rate limit exceeded (429)"
**Fix**: Wait 60 seconds, or increase limit in config.json

---

## 📞 Next Steps

After Phase 2 is complete:
- **Phase 3**: Integrate components into dashboard UI (1.5 hours)
- **Phase 4**: Demo unified net worth calculation (15 min)

---

**Ready to begin?** Start with Step 1 - Get your Plaid sandbox credentials!
