# Plaid OAuth Testing Instructions

## 🚀 Quick Start (Both Servers Running)

### Backend Server
**URL**: http://localhost:8000
**Status**: ✅ RUNNING
**Health Check**: http://localhost:8000/api/health

### Frontend Server
**URL**: http://localhost:3000 (starting now...)
**Framework**: React + Vite
**Wait**: ~5-10 seconds for compilation

---

## 📋 Step-by-Step OAuth Testing

### Step 1: Open Dashboard
1. Open browser to: **http://localhost:3000**
2. You should see the "Gary x Taleb Risk Dashboard"
3. Look for the **"Total Net Worth"** section at the top
4. Current display should show:
   - Trader-AI: $10,000
   - Banks: $0
   - Total: $10,000

### Step 2: Click "Connect Bank Account"
1. Find the **blue button** in the top-right of "Total Net Worth" section
2. Button text: "Connect Bank Account" or "Link Bank"
3. Click it!
4. **Plaid Link modal** should open

### Step 3: Complete Plaid OAuth (Sandbox)
1. **Search for bank**: Type "Wells Fargo" (or any bank)
2. **Select**: Click "Wells Fargo"
3. **Enter credentials** (Plaid Sandbox test account):
   ```
   Username: user_good
   Password: pass_good
   ```
4. **Select accounts**: Check all accounts you want to link
5. **Click Continue** through any additional screens
6. **Wait for success** message

### Step 4: Verify Integration
After OAuth completes, you should see:

1. **Network Requests** (Check DevTools Network tab):
   - `POST /api/plaid/create_link_token` → 200 OK
   - `POST /api/plaid/exchange_public_token` → 200 OK
   - `GET /api/bank/accounts` → 200 OK (with JWT)
   - `GET /api/networth` → Updated total

2. **UI Updates**:
   - "Connected Bank Accounts" section shows account cards
   - Each card displays:
     - Bank name (Wells Fargo)
     - Account type (Checking/Savings)
     - Balance
     - Last 4 digits
   - "Total Net Worth" updates:
     - Trader-AI: $10,000
     - Banks: $XXX (real balance from Plaid)
     - Total: $10,000 + $XXX

3. **Transactions Table**:
   - Recent transactions appear
   - Shows: Date, Merchant, Amount, Category
   - "Export CSV" button works

---

## 🐛 Troubleshooting

### Issue: Plaid Link doesn't open
**Fix**:
- Check browser console for errors
- Verify backend is running: `curl http://localhost:8000/api/health`
- Check link token generation: `curl -X POST http://localhost:8000/api/plaid/create_link_token -d '{"user_id":"test-user-001"}'`

### Issue: "Missing user authentication" after OAuth
**Fix**:
- Open DevTools → Application → Local Storage
- Check for JWT token stored
- If missing, check `/api/plaid/exchange_public_token` response
- Should return `{"access_token": "...", "token_type": "bearer"}`

### Issue: No accounts showing
**Fix**:
- Check Network tab for `/api/bank/accounts` request
- Should include header: `Authorization: Bearer <JWT>`
- Response should have `success: true` and array of accounts
- If `success: false, error: "No linked accounts"`:
  - Verify `exchange_public_token` was called
  - Check database: `C:\Users\17175\Desktop\trader-ai\data\bank_accounts.db`

### Issue: Frontend won't start
**Fix**:
```bash
cd C:\Users\17175\Desktop\trader-ai\src\dashboard\frontend
npm install  # Re-install dependencies
npm run dev
```

---

## 🧪 Manual API Testing (Alternative)

If frontend has issues, test backend directly:

### 1. Get Link Token
```bash
curl -X POST http://localhost:8000/api/plaid/create_link_token \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"test-user-001\"}"
```

### 2. Use Plaid Quickstart (Alternative OAuth Test)
```bash
git clone https://github.com/plaid/quickstart.git
cd quickstart
# Configure to point to our backend
# Follow Plaid's README
```

### 3. Test Endpoints with JWT
```bash
# Generate JWT
TOKEN=$(python -c "
import sys
sys.path.insert(0, 'C:/Users/17175/Desktop/trader-ai')
from dotenv import load_dotenv
load_dotenv('C:/Users/17175/Desktop/trader-ai/.env')
from src.security.auth import create_access_token
print(create_access_token({'sub': 'test-user-001'}))
")

# Test bank accounts
curl http://localhost:8000/api/bank/accounts \
  -H "Authorization: Bearer $TOKEN"
```

---

## ✅ Success Criteria

- [ ] Plaid Link modal opens on button click
- [ ] OAuth completes without errors
- [ ] Bank accounts appear in UI
- [ ] Net worth updates with bank balance
- [ ] Transactions table populates
- [ ] JWT authentication works (no 401 errors)
- [ ] Export CSV functionality works

---

## 📊 Expected Test Data (Plaid Sandbox)

When using `user_good` credentials, you should see:

**Wells Fargo Checking**:
- Balance: ~$100-$1,000 (varies)
- Account #: **** 1111
- Type: Depository - Checking

**Wells Fargo Savings**:
- Balance: ~$200-$2,000
- Account #: **** 2222
- Type: Depository - Savings

**Recent Transactions** (10-20):
- Starbucks: -$4.50
- Payroll Deposit: +$2,500
- Electric Bill: -$85.00
- etc.

---

## 🎉 When Complete

Take screenshots of:
1. Unified Net Worth card showing combined total
2. Connected Bank Accounts grid
3. Transactions table
4. Browser DevTools Network tab (successful API calls)

Then proceed to Phase 4: Final deployment checklist!

---

**Questions?** Check:
- Backend logs: `BashOutput` tool for server output
- Frontend console: Browser DevTools → Console
- Network activity: Browser DevTools → Network
- Documentation: `docs/PHASE2-COMPLETE-FINAL-REPORT.md`
