# Plaid Security Remediation Checklist

**Project**: trader-ai Banking Integration
**Date**: November 7, 2025
**Total Issues**: 12 (4 CRITICAL, 3 HIGH, 3 MEDIUM, 2 LOW)

---

## ⚡ CRITICAL PRIORITY (Fix Today)

### 1. Remove Access Token from API Response
**File**: `src/dashboard/run_server_simple.py:260-264`

- [ ] Delete line 262: `"access_token": access_token,`
- [ ] Add `item_id` to response instead
- [ ] Update frontend to store `item_id` instead of `access_token`
- [ ] Test token not exposed in browser Network tab

**Status**: ❌ **NOT STARTED**
**ETA**: 1 hour

---

### 2. Move Access Token from URL to Authorization Header
**Files**: `src/dashboard/run_server_simple.py:274-425`

- [ ] Refactor `/api/bank/accounts` endpoint (line 274)
- [ ] Refactor `/api/bank/balances` endpoint (line 325)
- [ ] Refactor `/api/bank/transactions` endpoint (line 361)
- [ ] Refactor `/api/networth` endpoint (line 427)
- [ ] Change from `access_token: str = None` to `authorization: str = Header(None)`
- [ ] Implement JWT session token system
- [ ] Update frontend to use `Authorization: Bearer <token>` headers
- [ ] Test no tokens in URL logs

**Status**: ❌ **NOT STARTED**
**ETA**: 4 hours

---

### 3. Fix Database File Permissions
**File**: `data/bank_accounts.db`

**Current**: `-rw-r--r--` (644)
**Target**: `-rw-------` (600)

```bash
chmod 600 data/bank_accounts.db
```

- [ ] Run chmod command
- [ ] Verify with `ls -la data/bank_accounts.db`
- [ ] Add to deployment scripts

**Status**: ❌ **NOT STARTED**
**ETA**: 5 minutes

---

### 4. Encrypt Access Tokens in Database
**File**: `src/finances/bank_database.py`

- [ ] Install cryptography: `pip install cryptography`
- [ ] Add encryption key to `.env`: `DATABASE_ENCRYPTION_KEY=<key>`
- [ ] Modify `BankDatabase.__init__()` to accept encryption key
- [ ] Modify `add_plaid_item()` to encrypt tokens before INSERT
- [ ] Add `get_decrypted_access_token()` method
- [ ] Update all callers to use new API
- [ ] Migrate existing tokens
- [ ] Test encryption/decryption

**Status**: ❌ **NOT STARTED**
**ETA**: 3 hours

---

## 🟠 HIGH PRIORITY (Fix This Week)

### 5. Add Rate Limiting
**Files**: `src/dashboard/run_server_simple.py` (all Plaid endpoints)

- [ ] Install slowapi: `pip install slowapi`
- [ ] Import and configure limiter
- [ ] Add `@limiter.limit("5/minute")` to `/api/plaid/create_link_token`
- [ ] Add `@limiter.limit("20/minute")` to all `/api/bank/*` endpoints
- [ ] Test rate limiting with rapid requests
- [ ] Verify 429 status code returned

**Status**: ❌ **NOT STARTED**
**ETA**: 2 hours

---

### 6. Sanitize Error Messages
**Files**: `src/finances/plaid_client.py:139-142, 171-174`

- [ ] Replace detailed error messages with generic ones
- [ ] Keep detailed logging server-side only
- [ ] Use `HTTPException` with safe messages
- [ ] Test error paths don't leak info

**Status**: ❌ **NOT STARTED**
**ETA**: 1 hour

---

### 7. Enforce HTTPS in Production
**File**: `src/dashboard/run_server_simple.py:119-127`

- [ ] Add environment detection
- [ ] Add `HTTPSRedirectMiddleware` for production
- [ ] Restrict CORS origins to HTTPS in production
- [ ] Update deployment docs

**Status**: ❌ **NOT STARTED**
**ETA**: 30 minutes

---

### 8. Add Authentication to API Endpoints
**Files**: All `/api/bank/*` endpoints

- [ ] Implement JWT session management
- [ ] Add `verify_session_token()` function
- [ ] Require authentication on all sensitive endpoints
- [ ] Add user<->item_id mapping table
- [ ] Test unauthorized access blocked (401)

**Status**: ❌ **NOT STARTED**
**ETA**: 4 hours

---

## 🟡 MEDIUM PRIORITY (Fix This Month)

### 9. Add Log Sanitization
**Files**: All logging statements

- [ ] Create `sanitize_log()` helper function
- [ ] Review all logging statements
- [ ] Apply sanitization to sensitive data
- [ ] Add automated log scanning to CI/CD

**Status**: ❌ **NOT STARTED**
**ETA**: 2 hours

---

### 10. Implement Token Rotation
**File**: `src/finances/bank_database.py`

- [ ] Add `token_expiration` column to `plaid_items` table
- [ ] Schedule periodic token refresh
- [ ] Handle expired tokens gracefully
- [ ] Test token rotation workflow

**Status**: ❌ **NOT STARTED**
**ETA**: 3 hours

---

### 11. Add Audit Logging
**Files**: All Plaid endpoints

- [ ] Create audit log table
- [ ] Log all token exchanges
- [ ] Log all bank account access
- [ ] Implement tamper-proof logging
- [ ] Add audit log viewer

**Status**: ❌ **NOT STARTED**
**ETA**: 4 hours

---

## 🔵 LOW PRIORITY (Enhancement)

### 12. Add Security Headers
**File**: `src/dashboard/run_server_simple.py`

- [ ] Add `X-Frame-Options: DENY`
- [ ] Add `X-Content-Type-Options: nosniff`
- [ ] Add `X-XSS-Protection: 1; mode=block`
- [ ] Add `Strict-Transport-Security`
- [ ] Test headers with security scanner

**Status**: ❌ **NOT STARTED**
**ETA**: 30 minutes

---

## Quick Commands

### Check Database Permissions
```bash
ls -la data/bank_accounts.db
```

### Fix Database Permissions
```bash
chmod 600 data/bank_accounts.db
```

### Generate Encryption Key
```python
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
```

### Test Rate Limiting
```bash
for i in {1..10}; do curl http://localhost:8000/api/plaid/create_link_token; done
```

### Check for Tokens in Logs
```bash
grep -r "access_token\|access-sandbox-\|public-sandbox-" logs/
```

### Run Security Tests
```bash
pytest tests/security/test_plaid_security.py -v
```

---

## Progress Tracker

| Priority | Issues | Fixed | Progress |
|----------|--------|-------|----------|
| CRITICAL | 4 | 0 | 0% ░░░░░░░░░░ |
| HIGH | 4 | 0 | 0% ░░░░░░░░░░ |
| MEDIUM | 3 | 0 | 0% ░░░░░░░░░░ |
| LOW | 1 | 0 | 0% ░░░░░░░░░░ |
| **TOTAL** | **12** | **0** | **0%** ░░░░░░░░░░ |

---

## Estimated Timeline

| Phase | Duration | Items | Start Date | End Date |
|-------|----------|-------|------------|----------|
| CRITICAL Fixes | 8 hours | 4 items | Nov 7 | Nov 7 |
| HIGH Fixes | 7.5 hours | 4 items | Nov 8 | Nov 8 |
| MEDIUM Fixes | 9 hours | 3 items | Nov 11 | Nov 15 |
| LOW Enhancements | 30 min | 1 item | Nov 18 | Nov 18 |
| **TOTAL** | **25 hours** | **12 items** | **Nov 7** | **Nov 18** |

---

## Sign-Off

Once all CRITICAL and HIGH issues are resolved:

- [ ] All security tests passing
- [ ] Manual penetration testing completed
- [ ] Security audit report reviewed
- [ ] Production deployment checklist completed
- [ ] Security team sign-off obtained
- [ ] Deployment to production approved

**Deployment Approval**:
- Security Lead: _________________ Date: _______
- Tech Lead: _________________ Date: _______
- Product Owner: _________________ Date: _______

---

## Notes

- **CRITICAL issues block production deployment**
- **HIGH issues should block production but can be hot-fixed**
- **MEDIUM issues should be in next sprint**
- **LOW issues are nice-to-have enhancements**

**Last Updated**: 2025-11-07
**Next Review**: After CRITICAL fixes completed
