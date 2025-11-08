# API Key & Secret Rotation Policy

**Last Updated**: 2025-11-08
**Owner**: Security Team
**Review Frequency**: Quarterly

---

## Purpose

This document establishes the policy and procedures for rotating API keys, secrets, and cryptographic keys used in the trader-ai trading system.

---

## Rotation Schedule

### Automatic Rotation (Quarterly)

| Key Type | Rotation Frequency | Last Rotated | Next Due |
|----------|-------------------|--------------|----------|
| `JWT_SECRET_KEY` | Every 90 days | - | - |
| `DATABASE_ENCRYPTION_KEY` | Every 90 days | - | - |

### Event-Driven Rotation (On Security Incident)

| Key Type | Rotation Trigger |
|----------|------------------|
| `PLAID_CLIENT_ID` | Security breach, unauthorized access, or Plaid notification |
| `PLAID_SECRET` | Security breach, unauthorized access, or Plaid notification |
| `ALPACA_API_KEY` | Security breach, unauthorized access, or Alpaca notification |
| `ALPACA_SECRET_KEY` | Security breach, unauthorized access, or Alpaca notification |
| `HF_TOKEN` | Security breach, unauthorized access, or HuggingFace notification |

---

## Rotation Procedures

### 1. JWT Secret Key Rotation

**Complexity**: Moderate
**Downtime Required**: Yes (< 5 minutes)
**Impact**: All active sessions invalidated

#### Steps:

1. **Generate New Secret**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Update Environment Variable**
   ```bash
   # .env file
   JWT_SECRET_KEY=<new-secret-key>
   ```

3. **Restart Backend Server**
   ```bash
   # This invalidates all existing sessions
   ./stop_all.bat
   ./start_ui.bat
   ```

4. **Notify Users**
   - All users will need to re-authenticate
   - Sessions will be invalidated

5. **Verification**
   ```bash
   # Test authentication endpoint
   curl -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test"}'
   ```

---

### 2. Database Encryption Key Rotation

**Complexity**: High
**Downtime Required**: Yes (10-30 minutes)
**Impact**: Requires re-encryption of all stored Plaid tokens

#### Steps:

1. **Generate New Encryption Key**
   ```bash
   python scripts/security/generate_encryption_key.py
   ```

2. **Create Migration Script**
   ```python
   # scripts/security/rotate_db_key.py
   from cryptography.fernet import Fernet
   import sqlite3

   old_key = b"<old-key>"
   new_key = b"<new-key>"

   old_cipher = Fernet(old_key)
   new_cipher = Fernet(new_key)

   # Re-encrypt all access tokens
   conn = sqlite3.connect('data/bank_accounts.db')
   cursor = conn.cursor()

   cursor.execute("SELECT item_id, access_token FROM plaid_items")
   for row in cursor.fetchall():
       item_id, encrypted_token = row
       decrypted = old_cipher.decrypt(encrypted_token.encode())
       re_encrypted = new_cipher.encrypt(decrypted).decode()
       cursor.execute(
           "UPDATE plaid_items SET access_token = ? WHERE item_id = ?",
           (re_encrypted, item_id)
       )

   conn.commit()
   conn.close()
   ```

3. **Run Migration**
   ```bash
   python scripts/security/rotate_db_key.py
   ```

4. **Update Environment Variable**
   ```bash
   # .env file
   DATABASE_ENCRYPTION_KEY=<new-key>
   ```

5. **Restart Server**
   ```bash
   ./stop_all.bat
   ./start_ui.bat
   ```

6. **Verification**
   ```bash
   # Test Plaid token retrieval
   python -c "from src.finances.bank_database_encrypted import *; test_retrieval()"
   ```

---

### 3. Plaid API Credentials Rotation

**Complexity**: Low
**Downtime Required**: No
**Impact**: Minimal (seamless transition)

#### Steps:

1. **Generate New Credentials**
   - Login to https://dashboard.plaid.com
   - Navigate to Team Settings → Keys
   - Create new API keys
   - Copy `client_id` and `secret`

2. **Update Environment Variables**
   ```bash
   # .env file
   PLAID_CLIENT_ID=<new-client-id>
   PLAID_SECRET=<new-secret>
   ```

3. **Restart Backend Server**
   ```bash
   ./stop_all.bat
   ./start_ui.bat
   ```

4. **Test Plaid Integration**
   ```bash
   python test_plaid.py
   ```

5. **Deactivate Old Keys**
   - Return to Plaid Dashboard
   - Disable/delete old API keys
   - Confirm no errors in logs

---

### 4. Alpaca API Credentials Rotation

**Complexity**: Low
**Downtime Required**: No
**Impact**: Minimal

#### Steps:

1. **Generate New API Keys**
   - Login to https://app.alpaca.markets
   - Navigate to Paper Trading → API Keys (or Live Trading)
   - Regenerate API keys
   - Copy `API Key ID` and `Secret Key`

2. **Update Environment Variables**
   ```bash
   # .env file (or config/config.json)
   ALPACA_API_KEY=<new-key>
   ALPACA_SECRET_KEY=<new-secret>
   ```

3. **Update Config File** (if using config.json)
   ```json
   {
     "api_key": "env:ALPACA_API_KEY",
     "secret_key": "env:ALPACA_SECRET_KEY"
   }
   ```

4. **Restart Trading Engine**
   ```bash
   python main.py --test  # Verify connection
   python main.py  # Start trading
   ```

5. **Deactivate Old Keys**
   - Return to Alpaca Dashboard
   - Delete old API keys

---

### 5. HuggingFace Token Rotation

**Complexity**: Low
**Downtime Required**: No
**Impact**: Minimal (only affects AI features)

#### Steps:

1. **Generate New Token**
   - Login to https://huggingface.co
   - Navigate to Settings → Access Tokens
   - Create new token with `read` permissions
   - Copy token value

2. **Update Environment Variable**
   ```bash
   # .env file
   HF_TOKEN=<new-token>
   ```

3. **Restart Server**
   ```bash
   ./stop_all.bat
   ./start_ui.bat
   ```

4. **Revoke Old Token**
   - Return to HuggingFace Settings
   - Revoke old access token

---

## Security Incident Response

### If Keys Are Compromised:

1. **Immediate Actions** (Within 1 hour):
   - [ ] Rotate ALL affected keys immediately
   - [ ] Review access logs for unauthorized usage
   - [ ] Notify security team
   - [ ] Document incident details

2. **Short-Term Actions** (Within 24 hours):
   - [ ] Investigate root cause of exposure
   - [ ] Review all API usage logs
   - [ ] Check for unauthorized transactions (Alpaca)
   - [ ] Check for unauthorized bank access (Plaid)
   - [ ] Rotate related keys as precaution

3. **Long-Term Actions** (Within 1 week):
   - [ ] Update security policies
   - [ ] Implement additional monitoring
   - [ ] Review access controls
   - [ ] Conduct security training

---

## Key Storage Best Practices

### Development Environment

✅ **DO:**
- Store keys in `.env` file (gitignored)
- Use `.env.example` as template (no real keys)
- Keep `.env` file permissions restricted (chmod 600)
- Use different keys for development vs production

❌ **DON'T:**
- Commit `.env` to version control
- Share `.env` file via email/Slack
- Use production keys in development
- Hardcode keys in source code

### Production Environment

✅ **DO:**
- Use AWS Secrets Manager, HashiCorp Vault, or similar
- Rotate keys automatically via CI/CD
- Monitor key usage with CloudWatch/Datadog
- Implement least-privilege access

❌ **DON'T:**
- Store keys in environment variables on servers
- Use the same keys across multiple environments
- Share keys between team members
- Log keys in application logs

---

## Monitoring & Alerting

### Key Usage Monitoring

Monitor the following metrics:

1. **API Call Frequency**
   - Alpaca API calls per hour
   - Plaid API calls per hour
   - Unusual spikes in activity

2. **Authentication Failures**
   - Failed JWT verifications
   - Invalid API key attempts
   - Unusual authentication patterns

3. **Geographic Anomalies**
   - API calls from unexpected locations
   - Access from blacklisted IPs

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Failed auth attempts | 10/hour | 50/hour |
| Alpaca API errors | 5/hour | 20/hour |
| Plaid API errors | 3/hour | 10/hour |
| API calls from new IPs | 1 | 5 |

---

## Audit Trail

### Key Rotation Log

Maintain a rotation log in this format:

| Date | Key Type | Reason | Rotated By | Verified By |
|------|----------|--------|------------|-------------|
| 2025-11-08 | Initial Setup | N/A | System | System |
| - | - | - | - | - |

### Verification Checklist

After each rotation:

- [ ] New key works in all environments
- [ ] Old key has been deactivated
- [ ] No errors in application logs
- [ ] API integrations functioning normally
- [ ] Users notified if sessions invalidated
- [ ] Rotation logged in audit trail
- [ ] Backup of encrypted database (if applicable)

---

## Appendix: Key Generation Commands

### JWT Secret Key
```bash
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

### Database Encryption Key
```bash
python -c "from cryptography.fernet import Fernet; print('DATABASE_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
```

### Random Password (General)
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Contact Information

**Security Team**: security@trader-ai.example.com
**On-Call**: +1-XXX-XXX-XXXX
**Incident Report**: security-incidents@trader-ai.example.com

---

**Policy Version**: 1.0
**Next Review Date**: 2026-02-08

