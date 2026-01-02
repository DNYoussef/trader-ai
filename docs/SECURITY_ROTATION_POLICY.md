# Security Key Rotation Policy

**Document Version**: 1.0
**Created**: 2026-01-02
**Owner**: Security Team

---

## Overview

This document defines the key rotation schedule and procedures for all secrets used in the trader-ai system.

## Rotation Schedule

| Secret | Rotation Period | Next Rotation | Risk if Compromised |
|--------|-----------------|---------------|---------------------|
| JWT_SECRET_KEY | 90 days (quarterly) | 2026-04-02 | Session hijacking |
| DATABASE_ENCRYPTION_KEY | 90 days (quarterly) | 2026-04-02 | Data exposure |
| ALPACA_API_KEY | On incident only | N/A | Unauthorized trades |
| PLAID_SECRET | On incident only | N/A | Bank data exposure |
| HF_TOKEN | On incident only | N/A | API abuse |

## Rotation Procedures

### JWT_SECRET_KEY Rotation

1. Generate new key:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Update `.env` with new key

3. Restart all services (active sessions will be invalidated)

4. Users will need to re-authenticate

### DATABASE_ENCRYPTION_KEY Rotation

**WARNING**: Requires data migration!

1. Export encrypted data:
   ```bash
   python scripts/export_encrypted_data.py --old-key
   ```

2. Generate new Fernet key:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

3. Update `.env` with new key

4. Re-encrypt and import data:
   ```bash
   python scripts/import_encrypted_data.py --new-key
   ```

5. Verify data integrity

### API Keys (Alpaca, Plaid, HuggingFace)

Rotate immediately if:
- Key exposed in logs, commits, or public channels
- Unauthorized access detected
- Employee with access leaves
- Suspicious API activity observed

Procedure:
1. Generate new key in provider dashboard
2. Update `.env`
3. Revoke old key in provider dashboard
4. Monitor for failures

## Emergency Rotation

If a key is compromised:

1. **IMMEDIATE**: Revoke compromised key at source
2. Generate new key
3. Update all environments (dev, staging, prod)
4. Document incident in security log
5. Review audit logs for unauthorized access
6. Notify affected parties if required

## Audit Trail

All rotation events must be logged:
- Date/time of rotation
- Person performing rotation
- Reason for rotation
- Verification of new key working

## Compliance

This policy supports:
- SOC 2 Type II requirements
- PCI DSS key management standards
- General security best practices

---

**Last Review**: 2026-01-02
**Next Review**: 2026-04-02
