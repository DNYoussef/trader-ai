# Token Encryption Quick Start Guide

Get Plaid token encryption up and running in 5 minutes.

## Prerequisites

✅ Python 3.8+
✅ trader-ai project installed
✅ Terminal/command prompt

## Step 1: Install Dependency (30 seconds)

```bash
cd C:\Users\17175\Desktop\trader-ai
pip install cryptography>=41.0.0
```

## Step 2: Generate Encryption Key (10 seconds)

```bash
python scripts/security/generate_encryption_key.py
```

**Output**:
```
======================================================================
Add this to your .env file:
======================================================================

DATABASE_ENCRYPTION_KEY=xH8f2k9L3mN6pQ9sT2uV5wX8zA1bC4dE5fG7hI9jK0lM2nO4pQ6rS8tU0vW2xY4zA6bC==
```

## Step 3: Add Key to .env File (30 seconds)

Copy the key and add to your `.env` file:

```bash
# .env file
DATABASE_ENCRYPTION_KEY=<paste_key_here>
```

**⚠️ NEVER commit .env to git!** (Already in .gitignore)

## Step 4: Migrate Existing Tokens (1 minute)

**If you have existing Plaid tokens:**

```bash
# Preview changes first
python scripts/security/migrate_encrypt_tokens.py --dry-run

# Perform migration
python scripts/security/migrate_encrypt_tokens.py
```

**If starting fresh:** Skip this step.

## Step 5: Verify Installation (30 seconds)

```bash
# Run unit tests
pytest tests/unit/security/test_token_encryption.py -v

# Expected output: 13 tests passed ✅
```

## Step 6: Update Code (2 minutes)

**Replace your imports:**

```python
# OLD (plaintext storage)
from src.finances.bank_database import BankDatabase

# NEW (encrypted storage)
from src.finances.bank_database_encrypted import BankDatabase
```

**That's it!** Tokens are now encrypted automatically.

## Usage Examples

### Add Plaid Item (Automatic Encryption)

```python
from src.finances.bank_database_encrypted import BankDatabase

db = BankDatabase("data/bank_accounts.db")

# Token is encrypted automatically before storage
item_id = db.add_plaid_item(
    access_token="access-sandbox-abc123def456",
    institution_name="Chase Bank"
)

print(f"Item added: {item_id}")
```

### Retrieve Token (Automatic Decryption)

```python
# Token is decrypted automatically on retrieval
decrypted_token = db.get_access_token(item_id)

print(f"Token: {decrypted_token}")
# Output: access-sandbox-abc123def456
```

### Manual Encryption/Decryption

```python
from src.security import TokenEncryption

encryptor = TokenEncryption()

# Encrypt
plaintext = "access-sandbox-test123"
encrypted = encryptor.encrypt_token(plaintext)
print(f"Encrypted: {encrypted[:50]}...")

# Decrypt
decrypted = encryptor.decrypt_token(encrypted)
print(f"Decrypted: {decrypted}")
```

## Troubleshooting

### Error: MissingEncryptionKeyError

**Cause**: `DATABASE_ENCRYPTION_KEY` not set

**Fix**:
```bash
python scripts/security/generate_encryption_key.py
# Add key to .env file
```

### Error: InvalidEncryptionKeyError

**Cause**: Key format is invalid

**Fix**:
```bash
# Generate new valid key
python scripts/security/generate_encryption_key.py
```

### Error: DecryptionFailedError

**Cause**: Wrong key or corrupted token

**Fix**:
1. Verify correct key in .env
2. Restore from backup:
   ```bash
   python scripts/security/migrate_encrypt_tokens.py --rollback
   ```

## Security Checklist

Before production:
- [ ] Generated unique encryption key
- [ ] Added key to production .env (not committed)
- [ ] Migrated all existing tokens
- [ ] Ran all tests (13/13 passing)
- [ ] Updated all database imports
- [ ] Stored key in AWS Secrets Manager (production)
- [ ] Set up quarterly key rotation reminder

## Quick Commands Reference

```bash
# Generate key
python scripts/security/generate_encryption_key.py

# Migrate tokens (dry run)
python scripts/security/migrate_encrypt_tokens.py --dry-run

# Migrate tokens (actual)
python scripts/security/migrate_encrypt_tokens.py

# Rollback migration
python scripts/security/migrate_encrypt_tokens.py --rollback

# Run tests
pytest tests/unit/security/test_token_encryption.py -v

# Test encryption manually
python -c "from src.security import TokenEncryption; e = TokenEncryption(); print(e.verify_roundtrip('test'))"
```

## Next Steps

✅ **You're done!** Tokens are now encrypted.

**Optional enhancements:**
1. Set up quarterly key rotation
2. Integrate with AWS Secrets Manager (production)
3. Add audit logging for encryption events
4. Set up monitoring for decryption failures

## Documentation

Full documentation:
- `docs/security/TOKEN_ENCRYPTION.md` - Comprehensive guide
- `docs/security/IMPLEMENTATION_SUMMARY.md` - Technical details

## Support

Issues? Check:
1. This quick start guide
2. Full documentation in `docs/security/`
3. Test examples in `tests/unit/security/`
4. Migration script logs

---

**Total Setup Time**: ~5 minutes
**Status**: ✅ Ready for production
