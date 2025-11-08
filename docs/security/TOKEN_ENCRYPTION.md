# Plaid Token Encryption Documentation

## Overview

This document describes the Fernet symmetric encryption system implemented for securing Plaid access tokens in the database. The system prevents plaintext exposure of sensitive API credentials.

## Architecture

### Components

1. **TokenEncryption Module** (`src/security/token_encryption.py`)
   - Fernet symmetric encryption (AES-128 CBC + HMAC-SHA256)
   - Key management from environment variables
   - Automatic encryption/decryption helpers

2. **BankDatabase Integration** (`src/finances/bank_database_encrypted.py`)
   - Automatic token encryption on write operations
   - Automatic token decryption on read operations
   - Graceful fallback if encryption unavailable

3. **Migration Script** (`scripts/security/migrate_encrypt_tokens.py`)
   - One-time migration of plaintext tokens to encrypted format
   - Backup/rollback capability
   - Dry-run mode for testing

4. **Key Generation** (`scripts/security/generate_encryption_key.py`)
   - Secure Fernet key generation
   - Usage instructions and security warnings

## Security Features

### Encryption Algorithm

- **Algorithm**: Fernet (symmetric encryption)
- **Cipher**: AES-128 in CBC mode with PKCS7 padding
- **Authentication**: HMAC-SHA256 for integrity verification
- **Encoding**: Base64 URL-safe for database storage
- **IV**: Random initialization vector for each encryption

### Key Management

- **Storage**: Environment variable `DATABASE_ENCRYPTION_KEY`
- **Format**: Base64-encoded 32-byte key (44 characters)
- **Rotation**: Support for key rotation with migration
- **Backup**: Recommended to keep backup during rotation

## Setup Guide

### 1. Generate Encryption Key

```bash
python scripts/security/generate_encryption_key.py
```

This will output a key like:
```
DATABASE_ENCRYPTION_KEY=xH8f2k9L3mN6pQ9sT2uV5wX8zA1bC4dE5fG7hI9jK0lM2nO4pQ6rS8tU0vW2xY4zA6bC==
```

### 2. Configure Environment

Add to your `.env` file:
```bash
DATABASE_ENCRYPTION_KEY=<your_generated_key>
```

**⚠️ Security Warning**: Never commit the `.env` file to version control!

### 3. Migrate Existing Tokens (if any)

If you have existing plaintext tokens in the database:

```bash
# Preview changes (dry run)
python scripts/security/migrate_encrypt_tokens.py --dry-run

# Perform migration
python scripts/security/migrate_encrypt_tokens.py

# Rollback if needed
python scripts/security/migrate_encrypt_tokens.py --rollback
```

## Usage Examples

### Direct Usage

```python
from src.security.token_encryption import TokenEncryption

# Initialize (loads key from DATABASE_ENCRYPTION_KEY env var)
encryptor = TokenEncryption()

# Encrypt token before storage
plaintext_token = "access-sandbox-abc123def456"
encrypted_token = encryptor.encrypt_token(plaintext_token)

# Decrypt token after retrieval
original_token = encryptor.decrypt_token(encrypted_token)

# Verify roundtrip
assert encryptor.verify_roundtrip(plaintext_token)
```

### Database Integration

The `BankDatabase` class automatically handles encryption:

```python
from src.finances.bank_database_encrypted import BankDatabase

# Initialize database (automatically loads encryption)
db = BankDatabase("data/bank_accounts.db")

# Add item - token is encrypted automatically
item_id = db.add_plaid_item(
    access_token="access-sandbox-abc123",
    institution_name="Chase Bank"
)

# Retrieve token - automatically decrypted
decrypted_token = db.get_access_token(item_id)
```

## Testing

### Run Unit Tests

```bash
pytest tests/unit/security/test_token_encryption.py -v
```

### Test Coverage

- ✅ Key generation
- ✅ Encryption/decryption roundtrip
- ✅ Invalid key handling
- ✅ Decryption failures with wrong key
- ✅ Corrupted token handling
- ✅ Empty token validation
- ✅ Token detection (encrypted vs plaintext)
- ✅ Unicode and long token handling

## Security Best Practices

### 1. Key Storage

**Development:**
- Store in `.env` file (included in `.gitignore`)
- Use different keys for each developer

**Production:**
- Use AWS Secrets Manager, HashiCorp Vault, or similar
- Never hardcode keys in source code
- Implement key rotation policy

### 2. Key Rotation

Recommended quarterly rotation:

```bash
# 1. Generate new key
python scripts/security/generate_encryption_key.py

# 2. Keep old key as backup
cp .env .env.backup

# 3. Update .env with new key
# DATABASE_ENCRYPTION_KEY=<new_key>

# 4. Re-encrypt all tokens
python scripts/security/migrate_encrypt_tokens.py
```

### 3. Backup Strategy

- Create database backups before migration
- Store encryption key backups securely
- Test restore procedures regularly

### 4. Monitoring

- Log encryption failures (without exposing tokens)
- Monitor decryption errors for security incidents
- Track key rotation schedule

## Error Handling

### MissingEncryptionKeyError

**Cause**: `DATABASE_ENCRYPTION_KEY` not set

**Solution**:
```bash
python scripts/security/generate_encryption_key.py
# Add key to .env file
```

### InvalidEncryptionKeyError

**Cause**: Key format is invalid

**Solution**:
```bash
# Generate new valid key
python scripts/security/generate_encryption_key.py
```

### DecryptionFailedError

**Cause**: Token corrupted or wrong key

**Solution**:
1. Verify correct key is loaded
2. Check database integrity
3. Restore from backup if needed

## Performance Impact

- **Encryption**: ~0.1ms per token
- **Decryption**: ~0.1ms per token
- **Storage**: ~33% size increase (encrypted tokens are longer)
- **Memory**: Negligible (<1MB for encryption context)

## Compliance

This encryption system helps meet:
- **PCI DSS**: Requirement 3.4 (render PAN unreadable)
- **GDPR**: Article 32 (appropriate security measures)
- **SOC 2**: CC6.7 (encryption of sensitive data)

## Migration Notes

### From Plaintext to Encrypted

The migration script:
1. Creates automatic backup
2. Validates encryption key
3. Tests roundtrip encryption
4. Encrypts each token individually
5. Provides rollback capability

### Rollback Procedure

If migration fails or causes issues:
```bash
python scripts/security/migrate_encrypt_tokens.py --rollback
```

## Troubleshooting

### Issue: Tokens not decrypting

**Check**:
1. Correct `DATABASE_ENCRYPTION_KEY` is set
2. Key hasn't changed since encryption
3. Database hasn't been corrupted

**Debug**:
```python
from src.security.token_encryption import TokenEncryption

encryptor = TokenEncryption()
test_token = "access-sandbox-test"
print(encryptor.verify_roundtrip(test_token))  # Should be True
```

### Issue: Migration failing

**Check**:
1. Database file permissions
2. Sufficient disk space for backup
3. No concurrent database access

**Retry**:
```bash
python scripts/security/migrate_encrypt_tokens.py --dry-run  # Preview first
```

## Additional Resources

- [Cryptography Library Documentation](https://cryptography.io/en/latest/)
- [Fernet Specification](https://github.com/fernet/spec/)
- [OWASP Cryptographic Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html)

## Support

For issues or questions:
1. Check this documentation
2. Review test cases in `tests/unit/security/`
3. Examine migration script logs
4. Create issue with detailed error messages
