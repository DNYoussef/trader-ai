"""
Unit Tests for Token Encryption Module

Tests the Fernet-based encryption system for Plaid access tokens.

Test Coverage:
- Key generation
- Encryption/decryption roundtrip
- Invalid key handling
- Decryption failures
- Token detection (encrypted vs plaintext)
- Error handling
"""

import pytest
import os
from cryptography.fernet import Fernet

from src.security.token_encryption import (
    TokenEncryption,
    TokenEncryptionError,
    MissingEncryptionKeyError,
    InvalidEncryptionKeyError,
    DecryptionFailedError
)


class TestTokenEncryption:
    """Test suite for TokenEncryption class."""

    def test_key_generation(self):
        """Test that generate_key produces valid Fernet keys."""
        key = TokenEncryption.generate_key()

        # Key should be 44 characters (32 bytes base64 encoded)
        assert len(key) == 44

        # Should be able to create a Fernet instance with it
        cipher = Fernet(key.encode())
        assert cipher is not None

    def test_encryption_roundtrip(self):
        """Test encrypt/decrypt roundtrip with valid token."""
        # Generate test key
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        # Test token
        original_token = "access-sandbox-abc123def456"

        # Encrypt
        encrypted = encryptor.encrypt_token(original_token)
        assert encrypted != original_token
        assert len(encrypted) > len(original_token)

        # Decrypt
        decrypted = encryptor.decrypt_token(encrypted)
        assert decrypted == original_token

    def test_verify_roundtrip(self):
        """Test verify_roundtrip helper method."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        token = "access-sandbox-test123"
        assert encryptor.verify_roundtrip(token) is True

    def test_encryption_produces_different_output(self):
        """Test that same token produces different encrypted values (IV randomization)."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        token = "access-sandbox-test"
        encrypted1 = encryptor.encrypt_token(token)
        encrypted2 = encryptor.encrypt_token(token)

        # Different encrypted values due to random IV
        assert encrypted1 != encrypted2

        # But both decrypt to same original
        assert encryptor.decrypt_token(encrypted1) == token
        assert encryptor.decrypt_token(encrypted2) == token

    def test_missing_encryption_key(self):
        """Test that missing key raises appropriate error."""
        # Ensure env var not set
        old_key = os.environ.pop('DATABASE_ENCRYPTION_KEY', None)

        try:
            with pytest.raises(MissingEncryptionKeyError):
                TokenEncryption()
        finally:
            # Restore env var if it existed
            if old_key:
                os.environ['DATABASE_ENCRYPTION_KEY'] = old_key

    def test_invalid_encryption_key_format(self):
        """Test that invalid key format raises error."""
        with pytest.raises(InvalidEncryptionKeyError):
            TokenEncryption(encryption_key="invalid-key-format")

    def test_decryption_with_wrong_key(self):
        """Test that decryption fails with wrong key."""
        # Encrypt with one key
        key1 = TokenEncryption.generate_key()
        encryptor1 = TokenEncryption(encryption_key=key1)
        encrypted = encryptor1.encrypt_token("access-sandbox-test")

        # Try to decrypt with different key
        key2 = TokenEncryption.generate_key()
        encryptor2 = TokenEncryption(encryption_key=key2)

        with pytest.raises(DecryptionFailedError):
            encryptor2.decrypt_token(encrypted)

    def test_decryption_of_corrupted_token(self):
        """Test that corrupted encrypted token raises error."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        # Create corrupted encrypted token
        corrupted_token = "Z0FBQUFBQmFiYWRjb3JydXB0ZWQ="

        with pytest.raises(DecryptionFailedError):
            encryptor.decrypt_token(corrupted_token)

    def test_empty_token_encryption(self):
        """Test that empty token raises error."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        with pytest.raises(TokenEncryptionError):
            encryptor.encrypt_token("")

    def test_empty_token_decryption(self):
        """Test that empty encrypted token raises error."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        with pytest.raises(TokenEncryptionError):
            encryptor.decrypt_token("")

    def test_is_encrypted_plaintext(self):
        """Test that plaintext Plaid token is correctly identified."""
        plaintext_token = "access-sandbox-abc123"
        assert TokenEncryption.is_encrypted(plaintext_token) is False

    def test_is_encrypted_encrypted(self):
        """Test that encrypted token is correctly identified."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        plaintext = "access-sandbox-abc123"
        encrypted = encryptor.encrypt_token(plaintext)

        assert TokenEncryption.is_encrypted(encrypted) is True

    def test_unicode_token_handling(self):
        """Test that Unicode characters in tokens are handled correctly."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        # Token with Unicode characters
        unicode_token = "access-sandbox-测试-🔐"

        encrypted = encryptor.encrypt_token(unicode_token)
        decrypted = encryptor.decrypt_token(encrypted)

        assert decrypted == unicode_token

    def test_long_token_handling(self):
        """Test that very long tokens are handled correctly."""
        key = TokenEncryption.generate_key()
        encryptor = TokenEncryption(encryption_key=key)

        # Very long token (typical Plaid tokens are 50-100 chars)
        long_token = "access-sandbox-" + ("x" * 200)

        encrypted = encryptor.encrypt_token(long_token)
        decrypted = encryptor.decrypt_token(encrypted)

        assert decrypted == long_token
        assert len(encrypted) > len(long_token)


class TestIntegrationWithDatabase:
    """Integration tests with database module."""

    def test_database_encryption_integration(self):
        """Test that database correctly encrypts/decrypts tokens."""
        # This would require importing BankDatabase
        # Left as a placeholder for integration testing
        pass


class TestMigrationScript:
    """Tests for token migration script."""

    def test_migration_dry_run(self):
        """Test migration script in dry-run mode."""
        # This would require mocking database and running migration script
        # Left as a placeholder for integration testing
        pass

    def test_migration_rollback(self):
        """Test migration rollback functionality."""
        # This would require testing backup/restore logic
        # Left as a placeholder for integration testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
