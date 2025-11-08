"""
Token Encryption Module using Fernet Symmetric Encryption

This module provides secure encryption and decryption for Plaid access tokens
before database storage to prevent plaintext exposure.

Security Features:
- Fernet symmetric encryption (AES-128 CBC + HMAC-SHA256)
- Key derivation from environment variable
- Base64 URL-safe encoding for database storage
- Automatic key generation for initial setup

Usage:
    from src.security.token_encryption import TokenEncryption

    # Initialize (loads key from DATABASE_ENCRYPTION_KEY env var)
    encryptor = TokenEncryption()

    # Encrypt before storage
    encrypted = encryptor.encrypt_token("access-sandbox-abc123")

    # Decrypt on retrieval
    original = encryptor.decrypt_token(encrypted)
"""

from cryptography.fernet import Fernet, InvalidToken
import os
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TokenEncryptionError(Exception):
    """Base exception for token encryption errors"""
    pass


class MissingEncryptionKeyError(TokenEncryptionError):
    """Raised when DATABASE_ENCRYPTION_KEY environment variable is not set"""
    pass


class InvalidEncryptionKeyError(TokenEncryptionError):
    """Raised when encryption key format is invalid"""
    pass


class DecryptionFailedError(TokenEncryptionError):
    """Raised when token decryption fails"""
    pass


class TokenEncryption:
    """
    Manages encryption and decryption of Plaid access tokens.

    Uses Fernet symmetric encryption with the following characteristics:
    - AES-128 in CBC mode with PKCS7 padding
    - HMAC-SHA256 for authentication
    - Timestamp for token expiration (optional)
    - URL-safe Base64 encoding

    Attributes:
        cipher: Fernet cipher instance for encrypt/decrypt operations
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize token encryption with key from environment or parameter.

        Args:
            encryption_key: Optional encryption key (for testing).
                          If None, loads from DATABASE_ENCRYPTION_KEY env var.

        Raises:
            MissingEncryptionKeyError: If no key found
            InvalidEncryptionKeyError: If key format is invalid
        """
        # Load key from parameter or environment
        key = encryption_key or os.getenv('DATABASE_ENCRYPTION_KEY')

        if not key:
            raise MissingEncryptionKeyError(
                "DATABASE_ENCRYPTION_KEY environment variable not set. "
                "Generate a key with: python scripts/security/generate_encryption_key.py"
            )

        try:
            # Validate key format and create cipher
            self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
            logger.info("Token encryption initialized successfully")
        except Exception as e:
            raise InvalidEncryptionKeyError(
                f"Invalid encryption key format: {e}. "
                "Generate a new key with: python scripts/security/generate_encryption_key.py"
            )

    def encrypt_token(self, token: str) -> str:
        """
        Encrypt access token before database storage.

        Args:
            token: Plaintext Plaid access token

        Returns:
            Base64 URL-safe encoded encrypted token

        Raises:
            TokenEncryptionError: If encryption fails

        Example:
            >>> encryptor = TokenEncryption()
            >>> encrypted = encryptor.encrypt_token("access-sandbox-abc123")
            >>> print(encrypted[:20])  # First 20 chars
            Z0FBQUFBQm1...
        """
        if not token:
            raise TokenEncryptionError("Cannot encrypt empty token")

        try:
            # Encrypt token bytes
            encrypted_bytes = self.cipher.encrypt(token.encode('utf-8'))

            # Base64 encode for safe database storage
            encrypted_b64 = base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')

            logger.debug(f"Token encrypted successfully (length: {len(encrypted_b64)})")
            return encrypted_b64

        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise TokenEncryptionError(f"Encryption failed: {e}")

    def decrypt_token(self, encrypted_token: str) -> str:
        """
        Decrypt access token from database.

        Args:
            encrypted_token: Base64 URL-safe encoded encrypted token

        Returns:
            Decrypted plaintext access token

        Raises:
            DecryptionFailedError: If decryption fails or token is invalid

        Example:
            >>> encryptor = TokenEncryption()
            >>> original = encryptor.decrypt_token(encrypted_token)
            >>> print(original)
            access-sandbox-abc123
        """
        if not encrypted_token:
            raise TokenEncryptionError("Cannot decrypt empty token")

        try:
            # Decode from Base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_token.encode('utf-8'))

            # Decrypt token
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            decrypted_token = decrypted_bytes.decode('utf-8')

            logger.debug("Token decrypted successfully")
            return decrypted_token

        except InvalidToken as e:
            logger.error(f"Invalid token or wrong decryption key: {e}")
            raise DecryptionFailedError(
                "Token decryption failed. Token may be corrupted or wrong encryption key is being used."
            )
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise DecryptionFailedError(f"Decryption failed: {e}")

    def verify_roundtrip(self, token: str) -> bool:
        """
        Verify encryption/decryption roundtrip for testing.

        Args:
            token: Token to test

        Returns:
            True if roundtrip successful, False otherwise
        """
        try:
            encrypted = self.encrypt_token(token)
            decrypted = self.decrypt_token(encrypted)
            return decrypted == token
        except Exception as e:
            logger.error(f"Roundtrip verification failed: {e}")
            return False

    @staticmethod
    def generate_key() -> str:
        """
        Generate new Fernet encryption key for initial setup.

        Returns:
            Base64-encoded 32-byte encryption key

        Example:
            >>> key = TokenEncryption.generate_key()
            >>> print(f"DATABASE_ENCRYPTION_KEY={key}")
            DATABASE_ENCRYPTION_KEY=xH8f2k9L...

        Note:
            Store this key securely in environment variables.
            NEVER commit to version control!
        """
        key = Fernet.generate_key().decode('utf-8')
        logger.info("New encryption key generated")
        return key

    @staticmethod
    def is_encrypted(token: str) -> bool:
        """
        Heuristic check if token appears encrypted.

        Args:
            token: Token string to check

        Returns:
            True if token looks encrypted (Base64), False if plaintext

        Note:
            This is a heuristic check. Plaid tokens start with "access-"
        """
        # Plaid tokens start with "access-" prefix
        if token.startswith('access-'):
            return False

        # Encrypted tokens are Base64 (longer and different character set)
        try:
            # Try to decode as Base64
            base64.urlsafe_b64decode(token.encode('utf-8'))
            return True
        except Exception:
            return False


def init_encryption() -> TokenEncryption:
    """
    Initialize and return a TokenEncryption instance.

    Returns:
        Initialized TokenEncryption instance

    Raises:
        MissingEncryptionKeyError: If DATABASE_ENCRYPTION_KEY not set

    Example:
        >>> encryptor = init_encryption()
        >>> encrypted = encryptor.encrypt_token("my-token")
    """
    return TokenEncryption()


if __name__ == "__main__":
    # Demo usage
    print("Token Encryption Module")
    print("=" * 50)

    # Generate new key
    print("\n1. Generate new encryption key:")
    new_key = TokenEncryption.generate_key()
    print(f"   DATABASE_ENCRYPTION_KEY={new_key}")
    print("   ⚠️  Store this securely in your .env file!")

    # Test encryption (using generated key)
    print("\n2. Test encryption/decryption:")
    encryptor = TokenEncryption(encryption_key=new_key)

    test_token = "access-sandbox-test123"
    print(f"   Original: {test_token}")

    encrypted = encryptor.encrypt_token(test_token)
    print(f"   Encrypted: {encrypted[:50]}...")

    decrypted = encryptor.decrypt_token(encrypted)
    print(f"   Decrypted: {decrypted}")

    print(f"\n   ✓ Roundtrip successful: {decrypted == test_token}")

    # Check if token is encrypted
    print("\n3. Token detection:")
    print(f"   Is plaintext encrypted? {TokenEncryption.is_encrypted(test_token)}")
    print(f"   Is encrypted encrypted? {TokenEncryption.is_encrypted(encrypted)}")
