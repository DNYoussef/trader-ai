"""
Security Module for Trader-AI

This module provides security features including:
- Token encryption/decryption (Fernet symmetric encryption)
- Secure credential management
- Database encryption for sensitive data

Components:
- token_encryption: Encrypt/decrypt Plaid access tokens
"""

from src.security.token_encryption import (
    TokenEncryption,
    TokenEncryptionError,
    MissingEncryptionKeyError,
    InvalidEncryptionKeyError,
    DecryptionFailedError,
    init_encryption
)

__all__ = [
    'TokenEncryption',
    'TokenEncryptionError',
    'MissingEncryptionKeyError',
    'InvalidEncryptionKeyError',
    'DecryptionFailedError',
    'init_encryption'
]
