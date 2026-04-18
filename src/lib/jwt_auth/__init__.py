"""
JWT Authentication Library

A reusable, project-agnostic JWT authentication module implementing
OWASP API2:2023 Broken Authentication mitigations.

Quick Start:
    from jwt_auth import JWTAuth, JWTConfig

    # Initialize with configuration
    auth = JWTAuth(JWTConfig(
        secret_key="your-256-bit-secret-key",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7
    ))

    # Or use a dictionary
    auth = JWTAuth({"secret_key": "your-secret"})

    # Password hashing
    hashed = auth.hash_password("user_password")
    is_valid = auth.verify_password("user_password", hashed)

    # Token creation
    access_token = auth.create_access_token({"sub": "user_id"})
    refresh_token = auth.create_refresh_token({"sub": "user_id"})

    # Token verification
    payload = auth.verify_token(access_token, token_type="access")
    user_id = auth.get_user_id_from_token(access_token)

    # Standalone utilities
    from jwt_auth import generate_secure_token, generate_api_key

    session_token = generate_secure_token(32)
    api_key = generate_api_key("sk")  # sk_a1b2c3...

Dependencies:
    pip install python-jose[cryptography] passlib[bcrypt]

For detailed documentation, see jwt_auth.py or the README.md file.
"""

from .jwt_auth import (
    # Main class
    JWTAuth,
    # Configuration
    JWTConfig,
    # Utility functions
    generate_secure_token,
    generate_api_key,
)

__all__ = [
    "JWTAuth",
    "JWTConfig",
    "generate_secure_token",
    "generate_api_key",
]

__version__ = "1.0.0"
__author__ = "David Youssef"
