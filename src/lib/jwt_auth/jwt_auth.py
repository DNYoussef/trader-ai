"""
JWT Authentication Library - Token Management & Password Hashing

A reusable, project-agnostic JWT authentication module implementing OWASP API2:2023
Broken Authentication mitigations.

Features:
    - JWT access and refresh token creation/verification
    - Bcrypt password hashing with automatic algorithm upgrades
    - Cryptographically secure token generation
    - Configurable expiration times
    - Token type validation (access vs refresh)

Usage:
    from jwt_auth import JWTAuth

    # Initialize with configuration
    auth = JWTAuth(
        secret_key="your-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7
    )

    # Create tokens
    access_token = auth.create_access_token({"sub": "user_id_123"})
    refresh_token = auth.create_refresh_token({"sub": "user_id_123"})

    # Verify tokens
    payload = auth.verify_token(access_token, token_type="access")

    # Password operations
    hashed = auth.hash_password("my_password")
    is_valid = auth.verify_password("my_password", hashed)

Dependencies:
    - python-jose[cryptography]>=3.3.0
    - passlib[bcrypt]>=1.7.4

Author: Extracted from Life OS Dashboard for cross-project reuse
License: MIT
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext


@dataclass
class JWTConfig:
    """
    Configuration for JWT authentication.

    Attributes:
        secret_key: Secret key for signing tokens. MUST be kept secure.
        algorithm: JWT signing algorithm (default: HS256)
        access_token_expire_minutes: Access token lifetime in minutes (default: 30)
        refresh_token_expire_days: Refresh token lifetime in days (default: 7)
        issuer: Optional JWT issuer claim
        audience: Optional JWT audience claim
    """
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: Optional[str] = None
    audience: Optional[str] = None


class JWTAuth:
    """
    JWT Authentication Manager.

    Provides comprehensive JWT token management and password hashing
    functionality. Designed for reuse across multiple projects.

    Thread Safety:
        This class is thread-safe for all operations.

    Security Notes:
        - Uses bcrypt for password hashing (resistant to GPU attacks)
        - Includes token type validation to prevent token confusion attacks
        - Refresh tokens include unique JTI for revocation support
        - All tokens include iat (issued at) claims for audit trails
        - Requires minimum 32-byte secret key for HS256 algorithm security

    Example:
        >>> auth = JWTAuth(JWTConfig(secret_key="my-secret-key"))
        >>> token = auth.create_access_token({"sub": "user123", "role": "admin"})
        >>> payload = auth.verify_token(token)
        >>> print(payload["sub"])
        'user123'
    """

    __slots__ = ('_config', '_pwd_context', '_logger')

    # Minimum secret key length in bytes for HS256 security
    MIN_SECRET_KEY_LENGTH = 32

    def __init__(self, config: Union[JWTConfig, dict[str, Any]]):
        """
        Initialize JWT authentication manager.

        Args:
            config: JWTConfig instance or dictionary with configuration values.
                   Required keys: secret_key
                   Optional keys: algorithm, access_token_expire_minutes,
                                 refresh_token_expire_days, issuer, audience

        Raises:
            ValueError: If secret_key is not provided, is empty, or is too short
        """
        # Initialize logger for audit purposes
        self._logger = logging.getLogger(__name__)

        if isinstance(config, dict):
            secret_key = config.get("secret_key")
            if not secret_key:
                raise ValueError("secret_key is required and cannot be empty")
            self._config = JWTConfig(**config)
        else:
            if not config.secret_key:
                raise ValueError("secret_key is required and cannot be empty")
            self._config = config

        # HIGH-JWT-01: Validate minimum secret key length for HS256 security
        # OWASP recommends at least 256 bits (32 bytes) for HMAC-SHA256
        if len(self._config.secret_key) < self.MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"secret_key must be at least {self.MIN_SECRET_KEY_LENGTH} bytes "
                f"for {self._config.algorithm} security. Got {len(self._config.secret_key)} bytes."
            )

        # Password hashing context using bcrypt
        # 'auto' deprecation allows seamless algorithm upgrades
        self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # ============ Password Hashing ============

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Uses bcrypt with automatic work factor adjustment. Safe for storing
        in databases.

        Args:
            password: Plain text password to hash

        Returns:
            Bcrypt hash string (60 characters, starts with $2b$)

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> hashed = auth.hash_password("my_secure_password")
            >>> hashed.startswith("$2b$")
            True
        """
        return self._pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            plain_password: Plain text password from user input
            hashed_password: Stored bcrypt hash to verify against

        Returns:
            True if password matches hash, False otherwise

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> hashed = auth.hash_password("correct_password")
            >>> auth.verify_password("correct_password", hashed)
            True
            >>> auth.verify_password("wrong_password", hashed)
            False
        """
        return self._pwd_context.verify(plain_password, hashed_password)

    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if a password hash needs to be upgraded.

        Returns True if the hash uses a deprecated algorithm or
        insufficient work factor. Useful for rolling hash upgrades.

        Args:
            hashed_password: Stored bcrypt hash to check

        Returns:
            True if hash should be regenerated, False otherwise

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> hashed = auth.hash_password("password")
            >>> auth.needs_rehash(hashed)
            False
        """
        return self._pwd_context.needs_update(hashed_password)

    # ============ JWT Token Creation ============

    def create_access_token(
        self,
        data: dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Access tokens are short-lived and used for API authentication.
        Include user identification and authorization claims.

        Args:
            data: Payload data to encode. Should include 'sub' (subject)
                  for user identification. Values can include Decimal types
                  which will be converted to float for JSON serialization.
            expires_delta: Custom expiration time. If None, uses configured
                          access_token_expire_minutes.

        Returns:
            Encoded JWT token string

        Token Structure:
            - sub: Subject (typically user_id)
            - exp: Expiration timestamp
            - iat: Issued at timestamp
            - type: "access"
            - iss: Issuer (if configured)
            - aud: Audience (if configured)
            - ... (additional claims from data)

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_access_token({
            ...     "sub": "user_123",
            ...     "role": "admin",
            ...     "permissions": ["read", "write"]
            ... })
        """
        to_encode = self._prepare_payload(data)

        # Set expiration
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self._config.access_token_expire_minutes
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        })

        # Add optional claims
        if self._config.issuer:
            to_encode["iss"] = self._config.issuer
        if self._config.audience:
            to_encode["aud"] = self._config.audience

        return jwt.encode(
            to_encode,
            self._config.secret_key,
            algorithm=self._config.algorithm
        )

    def create_refresh_token(
        self,
        data: dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token.

        Refresh tokens are long-lived and used to obtain new access tokens.
        Include a unique JTI (JWT ID) for token revocation support.

        Args:
            data: Payload data to encode. Should include 'sub' (subject)
                  for user identification.
            expires_delta: Custom expiration time. If None, uses configured
                          refresh_token_expire_days.

        Returns:
            Encoded JWT refresh token string

        Token Structure:
            - sub: Subject (typically user_id)
            - exp: Expiration timestamp
            - iat: Issued at timestamp
            - type: "refresh"
            - jti: Unique token identifier (for revocation)
            - iss: Issuer (if configured)
            - aud: Audience (if configured)

        Security Note:
            Store the 'jti' claim in a database to enable token revocation.
            When a user logs out, add the jti to a revocation list.

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> refresh_token = auth.create_refresh_token({"sub": "user_123"})
        """
        to_encode = self._prepare_payload(data)

        # Set expiration
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self._config.refresh_token_expire_days
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID for revocation
        })

        # Add optional claims
        if self._config.issuer:
            to_encode["iss"] = self._config.issuer
        if self._config.audience:
            to_encode["aud"] = self._config.audience

        return jwt.encode(
            to_encode,
            self._config.secret_key,
            algorithm=self._config.algorithm
        )

    # ============ JWT Token Verification ============

    def verify_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[dict[str, Any]]:
        """
        Verify and decode a JWT token.

        Validates signature, expiration, issuer (if configured), audience (if
        configured), and token type. Returns None for any validation failure.

        Args:
            token: JWT token string to verify
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            Optional[dict[str, Any]]: Decoded token payload as a dictionary
                containing all claims (sub, exp, iat, type, etc.) if the token
                is valid. Returns None if validation fails for any reason:
                - Invalid or tampered signature
                - Expired token (exp claim in the past)
                - Wrong token type (type claim doesn't match token_type arg)
                - Invalid issuer (iss claim doesn't match config.issuer)
                - Invalid audience (aud claim doesn't match config.audience)
                - Malformed token structure

        Security Notes:
            - Signature is cryptographically verified against secret_key
            - Expiration (exp) is automatically checked by jose library
            - Issuer (iss) is validated when config.issuer is set
            - Audience (aud) is validated when config.audience is set
            - Token type must match expected type to prevent confusion attacks
            - Returns None instead of raising exceptions (safe for auth flows)

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_access_token({"sub": "user_123"})
            >>> payload = auth.verify_token(token, token_type="access")
            >>> payload["sub"]
            'user_123'
            >>> auth.verify_token(token, token_type="refresh")  # Wrong type
            None
        """
        try:
            # Build decode options
            audience = self._config.audience if self._config.audience else None
            issuer = self._config.issuer if self._config.issuer else None

            # HIGH-JWT-02: Include issuer validation when configured
            # This prevents tokens from other issuers being accepted
            payload = jwt.decode(
                token,
                self._config.secret_key,
                algorithms=[self._config.algorithm],
                audience=audience,
                issuer=issuer,
                options={
                    "verify_iss": bool(issuer),  # Only verify if issuer is configured
                    "verify_aud": bool(audience)  # Only verify if audience is configured
                }
            )

            # Verify token type to prevent token confusion attacks
            if payload.get("type") != token_type:
                return None

            return payload

        except JWTError:
            return None

    def decode_token_unsafe(self, token: str) -> Optional[dict[str, Any]]:
        """
        Decode a JWT token WITHOUT signature verification.

        WARNING: This method does NOT verify the token signature.
        Use only for debugging, logging, or extracting claims from
        tokens that you know are already validated elsewhere.

        Note:
            All calls to this method are logged at WARNING level for
            security audit purposes. Monitor these logs in production.

        Args:
            token: JWT token string to decode

        Returns:
            Decoded token payload or None if malformed

        Security Warning:
            NEVER use this for authentication decisions.
            Tokens decoded this way may be forged or tampered with.

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_access_token({"sub": "user_123"})
            >>> payload = auth.decode_token_unsafe(token)
            >>> payload["type"]
            'access'
        """
        # MED-JWT-01: Log all unsafe decode operations for security audit
        # This helps track potentially dangerous usage patterns in production
        self._logger.warning(
            "decode_token_unsafe called - token decoded without signature verification. "
            "Ensure this is intentional and not used for authentication decisions."
        )

        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            return payload
        except JWTError:
            return None

    def get_subject_from_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[str]:
        """
        Extract the subject (sub) claim from a verified token.

        Convenience method that verifies the token and extracts the
        subject claim in a single operation.

        Args:
            token: JWT token to verify and extract from
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            Subject claim value as string, or None if token invalid

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_access_token({"sub": "user_123"})
            >>> auth.get_subject_from_token(token)
            'user_123'
        """
        payload = self.verify_token(token, token_type=token_type)
        if not payload:
            return None

        sub = payload.get("sub")
        return str(sub) if sub is not None else None

    def get_user_id_from_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[int]:
        """
        Extract numeric user ID from a verified token.

        Convenience method for the common case where 'sub' contains
        a numeric user ID.

        Args:
            token: JWT access token to verify and extract from
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            User ID as integer, or None if token invalid or sub is not numeric

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_access_token({"sub": 12345})
            >>> auth.get_user_id_from_token(token)
            12345
        """
        payload = self.verify_token(token, token_type=token_type)
        if not payload:
            return None

        user_id = payload.get("sub")
        if user_id is None:
            return None

        try:
            return int(user_id)
        except (ValueError, TypeError):
            return None

    def get_jti_from_token(self, token: str) -> Optional[str]:
        """
        Extract JTI (JWT ID) from a refresh token.

        Used for token revocation. Store returned JTIs in a revocation
        list to invalidate refresh tokens.

        Args:
            token: JWT refresh token

        Returns:
            JTI string if present, None otherwise

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> token = auth.create_refresh_token({"sub": "user_123"})
            >>> jti = auth.get_jti_from_token(token)
            >>> # Store jti in database for revocation tracking
        """
        payload = self.verify_token(token, token_type="refresh")
        if not payload:
            return None

        return payload.get("jti")

    def refresh_access_token(
        self,
        refresh_token: str,
        additional_claims: Optional[dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Exchange a valid refresh token for a new access token.

        This method verifies the refresh token and creates a new access token
        with the same subject (user ID) from the refresh token.

        Args:
            refresh_token: Valid JWT refresh token
            additional_claims: Optional additional claims to include in the
                             new access token (e.g., updated permissions)

        Returns:
            New access token string if refresh token is valid, None otherwise

        Security Notes:
            - Always check if the refresh token's JTI is in your revocation list
              before calling this method
            - Consider implementing refresh token rotation (issue new refresh
              token and revoke the old one)

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> refresh = auth.create_refresh_token({"sub": "user_123"})
            >>> # Later, when access token expires:
            >>> new_access = auth.refresh_access_token(refresh)
            >>> if new_access:
            ...     # Use the new access token
            ...     pass
            >>> else:
            ...     # Refresh token invalid, user must re-authenticate
            ...     pass
        """
        # Verify the refresh token
        payload = self.verify_token(refresh_token, token_type="refresh")
        if not payload:
            self._logger.warning("Failed to refresh access token: invalid refresh token")
            return None

        # Extract subject from refresh token
        subject = payload.get("sub")
        if subject is None:
            self._logger.warning("Failed to refresh access token: no subject in refresh token")
            return None

        # Build new access token claims
        new_claims: dict[str, Any] = {"sub": subject}

        # Carry over any non-standard claims from refresh token
        # (excluding JWT-specific claims)
        excluded_claims = {"sub", "exp", "iat", "type", "jti", "iss", "aud"}
        for key, value in payload.items():
            if key not in excluded_claims:
                new_claims[key] = value

        # Add any additional claims provided
        if additional_claims:
            new_claims.update(additional_claims)

        self._logger.info(f"Refreshed access token for subject: {subject}")
        return self.create_access_token(new_claims)

    def rotate_refresh_token(
        self,
        refresh_token: str
    ) -> Optional[tuple[str, str]]:
        """
        Rotate a refresh token: create new access and refresh tokens.

        This is a security best practice where each refresh token can only
        be used once. The old refresh token should be revoked after this call.

        Args:
            refresh_token: Valid JWT refresh token to rotate

        Returns:
            Tuple of (new_access_token, new_refresh_token) if valid,
            None if refresh token is invalid

        Security Notes:
            - After calling this method, immediately revoke the old refresh token
              by adding its JTI to your revocation list
            - If a revoked refresh token is used, it may indicate token theft -
              consider revoking ALL tokens for that user

        Example:
            >>> auth = JWTAuth(JWTConfig(secret_key="secret"))
            >>> refresh = auth.create_refresh_token({"sub": "user_123"})
            >>> old_jti = auth.get_jti_from_token(refresh)
            >>> result = auth.rotate_refresh_token(refresh)
            >>> if result:
            ...     new_access, new_refresh = result
            ...     # Revoke old_jti in your database
            ...     # Return new tokens to client
        """
        # Verify the refresh token
        payload = self.verify_token(refresh_token, token_type="refresh")
        if not payload:
            self._logger.warning("Failed to rotate refresh token: invalid token")
            return None

        # Extract subject
        subject = payload.get("sub")
        if subject is None:
            return None

        # Build claims for new tokens
        new_claims: dict[str, Any] = {"sub": subject}

        # Carry over non-standard claims
        excluded_claims = {"sub", "exp", "iat", "type", "jti", "iss", "aud"}
        for key, value in payload.items():
            if key not in excluded_claims:
                new_claims[key] = value

        # Create new token pair
        new_access = self.create_access_token(new_claims)
        new_refresh = self.create_refresh_token(new_claims)

        old_jti = payload.get("jti", "unknown")
        self._logger.info(
            f"Rotated refresh token for subject {subject}. Old JTI: {old_jti}"
        )

        return new_access, new_refresh

    # ============ Utility Methods ============

    def _prepare_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare payload data for JWT encoding.

        Converts Decimal types to float for JSON serialization.

        Args:
            data: Raw payload data

        Returns:
            Processed payload ready for JWT encoding
        """
        to_encode = {}
        for key, value in data.items():
            if not isinstance(value, Decimal):
                to_encode[key] = value
                continue
            to_encode[key] = float(value)
        return to_encode

    @property
    def config(self) -> JWTConfig:
        """Get the current JWT configuration (read-only)."""
        return self._config


# ============ Standalone Utility Functions ============

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Uses the secrets module for cryptographic-quality randomness.
    Suitable for session tokens, CSRF tokens, password reset tokens, etc.

    Args:
        length: Token length in bytes. The resulting string will be
               approximately 4/3 times this length due to base64 encoding.

    Returns:
        URL-safe base64-encoded token string

    Example:
        >>> token = generate_secure_token(32)
        >>> len(token)  # Approximately 43 characters
        43
    """
    return secrets.token_urlsafe(length)


def generate_api_key(prefix: str = "sk") -> str:
    """
    Generate an API key with a recognizable prefix.

    Format: {prefix}_{random_token}
    Example: sk_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

    Args:
        prefix: Key prefix for identification (default: "sk" for secret key)

    Returns:
        Prefixed API key string

    Example:
        >>> key = generate_api_key("pk")  # Public key style
        >>> key.startswith("pk_")
        True
    """
    return f"{prefix}_{secrets.token_urlsafe(32)}"
