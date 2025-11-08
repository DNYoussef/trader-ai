# Plaid Security Code Fixes

**Project**: trader-ai Banking Integration
**Date**: November 7, 2025

This document contains copy-paste ready code fixes for all CRITICAL and HIGH priority security issues.

---

## CRITICAL FIX 1: Remove Access Token from API Response

**File**: `src/dashboard/run_server_simple.py:235-272`

### BEFORE (INSECURE):
```python
@self.app.post("/api/plaid/exchange_public_token")
async def exchange_plaid_public_token(token_data: dict):
    """
    Exchange Plaid public token for access token.

    Body: {"public_token": "public-sandbox-xxx"}

    Returns access_token that should be stored securely.
    """
    try:
        public_token = token_data.get('public_token')
        if not public_token:
            return {
                "success": False,
                "error": "Missing public_token in request body"
            }

        from src.finances.plaid_client import create_plaid_client

        plaid_client = create_plaid_client()
        access_token = plaid_client.exchange_public_token(public_token)

        # TODO: Store access_token securely (database/encrypted storage)
        # For now, return it (in production, only return success status)

        return {
            "success": True,
            "access_token": access_token,  # ❌ CRITICAL SECURITY ISSUE
            "message": "Bank account connected successfully"
        }

    except Exception as e:
        logger.error(f"Failed to exchange public token: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "plaid_token_exchange_error"
        }
```

### AFTER (SECURE):
```python
from fastapi import HTTPException, Header
from src.finances.bank_database import BankDatabase

@self.app.post("/api/plaid/exchange_public_token")
async def exchange_plaid_public_token(
    token_data: dict,
    authorization: str = Header(None)
):
    """
    Exchange Plaid public token for access token.
    Stores token securely server-side, never returns it to client.

    Body: {"public_token": "public-sandbox-xxx", "institution_name": "Bank Name"}
    Headers: {"Authorization": "Bearer <session_token>"}

    Returns item_id that client can use for subsequent requests.
    """
    try:
        # Verify user session
        user_id = verify_session_token(authorization)
        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized")

        public_token = token_data.get('public_token')
        if not public_token:
            raise HTTPException(
                status_code=400,
                detail="Missing public_token in request body"
            )

        institution_name = token_data.get('institution_name', 'Unknown Bank')

        # Exchange token with Plaid
        from src.finances.plaid_client import create_plaid_client
        plaid_client = create_plaid_client()
        access_token = plaid_client.exchange_public_token(public_token)

        # Store encrypted access token in database
        encryption_key = get_encryption_key()
        db = BankDatabase("data/bank_accounts.db", encryption_key)
        item_id = db.add_plaid_item(access_token, institution_name)

        # Link item_id to user in session
        link_user_item(user_id, item_id)

        logger.info(f"User {user_id} connected bank account {item_id}")

        return {
            "success": True,
            "item_id": item_id,  # ✅ Safe to return - can't be used without auth
            "institution_name": institution_name,
            "message": "Bank account connected successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to exchange public token: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to connect bank account. Please try again."
        )
```

---

## CRITICAL FIX 2: Move Access Token from URL to Header

**File**: `src/dashboard/run_server_simple.py:274-323`

### BEFORE (INSECURE):
```python
@self.app.get("/api/bank/accounts")
async def get_bank_accounts(access_token: str = None):
    """
    Get all linked bank accounts.

    Query param: ?access_token=access-sandbox-xxx  # ❌ LOGGED EVERYWHERE!

    Returns list of bank accounts with account details.
    """
    try:
        if not access_token:
            return {
                "success": False,
                "error": "Missing access_token query parameter"
            }

        from src.finances.plaid_client import create_plaid_client

        plaid_client = create_plaid_client()
        accounts = plaid_client.get_accounts(access_token)

        # Convert dataclass to dict
        accounts_data = [
            {
                "account_id": acc.account_id,
                "name": acc.name,
                # ...
            }
            for acc in accounts
        ]

        return {
            "success": True,
            "accounts": accounts_data,
            "count": len(accounts_data)
        }

    except Exception as e:
        logger.error(f"Failed to get bank accounts: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "plaid_accounts_error"
        }
```

### AFTER (SECURE):
```python
from fastapi import Request, HTTPException, Header
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@self.app.get("/api/bank/accounts")
@limiter.limit("20/minute")  # Rate limiting
async def get_bank_accounts(
    request: Request,
    authorization: str = Header(None)
):
    """
    Get all linked bank accounts.

    Headers: {"Authorization": "Bearer <session_token>"}

    Returns list of bank accounts with account details.
    """
    try:
        # Verify user session
        user_id = verify_session_token(authorization)
        if not user_id:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Get user's item_id from database
        item_id = get_user_item_id(user_id)
        if not item_id:
            raise HTTPException(
                status_code=404,
                detail="No bank accounts linked. Please connect a bank account first."
            )

        # Retrieve encrypted access token from database
        encryption_key = get_encryption_key()
        db = BankDatabase("data/bank_accounts.db", encryption_key)
        access_token = db.get_decrypted_access_token(item_id)

        # Fetch accounts from Plaid
        from src.finances.plaid_client import create_plaid_client
        plaid_client = create_plaid_client()
        accounts = plaid_client.get_accounts(access_token)

        # Convert dataclass to dict
        accounts_data = [
            {
                "account_id": acc.account_id,
                "name": acc.name,
                "official_name": acc.official_name,
                "type": acc.type,
                "subtype": acc.subtype,
                "mask": acc.mask,
                "current_balance": acc.current_balance,
                "available_balance": acc.available_balance,
                "currency": acc.currency_code
            }
            for acc in accounts
        ]

        logger.info(f"User {user_id} fetched {len(accounts_data)} bank accounts")

        return {
            "success": True,
            "accounts": accounts_data,
            "count": len(accounts_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bank accounts for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve bank accounts. Please try again."
        )
```

---

## CRITICAL FIX 3: Fix Database Permissions

**Command Line**:
```bash
# Set secure permissions (owner read/write only)
chmod 600 data/bank_accounts.db

# Verify
ls -la data/bank_accounts.db
# Should show: -rw------- (600)

# Also fix directory permissions
chmod 700 data/

# Verify no other users can access
ls -la data/
# Should show: drwx------ (700)
```

**Add to deployment script** (`scripts/deployment/deploy.sh`):
```bash
#!/bin/bash

# Ensure data directory exists with secure permissions
mkdir -p data
chmod 700 data

# Set secure permissions on database
if [ -f data/bank_accounts.db ]; then
    chmod 600 data/bank_accounts.db
    echo "✅ Database permissions set to 600"
else
    echo "⚠️  Database file not found - will be created on first run"
fi

# Verify permissions
echo "Checking permissions..."
ls -la data/bank_accounts.db 2>/dev/null || echo "Database will be created with 600 permissions"
```

---

## CRITICAL FIX 4: Encrypt Access Tokens

**File**: `src/finances/bank_database.py`

### Add Encryption Support:

```python
from cryptography.fernet import Fernet
import os
import base64

class BankDatabase:
    """
    SQLite database manager for Plaid banking data with encryption.
    """

    def __init__(self, db_path: str, encryption_key: bytes = None):
        """
        Initialize database connection with encryption support.

        Args:
            db_path: Path to SQLite database file
            encryption_key: Fernet encryption key (32-byte URL-safe base64-encoded)
        """
        self.db_path = db_path

        # Initialize encryption
        if encryption_key:
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()
            self.cipher = Fernet(encryption_key)
        else:
            raise ValueError(
                "Encryption key required. Set DATABASE_ENCRYPTION_KEY environment variable."
            )

        # Ensure directory exists with secure permissions
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._init_schema()

        # Set secure file permissions (owner read/write only)
        os.chmod(db_path, 0o600)

        logger.info(f"Database initialized at {db_path} with encryption enabled")

    def add_plaid_item(self, access_token: str, institution_name: str) -> str:
        """
        Add a new Plaid item connection with encrypted access token.

        Args:
            access_token: Plaid access token (WILL BE ENCRYPTED)
            institution_name: Human-readable institution name

        Returns:
            item_id: Generated unique identifier for this connection
        """
        # Encrypt access token before storage
        encrypted_token = self.cipher.encrypt(access_token.encode())

        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate unique item_id
        item_id = f"item_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        cursor.execute("""
            INSERT INTO plaid_items (item_id, access_token, institution_name)
            VALUES (?, ?, ?)
        """, (item_id, encrypted_token, institution_name))

        conn.commit()
        conn.close()

        logger.info(f"Added encrypted Plaid item {item_id}")

        return item_id

    def get_decrypted_access_token(self, item_id: str) -> str:
        """
        Retrieve and decrypt access token for a Plaid item.

        Args:
            item_id: Plaid item ID

        Returns:
            Decrypted access token

        Raises:
            ValueError: If item not found
            cryptography.fernet.InvalidToken: If decryption fails
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT access_token FROM plaid_items WHERE item_id = ?
        """, (item_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Plaid item {item_id} not found")

        # Decrypt token
        encrypted_token = row['access_token']
        try:
            decrypted_token = self.cipher.decrypt(encrypted_token).decode()
            return decrypted_token
        except Exception as e:
            logger.error(f"Failed to decrypt token for item {item_id}: {e}")
            raise ValueError("Failed to decrypt access token. Encryption key may be incorrect.")

    def update_access_token(self, item_id: str, new_access_token: str) -> bool:
        """
        Update (rotate) access token for existing item.

        Args:
            item_id: Plaid item ID
            new_access_token: New access token (WILL BE ENCRYPTED)

        Returns:
            True if updated successfully
        """
        # Encrypt new token
        encrypted_token = self.cipher.encrypt(new_access_token.encode())

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE plaid_items
            SET access_token = ?, updated_at = CURRENT_TIMESTAMP
            WHERE item_id = ?
        """, (encrypted_token, item_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if updated:
            logger.info(f"Access token rotated for item {item_id}")

        return updated


# Helper functions for encryption key management
def get_encryption_key() -> bytes:
    """
    Get encryption key from environment variable.

    Returns:
        Fernet encryption key

    Raises:
        ValueError: If key not set or invalid
    """
    key_str = os.getenv('DATABASE_ENCRYPTION_KEY')

    if not key_str:
        raise ValueError(
            "DATABASE_ENCRYPTION_KEY environment variable not set. "
            "Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )

    try:
        # Validate key format
        key = key_str.encode() if isinstance(key_str, str) else key_str
        Fernet(key)  # Test key is valid
        return key
    except Exception as e:
        raise ValueError(f"Invalid DATABASE_ENCRYPTION_KEY: {e}")


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.

    Returns:
        Base64-encoded encryption key as string

    Example:
        >>> key = generate_encryption_key()
        >>> print(f"DATABASE_ENCRYPTION_KEY={key}")
    """
    return Fernet.generate_key().decode()


# Convenience function for creating database with encryption
def init_bank_database(db_path: str = "data/bank_accounts.db") -> BankDatabase:
    """
    Initialize BankDatabase with encryption from environment.

    Args:
        db_path: Path to database file

    Returns:
        Initialized BankDatabase instance with encryption enabled
    """
    encryption_key = get_encryption_key()
    return BankDatabase(db_path, encryption_key)
```

---

## Supporting Code: Session Management

**File**: `src/auth/session_manager.py` (NEW FILE)

```python
"""
Session management for API authentication.
Uses JWT tokens for stateless authentication.
"""

import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException

# Secret key from environment (MUST be secret!)
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable not set")

JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def create_session_token(user_id: str, additional_claims: Dict = None) -> str:
    """
    Create JWT session token for user.

    Args:
        user_id: User identifier
        additional_claims: Optional additional JWT claims

    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }

    if additional_claims:
        payload.update(additional_claims)

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_session_token(authorization: Optional[str]) -> str:
    """
    Verify JWT session token and extract user_id.

    Args:
        authorization: Authorization header value (e.g., "Bearer <token>")

    Returns:
        user_id extracted from token

    Raises:
        HTTPException: If token invalid or expired
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization[7:]  # Remove "Bearer " prefix

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# User-Item mapping (in production, store in database)
_user_item_mapping: Dict[str, str] = {}


def link_user_item(user_id: str, item_id: str) -> None:
    """Link item_id to user_id (store in database in production)."""
    _user_item_mapping[user_id] = item_id


def get_user_item_id(user_id: str) -> Optional[str]:
    """Get item_id for user_id (query database in production)."""
    return _user_item_mapping.get(user_id)
```

---

## Environment Setup

**File**: `.env` (chmod 600)

```bash
# Plaid API Credentials
PLAID_CLIENT_ID=your_plaid_client_id
PLAID_SECRET=your_plaid_secret
PLAID_ENV=sandbox

# Database Encryption Key (CRITICAL - KEEP SECRET!)
# Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
DATABASE_ENCRYPTION_KEY=your_generated_key_here

# JWT Secret Key (CRITICAL - KEEP SECRET!)
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET_KEY=your_jwt_secret_key_here

# Alpaca API (Trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Generate keys**:
```bash
# Database encryption key
python -c "from cryptography.fernet import Fernet; print('DATABASE_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"

# JWT secret key
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

---

## Installation Requirements

**File**: `requirements.txt`

Add these lines:
```txt
# Security dependencies
cryptography==41.0.7
PyJWT==2.8.0
slowapi==0.1.9
```

Install:
```bash
pip install cryptography PyJWT slowapi
```

---

## Migration Script for Existing Tokens

**File**: `scripts/migrate_encrypt_tokens.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""
Migrate existing plaintext tokens to encrypted format.
RUN THIS ONCE after implementing encryption.
"""

import sqlite3
import os
from cryptography.fernet import Fernet

def migrate_tokens(db_path: str, encryption_key: bytes):
    """Encrypt all existing plaintext tokens in database."""
    cipher = Fernet(encryption_key)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all items with plaintext tokens
    cursor.execute("SELECT item_id, access_token FROM plaid_items")
    items = cursor.fetchall()

    print(f"Found {len(items)} items to migrate")

    for item_id, access_token in items:
        # Check if already encrypted (encrypted tokens are longer)
        if len(access_token) < 50:
            print(f"Encrypting token for item {item_id}...")

            # Encrypt token
            encrypted_token = cipher.encrypt(access_token.encode())

            # Update database
            cursor.execute("""
                UPDATE plaid_items SET access_token = ? WHERE item_id = ?
            """, (encrypted_token, item_id))
        else:
            print(f"Item {item_id} already encrypted, skipping")

    conn.commit()
    conn.close()

    print("✅ Migration complete!")


if __name__ == "__main__":
    # Get encryption key from environment
    key = os.getenv('DATABASE_ENCRYPTION_KEY')
    if not key:
        print("❌ DATABASE_ENCRYPTION_KEY not set")
        exit(1)

    db_path = "data/bank_accounts.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        exit(1)

    # Backup database first
    import shutil
    backup_path = f"{db_path}.backup"
    shutil.copy2(db_path, backup_path)
    print(f"✅ Database backed up to {backup_path}")

    # Run migration
    migrate_tokens(db_path, key.encode())
```

**Usage**:
```bash
# Set encryption key
export DATABASE_ENCRYPTION_KEY="your_key_here"

# Run migration
python scripts/migrate_encrypt_tokens.py
```

---

## Testing

**File**: `tests/security/test_plaid_security.py` (NEW FILE)

```python
"""Security tests for Plaid integration."""

import pytest
from fastapi.testclient import TestClient
from src.dashboard.run_server_simple import SimpleDashboardServer
from cryptography.fernet import Fernet
import os

@pytest.fixture
def client():
    server = SimpleDashboardServer()
    return TestClient(server.app)

def test_access_token_not_in_response(client):
    """Verify access token never returned to client."""
    response = client.post("/api/plaid/exchange_public_token",
                           json={"public_token": "public-sandbox-test"},
                           headers={"Authorization": "Bearer test-token"})

    data = response.json()
    assert "access_token" not in data, "❌ Access token leaked in response!"
    assert "item_id" in data, "✅ item_id should be returned instead"

def test_access_token_not_in_url(client):
    """Verify endpoints don't accept access_token in URL."""
    response = client.get("/api/bank/accounts?access_token=test-token")
    assert response.status_code == 401, "❌ Should reject URL tokens"

def test_authorization_header_required(client):
    """Verify Authorization header required."""
    response = client.get("/api/bank/accounts")
    assert response.status_code == 401, "❌ Should require auth"

def test_database_permissions():
    """Verify database file has 600 permissions."""
    db_path = "data/bank_accounts.db"
    if os.path.exists(db_path):
        import stat
        st = os.stat(db_path)
        mode = stat.S_IMODE(st.st_mode)
        assert mode == 0o600, f"❌ Database permissions {oct(mode)} should be 0o600"
    else:
        pytest.skip("Database file doesn't exist yet")

def test_token_encryption():
    """Verify tokens encrypted in database."""
    from src.finances.bank_database import BankDatabase

    key = Fernet.generate_key()
    db = BankDatabase("test.db", key)

    item_id = db.add_plaid_item("test-access-token", "Test Bank")

    # Raw database read to verify encryption
    import sqlite3
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    cursor.execute("SELECT access_token FROM plaid_items WHERE item_id = ?", (item_id,))
    raw_token = cursor.fetchone()[0]
    conn.close()

    # Verify it's encrypted (not plaintext)
    assert raw_token != b"test-access-token", "❌ Token not encrypted!"
    assert len(raw_token) > 50, "❌ Encrypted tokens should be longer"

    # Verify decryption works
    decrypted = db.get_decrypted_access_token(item_id)
    assert decrypted == "test-access-token", "❌ Decryption failed!"

    # Cleanup
    os.remove("test.db")

def test_rate_limiting(client):
    """Verify rate limiting on Plaid endpoints."""
    # Make 10 rapid requests
    responses = []
    for _ in range(10):
        resp = client.post("/api/plaid/create_link_token",
                          json={"user_id": "test"})
        responses.append(resp)

    # At least one should be rate limited
    rate_limited = [r for r in responses if r.status_code == 429]
    assert len(rate_limited) > 0, "❌ No rate limiting detected!"

def test_error_messages_sanitized(client):
    """Verify error messages don't leak details."""
    response = client.get("/api/bank/accounts",
                          headers={"Authorization": "Bearer invalid-token"})

    error_detail = response.json().get("detail", "")

    # Should be generic, not expose internals
    sensitive_terms = ["plaid", "database", "exception", "traceback", "sql"]
    for term in sensitive_terms:
        assert term.lower() not in error_detail.lower(), \
            f"❌ Error message leaked sensitive term: {term}"
```

**Run tests**:
```bash
pytest tests/security/test_plaid_security.py -v
```

---

**END OF CODE FIXES DOCUMENT**
