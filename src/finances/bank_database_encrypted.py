"""
SQLite Database Layer for Plaid Banking Data Persistence with Token Encryption

This module provides a complete database abstraction for storing and retrieving:
- Plaid item connections (access tokens ENCRYPTED, institutions)
- Bank account details (balances, types, metadata)
- Transaction history (amounts, dates, merchants, categories)

Security:
- All Plaid access tokens are encrypted using Fernet before storage
- Automatic encryption/decryption on write/read operations
- Graceful fallback if encryption key not available (with warnings)

Schema Design:
- plaid_items: Root connection records (access_token is ENCRYPTED)
- bank_accounts: Account-level data with foreign key to items
- bank_transactions: Transaction-level data with foreign key to accounts

All timestamps are UTC. All monetary amounts are stored as REAL (floating point).
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

# Import token encryption
from src.security.token_encryption import TokenEncryption, TokenEncryptionError

logger = logging.getLogger(__name__)


class BankDatabase:
    """
    SQLite database manager for Plaid banking data with encryption support.

    Provides atomic operations for:
    - Connection management (Plaid items with ENCRYPTED access tokens)
    - Account synchronization (balances, metadata)
    - Transaction storage and retrieval
    - Aggregate reporting (total balance, recent activity)

    Security Features:
    - Automatic token encryption on write
    - Automatic token decryption on read
    - Graceful degradation if encryption unavailable
    """

    def __init__(self, db_path: str, encryption_key: Optional[str] = None):
        """
        Initialize database connection with encryption support.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
            encryption_key: Optional encryption key (for testing).
                          If None, loads from DATABASE_ENCRYPTION_KEY env var.
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize token encryption
        try:
            self.encryptor = TokenEncryption(encryption_key=encryption_key)
            self.encryption_enabled = True
            logger.info("Token encryption initialized for database")
        except TokenEncryptionError as e:
            logger.error(f"Token encryption REQUIRED but not available: {e}")
            logger.error("Set DATABASE_ENCRYPTION_KEY in .env file. Generate key: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""); raise RuntimeError(f"Token encryption is REQUIRED. Set DATABASE_ENCRYPTION_KEY. Error: {e}") from e

        # Initialize database schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory for dict-like access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """
        Create database tables if they don't exist.

        Tables:
        1. plaid_items: Stores Plaid connection metadata (tokens ENCRYPTED)
        2. bank_accounts: Stores account details linked to items
        3. bank_transactions: Stores transaction history linked to accounts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Plaid items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plaid_items (
                item_id TEXT PRIMARY KEY,
                access_token TEXT NOT NULL,
                institution_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Bank accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bank_accounts (
                account_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                name TEXT,
                official_name TEXT,
                type TEXT,
                subtype TEXT,
                mask TEXT,
                current_balance REAL,
                available_balance REAL,
                currency TEXT DEFAULT 'USD',
                last_synced TIMESTAMP,
                FOREIGN KEY(item_id) REFERENCES plaid_items(item_id) ON DELETE CASCADE
            )
        """)

        # Bank transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bank_transactions (
                transaction_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                amount REAL NOT NULL,
                date DATE NOT NULL,
                name TEXT,
                merchant_name TEXT,
                category TEXT,
                pending INTEGER DEFAULT 0,
                last_synced TIMESTAMP,
                FOREIGN KEY(account_id) REFERENCES bank_accounts(account_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_accounts_item
            ON bank_accounts(item_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_account
            ON bank_transactions(account_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_date
            ON bank_transactions(date)
        """)

        conn.commit()
        conn.close()

    def _encrypt_token(self, token: str) -> str:
        """
        Encrypt token if encryption is enabled.

        Args:
            token: Plaintext token

        Returns:
            Encrypted token or plaintext if encryption disabled
        """
        if self.encryption_enabled and self.encryptor:
            try:
                return self.encryptor.encrypt_token(token)
            except TokenEncryptionError as e:
                logger.error(f"Token encryption failed: {e}")
                raise
        else:
            logger.warning("Storing plaintext token (encryption not enabled)")
            return token

    def _decrypt_token(self, encrypted_token: str) -> str:
        """
        Decrypt token if encryption is enabled.

        Args:
            encrypted_token: Encrypted token from database

        Returns:
            Decrypted token or encrypted_token if encryption disabled
        """
        if self.encryption_enabled and self.encryptor:
            try:
                # Check if token is actually encrypted
                if TokenEncryption.is_encrypted(encrypted_token):
                    return self.encryptor.decrypt_token(encrypted_token)
                else:
                    logger.warning("Token appears to be plaintext, returning as-is")
                    return encrypted_token
            except TokenEncryptionError as e:
                logger.error(f"Token decryption failed: {e}")
                # Return encrypted token to avoid breaking application
                logger.warning("Returning encrypted token (decryption failed)")
                return encrypted_token
        else:
            return encrypted_token

    def add_plaid_item(self, access_token: str, institution_name: str) -> str:
        """
        Add a new Plaid item connection with encrypted access token.

        Args:
            access_token: Plaid access token for API calls (will be encrypted)
            institution_name: Human-readable institution name (e.g., "Chase Bank")

        Returns:
            item_id: Generated unique identifier for this connection

        Raises:
            TokenEncryptionError: If encryption fails
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate unique item_id (timestamp-based)
        item_id = f"item_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        # Encrypt access token before storage
        encrypted_token = self._encrypt_token(access_token)

        if self.encryption_enabled:
            logger.info(f"Access token encrypted for item {item_id}")
        else:
            logger.warning(f"Storing plaintext token for item {item_id}")

        cursor.execute("""
            INSERT INTO plaid_items (item_id, access_token, institution_name)
            VALUES (?, ?, ?)
        """, (item_id, encrypted_token, institution_name))

        conn.commit()
        conn.close()

        return item_id

    def get_access_token(self, item_id: str) -> Optional[str]:
        """
        Retrieve and decrypt access token for a Plaid item.

        Args:
            item_id: Plaid item ID

        Returns:
            Decrypted access token or None if item not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT access_token FROM plaid_items
            WHERE item_id = ?
        """, (item_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        encrypted_token = result['access_token']
        decrypted_token = self._decrypt_token(encrypted_token)

        if self.encryption_enabled:
            logger.debug(f"Access token decrypted for item {item_id}")

        return decrypted_token

    def update_accounts(self, item_id: str, accounts_data: List[Dict[str, Any]]) -> int:
        """
        Upsert account data for a Plaid item.

        Args:
            item_id: Plaid item ID
            accounts_data: List of account dictionaries from Plaid API

        Returns:
            Number of accounts updated/inserted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for account in accounts_data:
            balances = account.get('balances', {})

            cursor.execute("""
                INSERT INTO bank_accounts (
                    account_id, item_id, name, official_name, type, subtype,
                    mask, current_balance, available_balance, currency, last_synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    name = excluded.name,
                    official_name = excluded.official_name,
                    type = excluded.type,
                    subtype = excluded.subtype,
                    mask = excluded.mask,
                    current_balance = excluded.current_balance,
                    available_balance = excluded.available_balance,
                    currency = excluded.currency,
                    last_synced = excluded.last_synced
            """, (
                account['account_id'],
                item_id,
                account.get('name'),
                account.get('official_name'),
                account.get('type'),
                account.get('subtype'),
                account.get('mask'),
                balances.get('current'),
                balances.get('available'),
                balances.get('iso_currency_code', 'USD'),
                datetime.utcnow().isoformat()
            ))
            count += 1

        conn.commit()
        conn.close()

        return count

    def update_transactions(self, account_id: str, transactions_data: List[Dict[str, Any]]) -> int:
        """
        Upsert transaction data for an account.

        Args:
            account_id: Account ID
            transactions_data: List of transaction dictionaries from Plaid API

        Returns:
            Number of transactions updated/inserted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for txn in transactions_data:
            # Convert category list to string
            category = ', '.join(txn.get('category', [])) if txn.get('category') else None

            cursor.execute("""
                INSERT INTO bank_transactions (
                    transaction_id, account_id, amount, date, name,
                    merchant_name, category, pending, last_synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    amount = excluded.amount,
                    date = excluded.date,
                    name = excluded.name,
                    merchant_name = excluded.merchant_name,
                    category = excluded.category,
                    pending = excluded.pending,
                    last_synced = excluded.last_synced
            """, (
                txn['transaction_id'],
                account_id,
                txn['amount'],
                txn['date'],
                txn.get('name'),
                txn.get('merchant_name'),
                category,
                1 if txn.get('pending') else 0,
                datetime.utcnow().isoformat()
            ))
            count += 1

        conn.commit()
        conn.close()

        return count

    def get_all_accounts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all accounts with current balances.

        Returns:
            List of account dictionaries with balance information
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                a.account_id,
                a.item_id,
                a.name,
                a.official_name,
                a.type,
                a.subtype,
                a.mask,
                a.current_balance,
                a.available_balance,
                a.currency,
                a.last_synced,
                i.institution_name
            FROM bank_accounts a
            JOIN plaid_items i ON a.item_id = i.item_id
            ORDER BY a.current_balance DESC
        """)

        accounts = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return accounts

    def get_total_balance(self) -> float:
        """
        Calculate total balance across all accounts.

        Returns:
            Sum of all current balances
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COALESCE(SUM(current_balance), 0.0) as total
            FROM bank_accounts
        """)

        result = cursor.fetchone()
        total = result['total'] if result else 0.0

        conn.close()
        return total

    def get_recent_transactions(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve transactions from the last N days.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of transaction dictionaries sorted by date (newest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

        cursor.execute("""
            SELECT
                t.transaction_id,
                t.account_id,
                t.amount,
                t.date,
                t.name,
                t.merchant_name,
                t.category,
                t.pending,
                t.last_synced,
                a.name as account_name,
                a.mask as account_mask
            FROM bank_transactions t
            JOIN bank_accounts a ON t.account_id = a.account_id
            WHERE t.date >= ?
            ORDER BY t.date DESC, t.last_synced DESC
        """, (cutoff_date,))

        transactions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return transactions

    def get_transactions_by_account(self, account_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve transactions for a specific account.

        Args:
            account_id: Account ID to filter by
            limit: Maximum number of transactions to return (default: 100)

        Returns:
            List of transaction dictionaries sorted by date (newest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                transaction_id,
                account_id,
                amount,
                date,
                name,
                merchant_name,
                category,
                pending,
                last_synced
            FROM bank_transactions
            WHERE account_id = ?
            ORDER BY date DESC, last_synced DESC
            LIMIT ?
        """, (account_id, limit))

        transactions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return transactions

    def get_item_summary(self, item_id: str) -> Dict[str, Any]:
        """
        Get summary information for a Plaid item.

        Args:
            item_id: Plaid item ID

        Returns:
            Dictionary with item details, account count, and total balance
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get item details
        cursor.execute("""
            SELECT item_id, institution_name, created_at, updated_at
            FROM plaid_items
            WHERE item_id = ?
        """, (item_id,))

        item = cursor.fetchone()
        if not item:
            conn.close()
            return {}

        item_dict = dict(item)

        # Get account count and total balance
        cursor.execute("""
            SELECT
                COUNT(*) as account_count,
                COALESCE(SUM(current_balance), 0.0) as total_balance
            FROM bank_accounts
            WHERE item_id = ?
        """, (item_id,))

        stats = cursor.fetchone()
        item_dict['account_count'] = stats['account_count']
        item_dict['total_balance'] = stats['total_balance']

        conn.close()
        return item_dict

    def delete_item(self, item_id: str) -> bool:
        """
        Delete a Plaid item and all associated accounts/transactions.

        Args:
            item_id: Plaid item ID to delete

        Returns:
            True if item was deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM plaid_items WHERE item_id = ?", (item_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def get_spending_by_category(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Aggregate spending by category for the last N days.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of category/amount pairs sorted by spending (highest first)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

        cursor.execute("""
            SELECT
                category,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM bank_transactions
            WHERE date >= ?
            AND category IS NOT NULL
            AND amount > 0
            GROUP BY category
            ORDER BY total_amount DESC
        """, (cutoff_date,))

        categories = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return categories


def init_bank_database(db_path: str = "data/bank_accounts.db", encryption_key: Optional[str] = None) -> BankDatabase:
    """
    Initialize and return a BankDatabase instance with encryption.

    Args:
        db_path: Path to SQLite database file (default: data/bank_accounts.db)
        encryption_key: Optional encryption key (for testing)

    Returns:
        Initialized BankDatabase instance
    """
    return BankDatabase(db_path, encryption_key=encryption_key)


# Convenience functions for common operations
def add_plaid_item(access_token: str, institution_name: str, db_path: str = "data/bank_accounts.db") -> str:
    """Add Plaid item with encrypted token and return item_id."""
    db = init_bank_database(db_path)
    return db.add_plaid_item(access_token, institution_name)


def update_accounts(item_id: str, accounts_data: List[Dict], db_path: str = "data/bank_accounts.db") -> int:
    """Update accounts for an item."""
    db = init_bank_database(db_path)
    return db.update_accounts(item_id, accounts_data)


def update_transactions(account_id: str, transactions_data: List[Dict], db_path: str = "data/bank_accounts.db") -> int:
    """Update transactions for an account."""
    db = init_bank_database(db_path)
    return db.update_transactions(account_id, transactions_data)


def get_all_accounts(db_path: str = "data/bank_accounts.db") -> List[Dict]:
    """Get all accounts with balances."""
    db = init_bank_database(db_path)
    return db.get_all_accounts()


def get_total_balance(db_path: str = "data/bank_accounts.db") -> float:
    """Get total balance across all accounts."""
    db = init_bank_database(db_path)
    return db.get_total_balance()


def get_recent_transactions(days: int = 30, db_path: str = "data/bank_accounts.db") -> List[Dict]:
    """Get transactions from last N days."""
    db = init_bank_database(db_path)
    return db.get_recent_transactions(days)


def get_transactions_by_account(account_id: str, limit: int = 100, db_path: str = "data/bank_accounts.db") -> List[Dict]:
    """Get transactions for a specific account."""
    db = init_bank_database(db_path)
    return db.get_transactions_by_account(account_id, limit)


if __name__ == "__main__":
    # Demo usage
    print("Initializing bank database with encryption...")
    db = init_bank_database("data/bank_accounts.db")

    print("\n✓ Database schema created successfully")
    print(f"✓ Token encryption: {'ENABLED' if db.encryption_enabled else 'DISABLED'}")

    print("\nAvailable functions:")
    print("  - init_bank_database()")
    print("  - add_plaid_item(access_token, institution_name)  [ENCRYPTS TOKEN]")
    print("  - update_accounts(item_id, accounts_data)")
    print("  - update_transactions(account_id, transactions_data)")
    print("  - get_all_accounts()")
    print("  - get_total_balance()")
    print("  - get_recent_transactions(days=30)")
    print("  - get_transactions_by_account(account_id)")
    print("\nDatabase location: data/bank_accounts.db")
    print("Encryption: Set DATABASE_ENCRYPTION_KEY environment variable")
