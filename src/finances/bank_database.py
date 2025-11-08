"""
SQLite Database Layer for Plaid Banking Data Persistence

This module provides a complete database abstraction for storing and retrieving:
- Plaid item connections (access tokens, institutions)
- Bank account details (balances, types, metadata)
- Transaction history (amounts, dates, merchants, categories)

Schema Design:
- plaid_items: Root connection records
- bank_accounts: Account-level data with foreign key to items
- bank_transactions: Transaction-level data with foreign key to accounts

All timestamps are UTC. All monetary amounts are stored as REAL (floating point).
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json


class BankDatabase:
    """
    SQLite database manager for Plaid banking data.

    Provides atomic operations for:
    - Connection management (Plaid items)
    - Account synchronization (balances, metadata)
    - Transaction storage and retrieval
    - Aggregate reporting (total balance, recent activity)
    """

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

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
        1. plaid_items: Stores Plaid connection metadata
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

        # User sessions table for JWT authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                plaid_item_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY(plaid_item_id) REFERENCES plaid_items(item_id) ON DELETE CASCADE
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

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON user_sessions(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_item
            ON user_sessions(plaid_item_id)
        """)

        conn.commit()
        conn.close()

    def add_plaid_item(self, access_token: str, institution_name: str) -> str:
        """
        Add a new Plaid item connection.

        Args:
            access_token: Plaid access token for API calls
            institution_name: Human-readable institution name (e.g., "Chase Bank")

        Returns:
            item_id: Generated unique identifier for this connection
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate unique item_id (timestamp-based)
        item_id = f"item_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        cursor.execute("""
            INSERT INTO plaid_items (item_id, access_token, institution_name)
            VALUES (?, ?, ?)
        """, (item_id, access_token, institution_name))

        conn.commit()
        conn.close()

        return item_id

    def update_accounts(self, item_id: str, accounts_data: List[Dict[str, Any]]) -> int:
        """
        Upsert account data for a Plaid item.

        Args:
            item_id: Plaid item ID
            accounts_data: List of account dictionaries from Plaid API

        Returns:
            Number of accounts updated/inserted

        Example accounts_data format:
        [
            {
                "account_id": "xyz123",
                "name": "Checking",
                "official_name": "Premium Checking Account",
                "type": "depository",
                "subtype": "checking",
                "mask": "1234",
                "balances": {
                    "current": 1500.50,
                    "available": 1500.50
                }
            }
        ]
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

        Example transactions_data format:
        [
            {
                "transaction_id": "abc123",
                "amount": 50.00,
                "date": "2025-01-15",
                "name": "Coffee Shop",
                "merchant_name": "Starbucks",
                "category": ["Food and Drink", "Restaurants", "Coffee Shop"],
                "pending": False
            }
        ]
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

    def create_session(self, session_id: str, user_id: str, plaid_item_id: Optional[str], expires_at: str) -> bool:
        """
        Create a new user session.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            plaid_item_id: Optional Plaid item ID associated with session
            expires_at: ISO timestamp when session expires

        Returns:
            True if session was created successfully
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_sessions (session_id, user_id, plaid_item_id, expires_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, plaid_item_id, expires_at))

        conn.commit()
        conn.close()

        return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information.

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary or None if not found/expired
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, user_id, plaid_item_id, created_at, expires_at
            FROM user_sessions
            WHERE session_id = ?
            AND expires_at > datetime('now')
        """, (session_id,))

        session = cursor.fetchone()
        conn.close()

        return dict(session) if session else None

    def get_access_token_by_user(self, user_id: str) -> Optional[str]:
        """
        Get Plaid access token for a user's most recent session.

        Args:
            user_id: User identifier

        Returns:
            Plaid access_token or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.access_token
            FROM user_sessions s
            JOIN plaid_items p ON s.plaid_item_id = p.item_id
            WHERE s.user_id = ?
            AND s.expires_at > datetime('now')
            ORDER BY s.created_at DESC
            LIMIT 1
        """, (user_id,))

        result = cursor.fetchone()
        conn.close()

        return result['access_token'] if result else None

    def delete_expired_sessions(self) -> int:
        """
        Delete all expired sessions.

        Returns:
            Number of sessions deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM user_sessions
            WHERE expires_at <= datetime('now')
        """)

        deleted = cursor.rowcount

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


def init_bank_database(db_path: str = "data/bank_accounts.db") -> BankDatabase:
    """
    Initialize and return a BankDatabase instance.

    Args:
        db_path: Path to SQLite database file (default: data/bank_accounts.db)

    Returns:
        Initialized BankDatabase instance
    """
    return BankDatabase(db_path)


# Convenience functions for common operations
def add_plaid_item(access_token: str, institution_name: str, db_path: str = "data/bank_accounts.db") -> str:
    """Add Plaid item and return item_id."""
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
    print("Initializing bank database...")
    db = init_bank_database("data/bank_accounts.db")

    print("\n✓ Database schema created successfully")
    print("\nAvailable functions:")
    print("  - init_bank_database()")
    print("  - add_plaid_item(access_token, institution_name)")
    print("  - update_accounts(item_id, accounts_data)")
    print("  - update_transactions(account_id, transactions_data)")
    print("  - get_all_accounts()")
    print("  - get_total_balance()")
    print("  - get_recent_transactions(days=30)")
    print("  - get_transactions_by_account(account_id)")
    print("\nDatabase location: data/bank_accounts.db")
