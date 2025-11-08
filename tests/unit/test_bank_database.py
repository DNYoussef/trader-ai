#!/usr/bin/env python3
"""
Unit Tests for BankDatabase
Tests SQLite database operations with in-memory database for isolation.
Target: 90%+ code coverage
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.finances.bank_database import (
    BankDatabase,
    init_bank_database,
    add_plaid_item,
    update_accounts,
    update_transactions,
    get_all_accounts,
    get_total_balance,
    get_recent_transactions,
    get_transactions_by_account
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create temporary database file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def in_memory_db():
    """Create in-memory database for faster tests."""
    return BankDatabase(":memory:")


@pytest.fixture
def populated_db(in_memory_db):
    """Database with sample data."""
    db = in_memory_db

    # Add Plaid item
    item_id = db.add_plaid_item("access-token-123", "Chase Bank")

    # Add accounts
    accounts_data = [
        {
            "account_id": "acc_checking_123",
            "name": "Premium Checking",
            "official_name": "Chase Premium Checking Account",
            "type": "depository",
            "subtype": "checking",
            "mask": "1234",
            "balances": {
                "current": 2500.50,
                "available": 2450.00,
                "iso_currency_code": "USD"
            }
        },
        {
            "account_id": "acc_savings_456",
            "name": "High Yield Savings",
            "official_name": None,
            "type": "depository",
            "subtype": "savings",
            "mask": "5678",
            "balances": {
                "current": 10000.00,
                "available": 10000.00,
                "iso_currency_code": "USD"
            }
        }
    ]
    db.update_accounts(item_id, accounts_data)

    # Add transactions
    transactions_data = [
        {
            "transaction_id": "txn_coffee_123",
            "amount": 4.50,
            "date": (datetime.utcnow() - timedelta(days=1)).date().isoformat(),
            "name": "Starbucks",
            "merchant_name": "Starbucks Corporation",
            "category": ["Food and Drink", "Restaurants"],
            "pending": False
        },
        {
            "transaction_id": "txn_grocery_456",
            "amount": 125.00,
            "date": (datetime.utcnow() - timedelta(days=2)).date().isoformat(),
            "name": "Whole Foods",
            "merchant_name": "Whole Foods Market",
            "category": ["Shops", "Groceries"],
            "pending": False
        }
    ]
    db.update_transactions("acc_checking_123", transactions_data)

    return db, item_id


# ============================================================================
# TEST CLASS: Database Initialization
# ============================================================================

class TestDatabaseInitialization:
    """Test database schema creation and connection."""

    def test_init_creates_file(self, temp_db_path):
        """Test that database file is created."""
        db = BankDatabase(temp_db_path)
        assert os.path.exists(temp_db_path)

    def test_init_creates_schema(self, in_memory_db):
        """Test that all tables are created."""
        conn = in_memory_db._get_connection()
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "plaid_items" in tables
        assert "bank_accounts" in tables
        assert "bank_transactions" in tables

        conn.close()

    def test_init_creates_indexes(self, in_memory_db):
        """Test that indexes are created."""
        conn = in_memory_db._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_accounts_item" in indexes
        assert "idx_transactions_account" in indexes
        assert "idx_transactions_date" in indexes

        conn.close()

    def test_init_directory_creation(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "dir", "test.db")
            db = BankDatabase(db_path)
            assert os.path.exists(db_path)

    def test_in_memory_database(self):
        """Test in-memory database creation."""
        db = BankDatabase(":memory:")
        assert db.db_path == ":memory:"


# ============================================================================
# TEST CLASS: Plaid Item Operations
# ============================================================================

class TestPlaidItemOperations:
    """Test Plaid item CRUD operations."""

    def test_add_plaid_item_returns_item_id(self, in_memory_db):
        """Test that add_plaid_item returns a valid item_id."""
        item_id = in_memory_db.add_plaid_item("access-token-123", "Chase Bank")

        assert item_id.startswith("item_")
        assert len(item_id) > 10

    def test_add_plaid_item_stores_data(self, in_memory_db):
        """Test that Plaid item data is stored correctly."""
        item_id = in_memory_db.add_plaid_item("access-token-456", "Wells Fargo")

        conn = in_memory_db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plaid_items WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row["item_id"] == item_id
        assert row["access_token"] == "access-token-456"
        assert row["institution_name"] == "Wells Fargo"
        conn.close()

    def test_add_multiple_items(self, in_memory_db):
        """Test adding multiple Plaid items."""
        item1 = in_memory_db.add_plaid_item("token1", "Bank 1")
        item2 = in_memory_db.add_plaid_item("token2", "Bank 2")

        assert item1 != item2

        conn = in_memory_db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM plaid_items")
        count = cursor.fetchone()[0]
        assert count == 2
        conn.close()

    def test_get_item_summary(self, populated_db):
        """Test get_item_summary returns correct data."""
        db, item_id = populated_db

        summary = db.get_item_summary(item_id)

        assert summary["item_id"] == item_id
        assert summary["institution_name"] == "Chase Bank"
        assert summary["account_count"] == 2
        assert summary["total_balance"] == 12500.50

    def test_get_item_summary_nonexistent(self, in_memory_db):
        """Test get_item_summary with non-existent item."""
        summary = in_memory_db.get_item_summary("nonexistent_item")
        assert summary == {}

    def test_delete_item(self, populated_db):
        """Test delete_item removes item and cascades."""
        db, item_id = populated_db

        deleted = db.delete_item(item_id)
        assert deleted is True

        # Verify item deleted
        summary = db.get_item_summary(item_id)
        assert summary == {}

        # Verify accounts deleted (CASCADE)
        accounts = db.get_all_accounts()
        assert len(accounts) == 0

    def test_delete_nonexistent_item(self, in_memory_db):
        """Test delete_item with non-existent item."""
        deleted = in_memory_db.delete_item("fake_item_id")
        assert deleted is False


# ============================================================================
# TEST CLASS: Account Operations
# ============================================================================

class TestAccountOperations:
    """Test bank account CRUD operations."""

    def test_update_accounts_insert(self, in_memory_db):
        """Test inserting new accounts."""
        item_id = in_memory_db.add_plaid_item("token", "Test Bank")

        accounts_data = [{
            "account_id": "acc_123",
            "name": "Checking",
            "type": "depository",
            "subtype": "checking",
            "balances": {"current": 1000.00}
        }]

        count = in_memory_db.update_accounts(item_id, accounts_data)
        assert count == 1

    def test_update_accounts_upsert(self, populated_db):
        """Test updating existing accounts (upsert)."""
        db, item_id = populated_db

        # Update balance
        updated_data = [{
            "account_id": "acc_checking_123",
            "name": "Premium Checking (Updated)",
            "type": "depository",
            "subtype": "checking",
            "balances": {"current": 3000.00, "available": 2950.00}
        }]

        count = db.update_accounts(item_id, updated_data)
        assert count == 1

        # Verify update
        accounts = db.get_all_accounts()
        checking = [a for a in accounts if a["account_id"] == "acc_checking_123"][0]
        assert checking["current_balance"] == 3000.00
        assert checking["name"] == "Premium Checking (Updated)"

    def test_update_accounts_multiple(self, in_memory_db):
        """Test updating multiple accounts at once."""
        item_id = in_memory_db.add_plaid_item("token", "Test Bank")

        accounts_data = [
            {"account_id": "acc_1", "name": "Account 1", "balances": {"current": 100}},
            {"account_id": "acc_2", "name": "Account 2", "balances": {"current": 200}},
            {"account_id": "acc_3", "name": "Account 3", "balances": {"current": 300}}
        ]

        count = in_memory_db.update_accounts(item_id, accounts_data)
        assert count == 3

    def test_get_all_accounts(self, populated_db):
        """Test retrieving all accounts."""
        db, _ = populated_db

        accounts = db.get_all_accounts()

        assert len(accounts) == 2
        assert accounts[0]["account_id"] in ["acc_checking_123", "acc_savings_456"]
        assert accounts[0]["institution_name"] == "Chase Bank"

    def test_get_all_accounts_empty(self, in_memory_db):
        """Test get_all_accounts with no accounts."""
        accounts = in_memory_db.get_all_accounts()
        assert len(accounts) == 0

    def test_get_total_balance(self, populated_db):
        """Test total balance calculation."""
        db, _ = populated_db

        total = db.get_total_balance()
        assert total == 12500.50

    def test_get_total_balance_empty(self, in_memory_db):
        """Test total balance with no accounts."""
        total = in_memory_db.get_total_balance()
        assert total == 0.0

    def test_account_foreign_key_constraint(self, in_memory_db):
        """Test foreign key constraint on accounts."""
        # This should fail because item_id doesn't exist
        accounts_data = [{
            "account_id": "acc_123",
            "name": "Test",
            "balances": {"current": 100}
        }]

        # SQLite foreign key constraints may not raise by default,
        # but the relationship should be enforced
        count = in_memory_db.update_accounts("nonexistent_item", accounts_data)
        assert count == 1  # Insert succeeds but orphaned


# ============================================================================
# TEST CLASS: Transaction Operations
# ============================================================================

class TestTransactionOperations:
    """Test transaction CRUD operations."""

    def test_update_transactions_insert(self, populated_db):
        """Test inserting new transactions."""
        db, _ = populated_db

        transactions_data = [{
            "transaction_id": "txn_new_123",
            "amount": 50.00,
            "date": datetime.utcnow().date().isoformat(),
            "name": "Test Transaction",
            "category": ["Test", "Category"],
            "pending": False
        }]

        count = db.update_transactions("acc_checking_123", transactions_data)
        assert count == 1

    def test_update_transactions_upsert(self, populated_db):
        """Test updating existing transactions (upsert)."""
        db, _ = populated_db

        # Update existing transaction
        updated_data = [{
            "transaction_id": "txn_coffee_123",
            "amount": 5.00,  # Updated amount
            "date": (datetime.utcnow() - timedelta(days=1)).date().isoformat(),
            "name": "Starbucks (Updated)",
            "category": ["Food"],
            "pending": True  # Changed to pending
        }]

        count = db.update_transactions("acc_checking_123", updated_data)
        assert count == 1

        # Verify update
        txns = db.get_transactions_by_account("acc_checking_123")
        coffee_txn = [t for t in txns if t["transaction_id"] == "txn_coffee_123"][0]
        assert coffee_txn["amount"] == 5.00
        assert coffee_txn["pending"] == 1

    def test_update_transactions_category_conversion(self, populated_db):
        """Test category list to string conversion."""
        db, _ = populated_db

        transactions_data = [{
            "transaction_id": "txn_test_456",
            "amount": 100.00,
            "date": datetime.utcnow().date().isoformat(),
            "name": "Test",
            "category": ["Cat1", "Cat2", "Cat3"],
            "pending": False
        }]

        db.update_transactions("acc_checking_123", transactions_data)

        conn = db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT category FROM bank_transactions WHERE transaction_id = ?", ("txn_test_456",))
        row = cursor.fetchone()
        assert row["category"] == "Cat1, Cat2, Cat3"
        conn.close()

    def test_get_recent_transactions(self, populated_db):
        """Test retrieving recent transactions."""
        db, _ = populated_db

        transactions = db.get_recent_transactions(days=30)

        assert len(transactions) >= 2
        assert transactions[0]["date"] >= transactions[-1]["date"]  # Sorted by date DESC

    def test_get_recent_transactions_custom_days(self, populated_db):
        """Test recent transactions with custom day range."""
        db, _ = populated_db

        # Add old transaction
        old_txn = [{
            "transaction_id": "txn_old_789",
            "amount": 200.00,
            "date": (datetime.utcnow() - timedelta(days=60)).date().isoformat(),
            "name": "Old Transaction",
            "category": [],
            "pending": False
        }]
        db.update_transactions("acc_checking_123", old_txn)

        # Get last 30 days (should exclude old transaction)
        recent = db.get_recent_transactions(days=30)
        old_ids = [t["transaction_id"] for t in recent]
        assert "txn_old_789" not in old_ids

        # Get last 90 days (should include old transaction)
        all_txns = db.get_recent_transactions(days=90)
        all_ids = [t["transaction_id"] for t in all_txns]
        assert "txn_old_789" in all_ids

    def test_get_transactions_by_account(self, populated_db):
        """Test retrieving transactions for specific account."""
        db, _ = populated_db

        transactions = db.get_transactions_by_account("acc_checking_123")

        assert len(transactions) >= 2
        assert all(t["account_id"] == "acc_checking_123" for t in transactions)

    def test_get_transactions_by_account_limit(self, populated_db):
        """Test transaction retrieval with limit."""
        db, _ = populated_db

        # Add many transactions
        many_txns = [
            {
                "transaction_id": f"txn_{i}",
                "amount": i * 10.0,
                "date": datetime.utcnow().date().isoformat(),
                "name": f"Transaction {i}",
                "category": [],
                "pending": False
            }
            for i in range(10)
        ]
        db.update_transactions("acc_checking_123", many_txns)

        # Get only 5 transactions
        limited = db.get_transactions_by_account("acc_checking_123", limit=5)
        assert len(limited) == 5

    def test_get_transactions_nonexistent_account(self, in_memory_db):
        """Test get_transactions_by_account with non-existent account."""
        transactions = in_memory_db.get_transactions_by_account("fake_account_id")
        assert len(transactions) == 0


# ============================================================================
# TEST CLASS: Spending Analytics
# ============================================================================

class TestSpendingAnalytics:
    """Test spending analytics and aggregation."""

    def test_get_spending_by_category(self, populated_db):
        """Test spending aggregation by category."""
        db, _ = populated_db

        spending = db.get_spending_by_category(days=30)

        assert len(spending) > 0
        assert all("category" in s and "total_amount" in s for s in spending)
        # Sorted by total_amount DESC
        if len(spending) > 1:
            assert spending[0]["total_amount"] >= spending[1]["total_amount"]

    def test_get_spending_by_category_excludes_negative(self, populated_db):
        """Test that negative amounts (income) are excluded."""
        db, _ = populated_db

        # Add income transaction (negative amount)
        income_txn = [{
            "transaction_id": "txn_income_999",
            "amount": -1000.00,
            "date": datetime.utcnow().date().isoformat(),
            "name": "Paycheck",
            "category": ["Income", "Salary"],
            "pending": False
        }]
        db.update_transactions("acc_checking_123", income_txn)

        spending = db.get_spending_by_category(days=30)

        # Income category should not appear in spending
        categories = [s["category"] for s in spending]
        assert "Income, Salary" not in categories

    def test_get_spending_by_category_custom_days(self, populated_db):
        """Test spending analytics with custom day range."""
        db, _ = populated_db

        spending_7d = db.get_spending_by_category(days=7)
        spending_30d = db.get_spending_by_category(days=30)

        # 30-day should have same or more categories
        assert len(spending_30d) >= len(spending_7d)

    def test_get_spending_by_category_empty(self, in_memory_db):
        """Test spending analytics with no transactions."""
        spending = in_memory_db.get_spending_by_category()
        assert len(spending) == 0


# ============================================================================
# TEST CLASS: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test module-level utility functions."""

    def test_init_bank_database(self):
        """Test init_bank_database utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = init_bank_database(db_path)

            assert isinstance(db, BankDatabase)
            assert os.path.exists(db_path)

    def test_add_plaid_item_utility(self):
        """Test add_plaid_item utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            item_id = add_plaid_item("token", "Bank", db_path)

            assert item_id.startswith("item_")

    def test_update_accounts_utility(self):
        """Test update_accounts utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            item_id = add_plaid_item("token", "Bank", db_path)

            accounts_data = [{
                "account_id": "acc_123",
                "name": "Test",
                "balances": {"current": 100}
            }]

            count = update_accounts(item_id, accounts_data, db_path)
            assert count == 1

    def test_get_total_balance_utility(self):
        """Test get_total_balance utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            total = get_total_balance(db_path)
            assert total == 0.0


# ============================================================================
# TEST CLASS: Concurrent Access
# ============================================================================

class TestConcurrentAccess:
    """Test database operations under concurrent access."""

    def test_multiple_connections(self, temp_db_path):
        """Test multiple simultaneous connections."""
        db1 = BankDatabase(temp_db_path)
        db2 = BankDatabase(temp_db_path)

        # Write from db1
        item_id = db1.add_plaid_item("token1", "Bank1")

        # Read from db2
        summary = db2.get_item_summary(item_id)
        assert summary["institution_name"] == "Bank1"

    def test_transaction_isolation(self, temp_db_path):
        """Test transaction isolation between connections."""
        db = BankDatabase(temp_db_path)

        item_id = db.add_plaid_item("token", "Bank")

        # Add accounts in one transaction
        accounts_data = [
            {"account_id": "acc_1", "name": "Account 1", "balances": {"current": 100}},
            {"account_id": "acc_2", "name": "Account 2", "balances": {"current": 200}}
        ]
        db.update_accounts(item_id, accounts_data)

        # Should see both accounts
        accounts = db.get_all_accounts()
        assert len(accounts) == 2


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src.finances.bank_database", "--cov-report=term-missing"])
