"""
SQLite Database Layer for Plaid Banking Data Persistence

IMPORTANT: This module now re-exports from bank_database_encrypted.py to ensure
all Plaid access tokens are encrypted at rest (TRD-005 security fix).

This module provides a complete database abstraction for storing and retrieving:
- Plaid item connections (access tokens ENCRYPTED, institutions)
- Bank account details (balances, types, metadata)
- Transaction history (amounts, dates, merchants, categories)

Security:
- All Plaid access tokens are encrypted using Fernet before storage
- Automatic encryption/decryption on write/read operations
- Requires DATABASE_ENCRYPTION_KEY environment variable

ISS-025: Database Choice - SQLite
Rationale: SQLite is appropriate for this single-user trading application:
- No concurrent write scaling needed (single trader)
- Serverless deployment (no separate DB process)
- ACID-compliant for financial data integrity
- File-based portability for backup/restore

Schema Design:
- plaid_items: Root connection records (access_token is ENCRYPTED)
- bank_accounts: Account-level data with foreign key to items
- bank_transactions: Transaction-level data with foreign key to accounts

All timestamps are UTC. All monetary amounts are stored as REAL (floating point).
"""

# TRD-005: Re-export encrypted database implementation for backward compatibility
# All imports from this module now use encrypted token storage
from src.finances.bank_database_encrypted import (
    BankDatabase,
    init_bank_database,
    add_plaid_item,
    update_accounts,
    update_transactions,
    get_all_accounts,
    get_total_balance,
    get_recent_transactions,
    get_transactions_by_account,
)

__all__ = [
    'BankDatabase',
    'init_bank_database',
    'add_plaid_item',
    'update_accounts',
    'update_transactions',
    'get_all_accounts',
    'get_total_balance',
    'get_recent_transactions',
    'get_transactions_by_account',
]

# NOTE: The original plaintext implementation has been replaced.
# All existing code importing from this module will automatically
# use encrypted token storage without any changes needed.
#
# To generate an encryption key:
#   python scripts/security/generate_encryption_key.py
#
# Set in your environment:
#   DATABASE_ENCRYPTION_KEY=your_generated_key_here


if __name__ == "__main__":
    # Demo usage
    print("Initializing bank database with encryption...")
    db = init_bank_database("data/bank_accounts.db")

    print("\n[OK] Database schema created successfully")
    print(f"[OK] Token encryption: {'ENABLED' if db.encryption_enabled else 'DISABLED'}")

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
