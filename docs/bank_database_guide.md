# Bank Database Module - Usage Guide

## Overview

Complete SQLite database layer for persisting Plaid banking data with 10+ functions for managing items, accounts, and transactions.

**Location**: `src/finances/bank_database.py`
**Database**: `data/bank_accounts.db` (configured in `config/config.json`)

## Schema

### Tables

1. **plaid_items** - Root connection records
   - `item_id` (PK): Unique item identifier
   - `access_token`: Plaid API access token
   - `institution_name`: Bank/institution name
   - `created_at`: Creation timestamp
   - `updated_at`: Last update timestamp

2. **bank_accounts** - Account-level data
   - `account_id` (PK): Unique account identifier
   - `item_id` (FK): Links to plaid_items
   - `name`: Account name (e.g., "Checking")
   - `official_name`: Full official name
   - `type`: Account type (depository, credit, loan, etc.)
   - `subtype`: Account subtype (checking, savings, credit card, etc.)
   - `mask`: Last 4 digits
   - `current_balance`: Current balance
   - `available_balance`: Available balance
   - `currency`: Currency code (default: USD)
   - `last_synced`: Last sync timestamp

3. **bank_transactions** - Transaction history
   - `transaction_id` (PK): Unique transaction identifier
   - `account_id` (FK): Links to bank_accounts
   - `amount`: Transaction amount
   - `date`: Transaction date
   - `name`: Transaction name/description
   - `merchant_name`: Merchant name
   - `category`: Category string (comma-separated)
   - `pending`: Pending flag (0 or 1)
   - `last_synced`: Last sync timestamp

## Core Functions

### 1. Initialize Database

```python
from src.finances.bank_database import init_bank_database

# Default location (data/bank_accounts.db)
db = init_bank_database()

# Custom location
db = init_bank_database("custom/path/database.db")
```

### 2. Add Plaid Item

```python
item_id = db.add_plaid_item(
    access_token="access-sandbox-xxx",
    institution_name="Chase Bank"
)
# Returns: "item_20251107161532780835"
```

### 3. Update Accounts

```python
accounts_data = [
    {
        "account_id": "xyz123",
        "name": "Checking",
        "official_name": "Premium Checking Account",
        "type": "depository",
        "subtype": "checking",
        "mask": "1234",
        "balances": {
            "current": 1500.50,
            "available": 1500.50,
            "iso_currency_code": "USD"
        }
    }
]

count = db.update_accounts(item_id, accounts_data)
# Returns: 1 (number of accounts updated)
```

### 4. Update Transactions

```python
transactions_data = [
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

count = db.update_transactions(account_id, transactions_data)
# Returns: 1 (number of transactions updated)
```

### 5. Get All Accounts

```python
accounts = db.get_all_accounts()
# Returns: [
#     {
#         "account_id": "xyz123",
#         "item_id": "item_xxx",
#         "name": "Checking",
#         "official_name": "Premium Checking Account",
#         "type": "depository",
#         "subtype": "checking",
#         "mask": "1234",
#         "current_balance": 1500.50,
#         "available_balance": 1500.50,
#         "currency": "USD",
#         "last_synced": "2025-01-15T10:30:00.000000",
#         "institution_name": "Chase Bank"
#     }
# ]
```

### 6. Get Total Balance

```python
total = db.get_total_balance()
# Returns: 1500.50 (sum of all account balances)
```

### 7. Get Recent Transactions

```python
# Last 30 days (default)
recent = db.get_recent_transactions(days=30)

# Last 7 days
recent = db.get_recent_transactions(days=7)

# Returns: [
#     {
#         "transaction_id": "abc123",
#         "account_id": "xyz123",
#         "amount": 50.00,
#         "date": "2025-01-15",
#         "name": "Coffee Shop",
#         "merchant_name": "Starbucks",
#         "category": "Food and Drink, Restaurants, Coffee Shop",
#         "pending": 0,
#         "last_synced": "2025-01-15T10:30:00.000000",
#         "account_name": "Checking",
#         "account_mask": "1234"
#     }
# ]
```

### 8. Get Transactions by Account

```python
# Get up to 100 transactions (default)
transactions = db.get_transactions_by_account(account_id)

# Get specific number
transactions = db.get_transactions_by_account(account_id, limit=50)

# Returns: Same format as get_recent_transactions
```

### 9. Get Item Summary

```python
summary = db.get_item_summary(item_id)
# Returns: {
#     "item_id": "item_xxx",
#     "institution_name": "Chase Bank",
#     "created_at": "2025-01-15T10:00:00.000000",
#     "updated_at": "2025-01-15T10:30:00.000000",
#     "account_count": 3,
#     "total_balance": 5000.00
# }
```

### 10. Delete Item

```python
deleted = db.delete_item(item_id)
# Returns: True if deleted, False if not found
# Cascades deletion to all accounts and transactions
```

### 11. Get Spending by Category

```python
spending = db.get_spending_by_category(days=30)
# Returns: [
#     {
#         "category": "Food and Drink, Restaurants",
#         "transaction_count": 15,
#         "total_amount": 450.00,
#         "avg_amount": 30.00
#     }
# ]
```

## Convenience Functions

Shorthand functions that create database instance automatically:

```python
from src.finances.bank_database import (
    add_plaid_item,
    update_accounts,
    update_transactions,
    get_all_accounts,
    get_total_balance,
    get_recent_transactions,
    get_transactions_by_account
)

# Use directly without creating BankDatabase instance
item_id = add_plaid_item("token", "Bank Name")
accounts = get_all_accounts()
balance = get_total_balance()
```

## Integration with Plaid API

### Complete Sync Workflow

```python
from plaid import Client
from src.finances.bank_database import init_bank_database

# Initialize
plaid_client = Client(client_id="xxx", secret="xxx", environment="sandbox")
db = init_bank_database()

# 1. Add Plaid item
item_id = db.add_plaid_item(access_token, "Chase Bank")

# 2. Fetch and store accounts
accounts_response = plaid_client.Accounts.get(access_token)
db.update_accounts(item_id, accounts_response['accounts'])

# 3. Fetch and store transactions for each account
for account in accounts_response['accounts']:
    txns_response = plaid_client.Transactions.get(
        access_token,
        start_date="2024-01-01",
        end_date="2025-01-15",
        account_ids=[account['account_id']]
    )
    db.update_transactions(account['account_id'], txns_response['transactions'])

# 4. Query aggregated data
total = db.get_total_balance()
recent = db.get_recent_transactions(30)
spending = db.get_spending_by_category(30)
```

## Performance Features

- **Upsert Operations**: All updates use INSERT ... ON CONFLICT DO UPDATE for efficiency
- **Indexes**: Optimized indexes on foreign keys and date columns
- **Batch Operations**: All update functions support batch processing
- **Row Factory**: Dict-like access to query results
- **Cascade Deletes**: Foreign key constraints with cascade deletion

## Configuration

Add to `config/config.json`:

```json
{
  "bank_database_path": "data/bank_accounts.db"
}
```

## Testing

Run the module directly to verify schema creation:

```bash
python src/finances/bank_database.py
```

Output:
```
Initializing bank database...

✓ Database schema created successfully

Available functions:
  - init_bank_database()
  - add_plaid_item(access_token, institution_name)
  - update_accounts(item_id, accounts_data)
  - update_transactions(account_id, transactions_data)
  - get_all_accounts()
  - get_total_balance()
  - get_recent_transactions(days=30)
  - get_transactions_by_account(account_id)

Database location: data/bank_accounts.db
```

## Notes

- All timestamps stored in UTC
- All monetary amounts stored as REAL (floating point)
- Categories stored as comma-separated strings
- Pending transactions stored as INTEGER (0 = false, 1 = true)
- Database auto-creates on first use
- Thread-safe (each operation gets own connection)
- Cascading deletes prevent orphaned records
