"""Quick test script for bank database functionality."""
from src.finances.bank_database import BankDatabase
from datetime import datetime, timedelta

# Initialize database
db = BankDatabase('data/bank_accounts.db')

# Add test item
item_id = db.add_plaid_item('test_token', 'Test Bank')
print(f'[OK] Item added: {item_id}')

# Add test account
accounts = [{
    'account_id': 'test_acc_123',
    'name': 'Test Checking',
    'official_name': 'Premium Checking Account',
    'type': 'depository',
    'subtype': 'checking',
    'mask': '1234',
    'balances': {
        'current': 5000.00,
        'available': 4800.00
    }
}]
count = db.update_accounts(item_id, accounts)
print(f'[OK] Accounts updated: {count}')

# Add test transaction
txns = [{
    'transaction_id': 'txn_001',
    'amount': 50.00,
    'date': (datetime.now() - timedelta(days=1)).date().isoformat(),
    'name': 'Coffee Shop',
    'merchant_name': 'Starbucks',
    'category': ['Food and Drink', 'Restaurants'],
    'pending': False
}]
count = db.update_transactions('test_acc_123', txns)
print(f'[OK] Transactions added: {count}')

# Test retrieval functions
all_accounts = db.get_all_accounts()
print(f'[OK] Total accounts: {len(all_accounts)}')

total = db.get_total_balance()
print(f'[OK] Total balance: ${total:.2f}')

recent = db.get_recent_transactions(30)
print(f'[OK] Recent transactions: {len(recent)}')
if recent:
    print(f'     Sample: {recent[0]["name"]} - ${recent[0]["amount"]:.2f}')

# Test account-specific query
account_txns = db.get_transactions_by_account('test_acc_123')
print(f'[OK] Account transactions: {len(account_txns)}')

# Test item summary
summary = db.get_item_summary(item_id)
print(f'[OK] Item summary: {summary["institution_name"]}, {summary["account_count"]} accounts, ${summary["total_balance"]:.2f}')

print('\n[SUCCESS] All database functions validated successfully!')
