# Plaid Bank Account Redux Integration

Complete Redux integration for Plaid bank account state management with real-time balance updates.

## Overview

The Redux dashboard slice has been extended with comprehensive Plaid integration supporting:
- Bank account fetching and management
- Transaction history retrieval
- Plaid Link public token exchange
- Unified net worth calculation (trading portfolio + bank accounts)
- Real-time WebSocket balance updates
- Transaction analytics and categorization

## Installation

Ensure axios is installed:
```bash
npm install axios
```

## State Structure

### Extended DashboardState
```typescript
interface DashboardState {
  // Existing state...
  risk_metrics: RiskMetrics | null;
  positions: { [symbol: string]: PositionUpdate };
  connection: ConnectionStatus;
  historical_data: HistoricalData;

  // NEW: Plaid bank account state
  bankAccounts: BankAccount[];
  transactions: Transaction[];
  plaidLinked: boolean;
  plaidLoading: boolean;
  plaidError: string | null;
  lastBankSync: Date | null;
  totalBankBalance: number;
  unifiedNetWorth: number;
}
```

### Types
```typescript
interface BankAccount {
  account_id: string;
  name: string;
  official_name: string;
  type: string;           // checking, savings, credit, etc.
  subtype: string;        // specific account subtype
  mask: string;           // last 4 digits
  balance: number;
  available: number;
  currency: string;
  institution: string;
}

interface Transaction {
  transaction_id: string;
  account_id: string;
  amount: number;
  date: string;           // ISO date
  name: string;
  merchant_name: string | null;
  category: string;
  pending: boolean;
}
```

## Async Thunks (5 Total)

### 1. fetchBankAccounts()
Fetches all linked bank accounts.

```typescript
import { useDispatch } from 'react-redux';
import { fetchBankAccounts } from '@/store/dashboardSlice';

const dispatch = useDispatch();

// Fetch all bank accounts
dispatch(fetchBankAccounts())
  .unwrap()
  .then(data => {
    console.log('Bank accounts:', data.accounts);
  })
  .catch(error => {
    console.error('Failed to fetch accounts:', error);
  });
```

**API Endpoint:** `GET /api/bank/accounts`

**Response:**
```json
{
  "accounts": [
    {
      "account_id": "acc123",
      "name": "Checking",
      "balance": 5000.50,
      "available": 4800.25,
      "institution": "Chase"
    }
  ]
}
```

### 2. fetchTransactions(startDate, endDate)
Fetches transactions within a date range.

```typescript
import { fetchTransactions } from '@/store/dashboardSlice';

// Fetch last 30 days of transactions
const endDate = new Date().toISOString().split('T')[0];
const startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
  .toISOString()
  .split('T')[0];

dispatch(fetchTransactions({ startDate, endDate }))
  .unwrap()
  .then(data => {
    console.log('Transactions:', data.transactions);
  });
```

**API Endpoint:** `GET /api/bank/transactions?start_date=2024-01-01&end_date=2024-01-31`

**Response:**
```json
{
  "transactions": [
    {
      "transaction_id": "tx123",
      "account_id": "acc123",
      "amount": -45.67,
      "date": "2024-01-15",
      "name": "Amazon.com",
      "merchant_name": "Amazon",
      "category": "Shopping",
      "pending": false
    }
  ]
}
```

### 3. linkPlaidAccount(publicToken)
Exchanges Plaid Link public token for access token.

```typescript
import { linkPlaidAccount } from '@/store/dashboardSlice';

// After Plaid Link success
const publicToken = 'public-sandbox-12345678';

dispatch(linkPlaidAccount(publicToken))
  .unwrap()
  .then(data => {
    console.log('Account linked:', data.success);
  });
```

**API Endpoint:** `POST /api/plaid/exchange_public_token`

**Request:**
```json
{
  "public_token": "public-sandbox-12345678"
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "access-sandbox-abc123"
}
```

### 4. refreshBankData()
Comprehensive refresh of accounts, transactions, and net worth.

```typescript
import { refreshBankData } from '@/store/dashboardSlice';

// Refresh all bank data
dispatch(refreshBankData())
  .unwrap()
  .then(data => {
    console.log('Bank data refreshed at:', data.timestamp);
  });
```

**Workflow:**
1. Fetches all bank accounts
2. Fetches last 30 days of transactions
3. Fetches unified net worth
4. Updates `lastBankSync` timestamp

### 5. fetchUnifiedNetWorth()
Calculates combined net worth (trading + banking).

```typescript
import { fetchUnifiedNetWorth } from '@/store/dashboardSlice';

dispatch(fetchUnifiedNetWorth())
  .unwrap()
  .then(data => {
    console.log('Total net worth:', data.total_net_worth);
    console.log('Bank balance:', data.total_bank_balance);
    console.log('Portfolio value:', data.portfolio_value);
  });
```

**API Endpoint:** `GET /api/networth`

**Response:**
```json
{
  "total_net_worth": 15234.56,
  "total_bank_balance": 8000.00,
  "portfolio_value": 7234.56
}
```

## Reducers (3 Plaid-Specific)

### updateBankBalance
Real-time WebSocket balance updates.

```typescript
import { updateBankBalance } from '@/store/dashboardSlice';

// WebSocket message handler
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'bank_balance_update') {
    dispatch(updateBankBalance({
      account_id: data.account_id,
      balance: data.balance,
      available: data.available
    }));
  }
};
```

### clearPlaidState
Clear all Plaid state on disconnect.

```typescript
import { clearPlaidState } from '@/store/dashboardSlice';

// On logout or account disconnect
dispatch(clearPlaidState());
```

### setUnifiedNetWorth
Manually update unified net worth.

```typescript
import { setUnifiedNetWorth } from '@/store/dashboardSlice';

dispatch(setUnifiedNetWorth(25000.00));
```

## Selectors (10 Total)

### Basic Selectors

```typescript
import { useSelector } from 'react-redux';
import {
  selectBankAccounts,
  selectTotalBankBalance,
  selectUnifiedNetWorth,
  selectPlaidLinked,
  selectPlaidLoading,
  selectPlaidError,
  selectLastBankSync
} from '@/store/dashboardSlice';

function BankAccountDashboard() {
  const accounts = useSelector(selectBankAccounts);
  const totalBalance = useSelector(selectTotalBankBalance);
  const netWorth = useSelector(selectUnifiedNetWorth);
  const isLinked = useSelector(selectPlaidLinked);
  const loading = useSelector(selectPlaidLoading);
  const error = useSelector(selectPlaidError);
  const lastSync = useSelector(selectLastBankSync);

  return (
    <div>
      <h2>Total Bank Balance: ${totalBalance.toFixed(2)}</h2>
      <h2>Unified Net Worth: ${netWorth.toFixed(2)}</h2>
      <p>Last Sync: {lastSync?.toLocaleString()}</p>
      <p>Status: {isLinked ? 'Linked' : 'Not Linked'}</p>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      <h3>Accounts ({accounts.length})</h3>
      {accounts.map(acc => (
        <div key={acc.account_id}>
          <p>{acc.name} - ${acc.balance.toFixed(2)}</p>
        </div>
      ))}
    </div>
  );
}
```

### Advanced Selectors

#### selectRecentTransactions(limit)
```typescript
import { selectRecentTransactions } from '@/store/dashboardSlice';

function RecentTransactionsList() {
  const recent = useSelector(selectRecentTransactions(10));

  return (
    <div>
      <h3>Recent Transactions</h3>
      {recent.map(tx => (
        <div key={tx.transaction_id}>
          <p>{tx.name} - ${Math.abs(tx.amount).toFixed(2)}</p>
          <small>{tx.date}</small>
        </div>
      ))}
    </div>
  );
}
```

#### selectCombinedNetWorth
```typescript
import { selectCombinedNetWorth } from '@/store/dashboardSlice';

const combinedNetWorth = useSelector(selectCombinedNetWorth);
// Returns: portfolioValue + totalBankBalance
```

#### selectTransactionsByCategory
```typescript
import { selectTransactionsByCategory } from '@/store/dashboardSlice';

function SpendingBreakdown() {
  const categoryTotals = useSelector(selectTransactionsByCategory);

  return (
    <div>
      <h3>Spending by Category</h3>
      {categoryTotals.map(({ category, amount }) => (
        <div key={category}>
          <p>{category}: ${amount.toFixed(2)}</p>
        </div>
      ))}
    </div>
  );
}
```

#### selectTransactionsByAccount(accountId)
```typescript
import { selectTransactionsByAccount } from '@/store/dashboardSlice';

function AccountTransactions({ accountId }) {
  const transactions = useSelector(selectTransactionsByAccount(accountId));

  return (
    <div>
      <h3>Account Transactions ({transactions.length})</h3>
      {transactions.map(tx => (
        <div key={tx.transaction_id}>
          <p>{tx.name} - ${Math.abs(tx.amount).toFixed(2)}</p>
        </div>
      ))}
    </div>
  );
}
```

## Complete Usage Example

```typescript
import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  fetchBankAccounts,
  refreshBankData,
  selectBankAccounts,
  selectTotalBankBalance,
  selectUnifiedNetWorth,
  selectRecentTransactions,
  selectPlaidLinked,
  selectPlaidLoading,
  selectPlaidError,
  updateBankBalance
} from '@/store/dashboardSlice';

function PlaidDashboard() {
  const dispatch = useDispatch();

  // Selectors
  const accounts = useSelector(selectBankAccounts);
  const totalBalance = useSelector(selectTotalBankBalance);
  const netWorth = useSelector(selectUnifiedNetWorth);
  const recentTx = useSelector(selectRecentTransactions(5));
  const isLinked = useSelector(selectPlaidLinked);
  const loading = useSelector(selectPlaidLoading);
  const error = useSelector(selectPlaidError);

  // Initial load
  useEffect(() => {
    if (isLinked) {
      dispatch(fetchBankAccounts());
    }
  }, [isLinked, dispatch]);

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/bank-updates');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'bank_balance_update') {
        dispatch(updateBankBalance({
          account_id: data.account_id,
          balance: data.balance,
          available: data.available
        }));
      }
    };

    return () => ws.close();
  }, [dispatch]);

  // Manual refresh
  const handleRefresh = () => {
    dispatch(refreshBankData());
  };

  if (loading) return <div>Loading bank data...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h1>Plaid Banking Dashboard</h1>

      <div className="summary">
        <h2>Total Bank Balance: ${totalBalance.toFixed(2)}</h2>
        <h2>Unified Net Worth: ${netWorth.toFixed(2)}</h2>
        <button onClick={handleRefresh}>Refresh Data</button>
      </div>

      <div className="accounts">
        <h3>Bank Accounts ({accounts.length})</h3>
        {accounts.map(acc => (
          <div key={acc.account_id} className="account-card">
            <h4>{acc.name}</h4>
            <p>{acc.institution}</p>
            <p>Balance: ${acc.balance.toFixed(2)}</p>
            <p>Available: ${acc.available.toFixed(2)}</p>
            <p>Type: {acc.type} ({acc.subtype})</p>
            <small>****{acc.mask}</small>
          </div>
        ))}
      </div>

      <div className="transactions">
        <h3>Recent Transactions</h3>
        {recentTx.map(tx => (
          <div key={tx.transaction_id} className="transaction-row">
            <span>{tx.name}</span>
            <span>{tx.category}</span>
            <span className={tx.amount < 0 ? 'debit' : 'credit'}>
              ${Math.abs(tx.amount).toFixed(2)}
            </span>
            <span>{tx.date}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PlaidDashboard;
```

## WebSocket Integration

### Real-Time Balance Updates (Every 5 Minutes)

```typescript
// Backend sends updates via WebSocket
const handleWebSocketMessage = (data: any) => {
  switch (data.type) {
    case 'bank_balance_update':
      dispatch(updateBankBalance({
        account_id: data.account_id,
        balance: data.balance,
        available: data.available
      }));
      break;

    case 'transaction_posted':
      // Refresh transactions
      const endDate = new Date().toISOString().split('T')[0];
      const startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
        .toISOString()
        .split('T')[0];
      dispatch(fetchTransactions({ startDate, endDate }));
      break;
  }
};
```

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/bank/accounts` | GET | Fetch all linked bank accounts |
| `/api/bank/transactions` | GET | Fetch transactions with date filter |
| `/api/plaid/exchange_public_token` | POST | Link Plaid account |
| `/api/networth` | GET | Get unified net worth |

## Performance Optimizations

1. **Batch Updates**: Use `refreshBankData()` instead of individual calls
2. **Selective Re-renders**: Use specific selectors to avoid unnecessary renders
3. **WebSocket Updates**: Real-time balance updates without polling
4. **Memoized Selectors**: Transaction analytics computed only when data changes
5. **Date Range Filtering**: Fetch only necessary transaction history

## Error Handling

```typescript
dispatch(fetchBankAccounts())
  .unwrap()
  .catch(error => {
    // Error automatically stored in state.plaidError
    console.error('Bank account fetch failed:', error);
    // Show user-friendly error message
    toast.error('Failed to load bank accounts');
  });
```

## Testing

```typescript
// Mock data for testing
const mockBankAccounts: BankAccount[] = [
  {
    account_id: 'acc1',
    name: 'Checking',
    official_name: 'Premier Checking Account',
    type: 'depository',
    subtype: 'checking',
    mask: '1234',
    balance: 5000.00,
    available: 4800.00,
    currency: 'USD',
    institution: 'Chase'
  }
];

const mockTransactions: Transaction[] = [
  {
    transaction_id: 'tx1',
    account_id: 'acc1',
    amount: -45.67,
    date: '2024-01-15',
    name: 'Amazon.com',
    merchant_name: 'Amazon',
    category: 'Shopping',
    pending: false
  }
];
```

## Migration Guide

If you're upgrading from a previous version:

1. **Install axios**: `npm install axios`
2. **Update imports**: Import new thunks and selectors
3. **Update components**: Replace manual API calls with thunks
4. **Add WebSocket handler**: Integrate `updateBankBalance` action
5. **Test thoroughly**: Verify all bank account features work

## Best Practices

1. **Always use thunks** for API calls (never direct axios in components)
2. **Handle loading states** with `selectPlaidLoading`
3. **Display errors** from `selectPlaidError`
4. **Refresh regularly** using `refreshBankData()` or WebSocket
5. **Protect sensitive data** - never log access tokens
6. **Test with Plaid Sandbox** before production

## Support

For issues or questions:
- Check API endpoint responses in Network tab
- Verify Redux state with Redux DevTools
- Ensure backend is running on `http://localhost:8000`
- Test with Plaid Sandbox credentials first
