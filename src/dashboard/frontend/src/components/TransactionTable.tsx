import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { format } from 'date-fns';

export interface Transaction {
  transaction_id: string;
  date: string;
  merchant: string;
  category: string[];
  amount: number;
  account_id?: string;
  account_name?: string;
  pending?: boolean;
  payment_channel?: string;
}

interface TransactionTableProps {
  transactions: Transaction[];
  onRowClick?: (transaction: Transaction) => void;
  pageSize?: number;
  className?: string;
}

type SortField = 'date' | 'merchant' | 'amount';
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

// Export to CSV function
const exportToCSV = (transactions: Transaction[], filename: string = 'transactions.csv') => {
  const headers = ['Date', 'Merchant', 'Category', 'Amount', 'Account'];
  const rows = transactions.map(t => [
    t.date,
    t.merchant,
    t.category.join('; '),
    t.amount.toString(),
    t.account_name || t.account_id || '',
  ]);

  const csv = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
};

// Table header with sorting
const TableHeader: React.FC<{
  label: string;
  field?: SortField;
  sortConfig: SortConfig | null;
  onSort: (field: SortField) => void;
  className?: string;
}> = ({ label, field, sortConfig, onSort, className }) => {
  const isSorted = field && sortConfig?.field === field;
  const direction = isSorted ? sortConfig.direction : null;

  return (
    <th
      className={clsx(
        'px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider',
        field && 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors',
        className
      )}
      onClick={field ? () => onSort(field) : undefined}
    >
      <div className="flex items-center space-x-1">
        <span>{label}</span>
        {field && (
          <div className="flex flex-col">
            <svg
              className={clsx(
                'w-3 h-3 -mb-1',
                direction === 'asc' ? 'text-primary-500' : 'text-gray-300 dark:text-gray-600'
              )}
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M7 14l5-5 5 5z" />
            </svg>
            <svg
              className={clsx(
                'w-3 h-3',
                direction === 'desc' ? 'text-primary-500' : 'text-gray-300 dark:text-gray-600'
              )}
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M7 10l5 5 5-5z" />
            </svg>
          </div>
        )}
      </div>
    </th>
  );
};

// Transaction row
const TransactionRow: React.FC<{
  transaction: Transaction;
  onClick?: (transaction: Transaction) => void;
  index: number;
}> = ({ transaction, onClick, index }) => {
  const isPositive = transaction.amount < 0; // Plaid uses negative for credits
  const isNegative = transaction.amount > 0;

  const rowVariants = {
    initial: { opacity: 0, x: -20 },
    animate: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.3, delay: index * 0.05 }
    },
    exit: {
      opacity: 0,
      x: 20,
      transition: { duration: 0.2 }
    },
  };

  return (
    <motion.tr
      variants={rowVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className={clsx(
        'border-b border-gray-200 dark:border-gray-700',
        onClick && 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800',
        transaction.pending && 'opacity-60'
      )}
      onClick={() => onClick?.(transaction)}
    >
      {/* Date */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        <div className="flex flex-col">
          <span>{format(new Date(transaction.date), 'MMM dd, yyyy')}</span>
          {transaction.pending && (
            <span className="text-xs text-yellow-600 dark:text-yellow-400">Pending</span>
          )}
        </div>
      </td>

      {/* Merchant */}
      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
        {transaction.merchant}
      </td>

      {/* Category */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
        {transaction.category[0] || 'Uncategorized'}
      </td>

      {/* Amount */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-right">
        <span
          className={clsx(
            'font-medium',
            isPositive && 'text-success-600 dark:text-success-400',
            isNegative && 'text-danger-600 dark:text-danger-400',
            !isPositive && !isNegative && 'text-gray-900 dark:text-gray-100'
          )}
        >
          {isPositive ? '+' : ''}
          {numeral(Math.abs(transaction.amount)).format('$0,0.00')}
        </span>
      </td>

      {/* Account (optional) */}
      {transaction.account_name && (
        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
          {transaction.account_name}
        </td>
      )}
    </motion.tr>
  );
};

export const TransactionTable: React.FC<TransactionTableProps> = ({
  transactions,
  onRowClick,
  pageSize = 20,
  className,
}) => {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ field: 'date', direction: 'desc' });
  const [searchQuery, setSearchQuery] = useState('');
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({
    start: '',
    end: '',
  });
  const [currentPage, setCurrentPage] = useState(1);

  // Handle sorting
  const handleSort = (field: SortField) => {
    setSortConfig((prev) => ({
      field,
      direction: prev?.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
    }));
  };

  // Filter and sort transactions
  const filteredTransactions = useMemo(() => {
    let filtered = [...transactions];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((t) =>
        t.merchant.toLowerCase().includes(query) ||
        t.category.some(c => c.toLowerCase().includes(query))
      );
    }

    // Date range filter
    if (dateRange.start) {
      filtered = filtered.filter((t) => new Date(t.date) >= new Date(dateRange.start));
    }
    if (dateRange.end) {
      filtered = filtered.filter((t) => new Date(t.date) <= new Date(dateRange.end));
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal: any = a[sortConfig.field];
      let bVal: any = b[sortConfig.field];

      if (sortConfig.field === 'date') {
        aVal = new Date(aVal).getTime();
        bVal = new Date(bVal).getTime();
      }

      if (typeof aVal === 'string') {
        return sortConfig.direction === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return filtered;
  }, [transactions, searchQuery, dateRange, sortConfig]);

  // Pagination
  const totalPages = Math.ceil(filteredTransactions.length / pageSize);
  const paginatedTransactions = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredTransactions.slice(start, start + pageSize);
  }, [filteredTransactions, currentPage, pageSize]);

  // Handle export
  const handleExport = () => {
    exportToCSV(filteredTransactions, `transactions_${format(new Date(), 'yyyy-MM-dd')}.csv`);
  };

  return (
    <div className={clsx('bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden', className)}>
      {/* Header with filters */}
      <div className="bg-gray-50 dark:bg-gray-900 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Title and count */}
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
            Transactions ({filteredTransactions.length})
          </h3>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search */}
            <input
              type="text"
              placeholder="Search merchant..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />

            {/* Date range */}
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange((prev) => ({ ...prev, start: e.target.value }))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange((prev) => ({ ...prev, end: e.target.value }))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />

            {/* Export button */}
            <button
              onClick={handleExport}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium transition-colors focus:ring-2 focus:ring-primary-500 focus:outline-none"
            >
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <TableHeader label="Date" field="date" sortConfig={sortConfig} onSort={handleSort} />
              <TableHeader label="Merchant" field="merchant" sortConfig={sortConfig} onSort={handleSort} />
              <TableHeader label="Category" sortConfig={sortConfig} onSort={handleSort} />
              <TableHeader label="Amount" field="amount" sortConfig={sortConfig} onSort={handleSort} className="text-right" />
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            <AnimatePresence>
              {paginatedTransactions.map((transaction, index) => (
                <TransactionRow
                  key={transaction.transaction_id}
                  transaction={transaction}
                  onClick={onRowClick}
                  index={index}
                />
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-gray-50 dark:bg-gray-900 px-6 py-4 flex items-center justify-between border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}

      {/* Empty state */}
      {paginatedTransactions.length === 0 && (
        <div className="p-8 text-center">
          <p className="text-gray-500 dark:text-gray-400">
            {transactions.length === 0
              ? 'No transactions found. Transactions will appear here after connecting your bank.'
              : 'No transactions match your search criteria.'}
          </p>
        </div>
      )}
    </div>
  );
};

export default TransactionTable;
