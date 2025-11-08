import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { format } from 'date-fns';
import toast from 'react-hot-toast';
import axios from 'axios';

export interface BankAccount {
  account_id: string;
  name: string;
  type: string;
  mask: string;
  balance: number;
  institution: string;
  institution_logo?: string;
  lastSync: number;
  available_balance?: number;
  currency?: string;
}

interface BankAccountCardProps {
  account: BankAccount;
  onRefresh?: (accountId: string) => Promise<void>;
  onClick?: () => void;
  className?: string;
}

// Account type colors
const getAccountTypeColor = (type: string): { bg: string; text: string; icon: string } => {
  const typeMap: Record<string, { bg: string; text: string; icon: string }> = {
    depository: {
      bg: 'bg-blue-100 dark:bg-blue-900/20',
      text: 'text-blue-700 dark:text-blue-300',
      icon: 'text-blue-600',
    },
    credit: {
      bg: 'bg-purple-100 dark:bg-purple-900/20',
      text: 'text-purple-700 dark:text-purple-300',
      icon: 'text-purple-600',
    },
    loan: {
      bg: 'bg-orange-100 dark:bg-orange-900/20',
      text: 'text-orange-700 dark:text-orange-300',
      icon: 'text-orange-600',
    },
    investment: {
      bg: 'bg-green-100 dark:bg-green-900/20',
      text: 'text-green-700 dark:text-green-300',
      icon: 'text-green-600',
    },
  };

  return typeMap[type.toLowerCase()] || {
    bg: 'bg-gray-100 dark:bg-gray-900/20',
    text: 'text-gray-700 dark:text-gray-300',
    icon: 'text-gray-600',
  };
};

// Bank icon component
const BankIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"
    />
  </svg>
);

// Refresh icon component
const RefreshIcon: React.FC<{ className?: string; spinning?: boolean }> = ({ className, spinning }) => (
  <svg
    className={clsx(className, spinning && 'animate-spin')}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
    />
  </svg>
);

export const BankAccountCard: React.FC<BankAccountCardProps> = ({
  account,
  onRefresh,
  onClick,
  className,
}) => {
  const [refreshing, setRefreshing] = useState(false);
  const typeColors = getAccountTypeColor(account.type);

  // Handle refresh button click
  const handleRefresh = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (refreshing) return;

    setRefreshing(true);
    try {
      if (onRefresh) {
        await onRefresh(account.account_id);
      } else {
        // Default refresh implementation
        await axios.post('/api/plaid/refresh_account', {
          account_id: account.account_id,
        });
      }
      toast.success('Account refreshed successfully');
    } catch (error: any) {
      console.error('Failed to refresh account:', error);
      toast.error('Failed to refresh: ' + (error.response?.data?.error || error.message));
    } finally {
      setRefreshing(false);
    }
  };

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3 }
    },
    hover: onClick ? {
      scale: 1.02,
      transition: { duration: 0.2 }
    } : {},
  };

  return (
    <motion.div
      variants={cardVariants}
      initial="initial"
      animate="animate"
      whileHover="hover"
      className={clsx(
        'relative p-5 rounded-lg border shadow-sm transition-all duration-200',
        'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700',
        onClick && 'cursor-pointer hover:shadow-md',
        className
      )}
      onClick={onClick}
    >
      {/* Header with institution info */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          {/* Institution logo or default icon */}
          <div className="w-12 h-12 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center overflow-hidden">
            {account.institution_logo ? (
              <img
                src={account.institution_logo}
                alt={account.institution}
                className="w-full h-full object-cover"
              />
            ) : (
              <BankIcon className="w-6 h-6 text-primary-600 dark:text-primary-300" />
            )}
          </div>

          {/* Institution and account name */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              {account.institution}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {account.name}
            </p>
          </div>
        </div>

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            'hover:bg-gray-100 dark:hover:bg-gray-700',
            'focus:outline-none focus:ring-2 focus:ring-primary-500',
            refreshing && 'opacity-50 cursor-not-allowed'
          )}
          title="Refresh account"
        >
          <RefreshIcon
            className="w-5 h-5 text-gray-600 dark:text-gray-400"
            spinning={refreshing}
          />
        </button>
      </div>

      {/* Account type badge */}
      <div className="mb-3">
        <span
          className={clsx(
            'inline-block px-3 py-1 rounded-full text-xs font-medium',
            typeColors.bg,
            typeColors.text
          )}
        >
          {account.type}
        </span>
      </div>

      {/* Masked account number */}
      <div className="mb-3">
        <p className="text-sm text-gray-500 dark:text-gray-400">Account</p>
        <p className="text-base font-mono text-gray-900 dark:text-gray-100">
          •••• {account.mask}
        </p>
      </div>

      {/* Balance */}
      <div className="mb-3">
        <p className="text-sm text-gray-500 dark:text-gray-400">Current Balance</p>
        <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          {numeral(account.balance).format('$0,0.00')}
        </p>
        {account.available_balance !== undefined && account.available_balance !== account.balance && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Available: {numeral(account.available_balance).format('$0,0.00')}
          </p>
        )}
      </div>

      {/* Last sync timestamp */}
      <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Last synced: {format(new Date(account.lastSync), 'MMM dd, yyyy HH:mm')}
        </p>
      </div>
    </motion.div>
  );
};

// Grid container for multiple bank accounts
export const BankAccountGrid: React.FC<{
  accounts: BankAccount[];
  onRefresh?: (accountId: string) => Promise<void>;
  onAccountClick?: (account: BankAccount) => void;
  className?: string;
  columns?: 1 | 2 | 3;
}> = ({ accounts, onRefresh, onAccountClick, className, columns = 3 }) => {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3',
  };

  if (accounts.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
          <BankIcon className="w-8 h-8 text-gray-400" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          No Bank Accounts
        </h3>
        <p className="text-gray-500 dark:text-gray-400">
          Connect your bank accounts to see balances and transactions.
        </p>
      </div>
    );
  }

  return (
    <div className={clsx('grid gap-4', gridCols[columns], className)}>
      {accounts.map((account) => (
        <BankAccountCard
          key={account.account_id}
          account={account}
          onRefresh={onRefresh}
          onClick={onAccountClick ? () => onAccountClick(account) : undefined}
        />
      ))}
    </div>
  );
};

export default BankAccountCard;
