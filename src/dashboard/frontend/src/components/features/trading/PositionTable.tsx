import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { PositionTableProps, PositionUpdate } from '@/types';

// Sort configuration
type SortField = keyof PositionUpdate;
type SortDirection = 'asc' | 'desc';

interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

// Table header component with sorting
const TableHeader: React.FC<{
  label: string;
  field: SortField;
  sortConfig: SortConfig | null;
  onSort: (field: SortField) => void;
  className?: string;
}> = ({ label, field, sortConfig, onSort, className }) => {
  const isSorted = sortConfig?.field === field;
  const direction = isSorted ? sortConfig.direction : null;

  return (
    <th
      className={clsx(
        'px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors',
        className
      )}
      onClick={() => onSort(field)}
    >
      <div className="flex items-center space-x-1">
        <span>{label}</span>
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
      </div>
    </th>
  );
};

// Position row component
const PositionRow: React.FC<{
  position: PositionUpdate;
  onClick?: (position: PositionUpdate) => void;
  index: number;
}> = ({ position, onClick, index }) => {
  const isProfit = position.unrealized_pnl > 0;
  const isLoss = position.unrealized_pnl < 0;

  const changePercent = position.entry_price > 0
    ? ((position.current_price - position.entry_price) / position.entry_price) * 100
    : 0;

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
    hover: {
      backgroundColor: 'var(--hover-bg)',
      transition: { duration: 0.1 }
    }
  };

  return (
    <motion.tr
      variants={rowVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      whileHover={onClick ? "hover" : undefined}
      className={clsx(
        'border-b border-gray-200 dark:border-gray-700',
        onClick && 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800'
      )}
      onClick={() => onClick?.(position)}
      style={{
        '--hover-bg': 'rgb(249 250 251 / var(--tw-bg-opacity))'
      } as React.CSSProperties}
    >
      {/* Symbol */}
      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
        <div className="flex items-center">
          <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center mr-3">
            <span className="text-primary-600 dark:text-primary-300 font-semibold text-xs">
              {position.symbol.slice(0, 2)}
            </span>
          </div>
          {position.symbol}
        </div>
      </td>

      {/* Quantity */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        {numeral(position.quantity).format('0,0')}
      </td>

      {/* Current Price */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        {numeral(position.current_price).format('$0,0.00')}
      </td>

      {/* Market Value */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        {numeral(position.market_value).format('$0,0.00')}
      </td>

      {/* P&L */}
      <td className="px-4 py-3 whitespace-nowrap text-sm">
        <div className="flex flex-col">
          <span className={clsx(
            'font-medium',
            isProfit && 'text-success-600 dark:text-success-400',
            isLoss && 'text-danger-600 dark:text-danger-400',
            !isProfit && !isLoss && 'text-gray-900 dark:text-gray-100'
          )}>
            {numeral(position.unrealized_pnl).format('$0,0.00')}
          </span>
          <span className={clsx(
            'text-xs',
            isProfit && 'text-success-500',
            isLoss && 'text-danger-500',
            !isProfit && !isLoss && 'text-gray-500'
          )}>
            ({changePercent > 0 ? '+' : ''}{numeral(changePercent / 100).format('0.00%')})
          </span>
        </div>
      </td>

      {/* Weight */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        <div className="flex items-center">
          <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
            <div
              className="bg-primary-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(position.weight * 100, 100)}%` }}
            />
          </div>
          <span className="text-xs">{numeral(position.weight).format('0.0%')}</span>
        </div>
      </td>

      {/* Last Update */}
      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
        {new Date(position.last_update * 1000).toLocaleTimeString()}
      </td>
    </motion.tr>
  );
};

// Main PositionTable component
export const PositionTable: React.FC<PositionTableProps> = ({
  positions,
  onRowClick,
  sortBy,
  sortDirection,
}) => {
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(
    sortBy && sortDirection ? { field: sortBy, direction: sortDirection } : null
  );

  // Handle sorting
  const handleSort = (field: SortField) => {
    let direction: SortDirection = 'asc';

    if (sortConfig && sortConfig.field === field && sortConfig.direction === 'asc') {
      direction = 'desc';
    }

    setSortConfig({ field, direction });
  };

  // Sort positions
  const sortedPositions = useMemo(() => {
    if (!sortConfig) return positions;

    return [...positions].sort((a, b) => {
      const aVal = a[sortConfig.field];
      const bVal = b[sortConfig.field];

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortConfig.direction === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
      }

      return 0;
    });
  }, [positions, sortConfig]);

  // Calculate summary statistics
  const summary = useMemo(() => {
    return positions.reduce(
      (acc, position) => ({
        totalValue: acc.totalValue + position.market_value,
        totalPnL: acc.totalPnL + position.unrealized_pnl,
        winners: acc.winners + (position.unrealized_pnl > 0 ? 1 : 0),
        losers: acc.losers + (position.unrealized_pnl < 0 ? 1 : 0),
      }),
      { totalValue: 0, totalPnL: 0, winners: 0, losers: 0 }
    );
  }, [positions]);

  if (positions.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2M4 13h2m0 0V9a2 2 0 012-2h2m-4 4v4a2 2 0 002 2h2m0-6V9a2 2 0 012-2h2m-4 4v4a2 2 0 002 2h2" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No Positions
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            No active positions found. Positions will appear here when trades are executed.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
      {/* Summary header */}
      <div className="bg-gray-50 dark:bg-gray-900 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
            Positions ({positions.length})
          </h3>
          <div className="flex items-center space-x-6 text-sm">
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Total Value</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {numeral(summary.totalValue).format('$0,0.00')}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Unrealized P&L</div>
              <div className={clsx(
                'font-medium',
                summary.totalPnL > 0 && 'text-success-600 dark:text-success-400',
                summary.totalPnL < 0 && 'text-danger-600 dark:text-danger-400',
                summary.totalPnL === 0 && 'text-gray-900 dark:text-gray-100'
              )}>
                {numeral(summary.totalPnL).format('$0,0.00')}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Winners/Losers</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                <span className="text-success-600">{summary.winners}</span>
                {' / '}
                <span className="text-danger-600">{summary.losers}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <TableHeader
                label="Symbol"
                field="symbol"
                sortConfig={sortConfig}
                onSort={handleSort}
              />
              <TableHeader
                label="Quantity"
                field="quantity"
                sortConfig={sortConfig}
                onSort={handleSort}
                className="text-right"
              />
              <TableHeader
                label="Price"
                field="current_price"
                sortConfig={sortConfig}
                onSort={handleSort}
                className="text-right"
              />
              <TableHeader
                label="Market Value"
                field="market_value"
                sortConfig={sortConfig}
                onSort={handleSort}
                className="text-right"
              />
              <TableHeader
                label="Unrealized P&L"
                field="unrealized_pnl"
                sortConfig={sortConfig}
                onSort={handleSort}
                className="text-right"
              />
              <TableHeader
                label="Weight"
                field="weight"
                sortConfig={sortConfig}
                onSort={handleSort}
              />
              <TableHeader
                label="Last Update"
                field="last_update"
                sortConfig={sortConfig}
                onSort={handleSort}
              />
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            <AnimatePresence>
              {sortedPositions.map((position, index) => (
                <PositionRow
                  key={position.symbol}
                  position={position}
                  onClick={onRowClick}
                  index={index}
                />
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {/* Mobile-friendly cards view (hidden on desktop) */}
      <div className="block sm:hidden">
        <div className="p-4 space-y-4">
          <AnimatePresence>
            {sortedPositions.map((position, index) => (
              <motion.div
                key={position.symbol}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0, transition: { delay: index * 0.05 } }}
                exit={{ opacity: 0, y: -20 }}
                className={clsx(
                  'bg-gray-50 dark:bg-gray-700 rounded-lg p-4',
                  onRowClick && 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600'
                )}
                onClick={() => onRowClick?.(position)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-medium text-gray-900 dark:text-gray-100">
                    {position.symbol}
                  </div>
                  <div className={clsx(
                    'text-sm font-medium',
                    position.unrealized_pnl > 0 && 'text-success-600',
                    position.unrealized_pnl < 0 && 'text-danger-600',
                    position.unrealized_pnl === 0 && 'text-gray-900 dark:text-gray-100'
                  )}>
                    {numeral(position.unrealized_pnl).format('$0,0.00')}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Qty: </span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {numeral(position.quantity).format('0,0')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Price: </span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {numeral(position.current_price).format('$0,0.00')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Value: </span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {numeral(position.market_value).format('$0,0.00')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Weight: </span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {numeral(position.weight).format('0.0%')}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};