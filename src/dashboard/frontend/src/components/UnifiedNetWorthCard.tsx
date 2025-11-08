import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { format } from 'date-fns';

interface NetWorthData {
  timestamp: number;
  trading_value: number;
  banking_value: number;
  total: number;
}

interface UnifiedNetWorthCardProps {
  tradingNAV: number;
  totalBankBalance: number;
  historicalData: NetWorthData[];
  className?: string;
}

// Custom tooltip for the chart
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3"
    >
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
        {format(new Date(label), 'MMM dd, yyyy HH:mm')}
      </p>
      {payload.map((entry: any, index: number) => (
        <div key={index} className="flex items-center justify-between space-x-4 text-sm">
          <span
            className="flex items-center"
            style={{ color: entry.color }}
          >
            <span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: entry.color }} />
            {entry.name}:
          </span>
          <span className="font-medium text-gray-900 dark:text-gray-100">
            {numeral(entry.value).format('$0,0.00')}
          </span>
        </div>
      ))}
    </motion.div>
  );
};

export const UnifiedNetWorthCard: React.FC<UnifiedNetWorthCardProps> = ({
  tradingNAV,
  totalBankBalance,
  historicalData,
  className,
}) => {
  const combinedNetWorth = tradingNAV + totalBankBalance;

  // Calculate percentages
  const tradingPercent = combinedNetWorth > 0 ? (tradingNAV / combinedNetWorth) * 100 : 0;
  const bankingPercent = combinedNetWorth > 0 ? (totalBankBalance / combinedNetWorth) * 100 : 0;

  // Process chart data
  const chartData = useMemo(() => {
    return (historicalData || []).map((point) => ({
      timestamp: point.timestamp,
      'Trading Account': point.trading_value,
      'Banking': point.banking_value,
      Total: point.total,
    }));
  }, [historicalData]);

  // Calculate trend
  const trend = useMemo(() => {
    if (!historicalData || historicalData.length < 2) return { change: 0, percent: 0 };

    const latest = historicalData[historicalData.length - 1];
    const previous = historicalData[0];
    const change = latest.total - previous.total;
    const percent = previous.total > 0 ? (change / previous.total) * 100 : 0;

    return { change, percent };
  }, [historicalData]);

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3 }
    },
  };

  return (
    <motion.div
      variants={cardVariants}
      initial="initial"
      animate="animate"
      className={clsx(
        'bg-white dark:bg-gray-800 rounded-lg shadow p-6',
        className
      )}
    >
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-1">
          Combined Net Worth
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Trading + Banking
        </p>
      </div>

      {/* Main net worth display */}
      <div className="mb-6">
        <div className="flex items-end justify-between">
          <div>
            <p className="text-4xl font-bold text-gray-900 dark:text-gray-100">
              {numeral(combinedNetWorth).format('$0,0.00')}
            </p>
            {trend.change !== 0 && (
              <div className="flex items-center mt-2 space-x-2">
                <span
                  className={clsx(
                    'text-sm font-medium',
                    trend.change > 0 ? 'text-success-600 dark:text-success-400' : 'text-danger-600 dark:text-danger-400'
                  )}
                >
                  {trend.change > 0 ? '+' : ''}
                  {numeral(trend.change).format('$0,0.00')}
                </span>
                <span
                  className={clsx(
                    'text-sm',
                    trend.change > 0 ? 'text-success-500' : 'text-danger-500'
                  )}
                >
                  ({trend.percent > 0 ? '+' : ''}{numeral(trend.percent / 100).format('0.00%')})
                </span>
              </div>
            )}
          </div>

          {/* Trend icon */}
          {trend.change !== 0 && (
            <div
              className={clsx(
                'p-3 rounded-full',
                trend.change > 0 ? 'bg-success-100 dark:bg-success-900/20' : 'bg-danger-100 dark:bg-danger-900/20'
              )}
            >
              <svg
                className={clsx(
                  'w-6 h-6',
                  trend.change > 0 ? 'text-success-600' : 'text-danger-600'
                )}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {trend.change > 0 ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                )}
              </svg>
            </div>
          )}
        </div>
      </div>

      {/* Breakdown by account type */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Trading Account */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Trading
            </span>
            <span className="text-xs text-blue-600 dark:text-blue-400">
              {numeral(tradingPercent / 100).format('0.0%')}
            </span>
          </div>
          <p className="text-xl font-bold text-blue-900 dark:text-blue-100">
            {numeral(tradingNAV).format('$0,0.00')}
          </p>
        </div>

        {/* Banking */}
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              Banking
            </span>
            <span className="text-xs text-green-600 dark:text-green-400">
              {numeral(bankingPercent / 100).format('0.0%')}
            </span>
          </div>
          <p className="text-xl font-bold text-green-900 dark:text-green-100">
            {numeral(totalBankBalance).format('$0,0.00')}
          </p>
        </div>
      </div>

      {/* Historical chart */}
      {chartData.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Historical Net Worth
          </h4>
          <div style={{ height: '250px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-gray-600" />
                <XAxis
                  dataKey="timestamp"
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(ts) => format(new Date(ts), 'MM/dd')}
                  stroke="#9ca3af"
                  className="dark:stroke-gray-400"
                  style={{ fontSize: '12px' }}
                />
                <YAxis
                  tickFormatter={(value) => numeral(value).format('$0.0a')}
                  stroke="#9ca3af"
                  className="dark:stroke-gray-400"
                  style={{ fontSize: '12px' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="Trading Account"
                  stackId="1"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="Banking"
                  stackId="1"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Summary stats */}
      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-3 gap-4 text-center text-sm">
          <div>
            <p className="text-gray-500 dark:text-gray-400">Liquid</p>
            <p className="font-medium text-gray-900 dark:text-gray-100">
              {numeral(totalBankBalance).format('$0,0')}
            </p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Invested</p>
            <p className="font-medium text-gray-900 dark:text-gray-100">
              {numeral(tradingNAV).format('$0,0')}
            </p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Total</p>
            <p className="font-medium text-gray-900 dark:text-gray-100">
              {numeral(combinedNetWorth).format('$0,0')}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default UnifiedNetWorthCard;
