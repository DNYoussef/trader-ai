import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { format } from 'date-fns';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { RiskChartProps, ChartData } from '@/types';

// Custom tooltip component
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0];
  const timestamp = typeof label === 'number' ? label : Date.now();

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3"
    >
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-1">
        {format(new Date(timestamp), 'MMM dd, yyyy HH:mm')}
      </p>
      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
        <span
          className="inline-block w-3 h-3 rounded-full mr-2"
          style={{ backgroundColor: data.color }}
        />
        {data.name}: {typeof data.value === 'number' ? numeral(data.value).format('0,0.00') : data.value}
      </p>
    </motion.div>
  );
};

// Custom dot component for line chart
const CustomDot: React.FC<any> = (props) => {
  const { cx, cy, fill, payload } = props;

  // Only show dots for recent data points or significant changes
  const isRecent = payload && (Date.now() - payload.timestamp) < 300000; // 5 minutes
  if (!isRecent) return null;

  return (
    <motion.circle
      initial={{ r: 0 }}
      animate={{ r: 3 }}
      cx={cx}
      cy={cy}
      fill={fill}
      stroke="white"
      strokeWidth={2}
    />
  );
};

// Format X-axis labels
const formatXAxis = (timestamp: number) => {
  const now = Date.now();
  const diff = now - timestamp;

  if (diff < 3600000) { // Less than 1 hour
    return format(new Date(timestamp), 'HH:mm');
  } else if (diff < 86400000) { // Less than 1 day
    return format(new Date(timestamp), 'HH:mm');
  } else {
    return format(new Date(timestamp), 'MM/dd');
  }
};

// Format Y-axis labels based on data type
const formatYAxis = (value: number, format: string) => {
  switch (format) {
    case 'currency':
      return numeral(value).format('$0.0a');
    case 'percentage':
      return numeral(value).format('0.0%');
    case 'ratio':
      return numeral(value).format('0.0');
    default:
      return numeral(value).format('0.0a');
  }
};

// Main RiskChart component
export const RiskChart: React.FC<RiskChartProps> = ({
  data,
  title,
  color = '#3b82f6',
  height = 300,
  showGrid = true,
}) => {
  // Process data for chart
  const chartData = useMemo(() => {
    return data.map(point => ({
      ...point,
      timestamp: point.timestamp * 1000, // Convert to milliseconds if needed
    }));
  }, [data]);

  // Determine chart type based on data characteristics
  const chartType = useMemo(() => {
    if (title.toLowerCase().includes('portfolio') || title.toLowerCase().includes('value')) {
      return 'area';
    }
    return 'line';
  }, [title]);

  // Determine value format
  const valueFormat = useMemo(() => {
    if (title.toLowerCase().includes('portfolio') || title.toLowerCase().includes('value')) {
      return 'currency';
    }
    if (title.toLowerCase().includes('ratio') || title.toLowerCase().includes('sharpe')) {
      return 'ratio';
    }
    if (title.toLowerCase().includes('%') || title.toLowerCase().includes('ruin') || title.toLowerCase().includes('drawdown')) {
      return 'percentage';
    }
    return 'number';
  }, [title]);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
          {title}
        </h3>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-gray-500 dark:text-gray-400">No data available</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
          {title}
        </h3>
        <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <span>Last {chartData.length} points</span>
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
        </div>
      </div>

      <div style={{ height: `${height}px` }}>
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'area' ? (
            <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              {showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  className="dark:stroke-gray-600"
                />
              )}
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={formatXAxis}
                stroke="#9ca3af"
                className="dark:stroke-gray-400"
              />
              <YAxis
                tickFormatter={(value) => formatYAxis(value, valueFormat)}
                stroke="#9ca3af"
                className="dark:stroke-gray-400"
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="value"
                stroke={color}
                fill={color}
                fillOpacity={0.1}
                strokeWidth={2}
                dot={<CustomDot />}
                animationDuration={300}
              />
            </AreaChart>
          ) : (
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              {showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  className="dark:stroke-gray-600"
                />
              )}
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={formatXAxis}
                stroke="#9ca3af"
                className="dark:stroke-gray-400"
              />
              <YAxis
                tickFormatter={(value) => formatYAxis(value, valueFormat)}
                stroke="#9ca3af"
                className="dark:stroke-gray-400"
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={<CustomDot />}
                animationDuration={300}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Chart statistics */}
      {chartData.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Current</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {formatYAxis(chartData[chartData.length - 1]?.value || 0, valueFormat)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Min</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {formatYAxis(Math.min(...chartData.map(d => d.value)), valueFormat)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-500 dark:text-gray-400">Max</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {formatYAxis(Math.max(...chartData.map(d => d.value)), valueFormat)}
              </div>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};

// Specialized chart components
export const PortfolioValueChart: React.FC<{ data: ChartData[]; height?: number }> = ({
  data,
  height = 300
}) => (
  <RiskChart
    data={data}
    title="Portfolio Value"
    color="#10b981"
    height={height}
  />
);

export const PRuinChart: React.FC<{ data: ChartData[]; height?: number }> = ({
  data,
  height = 300
}) => (
  <RiskChart
    data={data}
    title="P(ruin) Over Time"
    color="#ef4444"
    height={height}
  />
);

export const VarChart: React.FC<{ data: ChartData[]; height?: number }> = ({
  data,
  height = 300
}) => (
  <RiskChart
    data={data}
    title="Value at Risk (95%)"
    color="#f59e0b"
    height={height}
  />
);

export const SharpeChart: React.FC<{ data: ChartData[]; height?: number }> = ({
  data,
  height = 300
}) => (
  <RiskChart
    data={data}
    title="Sharpe Ratio"
    color="#3b82f6"
    height={height}
  />
);

export const DrawdownChart: React.FC<{ data: ChartData[]; height?: number }> = ({
  data,
  height = 300
}) => (
  <RiskChart
    data={data}
    title="Max Drawdown"
    color="#dc2626"
    height={height}
  />
);

// Chart grid container
export const ChartGrid: React.FC<{
  children: React.ReactNode;
  className?: string;
  columns?: 1 | 2;
}> = ({ children, className, columns = 2 }) => {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 lg:grid-cols-2',
  };

  return (
    <div className={clsx(
      'grid gap-6',
      gridCols[columns],
      className
    )}>
      {children}
    </div>
  );
};