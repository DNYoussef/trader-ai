import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { format } from 'date-fns';

// Icons as inline SVGs
const TrendingUp = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TrendingDown = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
  </svg>
);

const AlertTriangle = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

interface InequalityMetrics {
  giniCoefficient: number;
  top1PercentWealth: number;
  top10PercentWealth: number;
  wageGrowthReal: number;
  corporateProfitsToGdp: number;
  householdDebtToIncome: number;
  luxuryVsDiscountSpend: number;
  wealthVelocity: number;
  consensusWrongScore: number;
}

interface WealthFlowData {
  source: string;
  target: string;
  value: number;
  color: string;
}

interface ContrarianSignal {
  topic: string;
  consensusView: string;
  realityView: string;
  conviction: number;
  opportunity: string;
}

interface InequalityPanelProps {
  metrics: InequalityMetrics;
  historicalData: Array<{ date: string; gini: number; top1: number; wageGrowth: number }>;
  wealthFlows: WealthFlowData[];
  contrarianSignals: ContrarianSignal[];
}

export const InequalityPanel: React.FC<InequalityPanelProps> = ({
  metrics,
  historicalData,
  wealthFlows,
  contrarianSignals
}) => {
  // Calculate trend indicators
  const trends = useMemo(() => {
    const recent = historicalData[historicalData.length - 1];
    const older = historicalData[Math.max(0, historicalData.length - 30)];

    return {
      giniTrend: recent?.gini > older?.gini ? 'up' : 'down',
      wealthTrend: recent?.top1 > older?.top1 ? 'up' : 'down',
      wageTrend: recent?.wageGrowth > older?.wageGrowth ? 'up' : 'down'
    };
  }, [historicalData]);

  // Color scheme for charts
  const colors = {
    primary: '#8b5cf6',
    danger: '#ef4444',
    warning: '#f59e0b',
    success: '#10b981',
    info: '#3b82f6'
  };

  // Format percentage
  const formatPercent = (value: number) => `${value.toFixed(1)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-800 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2">Inequality Hunter Dashboard</h2>
        <p className="text-purple-100">
          "Economists don't look at inequality. I do. That's how I win." - Gary Stevenson
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Gini Coefficient */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Gini Coefficient</span>
            {trends.giniTrend === 'up' ? (
              <span className="text-red-500"><TrendingUp /></span>
            ) : (
              <span className="text-green-500"><TrendingDown /></span>
            )}
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {metrics.giniCoefficient.toFixed(3)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Higher = More Inequality
          </div>
          <div className="mt-2 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-yellow-400 to-red-500 h-2 rounded-full"
              style={{ width: `${metrics.giniCoefficient * 100}%` }}
            />
          </div>
        </motion.div>

        {/* Top 1% Wealth Share */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Top 1% Wealth</span>
            {trends.wealthTrend === 'up' ? (
              <span className="text-red-500"><TrendingUp /></span>
            ) : (
              <span className="text-green-500"><TrendingDown /></span>
            )}
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatPercent(metrics.top1PercentWealth)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Of Total Wealth
          </div>
        </motion.div>

        {/* Real Wage Growth */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Real Wage Growth</span>
            {trends.wageTrend === 'up' ? (
              <span className="text-green-500"><TrendingUp /></span>
            ) : (
              <span className="text-red-500"><TrendingDown /></span>
            )}
          </div>
          <div className={`text-2xl font-bold ${metrics.wageGrowthReal < 0 ? 'text-red-500' : 'text-green-500'}`}>
            {formatPercent(metrics.wageGrowthReal)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Year over Year
          </div>
        </motion.div>

        {/* Consensus Wrong Score */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white p-4 rounded-lg shadow"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm">Gary Moment Score</span>
            <AlertTriangle />
          </div>
          <div className="text-2xl font-bold">
            {(metrics.consensusWrongScore * 100).toFixed(0)}%
          </div>
          <div className="text-xs mt-1">
            Consensus Blindness Level
          </div>
        </motion.div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Historical Inequality Trend */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
            Inequality Acceleration
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="date"
                tickFormatter={(date) => format(new Date(date), 'MMM')}
                stroke="#9ca3af"
              />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(17, 24, 39, 0.9)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'white'
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="gini"
                stroke={colors.primary}
                strokeWidth={2}
                name="Gini Coefficient"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="top1"
                stroke={colors.danger}
                strokeWidth={2}
                name="Top 1% Share"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Wealth Flow Visualization */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
            Wealth Flows (Gary's Key Insight)
          </h3>
          <div className="space-y-3">
            {wealthFlows.map((flow, index) => (
              <div key={index} className="relative">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {flow.source} → {flow.target}
                  </span>
                  <span className="text-sm font-semibold">
                    {formatPercent(flow.value)}
                  </span>
                </div>
                <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${flow.value}%` }}
                    transition={{ duration: 1, delay: index * 0.1 }}
                    className="h-3 rounded-full"
                    style={{ backgroundColor: flow.color }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
            <p className="text-sm text-purple-700 dark:text-purple-300">
              💡 Money flows from poor to rich through rents, profits, and debt service.
              This concentration drives asset prices regardless of economic fundamentals.
            </p>
          </div>
        </div>
      </div>

      {/* Contrarian Signals Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-purple-600 to-purple-800 text-white">
          <h3 className="text-lg font-semibold">Contrarian Opportunities (Consensus Blind Spots)</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Topic
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Consensus Says
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Reality Is
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Conviction
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Trade
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
              {contrarianSignals.map((signal, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                    {signal.topic}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                    {signal.consensusView}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900 dark:text-gray-100 font-medium">
                    {signal.realityView}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-2">
                        <div
                          className={`h-2 rounded-full ${
                            signal.conviction > 0.8 ? 'bg-green-500' :
                            signal.conviction > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${signal.conviction * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {(signal.conviction * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                      signal.opportunity.includes('Long') ?
                        'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100' :
                        'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
                    }`}>
                      {signal.opportunity}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bottom Insight Box */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-lg border border-purple-200 dark:border-purple-700"
      >
        <h4 className="text-lg font-semibold text-purple-900 dark:text-purple-100 mb-2">
          The Gary Trading Framework
        </h4>
        <p className="text-purple-700 dark:text-purple-300 mb-3">
          "You only need to identify ONE thing economists are missing. Growing inequality of wealth
          is destroying the economy. Everything can be predicted by understanding what will happen
          with growing inequality."
        </p>
        <div className="flex items-center justify-between">
          <div className="text-sm text-purple-600 dark:text-purple-400">
            Current Opportunity Level: <span className="font-bold text-lg">EXTREME</span>
          </div>
          <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
            View Full Analysis
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default InequalityPanel;