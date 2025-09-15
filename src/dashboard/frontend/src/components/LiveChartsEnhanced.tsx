import React, { useState, useEffect } from 'react';
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
  ReferenceLine,
  Brush
} from 'recharts';

interface LiveChartData {
  timestamp: number;
  portfolio_value: number;
  p_ruin: number;
  sharpe_ratio: number;
  var_95: number;
  price: number;
  high: number;
  low: number;
  volume: number;
  signal?: number;
  dpi_score?: number;
  ai_confidence?: number;
}

interface LiveChartsProps {
  data: LiveChartData[];
  isRealTime?: boolean;
}

export const TradingTerminalChart: React.FC<LiveChartsProps> = ({ data, isRealTime = true }) => {
  const [timeframe, setTimeframe] = useState('15m');
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [showSignals, setShowSignals] = useState(true);
  const [chartType, setChartType] = useState<'line' | 'area' | 'candlestick'>('line');

  // Generate enhanced data with trading signals
  const enhancedData = data.map((d, i) => ({
    ...d,
    high: d.portfolio_value + Math.random() * 50,
    low: d.portfolio_value - Math.random() * 50,
    signal: showSignals && i % 7 === 0 ? d.portfolio_value + 100 : null,
    dpi_score: 70 + Math.random() * 30,
    ai_confidence: 0.6 + Math.random() * 0.4
  }));

  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
  const symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIX'];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Professional Trading Terminal
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {selectedSymbol} - Real-time price action with AI signals
          </p>
        </div>

        {/* Live indicator */}
        {isRealTime && (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-600 font-medium">LIVE</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
        {/* Symbol selector */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Symbol:</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-3 py-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-sm"
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
        </div>

        {/* Timeframe selector */}
        <div className="flex space-x-1">
          {timeframes.map(tf => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-1 text-sm rounded ${
                timeframe === tf
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Chart type selector */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Type:</label>
          <select
            value={chartType}
            onChange={(e) => setChartType(e.target.value as any)}
            className="px-3 py-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-sm"
          >
            <option value="line">Line</option>
            <option value="area">Area</option>
            <option value="candlestick">Candlestick</option>
          </select>
        </div>

        {/* Show signals toggle */}
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={showSignals}
            onChange={(e) => setShowSignals(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">AI Signals</span>
        </label>
      </div>

      {/* Main Chart */}
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'area' ? (
            <AreaChart data={enhancedData}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(tick) => new Date(tick).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                stroke="#9ca3af"
              />
              <YAxis
                domain={['dataMin - 50', 'dataMax + 50']}
                tickFormatter={(tick) => `$${tick.toFixed(0)}`}
                stroke="#9ca3af"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '6px',
                  color: '#fff'
                }}
                formatter={(value: any, name: string) => {
                  if (name === 'portfolio_value') return [`$${value.toFixed(2)}`, 'Price'];
                  if (name === 'signal') return [`BUY at $${value.toFixed(2)}`, 'AI Signal'];
                  return [value, name];
                }}
                labelFormatter={(label) => new Date(label).toLocaleString()}
              />
              <Area
                type="monotone"
                dataKey="portfolio_value"
                stroke="#10b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorPrice)"
              />
              {showSignals && (
                <Line
                  type="monotone"
                  dataKey="signal"
                  stroke="#f59e0b"
                  strokeWidth={0}
                  dot={{ r: 6, fill: '#f59e0b' }}
                />
              )}
            </AreaChart>
          ) : (
            <LineChart data={enhancedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(tick) => new Date(tick).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                stroke="#9ca3af"
              />
              <YAxis
                domain={['dataMin - 50', 'dataMax + 50']}
                tickFormatter={(tick) => `$${tick.toFixed(0)}`}
                stroke="#9ca3af"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '6px',
                  color: '#fff'
                }}
                formatter={(value: any, name: string) => {
                  if (name === 'Price') return [`$${value.toFixed(2)}`, name];
                  if (name === 'High') return [`$${value.toFixed(2)}`, name];
                  if (name === 'Low') return [`$${value.toFixed(2)}`, name];
                  if (name === 'AI Signal') return [`BUY at $${value.toFixed(2)}`, name];
                  return [value, name];
                }}
                labelFormatter={(label) => new Date(label).toLocaleString()}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="portfolio_value"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Price"
              />
              <Line
                type="monotone"
                dataKey="high"
                stroke="#6b7280"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                name="High"
              />
              <Line
                type="monotone"
                dataKey="low"
                stroke="#6b7280"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                name="Low"
              />
              {showSignals && (
                <Line
                  type="monotone"
                  dataKey="signal"
                  stroke="#f59e0b"
                  strokeWidth={0}
                  dot={{ r: 6, fill: '#f59e0b' }}
                  name="AI Signal"
                />
              )}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Trading Panel */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Quick Trade */}
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Quick Trade</h4>
            <div className="flex space-x-2">
              <button className="flex-1 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 font-medium">
                BUY
              </button>
              <button className="flex-1 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 font-medium">
                SELL
              </button>
            </div>
          </div>

          {/* AI Signals */}
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">AI Signals</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-600 dark:text-gray-400">DPI Score</p>
                <p className="text-lg font-bold text-green-600">85%</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Signal</p>
                <p className="text-lg font-bold text-green-600">BUY</p>
              </div>
            </div>
          </div>

          {/* Market Data */}
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Market Data</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-600 dark:text-gray-400">Volume</p>
                <p className="text-lg font-bold text-blue-600">2.4M</p>
              </div>
              <div>
                <p className="text-gray-600 dark:text-gray-400">Spread</p>
                <p className="text-lg font-bold text-purple-600">$0.02</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const PortfolioPerformanceChart: React.FC<LiveChartsProps> = ({ data }) => {
  const [showBenchmark, setShowBenchmark] = useState(true);
  const [metric, setMetric] = useState<'value' | 'returns' | 'drawdown'>('value');

  const benchmarkData = data.map(d => ({
    ...d,
    benchmark: 25000 * Math.pow(1.07, (d.timestamp - data[0].timestamp) / (365 * 24 * 60 * 60 * 1000))
  }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Portfolio Performance Analysis
        </h3>

        <div className="flex items-center space-x-4">
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as any)}
            className="px-3 py-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-sm"
          >
            <option value="value">Portfolio Value</option>
            <option value="returns">Returns</option>
            <option value="drawdown">Drawdown</option>
          </select>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showBenchmark}
              onChange={(e) => setShowBenchmark(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Show Benchmark</span>
          </label>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={benchmarkData}>
            <defs>
              <linearGradient id="colorPortfolio" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
              stroke="#6b7280"
            />
            <YAxis
              tickFormatter={(tick) => `$${(tick / 1000).toFixed(1)}k`}
              stroke="#6b7280"
            />
            <Tooltip
              formatter={(value: any, name: string) => [
                `$${value.toFixed(2)}`,
                name === 'portfolio_value' ? 'Portfolio' : 'Benchmark'
              ]}
              labelFormatter={(label) => new Date(label).toLocaleDateString()}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '6px'
              }}
            />
            <Area
              type="monotone"
              dataKey="portfolio_value"
              stroke="#3b82f6"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorPortfolio)"
            />
            {showBenchmark && (
              <Area
                type="monotone"
                dataKey="benchmark"
                stroke="#ef4444"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorBenchmark)"
              />
            )}
            <Brush dataKey="timestamp" height={30} stroke="#8884d8" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Metrics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">Total Return</p>
          <p className="text-lg font-bold text-green-600">+18.2%</p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
          <p className="text-lg font-bold text-blue-600">1.85</p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
          <p className="text-lg font-bold text-red-600">-8.1%</p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">Win Rate</p>
          <p className="text-lg font-bold text-purple-600">67%</p>
        </div>
      </div>
    </div>
  );
};

export const RiskMetricsChart: React.FC<LiveChartsProps> = ({ data }) => {
  const [selectedMetrics, setSelectedMetrics] = useState(['p_ruin', 'var_95']);

  const metrics = [
    { key: 'p_ruin', label: 'P(ruin)', color: '#ef4444' },
    { key: 'var_95', label: 'VaR 95%', color: '#f59e0b' },
    { key: 'sharpe_ratio', label: 'Sharpe Ratio', color: '#10b981' }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Risk Metrics Evolution
        </h3>

        <div className="flex flex-wrap gap-2">
          {metrics.map(metric => (
            <label key={metric.key} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedMetrics.includes(metric.key)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedMetrics([...selectedMetrics, metric.key]);
                  } else {
                    setSelectedMetrics(selectedMetrics.filter(m => m !== metric.key));
                  }
                }}
                className="rounded"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">{metric.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(tick) => new Date(tick).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              stroke="#6b7280"
            />
            <YAxis stroke="#6b7280" />
            <Tooltip
              labelFormatter={(label) => new Date(label).toLocaleString()}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '6px'
              }}
            />
            <Legend />
            {metrics.map(metric => (
              selectedMetrics.includes(metric.key) && (
                <Line
                  key={metric.key}
                  type="monotone"
                  dataKey={metric.key}
                  stroke={metric.color}
                  strokeWidth={2}
                  dot={false}
                  name={metric.label}
                />
              )
            ))}
            <ReferenceLine y={0.15} stroke="#ef4444" strokeDasharray="5 5" label="Risk Threshold" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
export const LiveChartsEnhanced = TradingTerminalChart;
