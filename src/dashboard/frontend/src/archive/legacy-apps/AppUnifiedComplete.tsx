import React, { useState, useEffect } from 'react';
import './index.css';
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
  Legend
} from 'recharts';

// Import existing dashboard components
import {
  MetricCard,
  PRuinCard,
  PortfolioValueCard,
  VarCard,
  SharpeRatioCard,
  DrawdownCard
} from './components/MetricCard';
import { PositionTable } from './components/PositionTable';
import { AlertList } from './components/AlertList';
import { RiskChart } from './components/RiskChart';
import { EducationHub } from './components/education/EducationHub';

interface Metrics {
  portfolio_value: number;
  p_ruin: number;
  var_95: number;
  var_99: number;
  sharpe_ratio: number;
  max_drawdown: number;
  daily_pnl: number;
  unrealized_pnl: number;
  positions_count: number;
}

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'success' | 'error';
  title: string;
  message: string;
  timestamp: Date;
}

// Enhanced mock data
const mockMetrics: Metrics = {
  portfolio_value: 25432.18,
  p_ruin: 0.12,
  var_95: 1287.50,
  var_99: 2103.25,
  sharpe_ratio: 1.85,
  max_drawdown: 0.08,
  daily_pnl: 342.50,
  unrealized_pnl: 1205.30,
  positions_count: 5
};

const mockPositions: Position[] = [
  { symbol: 'SPY', quantity: 50, entry_price: 445.20, current_price: 448.75, pnl: 177.50, pnl_percent: 0.80 },
  { symbol: 'ULTY', quantity: 100, entry_price: 32.15, current_price: 33.20, pnl: 105.00, pnl_percent: 3.27 },
  { symbol: 'AMDY', quantity: 75, entry_price: 28.90, current_price: 29.15, pnl: 18.75, pnl_percent: 0.87 },
  { symbol: 'VTIP', quantity: 30, entry_price: 49.80, current_price: 49.95, pnl: 4.50, pnl_percent: 0.30 },
  { symbol: 'IAU', quantity: 200, entry_price: 41.25, current_price: 41.90, pnl: 130.00, pnl_percent: 1.58 }
];

const mockAlerts: Alert[] = [
  { id: '1', severity: 'warning', title: 'High P(ruin)', message: 'P(ruin) approaching threshold at 12%', timestamp: new Date() },
  { id: '2', severity: 'info', title: 'Market Update', message: 'SPY showing bullish momentum', timestamp: new Date() },
  { id: '3', severity: 'success', title: 'Trade Executed', message: 'Successfully bought 50 shares of SPY', timestamp: new Date() }
];

// Generate realistic chart data
const generateChartData = () => {
  return Array.from({ length: 20 }, (_, i) => ({
    timestamp: Date.now() - (19 - i) * 3600000,
    portfolio_value: 25000 + Math.random() * 1000 + i * 50,
    p_ruin: 0.08 + Math.random() * 0.08,
    sharpe_ratio: 1.5 + Math.random() * 0.7,
    var_95: 1200 + Math.random() * 200
  }));
};

/**
 * Complete Unified Trading Dashboard
 * Combines all functionality from original and enhanced versions
 */
const AppUnifiedComplete: React.FC = () => {
  // Core state
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'trading' | 'ai' | 'education'>('overview');
  const [metrics, setMetrics] = useState<Metrics>(mockMetrics);
  const [positions, setPositions] = useState<Position[]>(mockPositions);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
  const [chartData, setChartData] = useState(generateChartData());

  // Connection states
  const [apiConnected, setApiConnected] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);

  // API connection test
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/health');
        if (response.ok) {
          setApiConnected(true);
          // Fetch real data
          const metricsResponse = await fetch('http://localhost:8000/api/metrics/current');
          if (metricsResponse.ok) {
            const data = await metricsResponse.json();
            setMetrics(prev => ({ ...prev, ...data }));
          }
        }
      } catch (error) {
        console.log('Backend not available, using mock data');
        setApiConnected(false);
      }
    };

    testConnection();
    const interval = setInterval(testConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/unified_dashboard');

        ws.onopen = () => {
          console.log('WebSocket connected');
          setWsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'metrics_update') {
              setMetrics(prev => ({ ...prev, ...data.metrics }));
            } else if (data.type === 'position_update') {
              setPositions(data.positions || mockPositions);
            } else if (data.type === 'alert') {
              setAlerts(prev => [data.alert, ...prev].slice(0, 10));
            } else if (data.type === 'chart_update') {
              setChartData(prev => [...prev.slice(1), data.point]);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setWsConnected(false);
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setWsConnected(false);
          // Reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        return ws;
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        return null;
      }
    };

    const ws = connectWebSocket();

    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  // Simulate real-time updates when no WebSocket
  useEffect(() => {
    if (!wsConnected) {
      const interval = setInterval(() => {
        // Update metrics with small random changes
        setMetrics(prev => ({
          ...prev,
          portfolio_value: prev.portfolio_value + (Math.random() - 0.5) * 100,
          daily_pnl: prev.daily_pnl + (Math.random() - 0.5) * 20,
          p_ruin: Math.max(0, Math.min(1, prev.p_ruin + (Math.random() - 0.5) * 0.005))
        }));

        // Update chart data
        setChartData(prev => [
          ...prev.slice(1),
          {
            timestamp: Date.now(),
            portfolio_value: metrics.portfolio_value + (Math.random() - 0.5) * 100,
            p_ruin: metrics.p_ruin,
            sharpe_ratio: metrics.sharpe_ratio + (Math.random() - 0.5) * 0.1,
            var_95: metrics.var_95 + (Math.random() - 0.5) * 50
          }
        ]);
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [wsConnected, metrics]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'analysis', label: 'Inequality Analysis', icon: '📈' },
    { id: 'trading', label: 'Trading Terminal', icon: '💹' },
    { id: 'ai', label: 'AI Status', icon: '🤖' },
    { id: 'education', label: 'Education', icon: '🎓' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Gary×Taleb AI-Enhanced Trading System
              </h1>
            </div>

            {/* Status Indicators */}
            <div className="flex items-center space-x-6">
              <div className={`flex items-center space-x-2 ${apiConnected ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-3 h-3 rounded-full ${apiConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium">API</span>
              </div>
              <div className={`flex items-center space-x-2 ${wsConnected ? 'text-green-600' : 'text-yellow-600'}`}>
                <div className={`w-3 h-3 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                <span className="text-sm font-medium">Live Data</span>
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                ${metrics.portfolio_value?.toLocaleString('en-US', { minimumFractionDigits: 2 }) || '0'}
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="border-t border-gray-200 dark:border-gray-700">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'overview' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Portfolio Dashboard
            </h2>

            {/* Main Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <PortfolioValueCard value={metrics.portfolio_value} />
              <PRuinCard value={metrics.p_ruin} />
              <VarCard value={metrics.var_95} />
              <SharpeRatioCard value={metrics.sharpe_ratio} />
            </div>

            {/* Secondary Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <DrawdownCard value={metrics.max_drawdown} />
              <MetricCard
                title="Daily P&L"
                value={`$${metrics.daily_pnl?.toFixed(2) || '0.00'}`}
                trend={metrics.daily_pnl >= 0 ? "up" : "down"}
              />
              <MetricCard
                title="Unrealized P&L"
                value={`$${metrics.unrealized_pnl?.toFixed(2) || '0.00'}`}
                trend="up"
              />
            </div>

            {/* Charts and Tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Portfolio Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Portfolio Performance
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(tick) => new Date(tick).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                      stroke="#6b7280"
                    />
                    <YAxis
                      tickFormatter={(tick) => `$${(tick / 1000).toFixed(1)}k`}
                      stroke="#6b7280"
                    />
                    <Tooltip
                      formatter={(value: any) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                      contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px' }}
                    />
                    <Area
                      type="monotone"
                      dataKey="portfolio_value"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#colorValue)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Risk Metrics Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Risk Metrics
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(tick) => new Date(tick).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                      stroke="#6b7280"
                    />
                    <YAxis stroke="#6b7280" />
                    <Tooltip
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                      contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px' }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="p_ruin"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="P(ruin)"
                    />
                    <Line
                      type="monotone"
                      dataKey="sharpe_ratio"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                      name="Sharpe Ratio"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PositionTable positions={positions} />
              <AlertList alerts={alerts} />
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Inequality Analysis & Mathematical Framework
            </h2>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    DPI (Distributional Pressure Index)
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mt-2">
                    Advanced inequality analysis using Gary's mathematical framework.
                    DPI = Sum(w_i * (E[actual_i] - E[consensus_i])^2) where consensus blind spots create opportunity.
                  </p>
                  <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">85.3</div>
                    <div className="text-sm text-gray-500">Strong mispricing signal detected</div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Narrative Gap Analysis
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mt-2">
                    Tracking disconnect between consensus narratives and mathematical reality.
                  </p>
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="text-lg font-bold text-red-600">High</div>
                      <div className="text-sm text-gray-500">Wealth Concentration</div>
                    </div>
                    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="text-lg font-bold text-yellow-600">Medium</div>
                      <div className="text-sm text-gray-500">Market Pricing</div>
                    </div>
                    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="text-lg font-bold text-green-600">Low</div>
                      <div className="text-sm text-gray-500">Policy Response</div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Repricing Potential
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mt-2">
                    EVT-based tail risk assessment for antifragile positioning.
                  </p>
                  <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">12.7%</div>
                    <div className="text-sm text-gray-500">Expected repricing magnitude</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'trading' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Contrarian Trading Terminal
            </h2>

            {/* Market Overview */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
              {['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU'].map(symbol => (
                <div key={symbol} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">{symbol}</p>
                      <p className="text-lg font-bold">
                        ${(400 + Math.random() * 50).toFixed(2)}
                      </p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${
                      Math.random() > 0.5 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {Math.random() > 0.5 ? '+' : '-'}{(Math.random() * 3).toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Barbell Strategy Status */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Barbell Strategy Allocation
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Safe Bucket (80%)</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">VTIP (Treasury)</span>
                      <span className="text-sm font-medium">45%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">IAU (Gold)</span>
                      <span className="text-sm font-medium">25%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Cash</span>
                      <span className="text-sm font-medium">10%</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">Risky Bucket (20%)</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">SPY (Momentum)</span>
                      <span className="text-sm font-medium">12%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">ULTY (Contrarian)</span>
                      <span className="text-sm font-medium">5%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">AMDY (Tail)</span>
                      <span className="text-sm font-medium">3%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Trading Signals */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                AI-Generated Contrarian Opportunities
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">Asset</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">Signal</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">DPI Score</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">Conviction</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 px-4 font-medium">ULTY</td>
                      <td className="py-3 px-4">
                        <span className="inline-flex px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                          BUY
                        </span>
                      </td>
                      <td className="py-3 px-4">85.3</td>
                      <td className="py-3 px-4">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-green-600 h-2 rounded-full" style={{width: '78%'}}></div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <button className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
                          Execute
                        </button>
                      </td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-800">
                      <td className="py-3 px-4 font-medium">AMDY</td>
                      <td className="py-3 px-4">
                        <span className="inline-flex px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded">
                          WATCH
                        </span>
                      </td>
                      <td className="py-3 px-4">67.2</td>
                      <td className="py-3 px-4">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-yellow-600 h-2 rounded-full" style={{width: '54%'}}></div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <button className="px-3 py-1 bg-gray-400 text-white rounded text-sm">
                          Monitor
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'ai' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              AI Calibration & Performance
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Calibration Metrics
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Brier Score</span>
                    <span className="text-lg font-bold text-green-600">0.187</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">PIT Uniformity</span>
                    <span className="text-lg font-bold text-blue-600">0.923</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Prediction Accuracy</span>
                    <span className="text-lg font-bold text-purple-600">74.3%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Kelly Safety Factor</span>
                    <span className="text-lg font-bold text-yellow-600">0.25</span>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Mathematical Framework Status
                </h3>
                <div className="space-y-3">
                  {[
                    { name: 'DPI Engine', status: 'active' },
                    { name: 'Narrative Gap Tracking', status: 'active' },
                    { name: 'Repricing Potential', status: 'active' },
                    { name: 'Kelly Optimization', status: 'active' },
                    { name: 'EVT Risk Management', status: 'active' },
                    { name: 'Barbell Constraints', status: 'active' }
                  ].map((system) => (
                    <div key={system.name} className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">{system.name}</span>
                      <span className="inline-flex px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                        {system.status.toUpperCase()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                AI Decision Learning Progress
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    stroke="#6b7280"
                  />
                  <YAxis stroke="#6b7280" />
                  <Tooltip
                    labelFormatter={(label) => new Date(label).toLocaleDateString()}
                    contentStyle={{ backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="sharpe_ratio"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                    name="AI Decision Quality"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'education' && <EducationHub />}
      </div>
    </div>
  );
};

export default AppUnifiedComplete;