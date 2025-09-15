import React, { useState, useEffect } from 'react';
import './index.css';
import { motion, AnimatePresence } from 'framer-motion';
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

// Import ALL components from original dashboard
import { MetricCard, PRuinCard, PortfolioValueCard, VarCard, SharpeRatioCard, DrawdownCard } from './components/MetricCard';
import { PositionTable } from './components/PositionTable';
import { AlertList } from './components/AlertList';
import { RiskChart } from './components/RiskChart';

// Import ALL components from enhanced dashboard
import { InequalityPanel } from './components/InequalityPanel';
import { ContrarianTrades } from './components/ContrarianTrades';
import { AIStatusPanel } from './components/enhanced/AIStatusPanel';
import { DecisionTreeBuilder } from './components/enhanced/DecisionTreeBuilder';
import { CalibrationTrainer } from './components/enhanced/CalibrationTrainer';
import { EducationHub } from './components/education/EducationHub';

// Navigation Icons
const DashboardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const AnalysisIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TradingIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
  </svg>
);

const EducationIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const AIIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

/**
 * Unified App - Single Interface with ALL Components
 *
 * Combines original risk dashboard components with enhanced AI analysis
 * in one comprehensive interface without toggle mechanism.
 */
const UnifiedApp: React.FC = () => {
  // State for all data types
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analysis' | 'trading' | 'education' | 'ai'>('dashboard');

  // Enhanced mock data from the most complete version
  const mockMetrics = {
    portfolio_value: 25432.18,
    p_ruin: 0.12,
    var_95: 1287.50,
    var_99: 2103.25,
    sharpe_ratio: 1.85,
    max_drawdown: 0.08,
    daily_pnl: 342.50,
    unrealized_pnl: 1205.30,
    positions_count: 5,
    volatility: 0.1438,
    beta: 0.92,
    expected_shortfall: 10173.46
  };

  const mockPositions = [
    { symbol: 'SPY', quantity: 50, entry_price: 445.20, current_price: 448.75, pnl: 177.50, pnl_percent: 0.80 },
    { symbol: 'ULTY', quantity: 100, entry_price: 32.15, current_price: 33.20, pnl: 105.00, pnl_percent: 3.27 },
    { symbol: 'AMDY', quantity: 75, entry_price: 28.90, current_price: 29.15, pnl: 18.75, pnl_percent: 0.87 },
    { symbol: 'VTIP', quantity: 30, entry_price: 49.80, current_price: 49.95, pnl: 4.50, pnl_percent: 0.30 },
    { symbol: 'IAU', quantity: 200, entry_price: 41.25, current_price: 41.90, pnl: 130.00, pnl_percent: 1.58 }
  ];

  const mockAlerts = [
    { id: '1', severity: 'warning', title: 'High P(ruin)', message: 'P(ruin) approaching threshold at 12%', timestamp: new Date() },
    { id: '2', severity: 'info', title: 'Market Update', message: 'SPY showing bullish momentum', timestamp: new Date() },
    { id: '3', severity: 'success', title: 'Trade Executed', message: 'Successfully bought 50 shares of SPY', timestamp: new Date() }
  ];

  const mockChartData = Array.from({ length: 20 }, (_, i) => ({
    timestamp: Date.now() - (19 - i) * 3600000,
    value: 25000 + Math.random() * 1000,
    p_ruin: 0.08 + Math.random() * 0.08
  }));

  // Original dashboard data (starts with mock, gets updated by API)
  const [metrics, setMetrics] = useState(mockMetrics);
  const [positions, setPositions] = useState(mockPositions);
  const [alerts, setAlerts] = useState(mockAlerts);
  const [chartData, setChartData] = useState(mockChartData);

  // Enhanced dashboard data
  const [inequalityData, setInequalityData] = useState<any>(null);
  const [contrarianData, setContrarianData] = useState<any>(null);
  const [aiData, setAiData] = useState<any>(null);

  // Connection status
  const [wsConnected, setWsConnected] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);

  // Unified data fetching
  const fetchAllData = async () => {
    try {
      // Fetch original dashboard metrics
      const metricsResponse = await fetch('http://localhost:8000/api/metrics/current');
      const metricsData = await metricsResponse.json();
      setMetrics(metricsData);

      // Fetch inequality analysis data
      const inequalityResponse = await fetch('http://localhost:8000/api/inequality/data');
      const inequality = await inequalityResponse.json();
      setInequalityData(inequality);

      // Fetch contrarian opportunities
      const contrarianResponse = await fetch('http://localhost:8000/api/contrarian/opportunities');
      const contrarian = await contrarianResponse.json();
      setContrarianData(contrarian);

      // Fetch AI status data
      try {
        const aiResponse = await fetch('http://localhost:8000/api/ai/status');
        const ai = await aiResponse.json();
        setAiData(ai);
      } catch (error) {
        console.log('AI status endpoint not available, using mock data');
        setAiData({
          calibration: { brier_score: 0.23, accuracy: 0.87 },
          predictions: { total: 150, resolved: 142 },
          confidence: 0.74
        });
      }

      setApiConnected(true);
    } catch (error) {
      console.error('Failed to fetch data:', error);
      setApiConnected(false);
    }
  };

  // Unified WebSocket connection
  const setupWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/unified_dashboard_${Date.now()}`);

    ws.onopen = () => {
      console.log('Unified WebSocket connected');
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
      // Attempt to reconnect after 5 seconds
      setTimeout(setupWebSocket, 5000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
  };

  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'metrics_update':
        setMetrics(prev => ({ ...prev, ...message.data }));
        break;
      case 'position_update':
        setPositions(message.positions || []);
        break;
      case 'alert':
        setAlerts(prev => [message.alert, ...prev].slice(0, 10));
        break;
      case 'inequality_update':
        if (inequalityData) {
          setInequalityData(prev => ({
            ...prev,
            metrics: { ...prev.metrics, ...message.data }
          }));
        }
        break;
      case 'contrarian_update':
        setContrarianData(prev => ({
          ...prev,
          opportunities: message.opportunities || prev?.opportunities
        }));
        break;
      case 'ai_signal_update':
        console.log('AI signal update:', message);
        break;
      default:
        console.log('Unknown WebSocket message type:', message.type);
    }
  };

  // Initialize data and connections
  useEffect(() => {
    fetchAllData();
    setupWebSocket();

    // Set up periodic data refresh
    const interval = setInterval(fetchAllData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: DashboardIcon },
    { id: 'analysis', label: 'Analysis', icon: AnalysisIcon },
    { id: 'trading', label: 'Trading', icon: TradingIcon },
    { id: 'ai', label: 'AI Status', icon: AIIcon },
    { id: 'education', label: 'Education', icon: EducationIcon }
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
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 ${apiConnected ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium">API</span>
              </div>
              <div className={`flex items-center space-x-2 ${wsConnected ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm font-medium">Live</span>
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                ${metrics.portfolio_value?.toLocaleString() || '0'}
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="border-t border-gray-200 dark:border-gray-700">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                    }`}
                  >
                    <Icon />
                    <span className="ml-2">{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              {/* Main Dashboard Grid - 4 Column Layout */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                {/* Row 1: Core Metrics + AI */}
                <PortfolioValueCard value={metrics.portfolio_value} />
                <PRuinCard value={metrics.p_ruin} />
                <VarCard value={metrics.var_95} />
                <MetricCard
                  title="AI Confidence"
                  value={`${((aiData?.confidence || 0.74) * 100).toFixed(1)}%`}
                  trend="up"
                />

                {/* Row 2: Performance + AI Signals */}
                <SharpeRatioCard value={metrics.sharpe_ratio} />
                <DrawdownCard value={metrics.max_drawdown} />
                <MetricCard
                  title="DPI Signal"
                  value={inequalityData?.metrics?.mathematical_signal_strength ?
                    `${(inequalityData.metrics.mathematical_signal_strength * 100).toFixed(1)}%` : 'Loading...'}
                  trend="up"
                />
                <MetricCard
                  title="Gary Score"
                  value={contrarianData?.opportunities?.[0]?.garyMomentScore ?
                    `${(contrarianData.opportunities[0].garyMomentScore * 100).toFixed(1)}%` : 'Loading...'}
                  trend="up"
                />
              </div>

              {/* Secondary Grid - 2 Column Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Position Table */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Current Positions
                    </h3>
                    <PositionTable positions={positions} />
                  </div>
                </div>

                {/* AI Status Panel */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      AI Calibration Status
                    </h3>
                    <AIStatusPanel />
                  </div>
                </div>
              </div>

              {/* Tertiary Grid - Inequality & Contrarian */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Inequality Analysis
                    </h3>
                    <InequalityPanel />
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Contrarian Opportunities
                    </h3>
                    <ContrarianTrades />
                  </div>
                </div>
              </div>

              {/* Portfolio Performance Chart */}
              <div className="mt-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Portfolio Performance & Risk
                    </h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="timestamp"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
                          />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" />
                          <Tooltip
                            labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                            formatter={(value, name) => [
                              name === 'value' ? `$${value.toLocaleString()}` : `${(value * 100).toFixed(2)}%`,
                              name === 'value' ? 'Portfolio Value' : 'P(ruin)'
                            ]}
                          />
                          <Legend />
                          <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="value"
                            stroke="#2563eb"
                            strokeWidth={2}
                            name="Portfolio Value"
                            dot={false}
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="p_ruin"
                            stroke="#dc2626"
                            strokeWidth={2}
                            name="P(ruin)"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              {/* Alerts Section */}
              <div className="mt-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      System Alerts
                    </h3>
                    <AlertList alerts={alerts} />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'analysis' && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              {/* Analysis Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <MetricCard
                  title="Gini Coefficient"
                  value={inequalityData?.metrics?.giniCoefficient ?
                    inequalityData.metrics.giniCoefficient.toFixed(3) : '0.478'}
                  trend="up"
                />
                <MetricCard
                  title="Top 1% Wealth"
                  value={inequalityData?.metrics?.top1PercentWealth ?
                    `${inequalityData.metrics.top1PercentWealth.toFixed(1)}%` : '31.8%'}
                  trend="up"
                />
                <MetricCard
                  title="Consensus Wrong"
                  value={inequalityData?.metrics?.consensusWrongScore ?
                    `${(inequalityData.metrics.consensusWrongScore * 100).toFixed(1)}%` : '77.1%'}
                  trend="up"
                />
                <MetricCard
                  title="AI Prediction"
                  value={inequalityData?.metrics?.ai_prediction_accuracy ?
                    `${(inequalityData.metrics.ai_prediction_accuracy * 100).toFixed(1)}%` : '87.9%'}
                  trend="up"
                />
              </div>

              {/* Detailed Analysis Panels */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Inequality Analysis
                    </h3>
                    <InequalityPanel />
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Contrarian Opportunities
                    </h3>
                    <ContrarianTrades />
                  </div>
                </div>
              </div>

              {/* Advanced Risk Visualization */}
              <div className="mt-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Risk Analysis & Trends
                    </h3>
                    <RiskChart />
                  </div>
                </div>
              </div>

              {/* Wealth Flow Visualization */}
              <div className="mt-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                  <div className="p-6">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                      Wealth Flow Analysis
                    </h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="timestamp"
                            type="number"
                            scale="time"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
                          />
                          <YAxis />
                          <Tooltip
                            labelFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
                            formatter={(value) => [`$${value.toLocaleString()}`, 'Portfolio Value']}
                          />
                          <Area
                            type="monotone"
                            dataKey="value"
                            stroke="#16a34a"
                            fill="#16a34a"
                            fillOpacity={0.3}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'trading' && (
            <motion.div
              key="trading"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PositionTable positions={positions} />
                <DecisionTreeBuilder />
              </div>
              <div className="mt-6">
                <ContrarianTrades />
              </div>
            </motion.div>
          )}

          {activeTab === 'ai' && (
            <motion.div
              key="ai"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <AIStatusPanel />
                <CalibrationTrainer />
              </div>
            </motion.div>
          )}

          {activeTab === 'education' && (
            <motion.div
              key="education"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              <EducationHub />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default UnifiedApp;