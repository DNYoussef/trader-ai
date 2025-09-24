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

// Import components from current structure (will be reorganized)
import {
  MetricCard,
  PRuinCard,
  PortfolioValueCard,
  VarCard,
  SharpeRatioCard,
  DrawdownCard
} from './components/MetricCardSimple';
import { PositionTable } from './components/PositionTableSimple';
import { AlertList } from './components/AlertListSimple';
import { EducationHub } from './components/education/EducationHub';
import { LiveChartsEnhanced } from './components/LiveChartsEnhanced';
import { QuickTrade } from './components/QuickTrade';
import { AISignals } from './components/AISignals';
import { RecentAlerts } from './components/RecentAlerts';
import { GateProgression } from './components/GateProgression';
import { TradingJourney } from './components/TradingJourney';
import { InequalityPanelWrapper } from './components/InequalityPanelWrapper';
import { ContrarianTradesWrapper } from './components/ContrarianTradesWrapper';

interface AppMode {
  id: string;
  name: string;
  description: string;
  features: string[];
}

// Types are imported from hooks - no need to duplicate

// App modes for different use cases
const APP_MODES: AppMode[] = [
  {
    id: 'simple',
    name: 'Simple Dashboard',
    description: 'Basic risk dashboard with core metrics',
    features: ['metrics', 'positions', 'alerts']
  },
  {
    id: 'enhanced',
    name: 'Enhanced Trading',
    description: 'Full trading dashboard with AI features',
    features: ['metrics', 'positions', 'alerts', 'charts', 'ai', 'analysis']
  },
  {
    id: 'educational',
    name: 'Learning Mode',
    description: 'Educational interface with Guild of the Rose integration',
    features: ['metrics', 'education', 'ai']
  },
  {
    id: 'professional',
    name: 'Professional Trader',
    description: 'Complete professional trading environment',
    features: ['metrics', 'positions', 'alerts', 'charts', 'ai', 'analysis', 'trading']
  }
];

// Import data hooks for real data
import { useTradingData } from './hooks/useTradingData';
import { useAIData } from './hooks/useAIData';

/**
 * Unified App Component - Consolidates all implementations
 * Features adaptive UI based on selected mode
 */
const AppUnified: React.FC = () => {
  // Core state
  const [appMode, setAppMode] = useState<string>('enhanced');
  const [activeTab, setActiveTab] = useState<'overview' | 'terminal' | 'analysis' | 'learn' | 'progress'>('overview');

  // Use real data hooks instead of mock data
  const {
    metrics,
    positions,
    alerts,
    gateStatus,
    isConnected: apiConnected,
    loading: tradingLoading,
    error: tradingError
  } = useTradingData();

  const {
    timesfmVolatility,
    fingptSentiment,
    enhanced32D,
    loading: aiLoading,
    error: aiError
  } = useAIData();

  // WebSocket connection state
  const [wsConnected, setWsConnected] = useState(false);

  // Generate chart data from real metrics
  const [chartData, setChartData] = useState<any[]>([]);

  const currentMode = APP_MODES.find(mode => mode.id === appMode) || APP_MODES[1];

  // Update chart data when metrics change
  useEffect(() => {
    if (metrics) {
      setChartData(prev => {
        const newPoint = {
          timestamp: Date.now(),
          portfolio_value: metrics.portfolio_value,
          p_ruin: metrics.p_ruin,
          sharpe_ratio: metrics.sharpe_ratio,
          var_95: metrics.var_95
        };

        // Keep last 20 points
        const updated = [...prev.slice(-19), newPoint];
        return updated.length > 0 ? updated : Array(20).fill(newPoint);
      });
    }
  }, [metrics]);

  // WebSocket connection status is handled by useTradingData hook
  useEffect(() => {
    // The useTradingData hook manages WebSocket connection
    setWsConnected(apiConnected);
  }, [apiConnected]);

  const getAvailableTabs = () => {
    const features = currentMode.features;
    const tabs = [];

    tabs.push({ id: 'overview', label: 'Overview', icon: '📊' });
    if (features.includes('charts') || features.includes('trading')) {
      tabs.push({ id: 'terminal', label: 'Trading Terminal', icon: '💹' });
    }
    if (features.includes('analysis')) {
      tabs.push({ id: 'analysis', label: 'Analysis', icon: '📈' });
    }
    if (features.includes('education')) {
      tabs.push({ id: 'learn', label: 'Learn', icon: '📚' });
    }
    tabs.push({ id: 'progress', label: 'Progress', icon: '🏆' });

    return tabs;
  };

  const tabs = getAvailableTabs();

  // Ensure active tab is available in current mode
  useEffect(() => {
    const availableTabIds = tabs.map(tab => tab.id);
    if (!availableTabIds.includes(activeTab)) {
      setActiveTab(availableTabIds[0] as any);
    }
  }, [appMode, tabs, activeTab]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Gary×Taleb AI-Enhanced Trading System
              </h1>

              {/* Mode Selector */}
              <select
                value={appMode}
                onChange={(e) => setAppMode(e.target.value)}
                className="ml-4 px-3 py-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-sm"
              >
                {APP_MODES.map(mode => (
                  <option key={mode.id} value={mode.id}>
                    {mode.name}
                  </option>
                ))}
              </select>
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
          {tabs.length > 1 && (
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
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Mode Description */}
        <div className="mb-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <h2 className="text-lg font-medium text-blue-900 dark:text-blue-100">
            {currentMode.name}
          </h2>
          <p className="text-blue-700 dark:text-blue-300 text-sm mt-1">
            {currentMode.description}
          </p>
          <div className="mt-2 flex flex-wrap gap-1">
            {currentMode.features.map(feature => (
              <span
                key={feature}
                className="inline-flex px-2 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 rounded"
              >
                {feature}
              </span>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Portfolio Dashboard
            </h2>

            {/* Main Metrics Grid */}
            {currentMode.features.includes('metrics') && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <PortfolioValueCard value={metrics.portfolio_value} />
                <PRuinCard value={metrics.p_ruin} />
                <VarCard value={metrics.var_95} />
                <SharpeRatioCard value={metrics.sharpe_ratio} />
              </div>
            )}

            {/* Secondary Metrics */}
            {currentMode.features.includes('metrics') && (
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
            )}

            {/* Enhanced Overview with Real-time Charts and AI Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Main Portfolio Chart - 2/3 width */}
              <div className="lg:col-span-2">
                <LiveChartsEnhanced data={chartData} isRealTime={wsConnected} />
              </div>

              {/* Side Panel - 1/3 width */}
              <div className="space-y-6">
                <AISignals dpiScore={85} signal="BUY" confidence={92} />
                <QuickTrade symbol="SPY" />
              </div>
            </div>

            {/* Enhanced Tables with Recent Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {currentMode.features.includes('positions') && <PositionTable positions={positions} />}
              <RecentAlerts alerts={alerts?.map(alert => ({
                id: alert?.id || `alert-${Math.random()}`,
                type: (alert?.severity || 'info') as 'warning' | 'info' | 'success' | 'error',
                title: alert?.title || 'Alert',
                message: alert?.message || '',
                timestamp: alert?.timestamp ? alert.timestamp.toLocaleTimeString() : new Date().toLocaleTimeString()
              })) || []} />
            </div>
          </div>
        )}

        {activeTab === 'terminal' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Professional Trading Terminal
            </h2>
            <LiveChartsEnhanced data={chartData} isRealTime={wsConnected} showTerminal={true} />
          </div>
        )}

        {activeTab === 'analysis' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Inequality Analysis & Contrarian Opportunities
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <InequalityPanelWrapper />
              <ContrarianTradesWrapper />
            </div>
          </div>
        )}

        {activeTab === 'learn' && <EducationHub />}

        {activeTab === 'progress' && (
          <div>
            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">
              Your Trading Journey
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TradingJourney />
              <GateProgression currentCapital={metrics.portfolio_value} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AppUnified;