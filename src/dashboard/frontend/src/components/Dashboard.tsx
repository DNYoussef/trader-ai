import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import toast from 'react-hot-toast';

// Redux hooks and selectors
import { useAppDispatch, useAppSelector } from '@/store';
import {
  selectRiskMetrics,
  selectPositions,
  selectIsConnected,
  selectHistoricalData,
  selectLatestChartData,
} from '@/store/dashboardSlice';
import {
  selectAllAlerts,
  selectUnreadCount,
  selectAlertSummary,
} from '@/store/alertsSlice';
import {
  selectDarkMode,
  selectCompactView,
  selectShowAdvancedMetrics,
} from '@/store/settingsSlice';

// Hooks and services
import { useWebSocket, useWebSocketSubscription } from '@/hooks/useWebSocket';
import dashboardAPI from '@/services/api';

// Components
import {
  MetricCard,
  MetricGrid,
  PRuinCard,
  PortfolioValueCard,
  VarCard,
  SharpeRatioCard,
  DrawdownCard,
} from './MetricCard';
import { PositionTable } from './PositionTable';
import { AlertList } from './AlertList';
import {
  RiskChart,
  ChartGrid,
  PortfolioValueChart,
  PRuinChart,
  VarChart,
  SharpeChart,
  DrawdownChart,
} from './RiskChart';

// Connection status indicator
const ConnectionStatus: React.FC<{ connected: boolean; className?: string }> = ({
  connected,
  className,
}) => (
  <div className={clsx('flex items-center space-x-2', className)}>
    <div
      className={clsx(
        'w-3 h-3 rounded-full',
        connected ? 'bg-success-500 animate-pulse' : 'bg-danger-500'
      )}
    />
    <span className="text-sm text-gray-600 dark:text-gray-300">
      {connected ? 'Connected' : 'Disconnected'}
    </span>
  </div>
);

// Header component
const DashboardHeader: React.FC<{
  connected: boolean;
  unreadAlerts: number;
  onToggleSettings: () => void;
}> = ({ connected, unreadAlerts, onToggleSettings }) => (
  <motion.div
    initial={{ opacity: 0, y: -20 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700"
  >
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between h-16">
        {/* Title */}
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Risk Dashboard
          </h1>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Gary×Taleb Trading System
          </span>
        </div>

        {/* Status and controls */}
        <div className="flex items-center space-x-4">
          <ConnectionStatus connected={connected} />

          {/* Alert indicator */}
          {unreadAlerts > 0 && (
            <div className="relative">
              <div className="w-6 h-6 bg-danger-500 rounded-full flex items-center justify-center">
                <span className="text-xs font-medium text-white">
                  {unreadAlerts > 9 ? '9+' : unreadAlerts}
                </span>
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-danger-600 rounded-full animate-ping" />
            </div>
          )}

          {/* Settings button */}
          <button
            onClick={onToggleSettings}
            className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  </motion.div>
);

// Loading overlay component
const LoadingOverlay: React.FC<{ show: boolean }> = ({ show }) => {
  if (!show) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-50"
    >
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-xl">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 border-3 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-900 dark:text-gray-100">Loading dashboard...</span>
        </div>
      </div>
    </motion.div>
  );
};

// Main Dashboard component
export const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch();

  // Redux state
  const riskMetrics = useAppSelector(selectRiskMetrics);
  const positions = useAppSelector(selectPositions);
  const isConnected = useAppSelector(selectIsConnected);
  const historicalData = useAppSelector(selectHistoricalData);
  const alerts = useAppSelector(selectAllAlerts);
  const unreadCount = useAppSelector(selectUnreadCount);
  const alertSummary = useAppSelector(selectAlertSummary);
  const darkMode = useAppSelector(selectDarkMode);
  const compactView = useAppSelector(selectCompactView);
  const showAdvancedMetrics = useAppSelector(selectShowAdvancedMetrics);

  // Chart data selectors
  const portfolioChartData = useAppSelector(selectLatestChartData('portfolio_values', 100));
  const pRuinChartData = useAppSelector(selectLatestChartData('p_ruin_history', 100));
  const varChartData = useAppSelector(selectLatestChartData('var_history', 100));
  const sharpeChartData = useAppSelector(selectLatestChartData('sharpe_history', 100));
  const drawdownChartData = useAppSelector(selectLatestChartData('drawdown_history', 100));

  // Local state
  const [loading, setLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection
  const websocket = useWebSocket('ws://localhost:8000/ws', {
    auto_connect: true,
    reconnect_attempts: 5,
    heartbeat_interval: 30000,
  });

  // Subscribe to all data types
  useWebSocketSubscription('risk_metrics', websocket);
  useWebSocketSubscription('positions', websocket);
  useWebSocketSubscription('alerts', websocket);

  // Initial data load
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Load initial data from API
        const [metrics, positionsData, alertsData] = await Promise.all([
          dashboardAPI.getCurrentMetrics().catch(() => null),
          dashboardAPI.getPositions().catch(() => []),
          dashboardAPI.getAlerts().catch(() => []),
        ]);

        // Dispatch to store
        if (metrics || positionsData.length > 0) {
          dispatch({
            type: 'dashboard/setInitialData',
            payload: {
              risk_metrics: metrics,
              positions: positionsData
            }
          });
        }

        // Load alerts
        alertsData.forEach(alert => {
          dispatch({ type: 'alerts/addAlert', payload: alert });
        });

      } catch (err) {
        console.error('Failed to load initial data:', err);
        setError('Failed to load dashboard data');
        toast.error('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, [dispatch]);

  // Handle alert acknowledgment
  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await dashboardAPI.acknowledgeAlert(alertId);
      dispatch({ type: 'alerts/acknowledgeAlert', payload: alertId });
      toast.success('Alert acknowledged');
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
      toast.error('Failed to acknowledge alert');
    }
  };

  // Handle position row click
  const handlePositionClick = (position: any) => {
    toast.info(`Position details for ${position.symbol}`, {
      duration: 2000,
    });
  };

  // Connection error handling
  useEffect(() => {
    if (!isConnected && !loading) {
      toast.error('Connection lost. Attempting to reconnect...', {
        id: 'connection-error',
      });
    } else if (isConnected) {
      toast.dismiss('connection-error');
    }
  }, [isConnected, loading]);

  // Apply dark mode class to body
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-danger-100 dark:bg-danger-900 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-danger-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Dashboard Error
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
          >
            Reload Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <LoadingOverlay show={loading} />

      {/* Header */}
      <DashboardHeader
        connected={isConnected}
        unreadAlerts={unreadCount}
        onToggleSettings={() => setShowSettings(!showSettings)}
      />

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="space-y-6">
          {/* Key metrics */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
              Risk Metrics
            </h2>
            <MetricGrid columns={compactView ? 2 : 4}>
              {riskMetrics && (
                <>
                  <PRuinCard pRuin={riskMetrics.p_ruin} />
                  <PortfolioValueCard value={riskMetrics.portfolio_value} />
                  <VarCard var95={riskMetrics.var_95} portfolioValue={riskMetrics.portfolio_value} />
                  <SharpeRatioCard sharpe={riskMetrics.sharpe_ratio} />
                  {showAdvancedMetrics && (
                    <>
                      <DrawdownCard drawdown={riskMetrics.max_drawdown} />
                      <MetricCard
                        metric={{
                          title: 'Expected Shortfall',
                          value: riskMetrics.expected_shortfall,
                          format: 'currency',
                          description: 'Average loss beyond VaR',
                        }}
                      />
                      <MetricCard
                        metric={{
                          title: 'Beta',
                          value: riskMetrics.beta,
                          format: 'ratio',
                          description: 'Market correlation',
                        }}
                      />
                      <MetricCard
                        metric={{
                          title: 'Volatility',
                          value: riskMetrics.volatility,
                          format: 'percentage',
                          description: 'Annualized volatility',
                        }}
                      />
                    </>
                  )}
                </>
              )}
            </MetricGrid>
          </motion.section>

          {/* Charts */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
              Historical Trends
            </h2>
            <ChartGrid columns={compactView ? 1 : 2}>
              <PortfolioValueChart data={portfolioChartData} />
              <PRuinChart data={pRuinChartData} />
              {showAdvancedMetrics && (
                <>
                  <VarChart data={varChartData} />
                  <SharpeChart data={sharpeChartData} />
                  <DrawdownChart data={drawdownChartData} />
                </>
              )}
            </ChartGrid>
          </motion.section>

          {/* Positions and alerts */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Positions */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="lg:col-span-2"
            >
              <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Current Positions
              </h2>
              <PositionTable
                positions={positions}
                onRowClick={handlePositionClick}
                sortBy="market_value"
                sortDirection="desc"
              />
            </motion.section>

            {/* Alerts */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <AlertList
                alerts={alerts}
                onAcknowledge={handleAcknowledgeAlert}
                maxAlerts={20}
              />
            </motion.section>
          </div>
        </div>
      </main>
    </div>
  );
};