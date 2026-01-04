import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { API_ENDPOINTS, getWebSocketUrl } from '../config/api';

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
  cash_available?: number;
  margin_used?: number;
}

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  market_value?: number;
  weight?: number;
}

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'success' | 'error' | 'critical' | 'high';
  title: string;
  message: string;
  timestamp: Date;
}

interface Gate {
  id: string;
  name: string;
  range: string;
  status: 'completed' | 'current' | 'locked';
  requirements: string;
  progress?: number;
}

interface GateStatus {
  current_gate: string;
  current_capital: number;
  gates: Gate[];
}

interface TradingDataState {
  metrics: Metrics | null;
  positions: Position[];
  alerts: Alert[];
  gateStatus: GateStatus | null;
  isConnected: boolean;
  loading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export const useTradingData = (enableWebSocket: boolean = true) => {
  const [state, setState] = useState<TradingDataState>({
    metrics: null,
    positions: [],
    alerts: [],
    gateStatus: null,
    isConnected: false,
    loading: false,
    error: null,
    lastUpdate: null
  });

  // WebSocket connection for real-time updates
  const websocket = enableWebSocket ? useWebSocket(getWebSocketUrl(), {
    auto_connect: true,
    reconnect_attempts: 5,
    heartbeat_interval: 30000
  }) : null;

  // Fetch initial data from REST endpoints
  const fetchInitialData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // Fetch all trading data in parallel (using relative URLs for production compatibility)
      const [metricsRes, positionsRes, alertsRes, gatesRes] = await Promise.all([
        fetch(API_ENDPOINTS.metrics),
        fetch(API_ENDPOINTS.positions),
        fetch(API_ENDPOINTS.alerts),
        fetch(API_ENDPOINTS.gateStatus)
      ]);

      const [metrics, positions, alerts, gateStatus] = await Promise.all([
        metricsRes.json(),
        positionsRes.json(),
        alertsRes.json(),
        gatesRes.json()
      ]);

      setState({
        metrics: metrics.error ? getDefaultMetrics() : metrics,
        positions: positions.error ? [] : positions,
        alerts: alerts.error ? [] : alerts,
        gateStatus: gateStatus.error ? getDefaultGateStatus() : gateStatus,
        isConnected: true,
        loading: false,
        error: null,
        lastUpdate: new Date()
      });
    } catch (error) {
      console.error('Error fetching trading data:', error);
      setState(prev => ({
        ...prev,
        metrics: getDefaultMetrics(),
        positions: getDefaultPositions(),
        alerts: getDefaultAlerts(),
        gateStatus: getDefaultGateStatus(),
        isConnected: false,
        loading: false,
        error: 'Failed to connect to trading engine. Using demo data.',
        lastUpdate: new Date()
      }));
    }
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (!websocket) return;

    const { connection_status } = websocket;
    setState(prev => ({ ...prev, isConnected: connection_status.connected }));

    // Subscribe to trading data updates
    if (connection_status.connected) {
      websocket.subscribe('metrics');
      websocket.subscribe('positions');
      websocket.subscribe('alerts');
      websocket.subscribe('gates');
    }
  }, [websocket]);

  // Process incoming WebSocket messages (would need to be integrated with useWebSocket hook)
  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'metrics_update':
        setState(prev => ({
          ...prev,
          metrics: message.data,
          lastUpdate: new Date()
        }));
        break;
      case 'position_update':
        setState(prev => ({
          ...prev,
          positions: message.data,
          lastUpdate: new Date()
        }));
        break;
      case 'alert':
        setState(prev => ({
          ...prev,
          alerts: [message.data, ...prev.alerts].slice(0, 20), // Keep last 20 alerts
          lastUpdate: new Date()
        }));
        break;
      case 'gate_update':
        setState(prev => ({
          ...prev,
          gateStatus: message.data,
          lastUpdate: new Date()
        }));
        break;
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  // Periodic refresh fallback if WebSocket is not connected
  useEffect(() => {
    if (!enableWebSocket || !state.isConnected) {
      const interval = setInterval(fetchInitialData, 10000); // Refresh every 10 seconds
      return () => clearInterval(interval);
    }
  }, [enableWebSocket, state.isConnected, fetchInitialData]);

  // Calculate derived metrics
  const getDerivedMetrics = useCallback(() => {
    const { metrics, positions } = state;

    if (!metrics) return null;

    const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0);
    const winningPositions = positions.filter(p => p.pnl > 0).length;
    const losingPositions = positions.filter(p => p.pnl < 0).length;
    const winRate = positions.length > 0 ? winningPositions / positions.length : 0;

    return {
      totalPnL,
      winningPositions,
      losingPositions,
      winRate,
      avgPnL: positions.length > 0 ? totalPnL / positions.length : 0,
      largestWin: Math.max(...positions.map(p => p.pnl), 0),
      largestLoss: Math.min(...positions.map(p => p.pnl), 0)
    };
  }, [state]);

  return {
    ...state,
    derivedMetrics: getDerivedMetrics(),
    refresh: fetchInitialData,
    clearAlerts: () => setState(prev => ({ ...prev, alerts: [] })),
    acknowledgeAlert: (id: string) => {
      setState(prev => ({
        ...prev,
        alerts: prev.alerts.filter(a => a.id !== id)
      }));
    }
  };
};

// Default data providers for fallback
function getDefaultMetrics(): Metrics {
  return {
    portfolio_value: 25000 + Math.random() * 1000,
    p_ruin: 0.08 + Math.random() * 0.05,
    var_95: 1200 + Math.random() * 200,
    var_99: 2000 + Math.random() * 300,
    sharpe_ratio: 1.5 + Math.random() * 0.5,
    max_drawdown: 0.05 + Math.random() * 0.03,
    daily_pnl: (Math.random() - 0.5) * 500,
    unrealized_pnl: (Math.random() - 0.5) * 1000,
    positions_count: 5,
    cash_available: 5000 + Math.random() * 1000,
    margin_used: 15000 + Math.random() * 2000
  };
}

function getDefaultPositions(): Position[] {
  return [
    { symbol: 'SPY', quantity: 50, entry_price: 445.20, current_price: 448.75, pnl: 177.50, pnl_percent: 0.80 },
    { symbol: 'QQQ', quantity: 30, entry_price: 385.50, current_price: 387.25, pnl: 52.50, pnl_percent: 0.45 },
    { symbol: 'IWM', quantity: 40, entry_price: 195.30, current_price: 194.80, pnl: -20.00, pnl_percent: -0.26 },
    { symbol: 'GLD', quantity: 25, entry_price: 185.60, current_price: 186.40, pnl: 20.00, pnl_percent: 0.43 },
    { symbol: 'TLT', quantity: 35, entry_price: 92.40, current_price: 91.80, pnl: -21.00, pnl_percent: -0.65 }
  ];
}

function getDefaultAlerts(): Alert[] {
  const now = new Date();
  return [
    {
      id: '1',
      severity: 'info',
      title: 'Market Open',
      message: 'US markets are now open for trading',
      timestamp: new Date(now.getTime() - 30 * 60000)
    },
    {
      id: '2',
      severity: 'warning',
      title: 'Risk Level Elevated',
      message: 'P(ruin) approaching threshold at 12%',
      timestamp: new Date(now.getTime() - 15 * 60000)
    },
    {
      id: '3',
      severity: 'success',
      title: 'Trade Executed',
      message: 'Successfully bought 50 shares of SPY at $448.75',
      timestamp: new Date(now.getTime() - 5 * 60000)
    }
  ];
}

function getDefaultGateStatus(): GateStatus {
  return {
    current_gate: 'G1',
    current_capital: 342,
    gates: [
      {
        id: 'G0',
        name: 'Gate G0',
        range: '$200-499',
        status: 'completed',
        requirements: 'Initial capital secured'
      },
      {
        id: 'G1',
        name: 'Gate G1',
        range: '$500-999',
        status: 'current',
        requirements: '4 profitable trades required',
        progress: 68
      },
      {
        id: 'G2',
        name: 'Gate G2',
        range: '$1,000-2,499',
        status: 'locked',
        requirements: 'Reach $1,000 capital'
      },
      {
        id: 'G3',
        name: 'Gate G3',
        range: '$2,500-4,999',
        status: 'locked',
        requirements: 'Reach $2,500 capital'
      },
      {
        id: 'G4',
        name: 'Gate G4',
        range: '$5,000+',
        status: 'locked',
        requirements: 'Reach $5,000 capital'
      }
    ]
  };
}