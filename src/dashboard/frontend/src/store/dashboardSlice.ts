import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RiskMetrics, PositionUpdate, ConnectionStatus, HistoricalData } from '@/types';

interface DashboardState {
  risk_metrics: RiskMetrics | null;
  positions: { [symbol: string]: PositionUpdate };
  connection: ConnectionStatus;
  historical_data: HistoricalData;
  loading: boolean;
  error: string | null;
  last_update: number;
}

const initialState: DashboardState = {
  risk_metrics: null,
  positions: {},
  connection: {
    connected: false,
    last_heartbeat: 0,
    connection_time: 0,
    reconnect_attempts: 0,
  },
  historical_data: {
    timestamps: [],
    portfolio_values: [],
    p_ruin_history: [],
    var_history: [],
    sharpe_history: [],
    drawdown_history: [],
  },
  loading: false,
  error: null,
  last_update: 0,
};

export const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    updateConnection: (state, action: PayloadAction<Partial<ConnectionStatus>>) => {
      state.connection = { ...state.connection, ...action.payload };
    },

    setConnected: (state, action: PayloadAction<boolean>) => {
      state.connection.connected = action.payload;
      if (action.payload) {
        state.connection.connection_time = Date.now();
        state.connection.reconnect_attempts = 0;
        state.error = null;
      }
    },

    incrementReconnectAttempts: (state) => {
      state.connection.reconnect_attempts += 1;
    },

    updateHeartbeat: (state) => {
      state.connection.last_heartbeat = Date.now();
    },

    updateRiskMetrics: (state, action: PayloadAction<RiskMetrics>) => {
      const metrics = action.payload;
      state.risk_metrics = metrics;
      state.last_update = Date.now();

      // Update historical data
      const historical = state.historical_data;

      // Add new data point
      historical.timestamps.push(metrics.timestamp);
      historical.portfolio_values.push(metrics.portfolio_value);
      historical.p_ruin_history.push(metrics.p_ruin);
      historical.var_history.push(metrics.var_95);
      historical.sharpe_history.push(metrics.sharpe_ratio);
      historical.drawdown_history.push(metrics.max_drawdown);

      // Keep only last 500 data points for performance
      const maxPoints = 500;
      if (historical.timestamps.length > maxPoints) {
        historical.timestamps = historical.timestamps.slice(-maxPoints);
        historical.portfolio_values = historical.portfolio_values.slice(-maxPoints);
        historical.p_ruin_history = historical.p_ruin_history.slice(-maxPoints);
        historical.var_history = historical.var_history.slice(-maxPoints);
        historical.sharpe_history = historical.sharpe_history.slice(-maxPoints);
        historical.drawdown_history = historical.drawdown_history.slice(-maxPoints);
      }
    },

    updatePosition: (state, action: PayloadAction<PositionUpdate>) => {
      const position = action.payload;
      state.positions[position.symbol] = position;
      state.last_update = Date.now();
    },

    removePosition: (state, action: PayloadAction<string>) => {
      const symbol = action.payload;
      delete state.positions[symbol];
      state.last_update = Date.now();
    },

    clearPositions: (state) => {
      state.positions = {};
      state.last_update = Date.now();
    },

    resetHistoricalData: (state) => {
      state.historical_data = {
        timestamps: [],
        portfolio_values: [],
        p_ruin_history: [],
        var_history: [],
        sharpe_history: [],
        drawdown_history: [],
      };
    },

    // Bulk update for initial data load
    setInitialData: (state, action: PayloadAction<{
      risk_metrics?: RiskMetrics;
      positions?: PositionUpdate[];
    }>) => {
      const { risk_metrics, positions } = action.payload;

      if (risk_metrics) {
        state.risk_metrics = risk_metrics;
      }

      if (positions) {
        state.positions = {};
        positions.forEach(position => {
          state.positions[position.symbol] = position;
        });
      }

      state.last_update = Date.now();
    },

    // Performance optimization: batch updates
    batchUpdate: (state, action: PayloadAction<{
      risk_metrics?: RiskMetrics;
      positions?: PositionUpdate[];
    }>) => {
      const { risk_metrics, positions } = action.payload;

      if (risk_metrics) {
        dashboardSlice.caseReducers.updateRiskMetrics(state, {
          type: 'dashboard/updateRiskMetrics',
          payload: risk_metrics,
        });
      }

      if (positions) {
        positions.forEach(position => {
          state.positions[position.symbol] = position;
        });
      }

      state.last_update = Date.now();
    },
  },
});

export const {
  setLoading,
  setError,
  updateConnection,
  setConnected,
  incrementReconnectAttempts,
  updateHeartbeat,
  updateRiskMetrics,
  updatePosition,
  removePosition,
  clearPositions,
  resetHistoricalData,
  setInitialData,
  batchUpdate,
} = dashboardSlice.actions;

// Selectors
export const selectRiskMetrics = (state: { dashboard: DashboardState }) =>
  state.dashboard.risk_metrics;

export const selectPositions = (state: { dashboard: DashboardState }) =>
  Object.values(state.dashboard.positions);

export const selectPositionBySymbol = (symbol: string) =>
  (state: { dashboard: DashboardState }) =>
    state.dashboard.positions[symbol];

export const selectConnectionStatus = (state: { dashboard: DashboardState }) =>
  state.dashboard.connection;

export const selectHistoricalData = (state: { dashboard: DashboardState }) =>
  state.dashboard.historical_data;

export const selectIsConnected = (state: { dashboard: DashboardState }) =>
  state.dashboard.connection.connected;

export const selectPortfolioValue = (state: { dashboard: DashboardState }) =>
  state.dashboard.risk_metrics?.portfolio_value || 0;

export const selectTotalPositions = (state: { dashboard: DashboardState }) =>
  Object.keys(state.dashboard.positions).length;

export const selectTotalUnrealizedPnL = (state: { dashboard: DashboardState }) =>
  Object.values(state.dashboard.positions).reduce(
    (total, position) => total + position.unrealized_pnl,
    0
  );

export const selectTopPositions = (limit: number = 5) =>
  (state: { dashboard: DashboardState }) =>
    Object.values(state.dashboard.positions)
      .sort((a, b) => Math.abs(b.market_value) - Math.abs(a.market_value))
      .slice(0, limit);

export const selectPositionsByPerformance = (state: { dashboard: DashboardState }) => {
  const positions = Object.values(state.dashboard.positions);
  return {
    winners: positions.filter(p => p.unrealized_pnl > 0)
                     .sort((a, b) => b.unrealized_pnl - a.unrealized_pnl),
    losers: positions.filter(p => p.unrealized_pnl < 0)
                     .sort((a, b) => a.unrealized_pnl - b.unrealized_pnl),
  };
};

export const selectRiskLevel = (state: { dashboard: DashboardState }) => {
  const metrics = state.dashboard.risk_metrics;
  if (!metrics) return 'unknown';

  // Determine overall risk level based on multiple metrics
  if (metrics.p_ruin > 0.2 || metrics.max_drawdown > 0.2) return 'critical';
  if (metrics.p_ruin > 0.1 || metrics.max_drawdown > 0.1) return 'high';
  if (metrics.p_ruin > 0.05 || metrics.max_drawdown > 0.05) return 'medium';
  return 'low';
};

export const selectLatestChartData = (metric: keyof HistoricalData, points: number = 50) =>
  (state: { dashboard: DashboardState }) => {
    const historical = state.dashboard.historical_data;
    const data = historical[metric] as number[];
    const timestamps = historical.timestamps;

    if (!data || !timestamps || data.length === 0) return [];

    const startIndex = Math.max(0, data.length - points);
    return data.slice(startIndex).map((value, index) => ({
      timestamp: timestamps[startIndex + index],
      value,
    }));
  };

export default dashboardSlice.reducer;