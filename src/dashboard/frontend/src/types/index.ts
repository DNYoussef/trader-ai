// Real-time risk dashboard types for Gary×Taleb trading system

export interface RiskMetrics {
  timestamp: number;
  portfolio_value: number;
  p_ruin: number;
  var_95: number;
  var_99: number;
  expected_shortfall: number;
  max_drawdown: number;
  sharpe_ratio: number;
  volatility: number;
  beta: number;
  positions_count: number;
  cash_available: number;
  margin_used: number;
  unrealized_pnl: number;
  daily_pnl: number;
}

export interface PositionUpdate {
  timestamp: number;
  symbol: string;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  entry_price: number;
  current_price: number;
  weight: number;
  last_update: number;
}

export interface AlertEvent {
  timestamp: number;
  alert_id: string;
  alert_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  metric_name: string;
  current_value: number;
  threshold_value: number;
  acknowledged: boolean;
}

export interface WebSocketMessage {
  type: 'risk_metrics' | 'position_update' | 'alert' | 'heartbeat' | 'pong' | 'alert_acknowledged';
  data: any;
  timestamp?: number;
}

export interface ConnectionStatus {
  connected: boolean;
  last_heartbeat: number;
  connection_time: number;
  reconnect_attempts: number;
}

export interface DashboardState {
  risk_metrics: RiskMetrics | null;
  positions: { [symbol: string]: PositionUpdate };
  alerts: AlertEvent[];
  connection: ConnectionStatus;
  settings: DashboardSettings;
  historical_data: HistoricalData;
}

export interface DashboardSettings {
  auto_acknowledge_alerts: boolean;
  alert_sound_enabled: boolean;
  refresh_rate: number; // milliseconds
  chart_timeframe: '1h' | '4h' | '1d' | '1w';
  risk_thresholds: RiskThresholds;
  display_preferences: DisplayPreferences;
}

export interface RiskThresholds {
  p_ruin: { high: number; critical: number };
  var_95: { high: number; critical: number };
  max_drawdown: { high: number; critical: number };
  margin_used: { high: number; critical: number };
}

export interface DisplayPreferences {
  dark_mode: boolean;
  compact_view: boolean;
  show_advanced_metrics: boolean;
  currency_format: 'USD' | 'percentage';
  decimal_places: number;
}

export interface HistoricalData {
  timestamps: number[];
  portfolio_values: number[];
  p_ruin_history: number[];
  var_history: number[];
  sharpe_history: number[];
  drawdown_history: number[];
}

export interface ChartData {
  timestamp: number;
  value: number;
  label?: string;
}

export interface MetricCard {
  title: string;
  value: number;
  format: 'currency' | 'percentage' | 'ratio' | 'number';
  trend?: 'up' | 'down' | 'neutral';
  alert_level?: 'normal' | 'warning' | 'danger';
  description?: string;
}

// WebSocket hook types
export interface UseWebSocketOptions {
  auto_connect: boolean;
  reconnect_attempts: number;
  heartbeat_interval: number;
}

export interface UseWebSocketReturn {
  connection_status: ConnectionStatus;
  send_message: (message: any) => void;
  subscribe: (subscription: string) => void;
  unsubscribe: (subscription: string) => void;
}

// API response types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  timestamp: number;
}

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: number;
  connections: number;
  uptime?: number;
}

// Component prop types
export interface MetricCardProps {
  metric: MetricCard;
  className?: string;
  onClick?: () => void;
}

export interface AlertListProps {
  alerts: AlertEvent[];
  onAcknowledge: (alert_id: string) => void;
  maxAlerts?: number;
}

export interface PositionTableProps {
  positions: PositionUpdate[];
  onRowClick?: (position: PositionUpdate) => void;
  sortBy?: keyof PositionUpdate;
  sortDirection?: 'asc' | 'desc';
}

export interface RiskChartProps {
  data: ChartData[];
  title: string;
  color?: string;
  height?: number;
  showGrid?: boolean;
}

// Chart configuration types
export interface ChartConfig {
  type: 'line' | 'area' | 'bar';
  colors: {
    primary: string;
    secondary?: string;
    danger: string;
    warning: string;
    success: string;
  };
  animations: {
    enabled: boolean;
    duration: number;
  };
  grid: {
    show: boolean;
    color: string;
  };
  tooltip: {
    enabled: boolean;
    format: (value: number) => string;
  };
}

// Mobile responsiveness
export interface ViewportSize {
  width: number;
  height: number;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
}

// Error handling types
export interface ErrorState {
  has_error: boolean;
  error_message: string;
  error_code?: string;
  timestamp: number;
}

export interface RetryConfig {
  max_attempts: number;
  delay: number;
  backoff_factor: number;
}

// Performance monitoring
export interface PerformanceMetrics {
  render_time: number;
  update_frequency: number;
  memory_usage: number;
  network_latency: number;
}