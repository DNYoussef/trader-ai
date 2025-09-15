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

// Mock data for demonstration
const mockMetrics = {
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

/**
 * Fully Integrated Enhanced App with all components
 */
const AppEnhancedFull: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState(mockMetrics);
  const [positions, setPositions] = useState(mockPositions);
  const [alerts, setAlerts] = useState(mockAlerts);
  const [chartData, setChartData] = useState(mockChartData);
  const [wsConnected, setWsConnected] = useState(false);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/enhanced_dashboard');

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

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update metrics with small random changes
      setMetrics(prev => ({
        ...prev,
        portfolio_value: prev.portfolio_value + (Math.random() - 0.5) * 100,
        daily_pnl: prev.daily_pnl + (Math.random() - 0.5) * 50,
        p_ruin: Math.max(0, Math.min(1, prev.p_ruin + (Math.random() - 0.5) * 0.01))
      }));

      // Update chart data
      setChartData(prev => [
        ...prev.slice(1),
        {
          timestamp: Date.now(),
          value: metrics.portfolio_value + (Math.random() - 0.5) * 100,
          p_ruin: metrics.p_ruin
        }
      ]);
    }, 5000);

    return () => clearInterval(interval);
  }, [metrics]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '[CHART]' },
    { id: 'terminal', label: 'Trading Terminal', icon: '[TREND]' },
    { id: 'learn', label: 'Learn', icon: '[BRAIN]' },
    { id: 'progress', label: 'Progress', icon: '[TARGET]' }
  ];

  return (
    <div style={{ backgroundColor: '#f3f4f6', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{
        backgroundColor: 'white',
        padding: '20px',
        borderBottom: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: 0, color: '#111827', fontSize: '24px' }}>
              Gary×Taleb Trading System
            </h1>
            <p style={{ margin: '5px 0 0 0', color: '#6b7280' }}>
              Enhanced Trading Dashboard with AI Intelligence
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: wsConnected ? '#10b981' : '#ef4444'
            }} />
            <span style={{ color: '#6b7280', fontSize: '14px' }}>
              {wsConnected ? 'Connected' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e5e7eb',
        display: 'flex',
        gap: '0'
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '15px 25px',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #3b82f6' : '2px solid transparent',
              backgroundColor: activeTab === tab.id ? '#eff6ff' : 'white',
              color: activeTab === tab.id ? '#3b82f6' : '#6b7280',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s'
            }}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '30px' }}>
        {activeTab === 'overview' && (
          <div>
            {/* Metric Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '30px'
            }}>
              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#6b7280', fontSize: '14px' }}>Portfolio Value</h3>
                <p style={{ margin: 0, color: '#10b981', fontSize: '32px', fontWeight: 'bold' }}>
                  ${metrics.portfolio_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
                <p style={{ margin: '10px 0 0 0', color: metrics.daily_pnl >= 0 ? '#10b981' : '#ef4444', fontSize: '14px' }}>
                  {metrics.daily_pnl >= 0 ? '+' : ''}{metrics.daily_pnl.toFixed(2)} today
                </p>
              </div>

              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#6b7280', fontSize: '14px' }}>P(ruin)</h3>
                <p style={{
                  margin: 0,
                  color: metrics.p_ruin > 0.15 ? '#ef4444' : metrics.p_ruin > 0.10 ? '#f59e0b' : '#10b981',
                  fontSize: '32px',
                  fontWeight: 'bold'
                }}>
                  {(metrics.p_ruin * 100).toFixed(1)}%
                </p>
                <p style={{ margin: '10px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                  Risk of ruin probability
                </p>
              </div>

              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#6b7280', fontSize: '14px' }}>Sharpe Ratio</h3>
                <p style={{ margin: 0, color: '#8b5cf6', fontSize: '32px', fontWeight: 'bold' }}>
                  {metrics.sharpe_ratio.toFixed(2)}
                </p>
                <p style={{ margin: '10px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                  Risk-adjusted return
                </p>
              </div>

              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#6b7280', fontSize: '14px' }}>Max Drawdown</h3>
                <p style={{ margin: 0, color: '#f59e0b', fontSize: '32px', fontWeight: 'bold' }}>
                  {(metrics.max_drawdown * 100).toFixed(1)}%
                </p>
                <p style={{ margin: '10px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                  Peak to trough decline
                </p>
              </div>
            </div>

            {/* Charts and Tables */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              {/* Portfolio Chart */}
              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 20px 0', color: '#111827' }}>Portfolio Value</h3>
                <ResponsiveContainer width="100%" height={250}>
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
                      dataKey="value"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#colorValue)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Positions Table */}
              <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h3 style={{ margin: '0 0 20px 0', color: '#111827' }}>Open Positions</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', fontSize: '14px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                        <th style={{ padding: '10px', textAlign: 'left', color: '#6b7280' }}>Symbol</th>
                        <th style={{ padding: '10px', textAlign: 'right', color: '#6b7280' }}>Qty</th>
                        <th style={{ padding: '10px', textAlign: 'right', color: '#6b7280' }}>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.slice(0, 5).map(position => (
                        <tr key={position.symbol} style={{ borderBottom: '1px solid #f3f4f6' }}>
                          <td style={{ padding: '10px', fontWeight: '500' }}>{position.symbol}</td>
                          <td style={{ padding: '10px', textAlign: 'right' }}>{position.quantity}</td>
                          <td style={{
                            padding: '10px',
                            textAlign: 'right',
                            color: position.pnl >= 0 ? '#10b981' : '#ef4444'
                          }}>
                            ${position.pnl.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Alerts */}
            <div style={{ marginTop: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <h3 style={{ margin: '0 0 20px 0', color: '#111827' }}>Recent Alerts</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {alerts.map(alert => (
                  <div key={alert.id} style={{
                    padding: '12px',
                    borderRadius: '6px',
                    backgroundColor:
                      alert.severity === 'warning' ? '#fef3c7' :
                      alert.severity === 'success' ? '#d1fae5' :
                      '#dbeafe',
                    borderLeft: `4px solid ${
                      alert.severity === 'warning' ? '#f59e0b' :
                      alert.severity === 'success' ? '#10b981' :
                      '#3b82f6'
                    }`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                      <div>
                        <p style={{ margin: 0, fontWeight: '600', color: '#111827' }}>{alert.title}</p>
                        <p style={{ margin: '5px 0 0 0', color: '#6b7280', fontSize: '14px' }}>{alert.message}</p>
                      </div>
                      <span style={{ color: '#9ca3af', fontSize: '12px' }}>
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'terminal' && (
          <div>
            <h2 style={{ color: '#111827', marginBottom: '20px' }}>Professional Trading Terminal</h2>

            {/* Market Overview */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '15px',
              marginBottom: '30px'
            }}>
              {['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU'].map(symbol => (
                <div key={symbol} style={{
                  backgroundColor: 'white',
                  padding: '15px',
                  borderRadius: '8px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <div>
                      <p style={{ margin: 0, fontWeight: '600', color: '#111827' }}>{symbol}</p>
                      <p style={{ margin: '5px 0 0 0', fontSize: '20px', fontWeight: 'bold' }}>
                        ${(400 + Math.random() * 50).toFixed(2)}
                      </p>
                    </div>
                    <div style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: Math.random() > 0.5 ? '#d1fae5' : '#fee2e2',
                      color: Math.random() > 0.5 ? '#065f46' : '#991b1b',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {Math.random() > 0.5 ? '+' : '-'}{(Math.random() * 3).toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Main Chart Area */}
            <div style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              minHeight: '500px'
            }}>
              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ margin: 0, color: '#111827' }}>SPY - SPDR S&P 500 ETF</h3>
                <p style={{ margin: '5px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                  Real-time price action with AI signals
                </p>
              </div>

              {/* Chart Controls */}
              <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                {['1m', '5m', '15m', '1h', '4h', '1d'].map(tf => (
                  <button key={tf} style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: '1px solid #e5e7eb',
                    backgroundColor: tf === '15m' ? '#3b82f6' : 'white',
                    color: tf === '15m' ? 'white' : '#6b7280',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}>
                    {tf}
                  </button>
                ))}
              </div>

              {/* Chart Area */}
              <ResponsiveContainer width="100%" height={400}>
                <LineChart
                  data={chartData.map((d, i) => ({
                    ...d,
                    high: d.value + Math.random() * 50,
                    low: d.value - Math.random() * 50,
                    signal: i % 5 === 0 ? d.value + 100 : null
                  }))}
                >
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
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
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '6px' }}
                    labelStyle={{ color: '#9ca3af' }}
                    itemStyle={{ color: '#fff' }}
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
                    dataKey="value"
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
                  <Line
                    type="monotone"
                    dataKey="signal"
                    stroke="#f59e0b"
                    strokeWidth={0}
                    dot={{ r: 6, fill: '#f59e0b' }}
                    name="AI Signal"
                  />
                </LineChart>
              </ResponsiveContainer>

              {/* Trading Panel */}
              <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f9fafb', borderRadius: '8px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  <div>
                    <h4 style={{ margin: '0 0 15px 0', color: '#111827' }}>Quick Trade</h4>
                    <div style={{ display: 'flex', gap: '10px' }}>
                      <button style={{
                        flex: 1,
                        padding: '12px',
                        backgroundColor: '#10b981',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}>
                        BUY
                      </button>
                      <button style={{
                        flex: 1,
                        padding: '12px',
                        backgroundColor: '#ef4444',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontWeight: '600',
                        cursor: 'pointer'
                      }}>
                        SELL
                      </button>
                    </div>
                  </div>
                  <div>
                    <h4 style={{ margin: '0 0 15px 0', color: '#111827' }}>AI Signals</h4>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                      <div style={{ flex: 1 }}>
                        <p style={{ margin: 0, color: '#6b7280', fontSize: '12px' }}>DPI Score</p>
                        <p style={{ margin: '5px 0 0 0', color: '#10b981', fontSize: '20px', fontWeight: 'bold' }}>
                          85%
                        </p>
                      </div>
                      <div style={{ flex: 1 }}>
                        <p style={{ margin: 0, color: '#6b7280', fontSize: '12px' }}>Signal</p>
                        <p style={{ margin: '5px 0 0 0', color: '#10b981', fontSize: '20px', fontWeight: 'bold' }}>
                          BUY
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'learn' && (
          <EducationHub />
        )}

        {activeTab === 'progress' && (
          <div>
            <h2 style={{ color: '#111827', marginBottom: '20px' }}>Your Trading Journey</h2>

            {/* Stats Overview */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '20px',
              marginBottom: '30px'
            }}>
              {[
                { label: 'Total Trades', value: '147', icon: '[CHART]' },
                { label: 'Win Rate', value: '67%', icon: '[TARGET]' },
                { label: 'Avg Profit', value: '$342', icon: '[MONEY]' },
                { label: 'Best Month', value: '+12.3%', icon: '[TROPHY]' },
                { label: 'Risk Score', value: 'A+', icon: '[SHIELD]' },
                { label: 'Learning Hours', value: '24', icon: '📚' }
              ].map(stat => (
                <div key={stat.label} style={{
                  backgroundColor: 'white',
                  padding: '20px',
                  borderRadius: '8px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '32px', marginBottom: '10px' }}>{stat.icon}</div>
                  <p style={{ margin: '0 0 5px 0', color: '#6b7280', fontSize: '14px' }}>{stat.label}</p>
                  <p style={{ margin: 0, color: '#111827', fontSize: '24px', fontWeight: 'bold' }}>{stat.value}</p>
                </div>
              ))}
            </div>

            {/* Achievements Section */}
            <div style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              marginBottom: '30px'
            }}>
              <h3 style={{ margin: '0 0 20px 0', color: '#111827' }}>Recent Achievements</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px' }}>
                {[
                  { title: 'First Trade', desc: 'Executed your first trade', icon: '[PARTY]', date: '2 days ago' },
                  { title: 'Risk Master', desc: 'Maintained P(ruin) < 10% for 30 days', icon: '[SHIELD]', date: '1 week ago' },
                  { title: 'Profit Maker', desc: 'Achieved 5 consecutive profitable trades', icon: '[MONEY]', date: '2 weeks ago' },
                  { title: 'Knowledge Seeker', desc: 'Completed 3 educational courses', icon: '📚', date: '1 month ago' }
                ].map(achievement => (
                  <div key={achievement.title} style={{
                    padding: '15px',
                    borderRadius: '8px',
                    backgroundColor: '#fef3c7',
                    border: '1px solid #fcd34d'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'start', gap: '15px' }}>
                      <div style={{ fontSize: '32px' }}>{achievement.icon}</div>
                      <div style={{ flex: 1 }}>
                        <p style={{ margin: 0, fontWeight: '600', color: '#111827' }}>{achievement.title}</p>
                        <p style={{ margin: '5px 0', color: '#6b7280', fontSize: '14px' }}>{achievement.desc}</p>
                        <p style={{ margin: 0, color: '#9ca3af', fontSize: '12px' }}>{achievement.date}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Progress Timeline */}
            <div style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <h3 style={{ margin: '0 0 20px 0', color: '#111827' }}>Gate Progression</h3>
              <div style={{ position: 'relative' }}>
                {['G0', 'G1', 'G2', 'G3', 'G4'].map((gate, index) => (
                  <div key={gate} style={{
                    display: 'flex',
                    alignItems: 'center',
                    marginBottom: '20px'
                  }}>
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      backgroundColor: index === 0 ? '#10b981' : '#e5e7eb',
                      color: index === 0 ? 'white' : '#9ca3af',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: '600'
                    }}>
                      {index === 0 ? '✓' : gate}
                    </div>
                    <div style={{ marginLeft: '20px', flex: 1 }}>
                      <p style={{ margin: 0, fontWeight: '600', color: '#111827' }}>
                        Gate {gate} - ${index === 0 ? '200-499' : index === 1 ? '500-999' : index === 2 ? '1,000-2,499' : index === 3 ? '2,500-4,999' : '5,000+'}
                      </p>
                      <p style={{ margin: '5px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                        {index === 0 ? 'Completed - Great job!' : `Requirements: ${5 - index} more profitable trades`}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AppEnhancedFull;