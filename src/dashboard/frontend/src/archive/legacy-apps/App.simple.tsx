import React, { useEffect, useState } from 'react';
import './index.css';

interface RiskMetrics {
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

interface Position {
  symbol: string;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  entry_price: number;
  current_price: number;
  weight: number;
  last_updated: number;
}

function App() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [connected, setConnected] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Fetch initial data from API
    fetch('http://localhost:8000/api/metrics/current')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error('Failed to fetch metrics:', err));

    fetch('http://localhost:8000/api/positions')
      .then(res => res.json())
      .then(data => setPositions(data))
      .catch(err => console.error('Failed to fetch positions:', err));

    // Setup WebSocket connection
    const websocket = new WebSocket('ws://localhost:8000/ws/client_1');

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message:', data);

      if (data.type === 'risk_metrics') {
        setMetrics(data.data);
      } else if (data.type === 'position_update') {
        setPositions(data.data);
      } else if (data.type === 'initial_data') {
        setMetrics(data.data.metrics);
        setPositions(data.data.positions);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    setWs(websocket);

    // Cleanup
    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">
              Gary×Taleb Trading Dashboard
            </h1>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-gray-600">
                  {connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {/* Risk Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Portfolio Value */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Portfolio Value</h3>
            <p className="text-2xl font-bold text-gray-900">
              {metrics ? formatCurrency(metrics.portfolio_value) : '--'}
            </p>
            {metrics && (
              <p className={`text-sm mt-2 ${metrics.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                Daily P&L: {formatCurrency(metrics.daily_pnl)}
              </p>
            )}
          </div>

          {/* P(ruin) */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">P(ruin)</h3>
            <p className={`text-2xl font-bold ${metrics && metrics.p_ruin > 0.1 ? 'text-red-600' : 'text-gray-900'}`}>
              {metrics ? formatPercent(metrics.p_ruin) : '--'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Threshold: 10%
            </p>
          </div>

          {/* VaR 95% */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">VaR 95%</h3>
            <p className="text-2xl font-bold text-gray-900">
              {metrics ? formatCurrency(metrics.var_95) : '--'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              99%: {metrics ? formatCurrency(metrics.var_99) : '--'}
            </p>
          </div>

          {/* Sharpe Ratio */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Sharpe Ratio</h3>
            <p className="text-2xl font-bold text-gray-900">
              {metrics ? metrics.sharpe_ratio.toFixed(2) : '--'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Volatility: {metrics ? formatPercent(metrics.volatility) : '--'}
            </p>
          </div>
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Max Drawdown</h3>
            <p className="text-xl font-bold text-gray-900">
              {metrics ? formatPercent(metrics.max_drawdown) : '--'}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Cash Available</h3>
            <p className="text-xl font-bold text-gray-900">
              {metrics ? formatCurrency(metrics.cash_available) : '--'}
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Margin Used</h3>
            <p className="text-xl font-bold text-gray-900">
              {metrics ? formatCurrency(metrics.margin_used) : '--'}
            </p>
          </div>
        </div>

        {/* Positions Table */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              Positions ({positions.length})
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Entry Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Market Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Unrealized P&L
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Weight
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {positions.map((position) => (
                  <tr key={position.symbol}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {position.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {position.quantity}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(position.entry_price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(position.current_price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(position.market_value)}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                      position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatCurrency(position.unrealized_pnl)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatPercent(position.weight)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {positions.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No positions to display
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;