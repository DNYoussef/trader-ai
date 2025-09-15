import React from 'react';
import './index.css';
import { MetricCard, PRuinCard, PortfolioValueCard, VarCard, SharpeRatioCard, DrawdownCard } from './components/MetricCard';
import { PositionTable } from './components/PositionTable';
import { AlertList } from './components/AlertList';
import { RiskChart } from './components/RiskChart';
import { useState, useEffect } from 'react';

// Original Risk Dashboard UI
function App() {
  const [metrics, setMetrics] = useState({
    portfolio_value: 0,
    p_ruin: 0,
    var_95: 0,
    var_99: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    positions_count: 0,
    unrealized_pnl: 0,
    daily_pnl: 0
  });
  const [positions, setPositions] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // Fetch data from API
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/metrics/current');
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          Original Risk Dashboard - Gary×Taleb Trading
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <PortfolioValueCard value={metrics.portfolio_value} />
          <PRuinCard value={metrics.p_ruin} />
          <VarCard value={metrics.var_95} />
          <SharpeRatioCard value={metrics.sharpe_ratio} />
          <DrawdownCard value={metrics.max_drawdown} />
          <MetricCard
            title="Daily P&L"
            value={`$${metrics.daily_pnl?.toFixed(2) || '0.00'}`}
            trend="up"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PositionTable positions={positions} />
          <AlertList alerts={alerts} />
        </div>

        <div className="mt-6">
          <RiskChart />
        </div>
      </div>
    </div>
  );
}

export default App;