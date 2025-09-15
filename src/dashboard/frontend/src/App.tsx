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

// Inline styles for better control
const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '1.5rem 2rem',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  headerContent: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    maxWidth: '1400px',
    margin: '0 auto',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  logoIcon: {
    width: '48px',
    height: '48px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontWeight: 'bold',
    fontSize: '20px',
  },
  title: {
    fontSize: '28px',
    fontWeight: 'bold',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    margin: 0,
  },
  subtitle: {
    fontSize: '12px',
    color: '#6b7280',
    margin: 0,
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem 1rem',
    background: 'rgba(255, 255, 255, 0.8)',
    borderRadius: '20px',
  },
  connectionDot: (connected: boolean) => ({
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: connected ? '#10b981' : '#ef4444',
    animation: connected ? 'pulse 2s infinite' : 'none',
  }),
  main: {
    padding: '2rem',
    maxWidth: '1400px',
    margin: '0 auto',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.5rem',
    marginBottom: '2rem',
  },
  card: {
    background: 'white',
    borderRadius: '16px',
    padding: '1.5rem',
    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    cursor: 'pointer',
  },
  cardHover: {
    transform: 'translateY(-5px)',
    boxShadow: '0 15px 35px rgba(0, 0, 0, 0.15)',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '1rem',
  },
  cardTitle: {
    fontSize: '14px',
    color: '#6b7280',
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    margin: 0,
  },
  cardValue: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#111827',
    margin: '0.5rem 0',
  },
  cardSubtext: {
    fontSize: '13px',
    color: '#9ca3af',
  },
  portfolioCard: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    gridColumn: 'span 2',
  },
  riskCard: (risk: number) => ({
    borderLeft: `4px solid ${risk > 0.1 ? '#ef4444' : risk > 0.05 ? '#f59e0b' : '#10b981'}`,
  }),
  table: {
    width: '100%',
    background: 'white',
    borderRadius: '16px',
    overflow: 'hidden',
    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
  },
  tableHeader: {
    background: 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)',
    padding: '1.5rem',
    borderBottom: '1px solid #e5e7eb',
  },
  tableTitle: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#111827',
    margin: 0,
  },
  tableBody: {
    padding: '1rem',
  },
  tableRow: {
    display: 'grid',
    gridTemplateColumns: '150px 100px 120px 120px 120px 120px 100px',
    padding: '1rem',
    borderBottom: '1px solid #f3f4f6',
    transition: 'background 0.2s ease',
  },
  tableRowHover: {
    background: '#f9fafb',
  },
  symbolBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  symbolIcon: {
    width: '32px',
    height: '32px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '12px',
    fontWeight: 'bold',
  },
  pnlBadge: (positive: boolean) => ({
    display: 'inline-block',
    padding: '0.25rem 0.75rem',
    borderRadius: '12px',
    background: positive ? '#d1fae5' : '#fee2e2',
    color: positive ? '#065f46' : '#991b1b',
    fontWeight: 'bold',
    fontSize: '13px',
  }),
  progressBar: {
    width: '100%',
    height: '6px',
    background: '#e5e7eb',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  progressFill: (percent: number) => ({
    width: `${percent}%`,
    height: '100%',
    background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
    transition: 'width 0.5s ease',
  }),
};

function App() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [connected, setConnected] = useState(false);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

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
      if (data.type === 'risk_metrics') {
        setMetrics(data.data);
      } else if (data.type === 'position_update') {
        setPositions(data.data);
      } else if (data.type === 'initial_data') {
        setMetrics(data.data.metrics);
        setPositions(data.data.positions);
      }
    };

    websocket.onerror = () => setConnected(false);
    websocket.onclose = () => setConnected(false);

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
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>GT</div>
            <div>
              <h1 style={styles.title}>Gary×Taleb Trading</h1>
              <p style={styles.subtitle}>Autonomous Trading System</p>
            </div>
          </div>
          <div style={styles.connectionStatus}>
            <div style={styles.connectionDot(connected)} />
            <span>{connected ? 'Live' : 'Offline'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={styles.main}>
        {/* Metrics Grid */}
        <div style={styles.metricsGrid}>
          {/* Portfolio Value */}
          <div
            style={{
              ...styles.card,
              ...styles.portfolioCard,
              ...(hoveredCard === 'portfolio' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('portfolio')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <div style={styles.cardHeader}>
              <div>
                <h3 style={{ ...styles.cardTitle, color: 'rgba(255,255,255,0.9)' }}>Portfolio Value</h3>
                <p style={{ ...styles.cardValue, color: 'white' }}>
                  {metrics ? formatCurrency(metrics.portfolio_value) : '--'}
                </p>
                {metrics && (
                  <p style={{ ...styles.cardSubtext, color: 'rgba(255,255,255,0.8)' }}>
                    Daily P&L: {formatCurrency(metrics.daily_pnl)}
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* P(ruin) */}
          <div
            style={{
              ...styles.card,
              ...styles.riskCard(metrics?.p_ruin || 0),
              ...(hoveredCard === 'pruin' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('pruin')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <h3 style={styles.cardTitle}>Probability of Ruin</h3>
            <p style={{
              ...styles.cardValue,
              color: metrics && metrics.p_ruin > 0.1 ? '#ef4444' : '#111827',
            }}>
              {metrics ? formatPercent(metrics.p_ruin) : '--'}
            </p>
            <p style={styles.cardSubtext}>Threshold: 10%</p>
          </div>

          {/* VaR 95% */}
          <div
            style={{
              ...styles.card,
              ...(hoveredCard === 'var' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('var')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <h3 style={styles.cardTitle}>Value at Risk (95%)</h3>
            <p style={styles.cardValue}>
              {metrics ? formatCurrency(metrics.var_95) : '--'}
            </p>
            <p style={styles.cardSubtext}>
              99%: {metrics ? formatCurrency(metrics.var_99) : '--'}
            </p>
          </div>

          {/* Sharpe Ratio */}
          <div
            style={{
              ...styles.card,
              ...(hoveredCard === 'sharpe' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('sharpe')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <h3 style={styles.cardTitle}>Sharpe Ratio</h3>
            <p style={styles.cardValue}>
              {metrics ? metrics.sharpe_ratio.toFixed(2) : '--'}
            </p>
            <p style={styles.cardSubtext}>
              Volatility: {metrics ? formatPercent(metrics.volatility) : '--'}
            </p>
          </div>

          {/* Max Drawdown */}
          <div
            style={{
              ...styles.card,
              ...(hoveredCard === 'drawdown' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('drawdown')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <h3 style={styles.cardTitle}>Max Drawdown</h3>
            <p style={styles.cardValue}>
              {metrics ? formatPercent(metrics.max_drawdown) : '--'}
            </p>
          </div>

          {/* Cash Available */}
          <div
            style={{
              ...styles.card,
              ...(hoveredCard === 'cash' ? styles.cardHover : {}),
            }}
            onMouseEnter={() => setHoveredCard('cash')}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <h3 style={styles.cardTitle}>Cash Available</h3>
            <p style={styles.cardValue}>
              {metrics ? formatCurrency(metrics.cash_available) : '--'}
            </p>
          </div>
        </div>

        {/* Positions Table */}
        <div style={styles.table}>
          <div style={styles.tableHeader}>
            <h2 style={styles.tableTitle}>
              Active Positions ({positions.length})
            </h2>
          </div>
          <div style={styles.tableBody}>
            {/* Table Header */}
            <div style={{ ...styles.tableRow, fontWeight: 'bold', fontSize: '12px', color: '#6b7280' }}>
              <div>SYMBOL</div>
              <div>QTY</div>
              <div>ENTRY</div>
              <div>CURRENT</div>
              <div>VALUE</div>
              <div>P&L</div>
              <div>WEIGHT</div>
            </div>

            {/* Table Rows */}
            {positions.map((position, index) => (
              <div
                key={position.symbol}
                style={{
                  ...styles.tableRow,
                  ...(hoveredRow === index ? styles.tableRowHover : {}),
                }}
                onMouseEnter={() => setHoveredRow(index)}
                onMouseLeave={() => setHoveredRow(null)}
              >
                <div style={styles.symbolBadge}>
                  <div style={styles.symbolIcon}>
                    {position.symbol.substring(0, 2)}
                  </div>
                  <span style={{ fontWeight: 'bold' }}>{position.symbol}</span>
                </div>
                <div>{position.quantity}</div>
                <div>{formatCurrency(position.entry_price)}</div>
                <div>{formatCurrency(position.current_price)}</div>
                <div style={{ fontWeight: 'bold' }}>{formatCurrency(position.market_value)}</div>
                <div>
                  <span style={styles.pnlBadge(position.unrealized_pnl >= 0)}>
                    {position.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
                  </span>
                </div>
                <div>
                  <div style={styles.progressBar}>
                    <div style={styles.progressFill(position.weight * 100)} />
                  </div>
                  <span style={{ fontSize: '11px', color: '#6b7280' }}>
                    {formatPercent(position.weight)}
                  </span>
                </div>
              </div>
            ))}
            {positions.length === 0 && (
              <div style={{ textAlign: 'center', padding: '3rem', color: '#9ca3af' }}>
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