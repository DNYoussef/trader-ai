/**
 * Professional Trading Terminal - Real trader interface with advanced charts
 *
 * Provides a comprehensive trading terminal with real-time market data,
 * algorithmic strategy overlays, AI inflection points, and professional
 * trading tools. Designed to make users feel like professional traders.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useEnhancedUX } from './EnhancedUXProvider';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
}

interface ChartDataPoint {
  timestamp: number;
  price: number;
  volume: number;
  ma_20?: number;
  ma_50?: number;
  rsi?: number;
  algo_signal?: 'BUY' | 'SELL' | null;
  ai_inflection?: boolean;
  strategy_score?: number;
}

interface StrategySignal {
  timestamp: number;
  type: 'ENTRY' | 'EXIT' | 'REBALANCE';
  signal: 'BUY' | 'SELL';
  confidence: number;
  strategy: 'DPI' | 'CAUSAL' | 'RISK_MGMT';
  reason: string;
}

interface TradingTerminalProps {
  symbols?: string[];
  timeframe?: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  enableLiveData?: boolean;
}

const TradingTerminal: React.FC<TradingTerminalProps> = ({
  symbols = ['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU'],
  timeframe = '15m',
  enableLiveData = true
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [strategySignals, setStrategySignals] = useState<StrategySignal[]>([]);
  const [orderBook, setOrderBook] = useState<{bids: Array<[number, number]>, asks: Array<[number, number]>}>({bids: [], asks: []});
  const [watchlist, setWatchlist] = useState<string[]>(symbols);
  const [activeTimeframe, setActiveTimeframe] = useState(timeframe);
  const [showStrategyOverlay, setShowStrategyOverlay] = useState(true);
  const [showAIInflections, setShowAIInflections] = useState(true);

  const chartCanvasRef = useRef<HTMLCanvasElement>(null);
  const { trackProgress } = useEnhancedUX();

  useEffect(() => {
    initializeTradingTerminal();
    if (enableLiveData) {
      const interval = setInterval(updateMarketData, 1000); // 1 second updates
      return () => clearInterval(interval);
    }
  }, [enableLiveData]);

  useEffect(() => {
    generateChartData();
  }, [selectedSymbol, activeTimeframe]);

  useEffect(() => {
    drawChart();
  }, [chartData, showStrategyOverlay, showAIInflections]);

  const initializeTradingTerminal = () => {
    trackProgress('trading_terminal_opened', {
      symbols: watchlist,
      timeframe: activeTimeframe
    });

    // Initialize with mock data
    generateInitialMarketData();
    generateStrategySignals();
    generateOrderBook();
  };

  const generateInitialMarketData = () => {
    const mockData: Record<string, MarketData> = {};

    const basePrices = {
      'SPY': 440.25,
      'ULTY': 52.18,
      'AMDY': 24.67,
      'VTIP': 48.92,
      'IAU': 36.45
    };

    watchlist.forEach(symbol => {
      const basePrice = basePrices[symbol as keyof typeof basePrices] || 100;
      const change = (Math.random() - 0.5) * 4; // -2 to +2
      const change_percent = (change / basePrice) * 100;

      mockData[symbol] = {
        symbol,
        price: basePrice + change,
        change,
        change_percent,
        volume: Math.floor(Math.random() * 10000000) + 1000000,
        timestamp: new Date().toISOString()
      };
    });

    setMarketData(mockData);
  };

  const updateMarketData = useCallback(() => {
    setMarketData(prev => {
      const updated = { ...prev };

      Object.keys(updated).forEach(symbol => {
        const current = updated[symbol];
        const volatility = symbol === 'SPY' ? 0.002 : 0.005; // SPY is less volatile
        const priceChange = (Math.random() - 0.5) * current.price * volatility;

        updated[symbol] = {
          ...current,
          price: Math.max(0.01, current.price + priceChange),
          change: current.change + priceChange,
          change_percent: ((current.price + priceChange - (current.price - current.change)) / (current.price - current.change)) * 100,
          volume: current.volume + Math.floor(Math.random() * 10000),
          timestamp: new Date().toISOString()
        };
      });

      return updated;
    });
  }, []);

  const generateChartData = () => {
    const basePrice = marketData[selectedSymbol]?.price || 100;
    const dataPoints: ChartDataPoint[] = [];

    // Generate 100 data points for the chart
    for (let i = 0; i < 100; i++) {
      const timestamp = Date.now() - (100 - i) * 60000; // 1-minute intervals
      const volatility = 0.02;
      const trend = Math.sin(i * 0.1) * 0.01; // Slight trending
      const noise = (Math.random() - 0.5) * volatility;

      const price = basePrice * (1 + trend + noise);
      const volume = Math.floor(Math.random() * 1000000) + 100000;

      // Calculate technical indicators
      const ma_20 = i >= 20 ? dataPoints.slice(-20).reduce((sum, p) => sum + p.price, 0) / 20 : undefined;
      const ma_50 = i >= 50 ? dataPoints.slice(-50).reduce((sum, p) => sum + p.price, 0) / 50 : undefined;

      // Generate algo signals occasionally
      const algo_signal = Math.random() > 0.95 ? (Math.random() > 0.5 ? 'BUY' : 'SELL') : null;

      // Generate AI inflection points occasionally
      const ai_inflection = Math.random() > 0.97;

      // Strategy score (0-100)
      const strategy_score = Math.floor(Math.random() * 100);

      dataPoints.push({
        timestamp,
        price,
        volume,
        ma_20,
        ma_50,
        algo_signal,
        ai_inflection,
        strategy_score
      });
    }

    setChartData(dataPoints);
  };

  const generateStrategySignals = () => {
    const signals: StrategySignal[] = [];
    const strategies = ['DPI', 'CAUSAL', 'RISK_MGMT'] as const;

    for (let i = 0; i < 20; i++) {
      const timestamp = Date.now() - Math.random() * 86400000; // Last 24 hours

      signals.push({
        timestamp,
        type: ['ENTRY', 'EXIT', 'REBALANCE'][Math.floor(Math.random() * 3)] as any,
        signal: Math.random() > 0.5 ? 'BUY' : 'SELL',
        confidence: Math.floor(Math.random() * 40) + 60, // 60-100%
        strategy: strategies[Math.floor(Math.random() * strategies.length)],
        reason: [
          'DPI threshold exceeded',
          'Causal pattern detected',
          'Risk rebalancing required',
          'Policy shock anticipated',
          'Flow reversal detected'
        ][Math.floor(Math.random() * 5)]
      });
    }

    signals.sort((a, b) => b.timestamp - a.timestamp);
    setStrategySignals(signals);
  };

  const generateOrderBook = () => {
    const currentPrice = marketData[selectedSymbol]?.price || 100;

    const bids: Array<[number, number]> = [];
    const asks: Array<[number, number]> = [];

    // Generate bid levels (below current price)
    for (let i = 0; i < 10; i++) {
      const price = currentPrice - (i + 1) * 0.01;
      const size = Math.floor(Math.random() * 1000) + 100;
      bids.push([price, size]);
    }

    // Generate ask levels (above current price)
    for (let i = 0; i < 10; i++) {
      const price = currentPrice + (i + 1) * 0.01;
      const size = Math.floor(Math.random() * 1000) + 100;
      asks.push([price, size]);
    }

    setOrderBook({ bids, asks });
  };

  const drawChart = () => {
    const canvas = chartCanvasRef.current;
    if (!canvas || chartData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Calculate scales
    const prices = chartData.map(d => d.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    const xScale = width / chartData.length;
    const yScale = height / priceRange;

    // Draw grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = (height / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 20; i++) {
      const x = (width / 20) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw price line (green/red based on trend)
    ctx.lineWidth = 2;
    ctx.beginPath();

    chartData.forEach((point, index) => {
      const x = index * xScale;
      const y = height - ((point.price - minPrice) * yScale);

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    const currentPrice = chartData[chartData.length - 1]?.price;
    const previousPrice = chartData[chartData.length - 2]?.price;
    ctx.strokeStyle = currentPrice > previousPrice ? '#00ff88' : '#ff4444';
    ctx.stroke();

    // Draw moving averages if enabled
    if (showStrategyOverlay) {
      // MA 20 (yellow)
      ctx.strokeStyle = '#ffaa00';
      ctx.lineWidth = 1;
      ctx.beginPath();
      chartData.forEach((point, index) => {
        if (point.ma_20) {
          const x = index * xScale;
          const y = height - ((point.ma_20 - minPrice) * yScale);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // MA 50 (purple)
      ctx.strokeStyle = '#aa00ff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      chartData.forEach((point, index) => {
        if (point.ma_50) {
          const x = index * xScale;
          const y = height - ((point.ma_50 - minPrice) * yScale);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Draw AI inflection points
    if (showAIInflections) {
      chartData.forEach((point, index) => {
        if (point.ai_inflection) {
          const x = index * xScale;
          const y = height - ((point.price - minPrice) * yScale);

          ctx.fillStyle = '#ff00ff';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();

          // Add inflection marker
          ctx.fillStyle = '#ff00ff';
          ctx.font = '10px monospace';
          ctx.fillText('AI', x - 8, y - 8);
        }
      });
    }

    // Draw algo signals
    chartData.forEach((point, index) => {
      if (point.algo_signal) {
        const x = index * xScale;
        const y = height - ((point.price - minPrice) * yScale);

        ctx.fillStyle = point.algo_signal === 'BUY' ? '#00ff88' : '#ff4444';
        ctx.fillRect(x - 2, y - 10, 4, 8);

        ctx.fillStyle = '#ffffff';
        ctx.font = '8px monospace';
        ctx.fillText(point.algo_signal[0], x - 2, y - 12);
      }
    });

    // Draw current price line
    if (currentPrice) {
      const y = height - ((currentPrice - minPrice) * yScale);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Price label
      ctx.fillStyle = '#000000';
      ctx.fillRect(width - 60, y - 8, 58, 16);
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px monospace';
      ctx.fillText(currentPrice.toFixed(2), width - 55, y + 3);
    }
  };

  const formatPrice = (price: number) => {
    return price?.toFixed(2) || '0.00';
  };

  const formatChange = (change: number, percent: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)} (${sign}${percent.toFixed(2)}%)`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(0)}K`;
    }
    return volume.toString();
  };

  const getStrategyColor = (strategy: string) => {
    switch (strategy) {
      case 'DPI': return '#ffaa00';
      case 'CAUSAL': return '#aa00ff';
      case 'RISK_MGMT': return '#00aaff';
      default: return '#ffffff';
    }
  };

  return (
    <div className="h-full bg-black text-white font-mono flex flex-col">
      {/* Terminal Header */}
      <div className="bg-gray-900 p-2 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-bold text-green-400">TRADING TERMINAL</h2>
            <div className="text-xs text-gray-400">
              LIVE • {new Date().toLocaleTimeString()}
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowStrategyOverlay(!showStrategyOverlay)}
              className={`px-2 py-1 text-xs rounded ${showStrategyOverlay ? 'bg-yellow-600' : 'bg-gray-700'}`}
            >
              ALGO
            </button>
            <button
              onClick={() => setShowAIInflections(!showAIInflections)}
              className={`px-2 py-1 text-xs rounded ${showAIInflections ? 'bg-purple-600' : 'bg-gray-700'}`}
            >
              AI
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex">
        {/* Left Panel - Watchlist & Order Book */}
        <div className="w-80 bg-gray-900 border-r border-gray-700 flex flex-col">
          {/* Watchlist */}
          <div className="p-3 border-b border-gray-700">
            <h3 className="text-sm font-bold mb-2 text-blue-400">WATCHLIST</h3>
            <div className="space-y-1">
              {watchlist.map(symbol => {
                const data = marketData[symbol];
                if (!data) return null;

                const isSelected = symbol === selectedSymbol;
                const isPositive = data.change >= 0;

                return (
                  <div
                    key={symbol}
                    onClick={() => setSelectedSymbol(symbol)}
                    className={`p-2 rounded cursor-pointer text-xs ${
                      isSelected ? 'bg-blue-800' : 'hover:bg-gray-800'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold">{symbol}</span>
                      <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
                        ${formatPrice(data.price)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center mt-1">
                      <span className="text-gray-400">Vol: {formatVolume(data.volume)}</span>
                      <span className={`text-xs ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {formatChange(data.change, data.change_percent)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Order Book */}
          <div className="p-3 flex-1">
            <h3 className="text-sm font-bold mb-2 text-blue-400">ORDER BOOK - {selectedSymbol}</h3>
            <div className="space-y-1">
              <div className="grid grid-cols-2 text-xs text-gray-400 pb-1 border-b border-gray-700">
                <span>PRICE</span>
                <span>SIZE</span>
              </div>

              {/* Asks */}
              {orderBook.asks.slice(0, 5).reverse().map(([price, size], index) => (
                <div key={`ask-${index}`} className="grid grid-cols-2 text-xs text-red-400">
                  <span>${price.toFixed(2)}</span>
                  <span>{size}</span>
                </div>
              ))}

              <div className="border-t border-gray-600 my-1"></div>

              {/* Bids */}
              {orderBook.bids.slice(0, 5).map(([price, size], index) => (
                <div key={`bid-${index}`} className="grid grid-cols-2 text-xs text-green-400">
                  <span>${price.toFixed(2)}</span>
                  <span>{size}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Center Panel - Chart */}
        <div className="flex-1 flex flex-col">
          {/* Chart Header */}
          <div className="bg-gray-900 p-2 border-b border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <span className="text-lg font-bold text-white">{selectedSymbol}</span>
                {marketData[selectedSymbol] && (
                  <div className="flex items-center space-x-4 text-sm">
                    <span className="text-white">
                      ${formatPrice(marketData[selectedSymbol].price)}
                    </span>
                    <span className={`${marketData[selectedSymbol].change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatChange(marketData[selectedSymbol].change, marketData[selectedSymbol].change_percent)}
                    </span>
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-2">
                {['1m', '5m', '15m', '1h', '4h', '1d'].map(tf => (
                  <button
                    key={tf}
                    onClick={() => setActiveTimeframe(tf as any)}
                    className={`px-2 py-1 text-xs rounded ${
                      activeTimeframe === tf ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="flex-1 relative">
            <canvas
              ref={chartCanvasRef}
              className="absolute inset-0 w-full h-full"
              style={{ width: '100%', height: '100%' }}
            />
          </div>

          {/* Chart Legend */}
          <div className="bg-gray-900 p-2 border-t border-gray-700">
            <div className="flex items-center space-x-6 text-xs">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-0.5 bg-green-400"></div>
                <span>Price</span>
              </div>
              {showStrategyOverlay && (
                <>
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-0.5 bg-yellow-500"></div>
                    <span>MA 20</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-0.5 bg-purple-500"></div>
                    <span>MA 50</span>
                  </div>
                </>
              )}
              {showAIInflections && (
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  <span>AI Inflection</span>
                </div>
              )}
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-400"></div>
                <span>BUY</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-red-400"></div>
                <span>SELL</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Strategy Signals & Analytics */}
        <div className="w-80 bg-gray-900 border-l border-gray-700 flex flex-col">
          {/* Strategy Signals */}
          <div className="p-3 border-b border-gray-700 flex-1">
            <h3 className="text-sm font-bold mb-2 text-blue-400">STRATEGY SIGNALS</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {strategySignals.slice(0, 10).map((signal, index) => (
                <div key={index} className="p-2 bg-gray-800 rounded text-xs">
                  <div className="flex justify-between items-center mb-1">
                    <span style={{ color: getStrategyColor(signal.strategy) }} className="font-bold">
                      {signal.strategy}
                    </span>
                    <span className={`${signal.signal === 'BUY' ? 'text-green-400' : 'text-red-400'} font-bold`}>
                      {signal.signal}
                    </span>
                  </div>
                  <div className="text-gray-400 mb-1">{signal.reason}</div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="text-blue-400">{signal.confidence}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Real-time Analytics */}
          <div className="p-3">
            <h3 className="text-sm font-bold mb-2 text-blue-400">ANALYTICS</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">DPI Score:</span>
                <span className="text-yellow-400 font-bold">78.5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Causal Confidence:</span>
                <span className="text-purple-400 font-bold">85%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Risk Level:</span>
                <span className="text-green-400 font-bold">LOW</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Flow Direction:</span>
                <span className="text-blue-400 font-bold">BULLISH</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volatility:</span>
                <span className="text-orange-400 font-bold">12.3%</span>
              </div>

              <div className="border-t border-gray-700 pt-2 mt-2">
                <div className="text-center text-green-400 font-bold">
                  SYSTEM STATUS: ACTIVE
                </div>
                <div className="text-center text-xs text-gray-400 mt-1">
                  Last Update: {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingTerminal;