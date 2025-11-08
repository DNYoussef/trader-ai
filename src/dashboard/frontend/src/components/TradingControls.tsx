import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PlayIcon,
  StopIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ClockIcon,
  AdjustmentsHorizontalIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import toast from 'react-hot-toast';

interface TradingControlsProps {
  onTradeExecute?: (trade: TradeOrder) => void;
  currentStrategy?: string;
  features?: number[];
}

interface TradeOrder {
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  time_in_force: 'day' | 'gtc' | 'ioc' | 'fok';
  limit_price?: number;
  stop_price?: number;
  strategy?: string;
  timeframe?: string;
}

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

const timeframes = [
  { id: '1m', label: '1M', minutes: 1 },
  { id: '5m', label: '5M', minutes: 5 },
  { id: '15m', label: '15M', minutes: 15 },
  { id: '30m', label: '30M', minutes: 30 },
  { id: '1h', label: '1H', minutes: 60 },
  { id: '4h', label: '4H', minutes: 240 },
  { id: '1d', label: '1D', minutes: 1440 },
  { id: '1w', label: '1W', minutes: 10080 },
];

const symbols = [
  'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
  'TSLA', 'META', 'BRK.B', 'JPM', 'V',
  'GLD', 'SLV', 'USO', 'TLT', 'VXX'
];

const strategies = [
  'Manual Trading',
  'Inequality Mispricing Exploit',
  'Volatility Arbitrage',
  'Narrative Gap Trade',
  'Correlation Breakdown',
  'Barbell Position',
  'Antifragile Convex',
  'Black Swan Hunt',
  'Risk Parity Rebalance',
  'Momentum Follow',
  'Mean Reversion',
  'Pairs Trading',
  'Options Spread'
];

export const TradingControls: React.FC<TradingControlsProps> = ({
  onTradeExecute,
  currentStrategy,
  features
}) => {
  // Trading state
  const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [selectedStrategy, setSelectedStrategy] = useState(currentStrategy || 'Manual Trading');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop' | 'stop_limit'>('market');
  const [timeInForce, setTimeInForce] = useState<'day' | 'gtc' | 'ioc' | 'fok'>('day');
  const [quantity, setQuantity] = useState(1);
  const [limitPrice, setLimitPrice] = useState<number | ''>('');
  const [stopPrice, setStopPrice] = useState<number | ''>('');

  // UI state
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [positions, setPositions] = useState<Position[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre' | 'post'>('closed');

  // Update strategy when prop changes
  useEffect(() => {
    if (currentStrategy) {
      setSelectedStrategy(currentStrategy);
    }
  }, [currentStrategy]);

  // Fetch current price and market status
  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/market/quote/${selectedSymbol}`);
        setCurrentPrice(response.data.price || 0);
        setMarketStatus(response.data.market_status || 'closed');
      } catch (error) {
        console.error('Failed to fetch market data:', error);
      }
    };

    fetchMarketData();
    const interval = setInterval(fetchMarketData, 5000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  // Fetch positions
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/positions');
        setPositions(response.data.positions || []);
      } catch (error) {
        console.error('Failed to fetch positions:', error);
      }
    };

    fetchPositions();
    const interval = setInterval(fetchPositions, 10000);
    return () => clearInterval(interval);
  }, []);

  const executeTrade = async (side: 'buy' | 'sell') => {
    setIsExecuting(true);

    const trade: TradeOrder = {
      symbol: selectedSymbol,
      side,
      quantity,
      order_type: orderType,
      time_in_force: timeInForce,
      strategy: selectedStrategy,
      timeframe: selectedTimeframe,
    };

    if (orderType === 'limit' || orderType === 'stop_limit') {
      if (!limitPrice) {
        toast.error('Limit price is required for limit orders');
        setIsExecuting(false);
        return;
      }
      trade.limit_price = Number(limitPrice);
    }

    if (orderType === 'stop' || orderType === 'stop_limit') {
      if (!stopPrice) {
        toast.error('Stop price is required for stop orders');
        setIsExecuting(false);
        return;
      }
      trade.stop_price = Number(stopPrice);
    }

    try {
      // Execute trade via API
      const response = await axios.post('http://localhost:8000/api/trade/execute', trade);

      if (response.data.success) {
        toast.success(
          <div>
            <CheckCircleIcon className="w-5 h-5 inline mr-2" />
            {side.toUpperCase()} order executed: {quantity} {selectedSymbol}
          </div>
        );

        // Callback to parent
        if (onTradeExecute) {
          onTradeExecute(trade);
        }

        // Reset form
        setQuantity(1);
        setLimitPrice('');
        setStopPrice('');
      } else {
        toast.error(response.data.error || 'Trade execution failed');
      }
    } catch (error: any) {
      toast.error(
        <div>
          <XCircleIcon className="w-5 h-5 inline mr-2" />
          {error.response?.data?.detail || 'Failed to execute trade'}
        </div>
      );
    } finally {
      setIsExecuting(false);
    }
  };

  const closePosition = async (symbol: string) => {
    try {
      const response = await axios.post(`http://localhost:8000/api/trade/close/${symbol}`);
      if (response.data.success) {
        toast.success(`Closed position: ${symbol}`);
      }
    } catch (error) {
      toast.error('Failed to close position');
    }
  };

  const closeAllPositions = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/trade/close-all');
      if (response.data.success) {
        toast.success('All positions closed');
      }
    } catch (error) {
      toast.error('Failed to close all positions');
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header with Market Status */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
            <ChartBarIcon className="w-6 h-6 mr-2 text-indigo-500" />
            Trading Controls
          </h2>
          <div className="flex items-center space-x-4">
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              marketStatus === 'open'
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                : marketStatus === 'pre' || marketStatus === 'post'
                ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
            }`}>
              Market: {marketStatus.toUpperCase()}
            </span>
            <button
              onClick={() => setIsAdvancedMode(!isAdvancedMode)}
              className="text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <AdjustmentsHorizontalIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Timeframe Selection */}
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            TIMEFRAME
          </label>
          <div className="flex space-x-2">
            {timeframes.map((tf) => (
              <button
                key={tf.id}
                onClick={() => setSelectedTimeframe(tf.id)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  selectedTimeframe === tf.id
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>

        {/* Symbol and Strategy Selection */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              SYMBOL
            </label>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
            {currentPrice > 0 && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Current: ${currentPrice.toFixed(2)}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              STRATEGY
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
            >
              {strategies.map(strategy => (
                <option key={strategy} value={strategy}>{strategy}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Order Configuration */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              ORDER TYPE
            </label>
            <select
              value={orderType}
              onChange={(e) => setOrderType(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
            >
              <option value="market">Market</option>
              <option value="limit">Limit</option>
              <option value="stop">Stop</option>
              <option value="stop_limit">Stop Limit</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              QUANTITY
            </label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(Number(e.target.value))}
              min="1"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        {/* Advanced Order Options */}
        <AnimatePresence>
          {(orderType === 'limit' || orderType === 'stop_limit') && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6"
            >
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                LIMIT PRICE
              </label>
              <input
                type="number"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value ? Number(e.target.value) : '')}
                step="0.01"
                placeholder="Enter limit price"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
              />
            </motion.div>
          )}

          {(orderType === 'stop' || orderType === 'stop_limit') && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6"
            >
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                STOP PRICE
              </label>
              <input
                type="number"
                value={stopPrice}
                onChange={(e) => setStopPrice(e.target.value ? Number(e.target.value) : '')}
                step="0.01"
                placeholder="Enter stop price"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
              />
            </motion.div>
          )}

          {isAdvancedMode && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6"
            >
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                TIME IN FORCE
              </label>
              <select
                value={timeInForce}
                onChange={(e) => setTimeInForce(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500"
              >
                <option value="day">Day</option>
                <option value="gtc">Good Till Cancelled</option>
                <option value="ioc">Immediate or Cancel</option>
                <option value="fok">Fill or Kill</option>
              </select>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Trade Execution Buttons */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <button
            onClick={() => executeTrade('buy')}
            disabled={isExecuting || marketStatus === 'closed'}
            className={`py-3 px-4 rounded-lg font-bold text-white transition-all flex items-center justify-center ${
              isExecuting || marketStatus === 'closed'
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 active:scale-95'
            }`}
          >
            {isExecuting ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
            ) : (
              <>
                <ArrowTrendingUpIcon className="w-5 h-5 mr-2" />
                BUY {selectedSymbol}
              </>
            )}
          </button>

          <button
            onClick={() => executeTrade('sell')}
            disabled={isExecuting || marketStatus === 'closed'}
            className={`py-3 px-4 rounded-lg font-bold text-white transition-all flex items-center justify-center ${
              isExecuting || marketStatus === 'closed'
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 active:scale-95'
            }`}
          >
            {isExecuting ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
            ) : (
              <>
                <ArrowTrendingDownIcon className="w-5 h-5 mr-2" />
                SELL {selectedSymbol}
              </>
            )}
          </button>
        </div>

        {/* Open Positions */}
        {positions.length > 0 && (
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                OPEN POSITIONS ({positions.length})
              </h3>
              <button
                onClick={closeAllPositions}
                className="text-xs text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
              >
                Close All
              </button>
            </div>
            <div className="space-y-2">
              {positions.map((position) => (
                <div
                  key={position.symbol}
                  className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900 rounded"
                >
                  <div className="flex items-center">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {position.symbol}
                    </span>
                    <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                      {position.quantity} @ ${position.avg_price.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`text-sm font-medium ${
                      position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {position.pnl >= 0 ? '+' : ''}{position.pnl.toFixed(2)} ({position.pnl_percent.toFixed(2)}%)
                    </span>
                    <button
                      onClick={() => closePosition(position.symbol)}
                      className="text-xs px-2 py-1 bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 rounded hover:bg-red-200 dark:hover:bg-red-800"
                    >
                      Close
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingControls;