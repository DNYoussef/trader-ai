import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { useTradeExecution, formatTradeConfirmation } from '../hooks/useTradeExecution';

interface QuickTradeProps {
  symbol?: string;
}

export const QuickTrade: React.FC<QuickTradeProps> = ({ symbol = 'SPY' }) => {
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState<number>(1);
  const [limitPrice, setLimitPrice] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>('');

  const { executeQuickTrade, executeLimitOrder, isExecuting, lastTrade, error } = useTradeExecution();

  const handleBuy = async () => {
    setFeedback('');

    try {
      let result;
      if (orderType === 'market') {
        result = await executeQuickTrade(symbol, quantity, 'buy');
      } else {
        result = await executeLimitOrder(symbol, quantity, 'buy', limitPrice);
      }

      if (result.success) {
        setFeedback(`SUCCESS: ${formatTradeConfirmation(result)}`);
        // Reset quantity after successful trade
        setQuantity(1);
      } else {
        setFeedback(`FAILED: ${result.error || 'Trade failed'}`);
      }
    } catch (err) {
      setFeedback(`ERROR: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleSell = async () => {
    setFeedback('');

    try {
      let result;
      if (orderType === 'market') {
        result = await executeQuickTrade(symbol, quantity, 'sell');
      } else {
        result = await executeLimitOrder(symbol, quantity, 'sell', limitPrice);
      }

      if (result.success) {
        setFeedback(`SUCCESS: ${formatTradeConfirmation(result)}`);
        // Reset quantity after successful trade
        setQuantity(1);
      } else {
        setFeedback(`FAILED: ${result.error || 'Trade failed'}`);
      }
    } catch (err) {
      setFeedback(`ERROR: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Quick Trade</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Symbol:</span>
          <span className="font-medium">{symbol}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Quantity:</span>
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(Number(e.target.value))}
            className="w-20 px-2 py-1 border rounded text-center"
            min="1"
          />
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600">Order Type:</span>
          <select
            value={orderType}
            onChange={(e) => setOrderType(e.target.value as 'market' | 'limit')}
            className="px-2 py-1 border rounded"
          >
            <option value="market">Market</option>
            <option value="limit">Limit</option>
          </select>
        </div>

        {orderType === 'limit' && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Limit Price:</span>
            <input
              type="number"
              value={limitPrice}
              onChange={(e) => setLimitPrice(Number(e.target.value))}
              className="w-24 px-2 py-1 border rounded text-center"
              step="0.01"
              min="0"
            />
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 pt-4">
          <button
            onClick={handleBuy}
            disabled={isExecuting}
            className={`${
              isExecuting
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-500 hover:bg-green-600'
            } text-white font-semibold py-3 px-6 rounded-lg transition-colors`}
          >
            {isExecuting ? 'Processing...' : 'BUY'}
          </button>
          <button
            onClick={handleSell}
            disabled={isExecuting}
            className={`${
              isExecuting
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-red-500 hover:bg-red-600'
            } text-white font-semibold py-3 px-6 rounded-lg transition-colors`}
          >
            {isExecuting ? 'Processing...' : 'SELL'}
          </button>
        </div>

        {feedback && (
          <div className={`mt-3 text-sm text-center ${
            feedback.startsWith('SUCCESS') ? 'text-green-600' : 'text-red-600'
          }`}>
            {feedback}
          </div>
        )}
      </CardContent>
    </Card>
  );
};