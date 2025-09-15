import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface QuickTradeProps {
  symbol?: string;
}

export const QuickTrade: React.FC<QuickTradeProps> = ({ symbol = 'SPY' }) => {
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState<number>(1);

  const handleBuy = () => {
    console.log(`Buy ${quantity} shares of ${symbol} at ${orderType}`);
    // TODO: Integrate with trading API
  };

  const handleSell = () => {
    console.log(`Sell ${quantity} shares of ${symbol} at ${orderType}`);
    // TODO: Integrate with trading API
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

        <div className="grid grid-cols-2 gap-2 pt-4">
          <button
            onClick={handleBuy}
            className="bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
          >
            BUY
          </button>
          <button
            onClick={handleSell}
            className="bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
          >
            SELL
          </button>
        </div>
      </CardContent>
    </Card>
  );
};