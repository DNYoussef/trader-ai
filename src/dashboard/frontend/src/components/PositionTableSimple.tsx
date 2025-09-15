import React from 'react';

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

interface PositionTableProps {
  positions: Position[];
}

export const PositionTable: React.FC<PositionTableProps> = ({ positions }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Open Positions
      </h3>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-white">Symbol</th>
              <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">Quantity</th>
              <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">Entry Price</th>
              <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">Current Price</th>
              <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">P&L</th>
              <th className="text-right py-3 px-4 font-medium text-gray-900 dark:text-white">P&L %</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position, index) => (
              <tr key={position.symbol} className="border-b border-gray-100 dark:border-gray-800">
                <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">{position.symbol}</td>
                <td className="py-3 px-4 text-right text-gray-600 dark:text-gray-400">{position.quantity}</td>
                <td className="py-3 px-4 text-right text-gray-600 dark:text-gray-400">
                  ${position.entry_price.toFixed(2)}
                </td>
                <td className="py-3 px-4 text-right text-gray-600 dark:text-gray-400">
                  ${position.current_price.toFixed(2)}
                </td>
                <td className={`py-3 px-4 text-right font-medium ${
                  position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  ${position.pnl.toFixed(2)}
                </td>
                <td className={`py-3 px-4 text-right font-medium ${
                  position.pnl_percent >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {position.pnl_percent.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {positions.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No open positions
          </div>
        )}
      </div>
    </div>
  );
};