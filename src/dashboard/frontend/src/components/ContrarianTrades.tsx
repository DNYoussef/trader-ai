import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

// Icons
const TrendingUp = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TrendingDown = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
  </svg>
);

const AlertCircle = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const Fire = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14.015 9.879 16.12z" />
  </svg>
);

interface ContrarianOpportunity {
  id: string;
  symbol: string;
  thesis: string;
  consensusView: string;
  contrarianView: string;
  inequalityCorrelation: number;
  convictionScore: number;
  expectedPayoff: number;
  timeframeDays: number;
  entryPrice: number;
  targetPrice: number;
  stopLoss: number;
  currentPrice: number;
  historicalAccuracy: number;
  garyMomentScore: number;
  supportingData: {
    metric: string;
    value: number;
    trend: 'up' | 'down';
  }[];
}

interface ContrarianTradesProps {
  opportunities: ContrarianOpportunity[];
  onExecuteTrade?: (opportunity: ContrarianOpportunity) => void;
}

export const ContrarianTrades: React.FC<ContrarianTradesProps> = ({
  opportunities,
  onExecuteTrade
}) => {
  const [selectedOpportunity, setSelectedOpportunity] = useState<ContrarianOpportunity | null>(null);
  const [sortBy, setSortBy] = useState<'conviction' | 'payoff' | 'gary'>('gary');

  // Sort opportunities based on selected criteria
  const sortedOpportunities = useMemo(() => {
    const sorted = [...opportunities];
    sorted.sort((a, b) => {
      switch (sortBy) {
        case 'conviction':
          return b.convictionScore - a.convictionScore;
        case 'payoff':
          return b.expectedPayoff - a.expectedPayoff;
        case 'gary':
          return b.garyMomentScore - a.garyMomentScore;
        default:
          return 0;
      }
    });
    return sorted;
  }, [opportunities, sortBy]);

  // Calculate risk/reward for selected opportunity
  const calculateRiskReward = (opp: ContrarianOpportunity) => {
    const risk = ((opp.currentPrice - opp.stopLoss) / opp.currentPrice) * 100;
    const reward = ((opp.targetPrice - opp.currentPrice) / opp.currentPrice) * 100;
    return { risk, reward, ratio: reward / risk };
  };

  // Format conviction score color
  const getConvictionColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.7) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  // Gary Moment indicator
  const GaryMomentBadge: React.FC<{ score: number }> = ({ score }) => {
    if (score < 0.7) return null;

    return (
      <motion.div
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ repeat: Infinity, duration: 2 }}
        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-bold ${
          score >= 0.9 ? 'bg-red-500 text-white' :
          score >= 0.8 ? 'bg-orange-500 text-white' :
          'bg-yellow-500 text-white'
        }`}
      >
        <Fire />
        <span className="ml-1">Gary Moment {(score * 100).toFixed(0)}%</span>
      </motion.div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-2">Contrarian Opportunities</h2>
        <p className="text-purple-100">
          High-conviction trades where consensus is blind to inequality effects
        </p>
        <div className="mt-4 flex gap-4">
          <button
            onClick={() => setSortBy('gary')}
            className={`px-4 py-2 rounded ${sortBy === 'gary' ? 'bg-white text-purple-600' : 'bg-purple-500'}`}
          >
            Gary Score
          </button>
          <button
            onClick={() => setSortBy('conviction')}
            className={`px-4 py-2 rounded ${sortBy === 'conviction' ? 'bg-white text-purple-600' : 'bg-purple-500'}`}
          >
            Conviction
          </button>
          <button
            onClick={() => setSortBy('payoff')}
            className={`px-4 py-2 rounded ${sortBy === 'payoff' ? 'bg-white text-purple-600' : 'bg-purple-500'}`}
          >
            Expected Payoff
          </button>
        </div>
      </div>

      {/* Opportunities Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {sortedOpportunities.slice(0, 6).map((opp) => {
          const riskReward = calculateRiskReward(opp);

          return (
            <motion.div
              key={opp.id}
              whileHover={{ scale: 1.02 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-5 cursor-pointer border-2 border-transparent hover:border-purple-500 transition-all"
              onClick={() => setSelectedOpportunity(opp)}
            >
              {/* Header */}
              <div className="flex justify-between items-start mb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                      {opp.symbol}
                    </h3>
                    <GaryMomentBadge score={opp.garyMomentScore} />
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {opp.timeframeDays} day opportunity
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-purple-600">
                    {opp.expectedPayoff.toFixed(1)}x
                  </div>
                  <div className="text-xs text-gray-500">Expected Return</div>
                </div>
              </div>

              {/* Thesis */}
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
                  Contrarian Thesis:
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {opp.thesis}
                </p>
              </div>

              {/* Consensus vs Reality */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-red-50 dark:bg-red-900/20 p-2 rounded">
                  <p className="text-xs font-medium text-red-600 dark:text-red-400 mb-1">
                    Consensus Says:
                  </p>
                  <p className="text-xs text-red-700 dark:text-red-300">
                    {opp.consensusView}
                  </p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 p-2 rounded">
                  <p className="text-xs font-medium text-green-600 dark:text-green-400 mb-1">
                    Reality Is:
                  </p>
                  <p className="text-xs text-green-700 dark:text-green-300">
                    {opp.contrarianView}
                  </p>
                </div>
              </div>

              {/* Metrics Bar */}
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center gap-4">
                  <div>
                    <span className="text-xs text-gray-500">Conviction</span>
                    <div className="flex items-center">
                      <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                        <div
                          className={`h-2 rounded-full ${
                            opp.convictionScore >= 0.8 ? 'bg-green-500' :
                            opp.convictionScore >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${opp.convictionScore * 100}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium">{(opp.convictionScore * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">R/R Ratio</span>
                    <div className="text-sm font-bold text-gray-900 dark:text-gray-100">
                      {riskReward.ratio.toFixed(1)}:1
                    </div>
                  </div>
                </div>
              </div>

              {/* Price Targets */}
              <div className="flex justify-between items-center pt-3 border-t border-gray-200 dark:border-gray-700">
                <div className="text-xs">
                  <span className="text-gray-500">Entry: </span>
                  <span className="font-medium">${opp.entryPrice.toFixed(2)}</span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">Target: </span>
                  <span className="font-medium text-green-600">${opp.targetPrice.toFixed(2)}</span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">Stop: </span>
                  <span className="font-medium text-red-600">${opp.stopLoss.toFixed(2)}</span>
                </div>
              </div>

              {/* Execute Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={(e) => {
                  e.stopPropagation();
                  onExecuteTrade?.(opp);
                }}
                className="w-full mt-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-indigo-700 transition-all"
              >
                Execute Trade
              </motion.button>
            </motion.div>
          );
        })}
      </div>

      {/* Selected Opportunity Detail Modal */}
      <AnimatePresence>
        {selectedOpportunity && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedOpportunity(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex justify-between items-start mb-6">
                <div>
                  <div className="flex items-center gap-3">
                    <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                      {selectedOpportunity.symbol}
                    </h2>
                    <GaryMomentBadge score={selectedOpportunity.garyMomentScore} />
                  </div>
                  <p className="text-gray-600 dark:text-gray-400 mt-2">
                    {selectedOpportunity.thesis}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedOpportunity(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>

              {/* Supporting Data */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">Supporting Inequality Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {selectedOpportunity.supportingData.map((data, idx) => (
                    <div key={idx} className="bg-gray-50 dark:bg-gray-700 p-3 rounded">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-500">{data.metric}</span>
                        {data.trend === 'up' ? (
                          <span className="text-green-500"><TrendingUp /></span>
                        ) : (
                          <span className="text-red-500"><TrendingDown /></span>
                        )}
                      </div>
                      <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                        {data.value.toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Risk/Reward Visualization */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">Risk/Reward Analysis</h3>
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <span className="text-sm text-gray-500">Current Price</span>
                      <div className="text-2xl font-bold">${selectedOpportunity.currentPrice.toFixed(2)}</div>
                    </div>
                    <div className="text-center">
                      <span className="text-sm text-gray-500">Risk/Reward Ratio</span>
                      <div className="text-2xl font-bold text-purple-600">
                        {calculateRiskReward(selectedOpportunity).ratio.toFixed(1)}:1
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500">Expected Return</span>
                      <div className="text-2xl font-bold text-green-600">
                        +{calculateRiskReward(selectedOpportunity).reward.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Price Range Visualization */}
                  <div className="relative h-8 bg-gray-200 dark:bg-gray-600 rounded-full">
                    <div
                      className="absolute top-0 h-full bg-red-500 rounded-l-full"
                      style={{
                        left: '0%',
                        width: `${(selectedOpportunity.stopLoss / selectedOpportunity.targetPrice) * 100}%`
                      }}
                    />
                    <div
                      className="absolute top-0 h-full bg-green-500 rounded-r-full"
                      style={{
                        left: `${(selectedOpportunity.currentPrice / selectedOpportunity.targetPrice) * 100}%`,
                        width: `${((selectedOpportunity.targetPrice - selectedOpportunity.currentPrice) / selectedOpportunity.targetPrice) * 100}%`
                      }}
                    />
                    <div
                      className="absolute top-1/2 transform -translate-y-1/2 w-1 h-6 bg-black"
                      style={{
                        left: `${(selectedOpportunity.currentPrice / selectedOpportunity.targetPrice) * 100}%`
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-2 text-xs">
                    <span className="text-red-600">Stop: ${selectedOpportunity.stopLoss.toFixed(2)}</span>
                    <span className="font-bold">Current: ${selectedOpportunity.currentPrice.toFixed(2)}</span>
                    <span className="text-green-600">Target: ${selectedOpportunity.targetPrice.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Historical Performance */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">Historical Accuracy</h3>
                <div className="flex items-center gap-4">
                  <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                    <div
                      className="h-4 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"
                      style={{ width: `${selectedOpportunity.historicalAccuracy * 100}%` }}
                    />
                  </div>
                  <span className="text-lg font-bold">
                    {(selectedOpportunity.historicalAccuracy * 100).toFixed(0)}% Success Rate
                  </span>
                </div>
              </div>

              {/* Execute Trade Button */}
              <div className="flex gap-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    onExecuteTrade?.(selectedOpportunity);
                    setSelectedOpportunity(null);
                  }}
                  className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-indigo-700"
                >
                  Execute Trade - Bet on Inequality
                </motion.button>
                <button
                  onClick={() => setSelectedOpportunity(null)}
                  className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  Cancel
                </button>
              </div>

              {/* Gary Quote */}
              <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-700">
                <p className="text-sm text-purple-700 dark:text-purple-300 italic">
                  "The more people say I'm wrong, the more money I'm going to make. When everyone thinks
                  you're wrong about inequality's effects, that's when the payoff is massive."
                  - Gary Stevenson
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ContrarianTrades;