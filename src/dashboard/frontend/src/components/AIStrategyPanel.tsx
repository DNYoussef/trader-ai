import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  BoltIcon,
  ShieldCheckIcon,
  ArrowTrendingUpIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';

interface AIStrategyPanelProps {
  features?: number[];
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface StrategyPrediction {
  strategy_index: number;
  strategy_name: string;
  confidence: number;
  probabilities: number[];
  top_3_strategies: Array<{
    index: number;
    name: string;
    probability: number;
  }>;
  reasoning?: string;
  risk_level?: string;
}

interface ModelInfo {
  model_loaded: boolean;
  device: string;
  metadata: {
    iteration?: string;
    clean_accuracy?: number;
    noisy_accuracy?: number;
    loaded_from?: string;
  };
  strategies: string[];
}

const strategyIcons: Record<string, React.ElementType> = {
  'Inequality Mispricing Exploit': ChartBarIcon,
  'Volatility Arbitrage': BoltIcon,
  'Narrative Gap Trade': ArrowTrendingUpIcon,
  'Correlation Breakdown': ExclamationTriangleIcon,
  'Barbell Position': ShieldCheckIcon,
  'Antifragile Convex': SparklesIcon,
  'Black Swan Hunt': ExclamationTriangleIcon,
  'Risk Parity Rebalance': CpuChipIcon,
};

const strategyColors: Record<string, string> = {
  'Inequality Mispricing Exploit': 'bg-purple-500',
  'Volatility Arbitrage': 'bg-yellow-500',
  'Narrative Gap Trade': 'bg-blue-500',
  'Correlation Breakdown': 'bg-red-500',
  'Barbell Position': 'bg-green-500',
  'Antifragile Convex': 'bg-indigo-500',
  'Black Swan Hunt': 'bg-gray-800',
  'Risk Parity Rebalance': 'bg-teal-500',
};

const riskColors: Record<string, string> = {
  'LOW': 'text-green-600 bg-green-100',
  'MEDIUM': 'text-yellow-600 bg-yellow-100',
  'HIGH': 'text-red-600 bg-red-100',
};

export const AIStrategyPanel: React.FC<AIStrategyPanelProps> = ({
  features,
  autoRefresh = true,
  refreshInterval = 5000,
}) => {
  const [prediction, setPrediction] = useState<StrategyPrediction | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Fetch model info on mount
  useEffect(() => {
    fetchModelInfo();
  }, []);

  // Fetch predictions periodically or when features change
  useEffect(() => {
    if (features && features.length === 32) {
      fetchPrediction(features);
    } else if (autoRefresh) {
      fetchPrediction();
    }

    if (autoRefresh) {
      const interval = setInterval(() => {
        if (features && features.length === 32) {
          fetchPrediction(features);
        } else {
          fetchPrediction();
        }
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [features, autoRefresh, refreshInterval]);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/ai/model/info');
      setModelInfo(response.data);
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const fetchPrediction = async (customFeatures?: number[]) => {
    setIsLoading(true);
    setError(null);

    try {
      let response;
      if (customFeatures) {
        response = await axios.post('http://localhost:8000/api/ai/predict', {
          features: customFeatures,
        });
      } else {
        response = await axios.get('http://localhost:8000/api/ai/predict');
      }

      setPrediction(response.data);
      setLastUpdate(new Date());

      // Fetch explanation if available
      if (response.data) {
        const explainResponse = await axios.post('http://localhost:8000/api/ai/explain', {
          prediction: response.data,
        });
        setExplanation(explainResponse.data);
      }
    } catch (err) {
      console.error('Failed to fetch prediction:', err);
      setError('Failed to get AI prediction');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStrategySelect = (strategyName: string) => {
    // This could trigger strategy execution or show more details
    console.log('Selected strategy:', strategyName);
    // You can emit an event or call a parent callback here
  };

  if (isLoading && !prediction) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400">Analyzing market conditions...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
          <p className="text-red-600 dark:text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
              <CpuChipIcon className="w-6 h-6 mr-2 text-indigo-500" />
              AI Strategy Recommendations
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Powered by HRM 32D Model • Last update: {lastUpdate.toLocaleTimeString()}
            </p>
          </div>
          {modelInfo && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                {modelInfo.device.toUpperCase()}
              </span>
              {modelInfo.metadata.clean_accuracy && (
                <span className="ml-2">
                  Accuracy: {(modelInfo.metadata.clean_accuracy * 100).toFixed(1)}%
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {prediction && (
        <div className="p-6">
          {/* Primary Recommendation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-3">
              RECOMMENDED STRATEGY
            </h3>
            <div
              className={`relative rounded-lg p-6 cursor-pointer transition-transform hover:scale-[1.02] ${
                strategyColors[prediction.strategy_name]
              }`}
              onClick={() => handleStrategySelect(prediction.strategy_name)}
            >
              <div className="flex items-center justify-between text-white">
                <div className="flex items-center">
                  {React.createElement(
                    strategyIcons[prediction.strategy_name] || ChartBarIcon,
                    { className: 'w-8 h-8 mr-3' }
                  )}
                  <div>
                    <h4 className="text-2xl font-bold">{prediction.strategy_name}</h4>
                    <p className="text-sm opacity-90 mt-1">
                      Strategy #{prediction.strategy_index + 1} of 8
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm opacity-90">Confidence</div>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="mt-4 bg-white/20 rounded-full h-2 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${prediction.confidence * 100}%` }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-white/60"
                />
              </div>

              {/* Risk Level Badge */}
              {prediction.risk_level && (
                <div className="absolute top-4 right-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    riskColors[prediction.risk_level]
                  }`}>
                    {prediction.risk_level} RISK
                  </span>
                </div>
              )}
            </div>
          </motion.div>

          {/* Reasoning */}
          {prediction.reasoning && (
            <div className="mb-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">
                AI REASONING
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                {prediction.reasoning}
              </p>
            </div>
          )}

          {/* Alternative Strategies */}
          {prediction.top_3_strategies && prediction.top_3_strategies.length > 1 && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-3">
                ALTERNATIVE STRATEGIES
              </h3>
              <div className="space-y-2">
                {prediction.top_3_strategies.slice(1).map((strategy) => (
                  <motion.div
                    key={strategy.index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-colors"
                    onClick={() => handleStrategySelect(strategy.name)}
                  >
                    <div className="flex items-center">
                      {React.createElement(
                        strategyIcons[strategy.name] || ChartBarIcon,
                        { className: 'w-5 h-5 mr-2 text-gray-600 dark:text-gray-400' }
                      )}
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {strategy.name}
                      </span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-3">
                        <div
                          className="bg-indigo-500 h-2 rounded-full"
                          style={{ width: `${strategy.probability * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400 w-12 text-right">
                        {(strategy.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* All Strategy Probabilities */}
          <div>
            <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-3">
              ALL STRATEGY PROBABILITIES
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {modelInfo?.strategies.map((strategyName, idx) => (
                <div
                  key={idx}
                  className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900 rounded text-xs"
                >
                  <span className="text-gray-600 dark:text-gray-400 truncate mr-2">
                    {strategyName}
                  </span>
                  <span className="font-mono text-gray-700 dark:text-gray-300">
                    {prediction.probabilities[idx]
                      ? (prediction.probabilities[idx] * 100).toFixed(1)
                      : '0.0'}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-6 flex space-x-3">
            <button
              onClick={() => fetchPrediction(features)}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Refresh Analysis
            </button>
            <button
              onClick={() => handleStrategySelect(prediction.strategy_name)}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Execute Strategy
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIStrategyPanel;