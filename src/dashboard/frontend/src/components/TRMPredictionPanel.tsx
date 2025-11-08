import React, { useState, useEffect } from 'react';

interface TRMPrediction {
  timestamp: string;
  strategy_id: number;
  strategy_name: string;
  confidence: number;
  probabilities: {
    [key: string]: number;
  };
  raw_features?: number[];
  normalized_features?: number[];
  halt_probability?: number;
  mock_data?: boolean;
  model_metadata?: {
    recursion_cycles: number;
    latent_steps: number;
    effective_depth: number;
  };
}

export const TRMPredictionPanel: React.FC = () => {
  const [prediction, setPrediction] = useState<TRMPrediction | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Connect to TRM WebSocket on port 8001
    const ws = new WebSocket('ws://localhost:8001/ws/dashboard');

    ws.onopen = () => {
      console.log('Connected to TRM stream');
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'trm_prediction') {
          setPrediction(msg.data);
        }
      } catch (err) {
        console.error('Error parsing TRM message:', err);
      }
    };

    ws.onerror = () => {
      setError('Connection failed');
      setConnected(false);
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('Disconnected from TRM stream');
    };

    return () => {
      ws.close();
    };
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900 rounded-lg p-6">
        <h2 className="text-xl font-bold text-red-900 dark:text-red-100 mb-2">
          TRM Connection Error
        </h2>
        <p className="text-red-700 dark:text-red-300">
          {error} - Make sure TRM WebSocket server is running on port 8001
        </p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  if (!connected || !prediction) {
    return (
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">
            Connecting to TRM predictions...
          </p>
        </div>
      </div>
    );
  }

  // Get top 3 strategies
  const topStrategies = Object.entries(prediction.probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  // Strategy colors
  const strategyColors: { [key: string]: string } = {
    ultra_defensive: 'bg-blue-600',
    defensive: 'bg-blue-500',
    balanced_safe: 'bg-green-500',
    balanced_growth: 'bg-green-600',
    balanced_aggressive: 'bg-yellow-500',
    aggressive_growth: 'bg-orange-500',
    max_growth: 'bg-red-500',
    tactical_opportunity: 'bg-purple-500'
  };

  const getStrategyColor = (name: string) => {
    return strategyColors[name] || 'bg-gray-500';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              TRM Strategy Recommendation
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Real-time predictions • 10 base features
              {prediction.mock_data && ' • Using MOCK DATA'}
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* Connection status */}
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'} mr-2`}></div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {connected ? 'Live' : 'Disconnected'}
              </span>
            </div>

            {/* Timestamp */}
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {new Date(prediction.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>

      {/* Main Prediction */}
      <div className="p-6">
        <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg p-6 text-white mb-6">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm opacity-90 mb-1">Recommended Strategy</p>
              <h3 className="text-3xl font-bold capitalize">
                {prediction.strategy_name.replace(/_/g, ' ')}
              </h3>
            </div>
            <div className="text-right">
              <p className="text-sm opacity-90 mb-1">Confidence</p>
              <p className="text-4xl font-bold">
                {(prediction.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Confidence bar */}
          <div className="mt-4 bg-white bg-opacity-20 rounded-full h-3">
            <div
              className="bg-white rounded-full h-3 transition-all duration-500"
              style={{ width: `${prediction.confidence * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Top 3 Strategies */}
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Top 3 Strategy Probabilities
          </h4>
          <div className="space-y-3">
            {topStrategies.map(([name, prob], idx) => (
              <div key={name}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                    {idx + 1}. {name.replace(/_/g, ' ')}
                  </span>
                  <span className="text-sm font-mono font-semibold text-gray-900 dark:text-white">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`${getStrategyColor(name)} rounded-full h-2 transition-all duration-500`}
                    style={{ width: `${prob * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* All 8 Strategies (Compact) */}
        <details className="mb-6">
          <summary className="cursor-pointer text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            All 8 Strategy Probabilities
          </summary>
          <div className="grid grid-cols-2 gap-2 mt-3">
            {Object.entries(prediction.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, prob]) => (
                <div key={name} className="flex justify-between items-center text-xs">
                  <span className="text-gray-600 dark:text-gray-400 capitalize truncate">
                    {name.replace(/_/g, ' ')}
                  </span>
                  <span className="font-mono font-semibold text-gray-900 dark:text-white ml-2">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </details>

        {/* Model Metadata */}
        {prediction.model_metadata && (
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Model Architecture
            </h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {prediction.model_metadata.recursion_cycles}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Recursion Cycles
                </p>
              </div>
              <div>
                <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {prediction.model_metadata.latent_steps}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Latent Steps
                </p>
              </div>
              <div>
                <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {prediction.model_metadata.effective_depth}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Effective Depth
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="px-6 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        <div className="flex justify-between items-center text-sm">
          <div className="flex space-x-4">
            {prediction.halt_probability !== undefined && (
              <span className="text-gray-600 dark:text-gray-400">
                Halt Probability:{' '}
                <span className="font-mono text-gray-900 dark:text-white">
                  {(prediction.halt_probability * 100).toFixed(2)}%
                </span>
              </span>
            )}
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">Updates every 30s</span>
            {prediction.mock_data && (
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 rounded text-xs font-semibold">
                DEMO MODE
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TRMPredictionPanel;
