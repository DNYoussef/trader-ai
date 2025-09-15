import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Icons
const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const ChartIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const TargetIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const ActivityIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

interface AIStatusData {
  utility_parameters: {
    risk_aversion: number;
    loss_aversion: number;
    kelly_safety_factor: number;
    confidence_threshold: number;
    last_updated: string;
  };
  calibration_metrics: {
    total_predictions: number;
    resolved_predictions: number;
    overall_accuracy: number;
    brier_score: number;
    log_loss: number;
    calibration_error: number;
    pit_p_value: number;
    confidence_bins?: Record<string, any>;
  };
  mathematical_framework: {
    dpi_active: boolean;
    narrative_gap_tracking: boolean;
    repricing_potential_calculated: boolean;
    kelly_optimization: boolean;
    evt_risk_management: boolean;
    barbell_constraints: boolean;
  };
  streaming_status: {
    ai_processing: boolean;
    mispricing_detection: boolean;
    websocket_connections: number;
    last_update: string;
  };
}

export const AIStatusPanel: React.FC = () => {
  const [aiStatus, setAiStatus] = useState<AIStatusData | null>(null);
  const [activeTab, setActiveTab] = useState<'calibration' | 'framework' | 'performance'>('calibration');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Fetch initial AI status
    fetchAIStatus();

    // Set up periodic updates
    const interval = setInterval(fetchAIStatus, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchAIStatus = async () => {
    try {
      const response = await fetch('/api/ai/status');
      const data = await response.json();
      setAiStatus(data);
      setIsConnected(true);
    } catch (error) {
      console.error('Failed to fetch AI status:', error);
      setIsConnected(false);
    }
  };

  if (!aiStatus) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600" />
          <span className="ml-3 text-gray-600 dark:text-gray-400">Loading AI Status...</span>
        </div>
      </div>
    );
  }

  const getStatusColor = (active: boolean) => active ? 'text-green-500' : 'text-red-500';
  const getCalibrationColor = (error: number) => {
    if (error < 0.1) return 'text-green-500';
    if (error < 0.2) return 'text-yellow-500';
    return 'text-red-500';
  };

  const calibrationData = [
    { name: 'Accuracy', value: aiStatus.calibration_metrics.overall_accuracy * 100, color: '#10b981' },
    { name: 'Brier Score', value: (1 - aiStatus.calibration_metrics.brier_score) * 100, color: '#3b82f6' },
    { name: 'Calibration', value: (1 - aiStatus.calibration_metrics.calibration_error) * 100, color: '#8b5cf6' }
  ];

  const frameworkStatus = [
    { name: 'DPI Signals', active: aiStatus.mathematical_framework.dpi_active, description: 'Distributional Pressure Index' },
    { name: 'Narrative Gap', active: aiStatus.mathematical_framework.narrative_gap_tracking, description: 'AI vs Market Expectations' },
    { name: 'Repricing Potential', active: aiStatus.mathematical_framework.repricing_potential_calculated, description: 'Catalyst-Weighted Opportunities' },
    { name: 'Kelly Optimization', active: aiStatus.mathematical_framework.kelly_optimization, description: 'AI-Calibrated Position Sizing' },
    { name: 'EVT Risk Management', active: aiStatus.mathematical_framework.evt_risk_management, description: 'Extreme Value Theory Tails' },
    { name: 'Barbell Constraints', active: aiStatus.mathematical_framework.barbell_constraints, description: '80/20 Allocation Management' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BrainIcon />
            <div>
              <h2 className="text-2xl font-bold">AI Trading System Status</h2>
              <p className="text-purple-100">Mathematical Framework & Self-Calibration Monitor</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center gap-2 mb-2">
            <TargetIcon />
            <span className="text-sm text-gray-600 dark:text-gray-400">AI Accuracy</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {(aiStatus.calibration_metrics.overall_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">
            {aiStatus.calibration_metrics.resolved_predictions} predictions
          </div>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center gap-2 mb-2">
            <ChartIcon />
            <span className="text-sm text-gray-600 dark:text-gray-400">Calibration Error</span>
          </div>
          <div className={`text-2xl font-bold ${getCalibrationColor(aiStatus.calibration_metrics.calibration_error)}`}>
            {(aiStatus.calibration_metrics.calibration_error * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">
            Lower is better
          </div>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center gap-2 mb-2">
            <ActivityIcon />
            <span className="text-sm text-gray-600 dark:text-gray-400">Risk Aversion</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {aiStatus.utility_parameters.risk_aversion.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500">
            AI-learned parameter
          </div>
        </motion.div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow"
        >
          <div className="flex items-center gap-2 mb-2">
            <BrainIcon />
            <span className="text-sm text-gray-600 dark:text-gray-400">Kelly Factor</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {(aiStatus.utility_parameters.kelly_safety_factor * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500">
            Position sizing safety
          </div>
        </motion.div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex">
          {[
            { id: 'calibration', label: 'AI Calibration', icon: TargetIcon },
            { id: 'framework', label: 'Mathematical Framework', icon: ChartIcon },
            { id: 'performance', label: 'Live Performance', icon: ActivityIcon }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === id
                  ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              <Icon />
              {label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'calibration' && (
          <motion.div
            key="calibration"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Calibration Metrics Chart */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                AI Calibration Performance
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="80%" data={calibrationData}>
                  <RadialBar
                    dataKey="value"
                    cornerRadius={10}
                    fill={(entry) => entry.color}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(17, 24, 39, 0.9)',
                      border: 'none',
                      borderRadius: '8px',
                      color: 'white'
                    }}
                    formatter={(value) => [`${value.toFixed(1)}%`, 'Score']}
                  />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  Prediction Statistics
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Total Predictions:</span>
                    <span className="font-semibold">{aiStatus.calibration_metrics.total_predictions}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Resolved:</span>
                    <span className="font-semibold">{aiStatus.calibration_metrics.resolved_predictions}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Brier Score:</span>
                    <span className="font-semibold">{aiStatus.calibration_metrics.brier_score.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Log Loss:</span>
                    <span className="font-semibold">{aiStatus.calibration_metrics.log_loss.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">PIT P-Value:</span>
                    <span className={`font-semibold ${aiStatus.calibration_metrics.pit_p_value > 0.05 ? 'text-green-500' : 'text-red-500'}`}>
                      {aiStatus.calibration_metrics.pit_p_value.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
                  AI Utility Parameters
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-400">Risk Aversion (γ):</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-purple-500 h-2 rounded-full"
                          style={{ width: `${(aiStatus.utility_parameters.risk_aversion / 2) * 100}%` }}
                        />
                      </div>
                      <span className="font-semibold text-sm">{aiStatus.utility_parameters.risk_aversion.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-400">Loss Aversion:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${(aiStatus.utility_parameters.loss_aversion / 4) * 100}%` }}
                        />
                      </div>
                      <span className="font-semibold text-sm">{aiStatus.utility_parameters.loss_aversion.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-400">Kelly Safety:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${(aiStatus.utility_parameters.kelly_safety_factor / 0.5) * 100}%` }}
                        />
                      </div>
                      <span className="font-semibold text-sm">{(aiStatus.utility_parameters.kelly_safety_factor * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-400">Confidence Threshold:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${aiStatus.utility_parameters.confidence_threshold * 100}%` }}
                        />
                      </div>
                      <span className="font-semibold text-sm">{(aiStatus.utility_parameters.confidence_threshold * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'framework' && (
          <motion.div
            key="framework"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                Mathematical Framework Status
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {frameworkStatus.map((item, index) => (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
                  >
                    <div>
                      <div className="font-semibold text-gray-900 dark:text-gray-100">
                        {item.name}
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {item.description}
                      </div>
                    </div>
                    <div className={`flex items-center gap-2 ${getStatusColor(item.active)}`}>
                      <div className={`w-3 h-3 rounded-full ${item.active ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className="text-sm font-semibold">
                        {item.active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Mathematical Formulas */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                Active Mathematical Formulas
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="font-mono text-sm text-purple-800 dark:text-purple-200 mb-2">
                    DPI_t = Σ(ω_g^AI × ΔNetCashFlow_g)
                  </div>
                  <div className="text-xs text-purple-600 dark:text-purple-400">
                    Distributional Pressure Index with AI-learned cohort weights
                  </div>
                </div>
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="font-mono text-sm text-blue-800 dark:text-blue-200 mb-2">
                    NG_t^(i) = E^AI[Path_i] - E^market[Path_i]
                  </div>
                  <div className="text-xs text-blue-600 dark:text-blue-400">
                    Narrative Gap between AI model and market expectations
                  </div>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="font-mono text-sm text-green-800 dark:text-green-200 mb-2">
                    f* = μ/σ², f = k × f* (k ∈ [0.2, 0.5])
                  </div>
                  <div className="text-xs text-green-600 dark:text-green-400">
                    Kelly fraction with AI-calibrated safety factor
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'performance' && (
          <motion.div
            key="performance"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Streaming Status */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                Live System Performance
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className={`text-2xl font-bold ${getStatusColor(aiStatus.streaming_status.ai_processing)}`}>
                    {aiStatus.streaming_status.ai_processing ? 'ACTIVE' : 'INACTIVE'}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">AI Processing</div>
                </div>
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className={`text-2xl font-bold ${getStatusColor(aiStatus.streaming_status.mispricing_detection)}`}>
                    {aiStatus.streaming_status.mispricing_detection ? 'SCANNING' : 'PAUSED'}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Mispricing Detection</div>
                </div>
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {aiStatus.streaming_status.websocket_connections}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">WebSocket Connections</div>
                </div>
              </div>
            </div>

            {/* Real-time Updates */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Last Update
              </h4>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {new Date(aiStatus.streaming_status.last_update).toLocaleString()}
              </div>
              <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
                <div className="text-sm text-green-700 dark:text-green-300">
                  ✅ AI system is actively monitoring market conditions and updating mathematical signals in real-time.
                  All Gary×Taleb frameworks are operational with continuous calibration.
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AIStatusPanel;