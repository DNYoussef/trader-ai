import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface DecisionOption {
  name: string;
  description: string;
  position_size: number;
  entry_cost: number;
  risk_level: string;
  time_horizon_days: number;
}

interface Outcome {
  name: string;
  description: string;
  probability: number;
  expected_return: number;
  worst_case: number;
  best_case: number;
  utility_score: number;
}

interface CriticalVariable {
  name: string;
  description: string;
  impact_weight: number;
  uncertainty_level: number;
  current_state: Record<string, number>;
}

interface DecisionTreeData {
  root_decision: string;
  recommendation: string;
  expected_utility: number;
  confidence_interval: [number, number];
  options: DecisionOption[];
  outcomes: Outcome[];
  critical_variables: CriticalVariable[];
}

interface DecisionTreeBuilderProps {
  opportunity: any;
  onDecisionMade: (decision: string, confidence: number) => void;
  className?: string;
}

const TreeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M8 16l-4-4 4-4m8 8l4-4-4-4" />
  </svg>
);

const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const TargetIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

export const DecisionTreeBuilder: React.FC<DecisionTreeBuilderProps> = ({
  opportunity,
  onDecisionMade,
  className = ""
}) => {
  const [activeTab, setActiveTab] = useState<'tree' | 'variables' | 'outcomes'>('tree');
  const [selectedOption, setSelectedOption] = useState<string>('');
  const [userConfidence, setUserConfidence] = useState<number>(0.7);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Mock decision tree data - in real implementation, this would come from the backend
  const decisionTreeData: DecisionTreeData = useMemo(() => ({
    root_decision: `Trade: ${opportunity?.topic || 'Unknown Opportunity'}`,
    recommendation: 'Standard Position',
    expected_utility: 0.156,
    confidence_interval: [0.098, 0.214],
    options: [
      {
        name: 'No Trade',
        description: 'Do not enter this position',
        position_size: 0,
        entry_cost: 0,
        risk_level: 'none',
        time_horizon_days: 0
      },
      {
        name: 'Conservative Position',
        description: 'Small position with tight stops',
        position_size: 200,
        entry_cost: 200,
        risk_level: 'low',
        time_horizon_days: 30
      },
      {
        name: 'Standard Position',
        description: 'Normal position size based on conviction',
        position_size: 1000,
        entry_cost: 1000,
        risk_level: 'medium',
        time_horizon_days: 90
      },
      {
        name: 'Gary Moment',
        description: 'Large conviction bet when consensus is extremely wrong',
        position_size: 2000,
        entry_cost: 2000,
        risk_level: 'high',
        time_horizon_days: 180
      }
    ],
    outcomes: [
      {
        name: 'Bull Case',
        description: 'Gary thesis proves extremely correct',
        probability: 0.25,
        expected_return: 0.45,
        worst_case: 0.15,
        best_case: 0.75,
        utility_score: 0.289
      },
      {
        name: 'Base Case',
        description: 'Thesis plays out as expected',
        probability: 0.50,
        expected_return: 0.15,
        worst_case: 0.05,
        best_case: 0.30,
        utility_score: 0.167
      },
      {
        name: 'Bear Case',
        description: 'Consensus proves partially correct',
        probability: 0.20,
        expected_return: -0.075,
        worst_case: -0.15,
        best_case: 0.0,
        utility_score: -0.034
      },
      {
        name: 'Black Swan',
        description: 'Extreme negative outcome',
        probability: 0.05,
        expected_return: -0.30,
        worst_case: -0.60,
        best_case: -0.15,
        utility_score: -0.112
      }
    ],
    critical_variables: [
      {
        name: 'Inequality Trend',
        description: 'Direction and speed of wealth concentration',
        impact_weight: 0.4,
        uncertainty_level: 0.3,
        current_state: { accelerating: 0.4, stable: 0.4, reversing: 0.2 }
      },
      {
        name: 'Consensus Strength',
        description: 'How strongly held is the consensus view',
        impact_weight: 0.3,
        uncertainty_level: 0.2,
        current_state: { very_strong: 0.3, moderate: 0.5, weak: 0.2 }
      },
      {
        name: 'Policy Response',
        description: 'Government/central bank reaction to inequality',
        impact_weight: 0.2,
        uncertainty_level: 0.5,
        current_state: { pro_inequality: 0.6, neutral: 0.3, anti_inequality: 0.1 }
      }
    ]
  }), [opportunity]);

  const handleExecuteDecision = () => {
    if (!selectedOption) return;

    setIsAnalyzing(true);

    // Simulate analysis time
    setTimeout(() => {
      setIsAnalyzing(false);
      onDecisionMade(selectedOption, userConfidence);
    }, 2000);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'none': return 'text-gray-500 bg-gray-100';
      case 'low': return 'text-green-700 bg-green-100';
      case 'medium': return 'text-yellow-700 bg-yellow-100';
      case 'high': return 'text-red-700 bg-red-100';
      default: return 'text-gray-500 bg-gray-100';
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatCurrency = (value: number) => `$${value.toFixed(0)}`;

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
        <div className="flex items-center gap-3 mb-2">
          <BrainIcon />
          <h3 className="text-xl font-bold">Matt Freeman Decision Tree</h3>
        </div>
        <p className="text-blue-100 text-sm">
          Guild of the Rose 5-Step Rational Decision Process
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex">
          {[
            { id: 'tree', label: 'Decision Tree', icon: TreeIcon },
            { id: 'variables', label: 'Critical Variables', icon: TargetIcon },
            { id: 'outcomes', label: 'Scenarios', icon: BrainIcon }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
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
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'tree' && (
            <motion.div
              key="tree"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Decision Root */}
              <div className="text-center">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {decisionTreeData.root_decision}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  AI Recommendation: <span className="font-semibold text-purple-600">
                    {decisionTreeData.recommendation}
                  </span>
                </p>
              </div>

              {/* Options Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {decisionTreeData.options.map((option, index) => (
                  <motion.div
                    key={option.name}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                      selectedOption === option.name
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                    } ${option.name === decisionTreeData.recommendation
                        ? 'ring-2 ring-purple-500 ring-opacity-50' : ''}`}
                    onClick={() => setSelectedOption(option.name)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h5 className="font-semibold text-gray-900 dark:text-gray-100">
                        {option.name}
                      </h5>
                      <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(option.risk_level)}`}>
                        {option.risk_level}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {option.description}
                    </p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">Position:</span>
                        <div className="font-semibold">{formatCurrency(option.position_size)}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Timeline:</span>
                        <div className="font-semibold">{option.time_horizon_days} days</div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Confidence Slider */}
              {selectedOption && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
                >
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Your Confidence Level: {formatPercent(userConfidence)}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={userConfidence}
                    onChange={(e) => setUserConfidence(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low (10%)</span>
                    <span>Medium (50%)</span>
                    <span>High (100%)</span>
                  </div>
                </motion.div>
              )}

              {/* Execute Button */}
              {selectedOption && (
                <motion.button
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  onClick={handleExecuteDecision}
                  disabled={isAnalyzing}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Analyzing Decision...
                    </div>
                  ) : (
                    `Execute: ${selectedOption}`
                  )}
                </motion.button>
              )}
            </motion.div>
          )}

          {activeTab === 'variables' && (
            <motion.div
              key="variables"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                Critical Variables Analysis
              </h4>
              {decisionTreeData.critical_variables.map((variable, index) => (
                <motion.div
                  key={variable.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border border-gray-200 dark:border-gray-600 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h5 className="font-semibold text-gray-900 dark:text-gray-100">
                      {variable.name}
                    </h5>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">Impact:</span>
                      <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${variable.impact_weight * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {variable.description}
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">Uncertainty Level:</span>
                      <span className={`font-semibold ${
                        variable.uncertainty_level > 0.4 ? 'text-red-500' :
                        variable.uncertainty_level > 0.2 ? 'text-yellow-500' : 'text-green-500'
                      }`}>
                        {formatPercent(variable.uncertainty_level)}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {Object.entries(variable.current_state).map(([state, probability]) => (
                        <div key={state} className="text-center">
                          <div className="text-xs text-gray-500 capitalize">{state.replace('_', ' ')}</div>
                          <div className="font-semibold text-sm">{formatPercent(probability)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}

          {activeTab === 'outcomes' && (
            <motion.div
              key="outcomes"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                Probability-Weighted Scenarios
              </h4>
              {decisionTreeData.outcomes.map((outcome, index) => (
                <motion.div
                  key={outcome.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border border-gray-200 dark:border-gray-600 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h5 className={`font-semibold ${
                      outcome.name === 'Bull Case' ? 'text-green-600' :
                      outcome.name === 'Base Case' ? 'text-blue-600' :
                      outcome.name === 'Bear Case' ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {outcome.name}
                    </h5>
                    <span className="text-sm font-semibold text-gray-600 dark:text-gray-400">
                      {formatPercent(outcome.probability)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {outcome.description}
                  </p>
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div className="text-center">
                      <div className="text-gray-500">Worst</div>
                      <div className={`font-semibold ${outcome.worst_case < 0 ? 'text-red-500' : 'text-gray-900 dark:text-gray-100'}`}>
                        {formatPercent(outcome.worst_case)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-500">Expected</div>
                      <div className={`font-semibold ${outcome.expected_return < 0 ? 'text-red-500' : 'text-green-500'}`}>
                        {formatPercent(outcome.expected_return)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-500">Best</div>
                      <div className="font-semibold text-green-500">
                        {formatPercent(outcome.best_case)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-gray-500">Utility</div>
                      <div className={`font-semibold ${outcome.utility_score < 0 ? 'text-red-500' : 'text-green-500'}`}>
                        {outcome.utility_score.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default DecisionTreeBuilder;