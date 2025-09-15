import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Import existing components
import { InequalityPanel } from '../InequalityPanel';
import { ContrarianTrades } from '../ContrarianTrades';

// Import new AI components
import { AIStatusPanel } from './AIStatusPanel';
import { DecisionTreeBuilder } from './DecisionTreeBuilder';
import { CalibrationTrainer } from './CalibrationTrainer';

// Icons
const DashboardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const InequalityIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TradeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
  </svg>
);

const AIIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const LearnIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

interface EnhancedDashboardProps {
  className?: string;
}

export const EnhancedDashboard: React.FC<EnhancedDashboardProps> = ({ className = "" }) => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'inequality' | 'trades' | 'ai' | 'learn'>('dashboard');
  const [inequalityData, setInequalityData] = useState<any>(null);
  const [contrarianData, setContrarianData] = useState<any>(null);
  const [aiConnected, setAiConnected] = useState(false);
  const [selectedOpportunity, setSelectedOpportunity] = useState<any>(null);

  useEffect(() => {
    // Fetch initial data
    fetchDashboardData();

    // Set up WebSocket connection for real-time updates
    setupWebSocketConnection();

    // Set up periodic data refresh
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch inequality data
      const inequalityResponse = await fetch('http://localhost:8000/api/inequality/data');
      const inequality = await inequalityResponse.json();
      setInequalityData(inequality);

      // Fetch contrarian opportunities
      const contrarianResponse = await fetch('http://localhost:8000/api/contrarian/opportunities');
      const contrarian = await contrarianResponse.json();
      setContrarianData(contrarian);

      setAiConnected(true);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setAiConnected(false);
    }
  };

  const setupWebSocketConnection = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/dashboard_${Date.now()}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setAiConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setAiConnected(false);
      // Attempt to reconnect after 5 seconds
      setTimeout(setupWebSocketConnection, 5000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setAiConnected(false);
    };
  };

  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'dashboard_update':
        if (message.inequality_panel) {
          setInequalityData(message.inequality_panel);
        }
        if (message.contrarian_trades) {
          setContrarianData(message.contrarian_trades);
        }
        break;

      case 'inequality_update':
        // Real-time inequality metric update
        if (inequalityData) {
          setInequalityData(prev => ({
            ...prev,
            metrics: {
              ...prev.metrics,
              [message.stream]: message.value
            }
          }));
        }
        break;

      case 'new_opportunities':
        // New trading opportunities detected
        console.log('New opportunities detected:', message.opportunities);
        fetchDashboardData(); // Refresh data
        break;

      case 'ai_signal_update':
        // Real-time AI signal updates
        console.log('AI signal update:', message.stream_name, message.data);
        break;

      default:
        console.log('Unknown WebSocket message type:', message.type);
    }
  };

  const handleExecuteTrade = async (opportunity: any) => {
    try {
      const response = await fetch(`/api/trade/execute/${opportunity.symbol}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const result = await response.json();

      if (result.success) {
        console.log('Trade executed successfully:', result);
        // Show success notification
        alert(`Trade executed: ${result.action} ${result.asset} at ${result.entry_price}`);
      } else {
        console.error('Trade execution failed:', result.error);
        alert(`Trade failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Failed to execute trade:', error);
      alert('Failed to execute trade - check connection');
    }
  };

  const tabs = [
    { id: 'dashboard', label: 'Overview', icon: DashboardIcon },
    { id: 'inequality', label: 'Inequality Hunter', icon: InequalityIcon },
    { id: 'trades', label: 'Contrarian Trades', icon: TradeIcon },
    { id: 'ai', label: 'AI Status', icon: AIIcon },
    { id: 'learn', label: 'Learn & Train', icon: LearnIcon }
  ];

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${className}`}>
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Gary×Taleb Trading System
              </h1>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${aiConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {aiConnected ? 'AI Connected' : 'AI Disconnected'}
                </span>
              </div>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Mathematical Framework • Self-Calibrating AI • Real-time Signals
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === id
                    ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                <Icon />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* Dashboard Overview */}
              <div className="space-y-6">
                <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6 rounded-lg">
                  <h2 className="text-2xl font-bold mb-2">AI-Enhanced Trading Dashboard</h2>
                  <p className="text-purple-100">
                    Find mispricings where consensus is wrong about inequality effects using mathematical rigor
                  </p>
                </div>

                {/* Quick Stats */}
                {inequalityData && (
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Gary Moment Score</div>
                      <div className="text-2xl font-bold text-purple-600">
                        {(inequalityData.metrics.consensusWrongScore * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600 dark:text-gray-400">AI Confidence</div>
                      <div className="text-2xl font-bold text-green-600">
                        {(inequalityData.metrics.ai_confidence_level * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Signal Strength</div>
                      <div className="text-2xl font-bold text-blue-600">
                        {(inequalityData.metrics.mathematical_signal_strength * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Prediction Accuracy</div>
                      <div className="text-2xl font-bold text-orange-600">
                        {(inequalityData.metrics.ai_prediction_accuracy * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                )}

                {/* Recent Activity */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                  <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                    Recent AI Activity
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-sm text-green-700 dark:text-green-300">
                        AI detected new mispricing in Treasury bonds (TLT) - Consensus wrong about safe haven demand
                      </span>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                      <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      <span className="text-sm text-blue-700 dark:text-blue-300">
                        DPI signal strength increased to 0.73 - Wealth concentration accelerating
                      </span>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                      <div className="w-2 h-2 bg-purple-500 rounded-full" />
                      <span className="text-sm text-purple-700 dark:text-purple-300">
                        AI calibration improved - Prediction accuracy now at 67.3%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'inequality' && (
            <motion.div
              key="inequality"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {inequalityData ? (
                <InequalityPanel
                  metrics={inequalityData.metrics}
                  historicalData={inequalityData.historicalData}
                  wealthFlows={inequalityData.wealthFlows}
                  contrarianSignals={inequalityData.contrarianSignals}
                />
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <div className="animate-pulse">
                    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded mb-4" />
                    <div className="space-y-3">
                      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded" />
                      <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'trades' && (
            <motion.div
              key="trades"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {contrarianData ? (
                <ContrarianTrades
                  opportunities={contrarianData.opportunities}
                  onExecuteTrade={handleExecuteTrade}
                />
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <div className="animate-pulse">
                    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded mb-4" />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[1, 2, 3, 4].map(i => (
                        <div key={i} className="h-32 bg-gray-200 dark:bg-gray-700 rounded" />
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'ai' && (
            <motion.div
              key="ai"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <AIStatusPanel />
            </motion.div>
          )}

          {activeTab === 'learn' && (
            <motion.div
              key="learn"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="bg-gradient-to-r from-green-600 to-teal-600 text-white p-6 rounded-lg">
                <h2 className="text-2xl font-bold mb-2">AI Training & Education</h2>
                <p className="text-green-100">
                  Improve your decision-making and help train the AI system
                </p>
              </div>

              {/* Training Modules */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <CalibrationTrainer />
                </div>
                <div className="space-y-6">
                  {selectedOpportunity ? (
                    <DecisionTreeBuilder
                      opportunity={selectedOpportunity}
                      onDecisionMade={(decision, confidence) => {
                        console.log('Decision made:', decision, confidence);
                        setSelectedOpportunity(null);
                      }}
                    />
                  ) : (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
                        Decision Tree Builder
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        Select a contrarian opportunity from the Trades tab to analyze it with Matt Freeman's decision tree methodology.
                      </p>
                      <button
                        onClick={() => setActiveTab('trades')}
                        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
                      >
                        Go to Trades
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EnhancedDashboard;