import React from 'react';

// Simple stub components for testing
export const RiskChart: React.FC = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Risk Chart
    </h3>
    <div className="h-64 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
      <p className="text-gray-500 dark:text-gray-400">Chart visualization would go here</p>
    </div>
  </div>
);

export const InequalityPanel: React.FC = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Inequality Analysis
    </h3>
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Gini Coefficient</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">0.473</p>
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Top 1% Wealth</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">32.1%</p>
        </div>
      </div>
      <p className="text-gray-600 dark:text-gray-400 text-sm">
        Advanced inequality analysis using Gary's DPI methodology.
      </p>
    </div>
  </div>
);

export const ContrarianTrades: React.FC = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Contrarian Opportunities
    </h3>
    <div className="space-y-3">
      <div className="border border-gray-200 dark:border-gray-700 rounded p-3">
        <div className="flex justify-between items-start">
          <div>
            <p className="font-medium text-gray-900 dark:text-white">SPY</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">Market inefficiency detected</p>
          </div>
          <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
            BUY
          </span>
        </div>
        <div className="mt-2">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-green-600 h-2 rounded-full" style={{width: '78%'}}></div>
          </div>
          <p className="text-xs text-gray-500 mt-1">Conviction: 78%</p>
        </div>
      </div>
      <div className="border border-gray-200 dark:border-gray-700 rounded p-3">
        <div className="flex justify-between items-start">
          <div>
            <p className="font-medium text-gray-900 dark:text-white">TLT</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">Consensus narrative gap</p>
          </div>
          <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs">
            WATCH
          </span>
        </div>
        <div className="mt-2">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-yellow-600 h-2 rounded-full" style={{width: '65%'}}></div>
          </div>
          <p className="text-xs text-gray-500 mt-1">Conviction: 65%</p>
        </div>
      </div>
    </div>
  </div>
);

export const AIStatusPanel: React.FC = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
      AI Status & Calibration
    </h3>
    <div className="grid grid-cols-2 gap-4">
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Brier Score</p>
        <p className="text-2xl font-bold text-green-600">0.187</p>
      </div>
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Prediction Accuracy</p>
        <p className="text-2xl font-bold text-blue-600">74.3%</p>
      </div>
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Kelly Safety Factor</p>
        <p className="text-2xl font-bold text-yellow-600">0.25</p>
      </div>
      <div>
        <p className="text-sm text-gray-500 dark:text-gray-400">PIT Uniformity</p>
        <p className="text-2xl font-bold text-purple-600">0.923</p>
      </div>
    </div>
    <div className="mt-4">
      <p className="text-gray-600 dark:text-gray-400 text-sm">
        AI calibration metrics and mathematical framework status.
      </p>
    </div>
  </div>
);

export const EducationHub: React.FC = () => (
  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
      Guild of the Rose Education
    </h3>
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border border-gray-200 dark:border-gray-700 rounded p-4">
          <h4 className="font-medium text-gray-900 dark:text-white">Decision Theory</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Master decision-making under uncertainty
          </p>
          <div className="mt-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-blue-600 h-2 rounded-full" style={{width: '85%'}}></div>
            </div>
            <p className="text-xs text-gray-500 mt-1">Progress: 85%</p>
          </div>
        </div>
        <div className="border border-gray-200 dark:border-gray-700 rounded p-4">
          <h4 className="font-medium text-gray-900 dark:text-white">Antifragility</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Build portfolios that benefit from volatility
          </p>
          <div className="mt-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div className="bg-green-600 h-2 rounded-full" style={{width: '60%'}}></div>
            </div>
            <p className="text-xs text-gray-500 mt-1">Progress: 60%</p>
          </div>
        </div>
      </div>
      <p className="text-gray-600 dark:text-gray-400 text-sm">
        Interactive learning modules with Matt Freeman's rational decision theory.
      </p>
    </div>
  </div>
);