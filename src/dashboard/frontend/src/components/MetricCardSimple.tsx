import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  trend,
  className = ''
}) => {
  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-green-600';
      case 'down':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return '↗';
      case 'down':
        return '↘';
      default:
        return '';
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h3>
      <div className="flex items-center justify-between mt-2">
        <p className="text-2xl font-bold text-gray-900 dark:text-white">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </p>
        {trend && (
          <span className={`text-sm font-medium ${getTrendColor()}`}>
            {getTrendIcon()}
          </span>
        )}
      </div>
    </div>
  );
};

export const PRuinCard: React.FC<{ value: number; className?: string }> = ({ value, className }) => {
  const safeValue = value ?? 0;
  const getColor = () => {
    if (safeValue >= 0.2) return 'text-red-600';
    if (safeValue >= 0.1) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className ?? ''}`}>
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">P(ruin)</h3>
      <p className={`text-2xl font-bold mt-2 ${getColor()}`}>
        {(safeValue * 100).toFixed(1)}%
      </p>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
        Probability of ruin
      </p>
    </div>
  );
};

export const PortfolioValueCard: React.FC<{ value: number; className?: string }> = ({ value, className }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className ?? ''}`}>
    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Portfolio Value</h3>
    <p className="text-2xl font-bold text-green-600 mt-2">
      ${(value ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
    </p>
    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
      Total market value
    </p>
  </div>
);

export const VarCard: React.FC<{ value: number; className?: string }> = ({ value, className }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className ?? ''}`}>
    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">VaR 95%</h3>
    <p className="text-2xl font-bold text-red-600 mt-2">
      ${(value ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
    </p>
    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
      Value at Risk (95% confidence)
    </p>
  </div>
);

export const SharpeRatioCard: React.FC<{ value: number; className?: string }> = ({ value, className }) => {
  const safeValue = value ?? 0;
  const getColor = () => {
    if (safeValue > 1.5) return 'text-green-600';
    if (safeValue < 0.5) return 'text-red-600';
    return 'text-yellow-600';
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className ?? ''}`}>
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Sharpe Ratio</h3>
      <p className={`text-2xl font-bold mt-2 ${getColor()}`}>
        {safeValue.toFixed(2)}
      </p>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
        Risk-adjusted return
      </p>
    </div>
  );
};

export const DrawdownCard: React.FC<{ value: number; className?: string }> = ({ value, className }) => {
  const safeValue = value ?? 0;
  const getColor = () => {
    if (safeValue >= 0.2) return 'text-red-600';
    if (safeValue >= 0.1) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className ?? ''}`}>
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Max Drawdown</h3>
      <p className={`text-2xl font-bold mt-2 ${getColor()}`}>
        {(safeValue * 100).toFixed(1)}%
      </p>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
        Peak-to-trough decline
      </p>
    </div>
  );
};