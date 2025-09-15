import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import numeral from 'numeral';
import { MetricCardProps } from '@/types';

// Icon components for different metric types
const TrendUpIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TrendDownIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
  </svg>
);

const AlertIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z" />
  </svg>
);

const formatValue = (value: number, format: string, decimalPlaces: number = 2): string => {
  switch (format) {
    case 'currency':
      return numeral(value).format(`$0,0.${'0'.repeat(decimalPlaces)}`);
    case 'percentage':
      return numeral(value).format(`0.${'0'.repeat(decimalPlaces)}%`);
    case 'ratio':
      return numeral(value).format(`0.${'0'.repeat(decimalPlaces)}`);
    case 'number':
      return numeral(value).format(`0,0.${'0'.repeat(decimalPlaces)}`);
    default:
      return value.toString();
  }
};

const getAlertLevelStyles = (alertLevel?: string) => {
  switch (alertLevel) {
    case 'danger':
      return {
        container: 'border-danger-200 bg-danger-50 dark:border-danger-800 dark:bg-danger-900/20',
        title: 'text-danger-700 dark:text-danger-300',
        value: 'text-danger-900 dark:text-danger-100',
        icon: 'text-danger-500',
      };
    case 'warning':
      return {
        container: 'border-warning-200 bg-warning-50 dark:border-warning-800 dark:bg-warning-900/20',
        title: 'text-warning-700 dark:text-warning-300',
        value: 'text-warning-900 dark:text-warning-100',
        icon: 'text-warning-500',
      };
    default:
      return {
        container: 'border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800',
        title: 'text-gray-600 dark:text-gray-300',
        value: 'text-gray-900 dark:text-gray-100',
        icon: 'text-gray-400 dark:text-gray-500',
      };
  }
};

const getTrendIcon = (trend?: string) => {
  switch (trend) {
    case 'up':
      return <TrendUpIcon />;
    case 'down':
      return <TrendDownIcon />;
    default:
      return null;
  }
};

const getTrendStyles = (trend?: string) => {
  switch (trend) {
    case 'up':
      return 'text-success-500';
    case 'down':
      return 'text-danger-500';
    default:
      return 'text-gray-400';
  }
};

export const MetricCard: React.FC<MetricCardProps> = ({
  metric,
  className,
  onClick,
}) => {
  const alertStyles = getAlertLevelStyles(metric.alert_level);
  const trendIcon = getTrendIcon(metric.trend);
  const trendStyles = getTrendStyles(metric.trend);

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3 }
    },
    hover: {
      scale: 1.02,
      transition: { duration: 0.2 }
    },
    tap: {
      scale: 0.98,
      transition: { duration: 0.1 }
    }
  };

  const valueVariants = {
    initial: { scale: 1 },
    animate: {
      scale: [1, 1.05, 1],
      transition: { duration: 0.5 }
    }
  };

  return (
    <motion.div
      variants={cardVariants}
      initial="initial"
      animate="animate"
      whileHover={onClick ? "hover" : undefined}
      whileTap={onClick ? "tap" : undefined}
      className={clsx(
        'relative p-4 rounded-lg border shadow-sm transition-all duration-200',
        alertStyles.container,
        onClick && 'cursor-pointer hover:shadow-md',
        className
      )}
      onClick={onClick}
    >
      {/* Alert indicator */}
      {metric.alert_level && metric.alert_level !== 'normal' && (
        <div className={clsx(
          'absolute top-2 right-2 p-1 rounded-full',
          alertStyles.icon
        )}>
          <AlertIcon />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className={clsx(
          'text-sm font-medium',
          alertStyles.title
        )}>
          {metric.title}
        </h3>

        {/* Trend indicator */}
        {trendIcon && (
          <div className={clsx('flex items-center', trendStyles)}>
            {trendIcon}
          </div>
        )}
      </div>

      {/* Value */}
      <motion.div
        variants={valueVariants}
        animate="animate"
        key={metric.value} // Re-animate when value changes
        className={clsx(
          'text-2xl font-bold mb-1',
          alertStyles.value
        )}
      >
        {formatValue(metric.value, metric.format)}
      </motion.div>

      {/* Description */}
      {metric.description && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {metric.description}
        </p>
      )}

      {/* Loading state overlay */}
      {metric.value === undefined && (
        <div className="absolute inset-0 bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
        </div>
      )}
    </motion.div>
  );
};

// Specialized metric cards for common use cases
export const PRuinCard: React.FC<{ pRuin: number; className?: string }> = ({ pRuin, className }) => {
  const getAlertLevel = (value: number) => {
    if (value >= 0.2) return 'danger';
    if (value >= 0.1) return 'warning';
    return 'normal';
  };

  return (
    <MetricCard
      metric={{
        title: 'P(ruin)',
        value: pRuin,
        format: 'percentage',
        alert_level: getAlertLevel(pRuin),
        description: 'Probability of ruin based on Gary\'s DPI methodology',
      }}
      className={className}
    />
  );
};

export const PortfolioValueCard: React.FC<{
  value: number;
  previousValue?: number;
  className?: string;
}> = ({ value, previousValue, className }) => {
  const trend = previousValue ? (
    value > previousValue ? 'up' : value < previousValue ? 'down' : 'neutral'
  ) : undefined;

  return (
    <MetricCard
      metric={{
        title: 'Portfolio Value',
        value,
        format: 'currency',
        trend,
        description: 'Total portfolio market value',
      }}
      className={className}
    />
  );
};

export const VarCard: React.FC<{ var95: number; portfolioValue: number; className?: string }> = ({
  var95,
  portfolioValue,
  className
}) => {
  const varRatio = portfolioValue > 0 ? var95 / portfolioValue : 0;

  const getAlertLevel = (ratio: number) => {
    if (ratio >= 0.1) return 'danger';
    if (ratio >= 0.05) return 'warning';
    return 'normal';
  };

  return (
    <MetricCard
      metric={{
        title: 'VaR (95%)',
        value: var95,
        format: 'currency',
        alert_level: getAlertLevel(varRatio),
        description: `${(varRatio * 100).toFixed(1)}% of portfolio value`,
      }}
      className={className}
    />
  );
};

export const SharpeRatioCard: React.FC<{ sharpe: number; className?: string }> = ({
  sharpe,
  className
}) => {
  const getTrend = (value: number) => {
    if (value > 1.5) return 'up';
    if (value < 0.5) return 'down';
    return 'neutral';
  };

  return (
    <MetricCard
      metric={{
        title: 'Sharpe Ratio',
        value: sharpe,
        format: 'ratio',
        trend: getTrend(sharpe),
        description: 'Risk-adjusted return measure',
      }}
      className={className}
    />
  );
};

export const DrawdownCard: React.FC<{ drawdown: number; className?: string }> = ({
  drawdown,
  className
}) => {
  const getAlertLevel = (value: number) => {
    if (value >= 0.2) return 'danger';
    if (value >= 0.1) return 'warning';
    return 'normal';
  };

  return (
    <MetricCard
      metric={{
        title: 'Max Drawdown',
        value: drawdown,
        format: 'percentage',
        alert_level: getAlertLevel(drawdown),
        trend: 'down', // Drawdown is always negative trend
        description: 'Peak-to-trough decline',
      }}
      className={className}
    />
  );
};

// Grid container for metric cards
export const MetricGrid: React.FC<{
  children: React.ReactNode;
  className?: string;
  columns?: 1 | 2 | 3 | 4;
}> = ({ children, className, columns = 4 }) => {
  const gridCols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={clsx(
      'grid gap-4',
      gridCols[columns],
      className
    )}>
      {children}
    </div>
  );
};