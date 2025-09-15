import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { formatDistanceToNow } from 'date-fns';
import { AlertListProps, AlertEvent } from '@/types';

// Alert severity icons
const AlertIcons = {
  critical: () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
  ),
  high: () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
    </svg>
  ),
  medium: () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
  ),
  low: () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
  ),
};

// Get severity styles
const getSeverityStyles = (severity: AlertEvent['severity']) => {
  switch (severity) {
    case 'critical':
      return {
        container: 'border-l-danger-500 bg-danger-50 dark:bg-danger-900/20',
        icon: 'text-danger-500',
        title: 'text-danger-900 dark:text-danger-100',
        badge: 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200',
      };
    case 'high':
      return {
        container: 'border-l-warning-500 bg-warning-50 dark:bg-warning-900/20',
        icon: 'text-warning-500',
        title: 'text-warning-900 dark:text-warning-100',
        badge: 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200',
      };
    case 'medium':
      return {
        container: 'border-l-primary-500 bg-primary-50 dark:bg-primary-900/20',
        icon: 'text-primary-500',
        title: 'text-primary-900 dark:text-primary-100',
        badge: 'bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200',
      };
    case 'low':
      return {
        container: 'border-l-gray-500 bg-gray-50 dark:bg-gray-800',
        icon: 'text-gray-500',
        title: 'text-gray-900 dark:text-gray-100',
        badge: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
      };
    default:
      return {
        container: 'border-l-gray-500 bg-gray-50 dark:bg-gray-800',
        icon: 'text-gray-500',
        title: 'text-gray-900 dark:text-gray-100',
        badge: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
      };
  }
};

// Individual alert item component
const AlertItem: React.FC<{
  alert: AlertEvent;
  onAcknowledge: (alertId: string) => void;
  index: number;
}> = ({ alert, onAcknowledge, index }) => {
  const [isAcknowledging, setIsAcknowledging] = useState(false);
  const styles = getSeverityStyles(alert.severity);
  const IconComponent = AlertIcons[alert.severity];

  const handleAcknowledge = async () => {
    setIsAcknowledging(true);
    try {
      await onAcknowledge(alert.alert_id);
    } finally {
      setIsAcknowledging(false);
    }
  };

  const itemVariants = {
    initial: { opacity: 0, x: -20, scale: 0.95 },
    animate: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: {
        duration: 0.3,
        delay: index * 0.05,
      },
    },
    exit: {
      opacity: 0,
      x: 20,
      scale: 0.95,
      transition: { duration: 0.2 },
    },
  };

  return (
    <motion.div
      variants={itemVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      layout
      className={clsx(
        'border-l-4 p-4 rounded-r-lg shadow-sm transition-all duration-200',
        styles.container,
        alert.acknowledged && 'opacity-60'
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          {/* Icon */}
          <div className={clsx('flex-shrink-0 mt-0.5', styles.icon)}>
            <IconComponent />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <span className={clsx(
                'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
                styles.badge
              )}>
                {alert.severity.toUpperCase()}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {alert.metric_name}
              </span>
            </div>

            <p className={clsx('text-sm font-medium mb-1', styles.title)}>
              {alert.message}
            </p>

            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>
                {formatDistanceToNow(new Date(alert.timestamp * 1000), { addSuffix: true })}
              </span>
              <span>
                Threshold: {typeof alert.threshold_value === 'number'
                  ? alert.threshold_value.toFixed(2)
                  : alert.threshold_value}
              </span>
            </div>
          </div>
        </div>

        {/* Actions */}
        {!alert.acknowledged && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleAcknowledge}
            disabled={isAcknowledging}
            className={clsx(
              'flex-shrink-0 ml-4 px-3 py-1 text-xs font-medium rounded-full transition-colors',
              'bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600',
              'hover:bg-gray-50 dark:hover:bg-gray-600',
              'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              styles.title
            )}
          >
            {isAcknowledging ? (
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                <span>...</span>
              </div>
            ) : (
              'Acknowledge'
            )}
          </motion.button>
        )}

        {alert.acknowledged && (
          <div className="flex-shrink-0 ml-4 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
              </svg>
              <span>Acknowledged</span>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Filter controls component
const AlertFilters: React.FC<{
  severityFilter: string;
  onSeverityChange: (severity: string) => void;
  acknowledgedFilter: string;
  onAcknowledgedChange: (status: string) => void;
  alertCounts: Record<string, number>;
}> = ({
  severityFilter,
  onSeverityChange,
  acknowledgedFilter,
  onAcknowledgedChange,
  alertCounts,
}) => {
  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      {/* Severity filter */}
      <div className="flex items-center space-x-2">
        <span className="text-sm text-gray-500 dark:text-gray-400">Severity:</span>
        <select
          value={severityFilter}
          onChange={(e) => onSeverityChange(e.target.value)}
          className="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          <option value="all">All ({alertCounts.total || 0})</option>
          <option value="critical">Critical ({alertCounts.critical || 0})</option>
          <option value="high">High ({alertCounts.high || 0})</option>
          <option value="medium">Medium ({alertCounts.medium || 0})</option>
          <option value="low">Low ({alertCounts.low || 0})</option>
        </select>
      </div>

      {/* Status filter */}
      <div className="flex items-center space-x-2">
        <span className="text-sm text-gray-500 dark:text-gray-400">Status:</span>
        <select
          value={acknowledgedFilter}
          onChange={(e) => onAcknowledgedChange(e.target.value)}
          className="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
        >
          <option value="all">All</option>
          <option value="unread">Unread ({alertCounts.unread || 0})</option>
          <option value="acknowledged">Acknowledged</option>
        </select>
      </div>
    </div>
  );
};

// Main AlertList component
export const AlertList: React.FC<AlertListProps> = ({
  alerts,
  onAcknowledge,
  maxAlerts = 50,
}) => {
  const [severityFilter, setSeverityFilter] = useState('all');
  const [acknowledgedFilter, setAcknowledgedFilter] = useState('all');

  // Calculate alert counts
  const alertCounts = React.useMemo(() => {
    return {
      total: alerts.length,
      critical: alerts.filter(a => a.severity === 'critical').length,
      high: alerts.filter(a => a.severity === 'high').length,
      medium: alerts.filter(a => a.severity === 'medium').length,
      low: alerts.filter(a => a.severity === 'low').length,
      unread: alerts.filter(a => !a.acknowledged).length,
    };
  }, [alerts]);

  // Filter alerts
  const filteredAlerts = React.useMemo(() => {
    let filtered = alerts;

    // Filter by severity
    if (severityFilter !== 'all') {
      filtered = filtered.filter(alert => alert.severity === severityFilter);
    }

    // Filter by acknowledgment status
    if (acknowledgedFilter === 'unread') {
      filtered = filtered.filter(alert => !alert.acknowledged);
    } else if (acknowledgedFilter === 'acknowledged') {
      filtered = filtered.filter(alert => alert.acknowledged);
    }

    // Sort by timestamp (newest first)
    filtered = filtered.sort((a, b) => b.timestamp - a.timestamp);

    // Limit results
    return filtered.slice(0, maxAlerts);
  }, [alerts, severityFilter, acknowledgedFilter, maxAlerts]);

  if (alerts.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
          Risk Alerts
        </h3>
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-success-100 dark:bg-success-900 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h4 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            All Clear
          </h4>
          <p className="text-gray-500 dark:text-gray-400">
            No risk alerts at this time. The system is operating within normal parameters.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
            Risk Alerts
          </h3>
          <div className="flex items-center space-x-2">
            {alertCounts.unread > 0 && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200">
                {alertCounts.unread} unread
              </span>
            )}
            <button
              onClick={() => {
                alerts.filter(a => !a.acknowledged).forEach(alert => {
                  onAcknowledge(alert.alert_id);
                });
              }}
              disabled={alertCounts.unread === 0}
              className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-500 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              Acknowledge All
            </button>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="px-6 py-3 bg-gray-50 dark:bg-gray-900">
        <AlertFilters
          severityFilter={severityFilter}
          onSeverityChange={setSeverityFilter}
          acknowledgedFilter={acknowledgedFilter}
          onAcknowledgedChange={setAcknowledgedFilter}
          alertCounts={alertCounts}
        />
      </div>

      {/* Alert list */}
      <div className="px-6 py-4 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500 dark:text-gray-400">
              No alerts match the current filters.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence>
              {filteredAlerts.map((alert, index) => (
                <AlertItem
                  key={alert.alert_id}
                  alert={alert}
                  onAcknowledge={onAcknowledge}
                  index={index}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Footer */}
      {filteredAlerts.length > 0 && (
        <div className="px-6 py-3 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Showing {filteredAlerts.length} of {alerts.length} alerts
            {maxAlerts < alerts.length && ` (limited to ${maxAlerts})`}
          </p>
        </div>
      )}
    </div>
  );
};