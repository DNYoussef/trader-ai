import React from 'react';

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'success' | 'error';
  title: string;
  message: string;
  timestamp: Date;
}

interface AlertListProps {
  alerts: Alert[];
}

export const AlertList: React.FC<AlertListProps> = ({ alerts }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error':
        return 'bg-red-100 border-red-400 text-red-700';
      case 'warning':
        return 'bg-yellow-100 border-yellow-400 text-yellow-700';
      case 'success':
        return 'bg-green-100 border-green-400 text-green-700';
      default:
        return 'bg-blue-100 border-blue-400 text-blue-700';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return '❌';
      case 'warning':
        return '⚠️';
      case 'success':
        return '✅';
      default:
        return 'ℹ️';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
        Recent Alerts
      </h3>
      <div className="space-y-3">
        {alerts.map(alert => (
          <div
            key={alert.id}
            className={`p-3 rounded border-l-4 ${getSeverityColor(alert.severity)}`}
          >
            <div className="flex items-start">
              <span className="mr-2 text-lg">{getSeverityIcon(alert.severity)}</span>
              <div className="flex-1">
                <p className="font-medium">{alert.title}</p>
                <p className="text-sm mt-1">{alert.message}</p>
                <p className="text-xs mt-1 opacity-75">
                  {alert.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        ))}
        {alerts.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No alerts
          </div>
        )}
      </div>
    </div>
  );
};