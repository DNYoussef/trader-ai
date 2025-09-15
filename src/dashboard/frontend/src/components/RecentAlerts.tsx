import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface Alert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'error';
  title: string;
  message: string;
  timestamp: string;
}

interface RecentAlertsProps {
  alerts?: Alert[];
}

export const RecentAlerts: React.FC<RecentAlertsProps> = ({
  alerts = [
    {
      id: '1',
      type: 'warning',
      title: 'High P(ruin)',
      message: 'P(ruin) approaching threshold at 12%',
      timestamp: '12:34:52 PM'
    },
    {
      id: '2',
      type: 'info',
      title: 'Market Update',
      message: 'SPY showing bullish momentum',
      timestamp: '12:34:52 PM'
    },
    {
      id: '3',
      type: 'success',
      title: 'Trade Executed',
      message: 'Successfully bought 50 shares of SPY',
      timestamp: '12:34:52 PM'
    }
  ]
}) => {
  const getAlertColor = (type: string) => {
    switch (type) {
      case 'warning': return 'bg-yellow-50 border-l-yellow-400 text-yellow-700';
      case 'error': return 'bg-red-50 border-l-red-400 text-red-700';
      case 'success': return 'bg-green-50 border-l-green-400 text-green-700';
      case 'info': return 'bg-blue-50 border-l-blue-400 text-blue-700';
      default: return 'bg-gray-50 border-l-gray-400 text-gray-700';
    }
  };

  const getBadgeColor = (type: string) => {
    switch (type) {
      case 'warning': return 'bg-yellow-100 text-yellow-700';
      case 'error': return 'bg-red-100 text-red-700';
      case 'success': return 'bg-green-100 text-green-700';
      case 'info': return 'bg-blue-100 text-blue-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Recent Alerts</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {alerts.map((alert) => (
          <div
            key={alert.id}
            className={`p-3 rounded-lg border-l-4 ${getAlertColor(alert.type)}`}
          >
            <div className="flex justify-between items-start mb-1">
              <h4 className="font-semibold text-sm">{alert.title}</h4>
              <Badge className={getBadgeColor(alert.type)} variant="secondary">
                {alert.type}
              </Badge>
            </div>
            <p className="text-sm mb-2">{alert.message}</p>
            <span className="text-xs opacity-75">{alert.timestamp}</span>
          </div>
        ))}

        <div className="text-center pt-2">
          <button className="text-sm text-blue-600 hover:text-blue-800">
            View All Alerts
          </button>
        </div>
      </CardContent>
    </Card>
  );
};