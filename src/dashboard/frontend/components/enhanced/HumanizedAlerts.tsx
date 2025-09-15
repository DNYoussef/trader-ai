import React, { useState, useEffect } from 'react';
import { Alert, AlertDescription, AlertTitle } from '../../src/components/ui/alert';
import { Button } from '../../src/components/ui/button';
import { Badge } from '../../src/components/ui/badge';
import { Card, CardContent } from '../../src/components/ui/card';
import {
  AlertTriangle, CheckCircle, Info, TrendingUp, TrendingDown,
  Brain, Shield, Clock, Target, MessageCircle, X, ChevronDown, ChevronUp
} from 'lucide-react';

interface HumanizedAlert {
  id: string;
  type: 'risk' | 'opportunity' | 'gate_progress' | 'causal_insight' | 'celebration';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  humanMessage: string;
  technicalDetails?: string;
  actionRequired?: boolean;
  actionText?: string;
  actionCallback?: () => void;
  timestamp: Date;
  persona?: string;
  emotionalTone: 'encouraging' | 'warning' | 'informative' | 'celebratory' | 'urgent';
  relatedData?: any;
  dismissible?: boolean;
  expanded?: boolean;
}

interface HumanizedAlertsProps {
  alerts: HumanizedAlert[];
  onDismiss?: (alertId: string) => void;
  onAction?: (alertId: string, action: string) => void;
  persona?: string;
}

const HumanizedAlerts: React.FC<HumanizedAlertsProps> = ({
  alerts,
  onDismiss,
  onAction,
  persona = 'casual_investor'
}) => {
  const [expandedAlerts, setExpandedAlerts] = useState<Set<string>>(new Set());

  const getAlertIcon = (type: string, severity: string) => {
    switch (type) {
      case 'risk':
        return severity === 'critical' ? AlertTriangle : Shield;
      case 'opportunity':
        return TrendingUp;
      case 'gate_progress':
        return Target;
      case 'causal_insight':
        return Brain;
      case 'celebration':
        return CheckCircle;
      default:
        return Info;
    }
  };

  const getAlertColors = (type: string, severity: string, tone: string) => {
    if (tone === 'celebratory') {
      return {
        container: 'border-green-200 bg-green-50',
        icon: 'text-green-600',
        badge: 'bg-green-100 text-green-800'
      };
    }

    if (tone === 'urgent' || severity === 'critical') {
      return {
        container: 'border-red-200 bg-red-50',
        icon: 'text-red-600',
        badge: 'bg-red-100 text-red-800'
      };
    }

    if (tone === 'warning' || severity === 'high') {
      return {
        container: 'border-yellow-200 bg-yellow-50',
        icon: 'text-yellow-600',
        badge: 'bg-yellow-100 text-yellow-800'
      };
    }

    if (type === 'opportunity') {
      return {
        container: 'border-blue-200 bg-blue-50',
        icon: 'text-blue-600',
        badge: 'bg-blue-100 text-blue-800'
      };
    }

    return {
      container: 'border-gray-200 bg-gray-50',
      icon: 'text-gray-600',
      badge: 'bg-gray-100 text-gray-800'
    };
  };

  const toggleExpanded = (alertId: string) => {
    const newExpanded = new Set(expandedAlerts);
    if (newExpanded.has(alertId)) {
      newExpanded.delete(alertId);
    } else {
      newExpanded.add(alertId);
    }
    setExpandedAlerts(newExpanded);
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;

    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const renderAlert = (alert: HumanizedAlert) => {
    const IconComponent = getAlertIcon(alert.type, alert.severity);
    const colors = getAlertColors(alert.type, alert.severity, alert.emotionalTone);
    const isExpanded = expandedAlerts.has(alert.id);

    return (
      <Card key={alert.id} className={`${colors.container} border-l-4`}>
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <div className={`flex-shrink-0 ${colors.icon} mt-0.5`}>
              <IconComponent className="w-5 h-5" />
            </div>

            <div className="flex-1 space-y-2">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold text-gray-800">{alert.title}</h4>
                  <div className="flex items-center space-x-2 mt-1">
                    <Badge className={colors.badge} variant="outline">
                      {alert.type.replace('_', ' ')}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {formatTimeAgo(alert.timestamp)}
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-1">
                  {alert.technicalDetails && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleExpanded(alert.id)}
                      className="p-1 h-6 w-6"
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </Button>
                  )}

                  {alert.dismissible && onDismiss && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onDismiss(alert.id)}
                      className="p-1 h-6 w-6 text-gray-400 hover:text-gray-600"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>

              {/* Human message */}
              <div className="text-gray-700">
                {alert.humanMessage}
              </div>

              {/* Technical details (expandable) */}
              {isExpanded && alert.technicalDetails && (
                <div className="bg-white bg-opacity-50 p-3 rounded border">
                  <div className="text-sm text-gray-600 space-y-1">
                    <div className="font-medium text-gray-700 mb-1">Technical Details:</div>
                    {alert.technicalDetails}
                  </div>
                </div>
              )}

              {/* Related data visualization */}
              {isExpanded && alert.relatedData && (
                <div className="bg-white bg-opacity-50 p-3 rounded border">
                  <div className="text-sm">
                    {alert.relatedData.metrics && (
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(alert.relatedData.metrics).map(([key, value]) => (
                          <div key={key}>
                            <div className="text-gray-600 capitalize">
                              {key.replace('_', ' ')}
                            </div>
                            <div className="font-semibold text-gray-800">
                              {typeof value === 'number' ? value.toFixed(2) : String(value)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Action button */}
              {alert.actionRequired && alert.actionText && (
                <div className="pt-2">
                  <Button
                    size="sm"
                    onClick={() => {
                      if (alert.actionCallback) {
                        alert.actionCallback();
                      } else if (onAction) {
                        onAction(alert.id, alert.actionText!);
                      }
                    }}
                    className={
                      alert.emotionalTone === 'urgent'
                        ? 'bg-red-600 hover:bg-red-700 text-white'
                        : alert.emotionalTone === 'celebratory'
                        ? 'bg-green-600 hover:bg-green-700 text-white'
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                    }
                  >
                    {alert.actionText}
                  </Button>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  };

  if (!alerts || alerts.length === 0) {
    return (
      <div className="text-center py-6 text-gray-500">
        <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>No alerts right now - your system is running smoothly!</p>
      </div>
    );
  }

  // Sort alerts by priority and timestamp
  const sortedAlerts = [...alerts].sort((a, b) => {
    const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
    const aPriority = priorityOrder[a.severity] || 0;
    const bPriority = priorityOrder[b.severity] || 0;

    if (aPriority !== bPriority) {
      return bPriority - aPriority; // Higher priority first
    }

    return b.timestamp.getTime() - a.timestamp.getTime(); // Newer first
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">System Updates</h3>
        <Badge variant="outline" className="text-xs">
          {alerts.length} active
        </Badge>
      </div>

      <div className="space-y-3">
        {sortedAlerts.map(renderAlert)}
      </div>
    </div>
  );
};

// Helper function to create humanized alerts from system data
export const createHumanizedAlert = (
  systemAlert: any,
  persona: string = 'casual_investor'
): HumanizedAlert => {
  const personaStyles = {
    beginner: {
      tone: 'encouraging',
      language: 'simple',
      focus: 'safety'
    },
    casual_investor: {
      tone: 'informative',
      language: 'balanced',
      focus: 'performance'
    },
    active_trader: {
      tone: 'analytical',
      language: 'technical',
      focus: 'opportunity'
    },
    experienced_trader: {
      tone: 'professional',
      language: 'technical',
      focus: 'strategy'
    }
  };

  const style = personaStyles[persona as keyof typeof personaStyles] || personaStyles.casual_investor;

  // Convert technical alert to human message
  const humanizeMessage = (technicalMessage: string, type: string): string => {
    if (type === 'risk' && persona === 'beginner') {
      return "I noticed something that needs your attention to keep your account safe. Don't worry - this is exactly why we have protective systems in place.";
    }

    if (type === 'opportunity' && style.focus === 'performance') {
      return "Great news! I've spotted a potential opportunity that fits your trading strategy. Here's what I found...";
    }

    if (type === 'causal_insight') {
      return "My AI analysis just picked up on some market patterns that could affect your positions. Let me explain what this means for you...";
    }

    // Default humanization
    return technicalMessage.replace(/[A-Z_]+/g, (match) =>
      match.toLowerCase().replace(/_/g, ' ')
    );
  };

  return {
    id: systemAlert.id || Date.now().toString(),
    type: systemAlert.type || 'info',
    severity: systemAlert.severity || 'medium',
    title: systemAlert.title || 'System Update',
    humanMessage: humanizeMessage(systemAlert.message, systemAlert.type),
    technicalDetails: systemAlert.technicalDetails,
    actionRequired: systemAlert.actionRequired || false,
    actionText: systemAlert.actionText,
    actionCallback: systemAlert.actionCallback,
    timestamp: systemAlert.timestamp || new Date(),
    persona,
    emotionalTone: style.tone as any,
    relatedData: systemAlert.relatedData,
    dismissible: systemAlert.dismissible !== false
  };
};

export default HumanizedAlerts;