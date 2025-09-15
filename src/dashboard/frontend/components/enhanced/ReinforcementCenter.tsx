/**
 * Reinforcement Center - Display and manage psychological reinforcement events
 *
 * Provides real-time display of achievements, celebrations, and progress updates
 * with appropriate visual effects and user interaction capabilities.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useEnhancedUX } from './EnhancedUXProvider';

interface ReinforcementEvent {
  event_id: string;
  event_type: string;
  trigger_condition: string;
  intensity: string;
  title: string;
  message: string;
  visual_effect: string;
  sound_effect?: string;
  user_id: string;
  timestamp: string;
  call_to_action?: string;
  next_goal?: string;
  social_sharing_prompt?: string;
  engagement_score: number;
}

interface UserEngagementData {
  user_id: string;
  engagement_overview: {
    days_active: number;
    total_trades: number;
    total_profit: number;
    current_gate: string;
    concepts_mastered: number;
    tutorials_completed: number;
  };
  current_streaks: {
    daily_compliance: number;
    profit_streak: number;
    learning_streak: number;
  };
  milestone_progress: {
    next_trade_milestone: {
      target: number | string;
      progress: number;
      remaining: number;
      reward: string;
    };
    next_learning_milestone: {
      target: number | string;
      progress: number;
      remaining: number;
      reward: string;
    };
    next_gate_progress: {
      current_gate: string;
      capital_progress?: number;
      trade_progress?: number;
      estimated_days_remaining?: number;
    };
  };
  recent_achievements: Array<{
    type: string;
    gate?: string;
    timestamp: string;
  }>;
}

interface ReinforcementCenterProps {
  userId: string;
  onEventInteraction?: (eventId: string, interaction: string) => void;
  onSocialShare?: (eventId: string, platform: string) => void;
}

const ReinforcementCenter: React.FC<ReinforcementCenterProps> = ({
  userId,
  onEventInteraction,
  onSocialShare
}) => {
  const [activeEvents, setActiveEvents] = useState<ReinforcementEvent[]>([]);
  const [engagementData, setEngagementData] = useState<UserEngagementData | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [eventHistory, setEventHistory] = useState<ReinforcementEvent[]>([]);
  const [loading, setLoading] = useState(true);

  const { triggerSystemEvent } = useEnhancedUX();

  useEffect(() => {
    loadEngagementDashboard();
    setupEventListeners();
  }, [userId]);

  const loadEngagementDashboard = async () => {
    try {
      setLoading(true);

      // Mock data - in production this would call the backend API
      const mockEngagementData: UserEngagementData = {
        user_id: userId,
        engagement_overview: {
          days_active: 15,
          total_trades: 23,
          total_profit: 47.50,
          current_gate: 'G0',
          concepts_mastered: 2,
          tutorials_completed: 1
        },
        current_streaks: {
          daily_compliance: 5,
          profit_streak: 3,
          learning_streak: 1
        },
        milestone_progress: {
          next_trade_milestone: {
            target: 25,
            progress: 23,
            remaining: 2,
            reward: 'Milestone celebration at 25 trades'
          },
          next_learning_milestone: {
            target: 3,
            progress: 2,
            remaining: 1,
            reward: 'Learning achievement at 3 concepts'
          },
          next_gate_progress: {
            current_gate: 'G0',
            capital_progress: 9.5,
            trade_progress: 92,
            estimated_days_remaining: 3
          }
        },
        recent_achievements: [
          {
            type: 'concept_mastery',
            timestamp: new Date(Date.now() - 86400000).toISOString()
          },
          {
            type: 'profit_milestone',
            timestamp: new Date(Date.now() - 172800000).toISOString()
          }
        ]
      };

      setEngagementData(mockEngagementData);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load engagement dashboard:', error);
      setLoading(false);
    }
  };

  const setupEventListeners = () => {
    // In a real implementation, this would setup WebSocket listeners
    // for real-time reinforcement events

    // Mock event simulation
    const simulateEvent = () => {
      const mockEvent: ReinforcementEvent = {
        event_id: `event_${Date.now()}`,
        event_type: 'achievement',
        trigger_condition: 'profit_achieved',
        intensity: 'moderate',
        title: 'Nice Profit!',
        message: 'You just made $12.50 following the system - this proves it works!',
        visual_effect: 'confetti_burst',
        user_id: userId,
        timestamp: new Date().toISOString(),
        call_to_action: 'Keep following the system for more wins',
        engagement_score: 0.8
      };

      setActiveEvents(prev => [mockEvent, ...prev.slice(0, 2)]);

      // Auto-dismiss after 8 seconds
      setTimeout(() => {
        dismissEvent(mockEvent.event_id);
      }, 8000);
    };

    // Simulate random events for demo
    const interval = setInterval(() => {
      if (Math.random() > 0.7) { // 30% chance every 10 seconds
        simulateEvent();
      }
    }, 10000);

    return () => clearInterval(interval);
  };

  const dismissEvent = useCallback((eventId: string) => {
    setActiveEvents(prev => {
      const event = prev.find(e => e.event_id === eventId);
      if (event) {
        setEventHistory(hist => [event, ...hist.slice(0, 19)]); // Keep last 20
      }
      return prev.filter(e => e.event_id !== eventId);
    });

    if (onEventInteraction) {
      onEventInteraction(eventId, 'dismissed');
    }
  }, [onEventInteraction]);

  const handleEventClick = useCallback((eventId: string, action: string) => {
    const event = activeEvents.find(e => e.event_id === eventId);

    if (event && action === 'cta_clicked' && event.call_to_action) {
      triggerSystemEvent('reinforcement_cta_clicked', {
        event_id: eventId,
        event_type: event.event_type,
        cta_text: event.call_to_action
      });
    }

    if (onEventInteraction) {
      onEventInteraction(eventId, action);
    }
  }, [activeEvents, onEventInteraction, triggerSystemEvent]);

  const handleSocialShare = useCallback((eventId: string, platform: string) => {
    if (onSocialShare) {
      onSocialShare(eventId, platform);
    }

    triggerSystemEvent('social_share_attempted', {
      event_id: eventId,
      platform: platform
    });
  }, [onSocialShare, triggerSystemEvent]);

  const getVisualEffectClass = (effect: string, intensity: string) => {
    const baseClass = 'transition-all duration-500 ';

    switch (effect) {
      case 'confetti_burst':
        return baseClass + 'animate-bounce bg-gradient-to-r from-yellow-400 to-orange-500';
      case 'epic_celebration':
        return baseClass + 'animate-pulse bg-gradient-to-r from-purple-500 to-pink-500';
      case 'streak_animation':
        return baseClass + 'animate-pulse bg-gradient-to-r from-green-400 to-blue-500';
      case 'shield_effect':
        return baseClass + 'bg-gradient-to-r from-blue-500 to-cyan-500';
      case 'knowledge_burst':
        return baseClass + 'bg-gradient-to-r from-indigo-500 to-purple-500';
      case 'sparkle_effect':
        return baseClass + 'animate-pulse bg-gradient-to-r from-pink-400 to-red-500';
      default:
        return baseClass + 'bg-gradient-to-r from-gray-400 to-gray-600';
    }
  };

  const getIntensityBorder = (intensity: string) => {
    switch (intensity) {
      case 'subtle': return 'border-l-2 border-gray-300';
      case 'moderate': return 'border-l-4 border-blue-400';
      case 'strong': return 'border-l-4 border-yellow-400';
      case 'epic': return 'border-l-8 border-purple-500';
      default: return 'border-l-2 border-gray-300';
    }
  };

  const ProgressRing: React.FC<{ progress: number; size: number; strokeWidth: number }> = ({
    progress,
    size,
    strokeWidth
  }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDasharray = circumference;
    const strokeDashoffset = circumference - (progress / 100) * circumference;

    return (
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-gray-200"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          className="text-blue-600 transition-all duration-500"
        />
      </svg>
    );
  };

  const ReinforcementEventCard: React.FC<{ event: ReinforcementEvent; isActive?: boolean }> = ({
    event,
    isActive = true
  }) => (
    <div
      className={`${getVisualEffectClass(event.visual_effect, event.intensity)} ${getIntensityBorder(event.intensity)} rounded-lg shadow-lg p-4 text-white relative ${isActive ? 'mb-4' : 'mb-2 opacity-75'}`}
    >
      {isActive && (
        <button
          onClick={() => dismissEvent(event.event_id)}
          className="absolute top-2 right-2 text-white hover:text-gray-200 text-lg"
        >
          ×
        </button>
      )}

      <div className="mb-2">
        <h3 className="text-lg font-bold">{event.title}</h3>
        <p className="text-sm opacity-90">{event.message}</p>
      </div>

      {event.call_to_action && isActive && (
        <button
          onClick={() => handleEventClick(event.event_id, 'cta_clicked')}
          className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-4 py-2 rounded-md text-sm font-medium mr-2 mb-2"
        >
          {event.call_to_action}
        </button>
      )}

      {event.social_sharing_prompt && isActive && (
        <div className="flex space-x-2">
          <button
            onClick={() => handleSocialShare(event.event_id, 'twitter')}
            className="bg-white bg-opacity-20 hover:bg-opacity-30 text-white px-3 py-1 rounded-md text-xs"
          >
            Share
          </button>
        </div>
      )}

      <div className="text-xs opacity-75 mt-2">
        {new Date(event.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!engagementData) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Unable to load engagement data</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Active Reinforcement Events */}
      {activeEvents.length > 0 && (
        <div className="fixed top-20 right-6 z-50 w-80">
          {activeEvents.map(event => (
            <ReinforcementEventCard key={event.event_id} event={event} isActive={true} />
          ))}
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Engagement Dashboard</h1>
          <p className="text-gray-600">Track your progress and celebrate achievements</p>
        </div>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          {showHistory ? 'Hide History' : 'Show History'}
        </button>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{engagementData.engagement_overview.days_active}</div>
          <div className="text-sm text-gray-600">Days Active</div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-green-600">{engagementData.engagement_overview.total_trades}</div>
          <div className="text-sm text-gray-600">Total Trades</div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-green-600">${engagementData.engagement_overview.total_profit.toFixed(2)}</div>
          <div className="text-sm text-gray-600">Total Profit</div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">{engagementData.engagement_overview.current_gate}</div>
          <div className="text-sm text-gray-600">Current Gate</div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-indigo-600">{engagementData.engagement_overview.concepts_mastered}</div>
          <div className="text-sm text-gray-600">Concepts Mastered</div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-orange-600">{engagementData.engagement_overview.tutorials_completed}</div>
          <div className="text-sm text-gray-600">Tutorials Done</div>
        </div>
      </div>

      {/* Progress Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Current Streaks */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Current Streaks</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Daily Compliance</span>
              <span className="font-bold text-blue-600">{engagementData.current_streaks.daily_compliance} days</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Profit Streak</span>
              <span className="font-bold text-green-600">{engagementData.current_streaks.profit_streak} trades</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Learning Streak</span>
              <span className="font-bold text-purple-600">{engagementData.current_streaks.learning_streak} concepts</span>
            </div>
          </div>
        </div>

        {/* Next Milestones */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Next Milestones</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Trade Milestone</span>
                <span>{engagementData.milestone_progress.next_trade_milestone.progress} / {engagementData.milestone_progress.next_trade_milestone.target}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(engagementData.milestone_progress.next_trade_milestone.progress / (engagementData.milestone_progress.next_trade_milestone.target as number)) * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-600 mt-1">{engagementData.milestone_progress.next_trade_milestone.reward}</div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Learning Milestone</span>
                <span>{engagementData.milestone_progress.next_learning_milestone.progress} / {engagementData.milestone_progress.next_learning_milestone.target}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(engagementData.milestone_progress.next_learning_milestone.progress / (engagementData.milestone_progress.next_learning_milestone.target as number)) * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-600 mt-1">{engagementData.milestone_progress.next_learning_milestone.reward}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Gate Progress */}
      {engagementData.milestone_progress.next_gate_progress.capital_progress !== undefined && (
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Gate {engagementData.milestone_progress.next_gate_progress.current_gate} Progress
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="text-center">
              <div className="relative inline-block mb-2">
                <ProgressRing
                  progress={engagementData.milestone_progress.next_gate_progress.capital_progress || 0}
                  size={100}
                  strokeWidth={8}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-lg font-bold text-gray-800">
                    {Math.round(engagementData.milestone_progress.next_gate_progress.capital_progress || 0)}%
                  </span>
                </div>
              </div>
              <div className="text-sm text-gray-600">Capital Progress</div>
            </div>

            <div className="text-center">
              <div className="relative inline-block mb-2">
                <ProgressRing
                  progress={engagementData.milestone_progress.next_gate_progress.trade_progress || 0}
                  size={100}
                  strokeWidth={8}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-lg font-bold text-gray-800">
                    {Math.round(engagementData.milestone_progress.next_gate_progress.trade_progress || 0)}%
                  </span>
                </div>
              </div>
              <div className="text-sm text-gray-600">Trade Progress</div>
            </div>
          </div>

          {engagementData.milestone_progress.next_gate_progress.estimated_days_remaining !== undefined && (
            <div className="text-center mt-4">
              <span className="text-sm text-gray-600">
                Estimated {engagementData.milestone_progress.next_gate_progress.estimated_days_remaining} days to next gate
              </span>
            </div>
          )}
        </div>
      )}

      {/* Event History */}
      {showHistory && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Events</h2>
          {eventHistory.length > 0 ? (
            <div className="space-y-2">
              {eventHistory.map(event => (
                <ReinforcementEventCard key={event.event_id} event={event} isActive={false} />
              ))}
            </div>
          ) : (
            <p className="text-gray-600 text-center py-8">No recent events to display</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ReinforcementCenter;