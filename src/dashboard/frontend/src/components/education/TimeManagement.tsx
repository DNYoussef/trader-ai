import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';

interface TradingSession {
  id: string;
  name: string;
  startTime: string;
  endTime: string;
  duration: number; // minutes
  energy: number; // 1-100
  focus: number; // 1-100
  tasks: SessionTask[];
  marketCondition: 'volatile' | 'trending' | 'sideways' | 'news';
  performance?: number;
}

interface SessionTask {
  id: string;
  title: string;
  type: 'analysis' | 'execution' | 'review' | 'learning';
  timeAllocation: number; // minutes
  priority: 'high' | 'medium' | 'low';
  completed: boolean;
  energyCost: number;
}

interface ProductivityInsight {
  category: string;
  insight: string;
  action: string;
  priority: 'high' | 'medium' | 'low';
}

export const TimeManagement: React.FC = () => {
  const [currentSession, setCurrentSession] = useState<TradingSession | null>(null);
  const [sessionTimer, setSessionTimer] = useState<number>(0);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');

  // Session Templates based on Guild of the Rose time management principles
  const sessionTemplates: TradingSession[] = [
    {
      id: 'morning_momentum',
      name: 'Morning Momentum Session',
      startTime: '09:30',
      endTime: '11:00',
      duration: 90,
      energy: 85,
      focus: 90,
      marketCondition: 'volatile',
      tasks: [
        {
          id: 'pre_market',
          title: 'Pre-market Analysis',
          type: 'analysis',
          timeAllocation: 20,
          priority: 'high',
          completed: false,
          energyCost: 15
        },
        {
          id: 'momentum_scan',
          title: 'Momentum Stock Scan',
          type: 'analysis',
          timeAllocation: 15,
          priority: 'high',
          completed: false,
          energyCost: 20
        },
        {
          id: 'position_entry',
          title: 'Position Entries',
          type: 'execution',
          timeAllocation: 30,
          priority: 'high',
          completed: false,
          energyCost: 25
        },
        {
          id: 'position_monitoring',
          title: 'Active Position Monitoring',
          type: 'execution',
          timeAllocation: 20,
          priority: 'medium',
          completed: false,
          energyCost: 20
        },
        {
          id: 'session_notes',
          title: 'Session Notes & Reflection',
          type: 'review',
          timeAllocation: 5,
          priority: 'medium',
          completed: false,
          energyCost: 5
        }
      ]
    },
    {
      id: 'midday_review',
      name: 'Midday Review & Adjustment',
      startTime: '12:00',
      endTime: '12:45',
      duration: 45,
      energy: 70,
      focus: 75,
      marketCondition: 'sideways',
      tasks: [
        {
          id: 'position_review',
          title: 'Portfolio Position Review',
          type: 'review',
          timeAllocation: 15,
          priority: 'high',
          completed: false,
          energyCost: 15
        },
        {
          id: 'risk_check',
          title: 'Risk Exposure Check',
          type: 'analysis',
          timeAllocation: 10,
          priority: 'high',
          completed: false,
          energyCost: 10
        },
        {
          id: 'afternoon_prep',
          title: 'Afternoon Strategy Prep',
          type: 'analysis',
          timeAllocation: 15,
          priority: 'medium',
          completed: false,
          energyCost: 15
        },
        {
          id: 'learning_time',
          title: 'Educational Content',
          type: 'learning',
          timeAllocation: 5,
          priority: 'low',
          completed: false,
          energyCost: 5
        }
      ]
    },
    {
      id: 'closing_session',
      name: 'Market Close Session',
      startTime: '15:30',
      endTime: '16:30',
      duration: 60,
      energy: 60,
      focus: 80,
      marketCondition: 'volatile',
      tasks: [
        {
          id: 'closing_positions',
          title: 'End-of-Day Position Management',
          type: 'execution',
          timeAllocation: 20,
          priority: 'high',
          completed: false,
          energyCost: 20
        },
        {
          id: 'day_review',
          title: 'Trading Day Review',
          type: 'review',
          timeAllocation: 15,
          priority: 'high',
          completed: false,
          energyCost: 15
        },
        {
          id: 'next_day_prep',
          title: 'Next Day Preparation',
          type: 'analysis',
          timeAllocation: 15,
          priority: 'medium',
          completed: false,
          energyCost: 10
        },
        {
          id: 'journal_entry',
          title: 'Trading Journal Entry',
          type: 'review',
          timeAllocation: 10,
          priority: 'medium',
          completed: false,
          energyCost: 10
        }
      ]
    }
  ];

  const [insights] = useState<ProductivityInsight[]>([
    {
      category: 'Energy Management',
      insight: 'Your focus peaks between 9:30-11:00 AM',
      action: 'Schedule high-stakes trades during morning hours',
      priority: 'high'
    },
    {
      category: 'Task Switching',
      insight: 'Frequent switching between analysis and execution reduces performance',
      action: 'Batch similar tasks together in time blocks',
      priority: 'high'
    },
    {
      category: 'Decision Fatigue',
      insight: 'Decision quality decreases after 2 hours of active trading',
      action: 'Take 15-minute breaks every 90 minutes',
      priority: 'medium'
    },
    {
      category: 'Learning Integration',
      insight: 'Low-energy periods are optimal for educational content',
      action: 'Schedule learning during 12-2 PM energy dip',
      priority: 'medium'
    }
  ]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSessionActive && currentSession) {
      interval = setInterval(() => {
        setSessionTimer(prev => {
          const newTime = prev + 1;
          if (newTime >= currentSession.duration * 60) {
            setIsSessionActive(false);
            return 0;
          }
          return newTime;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isSessionActive, currentSession]);

  const startSession = (template: TradingSession) => {
    setCurrentSession({ ...template });
    setSessionTimer(0);
    setIsSessionActive(true);
  };

  const pauseSession = () => {
    setIsSessionActive(!isSessionActive);
  };

  const endSession = () => {
    setIsSessionActive(false);
    setCurrentSession(null);
    setSessionTimer(0);
  };

  const toggleTask = (taskId: string) => {
    if (!currentSession) return;

    setCurrentSession(prev => ({
      ...prev!,
      tasks: prev!.tasks.map(task =>
        task.id === taskId ? { ...task, completed: !task.completed } : task
      )
    }));
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const calculateSessionProgress = () => {
    if (!currentSession) return 0;
    const completedTasks = currentSession.tasks.filter(task => task.completed).length;
    return (completedTasks / currentSession.tasks.length) * 100;
  };

  const getEnergyColor = (energy: number) => {
    if (energy >= 80) return 'text-green-600';
    if (energy >= 60) return 'text-yellow-600';
    if (energy >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-700';
      case 'medium': return 'bg-yellow-100 text-yellow-700';
      case 'low': return 'bg-green-100 text-green-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="space-y-6">
      {/* Session Timer & Controls */}
      {currentSession && (
        <Card>
          <CardHeader>
            <CardTitle>⏱️ Active Session: {currentSession.name}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Timer Display */}
              <div className="text-center p-6 bg-blue-50 rounded-lg">
                <div className="text-6xl font-mono font-bold text-blue-600 mb-2">
                  {formatTime(sessionTimer)}
                </div>
                <div className="text-sm text-gray-600">
                  {Math.floor(sessionTimer / 60)} / {currentSession.duration} minutes
                </div>
                <Progress
                  value={(sessionTimer / (currentSession.duration * 60)) * 100}
                  className="mt-3"
                />
              </div>

              {/* Session Controls */}
              <div className="flex gap-3 justify-center">
                <Button
                  onClick={pauseSession}
                  variant={isSessionActive ? "destructive" : "default"}
                >
                  {isSessionActive ? 'Pause' : 'Resume'}
                </Button>
                <Button onClick={endSession} variant="outline">
                  End Session
                </Button>
              </div>

              {/* Session Progress */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <p className="text-sm text-gray-600">Progress</p>
                  <p className="text-xl font-bold">{calculateSessionProgress().toFixed(0)}%</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Energy</p>
                  <p className={`text-xl font-bold ${getEnergyColor(currentSession.energy)}`}>
                    {currentSession.energy}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Focus</p>
                  <p className={`text-xl font-bold ${getEnergyColor(currentSession.focus)}`}>
                    {currentSession.focus}%
                  </p>
                </div>
              </div>

              {/* Task List */}
              <div className="space-y-2">
                <h4 className="font-semibold">Session Tasks</h4>
                {currentSession.tasks.map(task => (
                  <div
                    key={task.id}
                    className="flex items-center gap-3 p-3 bg-white rounded border cursor-pointer hover:border-blue-400"
                    onClick={() => toggleTask(task.id)}
                  >
                    <input
                      type="checkbox"
                      checked={task.completed}
                      onChange={() => {}}
                      className="w-4 h-4"
                    />
                    <div className="flex-1">
                      <p className={`font-medium ${task.completed ? 'line-through text-gray-500' : ''}`}>
                        {task.title}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-xs">
                          {task.type}
                        </Badge>
                        <Badge className={`text-xs ${getPriorityColor(task.priority)}`}>
                          {task.priority}
                        </Badge>
                        <span className="text-xs text-gray-500">
                          {task.timeAllocation}min
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">Energy Cost</p>
                      <p className="text-lg font-bold">{task.energyCost}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Session Templates */}
      <Card>
        <CardHeader>
          <CardTitle>📅 Trading Session Templates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Pre-designed session templates optimized for different market conditions and energy levels.
              Based on Guild of the Rose's contrarian time management principles.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {sessionTemplates.map(template => (
                <div
                  key={template.id}
                  className="p-4 border rounded-lg hover:border-blue-400 transition-colors"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-semibold">{template.name}</h3>
                    <Badge variant="outline">
                      {template.duration}min
                    </Badge>
                  </div>

                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Time:</span>
                      <span>{template.startTime} - {template.endTime}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Market:</span>
                      <Badge variant="secondary" className="text-xs">
                        {template.marketCondition}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Energy Level:</span>
                      <span className={getEnergyColor(template.energy)}>
                        {template.energy}%
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2 mb-4">
                    <p className="text-xs text-gray-600">Tasks ({template.tasks.length}):</p>
                    {template.tasks.slice(0, 3).map(task => (
                      <div key={task.id} className="text-xs bg-gray-50 p-2 rounded">
                        {task.title} ({task.timeAllocation}min)
                      </div>
                    ))}
                    {template.tasks.length > 3 && (
                      <p className="text-xs text-gray-500">
                        +{template.tasks.length - 3} more tasks
                      </p>
                    )}
                  </div>

                  <Button
                    onClick={() => startSession(template)}
                    disabled={isSessionActive}
                    className="w-full"
                    size="sm"
                  >
                    Start Session
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Productivity Insights */}
      <Card>
        <CardHeader>
          <CardTitle>💡 Productivity Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Personalized insights based on your trading patterns and energy cycles.
            </p>

            {insights.map((insight, idx) => (
              <div key={idx} className="p-4 border rounded-lg bg-blue-50">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-blue-900">{insight.category}</h4>
                  <Badge className={getPriorityColor(insight.priority)}>
                    {insight.priority}
                  </Badge>
                </div>
                <p className="text-sm text-blue-800 mb-2">
                  <strong>Insight:</strong> {insight.insight}
                </p>
                <p className="text-sm text-blue-700">
                  <strong>Action:</strong> {insight.action}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Exobrain System */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 Trading Exobrain</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              External memory system for capturing and organizing trading insights,
              inspired by Guild of the Rose's exobrain methodology.
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold mb-2">📝 Quick Capture</h4>
                <textarea
                  placeholder="Capture a quick trading insight or observation..."
                  className="w-full p-2 border rounded text-sm"
                  rows={3}
                />
                <Button size="sm" className="mt-2">
                  Save to Exobrain
                </Button>
              </div>

              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold mb-2">🔗 Pattern Links</h4>
                <div className="space-y-2">
                  <div className="text-sm p-2 bg-white rounded border">
                    <strong>SPY Breakout Pattern</strong><br />
                    <span className="text-gray-600">Linked to: Morning volatility, Volume spike</span>
                  </div>
                  <div className="text-sm p-2 bg-white rounded border">
                    <strong>Risk Management Rule</strong><br />
                    <span className="text-gray-600">Linked to: Position sizing, P(ruin) calculation</span>
                  </div>
                </div>
                <Button size="sm" className="mt-2" variant="outline">
                  View All Patterns
                </Button>
              </div>
            </div>

            <div className="p-4 bg-yellow-50 rounded-lg">
              <h4 className="font-semibold mb-2">🔍 Smart Search</h4>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Search your trading knowledge base..."
                  className="flex-1 p-2 border rounded text-sm"
                />
                <Button size="sm">
                  Search
                </Button>
              </div>
              <div className="mt-3 space-y-1">
                <p className="text-xs text-gray-600">Suggested searches:</p>
                <div className="flex gap-2">
                  {['risk management', 'momentum breakout', 'volatility patterns'].map(term => (
                    <Badge key={term} variant="outline" className="text-xs cursor-pointer">
                      {term}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};