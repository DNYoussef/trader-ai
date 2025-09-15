import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';

interface Quest {
  id: string;
  title: string;
  description: string;
  gate: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  type: 'solo' | 'cohort' | 'challenge';
  stages: QuestStage[];
  rewards: QuestReward[];
  prerequisites: string[];
  estimatedTime: string;
  participants?: string[];
  status: 'locked' | 'available' | 'in_progress' | 'completed';
}

interface QuestStage {
  id: string;
  title: string;
  description: string;
  tasks: QuestTask[];
  completed: boolean;
}

interface QuestTask {
  id: string;
  title: string;
  type: 'practice' | 'theory' | 'simulation' | 'collaboration';
  completed: boolean;
  optional?: boolean;
}

interface QuestReward {
  type: 'badge' | 'skill_points' | 'unlock' | 'achievement';
  title: string;
  description: string;
  value?: number;
}

export const QuestSystem: React.FC = () => {
  const [activeQuest, setActiveQuest] = useState<string | null>(null);
  const [userLevel, setUserLevel] = useState(1);
  const [userGate, setUserGate] = useState('G0');
  const [completedQuests, setCompletedQuests] = useState<string[]>([]);

  const [quests] = useState<Quest[]>([
    {
      id: 'gate_basics',
      title: 'Gate System Fundamentals',
      description: 'Master the foundational concepts of progressive capital gates',
      gate: 'G0',
      difficulty: 'beginner',
      type: 'solo',
      estimatedTime: '2 hours',
      status: 'available',
      prerequisites: [],
      stages: [
        {
          id: 'stage1',
          title: 'Understanding Risk Limits',
          description: 'Learn how gates protect capital through position sizing',
          completed: false,
          tasks: [
            { id: 'task1', title: 'Calculate G0 position limits', type: 'practice', completed: false },
            { id: 'task2', title: 'Review risk percentage rules', type: 'theory', completed: false },
            { id: 'task3', title: 'Practice position sizing', type: 'simulation', completed: false }
          ]
        },
        {
          id: 'stage2',
          title: 'Gate Progression Rules',
          description: 'Understand when and how to advance between gates',
          completed: false,
          tasks: [
            { id: 'task4', title: 'Calculate progression thresholds', type: 'practice', completed: false },
            { id: 'task5', title: 'Understand siphon system', type: 'theory', completed: false }
          ]
        }
      ],
      rewards: [
        { type: 'badge', title: 'Gate Guardian', description: 'Completed basic gate training' },
        { type: 'skill_points', title: 'Risk Management +10', description: 'Improved risk assessment skills', value: 10 },
        { type: 'unlock', title: 'G1 Quest Access', description: 'Unlocked intermediate quests' }
      ]
    },
    {
      id: 'momentum_mastery',
      title: 'Dual Momentum Quest',
      description: 'Implement Gary Antonacci\'s momentum strategies with cohort collaboration',
      gate: 'G1',
      difficulty: 'intermediate',
      type: 'cohort',
      estimatedTime: '4 hours',
      status: 'locked',
      participants: ['You', 'Alice_Trader', 'Bob_Momentum', 'Carol_Risk'],
      prerequisites: ['gate_basics'],
      stages: [
        {
          id: 'collab1',
          title: 'Babble Phase: Strategy Ideas',
          description: 'Brainstorm momentum indicators with your cohort',
          completed: false,
          tasks: [
            { id: 'brainstorm1', title: 'Generate 20 momentum ideas', type: 'collaboration', completed: false },
            { id: 'brainstorm2', title: 'Research existing indicators', type: 'theory', completed: false },
            { id: 'brainstorm3', title: 'Document team findings', type: 'collaboration', completed: false }
          ]
        },
        {
          id: 'collab2',
          title: 'Prune Phase: Filter Best Ideas',
          description: 'Evaluate and select the most promising strategies',
          completed: false,
          tasks: [
            { id: 'filter1', title: 'Rank strategies by potential', type: 'collaboration', completed: false },
            { id: 'filter2', title: 'Backtest top 3 strategies', type: 'simulation', completed: false },
            { id: 'filter3', title: 'Present to cohort', type: 'collaboration', completed: false }
          ]
        },
        {
          id: 'collab3',
          title: 'Implementation Quest',
          description: 'Build and test the selected strategy',
          completed: false,
          tasks: [
            { id: 'build1', title: 'Code momentum calculator', type: 'practice', completed: false },
            { id: 'build2', title: 'Run paper trading simulation', type: 'simulation', completed: false },
            { id: 'build3', title: 'Document results for team', type: 'collaboration', completed: false }
          ]
        }
      ],
      rewards: [
        { type: 'badge', title: 'Momentum Master', description: 'Mastered dual momentum strategies' },
        { type: 'skill_points', title: 'Strategy Development +15', description: 'Advanced strategy building skills', value: 15 },
        { type: 'achievement', title: 'Team Player', description: 'Completed first cohort quest' }
      ]
    },
    {
      id: 'antifragile_architect',
      title: 'Antifragile Portfolio Challenge',
      description: 'Design portfolios that benefit from market volatility',
      gate: 'G2',
      difficulty: 'advanced',
      type: 'challenge',
      estimatedTime: '6 hours',
      status: 'locked',
      prerequisites: ['gate_basics', 'momentum_mastery'],
      stages: [
        {
          id: 'challenge1',
          title: 'Barbell Construction',
          description: 'Build portfolios with asymmetric risk/reward',
          completed: false,
          tasks: [
            { id: 'barbell1', title: 'Design 90/10 allocation', type: 'practice', completed: false },
            { id: 'barbell2', title: 'Optimize for black swans', type: 'simulation', completed: false },
            { id: 'barbell3', title: 'Test under stress scenarios', type: 'simulation', completed: false }
          ]
        },
        {
          id: 'challenge2',
          title: 'Volatility Harvesting',
          description: 'Create systems that profit from uncertainty',
          completed: false,
          tasks: [
            { id: 'vol1', title: 'Implement convex strategies', type: 'practice', completed: false },
            { id: 'vol2', title: 'Backtest through crises', type: 'simulation', completed: false }
          ]
        }
      ],
      rewards: [
        { type: 'badge', title: 'Antifragile Architect', description: 'Built robust antifragile systems' },
        { type: 'skill_points', title: 'Portfolio Design +20', description: 'Master-level portfolio construction', value: 20 },
        { type: 'unlock', title: 'Expert Quests', description: 'Access to expert-level challenges' }
      ]
    }
  ]);

  const getQuestStatusColor = (status: Quest['status']) => {
    switch (status) {
      case 'locked': return 'bg-gray-100 text-gray-600';
      case 'available': return 'bg-blue-100 text-blue-700';
      case 'in_progress': return 'bg-yellow-100 text-yellow-700';
      case 'completed': return 'bg-green-100 text-green-700';
    }
  };

  const getDifficultyColor = (difficulty: Quest['difficulty']) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-700';
      case 'intermediate': return 'bg-yellow-100 text-yellow-700';
      case 'advanced': return 'bg-orange-100 text-orange-700';
      case 'expert': return 'bg-red-100 text-red-700';
    }
  };

  const calculateQuestProgress = (quest: Quest) => {
    const totalTasks = quest.stages.reduce((sum, stage) => sum + stage.tasks.length, 0);
    const completedTasks = quest.stages.reduce((sum, stage) =>
      sum + stage.tasks.filter(task => task.completed).length, 0
    );
    return totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;
  };

  const toggleTaskComplete = (questId: string, stageId: string, taskId: string) => {
    // In a real app, this would update the backend
    console.log(`Toggling task ${taskId} in stage ${stageId} of quest ${questId}`);
  };

  const startQuest = (questId: string) => {
    setActiveQuest(questId);
  };

  const availableQuests = quests.filter(q => q.status === 'available' || q.status === 'in_progress');
  const lockedQuests = quests.filter(q => q.status === 'locked');
  const selectedQuest = quests.find(q => q.id === activeQuest);

  return (
    <div className="space-y-6">
      {/* Quest Overview */}
      <Card>
        <CardHeader>
          <CardTitle>🏛️ Quest System Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-2xl font-bold text-blue-600">{userLevel}</p>
              <p className="text-sm text-gray-600">Current Level</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-2xl font-bold text-green-600">{userGate}</p>
              <p className="text-sm text-gray-600">Current Gate</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-2xl font-bold text-purple-600">{completedQuests.length}</p>
              <p className="text-sm text-gray-600">Completed Quests</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <p className="text-2xl font-bold text-yellow-600">{availableQuests.length}</p>
              <p className="text-sm text-gray-600">Available Quests</p>
            </div>
          </div>

          {/* Quest Philosophy */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold mb-2">🎯 Quest Methodology</h4>
            <p className="text-sm text-gray-600 mb-3">
              Inspired by Guild of the Rose's structured progression system, our quests use:
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="font-medium text-sm">🗣️ Babble & Prune</p>
                <p className="text-xs text-gray-600">Generate many ideas, then filter the best</p>
              </div>
              <div>
                <p className="font-medium text-sm">👥 Cohort Learning</p>
                <p className="text-xs text-gray-600">Collaborative skill development with peers</p>
              </div>
              <div>
                <p className="font-medium text-sm">🔄 Iterative Improvement</p>
                <p className="text-xs text-gray-600">Set up, try out, iterate on strategies</p>
              </div>
              <div>
                <p className="font-medium text-sm">🎮 Gamified Progress</p>
                <p className="text-xs text-gray-600">Achievements, badges, and skill trees</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Available Quests */}
      <Card>
        <CardHeader>
          <CardTitle>📋 Available Quests</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {availableQuests.map(quest => (
              <div
                key={quest.id}
                className="p-4 border rounded-lg hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => setActiveQuest(quest.id)}
              >
                <div className="flex justify-between items-start mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="font-semibold">{quest.title}</h3>
                      <Badge className={getDifficultyColor(quest.difficulty)}>
                        {quest.difficulty}
                      </Badge>
                      <Badge className={getQuestStatusColor(quest.status)}>
                        {quest.status.replace('_', ' ')}
                      </Badge>
                      <Badge variant="outline">{quest.type}</Badge>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{quest.description}</p>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span>⏱️ {quest.estimatedTime}</span>
                      <span>🚪 {quest.gate}</span>
                      {quest.participants && <span>👥 {quest.participants.length} participants</span>}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{calculateQuestProgress(quest).toFixed(0)}%</p>
                    <Progress value={calculateQuestProgress(quest)} className="w-20 mt-1" />
                  </div>
                </div>

                {quest.status === 'available' && (
                  <Button onClick={() => startQuest(quest.id)} size="sm">
                    Start Quest
                  </Button>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Locked Quests */}
      {lockedQuests.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>🔒 Locked Quests</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {lockedQuests.map(quest => (
                <div key={quest.id} className="p-4 border rounded-lg bg-gray-50 opacity-60">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-semibold text-gray-600">{quest.title}</h3>
                      <p className="text-sm text-gray-500 mt-1">{quest.description}</p>
                      <p className="text-xs text-gray-400 mt-2">
                        Prerequisites: {quest.prerequisites.join(', ')}
                      </p>
                    </div>
                    <Badge className="bg-gray-200 text-gray-600">
                      Requires {quest.gate}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quest Detail View */}
      {selectedQuest && (
        <Card>
          <CardHeader>
            <CardTitle>🎯 Quest Details: {selectedQuest.title}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Quest Info */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm mb-3">{selectedQuest.description}</p>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-gray-600">Type</p>
                    <p className="font-medium">{selectedQuest.type}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Difficulty</p>
                    <p className="font-medium">{selectedQuest.difficulty}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Estimated Time</p>
                    <p className="font-medium">{selectedQuest.estimatedTime}</p>
                  </div>
                </div>
              </div>

              {/* Stages */}
              <div className="space-y-4">
                <h4 className="font-semibold">Quest Stages</h4>
                {selectedQuest.stages.map((stage, stageIdx) => (
                  <div key={stage.id} className="border rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="w-6 h-6 bg-blue-600 text-white rounded-full text-xs flex items-center justify-center">
                        {stageIdx + 1}
                      </span>
                      <h5 className="font-medium">{stage.title}</h5>
                      {stage.completed && <Badge className="bg-green-100 text-green-700">✓ Complete</Badge>}
                    </div>
                    <p className="text-sm text-gray-600 mb-3">{stage.description}</p>

                    {/* Tasks */}
                    <div className="space-y-2">
                      {stage.tasks.map(task => (
                        <div
                          key={task.id}
                          className="flex items-center gap-3 p-2 bg-gray-50 rounded cursor-pointer hover:bg-gray-100"
                          onClick={() => toggleTaskComplete(selectedQuest.id, stage.id, task.id)}
                        >
                          <input
                            type="checkbox"
                            checked={task.completed}
                            onChange={() => {}}
                            className="w-4 h-4"
                          />
                          <span className={`flex-1 text-sm ${task.completed ? 'line-through text-gray-500' : ''}`}>
                            {task.title}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {task.type}
                          </Badge>
                          {task.optional && (
                            <Badge variant="secondary" className="text-xs">
                              optional
                            </Badge>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Rewards */}
              <div>
                <h4 className="font-semibold mb-3">🏆 Quest Rewards</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {selectedQuest.rewards.map((reward, idx) => (
                    <div key={idx} className="p-3 border rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-lg">
                          {reward.type === 'badge' ? '🏅' :
                           reward.type === 'skill_points' ? '⭐' :
                           reward.type === 'unlock' ? '🔓' : '🏆'}
                        </span>
                        <p className="font-medium text-sm">{reward.title}</p>
                      </div>
                      <p className="text-xs text-gray-600">{reward.description}</p>
                      {reward.value && (
                        <p className="text-xs text-green-600 mt-1">+{reward.value} points</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};