import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';

// Import education modules
import { DecisionTheory } from './DecisionTheory';
import { Antifragility } from './Antifragility';
import { QuestSystem } from './QuestSystem';
import { CharacterSheet } from './CharacterSheet';
import { TimeManagement } from './TimeManagement';

interface EducationModule {
  id: string;
  title: string;
  description: string;
  icon: string;
  category: 'core' | 'advanced' | 'psychology' | 'systems';
  progress: number;
  estimatedTime: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  guildOfRoseConnection: string;
}

export const EducationHub: React.FC = () => {
  const [activeModule, setActiveModule] = useState<string>('overview');

  const modules: EducationModule[] = [
    {
      id: 'decision_theory',
      title: 'Decision Theory & Risk Calculators',
      description: 'Master decision-making under uncertainty with interactive calculators and decision trees',
      icon: '🧮',
      category: 'core',
      progress: 85,
      estimatedTime: '3 hours',
      difficulty: 'intermediate',
      guildOfRoseConnection: 'Applied Decision Theory 5, Decision Theory 6 (Ruin), Certain Equivalent workshops'
    },
    {
      id: 'antifragility',
      title: 'Antifragility & Barbell Strategy',
      description: 'Build portfolios that benefit from volatility and uncertainty',
      icon: '⚖️',
      category: 'advanced',
      progress: 60,
      estimatedTime: '4 hours',
      difficulty: 'advanced',
      guildOfRoseConnection: 'Chaos, Risk, and Antifragility workshop and Taleb methodology'
    },
    {
      id: 'quest_system',
      title: 'Progressive Quest System',
      description: 'Collaborative learning through structured quests and gate progression',
      icon: '🏛️',
      category: 'systems',
      progress: 40,
      estimatedTime: '2 hours',
      difficulty: 'beginner',
      guildOfRoseConnection: 'Quest Creation, Quest Planning workshops with Babble & Prune methodology'
    },
    {
      id: 'character_sheet',
      title: 'Character Sheet & Skill Trees',
      description: 'Track your trading skills development and energy management',
      icon: '📜',
      category: 'psychology',
      progress: 70,
      estimatedTime: '1.5 hours',
      difficulty: 'beginner',
      guildOfRoseConnection: 'Level Up Session 2.0, Metalearning, Soul Mapping workshops'
    },
    {
      id: 'time_management',
      title: 'Trading Time Management',
      description: 'Optimize your trading sessions with energy-aware scheduling',
      icon: '⏱️',
      category: 'psychology',
      progress: 55,
      estimatedTime: '2.5 hours',
      difficulty: 'intermediate',
      guildOfRoseConnection: 'Contrarian Time Management, Creating an Exobrain workshops'
    }
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-700';
      case 'intermediate': return 'bg-yellow-100 text-yellow-700';
      case 'advanced': return 'bg-orange-100 text-orange-700';
      case 'expert': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'core': return 'bg-blue-100 text-blue-700';
      case 'advanced': return 'bg-purple-100 text-purple-700';
      case 'psychology': return 'bg-pink-100 text-pink-700';
      case 'systems': return 'bg-indigo-100 text-indigo-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const overallProgress = modules.reduce((sum, module) => sum + module.progress, 0) / modules.length;

  const renderModuleContent = () => {
    switch (activeModule) {
      case 'decision_theory':
        return <DecisionTheory />;
      case 'antifragility':
        return <Antifragility />;
      case 'quest_system':
        return <QuestSystem />;
      case 'character_sheet':
        return <CharacterSheet />;
      case 'time_management':
        return <TimeManagement />;
      default:
        return (
          <div className="space-y-6">
            {/* Welcome Section */}
            <Card>
              <CardHeader>
                <CardTitle>🌹 Welcome to the Gary×Taleb Education Hub</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-gray-600">
                    This educational system combines Gary Antonacci's trading methodologies with
                    Nassim Taleb's antifragility principles, enhanced by learning techniques from
                    Guild of the Rose's systematic self-improvement approach.
                  </p>

                  {/* Overall Progress */}
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-semibold">Your Learning Progress</h3>
                      <span className="text-lg font-bold text-blue-600">{overallProgress.toFixed(0)}%</span>
                    </div>
                    <Progress value={overallProgress} className="mb-2" />
                    <p className="text-sm text-gray-600">
                      {modules.filter(m => m.progress >= 80).length} of {modules.length} modules mastered
                    </p>
                  </div>

                  {/* Guild of the Rose Integration */}
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <h3 className="font-semibold mb-2">🌹 Guild of the Rose Integration</h3>
                    <p className="text-sm text-gray-600 mb-3">
                      Our education system is inspired by Guild of the Rose's proven methodologies:
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="font-medium text-sm">🗣️ Babble & Prune</p>
                        <p className="text-xs text-gray-600">Generate many strategies, filter the best</p>
                      </div>
                      <div>
                        <p className="font-medium text-sm">📊 Decision Theory</p>
                        <p className="text-xs text-gray-600">Rational decision-making under uncertainty</p>
                      </div>
                      <div>
                        <p className="font-medium text-sm">🦢 Antifragility</p>
                        <p className="text-xs text-gray-600">Systems that benefit from stress</p>
                      </div>
                      <div>
                        <p className="font-medium text-sm">🎓 Metalearning</p>
                        <p className="text-xs text-gray-600">Learning how to learn effectively</p>
                      </div>
                    </div>
                  </div>

                  {/* Weekly Learning Schedule */}
                  <div className="p-4 bg-green-50 rounded-lg">
                    <h3 className="font-semibold mb-2">📅 Weekly Learning Schedule</h3>
                    <p className="text-sm text-gray-600 mb-3">
                      Synchronized with your trading cycle for optimal learning integration:
                    </p>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Monday: Market Analysis & Decision Theory</span>
                        <Badge variant="outline" className="text-xs">30 min</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Wednesday: Risk Management & Psychology</span>
                        <Badge variant="outline" className="text-xs">45 min</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Friday: Strategy Development & Review</span>
                        <Badge variant="outline" className="text-xs">60 min</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Module Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {modules.map(module => (
                <Card
                  key={module.id}
                  className="cursor-pointer hover:shadow-lg transition-shadow"
                  onClick={() => setActiveModule(module.id)}
                >
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">{module.icon}</span>
                        <div>
                          <CardTitle className="text-lg">{module.title}</CardTitle>
                          <div className="flex gap-2 mt-1">
                            <Badge className={getDifficultyColor(module.difficulty)} variant="secondary">
                              {module.difficulty}
                            </Badge>
                            <Badge className={getCategoryColor(module.category)} variant="secondary">
                              {module.category}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">{module.progress}%</p>
                        <Progress value={module.progress} className="w-20 mt-1" />
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600 mb-3">{module.description}</p>

                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>⏱️ {module.estimatedTime}</span>
                        <span>📚 Interactive</span>
                      </div>

                      <div className="p-2 bg-gray-50 rounded text-xs">
                        <p className="font-medium text-gray-700 mb-1">🌹 Guild Connection:</p>
                        <p className="text-gray-600">{module.guildOfRoseConnection}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Learning Philosophy */}
            <Card>
              <CardHeader>
                <CardTitle>🎯 Learning Philosophy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold mb-2">🔄 Iterative Learning</h4>
                    <p className="text-sm text-gray-600">
                      Set up, try out, iterate. Each module builds on the previous,
                      with continuous refinement of your understanding.
                    </p>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg">
                    <h4 className="font-semibold mb-2">👥 Collaborative Growth</h4>
                    <p className="text-sm text-gray-600">
                      Learn with cohorts of fellow traders, sharing insights and
                      challenging each other's assumptions.
                    </p>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <h4 className="font-semibold mb-2">🧠 Meta-Cognitive Awareness</h4>
                    <p className="text-sm text-gray-600">
                      Develop awareness of your thinking processes, biases, and
                      learning patterns for accelerated growth.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 p-1 bg-gray-100 rounded-lg">
        <button
          onClick={() => setActiveModule('overview')}
          className={`px-4 py-2 rounded-md transition-colors text-sm ${
            activeModule === 'overview' ? 'bg-white shadow' : ''
          }`}
        >
          📚 Overview
        </button>
        {modules.map(module => (
          <button
            key={module.id}
            onClick={() => setActiveModule(module.id)}
            className={`px-3 py-2 rounded-md transition-colors text-sm ${
              activeModule === module.id ? 'bg-white shadow' : ''
            }`}
          >
            <span className="mr-1">{module.icon}</span>
            {module.title.split(' ')[0]}
          </button>
        ))}
      </div>

      {/* Module Content */}
      {renderModuleContent()}
    </div>
  );
};