import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';

interface Skill {
  id: string;
  name: string;
  level: number;
  maxLevel: number;
  experience: number;
  experienceToNext: number;
  category: 'risk' | 'strategy' | 'psychology' | 'technical' | 'meta';
  description: string;
  unlocks: string[];
  prerequisites: string[];
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  dateEarned?: Date;
  type: 'milestone' | 'streak' | 'mastery' | 'special';
}

interface EnergySystem {
  current: number;
  max: number;
  regenRate: number;
  lastRegen: Date;
  activities: {
    trading: number;
    learning: number;
    analyzing: number;
    socializing: number;
  };
}

export const CharacterSheet: React.FC = () => {
  const [activeSkillTree, setActiveSkillTree] = useState<'risk' | 'strategy' | 'psychology' | 'technical' | 'meta'>('risk');
  const [availableSkillPoints, setAvailableSkillPoints] = useState(5);

  const [skills, setSkills] = useState<Skill[]>([
    // Risk Management Tree
    {
      id: 'position_sizing',
      name: 'Position Sizing',
      level: 3,
      maxLevel: 10,
      experience: 750,
      experienceToNext: 250,
      category: 'risk',
      description: 'Master the art of determining optimal position sizes',
      unlocks: ['kelly_criterion', 'volatility_scaling'],
      prerequisites: []
    },
    {
      id: 'kelly_criterion',
      name: 'Kelly Criterion',
      level: 1,
      maxLevel: 5,
      experience: 200,
      experienceToNext: 300,
      category: 'risk',
      description: 'Mathematical approach to position sizing',
      unlocks: ['advanced_kelly'],
      prerequisites: ['position_sizing']
    },
    {
      id: 'stop_loss_mastery',
      name: 'Stop Loss Mastery',
      level: 2,
      maxLevel: 8,
      experience: 400,
      experienceToNext: 200,
      category: 'risk',
      description: 'Advanced stop loss strategies and implementation',
      unlocks: ['trailing_stops', 'volatility_stops'],
      prerequisites: []
    },

    // Strategy Development Tree
    {
      id: 'momentum_strategies',
      name: 'Momentum Strategies',
      level: 4,
      maxLevel: 10,
      experience: 850,
      experienceToNext: 150,
      category: 'strategy',
      description: 'Gary Antonacci\'s dual momentum approach',
      unlocks: ['relative_momentum', 'absolute_momentum'],
      prerequisites: []
    },
    {
      id: 'mean_reversion',
      name: 'Mean Reversion',
      level: 2,
      maxLevel: 8,
      experience: 350,
      experienceToNext: 250,
      category: 'strategy',
      description: 'Counter-trend trading strategies',
      unlocks: ['pairs_trading', 'statistical_arbitrage'],
      prerequisites: []
    },

    // Psychology Tree
    {
      id: 'emotional_control',
      name: 'Emotional Control',
      level: 5,
      maxLevel: 10,
      experience: 1200,
      experienceToNext: 300,
      category: 'psychology',
      description: 'Manage emotions during volatile markets',
      unlocks: ['mindfulness_trading', 'loss_acceptance'],
      prerequisites: []
    },
    {
      id: 'discipline',
      name: 'Trading Discipline',
      level: 3,
      maxLevel: 10,
      experience: 600,
      experienceToNext: 400,
      category: 'psychology',
      description: 'Stick to your trading plan consistently',
      unlocks: ['routine_mastery', 'rule_following'],
      prerequisites: []
    },

    // Technical Analysis Tree
    {
      id: 'chart_reading',
      name: 'Chart Reading',
      level: 6,
      maxLevel: 10,
      experience: 1500,
      experienceToNext: 200,
      category: 'technical',
      description: 'Interpret price action and patterns',
      unlocks: ['advanced_patterns', 'volume_analysis'],
      prerequisites: []
    },

    // Meta Learning Tree
    {
      id: 'metacognition',
      name: 'Metacognition',
      level: 2,
      maxLevel: 8,
      experience: 300,
      experienceToNext: 200,
      category: 'meta',
      description: 'Thinking about your thinking processes',
      unlocks: ['learning_optimization', 'cognitive_biases'],
      prerequisites: []
    }
  ]);

  const [achievements, setAchievements] = useState<Achievement[]>([
    {
      id: 'first_quest',
      title: 'Quest Initiate',
      description: 'Completed your first trading quest',
      icon: '🎯',
      dateEarned: new Date('2024-01-15'),
      type: 'milestone'
    },
    {
      id: 'streak_7',
      title: 'Week Warrior',
      description: 'Maintained trading discipline for 7 consecutive days',
      icon: '🔥',
      dateEarned: new Date('2024-01-20'),
      type: 'streak'
    },
    {
      id: 'risk_master',
      title: 'Risk Management Master',
      description: 'Reached level 5 in all risk management skills',
      icon: '🛡️',
      type: 'mastery'
    },
    {
      id: 'antifragile',
      title: 'Antifragile Architect',
      description: 'Built a portfolio that benefits from volatility',
      icon: '🦢',
      dateEarned: new Date('2024-01-25'),
      type: 'special'
    }
  ]);

  const [energy, setEnergy] = useState<EnergySystem>({
    current: 85,
    max: 100,
    regenRate: 1, // per hour
    lastRegen: new Date(),
    activities: {
      trading: 75,
      learning: 90,
      analyzing: 80,
      socializing: 65
    }
  });

  const skillsByCategory = skills.filter(skill => skill.category === activeSkillTree);

  const levelUpSkill = (skillId: string) => {
    if (availableSkillPoints <= 0) return;

    setSkills(prev => prev.map(skill => {
      if (skill.id === skillId && skill.level < skill.maxLevel) {
        const prerequisites = skill.prerequisites.every(prereq =>
          skills.find(s => s.id === prereq)?.level ?? 0 > 0
        );

        if (prerequisites) {
          setAvailableSkillPoints(p => p - 1);
          return {
            ...skill,
            level: skill.level + 1,
            experience: 0,
            experienceToNext: skill.level < skill.maxLevel - 1 ? (skill.level + 1) * 100 : 0
          };
        }
      }
      return skill;
    }));
  };

  const getSkillTreeIcon = (category: string) => {
    switch (category) {
      case 'risk': return '🛡️';
      case 'strategy': return '📈';
      case 'psychology': return '🧠';
      case 'technical': return '📊';
      case 'meta': return '🎓';
      default: return '⭐';
    }
  };

  const getSkillTreeColor = (category: string) => {
    switch (category) {
      case 'risk': return 'text-green-600 border-green-200 bg-green-50';
      case 'strategy': return 'text-blue-600 border-blue-200 bg-blue-50';
      case 'psychology': return 'text-purple-600 border-purple-200 bg-purple-50';
      case 'technical': return 'text-orange-600 border-orange-200 bg-orange-50';
      case 'meta': return 'text-indigo-600 border-indigo-200 bg-indigo-50';
      default: return 'text-gray-600 border-gray-200 bg-gray-50';
    }
  };

  const calculateTotalLevel = () => {
    return skills.reduce((sum, skill) => sum + skill.level, 0);
  };

  const getEnergyColor = (value: number) => {
    if (value >= 80) return 'text-green-600';
    if (value >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Character Overview */}
      <Card>
        <CardHeader>
          <CardTitle>📜 Trading Character Sheet</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-6 mb-6">
            {/* Character Stats */}
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-3xl font-bold text-blue-600">{calculateTotalLevel()}</p>
              <p className="text-sm text-gray-600">Total Level</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-3xl font-bold text-green-600">{availableSkillPoints}</p>
              <p className="text-sm text-gray-600">Skill Points</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <p className="text-3xl font-bold text-purple-600">{achievements.filter(a => a.dateEarned).length}</p>
              <p className="text-sm text-gray-600">Achievements</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <p className={`text-3xl font-bold ${getEnergyColor(energy.current)}`}>{energy.current}</p>
              <p className="text-sm text-gray-600">Energy</p>
            </div>
          </div>

          {/* Energy Management */}
          <div className="p-4 bg-gray-50 rounded-lg mb-6">
            <h4 className="font-semibold mb-3">⚡ Energy Management</h4>
            <p className="text-sm text-gray-600 mb-3">
              Inspired by Guild of the Rose's energy awareness training. Monitor your energy for optimal performance.
            </p>
            <div className="grid grid-cols-4 gap-4">
              {Object.entries(energy.activities).map(([activity, level]) => (
                <div key={activity} className="text-center">
                  <p className="text-xs text-gray-500 capitalize">{activity}</p>
                  <p className={`text-lg font-bold ${getEnergyColor(level)}`}>{level}%</p>
                  <Progress value={level} className="mt-1" />
                </div>
              ))}
            </div>
            <div className="mt-4 p-3 bg-white rounded border-l-4 border-blue-400">
              <p className="text-sm">
                <strong>💡 Energy Tip:</strong> Your learning energy is high ({energy.activities.learning}%).
                This is optimal for skill development and educational quests.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Skill Trees */}
      <Card>
        <CardHeader>
          <CardTitle>🌳 Skill Trees</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Skill Tree Navigation */}
          <div className="flex gap-2 mb-6 p-1 bg-gray-100 rounded-lg">
            {(['risk', 'strategy', 'psychology', 'technical', 'meta'] as const).map(category => (
              <button
                key={category}
                onClick={() => setActiveSkillTree(category)}
                className={`flex-1 py-2 px-3 rounded-md transition-colors text-sm ${
                  activeSkillTree === category
                    ? 'bg-white shadow'
                    : ''
                }`}
              >
                <span className="mr-2">{getSkillTreeIcon(category)}</span>
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </button>
            ))}
          </div>

          {/* Active Skill Tree */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <span>{getSkillTreeIcon(activeSkillTree)}</span>
              {activeSkillTree.charAt(0).toUpperCase() + activeSkillTree.slice(1)} Skills
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {skillsByCategory.map(skill => {
                const isUnlocked = skill.prerequisites.every(prereq =>
                  skills.find(s => s.id === prereq)?.level ?? 0 > 0
                );
                const canLevelUp = isUnlocked && skill.level < skill.maxLevel && availableSkillPoints > 0;

                return (
                  <div
                    key={skill.id}
                    className={`p-4 border-2 rounded-lg ${
                      isUnlocked ? getSkillTreeColor(skill.category) : 'bg-gray-100 border-gray-300'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <h4 className={`font-semibold ${isUnlocked ? '' : 'text-gray-500'}`}>
                          {skill.name}
                        </h4>
                        <p className={`text-sm mt-1 ${isUnlocked ? 'text-gray-600' : 'text-gray-400'}`}>
                          {skill.description}
                        </p>
                      </div>
                      <div className="text-right">
                        <Badge variant={isUnlocked ? 'default' : 'secondary'}>
                          Lv. {skill.level}/{skill.maxLevel}
                        </Badge>
                      </div>
                    </div>

                    {/* Experience Bar */}
                    {isUnlocked && (
                      <div className="mb-3">
                        <div className="flex justify-between text-xs text-gray-600 mb-1">
                          <span>XP: {skill.experience}</span>
                          <span>Next: {skill.experienceToNext}</span>
                        </div>
                        <Progress
                          value={skill.experienceToNext > 0 ? (skill.experience / (skill.experience + skill.experienceToNext)) * 100 : 100}
                          className="h-2"
                        />
                      </div>
                    )}

                    {/* Prerequisites */}
                    {skill.prerequisites.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs text-gray-500 mb-1">Prerequisites:</p>
                        <div className="flex flex-wrap gap-1">
                          {skill.prerequisites.map(prereq => {
                            const prereqSkill = skills.find(s => s.id === prereq);
                            const isMet = (prereqSkill?.level ?? 0) > 0;
                            return (
                              <Badge
                                key={prereq}
                                variant={isMet ? 'default' : 'secondary'}
                                className="text-xs"
                              >
                                {prereqSkill?.name}
                              </Badge>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Unlocks */}
                    {skill.unlocks.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs text-gray-500 mb-1">Unlocks:</p>
                        <div className="flex flex-wrap gap-1">
                          {skill.unlocks.map(unlock => (
                            <Badge key={unlock} variant="outline" className="text-xs">
                              {unlock.replace('_', ' ')}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Level Up Button */}
                    <Button
                      onClick={() => levelUpSkill(skill.id)}
                      disabled={!canLevelUp}
                      size="sm"
                      className="w-full"
                    >
                      {!isUnlocked ? 'Locked' :
                       skill.level >= skill.maxLevel ? 'Maxed' :
                       availableSkillPoints <= 0 ? 'No Skill Points' :
                       'Level Up'}
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Achievements */}
      <Card>
        <CardHeader>
          <CardTitle>🏆 Achievements</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {achievements.map(achievement => (
              <div
                key={achievement.id}
                className={`p-4 border rounded-lg ${
                  achievement.dateEarned
                    ? 'bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200'
                    : 'bg-gray-50 border-gray-200 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">{achievement.icon}</span>
                  <div className="flex-1">
                    <h4 className={`font-semibold ${achievement.dateEarned ? '' : 'text-gray-500'}`}>
                      {achievement.title}
                    </h4>
                    <Badge variant="outline" className="text-xs">
                      {achievement.type}
                    </Badge>
                  </div>
                </div>
                <p className={`text-sm ${achievement.dateEarned ? 'text-gray-600' : 'text-gray-400'}`}>
                  {achievement.description}
                </p>
                {achievement.dateEarned && (
                  <p className="text-xs text-gray-500 mt-2">
                    Earned: {achievement.dateEarned.toLocaleDateString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};