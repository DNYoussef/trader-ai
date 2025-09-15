import React, { useState } from 'react';

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

export const EducationHubFull: React.FC = () => {
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
      case 'systems': return 'bg-teal-100 text-teal-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  // Filter modules by category for organization
  const categorizedModules = {
    core: modules.filter(m => m.category === 'core'),
    advanced: modules.filter(m => m.category === 'advanced'),
    psychology: modules.filter(m => m.category === 'psychology'),
    systems: modules.filter(m => m.category === 'systems')
  };

  if (activeModule !== 'overview') {
    const selectedModule = modules.find(m => m.id === activeModule);
    if (selectedModule) {
      return (
        <div className="space-y-6">
          {/* Back button */}
          <button
            onClick={() => setActiveModule('overview')}
            className="flex items-center text-blue-600 hover:text-blue-800"
          >
            ← Back to Education Hub
          </button>

          {/* Module header */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{selectedModule.icon}</span>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    {selectedModule.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400 mt-1">
                    {selectedModule.description}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(selectedModule.difficulty)}`}>
                  {selectedModule.difficulty}
                </span>
                <p className="text-sm text-gray-500 mt-1">{selectedModule.estimatedTime}</p>
              </div>
            </div>

            {/* Progress bar */}
            <div className="mt-4">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Progress</span>
                <span>{selectedModule.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${selectedModule.progress}%` }}
                ></div>
              </div>
            </div>

            {/* Guild connection */}
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
              <h4 className="font-medium text-blue-900 dark:text-blue-100 text-sm">
                Guild of the Rose Connection
              </h4>
              <p className="text-blue-700 dark:text-blue-300 text-sm mt-1">
                {selectedModule.guildOfRoseConnection}
              </p>
            </div>
          </div>

          {/* Module content */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            {selectedModule.id === 'decision_theory' && <DecisionTheoryModule />}
            {selectedModule.id === 'antifragility' && <AntifragilityModule />}
            {selectedModule.id === 'quest_system' && <QuestSystemModule />}
            {selectedModule.id === 'character_sheet' && <CharacterSheetModule />}
            {selectedModule.id === 'time_management' && <TimeManagementModule />}
          </div>
        </div>
      );
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Guild of the Rose Education Hub
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Interactive learning modules integrating Matt Freeman's rational decision theory with Gary×Taleb trading methodology.
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
          <p className="text-3xl font-bold text-blue-600">{modules.length}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Learning Modules</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
          <p className="text-3xl font-bold text-green-600">
            {Math.round(modules.reduce((acc, m) => acc + m.progress, 0) / modules.length)}%
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Overall Progress</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
          <p className="text-3xl font-bold text-purple-600">
            {modules.reduce((acc, m) => acc + parseFloat(m.estimatedTime), 0).toFixed(1)}h
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Total Learning Time</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
          <p className="text-3xl font-bold text-yellow-600">
            {modules.filter(m => m.progress > 80).length}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Nearly Complete</p>
        </div>
      </div>

      {/* Module categories */}
      {Object.entries(categorizedModules).map(([category, categoryModules]) => (
        <div key={category} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 capitalize">
            {category} Modules
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {categoryModules.map(module => (
              <div
                key={module.id}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => setActiveModule(module.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{module.icon}</span>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {module.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {module.description}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <div className="flex space-x-2">
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded ${getCategoryColor(module.category)}`}>
                      {module.category}
                    </span>
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(module.difficulty)}`}>
                      {module.difficulty}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">{module.estimatedTime}</span>
                </div>

                {/* Progress bar */}
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Progress</span>
                    <span>{module.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                      style={{ width: `${module.progress}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// Individual module components
const DecisionTheoryModule: React.FC = () => (
  <div className="space-y-6">
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
      Interactive Decision Theory & Risk Calculators
    </h3>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Expected Value Calculator</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Calculate expected values for trading decisions with probability inputs.
        </p>
        <button className="mt-3 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          Launch Calculator
        </button>
      </div>

      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Decision Trees</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Build interactive decision trees for complex trading scenarios.
        </p>
        <button className="mt-3 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
          Build Tree
        </button>
      </div>
    </div>

    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
      <h4 className="font-medium text-blue-900 dark:text-blue-100">Matt Freeman's 5-Step Process</h4>
      <ol className="mt-2 space-y-1 text-sm text-blue-700 dark:text-blue-300">
        <li>1. Define the decision clearly</li>
        <li>2. Generate comprehensive options</li>
        <li>3. Assess probabilities and outcomes</li>
        <li>4. Calculate expected values</li>
        <li>5. Account for uncertainty and update beliefs</li>
      </ol>
    </div>
  </div>
);

const AntifragilityModule: React.FC = () => (
  <div className="space-y-6">
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
      Antifragility & Barbell Strategy Builder
    </h3>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Barbell Portfolio Builder</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Design 80/20 portfolios that benefit from volatility.
        </p>
        <div className="mt-3 space-y-2">
          <div className="flex justify-between text-sm">
            <span>Safe Assets (80%)</span>
            <span className="text-green-600">Treasury Bills, Gold</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Risky Assets (20%)</span>
            <span className="text-red-600">Options, Startups</span>
          </div>
        </div>
      </div>

      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Black Swan Simulator</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Test portfolio resilience against extreme events.
        </p>
        <button className="mt-3 px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700">
          Run Simulation
        </button>
      </div>
    </div>
  </div>
);

const QuestSystemModule: React.FC = () => (
  <div className="space-y-6">
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
      Progressive Quest System
    </h3>
    <p className="text-gray-600 dark:text-gray-400">
      Collaborative learning through structured quests tied to trading gate progression.
    </p>

    <div className="space-y-3">
      {[
        { name: 'Gate G0 Quest', status: 'completed', reward: 'Basic Risk Management Badge' },
        { name: 'Gate G1 Quest', status: 'in_progress', reward: 'Position Sizing Mastery' },
        { name: 'Gate G2 Quest', status: 'locked', reward: 'Advanced Portfolio Theory' }
      ].map((quest, index) => (
        <div key={index} className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-700 rounded">
          <div>
            <h4 className="font-medium text-gray-900 dark:text-white">{quest.name}</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">Reward: {quest.reward}</p>
          </div>
          <span className={`px-2 py-1 text-xs rounded ${
            quest.status === 'completed' ? 'bg-green-100 text-green-700' :
            quest.status === 'in_progress' ? 'bg-yellow-100 text-yellow-700' :
            'bg-gray-100 text-gray-700'
          }`}>
            {quest.status.replace('_', ' ')}
          </span>
        </div>
      ))}
    </div>
  </div>
);

const CharacterSheetModule: React.FC = () => (
  <div className="space-y-6">
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
      Trading Character Sheet & Skill Trees
    </h3>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {[
        { skill: 'Risk Management', level: 8, maxLevel: 10, color: 'blue' },
        { skill: 'Technical Analysis', level: 6, maxLevel: 10, color: 'green' },
        { skill: 'Psychology Control', level: 4, maxLevel: 10, color: 'purple' }
      ].map((skill, index) => (
        <div key={index} className="p-4 border border-gray-200 dark:border-gray-700 rounded">
          <h4 className="font-medium text-gray-900 dark:text-white">{skill.skill}</h4>
          <div className="mt-2">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Level {skill.level}</span>
              <span>{skill.level}/{skill.maxLevel}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`bg-${skill.color}-600 h-2 rounded-full`}
                style={{ width: `${(skill.level / skill.maxLevel) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

const TimeManagementModule: React.FC = () => (
  <div className="space-y-6">
    <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
      Trading Time Management & Energy Optimization
    </h3>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Energy Tracking</h4>
        <div className="mt-3 space-y-2">
          <div className="flex justify-between">
            <span className="text-sm">Mental Energy</span>
            <span className="text-sm font-medium text-green-600">85%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-green-600 h-2 rounded-full" style={{ width: '85%' }}></div>
          </div>
        </div>
      </div>

      <div className="p-4 border border-gray-200 dark:border-gray-700 rounded">
        <h4 className="font-medium text-gray-900 dark:text-white">Optimal Trading Hours</h4>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Based on your energy patterns and market volatility.
        </p>
        <div className="mt-2 text-sm">
          <div className="text-green-600">🟢 9:30-11:00 AM (High Energy)</div>
          <div className="text-yellow-600">🟡 2:00-3:30 PM (Medium Energy)</div>
          <div className="text-red-600">🔴 After 8:00 PM (Low Energy)</div>
        </div>
      </div>
    </div>
  </div>
);