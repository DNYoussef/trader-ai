/**
 * Tabbed Dashboard - Main dashboard with multiple tabs including trading terminal
 *
 * Integrates all enhanced UX components into a unified tabbed interface,
 * with the professional trading terminal as one of the tabs.
 */

import React, { useState, useEffect } from 'react';
import { useEnhancedUX } from './EnhancedUXProvider';

// Import all enhanced components
import OnboardingWizard from './OnboardingWizard';
import CausalEducationHub from './CausalEducationHub';
import ReinforcementCenter from './ReinforcementCenter';
import TradingTerminal from './TradingTerminal';
import ProgressCelebration from './ProgressCelebration';

interface TabConfig {
  id: string;
  title: string;
  icon: string;
  component: React.ReactNode;
  badge?: string;
  requiresAuth?: boolean;
}

interface TabbedDashboardProps {
  userId: string;
  userLevel?: string;
  defaultTab?: string;
}

const TabbedDashboard: React.FC<TabbedDashboardProps> = ({
  userId,
  userLevel = 'beginner',
  defaultTab = 'overview'
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [userProgress, setUserProgress] = useState({
    onboardingComplete: false,
    tradingExperience: 'beginner',
    conceptsLearned: 0,
    totalTrades: 0,
    currentGate: 'G0'
  });

  const { triggerSystemEvent } = useEnhancedUX();

  useEffect(() => {
    loadUserProgress();
  }, [userId]);

  const loadUserProgress = async () => {
    // In production, this would load from backend
    const mockProgress = {
      onboardingComplete: localStorage.getItem(`onboarding_${userId}`) === 'complete',
      tradingExperience: localStorage.getItem(`experience_${userId}`) || 'beginner',
      conceptsLearned: parseInt(localStorage.getItem(`concepts_${userId}`) || '0'),
      totalTrades: parseInt(localStorage.getItem(`trades_${userId}`) || '0'),
      currentGate: localStorage.getItem(`gate_${userId}`) || 'G0'
    };

    setUserProgress(mockProgress);

    if (!mockProgress.onboardingComplete) {
      setShowOnboarding(true);
    }
  };

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);

    triggerSystemEvent('tab_changed', {
      from_tab: activeTab,
      to_tab: tabId,
      user_level: userLevel
    });

    // Special handling for trading terminal access
    if (tabId === 'terminal') {
      triggerSystemEvent('trading_terminal_accessed', {
        user_gate: userProgress.currentGate,
        user_experience: userProgress.tradingExperience
      });
    }
  };

  const handleOnboardingComplete = (results: any) => {
    setUserProgress(prev => ({
      ...prev,
      onboardingComplete: true,
      tradingExperience: results.persona || 'beginner'
    }));

    localStorage.setItem(`onboarding_${userId}`, 'complete');
    localStorage.setItem(`experience_${userId}`, results.persona || 'beginner');

    setShowOnboarding(false);
    setActiveTab('overview');

    triggerSystemEvent('onboarding_completed', {
      persona: results.persona,
      user_id: userId
    });
  };

  // Define all available tabs
  const tabs: TabConfig[] = [
    {
      id: 'overview',
      title: 'Overview',
      icon: '📊',
      component: (
        <div className="p-6">
          <h2 className="text-2xl font-bold mb-4">Trading System Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="font-semibold text-gray-800 mb-2">Current Gate</h3>
              <div className="text-3xl font-bold text-blue-600">{userProgress.currentGate}</div>
              <p className="text-sm text-gray-600 mt-1">Capital progression level</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="font-semibold text-gray-800 mb-2">Total Trades</h3>
              <div className="text-3xl font-bold text-green-600">{userProgress.totalTrades}</div>
              <p className="text-sm text-gray-600 mt-1">Executed trades</p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="font-semibold text-gray-800 mb-2">Concepts Learned</h3>
              <div className="text-3xl font-bold text-purple-600">{userProgress.conceptsLearned}</div>
              <p className="text-sm text-gray-600 mt-1">Educational progress</p>
            </div>
          </div>

          <div className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Quick Actions</h3>
            <div className="flex space-x-4">
              <button
                onClick={() => setActiveTab('terminal')}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              >
                Open Trading Terminal
              </button>
              <button
                onClick={() => setActiveTab('education')}
                className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700"
              >
                Continue Learning
              </button>
              <button
                onClick={() => setActiveTab('engagement')}
                className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700"
              >
                View Progress
              </button>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'terminal',
      title: 'Trading Terminal',
      icon: '📈',
      component: <TradingTerminal symbols={['SPY', 'ULTY', 'AMDY', 'VTIP', 'IAU']} />,
      badge: 'PRO'
    },
    {
      id: 'education',
      title: 'Learn',
      icon: '🧠',
      component: (
        <CausalEducationHub
          userId={userId}
          userLevel={userLevel}
          onConceptComplete={(conceptId) => {
            setUserProgress(prev => ({
              ...prev,
              conceptsLearned: prev.conceptsLearned + 1
            }));
            localStorage.setItem(`concepts_${userId}`, (userProgress.conceptsLearned + 1).toString());
          }}
        />
      )
    },
    {
      id: 'engagement',
      title: 'Progress',
      icon: '🎯',
      component: (
        <ReinforcementCenter
          userId={userId}
          onEventInteraction={(eventId, interaction) => {
            triggerSystemEvent('reinforcement_interaction', {
              event_id: eventId,
              interaction_type: interaction
            });
          }}
          onSocialShare={(eventId, platform) => {
            triggerSystemEvent('social_share', {
              event_id: eventId,
              platform: platform
            });
          }}
        />
      )
    }
  ];

  // Filter tabs based on user progress
  const availableTabs = tabs.filter(tab => {
    if (tab.id === 'terminal' && userProgress.currentGate === 'G0' && userProgress.totalTrades < 5) {
      return false; // Require some trading experience for terminal
    }
    return true;
  });

  const activeTabConfig = availableTabs.find(tab => tab.id === activeTab) || availableTabs[0];

  if (showOnboarding) {
    return (
      <div className="min-h-screen bg-gray-100">
        <OnboardingWizard
          onComplete={handleOnboardingComplete}
          onClose={() => setShowOnboarding(false)}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-full">
          <div className="flex space-x-0">
            {availableTabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => handleTabChange(tab.id)}
                className={`
                  flex items-center px-6 py-4 text-sm font-medium border-b-2 transition-colors
                  ${activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 bg-blue-50'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }
                `}
              >
                <span className="mr-2 text-base">{tab.icon}</span>
                <span>{tab.title}</span>
                {tab.badge && (
                  <span className="ml-2 px-2 py-1 text-xs bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-full">
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1">
        {activeTabConfig.component}
      </div>

      {/* Global Progress Celebration Overlay */}
      <ProgressCelebration />
    </div>
  );
};

export default TabbedDashboard;