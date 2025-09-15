import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import OnboardingWizard from './OnboardingWizard';
import ValueScreens from './ValueScreens';
import HumanizedAlerts, { createHumanizedAlert } from './HumanizedAlerts';
import ProgressCelebration, { createAchievement } from './ProgressCelebration';

interface UserProfile {
  id: string;
  persona: 'beginner' | 'casual_investor' | 'active_trader' | 'experienced_trader';
  onboardingCompleted: boolean;
  painPoints: string[];
  goals: string[];
  preferences: {
    celebrationStyle: 'full' | 'minimal' | 'off';
    alertStyle: 'human' | 'technical' | 'mixed';
    guidanceLevel: 'high' | 'medium' | 'low';
  };
  progressMetrics: {
    currentGate: string;
    totalProfits: number;
    tradingStreakDays: number;
    achievementsUnlocked: number;
  };
}

interface EnhancedUXState {
  userProfile: UserProfile | null;
  showOnboarding: boolean;
  showValueScreens: boolean;
  currentCelebration: any | null;
  alerts: any[];
  isInitialized: boolean;
}

interface EnhancedUXContextType {
  state: EnhancedUXState;
  actions: {
    startOnboarding: () => void;
    completeOnboarding: (result: any) => void;
    showCelebration: (achievement: any) => void;
    addAlert: (alert: any) => void;
    dismissAlert: (alertId: string) => void;
    updateUserProfile: (updates: Partial<UserProfile>) => void;
    triggerSystemEvent: (eventType: string, data: any) => void;
  };
}

const EnhancedUXContext = createContext<EnhancedUXContextType | null>(null);

export const useEnhancedUX = () => {
  const context = useContext(EnhancedUXContext);
  if (!context) {
    throw new Error('useEnhancedUX must be used within EnhancedUXProvider');
  }
  return context;
};

interface EnhancedUXProviderProps {
  children: ReactNode;
  initialUserProfile?: Partial<UserProfile>;
  tradingSystemData?: any;
}

export const EnhancedUXProvider: React.FC<EnhancedUXProviderProps> = ({
  children,
  initialUserProfile,
  tradingSystemData
}) => {
  const [state, setState] = useState<EnhancedUXState>({
    userProfile: null,
    showOnboarding: false,
    showValueScreens: false,
    currentCelebration: null,
    alerts: [],
    isInitialized: false
  });

  useEffect(() => {
    initializeUX();
  }, [initialUserProfile]);

  useEffect(() => {
    // Listen to trading system events
    if (tradingSystemData) {
      handleTradingSystemUpdates(tradingSystemData);
    }
  }, [tradingSystemData]);

  const initializeUX = () => {
    const defaultProfile: UserProfile = {
      id: 'user_001',
      persona: 'casual_investor',
      onboardingCompleted: false,
      painPoints: [],
      goals: [],
      preferences: {
        celebrationStyle: 'full',
        alertStyle: 'human',
        guidanceLevel: 'medium'
      },
      progressMetrics: {
        currentGate: 'G0',
        totalProfits: 0,
        tradingStreakDays: 0,
        achievementsUnlocked: 0
      },
      ...initialUserProfile
    };

    setState(prev => ({
      ...prev,
      userProfile: defaultProfile,
      showOnboarding: !defaultProfile.onboardingCompleted,
      isInitialized: true
    }));
  };

  const handleTradingSystemUpdates = (data: any) => {
    // Process trading system updates and convert to UX events
    if (data.gateProgression) {
      triggerSystemEvent('gate_progression', data.gateProgression);
    }

    if (data.profitUpdate) {
      if (data.profitUpdate.newMilestone) {
        triggerSystemEvent('profit_milestone', data.profitUpdate);
      }
    }

    if (data.riskAlerts) {
      data.riskAlerts.forEach((alert: any) => {
        const humanizedAlert = createHumanizedAlert(alert, state.userProfile?.persona || 'casual_investor');
        addAlert(humanizedAlert);
      });
    }
  };

  const startOnboarding = () => {
    setState(prev => ({
      ...prev,
      showOnboarding: true
    }));
  };

  const completeOnboarding = (result: any) => {
    const updatedProfile: UserProfile = {
      ...state.userProfile!,
      onboardingCompleted: true,
      persona: result.persona,
      painPoints: result.painPoints,
      goals: result.goals || []
    };

    setState(prev => ({
      ...prev,
      userProfile: updatedProfile,
      showOnboarding: false,
      showValueScreens: true
    }));

    // Trigger welcome celebration
    const welcomeAchievement = {
      id: 'onboarding_complete',
      title: 'Welcome to Gary×Taleb Trading!',
      description: 'You\'ve completed your personalized setup',
      type: 'learning',
      value: 'Setup Complete',
      unlockedFeatures: [
        'Personalized dashboard',
        'AI-guided trading decisions',
        'Real-time risk monitoring'
      ],
      rarity: 'common',
      earnedAt: new Date(),
      celebrationStyle: 'gentle'
    };

    setTimeout(() => showCelebration(welcomeAchievement), 1000);
  };

  const showCelebration = (achievement: any) => {
    if (state.userProfile?.preferences.celebrationStyle === 'off') {
      return;
    }

    setState(prev => ({
      ...prev,
      currentCelebration: achievement
    }));
  };

  const addAlert = (alert: any) => {
    setState(prev => ({
      ...prev,
      alerts: [...prev.alerts, { ...alert, id: alert.id || Date.now().toString() }]
    }));
  };

  const dismissAlert = (alertId: string) => {
    setState(prev => ({
      ...prev,
      alerts: prev.alerts.filter(alert => alert.id !== alertId)
    }));
  };

  const updateUserProfile = (updates: Partial<UserProfile>) => {
    setState(prev => ({
      ...prev,
      userProfile: prev.userProfile ? { ...prev.userProfile, ...updates } : null
    }));
  };

  const triggerSystemEvent = (eventType: string, data: any) => {
    const persona = state.userProfile?.persona || 'casual_investor';

    try {
      switch (eventType) {
        case 'gate_progression':
          const gateAchievement = createAchievement('gate_progression', data, persona);
          showCelebration(gateAchievement);

          updateUserProfile({
            progressMetrics: {
              ...state.userProfile!.progressMetrics,
              currentGate: data.newGate,
              achievementsUnlocked: state.userProfile!.progressMetrics.achievementsUnlocked + 1
            }
          });
          break;

        case 'profit_milestone':
          const profitAchievement = createAchievement('profit_milestone', data, persona);
          showCelebration(profitAchievement);

          updateUserProfile({
            progressMetrics: {
              ...state.userProfile!.progressMetrics,
              totalProfits: data.profit,
              achievementsUnlocked: state.userProfile!.progressMetrics.achievementsUnlocked + 1
            }
          });
          break;

        case 'trading_streak':
          const streakAchievement = createAchievement('trading_streak', data, persona);
          if (data.days % 7 === 0 || data.days >= 30) { // Celebrate weekly milestones and 30+ days
            showCelebration(streakAchievement);
          }

          updateUserProfile({
            progressMetrics: {
              ...state.userProfile!.progressMetrics,
              tradingStreakDays: data.days
            }
          });
          break;

        case 'risk_alert':
          const riskAlert = createHumanizedAlert({
            type: 'risk',
            severity: data.severity || 'medium',
            title: data.title || 'Risk Alert',
            message: data.message,
            technicalDetails: data.details,
            actionRequired: data.actionRequired,
            actionText: data.actionText,
            relatedData: data.metrics
          }, persona);
          addAlert(riskAlert);
          break;

        case 'opportunity_alert':
          const opportunityAlert = createHumanizedAlert({
            type: 'opportunity',
            severity: 'medium',
            title: data.title || 'Trading Opportunity',
            message: data.message,
            technicalDetails: data.details,
            actionRequired: true,
            actionText: 'Review Opportunity',
            relatedData: data.metrics
          }, persona);
          addAlert(opportunityAlert);
          break;

        default:
          console.warn(`Unknown system event type: ${eventType}`);
      }
    } catch (error) {
      console.error(`Error handling system event ${eventType}:`, error);
    }
  };

  const actions = {
    startOnboarding,
    completeOnboarding,
    showCelebration,
    addAlert,
    dismissAlert,
    updateUserProfile,
    triggerSystemEvent
  };

  return (
    <EnhancedUXContext.Provider value={{ state, actions }}>
      {children}

      {/* Onboarding Wizard */}
      {state.showOnboarding && state.userProfile && (
        <OnboardingWizard
          onComplete={completeOnboarding}
          onClose={() => setState(prev => ({ ...prev, showOnboarding: false }))}
        />
      )}

      {/* Value Screens */}
      {state.showValueScreens && state.userProfile && (
        <div className="fixed inset-0 bg-white z-40 overflow-y-auto">
          <div className="min-h-screen py-8">
            <ValueScreens
              persona={state.userProfile.persona}
              painPoints={state.userProfile.painPoints}
              goals={state.userProfile.goals}
              onContinue={() => setState(prev => ({ ...prev, showValueScreens: false }))}
              onSkip={() => setState(prev => ({ ...prev, showValueScreens: false }))}
            />
          </div>
        </div>
      )}

      {/* Progress Celebration */}
      {state.currentCelebration && state.userProfile && (
        <ProgressCelebration
          achievement={state.currentCelebration}
          persona={state.userProfile.persona}
          onClose={() => setState(prev => ({ ...prev, currentCelebration: null }))}
          onShare={() => {
            // Handle sharing logic
            console.log('Sharing achievement:', state.currentCelebration);
          }}
          onContinue={() => setState(prev => ({ ...prev, currentCelebration: null }))}
        />
      )}
    </EnhancedUXContext.Provider>
  );
};

// Hook for components to integrate with the trading system
export const useUXIntegration = () => {
  const { actions } = useEnhancedUX();

  return {
    // Easy methods for trading system to trigger UX events
    onGateProgression: (data: any) => actions.triggerSystemEvent('gate_progression', data),
    onProfitMilestone: (data: any) => actions.triggerSystemEvent('profit_milestone', data),
    onTradingStreak: (data: any) => actions.triggerSystemEvent('trading_streak', data),
    onRiskAlert: (data: any) => actions.triggerSystemEvent('risk_alert', data),
    onOpportunity: (data: any) => actions.triggerSystemEvent('opportunity_alert', data),

    // User preference updates
    updatePreferences: (prefs: any) => actions.updateUserProfile({ preferences: prefs }),

    // Manual triggers
    showOnboarding: actions.startOnboarding,
    celebrate: actions.showCelebration,
    alert: actions.addAlert
  };
};

export default EnhancedUXProvider;