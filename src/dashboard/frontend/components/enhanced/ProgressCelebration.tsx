import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '../../src/components/ui/card';
import { Button } from '../../src/components/ui/button';
import { Badge } from '../../src/components/ui/badge';
import { Progress } from '../../src/components/ui/progress';
import {
  Trophy, Star, TrendingUp, Target, Gift, Sparkles,
  CheckCircle, ArrowRight, Share2, X, Crown, Medal,
  Zap, DollarSign, BarChart3, Users
} from 'lucide-react';
import confetti from 'canvas-confetti';

interface Achievement {
  id: string;
  title: string;
  description: string;
  type: 'gate_progression' | 'profit_milestone' | 'streak' | 'learning' | 'risk_management';
  value: string | number;
  previousValue?: string | number;
  unlockedFeatures?: string[];
  rewardType?: 'feature' | 'recognition' | 'milestone';
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  earnedAt: Date;
  celebrationStyle: 'confetti' | 'sparkles' | 'fireworks' | 'gentle';
}

interface ProgressCelebrationProps {
  achievement: Achievement;
  persona: string;
  onClose: () => void;
  onShare?: () => void;
  onContinue?: () => void;
  showOnboarding?: boolean;
}

const ProgressCelebration: React.FC<ProgressCelebrationProps> = ({
  achievement,
  persona,
  onClose,
  onShare,
  onContinue,
  showOnboarding = false
}) => {
  const [animationPhase, setAnimationPhase] = useState<'entering' | 'celebrating' | 'stable'>('entering');
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    // Trigger celebration animation
    if (achievement.celebrationStyle === 'confetti') {
      const duration = 3000;
      const end = Date.now() + duration;

      const frame = () => {
        confetti({
          particleCount: 2,
          angle: 60,
          spread: 55,
          origin: { x: 0 },
          colors: ['#3B82F6', '#10B981', '#F59E0B']
        });
        confetti({
          particleCount: 2,
          angle: 120,
          spread: 55,
          origin: { x: 1 },
          colors: ['#3B82F6', '#10B981', '#F59E0B']
        });

        if (Date.now() < end) {
          requestAnimationFrame(frame);
        }
      };
      frame();
    }

    // Animation phases
    const enterTimer = setTimeout(() => setAnimationPhase('celebrating'), 200);
    const stableTimer = setTimeout(() => setAnimationPhase('stable'), 1000);

    return () => {
      clearTimeout(enterTimer);
      clearTimeout(stableTimer);
    };
  }, [achievement.celebrationStyle]);

  const getAchievementIcon = (type: string, rarity: string) => {
    const iconMap = {
      gate_progression: rarity === 'legendary' ? Crown : Target,
      profit_milestone: rarity === 'epic' ? Trophy : DollarSign,
      streak: Star,
      learning: Medal,
      risk_management: Shield
    };

    return iconMap[type as keyof typeof iconMap] || Trophy;
  };

  const getRarityColors = (rarity: string) => {
    const colorMap = {
      common: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        icon: 'text-blue-600',
        badge: 'bg-blue-100 text-blue-800'
      },
      rare: {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        icon: 'text-purple-600',
        badge: 'bg-purple-100 text-purple-800'
      },
      epic: {
        bg: 'bg-yellow-50',
        border: 'border-yellow-200',
        icon: 'text-yellow-600',
        badge: 'bg-yellow-100 text-yellow-800'
      },
      legendary: {
        bg: 'bg-gradient-to-br from-yellow-50 to-orange-50',
        border: 'border-orange-300',
        icon: 'text-orange-600',
        badge: 'bg-gradient-to-r from-yellow-400 to-orange-500 text-white'
      }
    };

    return colorMap[rarity as keyof typeof colorMap] || colorMap.common;
  };

  const getPersonalizedMessage = (achievement: Achievement, persona: string) => {
    const messages = {
      beginner: {
        gate_progression: "Amazing work! You've successfully moved to the next level. This shows you're building real trading discipline.",
        profit_milestone: "Congratulations on your first real profits! This proves the system works when you stick to it.",
        streak: "You're building incredible consistency! Each day you follow the system, you're becoming a better trader.",
        learning: "Your dedication to learning is paying off! Understanding these concepts will make you much more successful.",
        risk_management: "Excellent risk management! Protecting your capital is the most important skill in trading."
      },
      casual_investor: {
        gate_progression: "Great progress! You've unlocked new trading capabilities and proven your systematic approach works.",
        profit_milestone: "Solid returns! Your patient, systematic approach is delivering the results you wanted.",
        streak: "Impressive consistency! You're proving that systematic trading beats emotional decisions.",
        learning: "Your expanding knowledge is showing in your results. Keep building on this foundation.",
        risk_management: "Smart risk control! This is what separates successful traders from everyone else."
      },
      active_trader: {
        gate_progression: "Outstanding performance! You've earned access to advanced strategies and higher position limits.",
        profit_milestone: "Excellent execution! Your systematic approach is generating alpha consistently.",
        streak: "Remarkable discipline! This consistency is the hallmark of professional trading.",
        learning: "Your analytical skills are clearly advancing. This knowledge edge will compound over time.",
        risk_management: "Professional-level risk management! You understand that preservation of capital is paramount."
      }
    };

    const personaMessages = messages[persona as keyof typeof messages] || messages.casual_investor;
    return personaMessages[achievement.type as keyof typeof personaMessages] || "Congratulations on this achievement!";
  };

  const getNextSteps = (achievement: Achievement, persona: string) => {
    if (achievement.type === 'gate_progression') {
      return [
        "Explore your new trading capabilities",
        "Review updated position size limits",
        "Consider diversifying with new asset classes"
      ];
    }

    if (achievement.type === 'profit_milestone') {
      return [
        "Consider taking some profits (50/50 split)",
        "Review what strategies worked best",
        "Set your next profit target"
      ];
    }

    return [
      "Keep following your systematic approach",
      "Continue building good habits",
      "Stay focused on long-term success"
    ];
  };

  const IconComponent = getAchievementIcon(achievement.type, achievement.rarity);
  const colors = getRarityColors(achievement.rarity);
  const personalizedMessage = getPersonalizedMessage(achievement, persona);
  const nextSteps = getNextSteps(achievement, persona);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className={`
        w-full max-w-lg mx-4 transform transition-all duration-500
        ${animationPhase === 'entering' ? 'scale-95 opacity-0' : 'scale-100 opacity-100'}
        ${colors.bg} ${colors.border} border-2
      `}>
        <CardHeader className="text-center pb-2">
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <Badge className={`${colors.badge} mb-2 text-xs font-medium`}>
                {achievement.rarity.toUpperCase()} ACHIEVEMENT
              </Badge>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose} className="p-1">
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Achievement Icon */}
          <div className={`
            mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4
            ${colors.bg} ${colors.border} border-2
            ${animationPhase === 'celebrating' ? 'animate-pulse' : ''}
          `}>
            <IconComponent className={`w-8 h-8 ${colors.icon}`} />
            {achievement.rarity === 'legendary' && (
              <Sparkles className="w-4 h-4 absolute -top-1 -right-1 text-yellow-400" />
            )}
          </div>

          <h2 className="text-xl font-bold text-gray-800 mb-1">
            {achievement.title}
          </h2>
          <p className="text-gray-600 text-sm">
            {achievement.description}
          </p>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Achievement Value */}
          <div className="text-center">
            <div className="text-3xl font-bold text-gray-800 mb-1">
              {typeof achievement.value === 'number'
                ? achievement.value.toLocaleString()
                : achievement.value
              }
            </div>
            {achievement.previousValue && (
              <div className="text-sm text-gray-500">
                Up from {typeof achievement.previousValue === 'number'
                  ? achievement.previousValue.toLocaleString()
                  : achievement.previousValue
                }
              </div>
            )}
          </div>

          {/* Personalized Message */}
          <div className="bg-white bg-opacity-60 p-4 rounded-lg">
            <p className="text-gray-700 text-center">
              {personalizedMessage}
            </p>
          </div>

          {/* Unlocked Features */}
          {achievement.unlockedFeatures && achievement.unlockedFeatures.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-800 flex items-center">
                <Gift className="w-4 h-4 mr-2 text-green-600" />
                New Features Unlocked
              </h4>
              <div className="space-y-1">
                {achievement.unlockedFeatures.map((feature, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    <span className="text-gray-700">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Next Steps */}
          {!showOnboarding && (
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-800 flex items-center">
                <ArrowRight className="w-4 h-4 mr-2 text-blue-600" />
                What's Next
              </h4>
              <div className="space-y-1">
                {nextSteps.map((step, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <div className="w-2 h-2 bg-blue-500 rounded-full" />
                    <span className="text-gray-700">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-2 pt-2">
            {onShare && (
              <Button
                variant="outline"
                size="sm"
                onClick={onShare}
                className="flex-1"
              >
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </Button>
            )}

            <Button
              onClick={onContinue || onClose}
              className="flex-1 bg-gradient-to-r from-blue-600 to-green-600 hover:from-blue-700 hover:to-green-700 text-white"
            >
              {showOnboarding ? 'Continue Setup' : 'Keep Trading'}
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>

          {/* Progress indicator for series achievements */}
          {achievement.type === 'streak' && (
            <div className="pt-2">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>Current Streak</span>
                <span>Next Milestone</span>
              </div>
              <Progress
                value={((Number(achievement.value) % 10) / 10) * 100}
                className="h-2"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

// Helper function to create achievement from system events
export const createAchievement = (
  eventType: string,
  eventData: any,
  persona: string = 'casual_investor'
): Achievement => {
  const achievements = {
    gate_progression: {
      title: `Welcome to Gate ${eventData.newGate}!`,
      description: `You've successfully graduated to ${eventData.newGate} with enhanced trading capabilities`,
      type: 'gate_progression' as const,
      value: eventData.newGate,
      previousValue: eventData.previousGate,
      unlockedFeatures: eventData.unlockedFeatures || [
        'Higher position limits',
        'New asset classes',
        'Advanced strategies'
      ],
      rarity: eventData.newGate === 'G3' ? 'legendary' : 'epic' as const,
      celebrationStyle: 'confetti' as const
    },

    profit_milestone: {
      title: 'Profit Milestone Reached!',
      description: `Your systematic approach has generated real profits`,
      type: 'profit_milestone' as const,
      value: `$${eventData.profit.toFixed(2)}`,
      previousValue: eventData.previousProfit ? `$${eventData.previousProfit.toFixed(2)}` : undefined,
      rarity: eventData.profit > 1000 ? 'epic' : eventData.profit > 100 ? 'rare' : 'common' as const,
      celebrationStyle: 'sparkles' as const
    },

    trading_streak: {
      title: `${eventData.days} Day Trading Streak!`,
      description: 'Consistent execution of your systematic strategy',
      type: 'streak' as const,
      value: eventData.days,
      rarity: eventData.days >= 30 ? 'epic' : eventData.days >= 14 ? 'rare' : 'common' as const,
      celebrationStyle: 'gentle' as const
    }
  };

  const template = achievements[eventType as keyof typeof achievements];
  if (!template) {
    throw new Error(`Unknown achievement type: ${eventType}`);
  }

  return {
    id: `${eventType}_${Date.now()}`,
    ...template,
    earnedAt: new Date()
  };
};

export default ProgressCelebration;