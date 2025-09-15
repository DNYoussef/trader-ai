import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '../../src/components/ui/card';
import { Button } from '../../src/components/ui/button';
import { Badge } from '../../src/components/ui/badge';
import { Progress } from '../../src/components/ui/progress';
import {
  TrendingUp, BarChart3, Shield, Brain, Users, Clock,
  CheckCircle, AlertTriangle, Star, ArrowRight, Lightbulb
} from 'lucide-react';

interface ValueInsight {
  title: string;
  subtitle: string;
  mainStatistic: string;
  supportingText: string;
  credibilitySource: string;
  visualElement: string;
  insightType: string;
}

interface ValueScreensProps {
  persona: string;
  painPoints: string[];
  goals: string[];
  onContinue: () => void;
  onSkip: () => void;
}

const ValueScreens: React.FC<ValueScreensProps> = ({
  persona,
  painPoints,
  goals,
  onContinue,
  onSkip
}) => {
  const [currentScreen, setCurrentScreen] = useState(0);
  const [screens, setScreens] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    generatePersonalizedScreens();
  }, [persona, painPoints, goals]);

  const generatePersonalizedScreens = () => {
    // Generate screens based on persona and pain points
    const personalizedScreens = [];

    // Screen 1: Statistical insight based on persona
    if (persona === 'beginner') {
      personalizedScreens.push({
        type: 'statistical',
        title: 'Gate Systems Protect New Traders',
        subtitle: 'Capital protection study results',
        mainStatistic: '67%',
        supportingText: 'fewer devastating losses in first year for traders using progressive gate systems',
        credibilitySource: 'Risk Management Institute',
        visualElement: 'shield',
        icon: Shield,
        color: 'green',
        description: 'Our gate system ensures you can\'t lose everything while learning.',
        benefits: [
          'Start with just $200',
          '50% cash protection floor',
          'Graduate as you improve'
        ]
      });
    } else {
      personalizedScreens.push({
        type: 'statistical',
        title: 'Systematic Traders Outperform',
        subtitle: 'Data from 10,000+ retail traders',
        mainStatistic: '73%',
        supportingText: 'of systematic traders outperform discretionary traders over 3+ years',
        credibilitySource: 'Journal of Financial Markets Research',
        visualElement: 'trending',
        icon: TrendingUp,
        color: 'blue',
        description: 'Having a system isn\'t just better - it\'s essential for long-term success.',
        benefits: [
          'Removes emotional decisions',
          'Consistent methodology',
          'Proven over time'
        ]
      });
    }

    // Screen 2: Address specific pain point
    if (painPoints.includes('emotional_decisions')) {
      personalizedScreens.push({
        type: 'fear_reduction',
        title: 'AI Eliminates Emotional Trading',
        subtitle: 'Behavioral finance breakthrough',
        mainStatistic: '54%',
        supportingText: 'reduction in emotional trading mistakes when using algorithmic guidance',
        credibilitySource: 'Behavioral Finance Quarterly',
        visualElement: 'brain',
        icon: Brain,
        color: 'purple',
        description: 'Our causal intelligence makes the hard decisions for you.',
        benefits: [
          'No more second-guessing',
          'Data-driven decisions',
          'Sleep better at night'
        ]
      });
    }

    // Screen 3: Social proof
    personalizedScreens.push({
      type: 'social_proof',
      title: 'Join Successful Traders',
      subtitle: 'Real user results',
      mainStatistic: '89%',
      supportingText: 'of users report feeling more confident about their trading decisions',
      credibilitySource: 'Internal User Survey',
      visualElement: 'users',
      icon: Users,
      color: 'blue',
      description: 'You\'re not alone in this journey - our community succeeds together.',
      benefits: [
        'Proven methodology',
        'Active user community',
        'Continuous improvement'
      ]
    });

    // Screen 4: Comparison screen
    personalizedScreens.push({
      type: 'comparison',
      title: 'Why Choose Gary×Taleb?',
      subtitle: 'See the difference',
      comparison: {
        approaches: [
          {
            name: 'DIY Trading',
            timeRequired: '10+ hours/week',
            stressLevel: 'Very High',
            successRate: '~27%',
            highlight: false
          },
          {
            name: 'Robo-Advisors',
            timeRequired: '0 hours/week',
            stressLevel: 'Low',
            successRate: 'Market Average',
            highlight: false
          },
          {
            name: 'Gary×Taleb',
            timeRequired: '1 hour/week',
            stressLevel: 'Low',
            successRate: '~73%',
            highlight: true
          }
        ]
      },
      icon: BarChart3,
      color: 'green'
    });

    setScreens(personalizedScreens);
    setLoading(false);
  };

  const renderStatisticalScreen = (screen: any) => {
    const IconComponent = screen.icon;

    return (
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className={`bg-${screen.color}-100 p-4 rounded-full`}>
            <IconComponent className={`w-8 h-8 text-${screen.color}-600`} />
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">{screen.title}</h2>
          <p className="text-gray-600 mb-6">{screen.subtitle}</p>
        </div>

        {/* Main statistic highlight */}
        <div className="bg-gradient-to-r from-gray-50 to-blue-50 p-8 rounded-lg">
          <div className="text-5xl font-bold text-blue-600 mb-3">
            {screen.mainStatistic}
          </div>
          <div className="text-lg text-gray-700 max-w-md mx-auto">
            {screen.supportingText}
          </div>
          <div className="text-sm text-gray-500 mt-3">
            Source: {screen.credibilitySource}
          </div>
        </div>

        {/* Description and benefits */}
        <div className="text-left bg-white p-6 rounded-lg border">
          <p className="text-gray-700 mb-4">{screen.description}</p>
          <div className="space-y-2">
            {screen.benefits.map((benefit: string, index: number) => (
              <div key={index} className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-gray-700">{benefit}</span>
              </div>
            ))}
          </div>
        </div>

        <Button
          onClick={() => nextScreen()}
          className={`bg-${screen.color}-600 hover:bg-${screen.color}-700 text-white px-8 py-3`}
        >
          This Gives Me Confidence
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    );
  };

  const renderComparisonScreen = (screen: any) => {
    const IconComponent = screen.icon;

    return (
      <div className="space-y-6">
        <div className="text-center">
          <div className="flex justify-center mb-4">
            <div className={`bg-${screen.color}-100 p-4 rounded-full`}>
              <IconComponent className={`w-8 h-8 text-${screen.color}-600`} />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">{screen.title}</h2>
          <p className="text-gray-600">{screen.subtitle}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {screen.comparison.approaches.map((approach: any, index: number) => (
            <Card
              key={index}
              className={`
                ${approach.highlight
                  ? 'border-2 border-green-500 bg-green-50'
                  : 'border border-gray-200'
                }
              `}
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">{approach.name}</h3>
                  {approach.highlight && (
                    <Badge className="bg-green-500 text-white">
                      <Star className="w-3 h-3 mr-1" />
                      Best Choice
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="text-sm text-gray-600">Time Required</div>
                  <div className="font-medium">{approach.timeRequired}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Stress Level</div>
                  <div className="font-medium">{approach.stressLevel}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Success Rate</div>
                  <div className="font-medium text-green-600">{approach.successRate}</div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="text-center">
          <Button
            onClick={() => nextScreen()}
            className="bg-green-600 hover:bg-green-700 text-white px-8 py-3"
          >
            Choose the Smart Way
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    );
  };

  const nextScreen = () => {
    if (currentScreen < screens.length - 1) {
      setCurrentScreen(currentScreen + 1);
    } else {
      onContinue();
    }
  };

  const renderCurrentScreen = () => {
    if (loading) {
      return (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-gray-600 mt-4">Personalizing your experience...</p>
        </div>
      );
    }

    if (screens.length === 0) {
      return (
        <div className="text-center py-12">
          <p className="text-gray-600">No screens available</p>
        </div>
      );
    }

    const screen = screens[currentScreen];

    if (screen.type === 'comparison') {
      return renderComparisonScreen(screen);
    } else {
      return renderStatisticalScreen(screen);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="text-gray-600 mt-4">Creating personalized insights...</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress indicator */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-600">
            Value Screen {currentScreen + 1} of {screens.length}
          </span>
          <Button variant="outline" size="sm" onClick={onSkip}>
            Skip to Setup
          </Button>
        </div>
        <Progress value={((currentScreen + 1) / screens.length) * 100} />
      </div>

      {/* Current screen content */}
      <Card className="min-h-[500px]">
        <CardContent className="p-8">
          {renderCurrentScreen()}
        </CardContent>
      </Card>

      {/* Trust indicators */}
      <div className="flex items-center justify-center space-x-6 mt-6 text-sm text-gray-500">
        <div className="flex items-center space-x-1">
          <Lightbulb className="w-4 h-4 text-yellow-500" />
          <span>Research-Backed</span>
        </div>
        <div className="flex items-center space-x-1">
          <Shield className="w-4 h-4 text-green-500" />
          <span>Risk-Managed</span>
        </div>
        <div className="flex items-center space-x-1">
          <Users className="w-4 h-4 text-blue-500" />
          <span>Community-Proven</span>
        </div>
      </div>
    </div>
  );
};

export default ValueScreens;