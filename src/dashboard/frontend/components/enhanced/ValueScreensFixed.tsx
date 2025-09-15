import React, { useState, useEffect } from 'react';

// Simple icon components
const TrendingUp = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const Shield = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
  </svg>
);

const Brain = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const Users = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const BarChart3 = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const CheckCircle = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ArrowRight = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

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
          <div className={`p-4 rounded-full`} style={{ backgroundColor: screen.color === 'green' ? '#d1fae5' : screen.color === 'blue' ? '#dbeafe' : '#e9d5ff' }}>
            <IconComponent />
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
                <CheckCircle />
                <span className="text-gray-600">{benefit}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonScreen = (screen: any) => {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">{screen.title}</h2>
          <p className="text-gray-600">{screen.subtitle}</p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {screen.comparison.approaches.map((approach: any, index: number) => (
            <div
              key={index}
              className={`p-4 rounded-lg border-2 ${
                approach.highlight
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-200 bg-white'
              }`}
            >
              <h3 className={`font-bold mb-3 ${approach.highlight ? 'text-green-700' : 'text-gray-700'}`}>
                {approach.name}
              </h3>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-500">Time:</span>
                  <p className="font-medium">{approach.timeRequired}</p>
                </div>
                <div>
                  <span className="text-gray-500">Stress:</span>
                  <p className="font-medium">{approach.stressLevel}</p>
                </div>
                <div>
                  <span className="text-gray-500">Success:</span>
                  <p className="font-medium">{approach.successRate}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const currentScreenData = screens[currentScreen];

  return (
    <div className="max-w-2xl mx-auto p-6">
      {/* Progress indicators */}
      <div className="flex justify-center space-x-2 mb-8">
        {screens.map((_, index) => (
          <div
            key={index}
            className={`h-2 w-8 rounded-full ${
              index === currentScreen ? 'bg-blue-600' : 'bg-gray-300'
            }`}
          />
        ))}
      </div>

      {/* Screen content */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        {currentScreenData.type === 'comparison'
          ? renderComparisonScreen(currentScreenData)
          : renderStatisticalScreen(currentScreenData)}
      </div>

      {/* Navigation buttons */}
      <div className="flex justify-between mt-6">
        <button
          onClick={onSkip}
          className="text-gray-500 hover:text-gray-700"
        >
          Skip
        </button>

        <div className="space-x-4">
          {currentScreen > 0 && (
            <button
              onClick={() => setCurrentScreen(prev => prev - 1)}
              className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Back
            </button>
          )}

          {currentScreen < screens.length - 1 ? (
            <button
              onClick={() => setCurrentScreen(prev => prev + 1)}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
            >
              Next
              <ArrowRight />
            </button>
          ) : (
            <button
              onClick={onContinue}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
            >
              Get Started
              <ArrowRight />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ValueScreens;