import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../src/components/ui/card';
import { Button } from '../../src/components/ui/button';
import { Progress } from '../../src/components/ui/progress';
import { RadioGroup, RadioGroupItem } from '../../src/components/ui/radio-group';
import { Label } from '../../src/components/ui/label';
import { Badge } from '../../src/components/ui/badge';
import { CheckCircle, AlertTriangle, TrendingUp, Brain, Shield } from 'lucide-react';

interface OnboardingStep {
  id: string;
  title: string;
  type: 'welcome' | 'question' | 'value' | 'completion';
  content: any;
}

interface OnboardingWizardProps {
  onComplete: (result: any) => void;
  onClose: () => void;
}

const OnboardingWizard: React.FC<OnboardingWizardProps> = ({ onComplete, onClose }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [responses, setResponses] = useState<Record<string, any>>({});
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(10);

  // Mock onboarding steps - in production, these would come from the backend
  const [steps, setSteps] = useState<OnboardingStep[]>([
    {
      id: 'welcome',
      title: 'Welcome to Gary×Taleb Trading',
      type: 'welcome',
      content: {
        message: 'Let\'s create a trading strategy that works for you.',
        subtitle: 'This will take about 3 minutes.',
        icon: TrendingUp
      }
    },
    {
      id: 'frustrations',
      title: 'What frustrates you most about investing?',
      type: 'question',
      content: {
        question: 'What frustrates you most about investing right now?',
        options: [
          { value: 'unpredictable_returns', text: 'Unpredictable returns that keep me up at night', emotion: 'high' },
          { value: 'time_consuming', text: 'Too much time researching what to buy', emotion: 'medium' },
          { value: 'emotional_decisions', text: 'Making emotional decisions I regret later', emotion: 'high' },
          { value: 'lack_of_strategy', text: 'No clear strategy or system to follow', emotion: 'medium' },
          { value: 'market_timing', text: 'Never knowing when to buy or sell', emotion: 'high' }
        ]
      }
    },
    {
      id: 'value_insight',
      title: 'You\'re Not Alone',
      type: 'value',
      content: {
        statistic: '73%',
        message: 'of systematic traders outperform discretionary traders over 3+ years',
        source: 'Journal of Financial Markets Research',
        insight: 'The solution isn\'t trying harder - it\'s having a system.',
        icon: Brain
      }
    }
  ]);

  const handleResponse = (value: any) => {
    const newResponses = { ...responses, [steps[currentStep].id]: value };
    setResponses(newResponses);

    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setProgress(((currentStep + 1) / steps.length) * 100);
    } else {
      // Complete onboarding
      onComplete({
        sessionId,
        responses: newResponses,
        persona: determinePersona(newResponses),
        painPoints: extractPainPoints(newResponses)
      });
    }
  };

  const determinePersona = (responses: Record<string, any>) => {
    // Simple persona determination logic
    const frustration = responses.frustrations;

    if (frustration === 'unpredictable_returns' || frustration === 'emotional_decisions') {
      return 'beginner';
    } else if (frustration === 'lack_of_strategy') {
      return 'casual_investor';
    } else {
      return 'active_trader';
    }
  };

  const extractPainPoints = (responses: Record<string, any>) => {
    return Object.values(responses).filter(response =>
      typeof response === 'string' && response.length > 0
    );
  };

  const renderWelcomeStep = (step: OnboardingStep) => {
    const IconComponent = step.content.icon;

    return (
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="bg-blue-100 p-4 rounded-full">
            <IconComponent className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">{step.title}</h2>
          <p className="text-lg text-gray-600">{step.content.message}</p>
          <p className="text-sm text-gray-500 mt-2">{step.content.subtitle}</p>
        </div>
        <Button
          onClick={() => handleResponse('continue')}
          className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg"
        >
          Let's Get Started
        </Button>
      </div>
    );
  };

  const renderQuestionStep = (step: OnboardingStep) => {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-bold text-gray-800 mb-4">{step.title}</h2>
          <p className="text-gray-600 mb-6">Select the option that best describes your situation:</p>
        </div>

        <RadioGroup onValueChange={handleResponse} className="space-y-4">
          {step.content.options.map((option: any, index: number) => (
            <div
              key={option.value}
              className={`
                border rounded-lg p-4 cursor-pointer transition-all hover:border-blue-300
                ${option.emotion === 'high' ? 'border-l-4 border-l-yellow-400' : ''}
              `}
            >
              <div className="flex items-start space-x-3">
                <RadioGroupItem value={option.value} id={option.value} className="mt-1" />
                <Label htmlFor={option.value} className="flex-1 cursor-pointer">
                  <div className="text-gray-800">{option.text}</div>
                  {option.emotion === 'high' && (
                    <Badge variant="outline" className="mt-2 text-yellow-700 border-yellow-300">
                      <AlertTriangle className="w-3 h-3 mr-1" />
                      High Impact
                    </Badge>
                  )}
                </Label>
              </div>
            </div>
          ))}
        </RadioGroup>
      </div>
    );
  };

  const renderValueStep = (step: OnboardingStep) => {
    const IconComponent = step.content.icon;

    return (
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="bg-green-100 p-4 rounded-full">
            <IconComponent className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">{step.title}</h2>

          {/* Main statistic */}
          <div className="bg-gradient-to-r from-blue-50 to-green-50 p-6 rounded-lg mb-6">
            <div className="text-4xl font-bold text-blue-600 mb-2">
              {step.content.statistic}
            </div>
            <div className="text-lg text-gray-700">
              {step.content.message}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              Source: {step.content.source}
            </div>
          </div>

          {/* Insight */}
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded">
            <p className="text-gray-700 font-medium">{step.content.insight}</p>
          </div>
        </div>

        <Button
          onClick={() => handleResponse('understood')}
          className="bg-green-600 hover:bg-green-700 text-white px-8 py-3"
        >
          This Makes Sense - Continue
        </Button>
      </div>
    );
  };

  const renderCurrentStep = () => {
    const step = steps[currentStep];

    switch (step.type) {
      case 'welcome':
        return renderWelcomeStep(step);
      case 'question':
        return renderQuestionStep(step);
      case 'value':
        return renderValueStep(step);
      default:
        return <div>Unknown step type</div>;
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-blue-600" />
              <span className="text-sm font-medium text-gray-600">
                Step {currentStep + 1} of {steps.length}
              </span>
            </div>
            <Button variant="outline" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
          <Progress value={progress} className="mt-2" />
        </CardHeader>

        <CardContent className="pt-4">
          {renderCurrentStep()}
        </CardContent>

        {/* Trust indicators */}
        <div className="px-6 pb-6">
          <div className="flex items-center justify-center space-x-4 text-sm text-gray-500 pt-4 border-t">
            <div className="flex items-center space-x-1">
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span>Secure & Private</span>
            </div>
            <div className="flex items-center space-x-1">
              <Shield className="w-4 h-4 text-blue-500" />
              <span>Your Data Protected</span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default OnboardingWizard;