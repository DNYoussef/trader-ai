/**
 * Interactive Tutorial Component - Hands-on learning for causal concepts
 *
 * Provides step-by-step guided learning with immediate feedback,
 * gamification elements, and practical application of concepts.
 */

import React, { useState, useEffect } from 'react';
import { useEnhancedUX } from './EnhancedUXProvider';

interface TutorialStep {
  step: number;
  title: string;
  content: string;
  action: string;
  data?: string;
  choices?: string[];
  interactive?: boolean;
  reward?: string;
}

interface TutorialSession {
  session_id: string;
  tutorial: {
    title: string;
    description: string;
    estimated_time: string;
    reward: string;
    practical_outcome: string;
  };
  current_step: TutorialStep;
  progress: string;
  next_action: string;
}

interface InteractiveTutorialProps {
  tutorialId: string;
  userId: string;
  onComplete?: (results: any) => void;
  onClose?: () => void;
}

const InteractiveTutorial: React.FC<InteractiveTutorialProps> = ({
  tutorialId,
  userId,
  onComplete,
  onClose
}) => {
  const [session, setSession] = useState<TutorialSession | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [userResponse, setUserResponse] = useState<string>('');
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [pointsEarned, setPointsEarned] = useState(0);
  const [loading, setLoading] = useState(true);

  const { triggerCelebration, trackProgress } = useEnhancedUX();

  useEffect(() => {
    startTutorial();
  }, [tutorialId, userId]);

  const startTutorial = async () => {
    try {
      setLoading(true);

      // Mock tutorial sessions - in production this would call the backend
      const mockSessions: Record<string, TutorialSession> = {
        'money_flows_intro': {
          session_id: `${userId}_money_flows_intro_${Date.now()}`,
          tutorial: {
            title: 'Spot the Money Flow',
            description: 'Learn to identify where big money is moving in real market data',
            estimated_time: '3 minutes',
            reward: 'Flow Detection Badge + Next tutorial unlocked',
            practical_outcome: 'You can now spot institutional buying before price moves'
          },
          current_step: {
            step: 1,
            title: 'Observe the Pattern',
            content: 'Look at this chart showing unusual buying in tech stocks',
            action: 'identify_pattern',
            data: 'sample_flow_data'
          },
          progress: '1/3',
          next_action: 'Begin tutorial'
        },
        'cause_effect_detective': {
          session_id: `${userId}_cause_effect_detective_${Date.now()}`,
          tutorial: {
            title: 'Market Detective: Find the Real Cause',
            description: 'Distinguish between correlation and causation in market movements',
            estimated_time: '5 minutes',
            reward: 'Causal Detective Badge + Advanced concepts unlocked',
            practical_outcome: 'Never confuse correlation with causation again'
          },
          current_step: {
            step: 1,
            title: 'The Mystery',
            content: 'Tech stocks and crypto both fell 20%. Most think they\'re connected...',
            action: 'present_mystery'
          },
          progress: '1/3',
          next_action: 'Start Investigation'
        }
      };

      const tutorialSession = mockSessions[tutorialId];
      if (tutorialSession) {
        setSession(tutorialSession);
        trackProgress('tutorial_started', {
          tutorial_id: tutorialId,
          session_id: tutorialSession.session_id
        });
      }

      setLoading(false);
    } catch (error) {
      console.error('Failed to start tutorial:', error);
      setLoading(false);
    }
  };

  const handleStepResponse = async (response: any) => {
    if (!session) return;

    // Mock tutorial progression logic
    const mockResponses = {
      1: {
        feedback: "Great observation! You correctly identified the unusual volume pattern in tech stocks.",
        points: 10,
        explanation: "The key indicator was the 3x normal volume in AAPL and MSFT, suggesting institutional accumulation.",
        correct: true
      },
      2: {
        feedback: "Excellent prediction! You understood the flow patterns correctly.",
        points: 15,
        explanation: "When institutions accumulate quietly like this, prices typically rise 5-10% over 1-2 weeks.",
        correct: true
      },
      3: {
        feedback: "Perfect! You've mastered flow detection fundamentals.",
        points: 20,
        explanation: "You can now spot these patterns in real-time and position accordingly.",
        correct: true
      }
    };

    const stepResponse = mockResponses[session.current_step.step as keyof typeof mockResponses];
    setFeedback(stepResponse.feedback);
    setPointsEarned(prev => prev + stepResponse.points);
    setShowExplanation(true);

    trackProgress('tutorial_step_completed', {
      session_id: session.session_id,
      step: session.current_step.step,
      response: response,
      points_earned: stepResponse.points
    });

    // Trigger mini-celebration for correct answers
    if (stepResponse.correct) {
      triggerCelebration('step_mastery', {
        step_title: session.current_step.title,
        points: stepResponse.points
      });
    }
  };

  const advanceToNextStep = () => {
    if (!session) return;

    const nextStepIndex = currentStepIndex + 1;

    // Mock next steps
    const allSteps: Record<string, TutorialStep[]> = {
      'money_flows_intro': [
        {
          step: 1,
          title: 'Observe the Pattern',
          content: 'Look at this chart showing unusual buying in tech stocks. Notice the volume spikes in AAPL and MSFT that are 3x normal levels.',
          action: 'identify_pattern',
          data: 'sample_flow_data'
        },
        {
          step: 2,
          title: 'Predict the Outcome',
          content: 'Based on the flow pattern you observed, what do you think will happen to tech stock prices over the next 1-2 weeks?',
          action: 'make_prediction',
          choices: ['Prices rise 5-10%', 'Prices fall 10-15%', 'No significant change'],
          interactive: true
        },
        {
          step: 3,
          title: 'See the Results',
          content: 'Here\'s what actually happened - tech stocks rose 8% over the next 10 days! Your prediction was spot-on.',
          action: 'reveal_outcome',
          reward: 'Flow Detection Badge'
        }
      ],
      'cause_effect_detective': [
        {
          step: 1,
          title: 'The Mystery',
          content: 'Tech stocks and crypto both fell 20% last week. Most traders think they\'re connected, but let\'s investigate...',
          action: 'present_mystery'
        },
        {
          step: 2,
          title: 'Gather Evidence',
          content: 'Look closely at the timing: Crypto fell Monday, tech stocks fell Wednesday. What happened in between?',
          action: 'examine_timeline',
          interactive: true,
          choices: ['Interest rates rose', 'Institutional deleveraging', 'Retail panic selling']
        },
        {
          step: 3,
          title: 'Solve the Case',
          content: 'The real cause was institutional deleveraging! They sold crypto first to raise cash, then sold tech stocks. The correlation was just coincidental timing.',
          action: 'reveal_solution',
          reward: 'Causal Detective Badge'
        }
      ]
    };

    const steps = allSteps[tutorialId] || [];

    if (nextStepIndex < steps.length) {
      setCurrentStepIndex(nextStepIndex);
      setSession(prev => prev ? {
        ...prev,
        current_step: steps[nextStepIndex],
        progress: `${nextStepIndex + 1}/${steps.length}`
      } : null);
      setFeedback(null);
      setShowExplanation(false);
      setSelectedChoice(null);
      setUserResponse('');
    } else {
      // Tutorial completed
      completeTutorial();
    }
  };

  const completeTutorial = () => {
    if (!session) return;

    triggerCelebration('tutorial_completed', {
      tutorial_title: session.tutorial.title,
      total_points: pointsEarned,
      reward: session.tutorial.reward,
      practical_outcome: session.tutorial.practical_outcome
    });

    trackProgress('tutorial_completed', {
      session_id: session.session_id,
      tutorial_id: tutorialId,
      total_points: pointsEarned,
      completion_time: Date.now()
    });

    if (onComplete) {
      onComplete({
        tutorial_id: tutorialId,
        points_earned: pointsEarned,
        completed_at: new Date().toISOString()
      });
    }
  };

  const FlowVisualization: React.FC = () => (
    <div className="bg-gray-100 rounded-lg p-4 mb-6">
      <div className="text-center mb-4">
        <h4 className="font-semibold text-gray-800">Market Flow Pattern</h4>
        <p className="text-sm text-gray-600">Unusual institutional activity detected</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded p-3">
          <div className="text-sm font-medium text-gray-700">AAPL Volume</div>
          <div className="flex items-end space-x-1 h-12">
            <div className="bg-blue-300 w-2" style={{height: '20%'}}></div>
            <div className="bg-blue-300 w-2" style={{height: '30%'}}></div>
            <div className="bg-red-500 w-2" style={{height: '90%'}}></div>
            <div className="bg-red-500 w-2" style={{height: '85%'}}></div>
          </div>
          <div className="text-xs text-red-600 mt-1">3x normal volume</div>
        </div>

        <div className="bg-white rounded p-3">
          <div className="text-sm font-medium text-gray-700">MSFT Volume</div>
          <div className="flex items-end space-x-1 h-12">
            <div className="bg-blue-300 w-2" style={{height: '25%'}}></div>
            <div className="bg-blue-300 w-2" style={{height: '35%'}}></div>
            <div className="bg-red-500 w-2" style={{height: '80%'}}></div>
            <div className="bg-red-500 w-2" style={{height: '95%'}}></div>
          </div>
          <div className="text-xs text-red-600 mt-1">3.2x normal volume</div>
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <strong>Question:</strong> What does this pattern suggest about institutional activity?
      </div>
    </div>
  );

  const TimelineVisualization: React.FC = () => (
    <div className="bg-gray-100 rounded-lg p-4 mb-6">
      <div className="text-center mb-4">
        <h4 className="font-semibold text-gray-800">Event Timeline</h4>
        <p className="text-sm text-gray-600">Sequence of market events</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center">
          <div className="w-16 text-sm font-medium text-gray-600">Monday</div>
          <div className="flex-1 bg-red-200 rounded px-3 py-2">
            <div className="font-medium">Bitcoin falls 20%</div>
            <div className="text-sm text-gray-600">Major sell-off in crypto markets</div>
          </div>
        </div>

        <div className="flex items-center">
          <div className="w-16 text-sm font-medium text-gray-600">Tuesday</div>
          <div className="flex-1 bg-yellow-200 rounded px-3 py-2">
            <div className="font-medium">Institutional cash raising</div>
            <div className="text-sm text-gray-600">Major funds liquidating positions</div>
          </div>
        </div>

        <div className="flex items-center">
          <div className="w-16 text-sm font-medium text-gray-600">Wednesday</div>
          <div className="flex-1 bg-red-200 rounded px-3 py-2">
            <div className="font-medium">Tech stocks fall 20%</div>
            <div className="text-sm text-gray-600">AAPL, MSFT, GOOGL all down sharply</div>
          </div>
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <strong>Detective Question:</strong> What's the real causal relationship here?
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Tutorial not found</p>
        <button
          onClick={onClose}
          className="mt-4 bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700"
        >
          Close
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{session.tutorial.title}</h1>
          <p className="text-gray-600">{session.tutorial.description}</p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 text-2xl"
        >
          ×
        </button>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-600">Progress: {session.progress}</span>
          <span className="text-sm font-medium text-blue-600">
            Points: {pointsEarned}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${((currentStepIndex + 1) / 3) * 100}%` }}
          />
        </div>
      </div>

      {/* Current Step */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          {session.current_step.title}
        </h2>

        <p className="text-gray-700 mb-6">{session.current_step.content}</p>

        {/* Interactive Content */}
        {tutorialId === 'money_flows_intro' && session.current_step.step === 1 && (
          <FlowVisualization />
        )}

        {tutorialId === 'cause_effect_detective' && session.current_step.step === 2 && (
          <TimelineVisualization />
        )}

        {/* Multiple Choice */}
        {session.current_step.choices && (
          <div className="mb-6">
            <p className="font-medium text-gray-800 mb-3">Choose your answer:</p>
            <div className="space-y-2">
              {session.current_step.choices.map((choice, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedChoice(choice)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    selectedChoice === choice
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {choice}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Text Input */}
        {session.current_step.action === 'identify_pattern' && !session.current_step.choices && (
          <div className="mb-6">
            <label className="block font-medium text-gray-800 mb-2">
              What pattern do you notice?
            </label>
            <textarea
              value={userResponse}
              onChange={(e) => setUserResponse(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg"
              rows={3}
              placeholder="Describe the unusual pattern you see..."
            />
          </div>
        )}

        {/* Feedback */}
        {feedback && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
            <div className="flex items-start">
              <div className="text-green-600 mr-2">✓</div>
              <div>
                <p className="font-medium text-green-800">{feedback}</p>
                {showExplanation && (
                  <div className="mt-3">
                    <p className="text-green-700 text-sm">
                      <strong>Why this matters:</strong> The key indicator was the 3x normal volume in AAPL and MSFT, suggesting institutional accumulation.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-4">
          {!feedback && (
            <button
              onClick={() => handleStepResponse(selectedChoice || userResponse)}
              disabled={!selectedChoice && !userResponse}
              className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Submit Answer
            </button>
          )}

          {feedback && (
            <button
              onClick={advanceToNextStep}
              className="bg-green-600 text-white px-6 py-3 rounded-md font-medium hover:bg-green-700"
            >
              {currentStepIndex < 2 ? 'Continue' : 'Complete Tutorial'}
            </button>
          )}
        </div>
      </div>

      {/* Tutorial Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>Estimated time: {session.tutorial.estimated_time}</span>
          <span>Reward: {session.tutorial.reward}</span>
        </div>
      </div>
    </div>
  );
};

export default InteractiveTutorial;