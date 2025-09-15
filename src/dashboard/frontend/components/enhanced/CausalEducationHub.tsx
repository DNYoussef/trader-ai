/**
 * Causal Education Hub - Main educational interface for causal intelligence concepts
 *
 * Makes complex causal concepts accessible through progressive disclosure,
 * interactive tutorials, and personalized learning paths.
 */

import React, { useState, useEffect } from 'react';
import { useEnhancedUX } from './EnhancedUXProvider';

interface EducationalConcept {
  concept_id: string;
  title: string;
  explanation: string;
  curiosity_hook: string;
  confidence_builder: string;
  practical_benefit: string;
  analogy: string;
  visual_metaphor: string;
  real_world_example: string;
  what_unlocks_next: string[];
  estimated_read_time: string;
}

interface InteractiveTutorial {
  tutorial_id: string;
  title: string;
  description: string;
  estimated_time: string;
  reward: string;
  practical_outcome: string;
}

interface LearningProgress {
  user_id: string;
  current_level: string;
  next_concepts: EducationalConcept[];
  recommended_tutorials: InteractiveTutorial[];
  progress_percentage: number;
  achievements_unlocked: Array<{
    id: string;
    title: string;
    description: string;
    icon: string;
  }>;
  next_milestone: {
    title: string;
    description: string;
    progress: number;
    target: number;
    reward: string;
  };
}

interface CausalEducationHubProps {
  userId: string;
  userLevel?: string;
  onConceptComplete?: (conceptId: string) => void;
  onTutorialStart?: (tutorialId: string) => void;
}

const CausalEducationHub: React.FC<CausalEducationHubProps> = ({
  userId,
  userLevel = 'beginner',
  onConceptComplete,
  onTutorialStart
}) => {
  const [learningProgress, setLearningProgress] = useState<LearningProgress | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<EducationalConcept | null>(null);
  const [showDetailedView, setShowDetailedView] = useState(false);
  const [loading, setLoading] = useState(true);
  const { triggerCelebration, trackProgress } = useEnhancedUX();

  useEffect(() => {
    loadLearningDashboard();
  }, [userId, userLevel]);

  const loadLearningDashboard = async () => {
    try {
      setLoading(true);
      // In production, this would call the backend API
      const mockProgress: LearningProgress = {
        user_id: userId,
        current_level: userLevel,
        next_concepts: [
          {
            concept_id: 'distributional_flows',
            title: 'How Money Really Flows in Markets',
            explanation: 'Money doesn\'t just move randomly in markets - it flows from specific groups to others in predictable patterns. Our system tracks these money flows to spot opportunities.',
            curiosity_hook: 'Want to see where smart money is moving before everyone else?',
            confidence_builder: 'You already understand supply and demand - this is just more precise',
            practical_benefit: 'Spot major moves 1-3 weeks before they happen',
            analogy: 'Think of the market like a river system - money flows from small streams (retail) to big rivers (institutions). We\'re tracking where the water is going before others notice.',
            visual_metaphor: 'River system with tributaries and main channels',
            real_world_example: 'When GameStop spiked, our system would have detected unusual retail flow patterns weeks before the main move.',
            what_unlocks_next: ['policy_shocks', 'causal_dag'],
            estimated_read_time: '2-3 minutes'
          }
        ],
        recommended_tutorials: [
          {
            tutorial_id: 'money_flows_intro',
            title: 'Spot the Money Flow',
            description: 'Learn to identify where big money is moving in real market data',
            estimated_time: '3 minutes',
            reward: 'Flow Detection Badge + Next tutorial unlocked',
            practical_outcome: 'You can now spot institutional buying before price moves'
          }
        ],
        progress_percentage: 15,
        achievements_unlocked: [
          {
            id: 'first_concept',
            title: 'Curious Learner',
            description: 'Started your causal intelligence journey',
            icon: 'lightbulb'
          }
        ],
        next_milestone: {
          title: 'Causal Foundation',
          description: 'Complete 2 core concepts',
          progress: 0,
          target: 2,
          reward: 'Advanced concepts unlocked'
        }
      };

      setLearningProgress(mockProgress);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load learning dashboard:', error);
      setLoading(false);
    }
  };

  const handleConceptSelect = (concept: EducationalConcept) => {
    setSelectedConcept(concept);
    setShowDetailedView(true);

    trackProgress('concept_viewed', {
      concept_id: concept.concept_id,
      user_level: userLevel
    });
  };

  const handleConceptComplete = (conceptId: string) => {
    triggerCelebration('concept_mastered', {
      concept_title: selectedConcept?.title,
      next_unlocks: selectedConcept?.what_unlocks_next
    });

    if (onConceptComplete) {
      onConceptComplete(conceptId);
    }

    // Update progress
    if (learningProgress) {
      setLearningProgress({
        ...learningProgress,
        progress_percentage: learningProgress.progress_percentage + 20,
        next_milestone: {
          ...learningProgress.next_milestone,
          progress: learningProgress.next_milestone.progress + 1
        }
      });
    }

    setShowDetailedView(false);
  };

  const handleTutorialStart = (tutorial: InteractiveTutorial) => {
    if (onTutorialStart) {
      onTutorialStart(tutorial.tutorial_id);
    }

    trackProgress('tutorial_started', {
      tutorial_id: tutorial.tutorial_id,
      estimated_time: tutorial.estimated_time
    });
  };

  const ProgressBar: React.FC<{ progress: number; target: number }> = ({ progress, target }) => (
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
        style={{ width: `${(progress / target) * 100}%` }}
      />
    </div>
  );

  const ConceptCard: React.FC<{ concept: EducationalConcept }> = ({ concept }) => (
    <div
      className="bg-white rounded-lg shadow-md p-6 cursor-pointer hover:shadow-lg transition-shadow"
      onClick={() => handleConceptSelect(concept)}
    >
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{concept.title}</h3>
        <span className="text-sm text-gray-500">{concept.estimated_read_time}</span>
      </div>

      <div className="mb-4">
        <p className="text-blue-600 font-medium mb-2">{concept.curiosity_hook}</p>
        <p className="text-gray-700 text-sm">{concept.explanation}</p>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-green-600 text-sm font-medium">
          Benefit: {concept.practical_benefit}
        </span>
        <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
          Learn Now
        </button>
      </div>
    </div>
  );

  const TutorialCard: React.FC<{ tutorial: InteractiveTutorial }> = ({ tutorial }) => (
    <div className="bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg shadow-md p-6 text-white">
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-lg font-semibold">{tutorial.title}</h3>
        <span className="text-sm opacity-75">{tutorial.estimated_time}</span>
      </div>

      <p className="text-sm mb-4 opacity-90">{tutorial.description}</p>

      <div className="mb-4">
        <p className="text-sm font-medium mb-1">You'll gain:</p>
        <p className="text-sm opacity-75">{tutorial.practical_outcome}</p>
      </div>

      <button
        className="bg-white text-purple-600 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-100"
        onClick={() => handleTutorialStart(tutorial)}
      >
        Start Tutorial
      </button>
    </div>
  );

  const DetailedConceptView: React.FC = () => {
    if (!selectedConcept) return null;

    return (
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="flex items-start justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">{selectedConcept.title}</h2>
          <button
            onClick={() => setShowDetailedView(false)}
            className="text-gray-500 hover:text-gray-700"
          >
            Close
          </button>
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-blue-600 mb-2">The Hook</h3>
            <p className="text-gray-800">{selectedConcept.curiosity_hook}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-green-600 mb-2">Confidence Builder</h3>
            <p className="text-gray-800">{selectedConcept.confidence_builder}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">How It Works</h3>
            <p className="text-gray-700">{selectedConcept.explanation}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-purple-600 mb-2">Think of It Like This</h3>
            <p className="text-gray-700 italic">{selectedConcept.analogy}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-orange-600 mb-2">Real Example</h3>
            <p className="text-gray-700">{selectedConcept.real_world_example}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-red-600 mb-2">Your Practical Benefit</h3>
            <p className="text-gray-800 font-medium">{selectedConcept.practical_benefit}</p>
          </div>

          {selectedConcept.what_unlocks_next.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-indigo-600 mb-2">What This Unlocks</h3>
              <ul className="list-disc list-inside text-gray-700">
                {selectedConcept.what_unlocks_next.map((unlock, index) => (
                  <li key={index} className="capitalize">{unlock.replace('_', ' ')}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex space-x-4">
            <button
              className="bg-green-600 text-white px-6 py-3 rounded-md font-medium hover:bg-green-700"
              onClick={() => handleConceptComplete(selectedConcept.concept_id)}
            >
              I Understand This Concept
            </button>
            <button className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700">
              Take Practice Quiz
            </button>
          </div>
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

  if (!learningProgress) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Unable to load learning progress</p>
        <button
          onClick={loadLearningDashboard}
          className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (showDetailedView) {
    return <DetailedConceptView />;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Causal Intelligence Education</h1>
        <p className="text-gray-600">Master the hidden patterns that drive market movements</p>
      </div>

      {/* Progress Overview */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Your Progress</h2>
          <span className="text-sm text-gray-600">
            {learningProgress.progress_percentage}% Complete
          </span>
        </div>

        <div className="mb-6">
          <ProgressBar
            progress={learningProgress.progress_percentage}
            target={100}
          />
        </div>

        {/* Next Milestone */}
        <div className="bg-white rounded-lg p-4">
          <h3 className="font-semibold text-gray-800 mb-2">
            {learningProgress.next_milestone.title}
          </h3>
          <p className="text-sm text-gray-600 mb-3">
            {learningProgress.next_milestone.description}
          </p>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">
              Progress: {learningProgress.next_milestone.progress} / {learningProgress.next_milestone.target}
            </span>
            <span className="text-green-600 font-medium">
              Reward: {learningProgress.next_milestone.reward}
            </span>
          </div>
          <ProgressBar
            progress={learningProgress.next_milestone.progress}
            target={learningProgress.next_milestone.target}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Next Concepts */}
        <div className="lg:col-span-2">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Next Concepts to Learn</h2>
          <div className="space-y-6">
            {learningProgress.next_concepts.map((concept) => (
              <ConceptCard key={concept.concept_id} concept={concept} />
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Interactive Tutorials */}
          {learningProgress.recommended_tutorials.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Interactive Tutorials</h3>
              <div className="space-y-4">
                {learningProgress.recommended_tutorials.map((tutorial) => (
                  <TutorialCard key={tutorial.tutorial_id} tutorial={tutorial} />
                ))}
              </div>
            </div>
          )}

          {/* Achievements */}
          {learningProgress.achievements_unlocked.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Achievements</h3>
              <div className="space-y-3">
                {learningProgress.achievements_unlocked.map((achievement) => (
                  <div key={achievement.id} className="bg-white rounded-lg p-4 shadow-sm border">
                    <div className="flex items-center">
                      <div className="w-8 h-8 bg-yellow-400 rounded-full flex items-center justify-center mr-3">
                        <span className="text-xs">🏆</span>
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">{achievement.title}</h4>
                        <p className="text-sm text-gray-600">{achievement.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Daily Insight */}
          <div className="bg-gradient-to-r from-green-400 to-blue-500 rounded-lg p-6 text-white">
            <h3 className="font-semibold mb-2">Today's Causal Insight</h3>
            <p className="text-sm mb-4 opacity-90">
              Policy announcements create predictable sector rotations 2-3 days later
            </p>
            <button className="bg-white text-blue-600 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-100">
              See it in action
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CausalEducationHub;