import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface CalibrationQuestion {
  id: string;
  question: string;
  context: string;
  correctAnswer: boolean;
  explanation: string;
  difficulty: 'easy' | 'medium' | 'hard';
  category: 'market' | 'inequality' | 'economics' | 'psychology';
}

interface CalibrationPrediction {
  questionId: string;
  prediction: boolean;
  confidence: number;
  timestamp: Date;
  correct?: boolean;
}

interface CalibrationStats {
  totalPredictions: number;
  overallAccuracy: number;
  calibrationError: number;
  confidenceBins: Record<string, {
    accuracy: number;
    count: number;
    targetConfidence: number;
  }>;
}

const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const TargetIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const TrophyIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
  </svg>
);

export const CalibrationTrainer: React.FC = () => {
  const [currentQuestion, setCurrentQuestion] = useState<CalibrationQuestion | null>(null);
  const [userPrediction, setUserPrediction] = useState<boolean | null>(null);
  const [userConfidence, setUserConfidence] = useState<number>(0.7);
  const [predictions, setPredictions] = useState<CalibrationPrediction[]>([]);
  const [showAnswer, setShowAnswer] = useState(false);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [isTrainingActive, setIsTrainingActive] = useState(false);

  // Sample calibration questions based on Gary's trading philosophy
  const questions: CalibrationQuestion[] = useMemo(() => [
    {
      id: 'q1',
      question: 'Will wealth inequality in the US increase over the next 12 months?',
      context: 'Current Gini coefficient is 0.434, Federal Reserve is maintaining low rates, and corporate profits as % of GDP are at historic highs.',
      correctAnswer: true,
      explanation: 'Low interest rates disproportionately benefit asset holders (wealthy), while wage growth remains subdued. Corporate profit concentration continues driving wealth upward.',
      difficulty: 'medium',
      category: 'inequality'
    },
    {
      id: 'q2',
      question: 'Will the housing market crash within 6 months given current mortgage rates?',
      context: 'Mortgage rates at 7.5%, inventory low, but 40% of home purchases are cash buyers (wealthy investors).',
      correctAnswer: false,
      explanation: 'Cash buyers (concentrated wealth) can continue purchasing regardless of mortgage rates. Regular buyers are priced out, but wealthy demand supports prices.',
      difficulty: 'hard',
      category: 'market'
    },
    {
      id: 'q3',
      question: 'Will luxury goods stocks outperform discount retailers over 6 months?',
      context: 'Wealth concentration accelerating, top 10% hold 70% of wealth, consumer sentiment mixed.',
      correctAnswer: true,
      explanation: 'Wealth concentration means luxury demand stays strong (wealthy have money) while middle/lower class spending contracts (discount pressure).',
      difficulty: 'easy',
      category: 'inequality'
    },
    {
      id: 'q4',
      question: 'Will tech stock valuations remain elevated despite economic slowdown?',
      context: 'Economic indicators mixed, but tech companies have pricing power and wealthy investor base continues buying.',
      correctAnswer: true,
      explanation: 'Concentrated wealth flows into scarce assets (tech stocks). Wealthy don\'t stop investing during slowdowns; they often accelerate.',
      difficulty: 'medium',
      category: 'market'
    },
    {
      id: 'q5',
      question: 'Will economists correctly predict the next market move?',
      context: 'Consensus forecast shows market optimism, traditional models suggest stability, but inequality metrics show acceleration.',
      correctAnswer: false,
      explanation: 'Economists consistently miss inequality effects. They model "average" behavior, but concentrated wealth drives different market dynamics.',
      difficulty: 'easy',
      category: 'psychology'
    }
  ], []);

  const calibrationStats = useMemo((): CalibrationStats => {
    if (predictions.length === 0) {
      return {
        totalPredictions: 0,
        overallAccuracy: 0,
        calibrationError: 0,
        confidenceBins: {}
      };
    }

    const resolvedPredictions = predictions.filter(p => p.correct !== undefined);
    const correct = resolvedPredictions.filter(p => p.correct).length;
    const overallAccuracy = correct / resolvedPredictions.length;

    // Group by confidence bins
    const bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const confidenceBins: Record<string, any> = {};

    bins.forEach(bin => {
      const binPredictions = resolvedPredictions.filter(
        p => Math.abs(p.confidence - bin) <= 0.05
      );

      if (binPredictions.length > 0) {
        const binCorrect = binPredictions.filter(p => p.correct).length;
        confidenceBins[bin.toString()] = {
          accuracy: binCorrect / binPredictions.length,
          count: binPredictions.length,
          targetConfidence: bin
        };
      }
    });

    // Calculate calibration error
    const calibrationError = Object.values(confidenceBins).reduce((sum: number, bin: any) => {
      return sum + Math.abs(bin.accuracy - bin.targetConfidence) * bin.count;
    }, 0) / resolvedPredictions.length;

    return {
      totalPredictions: predictions.length,
      overallAccuracy,
      calibrationError,
      confidenceBins
    };
  }, [predictions]);

  useEffect(() => {
    if (isTrainingActive && currentQuestion === null) {
      setCurrentQuestion(questions[questionIndex]);
    }
  }, [isTrainingActive, currentQuestion, questions, questionIndex]);

  const startTraining = () => {
    setIsTrainingActive(true);
    setQuestionIndex(0);
    setCurrentQuestion(questions[0]);
    setShowAnswer(false);
    setUserPrediction(null);
    setUserConfidence(0.7);
  };

  const submitPrediction = () => {
    if (currentQuestion && userPrediction !== null) {
      const prediction: CalibrationPrediction = {
        questionId: currentQuestion.id,
        prediction: userPrediction,
        confidence: userConfidence,
        timestamp: new Date(),
        correct: userPrediction === currentQuestion.correctAnswer
      };

      setPredictions(prev => [...prev, prediction]);
      setShowAnswer(true);
    }
  };

  const nextQuestion = () => {
    if (questionIndex < questions.length - 1) {
      setQuestionIndex(prev => prev + 1);
      setCurrentQuestion(questions[questionIndex + 1]);
      setShowAnswer(false);
      setUserPrediction(null);
      setUserConfidence(0.7);
    } else {
      // Training complete
      setIsTrainingActive(false);
      setCurrentQuestion(null);
    }
  };

  const getCalibrationColor = (error: number) => {
    if (error < 0.1) return 'text-green-500 bg-green-100';
    if (error < 0.2) return 'text-yellow-500 bg-yellow-100';
    return 'text-red-500 bg-red-100';
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'hard': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;

  if (!isTrainingActive) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <BrainIcon />
            <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Calibration Training
            </h3>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Train your probability assessment skills. Learn to be accurate when you say you're 80% confident.
          </p>

          {/* Stats Dashboard */}
          {predictions.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-600">{calibrationStats.totalPredictions}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Total Predictions</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600">
                  {formatPercent(calibrationStats.overallAccuracy)}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Overall Accuracy</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className={`text-2xl font-bold ${
                  calibrationStats.calibrationError < 0.1 ? 'text-green-600' :
                  calibrationStats.calibrationError < 0.2 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {formatPercent(calibrationStats.calibrationError)}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Calibration Error</div>
              </div>
            </div>
          )}

          {/* Calibration Chart */}
          {Object.keys(calibrationStats.confidenceBins).length > 0 && (
            <div className="mb-6">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                Calibration by Confidence Level
              </h4>
              <div className="space-y-2">
                {Object.entries(calibrationStats.confidenceBins).map(([confidence, data]) => (
                  <div key={confidence} className="flex items-center gap-4">
                    <div className="w-16 text-sm text-gray-600 dark:text-gray-400">
                      {formatPercent(parseFloat(confidence))}
                    </div>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-4 relative">
                      <div
                        className="bg-blue-500 h-4 rounded-full"
                        style={{ width: `${(data as any).accuracy * 100}%` }}
                      />
                      <div
                        className="absolute top-0 w-1 h-4 bg-red-500"
                        style={{ left: `${parseFloat(confidence) * 100}%` }}
                      />
                    </div>
                    <div className="w-20 text-sm text-gray-600 dark:text-gray-400">
                      {formatPercent((data as any).accuracy)} ({(data as any).count})
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Red line = target accuracy. Blue bar = actual accuracy. Perfect calibration = red line matches end of blue bar.
              </p>
            </div>
          )}

          <button
            onClick={startTraining}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-8 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all"
          >
            {predictions.length > 0 ? 'Continue Training' : 'Start Calibration Training'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TargetIcon />
            <h3 className="text-xl font-bold">Calibration Training</h3>
          </div>
          <div className="text-blue-100">
            Question {questionIndex + 1} of {questions.length}
          </div>
        </div>
      </div>

      {/* Question Content */}
      {currentQuestion && (
        <div className="p-6">
          <AnimatePresence mode="wait">
            {!showAnswer ? (
              <motion.div
                key="question"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Question Header */}
                <div className="flex items-center justify-between">
                  <span className={`px-3 py-1 text-xs font-semibold rounded-full ${getDifficultyColor(currentQuestion.difficulty)}`}>
                    {currentQuestion.difficulty.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-500 capitalize">
                    {currentQuestion.category}
                  </span>
                </div>

                {/* Context */}
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Context:</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {currentQuestion.context}
                  </p>
                </div>

                {/* Question */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                    {currentQuestion.question}
                  </h4>

                  {/* Prediction Buttons */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <button
                      onClick={() => setUserPrediction(true)}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        userPrediction === true
                          ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                          : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                      }`}
                    >
                      <div className="text-lg font-semibold text-green-600">YES</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">This will happen</div>
                    </button>
                    <button
                      onClick={() => setUserPrediction(false)}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        userPrediction === false
                          ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                          : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                      }`}
                    >
                      <div className="text-lg font-semibold text-red-600">NO</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">This won't happen</div>
                    </button>
                  </div>

                  {/* Confidence Slider */}
                  {userPrediction !== null && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6"
                    >
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        How confident are you? {formatPercent(userConfidence)}
                      </label>
                      <input
                        type="range"
                        min="0.5"
                        max="1.0"
                        step="0.05"
                        value={userConfidence}
                        onChange={(e) => setUserConfidence(parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>50% (Guess)</span>
                        <span>75% (Likely)</span>
                        <span>100% (Certain)</span>
                      </div>
                    </motion.div>
                  )}

                  {/* Submit Button */}
                  {userPrediction !== null && (
                    <motion.button
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      onClick={submitPrediction}
                      className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all"
                    >
                      Submit Prediction
                    </motion.button>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="answer"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Result */}
                <div className={`text-center p-6 rounded-lg ${
                  userPrediction === currentQuestion.correctAnswer
                    ? 'bg-green-50 dark:bg-green-900/20 border border-green-200'
                    : 'bg-red-50 dark:bg-red-900/20 border border-red-200'
                }`}>
                  <div className="flex items-center justify-center gap-2 mb-2">
                    {userPrediction === currentQuestion.correctAnswer ? (
                      <TrophyIcon />
                    ) : (
                      <div className="w-5 h-5 text-red-500">❌</div>
                    )}
                    <h4 className={`text-xl font-bold ${
                      userPrediction === currentQuestion.correctAnswer ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {userPrediction === currentQuestion.correctAnswer ? 'Correct!' : 'Incorrect'}
                    </h4>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    You predicted: <strong>{userPrediction ? 'YES' : 'NO'}</strong> with {formatPercent(userConfidence)} confidence
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Correct answer: <strong>{currentQuestion.correctAnswer ? 'YES' : 'NO'}</strong>
                  </p>
                </div>

                {/* Explanation */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                    Gary's Analysis:
                  </h5>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    {currentQuestion.explanation}
                  </p>
                </div>

                {/* Continue Button */}
                <button
                  onClick={nextQuestion}
                  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all"
                >
                  {questionIndex < questions.length - 1 ? 'Next Question' : 'Complete Training'}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default CalibrationTrainer;