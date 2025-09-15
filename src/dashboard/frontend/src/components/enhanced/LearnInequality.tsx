import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from 'recharts';

// Icons
const BookOpen = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const PlayCircle = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const CheckCircle = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const LightBulb = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

interface Lesson {
  id: string;
  title: string;
  duration: string;
  completed: boolean;
  content: {
    theory: string;
    keyPoints: string[];
    example?: {
      scenario: string;
      consensus: string;
      reality: string;
      trade: string;
      outcome: string;
    };
  };
}

interface SimulationData {
  year: number;
  gini: number;
  assetPrices: number;
  medianWage: number;
  top1Wealth: number;
}

export const LearnInequality: React.FC = () => {
  const [activeLesson, setActiveLesson] = useState<Lesson | null>(null);
  const [completedLessons, setCompletedLessons] = useState<Set<string>>(new Set());
  const [simulationRunning, setSimulationRunning] = useState(false);

  const lessons: Lesson[] = [
    {
      id: 'lesson1',
      title: 'Why Economists Miss Inequality',
      duration: '15 min',
      completed: completedLessons.has('lesson1'),
      content: {
        theory: `Most economists treat the economy as if everyone has the same spending patterns and wealth levels. They look at aggregates like GDP, unemployment, and inflation without considering WHO has the money. Gary discovered that by focusing on wealth distribution, he could predict economic outcomes that consensus completely missed.`,
        keyPoints: [
          'Traditional economics assumes rational actors with equal resources',
          'Real economy driven by who has money to spend',
          'Rich and poor have fundamentally different spending patterns',
          'Wealth concentration changes economic dynamics completely',
          'Consensus models miss this because they aggregate everything'
        ],
        example: {
          scenario: '2008 Financial Crisis Recovery',
          consensus: 'Economy will recover quickly as stimulus takes effect',
          reality: 'Money went to banks/assets, not consumer spending',
          trade: 'Bet on prolonged weakness, zero rates for years',
          outcome: 'Gary made millions as rates stayed at zero until 2015'
        }
      }
    },
    {
      id: 'lesson2',
      title: 'Following Wealth Flows (Poor → Rich)',
      duration: '20 min',
      completed: completedLessons.has('lesson2'),
      content: {
        theory: `Money naturally flows from poor to rich through rent, interest, profits, and asset appreciation. During crises, governments print money that flows through the poor (who must spend it) to the rich (who accumulate it). This one-way flow drives asset bubbles despite weak economies.`,
        keyPoints: [
          'Poor spend 100% of income on necessities',
          'Rich save/invest majority of income',
          'Stimulus → Poor → Spending → Corporate Profits → Rich',
          'Rich accumulate cash, need somewhere to put it',
          'Result: Asset prices rise regardless of economy'
        ],
        example: {
          scenario: 'COVID Stimulus 2020',
          consensus: 'Stimulus will boost consumer spending and economy',
          reality: 'Money flowed through poor to rich, who bought assets',
          trade: 'Long gold, stocks, crypto despite recession',
          outcome: 'All assets mooned while real economy struggled'
        }
      }
    },
    {
      id: 'lesson3',
      title: 'Identifying Consensus Blind Spots',
      duration: '18 min',
      completed: completedLessons.has('lesson3'),
      content: {
        theory: `Consensus becomes wrong when it ignores inequality. Look for situations where standard models assume broad-based effects but inequality will concentrate the impact. The bigger the wealth gap, the more wrong consensus becomes.`,
        keyPoints: [
          'Find topics where distribution matters but is ignored',
          'Look for "average" statistics hiding extreme distributions',
          'Question assumptions about consumer behavior',
          'Identify policies that affect rich/poor differently',
          'Spot wealth concentration accelerators'
        ],
        example: {
          scenario: 'Interest Rate Hikes 2022-2023',
          consensus: 'High rates will crash housing and stocks',
          reality: 'Rich cash buyers support prices, poor priced out',
          trade: 'Housing and luxury goods stay strong',
          outcome: 'Prices stayed elevated despite 5% rates'
        }
      }
    },
    {
      id: 'lesson4',
      title: 'The Barbell Strategy for Safety + Upside',
      duration: '22 min',
      completed: completedLessons.has('lesson4'),
      content: {
        theory: `The barbell protects capital while hunting for massive wins. 80% stays in safe, liquid assets to survive anything. 20% bets on high-conviction contrarian opportunities where inequality blindness creates huge mispricings.`,
        keyPoints: [
          '80% conservative: index funds, bonds, gold',
          '20% aggressive: concentrated contrarian bets',
          'Never risk more than you can afford to lose',
          'Size positions based on conviction and payoff',
          'Rebalance periodically to maintain ratios'
        ],
        example: {
          scenario: 'Portfolio Construction 2024',
          consensus: 'Diversify equally across all sectors',
          reality: 'Inequality makes some sectors uninvestable',
          trade: '80% safe assets, 20% in 3-4 contrarian bets',
          outcome: 'Protected capital with 10x upside potential'
        }
      }
    },
    {
      id: 'lesson5',
      title: 'Position Sizing with Kelly Criterion',
      duration: '25 min',
      completed: completedLessons.has('lesson5'),
      content: {
        theory: `Kelly Criterion optimizes bet size based on edge and odds. For inequality trades with high conviction and big payoffs, Kelly suggests larger positions. But always use fractional Kelly (25-50%) for safety.`,
        keyPoints: [
          'Kelly formula: f = (p*b - q) / b',
          'p = win probability, b = payoff, q = loss probability',
          'Full Kelly is too aggressive for real trading',
          'Use Kelly/4 for safety with good returns',
          'Never bet more than max loss tolerance'
        ],
        example: {
          scenario: 'Sizing a Contrarian Bet',
          consensus: '2% position size for all trades',
          reality: '80% conviction, 5:1 payoff suggests bigger bet',
          trade: 'Kelly says 12%, use 3% for safety',
          outcome: 'Optimal risk/reward with downside protection'
        }
      }
    }
  ];

  // Simulation data showing inequality effects
  const simulationData: SimulationData[] = [
    { year: 2010, gini: 0.80, assetPrices: 100, medianWage: 100, top1Wealth: 25 },
    { year: 2012, gini: 0.82, assetPrices: 120, medianWage: 102, top1Wealth: 28 },
    { year: 2014, gini: 0.83, assetPrices: 145, medianWage: 103, top1Wealth: 31 },
    { year: 2016, gini: 0.84, assetPrices: 170, medianWage: 105, top1Wealth: 33 },
    { year: 2018, gini: 0.85, assetPrices: 200, medianWage: 106, top1Wealth: 35 },
    { year: 2020, gini: 0.87, assetPrices: 250, medianWage: 104, top1Wealth: 38 },
    { year: 2022, gini: 0.88, assetPrices: 320, medianWage: 102, top1Wealth: 42 },
    { year: 2024, gini: 0.89, assetPrices: 380, medianWage: 100, top1Wealth: 45 }
  ];

  const completeLesson = (lessonId: string) => {
    setCompletedLessons(new Set([...completedLessons, lessonId]));
  };

  const progressPercentage = (completedLessons.size / lessons.length) * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Trading on Inequality - The Gary Method</h2>
            <p className="text-purple-100">
              Master the art of finding what everyone else is missing
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{progressPercentage.toFixed(0)}%</div>
            <div className="text-sm text-purple-200">Complete</div>
          </div>
        </div>
        <div className="mt-4 bg-white/20 rounded-full h-3">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progressPercentage}%` }}
            transition={{ duration: 0.5 }}
            className="bg-white h-3 rounded-full"
          />
        </div>
      </div>

      {/* Lessons Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {lessons.map((lesson, index) => (
          <motion.div
            key={lesson.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.02 }}
            className={`bg-white dark:bg-gray-800 rounded-lg p-5 cursor-pointer border-2 ${
              lesson.completed ? 'border-green-500' : 'border-transparent'
            } hover:border-purple-500 transition-all`}
            onClick={() => setActiveLesson(lesson)}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${
                  lesson.completed ? 'bg-green-100 text-green-600' : 'bg-purple-100 text-purple-600'
                }`}>
                  {lesson.completed ? <CheckCircle /> : <BookOpen />}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                    Lesson {index + 1}: {lesson.title}
                  </h3>
                  <p className="text-sm text-gray-500">{lesson.duration}</p>
                </div>
              </div>
              <PlayCircle />
            </div>

            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              {lesson.content.theory.substring(0, 100)}...
            </p>

            {lesson.completed && (
              <div className="flex items-center text-green-600 text-sm font-medium">
                <CheckCircle />
                <span className="ml-1">Completed</span>
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Interactive Simulation */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            Inequality Simulation: Watch Wealth Concentrate
          </h3>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSimulationRunning(!simulationRunning)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700"
          >
            {simulationRunning ? 'Pause' : 'Run'} Simulation
          </motion.button>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={simulationData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="year" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                border: 'none',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="assetPrices"
              stroke="#8b5cf6"
              strokeWidth={3}
              name="Asset Prices"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="medianWage"
              stroke="#ef4444"
              strokeWidth={2}
              name="Median Wage"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="top1Wealth"
              stroke="#10b981"
              strokeWidth={2}
              name="Top 1% Wealth %"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-sm text-purple-700 dark:text-purple-300">
            <strong>Key Insight:</strong> As inequality rises (green line), asset prices (purple)
            explode while median wages (red) stagnate. This divergence creates massive trading
            opportunities when consensus expects them to move together.
          </p>
        </div>
      </div>

      {/* Case Studies */}
      <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-lg p-6">
        <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">
          Real-World Case Studies
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <div className="text-lg font-bold text-purple-600 mb-2">2008 Crisis</div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Consensus: Quick recovery
            </p>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Gary's Trade: Long bonds, rates stay zero
            </p>
            <div className="mt-2 text-green-600 font-bold">Result: +400%</div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <div className="text-lg font-bold text-purple-600 mb-2">COVID 2020</div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Consensus: Asset crash
            </p>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Gary's Trade: Long gold, stocks, crypto
            </p>
            <div className="mt-2 text-green-600 font-bold">Result: +250%</div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <div className="text-lg font-bold text-purple-600 mb-2">2023 Banking</div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Consensus: Systemic collapse
            </p>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Gary's Trade: Buy the dip, wealth protects
            </p>
            <div className="mt-2 text-green-600 font-bold">Result: +80%</div>
          </div>
        </div>
      </div>

      {/* Lesson Detail Modal */}
      <AnimatePresence>
        {activeLesson && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={() => setActiveLesson(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Lesson Header */}
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                    {activeLesson.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400">{activeLesson.duration}</p>
                </div>
                <button
                  onClick={() => setActiveLesson(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>

              {/* Theory */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-gray-100">
                  Core Concept
                </h3>
                <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                  {activeLesson.content.theory}
                </p>
              </div>

              {/* Key Points */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-gray-100">
                  Key Points
                </h3>
                <ul className="space-y-2">
                  {activeLesson.content.keyPoints.map((point, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-purple-600 mt-1">•</span>
                      <span className="text-gray-700 dark:text-gray-300">{point}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Example */}
              {activeLesson.content.example && (
                <div className="mb-6 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-3 text-purple-900 dark:text-purple-100">
                    Real Example: {activeLesson.content.example.scenario}
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="font-medium text-gray-700 dark:text-gray-300">Consensus: </span>
                      <span className="text-red-600">{activeLesson.content.example.consensus}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700 dark:text-gray-300">Reality: </span>
                      <span className="text-green-600">{activeLesson.content.example.reality}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700 dark:text-gray-300">Trade: </span>
                      <span className="text-purple-600">{activeLesson.content.example.trade}</span>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700 dark:text-gray-300">Outcome: </span>
                      <span className="font-bold text-gray-900 dark:text-gray-100">
                        {activeLesson.content.example.outcome}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Complete Button */}
              <div className="flex gap-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    completeLesson(activeLesson.id);
                    setActiveLesson(null);
                  }}
                  className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold"
                >
                  {activeLesson.completed ? 'Review Complete' : 'Mark as Complete'}
                </motion.button>
                <button
                  onClick={() => setActiveLesson(null)}
                  className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-semibold"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Final Quote */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6 rounded-lg">
        <div className="flex items-start gap-4">
          <LightBulb />
          <div>
            <h3 className="text-xl font-bold mb-2">Remember Gary's Golden Rule</h3>
            <p className="text-purple-100">
              "You only need to identify ONE thing economists are missing. Growing inequality of wealth
              is destroying the economy. Everything that happens can be predicted by understanding what
              will happen with growing inequality. The more people disagree with you, the more money
              you're going to make."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LearnInequality;