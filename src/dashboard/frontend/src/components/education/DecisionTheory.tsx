import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';

interface DecisionNode {
  id: string;
  question: string;
  options: {
    label: string;
    value: number;
    probability: number;
    nextNode?: string;
  }[];
  isTerminal?: boolean;
  expectedValue?: number;
}

export const DecisionTheory: React.FC = () => {
  const [currentNode, setCurrentNode] = useState('root');
  const [certainEquivalent, setCertainEquivalent] = useState(0);
  const [riskAversion, setRiskAversion] = useState(0.5);
  const [selectedPath, setSelectedPath] = useState<string[]>([]);

  // Decision tree for trade evaluation
  const decisionTree: Record<string, DecisionNode> = {
    root: {
      id: 'root',
      question: 'Market showing bullish signals. Enter position?',
      options: [
        {
          label: 'Enter Long Position',
          value: 1000,
          probability: 0.65,
          nextNode: 'position_size'
        },
        {
          label: 'Wait for Confirmation',
          value: 0,
          probability: 0.35,
          nextNode: 'wait'
        }
      ]
    },
    position_size: {
      id: 'position_size',
      question: 'How much capital to allocate?',
      options: [
        {
          label: 'Conservative (2% of capital)',
          value: 200,
          probability: 0.8,
          nextNode: 'stop_loss'
        },
        {
          label: 'Standard (5% of capital)',
          value: 500,
          probability: 0.6,
          nextNode: 'stop_loss'
        },
        {
          label: 'Aggressive (10% of capital)',
          value: 1000,
          probability: 0.4,
          nextNode: 'stop_loss'
        }
      ]
    },
    stop_loss: {
      id: 'stop_loss',
      question: 'Set stop loss level?',
      options: [
        {
          label: 'Tight Stop (2% loss)',
          value: -40,
          probability: 0.3
        },
        {
          label: 'Standard Stop (5% loss)',
          value: -100,
          probability: 0.2
        },
        {
          label: 'Wide Stop (10% loss)',
          value: -200,
          probability: 0.1
        }
      ],
      isTerminal: true
    },
    wait: {
      id: 'wait',
      question: 'Continue monitoring for better entry?',
      options: [
        {
          label: 'Monitor for 1 hour',
          value: 50,
          probability: 0.5
        },
        {
          label: 'Set alert and wait',
          value: 0,
          probability: 0.7
        }
      ],
      isTerminal: true
    }
  };

  const calculateCertainEquivalent = (expectedValue: number, variance: number) => {
    // CE = E(X) - (λ/2) * Var(X)
    // where λ is risk aversion parameter
    return expectedValue - (riskAversion / 2) * variance;
  };

  const calculateExpectedValue = (node: DecisionNode) => {
    if (!node.options) return 0;

    return node.options.reduce((sum, option) => {
      return sum + (option.value * option.probability);
    }, 0);
  };

  const handleDecision = (option: any) => {
    setSelectedPath([...selectedPath, option.label]);
    if (option.nextNode) {
      setCurrentNode(option.nextNode);
    } else {
      // Calculate final outcome
      const ev = calculateExpectedValue(decisionTree[currentNode]);
      setCertainEquivalent(calculateCertainEquivalent(ev, 100));
    }
  };

  const resetDecisionTree = () => {
    setCurrentNode('root');
    setSelectedPath([]);
    setCertainEquivalent(0);
  };

  const node = decisionTree[currentNode];

  return (
    <div className="space-y-6">
      {/* Decision Tree Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>🌳 Interactive Decision Tree</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Current Decision */}
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">{node.question}</h3>

              {/* Decision Path */}
              {selectedPath.length > 0 && (
                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2">Decision Path:</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedPath.map((decision, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-white rounded-full text-sm border"
                      >
                        {decision}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Options */}
              <div className="space-y-3">
                {node.options.map((option, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleDecision(option)}
                    className="w-full p-4 text-left bg-white rounded-lg border hover:border-blue-400 transition-colors"
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{option.label}</p>
                        <p className="text-sm text-gray-600 mt-1">
                          Expected Value: ${option.value} • Probability: {(option.probability * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-semibold text-green-600">
                          ${(option.value * option.probability).toFixed(0)}
                        </p>
                        <p className="text-xs text-gray-500">Weighted Value</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Result */}
            {node.isTerminal && (
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold mb-2">Decision Analysis Complete</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Expected Value</p>
                    <p className="text-xl font-bold">${calculateExpectedValue(node).toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Certain Equivalent</p>
                    <p className="text-xl font-bold">${certainEquivalent.toFixed(2)}</p>
                  </div>
                </div>
                <Button onClick={resetDecisionTree} className="mt-4">
                  Try Another Scenario
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Certain Equivalent Calculator */}
      <Card>
        <CardHeader>
          <CardTitle>💰 Certain Equivalent Calculator</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Risk Aversion Level: {riskAversion.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={riskAversion}
                onChange={(e) => setRiskAversion(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>Risk Seeking</span>
                <span>Risk Neutral</span>
                <span>Risk Averse</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm text-gray-600">Your Profile</p>
                <p className="font-semibold">
                  {riskAversion < 0.3 ? 'Risk Seeker' : riskAversion < 0.7 ? 'Balanced' : 'Conservative'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Recommended Position</p>
                <p className="font-semibold">
                  {riskAversion < 0.3 ? '5-10%' : riskAversion < 0.7 ? '2-5%' : '1-2%'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Max Drawdown</p>
                <p className="font-semibold">
                  {riskAversion < 0.3 ? '20%' : riskAversion < 0.7 ? '10%' : '5%'}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* P(ruin) Scenarios */}
      <Card>
        <CardHeader>
          <CardTitle>🎲 Avoiding Ruin Scenarios</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Practice identifying and avoiding paths that lead to account ruin
            </p>

            {[
              {
                scenario: 'Overleveraging',
                pRuin: 0.35,
                description: 'Using 50% of capital on single trade',
                lesson: 'Never risk more than 5% per trade'
              },
              {
                scenario: 'No Stop Loss',
                pRuin: 0.28,
                description: 'Trading without protective stops',
                lesson: 'Always define risk before entry'
              },
              {
                scenario: 'Revenge Trading',
                pRuin: 0.42,
                description: 'Doubling down after losses',
                lesson: 'Stick to predetermined position sizes'
              },
              {
                scenario: 'Optimal Kelly',
                pRuin: 0.02,
                description: 'Using Kelly Criterion sizing',
                lesson: 'Mathematical position sizing works'
              }
            ].map(scenario => (
              <div key={scenario.scenario} className="p-4 border rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold">{scenario.scenario}</h4>
                  <span className={`px-2 py-1 rounded text-sm ${
                    scenario.pRuin > 0.2 ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  }`}>
                    P(ruin): {(scenario.pRuin * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-2">{scenario.description}</p>
                <p className="text-sm font-medium text-blue-600">📚 {scenario.lesson}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};