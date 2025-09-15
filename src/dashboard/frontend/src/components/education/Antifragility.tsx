import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface BarbellPosition {
  id: string;
  asset: string;
  allocation: number;
  category: 'safe' | 'aggressive';
  expectedReturn: number;
  maxLoss: number;
  blackSwanExposure: number;
}

export const Antifragility: React.FC = () => {
  const [portfolio, setPortfolio] = useState<BarbellPosition[]>([
    { id: '1', asset: 'Treasury Bonds', allocation: 45, category: 'safe', expectedReturn: 2, maxLoss: -5, blackSwanExposure: -2 },
    { id: '2', asset: 'FDIC Cash', allocation: 35, category: 'safe', expectedReturn: 1, maxLoss: 0, blackSwanExposure: 0 },
    { id: '3', asset: 'Gold ETF', allocation: 10, category: 'safe', expectedReturn: 3, maxLoss: -10, blackSwanExposure: 5 },
    { id: '4', asset: 'Options (OTM Calls)', allocation: 5, category: 'aggressive', expectedReturn: -50, maxLoss: -100, blackSwanExposure: 500 },
    { id: '5', asset: 'Crypto Moonshots', allocation: 5, category: 'aggressive', expectedReturn: -80, maxLoss: -100, blackSwanExposure: 1000 }
  ]);

  const [selectedDomain, setSelectedDomain] = useState<'mediocristan' | 'extremistan'>('mediocristan');
  const [simulationResults, setSimulationResults] = useState<any[]>([]);

  // Calculate portfolio metrics
  const calculatePortfolioMetrics = () => {
    const safeAllocation = portfolio.filter(p => p.category === 'safe').reduce((sum, p) => sum + p.allocation, 0);
    const aggressiveAllocation = portfolio.filter(p => p.category === 'aggressive').reduce((sum, p) => sum + p.allocation, 0);

    const expectedReturn = portfolio.reduce((sum, p) => sum + (p.allocation * p.expectedReturn / 100), 0);
    const maxDrawdown = portfolio.reduce((sum, p) => sum + (p.allocation * p.maxLoss / 100), 0);
    const blackSwanPotential = portfolio.reduce((sum, p) => sum + (p.allocation * p.blackSwanExposure / 100), 0);

    return {
      safeAllocation,
      aggressiveAllocation,
      expectedReturn,
      maxDrawdown,
      blackSwanPotential,
      isAntifragile: safeAllocation >= 80 && aggressiveAllocation <= 20 && blackSwanPotential > 0
    };
  };

  const metrics = calculatePortfolioMetrics();

  // Simulate black swan events
  const simulateBlackSwan = () => {
    const results = [];
    const baseValue = 100000;

    for (let i = 0; i < 100; i++) {
      const isBlackSwan = Math.random() < 0.01; // 1% chance
      const marketCrash = Math.random() < 0.05; // 5% chance

      let value = baseValue;

      portfolio.forEach(position => {
        if (isBlackSwan && position.blackSwanExposure > 100) {
          // Positive black swan for aggressive positions
          value += (position.allocation / 100) * baseValue * (position.blackSwanExposure / 100);
        } else if (marketCrash) {
          // Market crash affects all positions
          value += (position.allocation / 100) * baseValue * (position.maxLoss / 100);
        } else {
          // Normal returns
          value += (position.allocation / 100) * baseValue * (position.expectedReturn / 100);
        }
      });

      results.push({
        month: i,
        value: value,
        event: isBlackSwan ? 'black_swan' : marketCrash ? 'crash' : 'normal'
      });
    }

    setSimulationResults(results);
  };

  // Mediocristan vs Extremistan examples
  const domainExamples = {
    mediocristan: [
      { name: 'Height', range: '4-7 ft', outlierImpact: 'Minimal', example: 'Tallest person adds <1% to average' },
      { name: 'Weight', range: '50-500 lbs', outlierImpact: 'Low', example: 'Heaviest person has limited impact' },
      { name: 'Daily Returns', range: '-3% to +3%', outlierImpact: 'Moderate', example: 'Single day rarely ruins portfolio' }
    ],
    extremistan: [
      { name: 'Wealth', range: '$0 to $200B+', outlierImpact: 'Extreme', example: 'Top 1% owns 50% of wealth' },
      { name: 'Book Sales', range: '0 to 500M+', outlierImpact: 'Massive', example: 'Harry Potter outsells 99.9% of books' },
      { name: 'Pandemic Impact', range: 'Local to Global', outlierImpact: 'Catastrophic', example: 'COVID-19 changed everything' }
    ]
  };

  const updateAllocation = (id: string, newAllocation: number) => {
    setPortfolio(prev => prev.map(p =>
      p.id === id ? { ...p, allocation: Math.max(0, Math.min(100, newAllocation)) } : p
    ));
  };

  return (
    <div className="space-y-6">
      {/* Barbell Strategy Builder */}
      <Card>
        <CardHeader>
          <CardTitle>⚖️ Barbell Portfolio Builder</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Portfolio Status */}
            <div className={`p-4 rounded-lg ${metrics.isAntifragile ? 'bg-green-50' : 'bg-yellow-50'}`}>
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-semibold">Portfolio Status</h3>
                <Badge variant={metrics.isAntifragile ? 'default' : 'secondary'}>
                  {metrics.isAntifragile ? '✅ Antifragile' : '⚠️ Fragile'}
                </Badge>
              </div>
              <div className="grid grid-cols-4 gap-4 mt-3">
                <div>
                  <p className="text-xs text-gray-600">Safe Assets</p>
                  <p className="text-lg font-bold">{metrics.safeAllocation}%</p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Risk Assets</p>
                  <p className="text-lg font-bold">{metrics.aggressiveAllocation}%</p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Expected Return</p>
                  <p className="text-lg font-bold">{metrics.expectedReturn.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-xs text-gray-600">Black Swan Upside</p>
                  <p className="text-lg font-bold text-green-600">+{metrics.blackSwanPotential.toFixed(0)}%</p>
                </div>
              </div>
            </div>

            {/* Allocation Controls */}
            <div className="space-y-3">
              <h4 className="font-medium">Adjust Allocations</h4>
              {portfolio.map(position => (
                <div key={position.id} className="p-3 border rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{position.asset}</span>
                      <Badge variant={position.category === 'safe' ? 'secondary' : 'destructive'}>
                        {position.category}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        value={position.allocation}
                        onChange={(e) => updateAllocation(position.id, parseInt(e.target.value))}
                        className="w-16 px-2 py-1 border rounded text-right"
                        min="0"
                        max="100"
                      />
                      <span className="text-sm text-gray-600">%</span>
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-gray-600">
                    <span>Return: {position.expectedReturn}%</span>
                    <span>Max Loss: {position.maxLoss}%</span>
                    <span>Black Swan: +{position.blackSwanExposure}%</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Barbell Visualization */}
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={portfolio}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="asset" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="allocation" fill={(entry: any) => entry.category === 'safe' ? '#10b981' : '#ef4444'} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Mediocristan vs Extremistan */}
      <Card>
        <CardHeader>
          <CardTitle>🌍 Domain Classification: Mediocristan vs Extremistan</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Domain Selector */}
            <div className="flex gap-2 p-1 bg-gray-100 rounded-lg">
              <button
                onClick={() => setSelectedDomain('mediocristan')}
                className={`flex-1 py-2 px-4 rounded-md transition-colors ${
                  selectedDomain === 'mediocristan' ? 'bg-white shadow' : ''
                }`}
              >
                Mediocristan
              </button>
              <button
                onClick={() => setSelectedDomain('extremistan')}
                className={`flex-1 py-2 px-4 rounded-md transition-colors ${
                  selectedDomain === 'extremistan' ? 'bg-white shadow' : ''
                }`}
              >
                Extremistan
              </button>
            </div>

            {/* Domain Examples */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold mb-3">
                {selectedDomain === 'mediocristan' ? '📊 Mediocristan (Bell Curve World)' : '🚀 Extremistan (Power Law World)'}
              </h4>
              <div className="space-y-2">
                {domainExamples[selectedDomain].map((example, idx) => (
                  <div key={idx} className="p-3 bg-white rounded border">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="font-medium">{example.name}</p>
                        <p className="text-sm text-gray-600 mt-1">{example.example}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-gray-500">Range</p>
                        <p className="text-sm font-medium">{example.range}</p>
                        <p className="text-xs text-gray-500 mt-1">Outlier Impact</p>
                        <p className={`text-sm font-medium ${
                          example.outlierImpact === 'Extreme' || example.outlierImpact === 'Catastrophic'
                            ? 'text-red-600'
                            : example.outlierImpact === 'Massive'
                            ? 'text-orange-600'
                            : 'text-green-600'
                        }`}>
                          {example.outlierImpact}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Key Insight */}
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm">
                <strong>💡 Key Insight:</strong> {
                  selectedDomain === 'mediocristan'
                    ? "In Mediocristan, no single observation can meaningfully change the aggregate. Traditional statistics work well here."
                    : "In Extremistan, a single observation can dwarf all others combined. This is where black swans live and traditional models fail."
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Black Swan Simulator */}
      <Card>
        <CardHeader>
          <CardTitle>🦢 Black Swan Event Simulator</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button onClick={simulateBlackSwan} className="w-full">
              Run 100-Month Simulation
            </Button>

            {simulationResults.length > 0 && (
              <>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={simulationResults}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={(props: any) => {
                          const { cx, cy, payload } = props;
                          if (payload.event === 'black_swan') {
                            return <circle cx={cx} cy={cy} r={6} fill="#10b981" />;
                          } else if (payload.event === 'crash') {
                            return <circle cx={cx} cy={cy} r={4} fill="#ef4444" />;
                          }
                          return null;
                        }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex gap-4 justify-center">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span className="text-sm">Positive Black Swan</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span className="text-sm">Market Crash</span>
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm">
                    <strong>Simulation Results:</strong> The barbell strategy shows limited downside during crashes
                    but massive upside during positive black swan events. This asymmetry is the essence of antifragility.
                  </p>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};