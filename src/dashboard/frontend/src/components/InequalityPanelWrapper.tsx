import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';

// Simplified version without framer-motion for now
export const InequalityPanelWrapper: React.FC = () => {
  const metrics = [
    { label: 'Wealth Concentration', value: 87.3, trend: 'up', description: 'Top 1% wealth share increasing' },
    { label: 'Interest Rate Spread', value: 4.2, trend: 'stable', description: 'Rich vs poor borrowing rates' },
    { label: 'Asset Inflation', value: 12.5, trend: 'up', description: 'Financial assets vs wages growth' },
    { label: 'DPI Score', value: 78, trend: 'up', description: "Gary's Debt Pyramid Index" }
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Inequality Analysis</CardTitle>
        <p className="text-sm text-gray-600">Gary Stevenson's Trading Framework</p>
      </CardHeader>
      <CardContent className="space-y-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="space-y-2">
            <div className="flex justify-between items-center">
              <div>
                <span className="font-medium">{metric.label}</span>
                <p className="text-sm text-gray-600">{metric.description}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold">{metric.value}%</span>
                <span className={`text-xl ${metric.trend === 'up' ? 'text-green-600' : 'text-gray-600'}`}>
                  {metric.trend === 'up' ? '↑' : '→'}
                </span>
              </div>
            </div>
            <Progress value={metric.value} className="h-2" />
          </div>
        ))}

        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
          <h4 className="font-semibold mb-2">Trading Insight</h4>
          <p className="text-sm text-gray-700">
            When wealth concentration exceeds 85%, bet on asset inflation.
            The rich getting richer means assets will keep rising regardless of economic fundamentals.
          </p>
        </div>

        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Wealth Gap: Widening</Badge>
          <Badge variant="outline">Signal: Bullish Assets</Badge>
          <Badge variant="outline">Risk: Systemic</Badge>
        </div>
      </CardContent>
    </Card>
  );
};