import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface AISignalsProps {
  dpiScore?: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

export const AISignals: React.FC<AISignalsProps> = ({
  dpiScore = 85,
  signal = 'BUY',
  confidence = 92
}) => {
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-100 text-green-700';
      case 'SELL': return 'bg-red-100 text-red-700';
      case 'HOLD': return 'bg-yellow-100 text-yellow-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getDPIColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">AI Signals</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">DPI Score</div>
          <div className={`text-3xl font-bold ${getDPIColor(dpiScore)}`}>
            {dpiScore}%
          </div>
        </div>

        <div className="text-center">
          <div className="text-sm text-gray-600 mb-2">Signal</div>
          <Badge className={`${getSignalColor(signal)} text-xl px-4 py-2`}>
            {signal}
          </Badge>
        </div>

        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Confidence</div>
          <div className="text-lg font-semibold">{confidence}%</div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>

        <div className="pt-2 text-xs text-gray-500 text-center">
          Based on Gary's DPI methodology and Taleb's antifragility principles
        </div>
      </CardContent>
    </Card>
  );
};