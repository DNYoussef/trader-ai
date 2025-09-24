import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { useAIData } from '../hooks/useAIData';

interface AISignalsProps {
  // Props are now optional as we fetch data
}

export const AISignals: React.FC<AISignalsProps> = () => {
  const { enhanced32D, getAggregateSignals, loading, error } = useAIData(5000); // 5 second refresh
  const [dpiScore, setDpiScore] = useState<number>(0);
  const [signal, setSignal] = useState<'BUY' | 'SELL' | 'HOLD'>('HOLD');
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    // Get aggregate signals from all AI components
    const aggregateSignals = getAggregateSignals();

    // Update state with real AI data
    setDpiScore(aggregateSignals.dpi_score || enhanced32D?.dpi_score || 75);
    setSignal(aggregateSignals.signal);
    setConfidence(aggregateSignals.confidence);
  }, [enhanced32D, getAggregateSignals]);
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

        {loading && (
          <div className="text-center text-sm text-gray-500">
            Loading AI signals...
          </div>
        )}

        {error && (
          <div className="text-center text-xs text-red-500">
            {error}
          </div>
        )}

        <div className="pt-2 text-xs text-gray-500 text-center">
          Based on 5 AI Components: TimesFM, FinGPT & 32D Features
        </div>
      </CardContent>
    </Card>
  );
};