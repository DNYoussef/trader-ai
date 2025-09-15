import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

// Simplified version without framer-motion for now
export const ContrarianTradesWrapper: React.FC = () => {
  const signals = [
    {
      symbol: 'SPY',
      signal: 'SELL',
      crowdSentiment: 'Bullish',
      talebPrinciple: 'Fragility detected - negative convexity',
      confidence: 82
    },
    {
      symbol: 'ULTY',
      signal: 'BUY',
      crowdSentiment: 'Bearish',
      talebPrinciple: 'Antifragile - positive optionality',
      confidence: 91
    },
    {
      symbol: 'GLD',
      signal: 'BUY',
      crowdSentiment: 'Neutral',
      talebPrinciple: 'Barbell strategy - tail hedge',
      confidence: 78
    },
    {
      symbol: 'VIX',
      signal: 'BUY',
      crowdSentiment: 'Bearish',
      talebPrinciple: 'Black swan protection',
      confidence: 95
    }
  ];

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-100 text-green-700';
      case 'SELL': return 'bg-red-100 text-red-700';
      default: return 'bg-yellow-100 text-yellow-700';
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish': return 'text-green-600';
      case 'Bearish': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Contrarian Trade Signals</CardTitle>
        <p className="text-sm text-gray-600">Taleb's Antifragility + Crowd Psychology</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {signals.map((signal) => (
            <div key={signal.symbol} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h4 className="font-semibold text-lg">{signal.symbol}</h4>
                  <p className="text-sm text-gray-600">
                    Crowd: <span className={getSentimentColor(signal.crowdSentiment)}>
                      {signal.crowdSentiment}
                    </span>
                  </p>
                </div>
                <div className="text-right">
                  <Badge className={getSignalColor(signal.signal)} variant="secondary">
                    {signal.signal}
                  </Badge>
                  <p className="text-sm mt-1">
                    {signal.confidence}% confidence
                  </p>
                </div>
              </div>

              <div className="mt-3 p-3 bg-gray-50 rounded">
                <p className="text-sm font-medium text-gray-700">Taleb Principle:</p>
                <p className="text-sm text-gray-600">{signal.talebPrinciple}</p>
              </div>

              {signal.signal !== 'HOLD' && (
                <div className="mt-3 flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    Contrarian: {signal.signal === 'BUY' && signal.crowdSentiment === 'Bearish' ? '✓' :
                               signal.signal === 'SELL' && signal.crowdSentiment === 'Bullish' ? '✓' : '○'}
                  </span>
                  <button className="text-sm text-blue-600 hover:text-blue-800">
                    Execute Trade →
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 p-4 bg-purple-50 rounded-lg">
          <h4 className="font-semibold mb-2">🦢 Black Swan Alert</h4>
          <p className="text-sm text-gray-700">
            Current market shows high fragility. Consider increasing tail hedges
            and reducing leverage. The crowd's extreme bullishness on tech suggests
            a potential regime change approaching.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};