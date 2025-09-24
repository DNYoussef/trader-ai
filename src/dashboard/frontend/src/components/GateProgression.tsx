import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { useTradingData } from '../hooks/useTradingData';

interface Gate {
  id: string;
  name: string;
  range: string;
  status: 'completed' | 'current' | 'locked';
  requirements: string;
  progress?: number;
}

interface GateProgressionProps {
  // Props are now optional as we fetch data
}

export const GateProgression: React.FC<GateProgressionProps> = () => {
  const { gateStatus, metrics } = useTradingData();
  const [gates, setGates] = useState<Gate[]>([]);
  const [currentCapital, setCurrentCapital] = useState<number>(0);

  useEffect(() => {
    if (gateStatus?.gates) {
      // Use real gate data from API
      setGates(gateStatus.gates);
      setCurrentCapital(gateStatus.current_capital);
    } else if (metrics?.portfolio_value) {
      // Fallback to metrics portfolio value if gate status not available
      setCurrentCapital(metrics.portfolio_value);

      // Generate default gates if not available from API
      const defaultGates = generateDefaultGates(metrics.portfolio_value);
      setGates(defaultGates);
    } else {
      // Use hardcoded fallback when no data available
      setGates(getFallbackGates());
      setCurrentCapital(342);
    }
  }, [gateStatus, metrics]);

  // Helper function to generate gates based on current capital
  const generateDefaultGates = (capital: number): Gate[] => {
    const gateRanges = [
      { id: 'G0', min: 200, max: 499 },
      { id: 'G1', min: 500, max: 999 },
      { id: 'G2', min: 1000, max: 2499 },
      { id: 'G3', min: 2500, max: 4999 },
      { id: 'G4', min: 5000, max: 9999 },
      { id: 'G5', min: 10000, max: 24999 },
      { id: 'G6', min: 25000, max: 49999 }
    ];

    return gateRanges.map(gate => {
      let status: 'completed' | 'current' | 'locked' = 'locked';
      let progress = 0;

      if (capital >= gate.max) {
        status = 'completed';
      } else if (capital >= gate.min && capital < gate.max) {
        status = 'current';
        progress = ((capital - gate.min) / (gate.max - gate.min)) * 100;
      }

      return {
        id: gate.id,
        name: `Gate ${gate.id}`,
        range: `$${gate.min.toLocaleString()}-$${gate.max.toLocaleString()}`,
        status,
        requirements: status === 'completed'
          ? 'Completed!'
          : `Reach $${gate.min.toLocaleString()} capital`,
        progress: status === 'current' ? progress : undefined
      };
    });
  };

  // Fallback gates when no data is available
  const getFallbackGates = (): Gate[] => [
    {
      id: 'G0',
      name: 'Gate G0',
      range: '$200-499',
      status: 'completed',
      requirements: 'Initial gate completed!'
    },
    {
      id: 'G1',
      name: 'Gate G1',
      range: '$500-999',
      status: 'current',
      requirements: 'Building capital...',
      progress: 68
    },
    {
      id: 'G2',
      name: 'Gate G2',
      range: '$1,000-2,499',
      status: 'locked',
      requirements: 'Reach $1,000 capital'
    },
    {
      id: 'G3',
      name: 'Gate G3',
      range: '$2,500-4,999',
      status: 'locked',
      requirements: 'Reach $2,500 capital'
    },
    {
      id: 'G4',
      name: 'Gate G4',
      range: '$5,000+',
      status: 'locked',
      requirements: 'Reach $5,000 capital'
    }
  ];

  // Display only the first 5 gates for UI consistency
  const displayGates = gates.slice(0, 5);
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500 text-white';
      case 'current': return 'bg-blue-500 text-white';
      case 'locked': return 'bg-gray-300 text-gray-600';
      default: return 'bg-gray-300 text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'current': return '→';
      case 'locked': return '🔒';
      default: return '';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Gate Progression</CardTitle>
        <p className="text-sm text-gray-600">Current Capital: ${currentCapital}</p>
      </CardHeader>
      <CardContent className="space-y-4">
        {displayGates.map((gate) => (
          <div key={gate.id} className="space-y-2">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${getStatusColor(gate.status)}`}>
                  {gate.status === 'completed' || gate.status === 'current' ? gate.id : getStatusIcon(gate.status)}
                </div>
                <div>
                  <h3 className="font-medium">{gate.name}</h3>
                  <p className="text-sm text-gray-600">{gate.range}</p>
                </div>
              </div>
              <Badge className={getStatusColor(gate.status)} variant="secondary">
                {gate.status}
              </Badge>
            </div>

            <div className="ml-11">
              <p className="text-sm text-gray-600 mb-1">{gate.requirements}</p>
              {gate.progress && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Progress</span>
                    <span>{gate.progress}%</span>
                  </div>
                  <Progress value={gate.progress} className="h-2" />
                </div>
              )}
            </div>
          </div>
        ))}

        <div className="pt-2 text-center">
          <button className="text-sm text-blue-600 hover:text-blue-800">
            Switch to Overview
          </button>
        </div>
      </CardContent>
    </Card>
  );
};