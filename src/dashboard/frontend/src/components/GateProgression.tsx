import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

interface Gate {
  id: string;
  name: string;
  range: string;
  status: 'completed' | 'current' | 'locked';
  requirements: string;
  progress?: number;
}

interface GateProgressionProps {
  currentCapital?: number;
  gates?: Gate[];
}

export const GateProgression: React.FC<GateProgressionProps> = ({
  currentCapital = 342,
  gates = [
    {
      id: 'G0',
      name: 'Gate G0',
      range: '$200-499',
      status: 'completed',
      requirements: 'Great job!'
    },
    {
      id: 'G1',
      name: 'Gate G1',
      range: '$500-999',
      status: 'current',
      requirements: '4 more profitable trades',
      progress: 68
    },
    {
      id: 'G2',
      name: 'Gate G2',
      range: '$1,000-2,499',
      status: 'locked',
      requirements: '3 more profitable trades'
    },
    {
      id: 'G3',
      name: 'Gate G3',
      range: '$2,500-4,999',
      status: 'locked',
      requirements: '2 more profitable trades'
    },
    {
      id: 'G4',
      name: 'Gate G4',
      range: '$5,000+',
      status: 'locked',
      requirements: '1 more profitable trades'
    }
  ]
}) => {
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
        {gates.map((gate) => (
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