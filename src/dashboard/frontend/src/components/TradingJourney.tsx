import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface JourneyStats {
  totalTrades: number;
  winRate: number;
  avgProfit: number;
  bestMonth: string;
  riskScore: string;
  learningHours: number;
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  earned: string;
}

interface TradingJourneyProps {
  stats?: JourneyStats;
  achievements?: Achievement[];
}

export const TradingJourney: React.FC<TradingJourneyProps> = ({
  stats = {
    totalTrades: 147,
    winRate: 67,
    avgProfit: 342,
    bestMonth: '+12.3%',
    riskScore: 'A+',
    learningHours: 24
  },
  achievements = [
    {
      id: '1',
      title: 'First Trade',
      description: 'Executed your first trade',
      icon: '🎯',
      earned: '3 days ago'
    },
    {
      id: '2',
      title: 'Risk Master',
      description: 'Maintained P(ruin) < 10% for 30 days',
      icon: '🛡️',
      earned: '1 week ago'
    },
    {
      id: '3',
      title: 'Profit Maker',
      description: 'Achieved 5 consecutive profitable trades',
      icon: '🏆',
      earned: '2 weeks ago'
    },
    {
      id: '4',
      title: 'Knowledge Seeker',
      description: 'Completed 3 educational courses',
      icon: '📚',
      earned: '1 month ago'
    }
  ]
}) => {
  const statCards = [
    { icon: '📊', label: 'Total Trades', value: stats.totalTrades },
    { icon: '🎯', label: 'Win Rate', value: `${stats.winRate}%` },
    { icon: '💰', label: 'Avg Profit', value: `$${stats.avgProfit}` },
    { icon: '🏆', label: 'Best Month', value: stats.bestMonth },
    { icon: '🛡️', label: 'Risk Score', value: stats.riskScore },
    { icon: '📚', label: 'Learning Hours', value: stats.learningHours }
  ];

  return (
    <div className="space-y-6">
      {/* Journey Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Your Trading Journey</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {statCards.map((stat, index) => (
              <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl mb-1">{stat.icon}</div>
                <div className="text-sm text-gray-600 mb-1">{stat.label}</div>
                <div className="text-xl font-bold">{stat.value}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Achievements */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Achievements</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {achievements.map((achievement) => (
            <div key={achievement.id} className="flex items-center gap-4 p-3 bg-yellow-50 rounded-lg">
              <div className="text-2xl">{achievement.icon}</div>
              <div className="flex-1">
                <h4 className="font-semibold">{achievement.title}</h4>
                <p className="text-sm text-gray-600">{achievement.description}</p>
              </div>
              <Badge variant="outline" className="text-xs">
                {achievement.earned}
              </Badge>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};