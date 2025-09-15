import React from 'react';
import './index.css';
import { EnhancedUXProvider } from '../components/enhanced/EnhancedUXProvider';
import TabbedDashboard from '../components/enhanced/TabbedDashboard';

/**
 * Enhanced App with Mobile Psychology Integration
 *
 * This is the main entry point for the enhanced trading dashboard
 * with all psychological reinforcement and professional trading features.
 */

const AppEnhanced: React.FC = () => {
  // Get user ID from localStorage or generate new one
  const getUserId = () => {
    let userId = localStorage.getItem('trading_user_id');
    if (!userId) {
      userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('trading_user_id', userId);
    }
    return userId;
  };

  const userId = getUserId();
  console.log('AppEnhanced rendering with userId:', userId);

  return (
    <EnhancedUXProvider>
      <div style={{ padding: '20px', backgroundColor: '#f3f4f6', minHeight: '100vh' }}>
        <h1>Gary×Taleb Trading System</h1>
        <TabbedDashboard
          userId={userId}
          userLevel="beginner"
          defaultTab="overview"
        />
      </div>
    </EnhancedUXProvider>
  );
};

export default AppEnhanced;