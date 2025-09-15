import React, { useState } from 'react';
import './index.css';

/**
 * Simplified Enhanced App - Testing version
 */

const AppEnhancedSimple: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'terminal', label: 'Trading Terminal', icon: '📈' },
    { id: 'learn', label: 'Learn', icon: '🧠' },
    { id: 'progress', label: 'Progress', icon: '🎯' }
  ];

  return (
    <div style={{ backgroundColor: '#f3f4f6', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{
        backgroundColor: 'white',
        padding: '20px',
        borderBottom: '1px solid #e5e7eb',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ margin: 0, color: '#111827', fontSize: '24px' }}>
          Gary×Taleb Trading System
        </h1>
        <p style={{ margin: '5px 0 0 0', color: '#6b7280' }}>
          Enhanced Trading Dashboard with AI Intelligence
        </p>
      </div>

      {/* Tabs */}
      <div style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e5e7eb',
        display: 'flex',
        gap: '0'
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '15px 25px',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #3b82f6' : '2px solid transparent',
              backgroundColor: activeTab === tab.id ? '#eff6ff' : 'white',
              color: activeTab === tab.id ? '#3b82f6' : '#6b7280',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s'
            }}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '30px' }}>
        {activeTab === 'overview' && (
          <div>
            <h2 style={{ color: '#111827', marginBottom: '20px' }}>Trading System Overview</h2>

            {/* Stats Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '30px'
            }}>
              {[
                { label: 'Current Gate', value: 'G0', color: '#3b82f6' },
                { label: 'Portfolio Value', value: '$25,432', color: '#10b981' },
                { label: 'Total Trades', value: '147', color: '#8b5cf6' },
                { label: 'Win Rate', value: '67%', color: '#f59e0b' }
              ].map(stat => (
                <div key={stat.label} style={{
                  backgroundColor: 'white',
                  padding: '20px',
                  borderRadius: '8px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <p style={{ margin: '0 0 10px 0', color: '#6b7280', fontSize: '14px' }}>
                    {stat.label}
                  </p>
                  <p style={{ margin: 0, color: stat.color, fontSize: '32px', fontWeight: 'bold' }}>
                    {stat.value}
                  </p>
                </div>
              ))}
            </div>

            {/* Quick Actions */}
            <div style={{
              backgroundColor: 'linear-gradient(to right, #eff6ff, #f3e8ff)',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              padding: '30px',
              borderRadius: '12px',
              color: 'white'
            }}>
              <h3 style={{ margin: '0 0 20px 0' }}>Quick Actions</h3>
              <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
                <button style={{
                  padding: '12px 24px',
                  backgroundColor: 'white',
                  color: '#3b82f6',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}>
                  Open Trading Terminal
                </button>
                <button style={{
                  padding: '12px 24px',
                  backgroundColor: 'rgba(255,255,255,0.2)',
                  color: 'white',
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}>
                  View Tutorial
                </button>
                <button style={{
                  padding: '12px 24px',
                  backgroundColor: 'rgba(255,255,255,0.2)',
                  color: 'white',
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}>
                  Check Progress
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'terminal' && (
          <div>
            <h2 style={{ color: '#111827' }}>Professional Trading Terminal</h2>
            <div style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              minHeight: '400px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#6b7280'
            }}>
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '48px', margin: '0 0 20px 0' }}>📈</p>
                <p>Advanced charting and trading features coming soon...</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'learn' && (
          <div>
            <h2 style={{ color: '#111827' }}>Causal Education Hub</h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '20px'
            }}>
              {[
                { title: 'Understanding Risk', emoji: '🎯', progress: 75 },
                { title: 'Gate System Basics', emoji: '🚪', progress: 100 },
                { title: 'Portfolio Theory', emoji: '📊', progress: 45 },
                { title: 'Trading Psychology', emoji: '🧠', progress: 30 }
              ].map(course => (
                <div key={course.title} style={{
                  backgroundColor: 'white',
                  padding: '20px',
                  borderRadius: '8px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ fontSize: '32px', marginBottom: '10px' }}>{course.emoji}</div>
                  <h3 style={{ margin: '0 0 10px 0', color: '#111827' }}>{course.title}</h3>
                  <div style={{
                    backgroundColor: '#e5e7eb',
                    height: '8px',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      backgroundColor: '#3b82f6',
                      height: '100%',
                      width: `${course.progress}%`,
                      transition: 'width 0.3s'
                    }} />
                  </div>
                  <p style={{ margin: '10px 0 0 0', color: '#6b7280', fontSize: '14px' }}>
                    {course.progress}% Complete
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'progress' && (
          <div>
            <h2 style={{ color: '#111827' }}>Your Trading Journey</h2>
            <div style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <div style={{ marginBottom: '30px' }}>
                <h3 style={{ color: '#111827', marginBottom: '15px' }}>Achievements</h3>
                <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
                  {['🏆 First Trade', '🎯 Risk Master', '📈 Profit Maker', '🛡️ Safe Trader'].map(badge => (
                    <div key={badge} style={{
                      padding: '10px 20px',
                      backgroundColor: '#fef3c7',
                      borderRadius: '20px',
                      fontSize: '14px',
                      fontWeight: '500'
                    }}>
                      {badge}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 style={{ color: '#111827', marginBottom: '15px' }}>Recent Milestones</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {[
                    'Completed Gate G0 requirements',
                    'Achieved 5 consecutive winning trades',
                    'Maintained risk below 2% for 30 days',
                    'Completed "Understanding Risk" course'
                  ].map(milestone => (
                    <div key={milestone} style={{
                      padding: '15px',
                      backgroundColor: '#f3f4f6',
                      borderRadius: '6px',
                      fontSize: '14px',
                      color: '#374151'
                    }}>
                      ✅ {milestone}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AppEnhancedSimple;