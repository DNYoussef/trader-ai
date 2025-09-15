import React, { useState } from 'react';
import './index.css';
import App from './App';
// import AppEnhanced from './AppEnhanced';
// import AppEnhanced from './AppEnhancedSimple'; // Simple version
import AppEnhanced from './AppEnhancedFull'; // Full integrated version

/**
 * App wrapper that allows toggling between original and enhanced UI
 */

const AppWithEnhanced: React.FC = () => {
  // Check for enhanced mode preference
  const [enhancedMode, setEnhancedMode] = useState(
    localStorage.getItem('ui_mode') === 'enhanced' || true // Default to enhanced
  );

  const toggleMode = () => {
    const newMode = !enhancedMode;
    setEnhancedMode(newMode);
    localStorage.setItem('ui_mode', newMode ? 'enhanced' : 'original');
  };

  // Mode toggle button (floating in corner)
  const ModeToggle = () => (
    <button
      onClick={toggleMode}
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 9999,
        padding: '10px 20px',
        backgroundColor: enhancedMode ? '#8b5cf6' : '#3b82f6',
        color: 'white',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        fontSize: '14px',
        fontWeight: 'bold',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}
    >
      Switch to {enhancedMode ? 'Original' : 'Enhanced'} UI
    </button>
  );

  // Debug: Add console log
  console.log('AppWithEnhanced rendering, enhancedMode:', enhancedMode);

  return (
    <>
      {enhancedMode ? <AppEnhanced /> : <App />}
      <ModeToggle />
    </>
  );
};

export default AppWithEnhanced;