import React from 'react'
import ReactDOM from 'react-dom/client'
import AppUnified from './AppUnified'
import './index.css'

// Unified App with MECE organization and feature modes
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppUnified />
  </React.StrictMode>,
)