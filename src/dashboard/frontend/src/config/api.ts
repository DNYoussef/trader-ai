/**
 * API Configuration
 * Uses relative URLs in production (served by FastAPI)
 * Uses absolute URLs in development (via Vite proxy)
 */

// In production, the frontend is served by FastAPI, so relative URLs work
// In development, Vite's proxy handles /api/* -> localhost:8000
export const API_BASE_URL = '';  // Empty string = relative URLs

// WebSocket URL
export const getWebSocketUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}`;
};

// API endpoints
export const API_ENDPOINTS = {
  metrics: '/api/metrics/current',
  positions: '/api/positions',
  alerts: '/api/alerts',
  gateStatus: '/api/gates/status',
  health: '/api/health',
} as const;
