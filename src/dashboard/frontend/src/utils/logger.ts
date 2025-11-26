/**
 * ISS-043: Centralized logger utility for dashboard frontend.
 * Uses environment variable to control debug output.
 */

const isDevelopment = process.env.NODE_ENV === 'development';
const isDebugEnabled = process.env.REACT_APP_DEBUG === 'true' || isDevelopment;

export const logger = {
  debug: (...args: unknown[]) => {
    if (isDebugEnabled) {
      console.log('[DEBUG]', ...args);
    }
  },

  info: (...args: unknown[]) => {
    console.info('[INFO]', ...args);
  },

  warn: (...args: unknown[]) => {
    console.warn('[WARN]', ...args);
  },

  error: (...args: unknown[]) => {
    console.error('[ERROR]', ...args);
  },

  // WebSocket specific logging
  ws: {
    connected: () => {
      if (isDebugEnabled) {
        console.log('[WS] Connected');
      }
    },
    disconnected: () => {
      if (isDebugEnabled) {
        console.log('[WS] Disconnected');
      }
    },
    message: (type: string, data?: unknown) => {
      if (isDebugEnabled) {
        console.log('[WS] Message:', type, data);
      }
    }
  }
};

export default logger;
