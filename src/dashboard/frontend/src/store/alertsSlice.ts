import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { AlertEvent } from '@/types';

interface AlertsState {
  alerts: AlertEvent[];
  unread_count: number;
  sound_enabled: boolean;
  auto_acknowledge: boolean;
  max_alerts: number;
}

const initialState: AlertsState = {
  alerts: [],
  unread_count: 0,
  sound_enabled: true,
  auto_acknowledge: false,
  max_alerts: 100,
};

export const alertsSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    addAlert: (state, action: PayloadAction<AlertEvent>) => {
      const alert = action.payload;

      // Prevent duplicate alerts within 1 minute
      const isDuplicate = state.alerts.some(
        existing =>
          existing.metric_name === alert.metric_name &&
          existing.severity === alert.severity &&
          (alert.timestamp - existing.timestamp) < 60000 // 1 minute
      );

      if (!isDuplicate) {
        state.alerts.unshift(alert); // Add to beginning

        if (!alert.acknowledged) {
          state.unread_count += 1;
        }

        // Maintain max alerts limit
        if (state.alerts.length > state.max_alerts) {
          const removed = state.alerts.splice(state.max_alerts);
          // Adjust unread count for removed unacknowledged alerts
          const removedUnread = removed.filter(a => !a.acknowledged).length;
          state.unread_count = Math.max(0, state.unread_count - removedUnread);
        }

        // Auto-acknowledge if enabled and severity is low
        if (state.auto_acknowledge && alert.severity === 'low') {
          const alertIndex = state.alerts.findIndex(a => a.alert_id === alert.alert_id);
          if (alertIndex !== -1) {
            state.alerts[alertIndex].acknowledged = true;
          }
        }
      }
    },

    acknowledgeAlert: (state, action: PayloadAction<string>) => {
      const alertId = action.payload;
      const alertIndex = state.alerts.findIndex(alert => alert.alert_id === alertId);

      if (alertIndex !== -1 && !state.alerts[alertIndex].acknowledged) {
        state.alerts[alertIndex].acknowledged = true;
        state.unread_count = Math.max(0, state.unread_count - 1);
      }
    },

    acknowledgeAllAlerts: (state) => {
      state.alerts.forEach(alert => {
        alert.acknowledged = true;
      });
      state.unread_count = 0;
    },

    clearAlert: (state, action: PayloadAction<string>) => {
      const alertId = action.payload;
      const alertIndex = state.alerts.findIndex(alert => alert.alert_id === alertId);

      if (alertIndex !== -1) {
        const alert = state.alerts[alertIndex];
        if (!alert.acknowledged) {
          state.unread_count = Math.max(0, state.unread_count - 1);
        }
        state.alerts.splice(alertIndex, 1);
      }
    },

    clearAllAlerts: (state) => {
      state.alerts = [];
      state.unread_count = 0;
    },

    clearOldAlerts: (state, action: PayloadAction<number>) => {
      const cutoffTime = action.payload;
      const beforeCount = state.alerts.length;
      const removedUnread = state.alerts.filter(
        alert => alert.timestamp < cutoffTime && !alert.acknowledged
      ).length;

      state.alerts = state.alerts.filter(alert => alert.timestamp >= cutoffTime);
      state.unread_count = Math.max(0, state.unread_count - removedUnread);
    },

    setSoundEnabled: (state, action: PayloadAction<boolean>) => {
      state.sound_enabled = action.payload;
    },

    setAutoAcknowledge: (state, action: PayloadAction<boolean>) => {
      state.auto_acknowledge = action.payload;
    },

    setMaxAlerts: (state, action: PayloadAction<number>) => {
      state.max_alerts = Math.max(10, Math.min(1000, action.payload));

      // Trim alerts if new limit is smaller
      if (state.alerts.length > state.max_alerts) {
        const removed = state.alerts.splice(state.max_alerts);
        const removedUnread = removed.filter(a => !a.acknowledged).length;
        state.unread_count = Math.max(0, state.unread_count - removedUnread);
      }
    },

    // Bulk operations for performance
    batchAddAlerts: (state, action: PayloadAction<AlertEvent[]>) => {
      const newAlerts = action.payload;

      newAlerts.forEach(alert => {
        // Check for duplicates
        const isDuplicate = state.alerts.some(
          existing =>
            existing.metric_name === alert.metric_name &&
            existing.severity === alert.severity &&
            (alert.timestamp - existing.timestamp) < 60000
        );

        if (!isDuplicate) {
          state.alerts.unshift(alert);
          if (!alert.acknowledged) {
            state.unread_count += 1;
          }
        }
      });

      // Maintain max alerts limit
      if (state.alerts.length > state.max_alerts) {
        const removed = state.alerts.splice(state.max_alerts);
        const removedUnread = removed.filter(a => !a.acknowledged).length;
        state.unread_count = Math.max(0, state.unread_count - removedUnread);
      }
    },

    // Reset for reconnection scenarios
    resetAlerts: (state) => {
      state.alerts = [];
      state.unread_count = 0;
    },
  },
});

export const {
  addAlert,
  acknowledgeAlert,
  acknowledgeAllAlerts,
  clearAlert,
  clearAllAlerts,
  clearOldAlerts,
  setSoundEnabled,
  setAutoAcknowledge,
  setMaxAlerts,
  batchAddAlerts,
  resetAlerts,
} = alertsSlice.actions;

// Selectors
export const selectAllAlerts = (state: { alerts: AlertsState }) =>
  state.alerts.alerts;

export const selectUnreadCount = (state: { alerts: AlertsState }) =>
  state.alerts.unread_count;

export const selectSoundEnabled = (state: { alerts: AlertsState }) =>
  state.alerts.sound_enabled;

export const selectAutoAcknowledge = (state: { alerts: AlertsState }) =>
  state.alerts.auto_acknowledge;

export const selectAlertsBySeverity = (state: { alerts: AlertsState }) => {
  const alerts = state.alerts.alerts;
  return {
    critical: alerts.filter(a => a.severity === 'critical'),
    high: alerts.filter(a => a.severity === 'high'),
    medium: alerts.filter(a => a.severity === 'medium'),
    low: alerts.filter(a => a.severity === 'low'),
  };
};

export const selectRecentAlerts = (minutes: number = 60) =>
  (state: { alerts: AlertsState }) => {
    const cutoffTime = Date.now() - (minutes * 60 * 1000);
    return state.alerts.alerts.filter(alert => alert.timestamp > cutoffTime);
  };

export const selectUnreadAlerts = (state: { alerts: AlertsState }) =>
  state.alerts.alerts.filter(alert => !alert.acknowledged);

export const selectCriticalAlerts = (state: { alerts: AlertsState }) =>
  state.alerts.alerts.filter(alert =>
    alert.severity === 'critical' && !alert.acknowledged
  );

export const selectAlertsByMetric = (metricName: string) =>
  (state: { alerts: AlertsState }) =>
    state.alerts.alerts.filter(alert => alert.metric_name === metricName);

export const selectLatestAlert = (state: { alerts: AlertsState }) =>
  state.alerts.alerts.length > 0 ? state.alerts.alerts[0] : null;

export const selectAlertSummary = (state: { alerts: AlertsState }) => {
  const alerts = state.alerts.alerts;
  const now = Date.now();
  const last24Hours = now - (24 * 60 * 60 * 1000);

  return {
    total: alerts.length,
    unread: state.alerts.unread_count,
    last24Hours: alerts.filter(a => a.timestamp > last24Hours).length,
    critical: alerts.filter(a => a.severity === 'critical').length,
    high: alerts.filter(a => a.severity === 'high').length,
    acknowledged: alerts.filter(a => a.acknowledged).length,
  };
};

export default alertsSlice.reducer;