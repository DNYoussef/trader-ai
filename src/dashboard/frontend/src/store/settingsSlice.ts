import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DashboardSettings, RiskThresholds, DisplayPreferences } from '@/types';

const defaultRiskThresholds: RiskThresholds = {
  p_ruin: { high: 0.1, critical: 0.2 },
  var_95: { high: 0.05, critical: 0.1 },
  max_drawdown: { high: 0.1, critical: 0.2 },
  margin_used: { high: 0.8, critical: 0.9 },
};

const defaultDisplayPreferences: DisplayPreferences = {
  dark_mode: false,
  compact_view: false,
  show_advanced_metrics: true,
  currency_format: 'USD',
  decimal_places: 2,
};

const initialState: DashboardSettings = {
  auto_acknowledge_alerts: false,
  alert_sound_enabled: true,
  refresh_rate: 1000, // 1 second
  chart_timeframe: '1h',
  risk_thresholds: defaultRiskThresholds,
  display_preferences: defaultDisplayPreferences,
};

export const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    updateRiskThresholds: (state, action: PayloadAction<Partial<RiskThresholds>>) => {
      state.risk_thresholds = { ...state.risk_thresholds, ...action.payload };
    },

    updateDisplayPreferences: (state, action: PayloadAction<Partial<DisplayPreferences>>) => {
      state.display_preferences = { ...state.display_preferences, ...action.payload };
    },

    setAutoAcknowledgeAlerts: (state, action: PayloadAction<boolean>) => {
      state.auto_acknowledge_alerts = action.payload;
    },

    setAlertSoundEnabled: (state, action: PayloadAction<boolean>) => {
      state.alert_sound_enabled = action.payload;
    },

    setRefreshRate: (state, action: PayloadAction<number>) => {
      // Clamp between 100ms and 10 seconds
      state.refresh_rate = Math.max(100, Math.min(10000, action.payload));
    },

    setChartTimeframe: (state, action: PayloadAction<'1h' | '4h' | '1d' | '1w'>) => {
      state.chart_timeframe = action.payload;
    },

    toggleDarkMode: (state) => {
      state.display_preferences.dark_mode = !state.display_preferences.dark_mode;
    },

    toggleCompactView: (state) => {
      state.display_preferences.compact_view = !state.display_preferences.compact_view;
    },

    toggleAdvancedMetrics: (state) => {
      state.display_preferences.show_advanced_metrics = !state.display_preferences.show_advanced_metrics;
    },

    setCurrencyFormat: (state, action: PayloadAction<'USD' | 'percentage'>) => {
      state.display_preferences.currency_format = action.payload;
    },

    setDecimalPlaces: (state, action: PayloadAction<number>) => {
      // Clamp between 0 and 6 decimal places
      state.display_preferences.decimal_places = Math.max(0, Math.min(6, action.payload));
    },

    // Bulk settings update
    updateSettings: (state, action: PayloadAction<Partial<DashboardSettings>>) => {
      return { ...state, ...action.payload };
    },

    // Reset to defaults
    resetToDefaults: () => {
      return initialState;
    },

    // Import/Export settings
    importSettings: (state, action: PayloadAction<DashboardSettings>) => {
      const importedSettings = action.payload;

      // Validate imported settings before applying
      if (importedSettings.refresh_rate) {
        importedSettings.refresh_rate = Math.max(100, Math.min(10000, importedSettings.refresh_rate));
      }

      if (importedSettings.display_preferences?.decimal_places !== undefined) {
        importedSettings.display_preferences.decimal_places = Math.max(0, Math.min(6, importedSettings.display_preferences.decimal_places));
      }

      return { ...state, ...importedSettings };
    },

    // Specific threshold updates
    updatePRuinThresholds: (state, action: PayloadAction<{ high: number; critical: number }>) => {
      state.risk_thresholds.p_ruin = action.payload;
    },

    updateVarThresholds: (state, action: PayloadAction<{ high: number; critical: number }>) => {
      state.risk_thresholds.var_95 = action.payload;
    },

    updateDrawdownThresholds: (state, action: PayloadAction<{ high: number; critical: number }>) => {
      state.risk_thresholds.max_drawdown = action.payload;
    },

    updateMarginThresholds: (state, action: PayloadAction<{ high: number; critical: number }>) => {
      state.risk_thresholds.margin_used = action.payload;
    },
  },
});

export const {
  updateRiskThresholds,
  updateDisplayPreferences,
  setAutoAcknowledgeAlerts,
  setAlertSoundEnabled,
  setRefreshRate,
  setChartTimeframe,
  toggleDarkMode,
  toggleCompactView,
  toggleAdvancedMetrics,
  setCurrencyFormat,
  setDecimalPlaces,
  updateSettings,
  resetToDefaults,
  importSettings,
  updatePRuinThresholds,
  updateVarThresholds,
  updateDrawdownThresholds,
  updateMarginThresholds,
} = settingsSlice.actions;

// Selectors
export const selectRiskThresholds = (state: { settings: DashboardSettings }) =>
  state.settings.risk_thresholds;

export const selectDisplayPreferences = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences;

export const selectAutoAcknowledgeAlerts = (state: { settings: DashboardSettings }) =>
  state.settings.auto_acknowledge_alerts;

export const selectAlertSoundEnabled = (state: { settings: DashboardSettings }) =>
  state.settings.alert_sound_enabled;

export const selectRefreshRate = (state: { settings: DashboardSettings }) =>
  state.settings.refresh_rate;

export const selectChartTimeframe = (state: { settings: DashboardSettings }) =>
  state.settings.chart_timeframe;

export const selectDarkMode = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.dark_mode;

export const selectCompactView = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.compact_view;

export const selectShowAdvancedMetrics = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.show_advanced_metrics;

export const selectCurrencyFormat = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.currency_format;

export const selectDecimalPlaces = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.decimal_places;

export const selectAllSettings = (state: { settings: DashboardSettings }) =>
  state.settings;

// Computed selectors
export const selectFormattedRefreshRate = (state: { settings: DashboardSettings }) => {
  const rate = state.settings.refresh_rate;
  if (rate < 1000) return `${rate}ms`;
  return `${(rate / 1000).toFixed(1)}s`;
};

export const selectThemeClass = (state: { settings: DashboardSettings }) =>
  state.settings.display_preferences.dark_mode ? 'dark' : 'light';

export default settingsSlice.reducer;