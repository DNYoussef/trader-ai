import { useState, useEffect, useCallback } from 'react';

// API endpoints - use relative URLs for production compatibility
const AI_API_ENDPOINTS = {
  volatility: '/api/ai/timesfm/volatility',
  risk: '/api/ai/timesfm/risk',
  sentiment: '/api/ai/fingpt/sentiment',
  forecast: '/api/ai/fingpt/forecast',
  features: '/api/ai/features/32d',
};

interface TimesFMVolatility {
  horizons: number[];
  vix_forecast: number[];
  spike_probability: number;
  crisis_probability: number;
  regime: string[];
  confidence: number;
  model: string;
}

interface TimesFMRisk {
  portfolio_risk: {
    current_var: number;
    forecasted_var: number[];
    var_confidence: [number, number];
  };
  extreme_events: {
    black_swan_probability: number;
    tail_risk_metrics: any;
  };
  regime_predictions: string[];
  risk_contributors: any;
  model: string;
}

interface FinGPTSentiment {
  overall_sentiment: string;
  sentiment_score: number;
  confidence: number;
  trending_topics: string[];
  contrarian_signals: any[];
  news_volume: number;
  social_buzz: number;
  model: string;
}

interface FinGPTForecast {
  predictions: {
    [symbol: string]: {
      direction: string;
      probability: number;
      expected_return: number;
      timeframe: string;
    };
  };
  market_outlook: string;
  confidence_level: number;
  model: string;
}

interface Enhanced32DFeatures {
  feature_vector: number[];
  feature_names: string[];
  dpi_score: number;
  signal_strength: number;
  ai_confidence: number;
  feature_importance: any;
  model: string;
}

interface AIDataState {
  timesfmVolatility: TimesFMVolatility | null;
  timesfmRisk: TimesFMRisk | null;
  fingptSentiment: FinGPTSentiment | null;
  fingptForecast: FinGPTForecast | null;
  enhanced32D: Enhanced32DFeatures | null;
  loading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export const useAIData = (refreshInterval: number = 5000) => {
  const [state, setState] = useState<AIDataState>({
    timesfmVolatility: null,
    timesfmRisk: null,
    fingptSentiment: null,
    fingptForecast: null,
    enhanced32D: null,
    loading: false,
    error: null,
    lastUpdate: null
  });

  const fetchAIData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // Fetch all AI data in parallel for performance (using relative URLs for production)
      const [volatility, risk, sentiment, forecast, features] = await Promise.all([
        fetch(AI_API_ENDPOINTS.volatility).then(r => r.json()),
        fetch(AI_API_ENDPOINTS.risk).then(r => r.json()),
        fetch(AI_API_ENDPOINTS.sentiment).then(r => r.json()),
        fetch(AI_API_ENDPOINTS.forecast).then(r => r.json()),
        fetch(AI_API_ENDPOINTS.features).then(r => r.json())
      ]);

      setState({
        timesfmVolatility: volatility.error ? null : volatility,
        timesfmRisk: risk.error ? null : risk,
        fingptSentiment: sentiment.error ? null : sentiment,
        fingptForecast: forecast.error ? null : forecast,
        enhanced32D: features.error ? null : features,
        loading: false,
        error: null,
        lastUpdate: new Date()
      });
    } catch (error) {
      console.error('Error fetching AI data:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: 'Failed to fetch AI data. Using fallback values.',
        lastUpdate: new Date()
      }));

      // Provide fallback data if API is unavailable
      provideFallbackData();
    }
  }, []);

  const provideFallbackData = () => {
    setState(prev => ({
      ...prev,
      timesfmVolatility: {
        horizons: [1, 24, 72, 168],
        vix_forecast: [18.5, 19.2, 20.1, 19.8],
        spike_probability: 0.12,
        crisis_probability: 0.03,
        regime: ['normal', 'normal', 'volatile', 'normal'],
        confidence: 0.85,
        model: 'TimesFM 200M (fallback)'
      },
      timesfmRisk: {
        portfolio_risk: {
          current_var: 1250.50,
          forecasted_var: [1300, 1350, 1400, 1380],
          var_confidence: [1200, 1500]
        },
        extreme_events: {
          black_swan_probability: 0.02,
          tail_risk_metrics: { cvar: 2500, max_loss: 5000 }
        },
        regime_predictions: ['stable', 'stable', 'transition', 'stable'],
        risk_contributors: { market: 0.6, specific: 0.4 },
        model: 'TimesFM Risk (fallback)'
      },
      fingptSentiment: {
        overall_sentiment: 'neutral',
        sentiment_score: 0.52,
        confidence: 0.78,
        trending_topics: ['Fed', 'Earnings', 'AI', 'China', 'Oil'],
        contrarian_signals: [],
        news_volume: 1250,
        social_buzz: 0.65,
        model: 'FinGPT Sentiment (fallback)'
      },
      fingptForecast: {
        predictions: {
          SPY: { direction: 'up', probability: 0.58, expected_return: 0.02, timeframe: '1d' },
          QQQ: { direction: 'up', probability: 0.61, expected_return: 0.03, timeframe: '1d' },
          IWM: { direction: 'down', probability: 0.52, expected_return: -0.01, timeframe: '1d' },
          VIX: { direction: 'down', probability: 0.55, expected_return: -0.05, timeframe: '1d' },
          GLD: { direction: 'up', probability: 0.54, expected_return: 0.01, timeframe: '1d' }
        },
        market_outlook: 'cautiously optimistic',
        confidence_level: 0.57,
        model: 'FinGPT Forecast (fallback)'
      },
      enhanced32D: {
        feature_vector: Array(32).fill(0).map(() => Math.random() * 2 - 1),
        feature_names: Array(32).fill(0).map((_, i) => `feature_${i + 1}`),
        dpi_score: 75,
        signal_strength: 0.68,
        ai_confidence: 0.82,
        feature_importance: {},
        model: 'Enhanced HRM 32D (fallback)'
      }
    }));
  };

  // Calculate aggregate AI signals
  const getAggregateSignals = useCallback(() => {
    const { timesfmVolatility, fingptSentiment, fingptForecast, enhanced32D } = state;

    if (!timesfmVolatility || !fingptSentiment || !fingptForecast || !enhanced32D) {
      return { signal: 'HOLD', confidence: 0, dpi_score: 0 };
    }

    // Aggregate signals from all AI components
    let bullishScore = 0;
    let totalWeight = 0;

    // TimesFM signals (weight: 0.25)
    if (timesfmVolatility.spike_probability < 0.2) {
      bullishScore += 0.25;
    }
    totalWeight += 0.25;

    // FinGPT sentiment (weight: 0.25)
    if (fingptSentiment.sentiment_score > 0.6) {
      bullishScore += 0.25;
    } else if (fingptSentiment.sentiment_score < 0.4) {
      bullishScore -= 0.25;
    }
    totalWeight += 0.25;

    // FinGPT forecast (weight: 0.25)
    const spyPrediction = fingptForecast.predictions.SPY;
    if (spyPrediction && spyPrediction.direction === 'up') {
      bullishScore += 0.25 * spyPrediction.probability;
    }
    totalWeight += 0.25;

    // Enhanced features DPI score (weight: 0.25)
    if (enhanced32D.dpi_score > 70) {
      bullishScore += 0.25;
    } else if (enhanced32D.dpi_score < 30) {
      bullishScore -= 0.25;
    }
    totalWeight += 0.25;

    const normalizedScore = bullishScore / totalWeight;
    const signal = normalizedScore > 0.2 ? 'BUY' : normalizedScore < -0.2 ? 'SELL' : 'HOLD';
    const confidence = Math.abs(normalizedScore) * 100;

    return {
      signal,
      confidence: Math.min(confidence, 95),
      dpi_score: enhanced32D.dpi_score
    };
  }, [state]);

  // Initial fetch
  useEffect(() => {
    fetchAIData();
  }, [fetchAIData]);

  // Periodic refresh
  useEffect(() => {
    const interval = setInterval(fetchAIData, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchAIData, refreshInterval]);

  return {
    ...state,
    getAggregateSignals,
    refresh: fetchAIData
  };
};