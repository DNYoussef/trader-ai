import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { RiskMetrics, PositionUpdate, AlertEvent, HealthCheck, ApiResponse } from '@/types';

// Base API configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class DashboardAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for authentication/logging
    this.client.interceptors.request.use(
      (config) => {
        // Add timestamp to all requests
        config.params = {
          ...config.params,
          _t: Date.now(),
        };

        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error);

        // Handle different error types
        if (error.response) {
          // Server responded with error status
          const { status, data } = error.response;
          console.error(`API Error ${status}:`, data);

          switch (status) {
            case 401:
              // Handle authentication error
              console.error('Authentication required');
              break;
            case 403:
              // Handle authorization error
              console.error('Access forbidden');
              break;
            case 404:
              // Handle not found
              console.error('Resource not found');
              break;
            case 500:
              // Handle server error
              console.error('Internal server error');
              break;
            default:
              console.error('Unknown server error');
          }
        } else if (error.request) {
          // Network error
          console.error('Network error - no response received');
        } else {
          // Request setup error
          console.error('Request setup error:', error.message);
        }

        return Promise.reject(error);
      }
    );
  }

  // Health check endpoint
  async healthCheck(): Promise<HealthCheck> {
    try {
      const response = await this.client.get<HealthCheck>('/api/health');
      return response.data;
    } catch (error) {
      throw new Error('Health check failed');
    }
  }

  // Get current risk metrics
  async getCurrentMetrics(): Promise<RiskMetrics | null> {
    try {
      const response = await this.client.get<ApiResponse<RiskMetrics>>('/api/metrics/current');

      if (response.data.error) {
        console.warn('No metrics available:', response.data.error);
        return null;
      }

      return response.data.data || null;
    } catch (error) {
      console.error('Failed to fetch current metrics:', error);
      throw error;
    }
  }

  // Get all positions
  async getPositions(): Promise<PositionUpdate[]> {
    try {
      const response = await this.client.get<PositionUpdate[]>('/api/positions');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch positions:', error);
      throw error;
    }
  }

  // Get alerts
  async getAlerts(): Promise<AlertEvent[]> {
    try {
      const response = await this.client.get<AlertEvent[]>('/api/alerts');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      throw error;
    }
  }

  // Acknowledge alert
  async acknowledgeAlert(alertId: string): Promise<{ status: string }> {
    try {
      const response = await this.client.post<{ status: string }>(
        `/api/alerts/${alertId}/acknowledge`
      );
      return response.data;
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
      throw error;
    }
  }

  // Historical data endpoints (if implemented on backend)
  async getHistoricalMetrics(
    startTime: number,
    endTime: number,
    resolution: string = '1m'
  ): Promise<RiskMetrics[]> {
    try {
      const response = await this.client.get<RiskMetrics[]>('/api/metrics/historical', {
        params: {
          start: startTime,
          end: endTime,
          resolution,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to fetch historical metrics:', error);
      throw error;
    }
  }

  // Get position history
  async getPositionHistory(
    symbol: string,
    startTime: number,
    endTime: number
  ): Promise<PositionUpdate[]> {
    try {
      const response = await this.client.get<PositionUpdate[]>(
        `/api/positions/${symbol}/history`,
        {
          params: {
            start: startTime,
            end: endTime,
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Failed to fetch position history:', error);
      throw error;
    }
  }

  // Risk analysis endpoints
  async getPortfolioAnalysis(): Promise<any> {
    try {
      const response = await this.client.get('/api/analysis/portfolio');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch portfolio analysis:', error);
      throw error;
    }
  }

  async getPerformanceReport(period: string = '1d'): Promise<any> {
    try {
      const response = await this.client.get('/api/reports/performance', {
        params: { period },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to fetch performance report:', error);
      throw error;
    }
  }

  // Settings endpoints
  async getUserSettings(): Promise<any> {
    try {
      const response = await this.client.get('/api/settings');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch user settings:', error);
      throw error;
    }
  }

  async updateUserSettings(settings: any): Promise<any> {
    try {
      const response = await this.client.put('/api/settings', settings);
      return response.data;
    } catch (error) {
      console.error('Failed to update user settings:', error);
      throw error;
    }
  }

  // Export/backup endpoints
  async exportData(startTime: number, endTime: number): Promise<Blob> {
    try {
      const response = await this.client.get('/api/export', {
        params: {
          start: startTime,
          end: endTime,
        },
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      console.error('Failed to export data:', error);
      throw error;
    }
  }

  // System status and debugging
  async getSystemStatus(): Promise<any> {
    try {
      const response = await this.client.get('/api/system/status');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      throw error;
    }
  }

  async getConnectionMetrics(): Promise<any> {
    try {
      const response = await this.client.get('/api/system/connections');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch connection metrics:', error);
      throw error;
    }
  }

  // Utility methods
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      return false;
    }
  }

  // Update base URL (useful for environment switching)
  updateBaseURL(newBaseURL: string): void {
    this.client.defaults.baseURL = newBaseURL;
  }

  // Add custom headers (useful for authentication)
  setAuthToken(token: string): void {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  removeAuthToken(): void {
    delete this.client.defaults.headers.common['Authorization'];
  }

  // Request cancellation support
  createCancelToken() {
    return axios.CancelToken.source();
  }

  // Batch requests with concurrency control
  async batchRequests<T>(
    requests: (() => Promise<T>)[],
    concurrency: number = 5
  ): Promise<T[]> {
    const results: T[] = [];
    const executing: Promise<any>[] = [];

    for (const request of requests) {
      const promise = request().then((result) => {
        executing.splice(executing.indexOf(promise), 1);
        return result;
      });

      results.push(promise as any);
      executing.push(promise);

      if (executing.length >= concurrency) {
        await Promise.race(executing);
      }
    }

    return Promise.all(results);
  }
}

// Create singleton instance
const dashboardAPI = new DashboardAPI();

// Export both the class and instance
export { DashboardAPI };
export default dashboardAPI;

// Utility functions for common API patterns
export const withRetry = async <T>(
  apiCall: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: Error;

  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await apiCall();
    } catch (error) {
      lastError = error as Error;

      if (i === maxRetries) {
        throw lastError;
      }

      // Exponential backoff
      const waitTime = delay * Math.pow(2, i);
      console.log(`API call failed, retrying in ${waitTime}ms (attempt ${i + 1}/${maxRetries + 1})`);

      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }

  throw lastError!;
};

export const withTimeout = <T>(
  apiCall: () => Promise<T>,
  timeoutMs: number = 5000
): Promise<T> => {
  return Promise.race([
    apiCall(),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('API call timeout')), timeoutMs)
    ),
  ]);
};