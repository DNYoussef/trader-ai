import axios from 'axios';

const API_BASE = 'http://localhost:8000';

let jwtToken: string | null = null;

/**
 * Set the JWT authentication token
 * Stores in memory and localStorage for persistence
 */
export const setAuthToken = (token: string): void => {
  jwtToken = token;
  // Store in localStorage (httpOnly cookie would be better in production)
  localStorage.setItem('jwt_token', token);
};

/**
 * Get the current JWT authentication token
 * Retrieves from memory or localStorage
 */
export const getAuthToken = (): string | null => {
  if (!jwtToken) {
    jwtToken = localStorage.getItem('jwt_token');
  }
  return jwtToken;
};

/**
 * Clear the JWT authentication token
 * Removes from memory and localStorage
 */
export const clearAuthToken = (): void => {
  jwtToken = null;
  localStorage.removeItem('jwt_token');
};

/**
 * Exchange Plaid public token for JWT access token
 * @param publicToken - Plaid public token from Link flow
 * @returns JWT token for authenticated API requests
 */
export const exchangePublicToken = async (publicToken: string): Promise<string> => {
  const response = await axios.post(`${API_BASE}/api/plaid/exchange_public_token`, {
    public_token: publicToken,
  });

  const { jwt_token } = response.data;

  if (!jwt_token) {
    throw new Error('No JWT token received from server');
  }

  setAuthToken(jwt_token);
  return jwt_token;
};

/**
 * Check if user is authenticated
 */
export const isAuthenticated = (): boolean => {
  return getAuthToken() !== null;
};
