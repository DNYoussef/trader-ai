import { useEffect, useRef, useCallback, useState } from 'react';
import { useAppDispatch, useAppSelector } from '@/store';
import {
  setConnected,
  incrementReconnectAttempts,
  updateHeartbeat,
  updateRiskMetrics,
  updatePosition,
  setError,
} from '@/store/dashboardSlice';
import { addAlert, acknowledgeAlert } from '@/store/alertsSlice';
import { selectRefreshRate, selectAlertSoundEnabled } from '@/store/settingsSlice';
import { WebSocketMessage, UseWebSocketOptions, UseWebSocketReturn } from '@/types';

const DEFAULT_OPTIONS: UseWebSocketOptions = {
  auto_connect: true,
  reconnect_attempts: 5,
  heartbeat_interval: 30000, // 30 seconds
};

export const useWebSocket = (
  url: string,
  options: Partial<UseWebSocketOptions> = {}
): UseWebSocketReturn => {
  const dispatch = useAppDispatch();
  const refreshRate = useAppSelector(selectRefreshRate);
  const alertSoundEnabled = useAppSelector(selectAlertSoundEnabled);

  const config = { ...DEFAULT_OPTIONS, ...options };
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    last_heartbeat: 0,
    connection_time: 0,
    reconnect_attempts: 0,
  });

  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const subscriptionsRef = useRef<Set<string>>(new Set());

  // Sound effects for alerts
  const playAlertSound = useCallback((severity: string) => {
    if (!alertSoundEnabled) return;

    try {
      const audio = new Audio();
      // Different sounds for different severities
      switch (severity) {
        case 'critical':
          audio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmYeAy2P0+zPfSwBJG/A7t+SQwsWYrrr7K1XFgU/ld3wuGYgBi6S2OzNeCAAI2+/8d+PQQkUW7Ps7bJYFQU8ld/vt2ciBC2P0+7PeCEAImu59+CORwgUV7Pp7K1ZFgQ7k9/xumciBC2O0e7QeSEAIWm398OQQg8UW7bp6q5ZGAo';
          break;
        case 'high':
          audio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmYeAy2P0+zPfSwBJG/A7t+SQwsWYrrr7K1XFgU/ld3wuGYgBi6S2OzNeCAAI2+/8d+PQQkUW7Ps7bJYFQU8ld/vt2ciBC2P0+7PeCEAImu59+CORwgUV7Pp7K1ZFgQ7k9/xumciBC2O0e7QeSEAIWm398OQQg8UW7bp6q5ZGAo';
          break;
        default:
          audio.src = 'data:audio/wav;base64,UklGRkoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAA=';
      }
      audio.play().catch(() => {
        // Silently fail if audio playback is not allowed
      });
    } catch (error) {
      console.warn('Failed to play alert sound:', error);
    }
  }, [alertSoundEnabled]);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      switch (message.type) {
        case 'risk_metrics':
          dispatch(updateRiskMetrics(message.data));
          break;

        case 'position_update':
          dispatch(updatePosition(message.data));
          break;

        case 'alert':
          dispatch(addAlert(message.data));
          playAlertSound(message.data.severity);
          break;

        case 'alert_acknowledged':
          dispatch(acknowledgeAlert(message.data.alert_id));
          break;

        case 'heartbeat':
          dispatch(updateHeartbeat());
          setConnectionStatus(prev => ({
            ...prev,
            last_heartbeat: Date.now(),
          }));
          break;

        case 'pong':
          // Handle ping/pong for connection health
          break;

        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [dispatch, playAlertSound]);

  // Send ping to server
  const sendPing = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
    }
  }, [socket]);

  // Start heartbeat interval
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    heartbeatIntervalRef.current = setInterval(sendPing, config.heartbeat_interval);
  }, [sendPing, config.heartbeat_interval]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    try {
      const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const wsUrl = `${url}/${clientId}`;

      const newSocket = new WebSocket(wsUrl);

      newSocket.onopen = () => {
        console.log('WebSocket connected');
        setSocket(newSocket);
        dispatch(setConnected(true));
        dispatch(setError(null));

        setConnectionStatus(prev => ({
          ...prev,
          connected: true,
          connection_time: Date.now(),
          reconnect_attempts: 0,
        }));

        reconnectAttemptsRef.current = 0;
        startHeartbeat();

        // Re-subscribe to previous subscriptions
        subscriptionsRef.current.forEach(subscription => {
          newSocket.send(JSON.stringify({
            type: 'subscribe',
            subscription,
          }));
        });
      };

      newSocket.onmessage = handleMessage;

      newSocket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setSocket(null);
        dispatch(setConnected(false));

        setConnectionStatus(prev => ({
          ...prev,
          connected: false,
        }));

        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        // Attempt reconnection if not a manual close
        if (event.code !== 1000 && reconnectAttemptsRef.current < config.reconnect_attempts) {
          reconnectAttemptsRef.current += 1;
          dispatch(incrementReconnectAttempts());

          setConnectionStatus(prev => ({
            ...prev,
            reconnect_attempts: reconnectAttemptsRef.current,
          }));

          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

          reconnectTimeoutRef.current = setTimeout(connect, delay);
        } else {
          dispatch(setError('Connection failed. Please refresh the page.'));
        }
      };

      newSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        dispatch(setError('WebSocket connection error'));
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      dispatch(setError('Failed to create connection'));
    }
  }, [url, dispatch, handleMessage, startHeartbeat, config.reconnect_attempts]);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    if (socket) {
      socket.close(1000, 'Manual disconnect');
      setSocket(null);
    }

    dispatch(setConnected(false));
    setConnectionStatus(prev => ({
      ...prev,
      connected: false,
    }));
  }, [socket, dispatch]);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, [socket]);

  // Subscribe to data type
  const subscribe = useCallback((subscription: string) => {
    subscriptionsRef.current.add(subscription);
    return sendMessage({
      type: 'subscribe',
      subscription,
    });
  }, [sendMessage]);

  // Unsubscribe from data type
  const unsubscribe = useCallback((subscription: string) => {
    subscriptionsRef.current.delete(subscription);
    return sendMessage({
      type: 'unsubscribe',
      subscription,
    });
  }, [sendMessage]);

  // Effect for auto-connection
  useEffect(() => {
    if (config.auto_connect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [config.auto_connect]); // Only depend on auto_connect, not connect/disconnect

  // Effect for cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, []);

  return {
    connection_status: connectionStatus,
    send_message: sendMessage,
    subscribe,
    unsubscribe,
  };
};

// Hook for easy subscription management
export const useWebSocketSubscription = (
  subscription: string,
  websocket: UseWebSocketReturn
) => {
  useEffect(() => {
    if (websocket.connection_status.connected) {
      websocket.subscribe(subscription);

      return () => {
        websocket.unsubscribe(subscription);
      };
    }
  }, [subscription, websocket, websocket.connection_status.connected]);
};

// Hook for automatic reconnection with exponential backoff
export const useWebSocketWithRetry = (url: string, maxRetries: number = 5) => {
  const websocket = useWebSocket(url, {
    auto_connect: true,
    reconnect_attempts: maxRetries,
    heartbeat_interval: 30000,
  });

  return websocket;
};