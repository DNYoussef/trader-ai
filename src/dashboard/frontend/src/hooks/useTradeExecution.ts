import { useState, useCallback } from 'react';

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';

interface TradeRequest {
  symbol: string;
  quantity: number;
  side: OrderSide;
  order_type?: OrderType;
  limit_price?: number;
  stop_price?: number;
}

interface TradeResponse {
  success: boolean;
  order_id?: string;
  symbol?: string;
  quantity?: number;
  executed_price?: number;
  status?: string;
  error?: string;
  timestamp?: Date;
}

interface ExecutionState {
  isExecuting: boolean;
  lastTrade: TradeResponse | null;
  history: TradeResponse[];
  error: string | null;
}

export const useTradeExecution = () => {
  const [state, setState] = useState<ExecutionState>({
    isExecuting: false,
    lastTrade: null,
    history: [],
    error: null
  });

  const executeTrade = useCallback(async (tradeRequest: TradeRequest): Promise<TradeResponse> => {
    setState(prev => ({ ...prev, isExecuting: true, error: null }));

    try {
      // Validate trade request
      if (!tradeRequest.symbol || !tradeRequest.quantity || !tradeRequest.side) {
        throw new Error('Invalid trade request: missing required fields');
      }

      if (tradeRequest.quantity <= 0) {
        throw new Error('Invalid quantity: must be greater than 0');
      }

      // Make API call to execute trade
      const response = await fetch('http://localhost:8000/api/trading/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: tradeRequest.symbol.toUpperCase(),
          quantity: tradeRequest.quantity,
          side: tradeRequest.side,
          order_type: tradeRequest.order_type || 'market',
          limit_price: tradeRequest.limit_price,
          stop_price: tradeRequest.stop_price
        })
      });

      const result: TradeResponse = await response.json();
      result.timestamp = new Date();

      // Update state with trade result
      setState(prev => ({
        ...prev,
        isExecuting: false,
        lastTrade: result,
        history: [result, ...prev.history].slice(0, 50), // Keep last 50 trades
        error: result.success ? null : result.error || 'Trade execution failed'
      }));

      // Show notification based on result
      if (result.success) {
        console.log(`✅ Trade executed: ${result.side} ${result.quantity} ${result.symbol} @ $${result.executed_price}`);
      } else {
        console.error(`❌ Trade failed: ${result.error}`);
      }

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      const failedResult: TradeResponse = {
        success: false,
        error: errorMessage,
        timestamp: new Date()
      };

      setState(prev => ({
        ...prev,
        isExecuting: false,
        lastTrade: failedResult,
        history: [failedResult, ...prev.history].slice(0, 50),
        error: errorMessage
      }));

      return failedResult;
    }
  }, []);

  const executeQuickTrade = useCallback(async (
    symbol: string,
    quantity: number,
    side: OrderSide
  ): Promise<TradeResponse> => {
    return executeTrade({
      symbol,
      quantity,
      side,
      order_type: 'market'
    });
  }, [executeTrade]);

  const executeLimitOrder = useCallback(async (
    symbol: string,
    quantity: number,
    side: OrderSide,
    limit_price: number
  ): Promise<TradeResponse> => {
    return executeTrade({
      symbol,
      quantity,
      side,
      order_type: 'limit',
      limit_price
    });
  }, [executeTrade]);

  const executeStopOrder = useCallback(async (
    symbol: string,
    quantity: number,
    side: OrderSide,
    stop_price: number
  ): Promise<TradeResponse> => {
    return executeTrade({
      symbol,
      quantity,
      side,
      order_type: 'stop',
      stop_price
    });
  }, [executeTrade]);

  const clearHistory = useCallback(() => {
    setState(prev => ({
      ...prev,
      history: [],
      lastTrade: null
    }));
  }, []);

  const getTradeStats = useCallback(() => {
    const { history } = state;
    const successfulTrades = history.filter(t => t.success);
    const failedTrades = history.filter(t => !t.success);

    return {
      totalTrades: history.length,
      successfulTrades: successfulTrades.length,
      failedTrades: failedTrades.length,
      successRate: history.length > 0 ? successfulTrades.length / history.length : 0,
      lastTradeTime: history[0]?.timestamp
    };
  }, [state.history]);

  // Mock execution for testing when backend is not available
  const executeMockTrade = useCallback(async (tradeRequest: TradeRequest): Promise<TradeResponse> => {
    setState(prev => ({ ...prev, isExecuting: true }));

    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const mockResult: TradeResponse = {
      success: Math.random() > 0.1, // 90% success rate
      order_id: `MOCK-${Date.now()}`,
      symbol: tradeRequest.symbol,
      quantity: tradeRequest.quantity,
      executed_price: 100 + Math.random() * 50,
      status: 'filled',
      timestamp: new Date()
    };

    if (!mockResult.success) {
      mockResult.error = 'Mock error: Insufficient buying power';
    }

    setState(prev => ({
      ...prev,
      isExecuting: false,
      lastTrade: mockResult,
      history: [mockResult, ...prev.history].slice(0, 50),
      error: mockResult.success ? null : mockResult.error
    }));

    return mockResult;
  }, []);

  return {
    ...state,
    executeTrade,
    executeQuickTrade,
    executeLimitOrder,
    executeStopOrder,
    executeMockTrade,
    clearHistory,
    getTradeStats
  };
};

// Helper function to format trade confirmation message
export const formatTradeConfirmation = (trade: TradeResponse): string => {
  if (!trade.success) {
    return `Trade failed: ${trade.error}`;
  }

  const action = trade.side === 'buy' ? 'Bought' : 'Sold';
  return `${action} ${trade.quantity} shares of ${trade.symbol} at $${trade.executed_price?.toFixed(2)}`;
};

// Helper function to calculate trade cost
export const calculateTradeCost = (
  quantity: number,
  price: number,
  commission: number = 0
): number => {
  return quantity * price + commission;
};