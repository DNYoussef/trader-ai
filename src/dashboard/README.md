# Gary×Taleb Risk Dashboard

Real-time risk monitoring dashboard for the Gary×Taleb trading system. Provides live updates of P(ruin) calculations, portfolio metrics, position monitoring, and risk alerts with <1 second latency.

## Features

### 🔴 Real-Time Risk Metrics
- **P(ruin) Calculation**: Live probability of ruin using Gary's DPI methodology
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Expected Shortfall**: Conditional VaR for tail risk assessment
- **Maximum Drawdown**: Peak-to-trough decline monitoring
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Beta & Volatility**: Market correlation and volatility tracking

### 📊 Live Position Monitoring
- Real-time position tracking with market values
- Unrealized P&L calculations
- Position weights and allocation monitoring
- Winners/losers analysis
- Last update timestamps

### 🚨 Alert Management System
- Configurable risk thresholds
- Real-time alert notifications
- Severity levels (Critical, High, Medium, Low)
- Alert acknowledgment and tracking
- Sound notifications (configurable)

### 📱 Mobile-Responsive Design
- Optimized for desktop, tablet, and mobile devices
- Touch-friendly interface
- Adaptive layout based on screen size
- Dark mode support

### ⚡ WebSocket Real-Time Updates
- Sub-second latency (<1000ms)
- Automatic reconnection with exponential backoff
- Connection status monitoring
- Efficient data streaming

## Architecture

### Backend (Python)
- **FastAPI**: High-performance async web framework
- **WebSockets**: Real-time bidirectional communication
- **Redis**: Pub/sub for real-time data streaming (optional)
- **NumPy/SciPy**: Financial calculations and risk metrics
- **Uvicorn**: ASGI server for production deployment

### Frontend (React/TypeScript)
- **React 18**: Modern UI framework with hooks
- **TypeScript**: Type-safe development
- **Redux Toolkit**: State management
- **Recharts**: Financial chart visualization
- **Tailwind CSS**: Utility-first styling
- **Framer Motion**: Smooth animations
- **Vite**: Fast build tool and development server

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis (optional, for production)

### 1. Backend Setup

```bash
# Navigate to dashboard directory
cd src/dashboard

# Install Python dependencies
pip install -r server/requirements.txt

# Start the WebSocket server
python run_server.py
```

The server will start on `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### 3. Production Build

```bash
# Build for production
cd frontend
npm run build

# Serve production build
npm run preview
```

## API Endpoints

### REST API
- `GET /api/health` - Health check
- `GET /api/metrics/current` - Current risk metrics
- `GET /api/positions` - Current positions
- `GET /api/alerts` - Risk alerts
- `POST /api/alerts/{id}/acknowledge` - Acknowledge alert

### WebSocket
- `ws://localhost:8000/ws/{client_id}` - Real-time updates

### WebSocket Message Types
```typescript
// Incoming messages
interface WebSocketMessage {
  type: 'risk_metrics' | 'position_update' | 'alert' | 'heartbeat';
  data: any;
  timestamp?: number;
}

// Outgoing messages
interface ClientMessage {
  type: 'subscribe' | 'unsubscribe' | 'ping';
  subscription?: string;
}
```

## Configuration

### Environment Variables
```bash
# Server configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
REDIS_HOST=localhost
REDIS_PORT=6379

# Frontend configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Risk Thresholds
Default alert thresholds (configurable in UI):
- **P(ruin)**: High: 10%, Critical: 20%
- **VaR 95%**: High: 5%, Critical: 10%
- **Max Drawdown**: High: 10%, Critical: 20%
- **Margin Used**: High: 80%, Critical: 90%

## Integration with Trading System

The dashboard integrates with the existing Gary×Taleb trading engine:

```python
from src.trading_engine import TradingEngine
from dashboard.server.websocket_server import RiskDashboardServer

# Initialize components
trading_engine = TradingEngine()
dashboard_server = RiskDashboardServer()

# Integration layer
integration = DashboardIntegration(trading_engine, dashboard_server)
```

### Data Flow
1. **Trading Engine** → **Integration Layer** → **Dashboard Server**
2. **Dashboard Server** → **WebSocket** → **Frontend**
3. **Frontend** → **Redux Store** → **UI Components**

## Performance Optimizations

### Backend
- Async/await throughout for non-blocking operations
- Connection pooling for database operations
- Efficient data structures for real-time calculations
- Background tasks for periodic cleanup

### Frontend
- Component memoization to prevent unnecessary re-renders
- Virtual scrolling for large position lists
- Lazy loading of historical data
- Optimized chart rendering with D3.js

### Network
- WebSocket compression for reduced bandwidth
- Batched updates to minimize message frequency
- Intelligent reconnection with backoff

## Monitoring and Diagnostics

### Health Checks
```bash
curl http://localhost:8000/api/health
```

### Connection Metrics
```bash
curl http://localhost:8000/api/system/connections
```

### Logs
- Server logs: `logs/dashboard_server.log`
- Browser console for frontend debugging
- WebSocket connection status in UI

## Security Considerations

### Production Deployment
- Enable HTTPS/WSS in production
- Configure CORS appropriately
- Implement rate limiting
- Use environment variables for secrets
- Enable security headers

### Authentication (Future Enhancement)
```python
# JWT-based authentication
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication
```

## Testing

### Backend Tests
```bash
# Run server tests
pytest server/tests/

# Test WebSocket connections
python server/tests/test_websocket.py
```

### Frontend Tests
```bash
# Run frontend tests
npm test

# E2E tests with Playwright
npm run test:e2e
```

## Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY server/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_server.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: risk-dashboard
  template:
    metadata:
      labels:
        app: risk-dashboard
    spec:
      containers:
      - name: dashboard
        image: risk-dashboard:latest
        ports:
        - containerPort: 8000
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if server is running on port 8000
   - Verify firewall settings
   - Check browser console for errors

2. **No Data Updates**
   - Verify trading engine connection
   - Check server logs for errors
   - Ensure WebSocket subscriptions are active

3. **Performance Issues**
   - Reduce chart data points
   - Check browser memory usage
   - Verify network latency

### Debug Mode
```bash
# Start server in debug mode
DEBUG=1 python run_server.py

# Start frontend in debug mode
npm run dev -- --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

### Code Style
- Python: Black formatter, isort imports
- TypeScript: Prettier formatter, ESLint rules
- CSS: Tailwind CSS utilities preferred

## License

This project is part of the Gary×Taleb trading system and is proprietary software.

## Support

For technical support or questions:
- Check logs in `logs/` directory
- Review browser console for frontend issues
- Submit issues with detailed reproduction steps