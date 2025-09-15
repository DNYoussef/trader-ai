# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Commands
```bash
# Python development
python main.py                    # Run trading system (paper mode default)
python main.py --mode live        # Run in live trading mode (requires confirmation)
python main.py --test            # Test mode - initialize and exit

# Start UI Dashboard
start_ui.bat                     # Windows: Launch dashboard (opens backend + frontend)
./start_ui.sh                    # Linux/Mac: Launch dashboard

# Build and compilation
npm run build                     # TypeScript compilation
npm run build:watch              # Watch mode compilation
npm run clean                    # Clean build artifacts
npm run compile                  # Full build pipeline
npm run copy-compiled            # Copy compiled files

# Testing
python -m pytest tests/ -v       # Run all Python tests
npm run test                     # Run performance tests
npm run test:all                 # Run all test suites
npm run validate                 # Full validation pipeline
pytest tests/unit/ -v            # Unit tests only
pytest tests/integration/ -v     # Integration tests only
pytest tests/performance/ -v     # Performance tests only

# Performance testing
npm run performance:test         # Basic performance tests
npm run performance:benchmark    # Benchmarking suite
npm run performance:load         # Load testing
npm run performance:integration  # Integration performance tests

# Python dependency management
pip install -r requirements.txt  # Install all dependencies
```

## Architecture Overview

This is a **hybrid Python/TypeScript trading system** implementing Gary's capital progression methodology with Taleb's antifragility principles.

### Core System Architecture

**Main Entry Point**: `main.py` - CLI entry with mode selection and signal handling
**Trading Engine**: `src/trading_engine.py` - Central orchestration with async/await integration
**Broker Integration**: `src/brokers/` - Alpaca API adapter with mock fallback
**Gate System**: `src/gates/` - Capital progression from $200 through G0-G12 gates
**Weekly Cycles**: `src/cycles/` - Friday 4:10pm/6:00pm ET trading automation

### Key Components

1. **Trading Engine** (`src/trading_engine.py`)
   - Async orchestration of all trading components
   - Kill switch functionality and audit logging
   - Mode switching between paper/live trading

2. **Broker System** (`src/brokers/`)
   - `AlpacaAdapter`: Live/paper trading via Alpaca API
   - `BrokerInterface`: Abstract interface for all brokers
   - Mock implementations for development without API keys

3. **Gate Management** (`src/gates/`)
   - Progressive capital gates (G0: $200-499, G1: $500-999, etc.)
   - Risk constraints and position sizing rules
   - Automatic progression validation

4. **Market Data & Execution** (`src/market/`, `src/trading/`)
   - Real-time market data integration
   - Trade execution with fractional shares (6-decimal precision)
   - Portfolio management and rebalancing

5. **Safety Systems** (`src/safety/`)
   - Circuit breakers and kill switches
   - Multi-layer safety validation
   - Hardware authentication for live trading

6. **Performance Testing Infrastructure** (TypeScript)
   - `src/performance/` - Benchmarking and load testing
   - Real-time performance monitoring
   - CI/CD integration tests

### Development Modes

**Mock Mode** (Default for development):
- No external API dependencies
- Deterministic test results
- 100% feature coverage offline
- Set in `config/config.json` with `"mode": "paper"` and no API keys

**Paper Trading**:
- Live API integration with fake money
- Requires Alpaca paper trading credentials
- Real market data and timing

**Live Trading**:
- Real money trading (requires confirmation)
- Minimum $200 account balance
- Production audit logging

## Configuration

### Main Config: `config/config.json`
```json
{
    "mode": "paper",                    // paper or live
    "broker": "alpaca",                 // broker type
    "initial_capital": 200,             // starting capital
    "siphon_enabled": true,             // 50/50 profit split
    "audit_enabled": true               // WORM audit logging
}
```

### TypeScript Config: `tsconfig.json`
- Target: ES2020
- Output: `./dist`
- Source maps and declarations enabled
- Excludes tests and node_modules

## Testing Strategy

### Test Structure
- `tests/unit/` - Fast unit tests, no dependencies
- `tests/integration/` - Integration with mocked external services
- `tests/performance/` - Performance and benchmarking tests
- `tests/workflow-validation/` - End-to-end workflow validation

### Mock-First Development
All tests designed to work without external dependencies:
- Mock Alpaca API responses
- Deterministic market data
- Simulated time progression for weekly cycles

### Performance Requirements
- Order validation: <100ms
- Weekly cycle execution: <2 minutes
- Memory usage: <100MB during trading
- Full test suite: <30 seconds

## Dashboard UI

### Real-Time Risk Dashboard
Located in `src/dashboard/`, provides live monitoring of trading system:

**Backend API** (FastAPI + WebSocket):
- `src/dashboard/run_server_simple.py` - Simplified server without Redis
- Runs on `http://localhost:8000`
- WebSocket endpoint: `ws://localhost:8000/ws/{client_id}`
- REST endpoints: `/api/health`, `/api/metrics/current`, `/api/positions`

**Frontend** (React + TypeScript + Vite):
- `src/dashboard/frontend/` - React application
- Runs on `http://localhost:3000`
- Components: MetricCard, PositionTable, RiskChart, AlertList
- Real-time updates via WebSocket
- Redux for state management

### Starting the Dashboard
```bash
# Quick start (Windows)
start_ui.bat

# Manual start - Backend
cd src/dashboard
python run_server_simple.py

# Manual start - Frontend
cd src/dashboard/frontend
npm install  # First time only
npm run dev
```

### Dashboard Features
- **Real-time Risk Metrics**: P(ruin), VaR, Sharpe ratio, volatility
- **Position Monitoring**: Live position updates with P&L
- **Alert System**: Configurable risk thresholds and notifications
- **WebSocket Updates**: Sub-second latency for all metrics
- **Mock Data Mode**: Development mode with simulated data

## Project Structure

```
src/                          # Core source code
├── trading_engine.py          # Main orchestration engine
├── brokers/                   # Broker integrations (Alpaca + mocks)
├── gates/                     # Capital progression system
├── cycles/                    # Weekly automation (Friday timing)
├── safety/                    # Kill switches and circuit breakers
├── portfolio/                 # Portfolio management
├── market/                    # Market data integration
├── trading/                   # Trade execution
├── performance/              # TypeScript performance testing
├── security/                 # Security and compliance
├── intelligence/             # AI and pattern recognition
└── enterprise/               # Enterprise features

scripts/                      # Organized utility scripts
├── demos/                    # Demo and example scripts
├── training/                 # ML training scripts
├── validation/               # System validation scripts
└── deployment/               # Deployment scripts

tests/                        # Test suites
├── integration/              # Integration tests
├── unit/                     # Unit tests
├── performance/              # Performance tests
└── validation/               # Validation tests

docs/                         # Documentation
├── reports/                  # Phase and completion reports
├── plans/                    # Planning documents
├── specs/                    # Specifications
├── pre-mortems/             # Pre-mortem analyses
└── architecture/            # Architecture documentation

archive/                      # Historical files
├── legacy-plans/            # Old planning files
├── phase-reports/           # Archived phase reports
└── metrics/                 # Historical metrics

config/                       # Configuration files
data/                         # Data files
logs/                         # Log files
```

## Development Workflow

### Starting Development
1. Use mock mode by default (no API keys needed)
2. Run `python main.py --test` to validate setup
3. Use `pytest tests/unit/` for fast feedback
4. Run full test suite before committing

### Adding Features
1. Implement in mock mode first
2. Add comprehensive unit tests
3. Validate with integration tests
4. Performance test if needed
5. Update audit trail validation

### Production Deployment
1. All tests must pass
2. Performance benchmarks met
3. Security validation complete
4. Audit logging verified
5. Kill switch functionality tested

## Common Development Tasks

### Running Single Tests
```bash
# Test specific component
pytest tests/unit/test_trading_engine.py -v
pytest tests/integration/test_broker_integration.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html

# Performance testing
pytest tests/performance/ --benchmark-only
```

### Development Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --test

# Run with profiling
python -m cProfile -o profile.stats main.py --test
```

### Working with Gates
```bash
# Test gate progression
python -c "
from src.gates.gate_manager import GateManager
manager = GateManager()
print(f'Current gate: {manager.current_gate.value}')
print(f'Status: {manager.get_status_report()}')
"
```

### Running Organized Scripts
```bash
# Demo scripts
python scripts/demos/demo_enhanced_trading.py
python scripts/demos/demo_dpi_system.py

# Training scripts
python scripts/training/simple_train.py
python scripts/training/execute_training.py

# Validation scripts
python scripts/validation/simple_validation.py
python scripts/validation/validate_alpha_systems.py

# Deployment scripts
python scripts/deployment/launch_enhanced_paper_trading.py

# Integration tests
python scripts/tests/integration/test_complete_integration.py
```

## Security Considerations

- Never commit API keys or secrets
- All production credentials via environment variables
- Audit logs are append-only with cryptographic verification
- Kill switch activation logs all activity
- Paper mode has additional safety constraints

## Performance Monitoring

Key metrics tracked:
- Trade execution latency
- Memory usage during cycles
- API response times
- Test suite execution time
- System resource utilization

Performance tests run automatically via `npm run performance:test` and integrate with CI/CD pipelines.