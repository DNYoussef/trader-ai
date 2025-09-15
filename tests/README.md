# Foundation Phase Test Suite

A comprehensive test suite for the Foundation phase components including broker integration, gate management, weekly cycles, and end-to-end integration testing.

## 🎯 Overview

This test suite provides >80% coverage testing for:

- **Broker Integration** - Connection management, order processing, position tracking
- **Gate Management** - Risk validation, progression tracking, violation management  
- **Weekly Cycles** - Timing, allocations, delta calculations, phase progression
- **Integration** - End-to-end scenarios combining all components
- **Error Handling** - Comprehensive edge case and error scenario testing
- **Performance** - Concurrency, thread safety, and stress testing

## 📁 Structure

```
tests/
├── foundation/                    # Foundation phase test modules
│   ├── __init__.py
│   ├── test_broker_integration.py # Broker connectivity & trading tests
│   ├── test_gate_manager.py       # Risk gate validation tests
│   ├── test_weekly_cycle.py       # Cycle management tests
│   └── test_integration.py        # End-to-end integration tests
├── mocks/                         # Mock objects for testing
│   ├── __init__.py
│   ├── mock_broker.py            # Comprehensive broker simulation
│   ├── mock_gate_manager.py      # Gate validation simulation
│   └── mock_weekly_cycle.py      # Weekly cycle simulation
├── conftest.py                   # Shared fixtures and configuration
├── pytest.ini                   # Pytest configuration
├── requirements.txt              # Testing dependencies
├── run_tests.py                 # Test runner script
└── README.md                    # This file
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
# Run complete test suite with coverage
python tests/run_tests.py

# Or use pytest directly
pytest tests/ --cov=foundation --cov-report=html
```

### Run Specific Components

```bash
# Broker tests only
python tests/run_tests.py --component broker

# Gate manager tests only  
python tests/run_tests.py --component gates

# Weekly cycle tests only
python tests/run_tests.py --component cycle

# Integration tests only
python tests/run_tests.py --component integration
```

### Quick Smoke Tests

```bash
# Fast validation (no slow tests)
python tests/run_tests.py --smoke-only

# Or with pytest
pytest tests/ -m "not slow" --maxfail=5
```

## 📊 Test Categories

### Unit Tests
- **Broker Integration** (`test_broker_integration.py`)
  - Connection management (delays, failures, recovery)
  - Order placement (market, limit, stop orders)
  - Position tracking (accumulation, reduction, P&L)
  - Account management (balance, buying power)
  - Error handling (network issues, order rejections)
  - Performance metrics (trade counting, success rates)

- **Gate Management** (`test_gate_manager.py`) 
  - Gate registration and lifecycle
  - Validation logic (risk, position, loss limits)
  - Violation tracking and resolution
  - Bulk validation and fail-fast behavior
  - Performance metrics and callbacks
  - Thread safety and concurrent operations

- **Weekly Cycles** (`test_weekly_cycle.py`)
  - Cycle creation and phase progression
  - Allocation management (creation, updates, approval)
  - Delta calculations (current vs target positions)
  - Execution workflow (validation, ordering, completion)
  - Timing and market hours validation
  - Performance tracking across cycles

### Integration Tests
- **End-to-End Workflows** (`test_integration.py`)
  - Complete trading cycle (start to completion)
  - Risk-based allocation approval
  - Coordinated execution across all components
  - Error cascade handling and recovery
  - Multi-cycle performance scenarios
  - Stress testing with high volumes

### Mock Objects
All tests use sophisticated mock objects that simulate realistic behavior:

- **MockBroker** - Full broker simulation with configurable delays, errors, market data
- **MockGateManager** - Risk validation with customizable gates and violation tracking  
- **MockWeeklyCycleManager** - Complete cycle management with phase progression

## 🔧 Configuration

### Pytest Markers
- `@pytest.mark.broker` - Broker-related tests
- `@pytest.mark.gates` - Gate manager tests
- `@pytest.mark.cycle` - Weekly cycle tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests (use `--runslow`)
- `@pytest.mark.concurrent` - Concurrency tests
- `@pytest.mark.error_handling` - Error handling tests

### Coverage Requirements
- **Overall Coverage**: ≥80%
- **Individual Files**: ≥75% recommended
- **Critical Paths**: ≥90% (risk validation, execution)

### Test Fixtures
- `fast_broker` - No delays, immediate responses
- `reliable_broker` - Realistic delays, no errors
- `unreliable_broker` - Includes random failures
- `strict_gate_manager` - Tight risk thresholds  
- `lenient_gate_manager` - Loose risk thresholds
- `foundation_system_fast` - Integrated system for testing
- `sample_portfolio_data` - Realistic portfolio test data

## 📋 Test Scenarios

### Broker Integration Scenarios
- ✅ Successful connection and disconnection
- ✅ Order placement (buy/sell, different types)  
- ✅ Position tracking and P&L calculations
- ✅ Error handling (connection failures, order rejections)
- ✅ Concurrent operations and thread safety
- ✅ Performance under high frequency trading

### Gate Management Scenarios  
- ✅ Risk gate validation (portfolio limits)
- ✅ Violation detection and tracking
- ✅ Multi-gate validation with fail-fast
- ✅ Custom validation logic and callbacks
- ✅ Concurrent validation operations
- ✅ Historical violation analysis

### Weekly Cycle Scenarios
- ✅ Complete cycle progression (all phases)
- ✅ Allocation planning and delta calculations
- ✅ Approval workflow with risk validation
- ✅ Execution coordination with broker
- ✅ Performance tracking across cycles
- ✅ Multi-cycle portfolio evolution

### Integration Scenarios
- ✅ End-to-end trading workflow
- ✅ Risk-gated allocation approval
- ✅ Coordinated multi-component execution
- ✅ Error cascade handling and recovery
- ✅ Realistic trading simulation
- ✅ Large portfolio stress testing

## 🐛 Testing Edge Cases

### Error Conditions
- Network connectivity failures
- Broker API errors and timeouts
- Invalid allocation parameters
- Risk threshold violations
- Concurrent access conflicts
- Resource exhaustion scenarios

### Boundary Conditions
- Zero portfolio values
- Maximum position limits
- 100% allocation scenarios
- Tiny trade amounts
- Market hours edge cases
- Phase transition timing

### Performance Scenarios
- High-frequency allocation updates
- Large numbers of concurrent operations
- Long-running cycle progressions
- Memory usage under stress
- Thread safety validation

## 📈 Coverage Analysis

### Current Coverage Targets
- **Broker Integration**: >85%
- **Gate Management**: >85% 
- **Weekly Cycles**: >85%
- **Integration**: >75%
- **Mock Objects**: >70%

### Critical Path Coverage
- Risk validation logic: >95%
- Order execution paths: >90%
- Error handling: >80%
- Thread safety: >75%

## 🚨 Running Tests

### Development Workflow
```bash
# Quick validation during development
pytest tests/foundation/test_broker_integration.py::TestBrokerConnection -v

# Full component test
python tests/run_tests.py --component broker

# Coverage check
pytest tests/ --cov=foundation --cov-report=term-missing

# Performance test  
pytest tests/ -m performance --tb=short
```

### CI/CD Pipeline
```bash
# Complete validation for CI
python tests/run_tests.py --coverage-target 80

# Parallel execution for speed
pytest tests/ -n auto --cov=foundation

# Generate reports
pytest tests/ --junitxml=results/junit.xml --cov-report=xml:results/coverage.xml
```

### Debugging Failed Tests
```bash
# Stop on first failure with full traceback
pytest tests/ -x --tb=long

# Run specific failing test with output
pytest tests/foundation/test_integration.py::TestFoundationSystemIntegration::test_complete_trading_workflow -s -vv

# Debug with pdb
pytest tests/ --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

## 📊 Test Reports

Test execution generates several reports:

- **Coverage HTML**: `coverage/html/index.html`
- **Coverage JSON**: `coverage/coverage.json` 
- **JUnit XML**: `coverage/junit.xml`
- **Performance**: Console output with timing

## 🔍 Best Practices

### Writing New Tests
1. Use descriptive test names explaining what is tested
2. Follow AAA pattern: Arrange, Act, Assert
3. Use appropriate fixtures for test data
4. Mock external dependencies properly
5. Test both success and failure scenarios
6. Include performance considerations
7. Document complex test logic

### Mock Object Usage
1. Configure realistic delays for timing tests
2. Use error rates to test reliability
3. Set up callbacks to verify integration
4. Reset state between tests
5. Validate mock behavior matches real systems

### Coverage Guidelines  
1. Aim for >80% overall coverage
2. Focus on critical business logic first
3. Test error handling paths
4. Include edge cases and boundary conditions
5. Don't ignore hard-to-test code
6. Use coverage reports to guide testing

## 🤝 Contributing

When adding new tests:

1. Place in appropriate test module
2. Add relevant pytest markers
3. Update this README if adding new categories
4. Ensure tests pass in isolation and with full suite
5. Maintain or improve coverage percentage
6. Add fixtures to `conftest.py` if reusable

## 📝 Notes

- All tests are designed to run independently
- Mock objects provide deterministic behavior
- Integration tests validate component interaction
- Performance tests ensure scalability
- Error handling tests verify robustness
- Thread safety tests validate concurrent usage

The test suite is continuously maintained to ensure reliability and comprehensive coverage of all Foundation phase functionality.