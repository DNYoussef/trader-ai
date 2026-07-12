# 🗺️ Complete UI Feature & Connection Map

**System**: trader-ai AppUnified + Plaid Banking Integration
**Date**: 2025-11-07
**Purpose**: Comprehensive mapping of all UI features to backend connections

---

## 📊 HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + TypeScript)                │
│                    http://localhost:3000                         │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ├─── HTTP/REST Requests
                    │
                    ├─── WebSocket Connection (ws://localhost:8000/ws)
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                  BACKEND (Python + FastAPI)                      │
│                    http://localhost:8000                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Trading APIs │  │ Plaid APIs   │  │ AI/ML APIs   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ├─── Trading Engine (src/trading_engine.py)
                    │
                    ├─── Gate Manager (src/gates/gate_manager.py)
                    │
                    ├─── Plaid Client (src/finances/plaid_client.py)
                    │
                    └─── Alpaca Broker (src/brokers/alpaca_adapter.py)
```

---

## 🎯 APP STRUCTURE: 4 MODES × 5 TABS

### Mode Selector (Dropdown)
**Location**: Header top-right
**Component**: `AppUnified.tsx` line 208-219
**State**: `const [appMode, setAppMode] = useState('enhanced')`
**Backend Connection**: NONE (pure frontend state)
**Available Modes**:
1. **Simple** - Basic dashboard
2. **Enhanced** - Full trading (default)
3. **Educational** - Learning mode
4. **Professional** - Complete pro environment

**Affects**: Which tabs and components are visible

---

### Navigation Tabs
**Location**: Header below mode selector
**Component**: `AppUnified.tsx` line 238-260
**State**: `const [activeTab, setActiveTab] = useState('overview')`
**Backend Connection**: NONE (pure frontend routing)
**Available Tabs**:
1. **Overview** (all modes)
2. **Terminal** (Enhanced, Professional)
3. **Analysis** (Enhanced, Professional)
4. **Learn** (Educational)
5. **Progress** (all modes)

---

## 📍 TAB 1: OVERVIEW

### Section 1.1: Unified Net Worth Card (NEW - Plaid Integration)
**Location**: Top of Overview tab
**Component**: `UnifiedNetWorthCard.tsx` (285 lines)
**File**: `src/dashboard/frontend/src/components/UnifiedNetWorthCard.tsx`

**Props Required**:
```typescript
{
  tradingNAV: number,           // From /api/metrics/current → portfolio_value
  totalBankBalance: number,     // From /api/networth → bank_total
  historicalData: NetWorthData[] // Combined historical data
}
```

**Backend Connections**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/metrics/current` | GET | `portfolio_value` | Every 10s (WebSocket) |
| `/api/networth` | GET | `bank_total`, `trader_ai_nav` | On demand |
| `/api/bank/balances` | GET | Bank account balances | After OAuth |

**Displayed Data**:
- Combined total: Trading + Banking
- Trading value (blue card)
- Banking value (green card)
- Breakdown percentages
- Historical chart (Area chart)
- Trend indicators (up/down arrows)
- Summary stats (Liquid, Invested, Total)

**User Actions**:
- View combined net worth
- See breakdown by account type
- View historical trends

---

### Section 1.2: Plaid Connect Button
**Location**: Top-right of Unified Net Worth Card
**Component**: `PlaidLinkButton.tsx` (6.0 KB)
**File**: `src/dashboard/frontend/src/components/PlaidLinkButton.tsx`

**Backend Connections**:
| API Endpoint | Method | Purpose | Authentication |
|--------------|--------|---------|----------------|
| `/api/plaid/create_link_token` | POST | Generate OAuth link token | None |
| `/api/plaid/exchange_public_token` | POST | Exchange for JWT | None (returns JWT) |

**Flow**:
```
1. User clicks "Connect Bank Account"
   ↓
2. Frontend → POST /api/plaid/create_link_token
   ← Returns: { link_token }
   ↓
3. react-plaid-link opens OAuth modal
   ↓
4. User authenticates with Plaid (phone/guest)
   ↓
5. Plaid returns public_token
   ↓
6. Frontend → POST /api/plaid/exchange_public_token
   ← Returns: { jwt_token }
   ↓
7. JWT stored in localStorage
   ↓
8. Frontend can now call protected endpoints:
   - /api/bank/accounts (GET with JWT)
   - /api/bank/balances (GET with JWT)
   - /api/bank/transactions (GET with JWT)
```

**User Actions**:
- Click to initiate OAuth
- Connect bank accounts via Plaid

---

### Section 1.3: Risk Metrics Grid
**Location**: Below Unified Net Worth
**Components**: 4-8 metric cards
**File**: `src/dashboard/frontend/src/components/MetricCardSimple.tsx`

**Backend Connection**:
| API Endpoint | Method | Hook | Update Frequency |
|--------------|--------|------|------------------|
| `/api/metrics/current` | GET | `useTradingData()` | Every 10s |

**Individual Cards**:

#### Card 1.3.1: Portfolio Value
**Component**: `<PortfolioValueCard value={metrics.portfolio_value} />`
**Data Source**: `metrics.portfolio_value` from API
**Format**: `$10,000.00`
**Calculation**: From Alpaca account balance

#### Card 1.3.2: P(ruin)
**Component**: `<PRuinCard value={metrics.p_ruin} />`
**Data Source**: `metrics.p_ruin` from API
**Format**: `12.5%`
**Calculation**: Probability of ruin based on Kelly criterion

#### Card 1.3.3: VaR 95
**Component**: `<VarCard value={metrics.var_95} />`
**Data Source**: `metrics.var_95` from API
**Format**: `$200.00`
**Calculation**: Value at Risk (95% confidence)

#### Card 1.3.4: Sharpe Ratio
**Component**: `<SharpeRatioCard value={metrics.sharpe_ratio} />`
**Data Source**: `metrics.sharpe_ratio` from API
**Format**: `1.50`
**Calculation**: Risk-adjusted return metric

#### Card 1.3.5: Drawdown (Secondary)
**Component**: `<DrawdownCard value={metrics.max_drawdown} />`
**Data Source**: `metrics.max_drawdown` from API
**Format**: `5.2%`
**Calculation**: Maximum portfolio decline

#### Cards 1.3.6-8: Advanced Metrics (if enabled)
**Visibility**: `showAdvancedMetrics` setting
**Additional Metrics**:
- Expected Shortfall: `metrics.expected_shortfall`
- Beta: `metrics.beta`
- Volatility: `metrics.volatility`

---

### Section 1.4: Live Charts
**Location**: Below metrics grid
**Component**: `LiveChartsEnhanced.tsx`
**File**: `src/dashboard/frontend/src/components/LiveChartsEnhanced.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/metrics/current` | GET | All metrics | Every 10s |
| WebSocket `ws://localhost:8000/ws` | WS | Real-time updates | Sub-second |

**Chart Data Generation**:
```typescript
// AppUnified.tsx lines 126-142
const [chartData, setChartData] = useState([]);

useEffect(() => {
  if (metrics) {
    setChartData(prev => {
      const newPoint = {
        timestamp: Date.now(),
        portfolio_value: metrics.portfolio_value,  // From API
        p_ruin: metrics.p_ruin,                    // From API
        sharpe_ratio: metrics.sharpe_ratio,        // From API
        var_95: metrics.var_95                     // From API
      };
      return [...prev.slice(-19), newPoint]; // Keep last 20 points
    });
  }
}, [metrics]);
```

**Charts Rendered**:
1. **Portfolio Value Chart** (Area)
2. **P(ruin) Chart** (Line)
3. **VaR Chart** (Area)
4. **Sharpe Ratio Chart** (Line)
5. **Drawdown Chart** (Area)

**Library**: Recharts (real SVG charts)

---

### Section 1.5: AI Strategy Panel
**Location**: Right side of charts
**Component**: `AIStrategyPanel.tsx`
**File**: `src/dashboard/frontend/src/components/AIStrategyPanel.tsx`

**Backend Connections**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/ai/timesfm/volatility` | GET | VIX forecasts | Every 5s |
| `/api/ai/fingpt/sentiment` | GET | Market sentiment | Every 5s |
| `/api/ai/fingpt/forecast` | GET | Price predictions | Every 5s |
| `/api/ai/features/32d` | GET | DPI score | Every 5s |

**Hook**: `useAIData(5000)` - auto-refresh every 5 seconds

**Displayed Data**:
- Aggregate AI signal (BUY/SELL/HOLD)
- Confidence percentage
- DPI Score (0-100)
- TimesFM volatility forecasts
- FinGPT sentiment analysis

**Fallback**: If AI models not loaded, uses realistic mock data

---

### Section 1.6: AI Signals Card
**Location**: Below AI Strategy Panel
**Component**: `AISignals.tsx`
**File**: `src/dashboard/frontend/src/components/AISignals.tsx`

**Backend Connection**: Same as AI Strategy Panel (aggregate signals)

**Displayed Data**:
- Current signal (BUY/SELL/HOLD)
- Confidence level
- DPI score

---

### Section 1.7: Quick Trade Card
**Location**: Below AI Signals
**Component**: `QuickTrade.tsx`
**File**: `src/dashboard/frontend/src/components/QuickTrade.tsx`

**Backend Connection**:
| API Endpoint | Method | Purpose | Authentication |
|--------------|--------|---------|----------------|
| `/api/trading/execute` | POST | Execute trade | None (should add) |
| `/api/market/quote/{symbol}` | GET | Get current price | None |

**User Actions**:
- Enter symbol
- Enter quantity
- Select BUY/SELL
- Execute trade

**Trade Execution Flow**:
```
1. User enters trade details
   ↓
2. Frontend → POST /api/trading/execute
   Body: { symbol, quantity, side, order_type }
   ↓
3. Backend → Trading Engine → Alpaca API
   ↓
4. Order executed
   ↓
5. Response: { success, order_id, filled_price }
```

---

### Section 1.8: Position Table
**Location**: Bottom left
**Component**: `PositionTable.tsx` or `PositionTableSimple.tsx`
**File**: `src/dashboard/frontend/src/components/PositionTableSimple.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/positions` | GET | All positions | Every 10s |

**Hook**: `useTradingData()` → `positions` array

**Table Columns**:
- Symbol
- Quantity
- Entry Price
- Current Price
- Market Value
- P&L (profit/loss)
- P&L %
- Weight (% of portfolio)

**User Actions**:
- View current positions
- Click row for details
- Sort by column

---

### Section 1.9: Recent Alerts
**Location**: Bottom right
**Component**: `RecentAlerts.tsx`
**File**: `src/dashboard/frontend/src/components/RecentAlerts.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/alerts` | GET | Alert list | Every 10s |
| WebSocket | WS | Real-time alerts | Sub-second |

**Hook**: `useTradingData()` → `alerts` array

**Alert Types**:
- Info (blue)
- Warning (yellow)
- Error (red)
- Success (green)

**Displayed Data**:
- Alert title
- Alert message
- Timestamp
- Severity level

**User Actions**:
- View recent alerts
- Acknowledge/dismiss alerts

---

### Section 1.10: Connected Bank Accounts (NEW)
**Location**: Bottom of Overview tab
**Component**: `BankAccountCard.tsx` (placeholder currently)
**File**: `src/dashboard/frontend/src/components/BankAccountCard.tsx`

**Backend Connections**:
| API Endpoint | Method | Data Used | Authentication |
|--------------|--------|-----------|----------------|
| `/api/bank/accounts` | GET | Account list | JWT Required |
| `/api/bank/balances` | GET | Real-time balances | JWT Required |

**Displayed Data** (after OAuth):
- Bank name
- Account type (Checking/Savings)
- Balance
- Last 4 digits
- Account status

**Current State**: Placeholder with instructions

**Future State**: Grid of BankAccountCard components

---

## 📍 TAB 2: TERMINAL (Trading Terminal)

### Section 2.1: Trading Controls
**Location**: Top of Terminal tab
**Component**: `TradingControls.tsx`
**File**: `src/dashboard/frontend/src/components/TradingControls.tsx`

**Backend Connections**:
| API Endpoint | Method | Purpose |
|--------------|--------|---------|
| `/api/trading/execute` | POST | Execute trade |
| `/api/positions/{symbol}/close` | POST | Close position |
| `/api/positions/close_all` | POST | Close all positions |
| `/api/market/quote/{symbol}` | GET | Get quote |

**Features**:
- Symbol input
- Quantity input
- Order type (Market/Limit)
- Side (BUY/SELL)
- Price input (for limits)
- Execute button

**Trade Execution Flow**: Same as Quick Trade

---

### Section 2.2: AI Strategy Panel (Terminal)
**Location**: Left column
**Component**: Same as Overview, with `autoRefresh={true}, refreshInterval={5000}`

**Backend Connections**: Same as Section 1.5

---

### Section 2.3: 32 Features Panel
**Location**: Right column
**Component**: `Feature32Panel.tsx`
**File**: `src/dashboard/frontend/src/components/Feature32Panel.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/features/realtime` | GET | 32 feature values | Every 5s (from AppUnified) |

**Data Format**:
```json
{
  "values": [20.0, 0.5, 0.0, ...], // 32 float values
  "timestamp": 1762550287
}
```

**Displayed Data**:
- Feature heatmap visualization
- Feature importance scores
- DPI score calculation
- Real-time updates

---

### Section 2.4: Live Charts (Terminal Mode)
**Location**: Below AI panels
**Component**: `LiveChartsEnhanced` with `showTerminal={true}`

**Backend Connections**: Same as Section 1.4

**Additional Features**:
- Larger charts
- More technical indicators
- Volume data

---

### Section 2.5: Active Positions Table
**Location**: Bottom of Terminal tab
**Component**: Same as Section 1.8

**Backend Connections**: Same as Section 1.8

---

## 📍 TAB 3: ANALYSIS

### Section 3.1: Inequality Panel
**Location**: Left column
**Component**: `InequalityPanelWrapper.tsx`
**File**: `src/dashboard/frontend/src/components/InequalityPanelWrapper.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/inequality` | GET | Inequality metrics | On demand |

**Displayed Data**:
- Gini coefficient
- Income distribution
- Wealth gaps
- Economic indicators

---

### Section 3.2: Contrarian Trades
**Location**: Right column
**Component**: `ContrarianTradesWrapper.tsx`
**File**: `src/dashboard/frontend/src/components/ContrarianTradesWrapper.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/contrarian` | GET | Contrarian opportunities | On demand |

**Displayed Data**:
- Contrarian trade ideas
- Sentiment divergence
- Crowded trades
- Opportunity scores

---

## 📍 TAB 4: LEARN (Educational)

### Section 4.1: Education Hub
**Location**: Full tab
**Component**: `EducationHub.tsx`
**File**: `src/dashboard/frontend/src/components/education/EducationHub.tsx`

**Backend Connection**: NONE (pure frontend educational content)

**Sub-Modules**:
1. **Decision Theory** (`DecisionTheory.tsx`)
   - Interactive decision trees
   - Risk calculators
   - Kelly criterion examples

2. **Antifragility** (`Antifragility.tsx`)
   - Barbell strategy builder
   - Black swan simulator
   - Convexity examples

3. **Quest System** (`QuestSystem.tsx`)
   - Collaborative learning quests
   - Gate progression integration
   - Achievement tracking

4. **Character Sheet** (`CharacterSheet.tsx`)
   - Skill trees
   - Energy management
   - Trader stats

5. **Time Management** (`TimeManagement.tsx`)
   - Session optimization
   - Exobrain notes
   - Workflow tracking

---

## 📍 TAB 5: PROGRESS

### Section 5.1: Trading Journey
**Location**: Left column
**Component**: `TradingJourney.tsx`
**File**: `src/dashboard/frontend/src/components/TradingJourney.tsx`

**Backend Connection**: Uses metrics from `useTradingData()`

**Displayed Data**:
- Progress timeline
- Milestone achievements
- Historical performance
- Learning progression

---

### Section 5.2: Gate Progression
**Location**: Right column
**Component**: `GateProgression.tsx`
**File**: `src/dashboard/frontend/src/components/GateProgression.tsx`

**Backend Connection**:
| API Endpoint | Method | Data Used | Update Frequency |
|--------------|--------|-----------|------------------|
| `/api/gates/status` | GET | Gate info | On demand |

**Response Format** (NOW FIXED):
```json
{
  "current_gate": "G0",
  "current_capital": 0.0,
  "gates": [
    {
      "id": "G0",
      "name": "Gate G0",
      "range": "$200-$500",
      "status": "current",
      "requirements": "2 allowed assets, 50% cash floor",
      "progress": 0
    },
    {
      "id": "G1",
      "name": "Gate G1",
      "range": "$500-$1,000",
      "status": "locked",
      "requirements": "5 allowed assets, 60% cash floor",
      "progress": null
    },
    ... (G2, G3)
  ]
}
```

**Displayed Data**:
- Current gate (G0-G3)
- Current capital
- Progress to next gate (%)
- Gate requirements
- Allowed assets per gate
- Cash floor requirements

**Gate Progression Rules**:
- **G0** ($200-499): ULTY, AMDY only, 50% cash floor
- **G1** ($500-999): +IAU, GLDM, VTIP, 60% cash floor
- **G2** ($1k-2.5k): +Factor ETFs, 65% cash floor
- **G3** ($2.5k-5k): +Options enabled, 70% cash floor, 0.5% theta limit

---

## 🔌 BACKEND API COMPLETE REFERENCE

### Core Trading APIs

| Endpoint | Method | Auth | Response | Update Source |
|----------|--------|------|----------|---------------|
| `/api/health` | GET | None | `{"status": "healthy"}` | System |
| `/api/metrics/current` | GET | None | Portfolio metrics | Trading Engine |
| `/api/positions` | GET | None | Position array | Alpaca API |
| `/api/alerts` | GET | None | Alert array | Alert System |

### Plaid Banking APIs

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/plaid/create_link_token` | POST | None | Start OAuth |
| `/api/plaid/exchange_public_token` | POST | None | Get JWT |
| `/api/bank/accounts` | GET | JWT | List accounts |
| `/api/bank/balances` | GET | JWT | Get balances |
| `/api/bank/transactions` | GET | JWT | Get transactions |
| `/api/networth` | GET | None | Combined net worth |

### AI/ML APIs

| Endpoint | Method | Auth | Response |
|----------|--------|------|----------|
| `/api/features/realtime` | GET | None | 32 feature values |
| `/api/ai/timesfm/volatility` | GET | None | VIX forecasts |
| `/api/ai/timesfm/risk` | GET | None | Risk analysis |
| `/api/ai/fingpt/sentiment` | GET | None | Market sentiment |
| `/api/ai/fingpt/forecast` | GET | None | Price predictions |
| `/api/ai/features/32d` | GET | None | Enhanced features |

### Gate System APIs

| Endpoint | Method | Auth | Response |
|----------|--------|------|----------|
| `/api/gates/status` | GET | None | Gate progression ✅ FIXED |

### Trading Execution APIs

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/trading/execute` | POST | None | Execute trade |
| `/api/positions/{symbol}/close` | POST | None | Close position |
| `/api/positions/close_all` | POST | None | Close all |
| `/api/market/quote/{symbol}` | GET | None | Get quote |

### Analysis APIs

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/inequality` | GET | None | Inequality metrics |
| `/api/contrarian` | GET | None | Contrarian opportunities |

### WebSocket

| Endpoint | Type | Purpose |
|----------|------|---------|
| `ws://localhost:8000/ws/{client_id}` | WebSocket | Real-time updates |

**Subscriptions**:
- `metrics` - Portfolio metrics updates
- `positions` - Position changes
- `alerts` - New alerts
- `gates` - Gate progression

---

## 📡 DATA FLOW PATTERNS

### Pattern 1: Polling (REST)
```
Frontend Component
  ↓ useEffect + setInterval (every 10s)
  → API Call (fetch)
  ↓ Response
  ← Data
  → Update State
  → Re-render
```

**Used By**:
- Risk metrics
- Positions
- Alerts (fallback)

---

### Pattern 2: WebSocket (Real-time)
```
Frontend Component
  ↓ useWebSocket hook
  → WebSocket connection
  → Subscribe to channels
  ↓ Server pushes updates
  ← Real-time data
  → Update Redux store
  → Re-render
```

**Used By**:
- Live charts
- Real-time alerts
- Position updates

---

### Pattern 3: On-Demand (User Action)
```
User clicks button
  ↓
Frontend Component
  → API Call
  ↓ Response
  ← Data
  → Update State
  → Re-render
```

**Used By**:
- Trade execution
- Plaid OAuth
- Gate status refresh
- Analysis panels

---

### Pattern 4: OAuth Flow (Multi-step)
```
1. User clicks "Connect Bank"
   ↓
2. POST /api/plaid/create_link_token
   ← {link_token}
   ↓
3. Open Plaid Link modal
   ↓
4. User authenticates
   ← {public_token}
   ↓
5. POST /api/plaid/exchange_public_token
   ← {jwt_token}
   ↓
6. Store JWT in localStorage
   ↓
7. Use JWT for protected calls:
   - GET /api/bank/accounts (with Authorization: Bearer {jwt})
   - GET /api/bank/balances (with Authorization: Bearer {jwt})
   - GET /api/bank/transactions (with Authorization: Bearer {jwt})
```

---

## 🎨 COMPONENT DEPENDENCY TREE

```
AppUnified
├── Header
│   ├── Mode Selector → appMode state
│   ├── Navigation Tabs → activeTab state
│   ├── Status Indicators
│   │   ├── API Status → isConnected from useTradingData()
│   │   └── Live Data Status → wsConnected state
│   └── Portfolio Value → metrics.portfolio_value
│
├── Tab: Overview
│   ├── UnifiedNetWorthCard → tradingNAV, bankTotal, historicalData
│   │   ├── PlaidLinkButton → /api/plaid/*
│   │   ├── Trading Value → metrics.portfolio_value
│   │   ├── Banking Value → /api/networth
│   │   └── Historical Chart → combined historical data
│   ├── Risk Metrics Grid
│   │   ├── PortfolioValueCard → metrics.portfolio_value
│   │   ├── PRuinCard → metrics.p_ruin
│   │   ├── VarCard → metrics.var_95
│   │   ├── SharpeRatioCard → metrics.sharpe_ratio
│   │   └── DrawdownCard → metrics.max_drawdown
│   ├── Charts Section
│   │   ├── LiveChartsEnhanced → chartData (generated from metrics)
│   │   ├── AIStrategyPanel → useAIData() → /api/ai/*
│   │   ├── AISignals → aggregate signals
│   │   └── QuickTrade → /api/trading/execute
│   ├── Tables Section
│   │   ├── PositionTable → positions from useTradingData()
│   │   └── RecentAlerts → alerts from useTradingData()
│   └── Bank Accounts Section
│       └── BankAccountCard (placeholder) → /api/bank/accounts
│
├── Tab: Terminal
│   ├── TradingControls → /api/trading/execute
│   ├── AIStrategyPanel → /api/ai/*
│   ├── Feature32Panel → /api/features/realtime
│   ├── LiveChartsEnhanced (terminal mode)
│   └── PositionTable → positions
│
├── Tab: Analysis
│   ├── InequalityPanelWrapper → /api/inequality
│   └── ContrarianTradesWrapper → /api/contrarian
│
├── Tab: Learn
│   └── EducationHub (no backend)
│       ├── DecisionTheory
│       ├── Antifragility
│       ├── QuestSystem
│       ├── CharacterSheet
│       └── TimeManagement
│
└── Tab: Progress
    ├── TradingJourney → metrics
    └── GateProgression → /api/gates/status
```

---

## 🔄 STATE MANAGEMENT ARCHITECTURE

### React Hooks (Local State)
```typescript
// AppUnified.tsx
const [appMode, setAppMode] = useState('enhanced');
const [activeTab, setActiveTab] = useState('overview');
const [wsConnected, setWsConnected] = useState(false);
const [chartData, setChartData] = useState([]);
const [features32, setFeatures32] = useState(null);
```

### Custom Hooks (Data Fetching)
```typescript
// useTradingData.ts - Lines 162-324
const {
  metrics,           // From /api/metrics/current
  positions,         // From /api/positions
  alerts,            // From /api/alerts
  gateStatus,        // From /api/gates/status
  isConnected,       // Connection status
  loading,           // Loading state
  error              // Error state
} = useTradingData();

// useAIData.ts - Lines 74-251
const {
  timesfmVolatility,  // From /api/ai/