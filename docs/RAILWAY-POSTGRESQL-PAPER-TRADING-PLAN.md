# Railway PostgreSQL Paper Trading Deployment Plan

**Created**: 2026-01-03
**Status**: READY FOR EXECUTION
**Goal**: Enable 24/7 cloud-based paper trading with persistent state

---

## Current State

### Validated Components
- [x] Local paper trading script works (`launch_enhanced_paper_trading.py`)
- [x] Phase 5 components functional (NarrativeGap, BrierTracker, DPI)
- [x] First trade executed locally with full monitoring
- [x] Railway service healthy at `trader-ai-production.up.railway.app`
- [x] GitHub repo: `DNYoussef/trader-ai` (main branch)
- [x] Auto-deploy via GitHub push configured

### Missing for Cloud Execution
- [ ] PostgreSQL database for trade persistence
- [ ] DATABASE_URL environment variable
- [ ] Trade/state persistence layer
- [ ] Scheduled execution mechanism

---

## Architecture

```
                    +------------------+
                    |   GitHub Push    |
                    +--------+---------+
                             |
                             v
+------------------+   +------------------+   +------------------+
|   trader-ai      |   |   PostgreSQL     |   |   Worker/Cron    |
|   (Dashboard)    |<->|   (New Service)  |<->|   (Paper Trade)  |
|   Port 8080      |   |   DATABASE_URL   |   |   Scheduled      |
+------------------+   +------------------+   +------------------+
                             |
                             v
                    +------------------+
                    |   Trade History  |
                    |   Portfolio State|
                    |   Phase 5 Metrics|
                    +------------------+
```

---

## Execution Steps

### Step 1: Add PostgreSQL to Railway (5 min)

**Option A: Via Railway Dashboard**
1. Go to https://railway.com/project/e211a4c5-bc03-48f2-ab2f-bc67b27b3a9d
2. Click "+ Create" button (top right)
3. Select "Database" -> "PostgreSQL"
4. Railway auto-provisions and sets `DATABASE_URL`

**Option B: Via Railway CLI**
```bash
cd D:\Projects\trader-ai
railway add --database postgres
```

### Step 2: Create Database Schema (10 min)

Create file: `src/database/trading_schema.py`

```python
"""
Trading Database Schema for Railway PostgreSQL
"""
import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///local_trading.db')

# Handle Railway's postgres:// vs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(20))
    direction = Column(String(10))  # 'long' or 'short'
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    ng_score = Column(Float)
    brier_adjustment = Column(Float)
    dpi_enhancement = Column(Float)
    is_simulation = Column(Boolean, default=True)

class PortfolioState(Base):
    __tablename__ = 'portfolio_state'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    capital = Column(Float)
    positions_json = Column(String(2000))  # JSON serialized
    total_pnl = Column(Float)
    trade_count = Column(Integer)

class Phase5Metrics(Base):
    __tablename__ = 'phase5_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    brier_score = Column(Float)
    ng_signal_count = Column(Integer)
    avg_ng_score = Column(Float)
    dpi_regime = Column(String(20))
    health_status = Column(String(50))

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
```

### Step 3: Update Paper Trading Script (15 min)

Modify `scripts/deployment/launch_enhanced_paper_trading.py`:

1. Add database imports at top
2. Replace in-memory state with database writes
3. Add session persistence

Key changes:
```python
from src.database.trading_schema import SessionLocal, Trade, PortfolioState, Phase5Metrics, init_db

class EnhancedPaperTradingSystem:
    def __init__(self, ...):
        # ... existing init ...
        init_db()  # Create tables if not exist
        self.db = SessionLocal()
        self._load_state_from_db()

    def _save_trade(self, trade_data):
        trade = Trade(**trade_data)
        self.db.add(trade)
        self.db.commit()

    def _save_state(self):
        state = PortfolioState(
            capital=self.current_capital,
            positions_json=json.dumps(self.positions),
            total_pnl=self.total_pnl,
            trade_count=len(self.trades)
        )
        self.db.add(state)
        self.db.commit()
```

### Step 4: Add Worker Service for Scheduled Trading (10 min)

**Option A: Railway Cron Service**

Create `railway-cron.json`:
```json
{
  "cron": "*/5 * * * *",
  "command": "python -m scripts.deployment.run_trading_cycle"
}
```

**Option B: Use Railway's scheduled deployments**

Add to `railway.toml`:
```toml
[cron]
enabled = true
schedule = "*/5 * * * *"
command = "python -m scripts.deployment.run_trading_cycle"
```

### Step 5: Update Requirements (2 min)

Add to `requirements.txt`:
```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

### Step 6: Deploy (2 min)

```bash
cd D:\Projects\trader-ai
git add .
git commit -m "feat: Add PostgreSQL persistence for cloud paper trading"
git push origin main
```

Railway auto-deploys on push.

### Step 7: Verify Deployment (5 min)

1. Check Railway logs:
   ```bash
   railway logs --service trader-ai
   ```

2. Check database tables created:
   ```bash
   railway connect postgres
   \dt  # List tables
   SELECT COUNT(*) FROM trades;
   ```

3. Monitor paper trading:
   ```bash
   railway logs --service trader-ai | grep TRADE
   ```

---

## Environment Variables Needed

| Variable | Source | Description |
|----------|--------|-------------|
| DATABASE_URL | Railway auto-provides | PostgreSQL connection string |
| TRADING_MODE | Set manually | `paper` for paper trading |
| ALPACA_API_KEY | Optional | Only for real Alpaca paper API |
| ALPACA_SECRET_KEY | Optional | Only for real Alpaca paper API |

---

## Rollback Plan

If issues occur:
1. Railway dashboard -> Deployments -> Rollback to previous
2. Or: `git revert HEAD && git push`

---

## Success Criteria

- [ ] PostgreSQL service running in Railway
- [ ] Tables created (trades, portfolio_state, phase5_metrics)
- [ ] Paper trading executing on schedule
- [ ] Trades persisting across restarts
- [ ] Dashboard showing trade history

---

## Estimated Time

| Step | Duration |
|------|----------|
| Add PostgreSQL | 5 min |
| Create Schema | 10 min |
| Update Script | 15 min |
| Add Worker | 10 min |
| Update Requirements | 2 min |
| Deploy | 2 min |
| Verify | 5 min |
| **Total** | **~50 min** |
