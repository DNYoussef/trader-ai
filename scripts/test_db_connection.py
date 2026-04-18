#!/usr/bin/env python3
"""Test database connection and table creation"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.database.trading_schema import test_connection, init_db, get_session, Trade, PortfolioState

print("=" * 50)
print("TRADER-AI DATABASE CONNECTION TEST")
print("=" * 50)

# Test 1: Connection
print("\n1. Testing connection...")
if test_connection():
    print("   Connection: OK")
else:
    print("   Connection: FAILED")
    sys.exit(1)

# Test 2: Table creation
print("\n2. Creating tables...")
if init_db():
    print("   Tables: OK")
else:
    print("   Tables: FAILED")
    sys.exit(1)

# Test 3: Query tables
print("\n3. Querying tables...")
try:
    session = get_session()
    trade_count = session.query(Trade).count()
    state_count = session.query(PortfolioState).count()
    print(f"   Trades: {trade_count}")
    print(f"   Portfolio States: {state_count}")
    session.close()
    print("   Query: OK")
except Exception as e:
    print(f"   Query: FAILED - {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("DATABASE STATUS: READY")
print("=" * 50)
