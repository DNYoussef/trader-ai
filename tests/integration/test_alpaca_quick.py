#!/usr/bin/env python3
"""Quick Alpaca connection test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient

# Your credentials
API_KEY = "PKMQWWO2BXYFSE7RCTPHUTS2T4"
SECRET_KEY = "7LQY1SqAgLPcHE6fziYu5WxLncAp97sDeevHY5Ci8432"

print("Testing Alpaca connection...")
try:
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = client.get_account()

    print(f"✓ Connected successfully!")
    print(f"  Account Value: ${float(account.equity):,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")

except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)
