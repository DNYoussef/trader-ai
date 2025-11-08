#!/usr/bin/env python3
"""Direct Alpaca test without async wrapper"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient

# Your credentials
API_KEY = "PKMQWWO2BXYFSE7RCTPHUTS2T4"
SECRET_KEY = "7LQY1SqAgLPcHE6fziYu5WxLncAp97sDeevHY5Ci8432"

print("Testing direct Alpaca connection (no async)...")
print(f"Using paper trading: True")
print(f"Base URL: https://paper-api.alpaca.markets")
print()

try:
    # Initialize client with paper=True
    client = TradingClient(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        paper=True  # Paper trading
    )

    print("[OK] Client initialized")

    # Test connection by getting account
    print("Fetching account info...")
    account = client.get_account()

    print(f"\n[OK] Connected successfully!")
    print(f"  Account Number: {account.account_number}")
    print(f"  Status: {account.status}")
    print(f"  Currency: {account.currency}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Equity: ${float(account.equity):,.2f}")

    # Test getting market clock
    print("\nFetching market clock...")
    clock = client.get_clock()
    print(f"  Market is: {'OPEN' if clock.is_open else 'CLOSED'}")
    print(f"  Next open: {clock.next_open}")
    print(f"  Next close: {clock.next_close}")

except Exception as e:
    print(f"\n[FAIL] Connection failed!")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error message: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
