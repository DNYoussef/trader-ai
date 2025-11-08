#!/usr/bin/env python3
"""Test $5 SPY paper trade"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_engine import TradingEngine
from decimal import Decimal
import asyncio

print('Initializing trading engine...')
engine = TradingEngine()

if not engine.initialize():
    print('ERROR: Failed to initialize engine')
    sys.exit(1)

print('✓ Engine initialized successfully!')
print('✓ Account connected: PA3AQP89GW63')
print()
print('Executing $5 SPY buy order...')
print()

result = asyncio.run(engine.execute_manual_trade('SPY', Decimal('5.00'), 'buy'))

print('=' * 50)
print('TRADE RESULT')
print('=' * 50)
print(f'Success: {result.get("success", False)}')

if result.get('success'):
    print(f'\n✓ Trade executed successfully!')
    print(f'  Order ID: {result.get("order_id")}')
    print(f'  Symbol: {result.get("symbol")}')
    print(f'  Quantity: {result.get("quantity")}')
    print(f'  Side: {result.get("side").upper()}')
    print(f'  Status: {result.get("status")}')
    if result.get('filled_avg_price'):
        print(f'  Filled Price: ${float(result.get("filled_avg_price")):.2f}')
    print()
    print('🎉 Your first paper trade is complete!')
else:
    print(f'\n✗ Trade failed!')
    print(f'  Error: {result.get("error")}')
    sys.exit(1)
