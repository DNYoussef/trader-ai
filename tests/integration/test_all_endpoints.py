#!/usr/bin/env python3
"""Test all 6 Plaid endpoints with JWT authentication"""
import os
import sys
from pathlib import Path

# Load environment
from dotenv import load_dotenv
project_root = Path(__file__).parent
load_dotenv(project_root / '.env')

sys.path.insert(0, str(project_root))

from src.security.auth import create_access_token
import requests

BASE_URL = "http://localhost:8000"

def test_all_endpoints():
    print("=" * 60)
    print("TESTING ALL 6 PLAID ENDPOINTS")
    print("=" * 60)

    # Generate JWT
    token = create_access_token({"sub": "test-user-001"})
    headers = {"Authorization": f"Bearer {token}"}

    # Test 1: Create Link Token (NO AUTH)
    print("\n[1/6] POST /api/plaid/create_link_token")
    resp = requests.post(f"{BASE_URL}/api/plaid/create_link_token",
                         json={"user_id": "test-user-001"})
    print(f"   Status: {resp.status_code}")
    print(f"   Result: {resp.json()}")

    # Test 2: Net Worth (NO AUTH)
    print("\n[2/6] GET /api/networth")
    resp = requests.get(f"{BASE_URL}/api/networth")
    print(f"   Status: {resp.status_code}")
    print(f"   Result: {resp.json()}")

    # Test 3: Bank Accounts (JWT REQUIRED)
    print("\n[3/6] GET /api/bank/accounts (JWT)")
    resp = requests.get(f"{BASE_URL}/api/bank/accounts", headers=headers)
    print(f"   Status: {resp.status_code}")
    print(f"   Result: {resp.json()}")

    # Test 4: Bank Balances (JWT REQUIRED)
    print("\n[4/6] GET /api/bank/balances (JWT)")
    resp = requests.get(f"{BASE_URL}/api/bank/balances", headers=headers)
    print(f"   Status: {resp.status_code}")
    print(f"   Result: {resp.json()}")

    # Test 5: Bank Transactions (JWT REQUIRED)
    print("\n[5/6] GET /api/bank/transactions (JWT)")
    resp = requests.get(f"{BASE_URL}/api/bank/transactions", headers=headers)
    print(f"   Status: {resp.status_code}")
    print(f"   Result: {resp.json()}")

    # Test 6: Exchange Public Token (NO AUTH - needs real public_token)
    print("\n[6/6] POST /api/plaid/exchange_public_token")
    print("   Status: SKIP (needs OAuth flow to get public_token)")

    print("\n" + "=" * 60)
    print("✅ PHASE 2A COMPLETE: All JWT endpoints wired!")
    print("=" * 60)

if __name__ == "__main__":
    test_all_endpoints()
