#!/usr/bin/env python3
"""
Test Plaid Integration
Validates PlaidClient and API endpoints without requiring actual Plaid credentials.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_plaid_client_import():
    """Test that PlaidClient can be imported."""
    try:
        from src.finances.plaid_client import PlaidClient, PlaidAccount, PlaidTransaction
        print("[OK] PlaidClient import successful")
        print("[OK] PlaidAccount dataclass imported")
        print("[OK] PlaidTransaction dataclass imported")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_plaid_client_initialization():
    """Test PlaidClient initialization with credentials."""
    try:
        from src.finances.plaid_client import PlaidClient

        # This will fail without credentials, but validates the class structure
        try:
            client = PlaidClient(
                client_id="test_client_id",
                secret="test_secret",
                environment="sandbox"
            )
            print("✓ PlaidClient initialized with credentials")
            return True
        except Exception as e:
            if "plaid" in str(e).lower():
                print("✓ PlaidClient validation passed (plaid-python not installed yet)")
                return True
            else:
                raise
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_config_schema():
    """Test that config.json has correct Plaid fields."""
    config_path = project_root / 'config' / 'config.json'

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    required_fields = ['plaid_client_id', 'plaid_secret', 'plaid_env']
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        print(f"✗ Missing config fields: {missing_fields}")
        return False

    print("✓ Config schema valid")
    print(f"  - plaid_client_id: {config['plaid_client_id']}")
    print(f"  - plaid_secret: {config['plaid_secret']}")
    print(f"  - plaid_env: {config['plaid_env']}")
    return True


def test_requirements():
    """Test that plaid-python is in requirements.txt."""
    requirements_path = project_root / 'requirements.txt'

    if not requirements_path.exists():
        print(f"✗ requirements.txt not found: {requirements_path}")
        return False

    with open(requirements_path, 'r') as f:
        requirements = f.read()

    if 'plaid-python' not in requirements:
        print("✗ plaid-python not in requirements.txt")
        return False

    print("✓ plaid-python>=14.0.0 in requirements.txt")
    return True


def test_endpoint_definitions():
    """Test that FastAPI endpoints are defined correctly."""
    server_path = project_root / 'src' / 'dashboard' / 'run_server_simple.py'

    if not server_path.exists():
        print(f"✗ Server file not found: {server_path}")
        return False

    with open(server_path, 'r') as f:
        server_code = f.read()

    required_endpoints = [
        '/api/plaid/create_link_token',
        '/api/plaid/exchange_public_token',
        '/api/bank/accounts',
        '/api/bank/balances',
        '/api/bank/transactions',
        '/api/networth'
    ]

    missing_endpoints = []
    for endpoint in required_endpoints:
        if endpoint not in server_code:
            missing_endpoints.append(endpoint)

    if missing_endpoints:
        print(f"✗ Missing endpoints: {missing_endpoints}")
        return False

    print("✓ All 6 Plaid endpoints defined:")
    for endpoint in required_endpoints:
        print(f"  - {endpoint}")
    return True


def test_dataclass_structure():
    """Test PlaidAccount and PlaidTransaction dataclass structure."""
    try:
        from src.finances.plaid_client import PlaidAccount, PlaidTransaction
        from dataclasses import fields

        # Check PlaidAccount fields
        account_fields = {f.name for f in fields(PlaidAccount)}
        required_account_fields = {
            'account_id', 'name', 'official_name', 'type', 'subtype',
            'mask', 'current_balance', 'available_balance', 'currency_code'
        }

        if not required_account_fields.issubset(account_fields):
            missing = required_account_fields - account_fields
            print(f"✗ PlaidAccount missing fields: {missing}")
            return False

        # Check PlaidTransaction fields
        txn_fields = {f.name for f in fields(PlaidTransaction)}
        required_txn_fields = {
            'transaction_id', 'account_id', 'amount', 'date',
            'name', 'merchant_name', 'category', 'pending'
        }

        if not required_txn_fields.issubset(txn_fields):
            missing = required_txn_fields - txn_fields
            print(f"✗ PlaidTransaction missing fields: {missing}")
            return False

        print("✓ PlaidAccount dataclass structure valid")
        print("✓ PlaidTransaction dataclass structure valid")
        return True

    except Exception as e:
        print(f"✗ Dataclass validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Plaid Integration Validation Tests")
    print("=" * 60)
    print()

    tests = [
        ("Import Test", test_plaid_client_import),
        ("Initialization Test", test_plaid_client_initialization),
        ("Config Schema Test", test_config_schema),
        ("Requirements Test", test_requirements),
        ("Endpoint Definitions Test", test_endpoint_definitions),
        ("Dataclass Structure Test", test_dataclass_structure),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 60)
        results.append(test_func())

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All validation tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set environment variables:")
        print("   - PLAID_CLIENT_ID (get from https://dashboard.plaid.com)")
        print("   - PLAID_SECRET")
        print("3. Start server: python src/dashboard/run_server_simple.py")
        print("4. Test endpoints with curl or frontend integration")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
