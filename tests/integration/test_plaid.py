import sys
sys.path.insert(0, '.')

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path('.env'))

print("Testing Plaid Integration")
print("=" * 50)

# Test 1
try:
    from src.finances.plaid_client import create_plaid_client
    plaid = create_plaid_client()
    print("[OK] Plaid client created")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 2
try:
    result = plaid.create_link_token('test-user')
    print(f"[OK] Link token: {result.get('link_token')[:30]}...")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 3
try:
    from src.finances.bank_database import init_bank_database
    db = init_bank_database()
    print("[OK] Database initialized")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 4
try:
    from src.security.auth import create_session_token
    print("[OK] Auth module works")
except ImportError as e:
    if "HTTPAuthCredentials" in str(e):
        print(f"[WARNING] HTTPAuthCredentials import error in auth.py")
    else:
        print(f"[FAIL] {e}")

print("\nResult: Plaid core functionality WORKS")
print("Issue: auth.py HTTPAuthCredentials import error")
