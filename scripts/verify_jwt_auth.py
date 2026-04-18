#!/usr/bin/env python3
"""
JWT Authentication Verification Script

This script verifies that JWT authentication is properly configured
and working for the trader-ai API endpoints.

Usage:
    python scripts/verify_jwt_auth.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check if JWT_SECRET_KEY is set"""
    print("=" * 60)
    print("JWT AUTHENTICATION VERIFICATION")
    print("=" * 60)
    print()

    print("1. Checking environment variables...")
    jwt_secret = os.getenv("JWT_SECRET_KEY")

    if jwt_secret:
        print("   [OK] JWT_SECRET_KEY is set")
        print(f"   Key length: {len(jwt_secret)} characters")
        if len(jwt_secret) < 32:
            print("   [WARNING] Key is shorter than recommended 32 characters")
    else:
        print("   [ERROR] JWT_SECRET_KEY is NOT set")
        print("   Set it with: export JWT_SECRET_KEY='your-secret-key'")
        return False

    print()
    return True


def check_files():
    """Check if required files exist"""
    print("2. Checking required files...")

    files_to_check = [
        ("src/security/auth.py", "JWT token creation/verification"),
        ("src/security/auth_middleware.py", "JWT authentication middleware"),
        ("src/dashboard/run_server_simple.py", "FastAPI application"),
        ("tests/test_jwt_auth_middleware.py", "Test suite"),
    ]

    all_exist = True
    for file_path, description in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   [OK] {file_path}")
            print(f"        ({description})")
        else:
            print(f"   [ERROR] {file_path} NOT FOUND")
            all_exist = False

    print()
    return all_exist


def check_imports():
    """Check if required modules can be imported"""
    print("3. Checking Python imports...")

    imports_to_check = [
        ("fastapi", "FastAPI framework"),
        ("jose", "JWT library (python-jose)"),
        ("starlette", "ASGI framework"),
    ]

    all_imported = True
    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            print(f"   [OK] {module_name}")
            print(f"        ({description})")
        except ImportError:
            print(f"   [ERROR] {module_name} NOT FOUND")
            print(f"        Install with: pip install {module_name}")
            all_imported = False

    print()
    return all_imported


def check_middleware_config():
    """Check if middleware is configured in the app"""
    print("4. Checking middleware configuration...")

    app_file = project_root / "src/dashboard/run_server_simple.py"
    content = app_file.read_text()

    checks = [
        ("from src.security.auth_middleware import", "Middleware import"),
        ("JWT_AUTH_AVAILABLE", "JWT availability flag"),
        ("configure_jwt_auth_middleware", "Middleware configuration call"),
    ]

    all_configured = True
    for check_string, description in checks:
        if check_string in content:
            print(f"   [OK] {description}")
        else:
            print(f"   [ERROR] {description} NOT FOUND")
            all_configured = False

    print()
    return all_configured


def generate_test_token():
    """Generate a test JWT token"""
    print("5. Generating test JWT token...")

    try:
        from src.security.auth import create_access_token
        from datetime import timedelta

        # Generate test token
        test_token = create_access_token(
            data={"sub": "test_user_123"},
            expires_delta=timedelta(minutes=30)
        )

        print("   [OK] Test token generated successfully")
        print(f"   Token (first 50 chars): {test_token[:50]}...")
        print()

        print("   Use this token to test API endpoints:")
        print()
        print('   curl -H "Authorization: Bearer ' + test_token + '" \\')
        print("        http://localhost:8000/api/positions")
        print()

        return True

    except Exception as e:
        print(f"   [ERROR] Failed to generate token: {e}")
        print()
        return False


def show_protected_endpoints():
    """Display list of protected endpoints"""
    print("6. Protected API Endpoints (require JWT):")
    print()

    endpoints = [
        ("POST", "/api/trade/execute/{asset}", "Execute AI-recommended trade"),
        ("POST", "/api/trading/execute", "Execute real trades"),
        ("GET", "/api/metrics/current", "Current risk metrics"),
        ("GET", "/api/positions", "Portfolio positions"),
        ("GET", "/api/alerts", "Active alerts"),
        ("GET", "/api/engine/status", "Trading engine status"),
        ("GET", "/api/signals/recent", "Recent scan signals"),
        ("GET", "/api/signals/stats", "Signal statistics"),
        ("GET", "/api/inequality/data", "Inequality analysis"),
        ("GET", "/api/contrarian/opportunities", "Contrarian opportunities"),
        ("GET", "/api/ai/status", "AI calibration status"),
        ("GET", "/api/barbell/allocation", "Barbell allocation"),
        ("GET", "/api/gates/status", "Gate progression"),
    ]

    for method, path, description in endpoints:
        print(f"   {method:6} {path:35} - {description}")

    print()


def show_public_endpoints():
    """Display list of public endpoints"""
    print("7. Public Endpoints (no authentication required):")
    print()

    endpoints = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/api/health", "API health check"),
        ("GET", "/docs", "API documentation"),
        ("GET", "/redoc", "Alternative API docs"),
    ]

    for method, path, description in endpoints:
        print(f"   {method:6} {path:35} - {description}")

    print()


def main():
    """Run all verification checks"""
    checks_passed = []

    # Run checks
    checks_passed.append(("Environment", check_environment()))
    checks_passed.append(("Files", check_files()))
    checks_passed.append(("Imports", check_imports()))
    checks_passed.append(("Configuration", check_middleware_config()))
    checks_passed.append(("Token Generation", generate_test_token()))

    # Show endpoint lists
    show_protected_endpoints()
    show_public_endpoints()

    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print()

    all_passed = all(passed for _, passed in checks_passed)

    for check_name, passed in checks_passed:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {check_name}")

    print()

    if all_passed:
        print("ALL CHECKS PASSED")
        print()
        print("JWT authentication is properly configured!")
        print()
        print("Next steps:")
        print("1. Start the server: python src/dashboard/run_server_simple.py")
        print("2. Test public endpoint: curl http://localhost:8000/health")
        print("3. Test protected endpoint: curl http://localhost:8000/api/positions")
        print("   (Should return 401 Unauthorized)")
        print("4. Run test suite: pytest tests/test_jwt_auth_middleware.py -v")
        print()
        return 0
    else:
        print("SOME CHECKS FAILED")
        print()
        print("Please fix the issues above before proceeding.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
