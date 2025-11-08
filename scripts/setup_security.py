#!/usr/bin/env python3
"""
Security setup script for trader-ai application.

This script:
1. Generates secure encryption and JWT keys
2. Sets proper file permissions on sensitive files
3. Validates security configuration
4. Creates a secure .env file from template
"""

import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet
import secrets
import stat

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_encryption_key() -> str:
    """Generate a secure Fernet encryption key."""
    return Fernet.generate_key().decode()


def generate_jwt_secret() -> str:
    """Generate a secure JWT secret key."""
    return secrets.token_urlsafe(32)


def set_file_permissions(file_path: Path, mode: int = 0o600) -> None:
    """
    Set restrictive file permissions (owner read/write only).

    Args:
        file_path: Path to file
        mode: Permission mode (default: 0o600 = owner read/write only)
    """
    if file_path.exists():
        try:
            os.chmod(file_path, mode)
            print(f"✓ Set permissions {oct(mode)} on {file_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not set permissions on {file_path}: {e}")
            if os.name == 'nt':
                print("  Note: Windows uses different permission model than Unix")
    else:
        print(f"✗ File not found: {file_path}")


def create_secure_env_file() -> None:
    """Create .env file from .env.example with generated keys."""
    env_example = project_root / '.env.example'
    env_file = project_root / '.env'

    if env_file.exists():
        response = input(f".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping .env creation")
            return

    if not env_example.exists():
        print(f"✗ .env.example not found at {env_example}")
        return

    # Read template
    with open(env_example, 'r') as f:
        content = f.read()

    # Generate secure keys
    encryption_key = generate_encryption_key()
    jwt_secret = generate_jwt_secret()

    # Replace placeholders
    content = content.replace('REPLACE_WITH_GENERATED_KEY', encryption_key)
    content = content.replace('REPLACE_WITH_GENERATED_KEY', jwt_secret)

    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)

    print(f"✓ Created .env file at {env_file}")

    # Set restrictive permissions
    set_file_permissions(env_file, 0o600)


def secure_database_files() -> None:
    """Set secure permissions on database files."""
    data_dir = project_root / 'data'

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created data directory at {data_dir}")

    # Secure all .db files
    for db_file in data_dir.glob('*.db'):
        set_file_permissions(db_file, 0o600)

    # Secure the data directory itself
    try:
        os.chmod(data_dir, 0o700)
        print(f"✓ Set permissions 0o700 on {data_dir}")
    except Exception as e:
        print(f"⚠ Warning: Could not set permissions on {data_dir}: {e}")


def validate_security_config() -> bool:
    """
    Validate security configuration.

    Returns:
        True if all checks pass, False otherwise
    """
    print("\n=== Security Configuration Validation ===\n")

    all_checks_passed = True

    # Check .env file exists
    env_file = project_root / '.env'
    if env_file.exists():
        print("✓ .env file exists")

        # Check for placeholder values
        with open(env_file, 'r') as f:
            content = f.read()
            if 'REPLACE_WITH_GENERATED_KEY' in content:
                print("✗ .env file contains placeholder values - run setup again")
                all_checks_passed = False
            else:
                print("✓ .env file has no placeholder values")
    else:
        print("✗ .env file not found")
        all_checks_passed = False

    # Check config.json
    config_file = project_root / 'config' / 'config.json'
    if config_file.exists():
        print("✓ config.json exists")

        import json
        with open(config_file, 'r') as f:
            config = json.load(f)

            # Check environment setting
            if config.get('environment') in ['development', 'production']:
                print(f"✓ Environment set to: {config.get('environment')}")
            else:
                print("⚠ Environment not properly configured")
                all_checks_passed = False

            # Check HTTPS enforcement in production
            if config.get('environment') == 'production' and not config.get('enforce_https'):
                print("⚠ Warning: HTTPS not enforced in production mode")
                all_checks_passed = False

            # Check rate limiting
            if config.get('rate_limit_per_minute'):
                print(f"✓ Rate limiting configured: {config.get('rate_limit_per_minute')}/min")
            else:
                print("⚠ Rate limiting not configured")
    else:
        print("✗ config.json not found")
        all_checks_passed = False

    # Check database permissions (Unix only)
    if os.name != 'nt':
        data_dir = project_root / 'data'
        for db_file in data_dir.glob('*.db'):
            file_stat = os.stat(db_file)
            mode = stat.S_IMODE(file_stat.st_mode)
            if mode == 0o600:
                print(f"✓ {db_file.name} has secure permissions (600)")
            else:
                print(f"⚠ {db_file.name} has permissions {oct(mode)} (should be 600)")
                all_checks_passed = False

    return all_checks_passed


def main():
    """Main setup script."""
    print("=== Trader-AI Security Setup ===\n")

    print("Step 1: Creating secure .env file...")
    create_secure_env_file()

    print("\nStep 2: Securing database files...")
    secure_database_files()

    print("\nStep 3: Validating security configuration...")
    if validate_security_config():
        print("\n✓ All security checks passed!")
    else:
        print("\n⚠ Some security checks failed - please review warnings above")
        sys.exit(1)

    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Edit .env file and add your Plaid credentials")
    print("2. Review config/config.json security settings")
    print("3. For production: Set ENVIRONMENT=production and ENFORCE_HTTPS=true")
    print("4. Run: python main.py --test")


if __name__ == "__main__":
    main()
