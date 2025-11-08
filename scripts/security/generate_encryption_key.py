#!/usr/bin/env python3
"""
Generate Encryption Key for Plaid Token Encryption

This script generates a new Fernet encryption key for securing Plaid access tokens
in the database.

Usage:
    python scripts/security/generate_encryption_key.py

The generated key should be stored in your .env file as:
    DATABASE_ENCRYPTION_KEY=<generated_key>

Security Notes:
- NEVER commit the .env file to version control
- Store keys in secure secret management systems in production
- Rotate keys periodically
- Keep backup of old keys during rotation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.security.token_encryption import TokenEncryption


def main():
    """Generate and display new encryption key."""
    print("=" * 70)
    print("Plaid Token Encryption Key Generator")
    print("=" * 70)
    print()

    # Generate new key
    print("Generating new Fernet encryption key...")
    key = TokenEncryption.generate_key()

    print("\n✓ Key generated successfully!")
    print("\n" + "=" * 70)
    print("Add this to your .env file:")
    print("=" * 70)
    print()
    print(f"DATABASE_ENCRYPTION_KEY={key}")
    print()
    print("=" * 70)

    print("\n⚠️  SECURITY WARNINGS:")
    print("   1. NEVER commit this key to version control")
    print("   2. Store in .env file (already in .gitignore)")
    print("   3. Use secret management in production (AWS Secrets Manager, etc.)")
    print("   4. Keep backup of old keys during rotation")
    print("   5. Rotate keys periodically (quarterly recommended)")

    print("\n📝 Next Steps:")
    print("   1. Copy the key above to your .env file")
    print("   2. If you have existing plaintext tokens, run:")
    print("      python scripts/security/migrate_encrypt_tokens.py")
    print("   3. Verify encryption with:")
    print("      python -c 'from src.security import TokenEncryption; print(TokenEncryption().generate_key())'")

    print()

if __name__ == "__main__":
    main()
