#!/usr/bin/env python3
"""
Migrate Plaintext Plaid Tokens to Encrypted Format

This script encrypts all existing plaintext Plaid access tokens in the database.
It should be run ONCE after setting up DATABASE_ENCRYPTION_KEY for the first time.

Safety Features:
- Backs up database before migration
- Validates encryption key before starting
- Tests roundtrip encryption before applying to all tokens
- Provides rollback capability
- Logs all operations

Usage:
    # Dry run (preview changes without modifying)
    python scripts/security/migrate_encrypt_tokens.py --dry-run

    # Actual migration
    python scripts/security/migrate_encrypt_tokens.py

    # Rollback to backup (if needed)
    python scripts/security/migrate_encrypt_tokens.py --rollback

Requirements:
    - DATABASE_ENCRYPTION_KEY must be set in environment
    - Database must be in src/finances/bank_database.py location
"""

import sys
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.security.token_encryption import TokenEncryption, TokenEncryptionError
from src.finances.bank_database import BankDatabase


def backup_database(db_path: str) -> str:
    """
    Create backup of database before migration.

    Args:
        db_path: Path to database file

    Returns:
        Path to backup file
    """
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"

    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"✓ Backup created successfully")

    return backup_path


def get_plaintext_tokens(db: BankDatabase) -> list:
    """
    Get all plaid_items with potentially plaintext tokens.

    Args:
        db: BankDatabase instance

    Returns:
        List of (item_id, access_token, institution_name) tuples
    """
    conn = db._get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT item_id, access_token, institution_name
        FROM plaid_items
    """)

    items = cursor.fetchall()
    conn.close()

    return [(item['item_id'], item['access_token'], item['institution_name']) for item in items]


def is_token_encrypted(token: str) -> bool:
    """Check if token appears to be encrypted."""
    return TokenEncryption.is_encrypted(token)


def migrate_tokens(db_path: str, dry_run: bool = False) -> dict:
    """
    Migrate all plaintext tokens to encrypted format.

    Args:
        db_path: Path to database file
        dry_run: If True, preview changes without modifying database

    Returns:
        Dictionary with migration statistics
    """
    print("\n" + "=" * 70)
    print("Plaid Token Encryption Migration")
    print("=" * 70)

    # Initialize encryption
    try:
        encryptor = TokenEncryption()
        print("✓ Encryption initialized successfully")
    except TokenEncryptionError as e:
        print(f"✗ Encryption initialization failed: {e}")
        return {"success": False, "error": str(e)}

    # Test encryption roundtrip
    test_token = "access-sandbox-test123"
    if not encryptor.verify_roundtrip(test_token):
        print("✗ Encryption roundtrip test failed")
        return {"success": False, "error": "Roundtrip test failed"}
    print("✓ Encryption roundtrip test passed")

    # Load database
    db = BankDatabase(db_path)

    # Get all items
    items = get_plaintext_tokens(db)
    print(f"\nFound {len(items)} Plaid items in database")

    # Analyze tokens
    plaintext_count = 0
    encrypted_count = 0
    tokens_to_encrypt = []

    for item_id, token, institution in items:
        if is_token_encrypted(token):
            encrypted_count += 1
            print(f"  {item_id} ({institution}): Already encrypted ✓")
        else:
            plaintext_count += 1
            tokens_to_encrypt.append((item_id, token, institution))
            print(f"  {item_id} ({institution}): Plaintext → needs encryption")

    print(f"\nSummary:")
    print(f"  Already encrypted: {encrypted_count}")
    print(f"  Need encryption:   {plaintext_count}")

    if plaintext_count == 0:
        print("\n✓ All tokens already encrypted. Nothing to do.")
        return {"success": True, "encrypted": 0, "skipped": encrypted_count}

    if dry_run:
        print("\n[DRY RUN] Would encrypt the following tokens:")
        for item_id, token, institution in tokens_to_encrypt:
            encrypted_preview = encryptor.encrypt_token(token)[:50] + "..."
            print(f"  {item_id}: {token[:20]}... → {encrypted_preview}")
        print("\nRun without --dry-run to perform actual encryption")
        return {"success": True, "dry_run": True, "would_encrypt": plaintext_count}

    # Create backup
    print()
    backup_path = backup_database(db_path)

    # Encrypt tokens
    print(f"\nEncrypting {plaintext_count} tokens...")
    conn = db._get_connection()
    cursor = conn.cursor()

    encrypted = 0
    failed = 0

    try:
        for item_id, token, institution in tokens_to_encrypt:
            try:
                encrypted_token = encryptor.encrypt_token(token)

                cursor.execute("""
                    UPDATE plaid_items
                    SET access_token = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE item_id = ?
                """, (encrypted_token, item_id))

                encrypted += 1
                print(f"  ✓ Encrypted {item_id} ({institution})")

            except Exception as e:
                failed += 1
                print(f"  ✗ Failed to encrypt {item_id}: {e}")

        conn.commit()
        print(f"\n✓ Migration completed successfully")
        print(f"  Encrypted: {encrypted}")
        print(f"  Failed: {failed}")
        print(f"  Backup: {backup_path}")

    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        print(f"Database not modified. Backup available at: {backup_path}")
        return {"success": False, "error": str(e), "backup": backup_path}

    finally:
        conn.close()

    return {
        "success": True,
        "encrypted": encrypted,
        "failed": failed,
        "skipped": encrypted_count,
        "backup": backup_path
    }


def rollback_migration(db_path: str, backup_path: str = None):
    """
    Rollback to backup database.

    Args:
        db_path: Path to current database
        backup_path: Path to backup (or will find most recent)
    """
    if not backup_path:
        # Find most recent backup
        backup_pattern = f"{db_path}.backup_*"
        backups = sorted(Path(db_path).parent.glob(Path(backup_pattern).name), reverse=True)

        if not backups:
            print("✗ No backup files found")
            return False

        backup_path = str(backups[0])

    print(f"Rolling back to backup: {backup_path}")
    shutil.copy2(backup_path, db_path)
    print("✓ Rollback completed successfully")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate plaintext Plaid tokens to encrypted format"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying database'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback to most recent backup'
    )
    parser.add_argument(
        '--db-path',
        default='data/bank_accounts.db',
        help='Path to database file (default: data/bank_accounts.db)'
    )

    args = parser.parse_args()

    db_path = args.db_path

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"✗ Database not found: {db_path}")
        sys.exit(1)

    if args.rollback:
        success = rollback_migration(db_path)
        sys.exit(0 if success else 1)

    # Run migration
    result = migrate_tokens(db_path, dry_run=args.dry_run)

    if result.get("success"):
        print("\n" + "=" * 70)
        print("Migration completed successfully!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("Migration failed!")
        print("=" * 70)
        if "error" in result:
            print(f"Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
