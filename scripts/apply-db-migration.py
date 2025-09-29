#!/usr/bin/env python3
"""
Database Schema Migration Tool for llama.cpp Performance Testing

This script applies schema migrations to extend the existing llama-bench
SQLite database with baseline tracking, historical data, and regression alerting.
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger("apply-db-migration")


def apply_migration(db_path: str, migration_sql_path: str, dry_run: bool = False) -> bool:
    """
    Apply database schema migration.

    Args:
        db_path: Path to SQLite database
        migration_sql_path: Path to SQL migration script
        dry_run: If True, print migration without applying

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(migration_sql_path):
        logger.error(f"Migration script not found: {migration_sql_path}")
        return False

    with open(migration_sql_path, 'r') as f:
        migration_sql = f.read()

    if dry_run:
        logger.info("Dry run mode - migration would execute:")
        print(migration_sql)
        return True

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.executescript(migration_sql)
        conn.commit()

        logger.info(f"Migration applied successfully to {db_path}")

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Database tables: {', '.join(tables)}")

        conn.close()
        return True

    except sqlite3.Error as e:
        logger.error(f"Migration failed: {e}")
        return False


def check_migration_status(db_path: str) -> dict:
    """
    Check if migration has been applied to the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary with migration status information
    """
    if not os.path.exists(db_path):
        return {"exists": False, "migrated": False, "tables": []}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        migration_tables = [
            "performance_baselines",
            "performance_history",
            "regression_alerts",
            "memory_leak_logs"
        ]

        migrated = all(table in tables for table in migration_tables)

        conn.close()

        return {
            "exists": True,
            "migrated": migrated,
            "tables": tables,
            "migration_tables_present": [t for t in migration_tables if t in tables]
        }

    except sqlite3.Error as e:
        logger.error(f"Error checking database: {e}")
        return {"exists": True, "migrated": False, "error": str(e)}


def main():
    """Main entry point for migration tool."""
    parser = argparse.ArgumentParser(
        description="Apply database schema migrations for performance testing"
    )
    parser.add_argument(
        "--database",
        "-d",
        required=True,
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--migration",
        "-m",
        help="Path to migration SQL script (default: scripts/db-schema-migration.sql)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration without applying"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check migration status without applying"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not args.migration:
        script_dir = Path(__file__).parent
        args.migration = script_dir / "db-schema-migration.sql"

    if args.check:
        status = check_migration_status(args.database)
        print(f"Database exists: {status.get('exists', False)}")
        print(f"Migration applied: {status.get('migrated', False)}")
        if status.get('tables'):
            print(f"Tables present: {', '.join(status['tables'])}")
        if status.get('migration_tables_present'):
            print(f"Migration tables: {', '.join(status['migration_tables_present'])}")
        sys.exit(0 if status.get('migrated', False) else 1)

    success = apply_migration(args.database, str(args.migration), args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
