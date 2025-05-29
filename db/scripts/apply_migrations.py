#!/usr/bin/env python3
"""
Apply database migrations that weren't run during docker-compose initialization.

This script can be run manually or as part of your deployment process to ensure
all migrations are applied, including to existing databases.
"""

import os
import sys
import time
import psycopg2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vector_rag.config import Config


def wait_for_db(config: Config, max_attempts: int = 30):
    """Wait for database to be ready."""
    print("Waiting for database to be ready...")
    
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(config.DB_URL)
            conn.close()
            print("Database is ready!")
            return True
        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                time.sleep(1)
            else:
                print("Failed to connect to database after 30 seconds")
                return False
    return False


def apply_migration(config: Config, migration_file: Path):
    """Apply a single migration file."""
    print(f"\nApplying migration: {migration_file.name}")
    
    try:
        conn = psycopg2.connect(config.DB_URL)
        cur = conn.cursor()
        
        # Read and execute the migration
        with open(migration_file, 'r') as f:
            sql = f.read()
            cur.execute(sql)
        
        conn.commit()
        print(f"✓ Successfully applied {migration_file.name}")
        
    except psycopg2.Error as e:
        print(f"✗ Error applying {migration_file.name}: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    """Apply all pending migrations."""
    config = Config()
    
    # Wait for database to be ready
    if not wait_for_db(config):
        sys.exit(1)
    
    # Get all SQL files in the migrations directory
    migrations_dir = Path(__file__).parent.parent / "sql"
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    print(f"\nFound {len(migration_files)} migration files")
    
    # Apply each migration
    for migration_file in migration_files:
        # Skip init.sql as it's handled by docker-compose
        if migration_file.name == "init.sql":
            print(f"\nSkipping {migration_file.name} (handled by docker-compose)")
            continue
            
        apply_migration(config, migration_file)
    
    print("\n✓ All migrations completed successfully!")


if __name__ == "__main__":
    main()