#!/usr/bin/env python3
"""
Database initialization script for football data pipeline.
Sets up PostgreSQL tables and indexes according to defined schema.

Usage:
  python init_database.py --create-db  # Create the database and tables
  python init_database.py --reset-db   # Drop and recreate all tables (warning: destructive)
"""

import os
import sys
import argparse
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_init")

def get_connection_string():
    """Get database connection string from environment variables"""
    pg_uri = os.getenv('PG_URI')
    if pg_uri:
        return pg_uri
    
    # Construct from individual parameters if URI not provided
    db_name = os.getenv('PG_DB_NAME', 'fbref')
    user = os.getenv('PG_USER', 'postgres')
    password = os.getenv('PG_PASSWORD', 'password')
    host = os.getenv('PG_HOST', 'localhost')
    port = os.getenv('PG_PORT', '5432')
    
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

def create_database(conn_string):
    """Create the database if it doesn't exist"""
    # Extract database name from connection string
    db_name = conn_string.split('/')[-1]
    
    # Connect to default postgres database
    base_conn = conn_string.rsplit('/', 1)[0] + '/postgres'
    engine = create_engine(base_conn)
    
    try:
        # Check if database exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            exists = result.scalar() is not None
            
            if not exists:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                conn.commit()
                logger.info(f"Created database '{db_name}'")
            else:
                logger.info(f"Database '{db_name}' already exists")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise
    finally:
        engine.dispose()

def setup_schema(conn_string, reset=False):
    """Set up or reset the database schema"""
    engine = create_engine(conn_string)
    
    try:
        with engine.connect() as conn:
            if reset:
                # Drop tables in reverse order to handle foreign key constraints
                logger.info("Dropping existing tables...")
                conn.execute(text("DROP TABLE IF EXISTS league_table CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS players CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS teams CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS recent_matches CASCADE"))
                conn.commit()
            
            # Create schema from SQL file
            schema_path = os.path.join(os.path.dirname(__file__), 'db_schema.sql')
            
            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found: {schema_path}")
                return False
            
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema SQL
            conn.execute(text(schema_sql))
            conn.commit()
            logger.info("Database schema successfully applied")
            
            # Verify tables were created
            tables = ['recent_matches', 'teams', 'players', 'league_table']
            missing = []
            
            for table in tables:
                result = conn.execute(text(f"SELECT to_regclass('public.{table}')"))
                if result.scalar() is None:
                    missing.append(table)
            
            if missing:
                logger.warning(f"Some tables were not created: {', '.join(missing)}")
                return False
                
            return True
    except Exception as e:
        logger.error(f"Error setting up schema: {e}")
        return False
    finally:
        engine.dispose()

def main():
    parser = argparse.ArgumentParser(description="Initialize database for football data pipeline")
    parser.add_argument('--create-db', action='store_true', help='Create the database if it does not exist')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database (drop and recreate tables)')
    
    args = parser.parse_args()
    
    # Get connection string
    conn_string = get_connection_string()
    
    if args.create_db:
        try:
            create_database(conn_string)
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return 1
    
    # Set up schema
    success = setup_schema(conn_string, reset=args.reset_db)
    
    if success:
        logger.info("Database initialization complete")
        return 0
    else:
        logger.error("Database initialization failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())