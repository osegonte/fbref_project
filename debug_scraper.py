# debug_scraper.py
import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("1. Script started")

# Try to import required packages
try:
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    print("2. All required packages imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import required packages: {e}")
    sys.exit(1)

# Check if logs directory exists and is writable
try:
    os.makedirs("logs", exist_ok=True)
    with open("logs/test.log", "w") as f:
        f.write("Test log entry")
    print("3. Logs directory is writable")
except Exception as e:
    print(f"ERROR: Cannot write to logs directory: {e}")

# Check database connection
try:
    db_uri = os.getenv('PG_URI')
    if db_uri:
        print(f"4. Found database URI: {db_uri[:10]}...")
        # Extract connection parameters from URI
        uri_parts = db_uri.replace('postgresql+psycopg2://', '').split('@')
        user_pass = uri_parts[0].split(':')
        host_port_db = uri_parts[1].split('/')
        host_port = host_port_db[0].split(':')
        
        connection_params = {
            "dbname": host_port_db[1],
            "user": user_pass[0],
            "password": user_pass[1],
            "host": host_port[0],
            "port": host_port[1] if len(host_port) > 1 else "5432"
        }
    else:
        connection_params = {
            "dbname": os.getenv('PG_DB_NAME', 'fbref'),
            "user": os.getenv('PG_USER', 'postgres'),
            "password": os.getenv('PG_PASSWORD', 'password'),
            "host": os.getenv('PG_HOST', 'localhost'),
            "port": os.getenv('PG_PORT', '5432')
        }
    
    print(f"5. Connecting to database {connection_params['dbname']} on {connection_params['host']}")
    conn = psycopg2.connect(**connection_params)
    print("6. Database connection successful")
    
    # Try to create test table
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS debug_test (
        id SERIAL PRIMARY KEY,
        test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    print("7. Test table created successfully")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"ERROR: Database connection failed: {e}")

# Test FBref connection
try:
    print("8. Testing connection to FBref...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get("https://fbref.com/en/comps/9/Premier-League-Stats", headers=headers, timeout=10)
    print(f"9. FBref connection status: {response.status_code}")
    if response.status_code == 200:
        print("10. Successfully connected to FBref")
    else:
        print(f"WARNING: FBref returned non-200 status code: {response.status_code}")
except Exception as e:
    print(f"ERROR: FBref connection failed: {e}")

print("Debug script completed")