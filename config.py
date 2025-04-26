# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# PostgreSQL connection parameters (unused by the CSV-only scraper, but retained)
DB_CONFIG = {
    'db_name': os.getenv('PG_DB_NAME', 'fbref'),
    'user':    os.getenv('PG_USER', 'postgres'),
    'password':os.getenv('PG_PASSWORD', 'password'),
    'host':    os.getenv('PG_HOST', 'localhost'),
    'port':    os.getenv('PG_PORT', '5432'),
}

# Defaults for the scraper entry-point (used when running with no flags)
SCRAPER_CONFIG = {
    # Base FBref URL for default league (Premier League)
    'base_url': os.getenv(
        'SCRAPER_BASE_URL',
        'https://fbref.com/en/comps/9/Premier-League-Stats'
    ),
    # How many past matches to fetch per team
    'matches_to_keep': int(os.getenv('SCRAPER_LOOKBACK', '7')),
}

# Logging configuration (if you ever re-enable file‐ or DB‐logging)
LOG_CONFIG = {
    'log_file':      os.getenv('LOG_FILE', 'logs/universal_scraper.log'),
    'log_level':     os.getenv('LOG_LEVEL', 'INFO'),
    'log_format':    '%(asctime)s - %(levelname)s - %(message)s',
    'rotate_logs':   True,
    'max_log_size_mb': 10,
    'backup_count':  5,
}
