import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration - Uses environment variables with fallbacks
DB_CONFIG = {
    'db_name': os.getenv('PG_DB_NAME', 'fbref'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'password'),
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': os.getenv('PG_PORT', '5432')
}

# Alternatively, use the URI directly if you've set it in .env
DB_URI = os.getenv('PG_URI', f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['db_name']}")

# Scraper configuration
SCRAPER_CONFIG = {
    'base_url': 'https://fbref.com/en/comps/9/Premier-League-Stats',
    'matches_to_keep': 7,
    'sleep_time_range': (10, 20),  # Increased delay between requests (10-20 seconds)
    'data_dir': 'data',
    'max_cache_age': 48,  # Cache HTML files for 48 hours
    'max_retries': 3,     # Maximum retry attempts on rate limit
    'use_batch_mode': os.getenv('USE_BATCH_MODE', 'False').lower() == 'true',
    'batch_size': 3,      # Number of teams to scrape in each batch
    'batch_delay': 300    # Delay between batches in seconds
}

# Logging configuration
LOG_CONFIG = {
    'log_file': 'logs/fbref_scraper.log',
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'rotate_logs': True,
    'max_log_size_mb': 10,
    'backup_count': 5
}