"""Configuration settings for FBref Toolkit.

This module uses environment variables with fallbacks to default values,
making it more flexible for different deployment environments.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Database settings - Use environment variables with fallbacks
POSTGRES_URI = os.environ.get(
    "FBREF_DB_URI", 
    "postgresql+psycopg2://postgres:password@localhost:5432/fbref"
)

# Scraping settings
REQUEST_DELAY = float(os.environ.get("FBREF_REQUEST_DELAY", "1.0"))
CACHE_DIR = os.environ.get("FBREF_CACHE_DIR", "~/.fbref_cache")
LOG_DIR = os.environ.get("FBREF_LOG_DIR", "~/.fbref_logs")

# Expand user paths
CACHE_DIR_PATH = Path(os.path.expanduser(CACHE_DIR))
LOG_DIR_PATH = Path(os.path.expanduser(LOG_DIR))

# Cache TTL in seconds (default: 1 day)
CACHE_TTL = int(os.environ.get("FBREF_CACHE_TTL", "86400"))

# Rate limiting
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("FBREF_MAX_RPM", "60"))
BACKOFF_FACTOR = float(os.environ.get("FBREF_BACKOFF_FACTOR", "1.5"))
MAX_RETRIES = int(os.environ.get("FBREF_MAX_RETRIES", "3"))

# League settings
LEAGUES = {
    "Premier League": {"id": "9", "country": "England"},
    "La Liga": {"id": "12", "country": "Spain"},
    "Bundesliga": {"id": "20", "country": "Germany"},
    "Serie A": {"id": "11", "country": "Italy"},
    "Ligue 1": {"id": "13", "country": "France"},
}

# API settings
API_HOST = os.environ.get("FBREF_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("FBREF_API_PORT", "8000"))

# User agent for requests
USER_AGENT = os.environ.get(
    "FBREF_USER_AGENT", 
    "Mozilla/5.0 (FBrefToolkit; +https://github.com/your/repo)"
)

def get_config() -> Dict[str, Any]:
    """Return all configuration as a dictionary."""
    return {
        "postgres_uri": POSTGRES_URI,
        "request_delay": REQUEST_DELAY,
        "cache_dir": CACHE_DIR,
        "log_dir": LOG_DIR,
        "cache_ttl": CACHE_TTL,
        "max_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
        "backoff_factor": BACKOFF_FACTOR,
        "max_retries": MAX_RETRIES,
        "leagues": LEAGUES,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "user_agent": USER_AGENT,
    }