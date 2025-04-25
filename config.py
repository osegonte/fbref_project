"""Configuration settings for FBref Toolkit."""

# Database settings
POSTGRES_URI = "postgresql+psycopg2://postgres:password@localhost:5432/fbref"

# Scraping settings
REQUEST_DELAY = 1.0
CACHE_DIR = "~/.fbref_cache"
LOG_DIR = "~/.fbref_logs"

# League settings
LEAGUES = {
    "Premier League": {"id": "9", "country": "England"},
    "La Liga": {"id": "12", "country": "Spain"},
    "Bundesliga": {"id": "20", "country": "Germany"},
    "Serie A": {"id": "11", "country": "Italy"},
    "Ligue 1": {"id": "13", "country": "France"},
}