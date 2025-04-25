import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random
import psycopg2
from psycopg2.extras import execute_values
import io
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fbref_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_scraper")

# Rotate user agents to appear more like different browsers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
]

class FBrefScraper:
    """Class to scrape football data from FBref.com"""
    
    def __init__(self, base_url: str = "https://fbref.com/en/comps/9/Premier-League-Stats", 
                 years: List[int] = None, 
                 min_delay: int = 5, 
                 max_delay: int = 10,
                 cache_dir: str = "data/cache",
                 max_cache_age: int = 24):
        """
        Initialize the scraper with base URL and years to scrape
        
        Args:
            base_url: Starting URL for the Premier League stats
            years: List of years to scrape (e.g. [2024, 2023, 2022])
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            cache_dir: Directory to store cached HTML files
            max_cache_age: Maximum age of cached files in hours
        """
        self.base_url = base_url
        self.years = years or [datetime.now().year]
        self.all_matches = []
        self.team_urls = []
        self.data_dir = "data"
        self.cache_dir = cache_dir
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_cache_age = max_cache_age
        
        # Create required directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_random_headers(self):
        """Get random user agent headers to avoid detection"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    def get_cached_html(self, url: str) -> Optional[str]:
        """
        Get HTML from cache if available and not expired
        
        Args:
            url: URL to check cache for
            
        Returns:
            Cached HTML content or None if not cached or expired
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        if os.path.exists(cache_file):
            file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            
            if file_age_hours < self.max_cache_age:
                logger.debug(f"Using cached version of {url}")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read cache file for {url}: {e}")
        
        return None
    
    def save_to_cache(self, url: str, html_content: str) -> bool:
        """
        Save HTML content to cache
        
        Args:
            url: URL associated with the content
            html_content: HTML content to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            logger.warning(f"Failed to cache content for {url}: {e}")
            return False

    def get_html(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Download HTML from URL with error handling, caching and exponential backoff
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content as string or None if request failed
        """
        # Check cache first
        cached_html = self.get_cached_html(url)
        if cached_html:
            return cached_html
        
        # Not cached or expired, fetch it
        retry_count = 0
        base_delay = self.min_delay
        
        while retry_count <= max_retries:
            try:
                # Random delay between requests
                current_delay = base_delay + random.random() * (self.max_delay - self.min_delay)
                logger.debug(f"Waiting {current_delay:.2f} seconds before fetching {url}")
                time.sleep(current_delay)
                
                # Get with random headers
                headers = self.get_random_headers()
                response = requests.get(url, headers=headers, timeout=15)
                
                # If rate limited, back off and retry
                if response.status_code == 429:
                    retry_count += 1
                    base_delay *= 2  # Exponential backoff
                    
                    if retry_count <= max_retries:
                        logger.warning(f"Rate limited, retrying in {base_delay:.2f} seconds (attempt {retry_count}/{max_retries})")
                        time.sleep(base_delay)
                        continue
                    else:
                        logger.error(f"Maximum retries reached for {url}")
                        return None
                
                # For other errors, fail fast
                response.raise_for_status()
                
                # Cache successful response
                self.save_to_cache(url, response.text)
                return response.text
                
            except requests.RequestException as e:
                logger.error(f"Failed to fetch {url}: {e}")
                retry_count += 1
                
                if retry_count <= max_retries and isinstance(e, requests.exceptions.ConnectionError):
                    logger.info(f"Connection error, retrying in {base_delay} seconds ({retry_count}/{max_retries})")
                    time.sleep(base_delay)
                    base_delay *= 2
                else:
                    return None
        
        return None

    def extract_team_urls(self, html: str) -> List[str]:
        """
        Extract team URLs from the standings page
        
        Args:
            html: HTML content of the standings page
            
        Returns:
            List of team URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        try:
            standings_table = soup.select('table.stats_table')[0]
            links = [l.get("href") for l in standings_table.find_all('a')]
            links = [l for l in links if l and '/squads/' in l]
            return [f"https://fbref.com{l}" for l in links]
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to extract team URLs: {e}")
            return []

    def get_previous_season_url(self, html: str) -> Optional[str]:
        """
        Get URL for the previous season
        
        Args:
            html: HTML content of the current season page
            
        Returns:
            URL for the previous season or None if not found
        """
        soup = BeautifulSoup(html, 'html.parser')
        try:
            prev_link = soup.select("a.prev")[0]
            return f"https://fbref.com{prev_link.get('href')}"
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to extract previous season URL: {e}")
            return None

    def extract_team_name(self, team_url: str) -> str:
        """
        Extract clean team name from team URL
        
        Args:
            team_url: URL of the team page
            
        Returns:
            Clean team name
        """
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        return team_name

    def get_shooting_link(self, html: str) -> Optional[str]:
        """
        Extract shooting stats link from team page
        
        Args:
            html: HTML content of the team page
            
        Returns:
            URL for the shooting stats or None if not found
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        
        if not links:
            logger.warning("No shooting stats link found")
            return None
            
        return f"https://fbref.com{links[0]}"

    def parse_matches_and_shooting(self, team_url: str, year: int) -> Optional[pd.DataFrame]:
        """
        Parse matches and shooting data for a team
        
        Args:
            team_url: URL of the team page
            year: Season year
            
        Returns:
            DataFrame with combined match and shooting data or None if failed
        """
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {team_name} for {year} season")
        
        # Get team page HTML
        team_html = self.get_html(team_url)
        if not team_html:
            logger.error(f"Failed to get team page HTML for {team_name}")
            return None
            
        # Parse matches
        try:
            # Using StringIO to handle the warning about literal HTML
            html_io = io.StringIO(team_html)
            matches = pd.read_html(html_io, match="Scores & Fixtures")[0]
        except Exception as e:
            logger.error(f"Failed to parse matches for {team_name}: {e}")
            return None
            
        # Get shooting stats
        shooting_link = self.get_shooting_link(team_html)
        if not shooting_link:
            logger.warning(f"No shooting link found for {team_name}")
            return None
            
        shooting_html = self.get_html(shooting_link)
        if not shooting_html:
            logger.error(f"Failed to get shooting stats HTML for {team_name}")
            return None
            
        # Parse shooting stats
        try:
            # Using StringIO to handle the warning about literal HTML
            html_io = io.StringIO(shooting_html)
            shooting = pd.read_html(html_io, match="Shooting")[0]
            # Fix multi-level column headers
            if isinstance(shooting.columns, pd.MultiIndex):
                shooting.columns = shooting.columns.droplevel(0)
        except Exception as e:
            logger.error(f"Failed to parse shooting stats for {team_name}: {e}")
            return None
            
        # Merge data
        try:
            # Get essential shooting columns
            shooting_cols = ["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
            available_cols = [col for col in shooting_cols if col in shooting.columns]
            
            team_data = matches.merge(shooting[available_cols], on="Date", how="inner")
            
            # Filter for Premier League matches only
            team_data = team_data[team_data["Comp"] == "Premier League"]
            
            # Add team and season info
            team_data["Season"] = year
            team_data["Team"] = team_name
            
            return team_data
            
        except Exception as e:
            logger.error(f"Failed to merge data for {team_name}: {e}")
            return None

    def scrape_season(self, url: str, year: int) -> Tuple[List[pd.DataFrame], Optional[str]]:
        """
        Scrape all teams for a specific season
        
        Args:
            url: URL of the season standings page
            year: Season year
            
        Returns:
            Tuple of (list of team DataFrames, URL for previous season)
        """
        logger.info(f"Scraping season {year} from {url}")
        
        # Get standings page HTML
        html = self.get_html(url)
        if not html:
            logger.error(f"Failed to get season page HTML for {year}")
            return [], None
        
        # Extract team URLs
        team_urls = self.extract_team_urls(html)
        logger.info(f"Found {len(team_urls)} teams")
        
        # Get previous season URL
        previous_season_url = self.get_previous_season_url(html)
        
        # Process each team
        season_dfs = []
        for team_url in team_urls:
            team_df = self.parse_matches_and_shooting(team_url, year)
            if team_df is not None and not team_df.empty:
                season_dfs.append(team_df)
            else:
                logger.warning(f"No valid data found for team at {team_url}")
        
        return season_dfs, previous_season_url

    def scrape(self) -> pd.DataFrame:
        """
        Main method to scrape data for all configured years
        
        Returns:
            Combined DataFrame with all matches
        """
        standings_url = self.base_url
        season_dfs = []
        
        for year in self.years:
            year_dfs, next_url = self.scrape_season(standings_url, year)
            season_dfs.extend(year_dfs)
            
            if next_url:
                standings_url = next_url
            else:
                logger.warning(f"Could not find previous season URL after {year}")
                break
        
        if not season_dfs:
            logger.error("No data collected")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(season_dfs, ignore_index=True)
        
        # Standardize column names to lowercase
        combined_df.columns = [c.lower() for c in combined_df.columns]
        
        return combined_df
    
    def get_recent_matches(self, limit=7) -> pd.DataFrame:
        """
        Extract only the most recent matches from the scraped data
        
        Args:
            limit: Number of recent matches to retrieve per team
        
        Returns:
            DataFrame with only recent matches
        """
        # Get all matches
        all_matches = self.scrape()
        
        if all_matches.empty:
            logger.warning("No matches found to extract recent matches from")
            return pd.DataFrame()
        
        # Convert date to datetime for sorting
        all_matches['date'] = pd.to_datetime(all_matches['date'])
        
        # Create empty dataframe for recent matches
        recent_matches = pd.DataFrame()
        
        # For each team, get their most recent matches
        for team in all_matches['team'].unique():
            team_matches = all_matches[all_matches['team'] == team]
            team_recent = team_matches.sort_values('date', ascending=False).head(limit)
            recent_matches = pd.concat([recent_matches, team_recent])
        
        # Add scrape date
        recent_matches['scrape_date'] = datetime.now().strftime("%Y-%m-%d")
        
        # Generate match_id (unique identifier)
        recent_matches['match_id'] = recent_matches.apply(
            lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{row['team']}_{row['opponent']}",
            axis=1
        )
        
        # Add is_home field
        recent_matches['is_home'] = recent_matches['venue'].str.lower() == 'home'
        
        # Add points based on result
        result_to_points = {'W': 3, 'D': 1, 'L': 0}
        recent_matches['points'] = recent_matches['result'].map(result_to_points)
        
        return recent_matches

    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the scraped data to CSV
        
        Args:
            df: DataFrame to save
            filename: Optional filename, will use timestamp if not provided
            
        Returns:
            Path to the saved file
        """
        if df.empty:
            logger.error("Cannot save empty DataFrame")
            return ""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fbref_matches_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        
        return filepath


class FBrefDatabaseManager:
    """Class to manage PostgreSQL operations for FBref data"""
    
    def __init__(self, db_name="fbref", user="postgres", 
                 password="password", host="localhost", port="5432",
                 connection_uri=None):
        """Initialize the database manager"""
        if connection_uri:
            self.connection_uri = connection_uri
            # Extract components for psycopg2 connection
            # Format: postgresql+psycopg2://user:password@host:port/dbname
            uri_parts = connection_uri.replace('postgresql+psycopg2://', '').split('@')
            user_pass = uri_parts[0].split(':')
            host_port_db = uri_parts[1].split('/')
            host_port = host_port_db[0].split(':')
            
            self.connection_params = {
                "dbname": host_port_db[1],
                "user": user_pass[0],
                "password": user_pass[1],
                "host": host_port[0],
                "port": host_port[1] if len(host_port) > 1 else "5432"
            }
        else:
            self.connection_params = {
                "dbname": db_name,
                "user": user,
                "password": password,
                "host": host,
                "port": port
            }
            self.connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        
    def get_connection(self):
        """Get a database connection with context manager"""
        return psycopg2.connect(**self.connection_params)
        
    def initialize_db(self):
        """Create database tables if they don't exist"""
        # SQL to create the recent_matches table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS recent_matches (
            match_id VARCHAR(100) PRIMARY KEY,
            date DATE NOT NULL,
            team VARCHAR(50) NOT NULL,
            opponent VARCHAR(50) NOT NULL,
            venue VARCHAR(20),
            result CHAR(1),
            gf INTEGER,
            ga INTEGER,
            points INTEGER,
            sh INTEGER,
            sot INTEGER,
            dist FLOAT,
            fk FLOAT,
            pk INTEGER,
            pkatt INTEGER,
            possession FLOAT,
            corners_for INTEGER,
            corners_against INTEGER,
            xg FLOAT,
            xga FLOAT,
            comp VARCHAR(50),
            round VARCHAR(50),
            season INTEGER,
            is_home BOOLEAN,
            scrape_date TIMESTAMP
        );
        
        -- Create indexes for faster queries
        CREATE INDEX IF NOT EXISTS idx_recent_matches_date ON recent_matches(date);
        CREATE INDEX IF NOT EXISTS idx_recent_matches_team ON recent_matches(team);
        CREATE INDEX IF NOT EXISTS idx_recent_matches_season ON recent_matches(season);
        
        -- Teams Table for team information
        CREATE TABLE IF NOT EXISTS teams (
            team_id SERIAL PRIMARY KEY,
            team_name VARCHAR(50) UNIQUE NOT NULL,
            country VARCHAR(50),
            league VARCHAR(50),
            logo_url VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Players Table for player information
        CREATE TABLE IF NOT EXISTS players (
            player_id SERIAL PRIMARY KEY,
            player_name VARCHAR(100) NOT NULL,
            team_id INTEGER REFERENCES teams(team_id),
            position VARCHAR(20),
            nationality VARCHAR(50),
            birth_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (player_name, team_id)
        );
        
        -- League Table for storing standings
        CREATE TABLE IF NOT EXISTS league_table (
            id SERIAL PRIMARY KEY,
            team_id INTEGER REFERENCES teams(team_id),
            season INTEGER NOT NULL,
            rank INTEGER,
            matches_played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_diff INTEGER,
            points INTEGER,
            points_per_match FLOAT,
            xg FLOAT,
            xga FLOAT,
            xg_diff FLOAT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (team_id, season)
        );
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Create table and indexes
                cursor.execute(create_table_sql)
                
                # Commit changes happens automatically with context manager
                logger.info("Database initialized successfully")
    
    def store_recent_matches(self, df):
        """Store recent matches with upsert logic"""
        if df.empty:
            logger.warning("No matches to store")
            return 0
        
        # Prepare the upsert query
        upsert_query = """
        INSERT INTO recent_matches (
            match_id, date, team, opponent, venue, result, 
            gf, ga, points, sh, sot, dist, fk, pk, pkatt,
            comp, round, season, is_home, scrape_date
        ) VALUES %s
        ON CONFLICT (match_id) 
        DO UPDATE SET
            result = EXCLUDED.result,
            gf = EXCLUDED.gf,
            ga = EXCLUDED.ga,
            points = EXCLUDED.points,
            sh = EXCLUDED.sh,
            sot = EXCLUDED.sot,
            dist = EXCLUDED.dist,
            fk = EXCLUDED.fk,
            pk = EXCLUDED.pk,
            pkatt = EXCLUDED.pkatt,
            scrape_date = EXCLUDED.scrape_date;
        """
        
        # Prepare data for insertion
        columns = ['match_id', 'date', 'team', 'opponent', 'venue', 'result', 
                  'gf', 'ga', 'points', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt',
                  'comp', 'round', 'season', 'is_home', 'scrape_date']
        
        # Ensure all required columns exist
        for col in columns:
            if col not in df.columns and col not in ['is_home', 'scrape_date']:
                logger.error(f"Required column '{col}' missing from DataFrame")
                return 0
        
        # Convert values to list of tuples for execute_values
        values = []
        for _, row in df.iterrows():
            value = []
            for col in columns:
                if col in df.columns:
                    value.append(row[col])
                elif col == 'is_home':
                    value.append(row['venue'].lower() == 'home')
                elif col == 'scrape_date':
                    value.append(datetime.now())
                else:
                    value.append(None)
            values.append(tuple(value))
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Execute upsert
                execute_values(cursor, upsert_query, values)
                # Commit happens automatically with context manager
                
                logger.info(f"Stored/updated {len(df)} matches in the database")
                return len(df)
    
    def get_recent_team_matches(self, team, limit=7):
        """Get recent matches for a specific team"""
        query = """
        SELECT * FROM recent_matches
        WHERE team = %s
        ORDER BY date DESC
        LIMIT %s
        """
        
        with self.get_connection() as conn:
            # Using SQLAlchemy's connection for pandas
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_uri)
            df = pd.read_sql_query(query, engine, params=(team, limit))
            
            logger.info(f"Retrieved {len(df)} recent matches for {team}")
            return df
    
    def get_all_recent_matches(self, days=30):
        """Get all matches in the last N days"""
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = """
        SELECT * FROM recent_matches
        WHERE date >= %s
        ORDER BY date DESC
        """
        
        with self.get_connection() as conn:
            # Using SQLAlchemy's connection for pandas
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_uri)
            df = pd.read_sql_query(query, engine, params=(cutoff_date,))
            
            logger.info(f"Retrieved {len(df)} matches from the last {days} days")
            return df
    
    def execute_query(self, query, params=None):
        """Execute a custom SQL query"""
        try:
            # Using SQLAlchemy's connection for pandas
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_uri)
            
            # Query data and load into DataFrame
            if params:
                df = pd.read_sql_query(query, engine, params=params)
            else:
                df = pd.read_sql_query(query, engine)
            
            return df
            
        except Exception as error:
            logger.error(f"Error executing query: {error}")
            return pd.DataFrame()


def main():
    """Main function to run the scraper and store data in PostgreSQL"""
    # Import config if it exists, otherwise use environment variables
    try:
        from config import DB_CONFIG, SCRAPER_CONFIG, DB_URI
        # Use values from config
        db_uri = DB_URI
        db_name = DB_CONFIG['db_name']
        db_user = DB_CONFIG['user']
        db_password = DB_CONFIG['password']
        db_host = DB_CONFIG['host']
        db_port = DB_CONFIG['port']
        base_url = SCRAPER_CONFIG['base_url']
        matches_to_keep = SCRAPER_CONFIG['matches_to_keep']
        min_delay = SCRAPER_CONFIG.get('sleep_time_range', (5, 10))[0]
        max_delay = SCRAPER_CONFIG.get('sleep_time_range', (5, 10))[1]
    except (ImportError, KeyError):
        # Fallback to environment variables
        db_uri = os.getenv('PG_URI')
        db_name = os.getenv('PG_DB_NAME', 'fbref')
        db_user = os.getenv('PG_USER', 'postgres')
        db_password = os.getenv('PG_PASSWORD', 'password')
        db_host = os.getenv('PG_HOST', 'localhost')
        db_port = os.getenv('PG_PORT', '5432')
        base_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        matches_to_keep = 7
        min_delay = 5
        max_delay = 10
    
    # Define the year(s) to scrape (current season)
    current_year = datetime.now().year
    
    # Initialize the scraper
    logger.info(f"Initializing scraper with base URL: {base_url}")
    scraper = FBrefScraper(
        base_url=base_url, 
        years=[current_year],
        min_delay=min_delay,
        max_delay=max_delay
    )
    
    # Initialize database manager
    if db_uri:
        logger.info(f"Connecting to database using URI")
        db_manager = FBrefDatabaseManager(connection_uri=db_uri)
    else:
        logger.info(f"Connecting to database: {db_name} on {db_host}")
        db_manager = FBrefDatabaseManager(
            db_name=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
    
    # Initialize database (create tables if needed)
    db_manager.initialize_db()
    
    # Scrape only recent matches
    logger.info("Starting to scrape recent matches...")
    recent_matches = scraper.get_recent_matches(limit=matches_to_keep)
    
    if not recent_matches.empty:
        # Save to CSV for backup
        csv_path = scraper.save_data(recent_matches, "recent_matches.csv")
        logger.info(f"Saved backup to {csv_path}")
        
        # Store in PostgreSQL
        stored_count = db_manager.store_recent_matches(recent_matches)
        print(f"\nStored {stored_count} recent matches in the database.")
        
        # Example: pull back data for the first team scraped
        if len(recent_matches['team'].unique()) > 0:
            team_name = recent_matches['team'].unique()[0]
            team_matches = db_manager.get_recent_team_matches(team_name)
            print(f"\n{team_name}'s recent matches:")
            if not team_matches.empty:
                print(team_matches[['date','opponent','result','gf','ga']].to_string(index=False))
        
        # Show top teams by form (last 5 matches)
        print("\nTeam Form (last 5 matches):")
        form_query = """
        WITH recent_form AS (
            SELECT 
                team,
                date,
                result,
                points,
                ROW_NUMBER() OVER (PARTITION BY team ORDER BY date DESC) as match_num
            FROM recent_matches
        )
        SELECT 
            team,
            SUM(points) as points,
            string_agg(result, '' ORDER BY date DESC) as form
        FROM recent_form
        WHERE match_num <= 5
        GROUP BY team
        ORDER BY points DESC
        LIMIT 10;
        """
        form_table = db_manager.execute_query(form_query)
        if not form_table.empty:
            print(form_table.to_string(index=False))
    else:
        logger.error("No recent matches found—nothing to store.")
        print("No matches were found. Check the logs for more details.")


if __name__ == "__main__":
    print("Starting FBref scraper...")
    main()
    print("Completed scraping.")
    # The duplicate main() call has been removed