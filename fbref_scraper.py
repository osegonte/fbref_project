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
from rate_limit_handler import RateLimitHandler
from proxy_helper import ProxyManager

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
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36 OPR/82.0.4227.33'
]

class FBrefScraper:
    """Class to scrape football data from FBref.com"""
    
    def __init__(self, base_url: str = "https://fbref.com/en/comps/9/Premier-League-Stats", 
                 years: List[int] = None, 
                 min_delay: int = 10, 
                 max_delay: int = 20,
                 cache_dir: str = "data/cache",
                 max_cache_age: int = 48):
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
        
        # Create a session for persistent connections
        self.session = requests.Session()
        
        # Initialize the rate limit handler
        self.rate_handler = RateLimitHandler(min_delay=min_delay, max_delay=max_delay)
        
        # Initialize the proxy manager
        self.proxy_manager = ProxyManager()
        
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
            'Cache-Control': 'max-age=0',
            'Referer': 'https://fbref.com/',
            'DNT': '1'
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
        
        while retry_count <= max_retries:
            try:
                # Wait before request (handled by rate_handler)
                self.rate_handler.wait_before_request()
                
                # Get with random headers
                headers = self.get_random_headers()
                
                # Get proxy if available
                proxies = self.proxy_manager.get_proxy()
                
                # Make the request
                response = self.session.get(url, headers=headers, proxies=proxies, timeout=20)
                
                # If rate limited, back off and retry
                if response.status_code == 429:
                    retry_count += 1
                    
                    # Handle rate limit with rate_handler
                    should_retry, backoff_time = self.rate_handler.handle_rate_limit(retry_count, max_retries)
                    
                    if should_retry:
                        time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(f"Maximum retries reached for {url}")
                        return None
                
                # For other errors, fail fast
                response.raise_for_status()
                
                # Reset rate limit counter on success
                self.rate_handler.reset_after_success()
                
                # Cache successful response
                self.save_to_cache(url, response.text)
                return response.text
                
            except requests.RequestException as e:
                logger.error(f"Failed to fetch {url}: {e}")
                retry_count += 1
                
                if retry_count <= max_retries and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    backoff_time = self.min_delay * (2 ** retry_count)
                    logger.info(f"Connection error or timeout, retrying in {backoff_time} seconds ({retry_count}/{max_retries})")
                    time.sleep(backoff_time)
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

    def get_standard_stats_link(self, html: str) -> Optional[str]:
        """
        Extract standard stats link from team page (for corners data)
        
        Args:
            html: HTML content of the team page
            
        Returns:
            URL for the standard stats or None if not found
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/stats/' in l]
        
        if not links:
            logger.warning("No standard stats link found")
            return None
            
        return f"https://fbref.com{links[0]}"

    def parse_matches_and_stats(self, team_url: str, year: int) -> Optional[pd.DataFrame]:
        """
        Parse matches, shooting data, and corners data for a team
        
        Args:
            team_url: URL of the team page
            year: Season year
            
        Returns:
            DataFrame with combined match and stats data or None if failed
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
        
        # Get standard stats for corners
        standard_stats_link = self.get_standard_stats_link(team_html)
        corners_data = None
        
        if standard_stats_link:
            standard_stats_html = self.get_html(standard_stats_link)
            if standard_stats_html:
                try:
                    html_io = io.StringIO(standard_stats_html)
                    stats_tables = pd.read_html(html_io)
                    
                    # Find the table with corners data
                    for stats_table in stats_tables:
                        if isinstance(stats_table.columns, pd.MultiIndex):
                            stats_table.columns = stats_table.columns.droplevel(0)
                        
                        # Look for corner kicks columns (CK or similar)
                        if 'CK' in stats_table.columns:
                            corners_data = stats_table[['Date', 'CK']]
                            break
                        elif 'Corners' in stats_table.columns:
                            corners_data = stats_table[['Date', 'Corners']]
                            corners_data.rename(columns={'Corners': 'CK'}, inplace=True)
                            break
                except Exception as e:
                    logger.warning(f"Failed to parse corners data for {team_name}: {e}")
        
        # Merge data
        try:
            # Get essential shooting columns
            shooting_cols = ["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
            available_cols = [col for col in shooting_cols if col in shooting.columns]
            
            team_data = matches.merge(shooting[available_cols], on="Date", how="inner")
            
            # Add corners data if available
            if corners_data is not None and not corners_data.empty:
                team_data = team_data.merge(corners_data, on="Date", how="left")
            
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
            team_df = self.parse_matches_and_stats(team_url, year)
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
        
        # Process corners data if available
        if 'ck' in recent_matches.columns:
            # For home matches, CK is corners for
            recent_matches.loc[recent_matches['is_home'], 'corners_for'] = recent_matches.loc[recent_matches['is_home'], 'ck']
            # We don't have corners against directly, would need opponent's data
            
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

    def scrape_in_batches(self, batch_size=5, batch_delay=300):
        """
        Scrape teams in batches with delays between batches to avoid rate limiting
        
        Args:
            batch_size: Number of teams to scrape in each batch
            batch_delay: Delay in seconds between batches
            
        Returns:
            Combined DataFrame with all matches
        """
        logger.info(f"Starting batch scraping with {batch_size} teams per batch")
        
        # Get all team URLs first
        html = self.get_html(self.base_url)
        if not html:
            logger.error("Failed to get league page HTML")
            return pd.DataFrame()
        
        team_urls = self.extract_team_urls(html)
        if not team_urls:
            logger.error("No team URLs found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(team_urls)} teams, will scrape in batches of {batch_size}")
        
        # Split into batches
        batches = [team_urls[i:i+batch_size] for i in range(0, len(team_urls), batch_size)]
        all_team_dfs = []
        
        for batch_num, batch_urls in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_urls)} teams")
            
            # Process this batch
            for team_url in batch_urls:
                team_name = self.extract_team_name(team_url)
                logger.info(f"Scraping {team_name}")
                
                team_df = self.parse_matches_and_stats(team_url, self.years[0])
                if team_df is not None and not team_df.empty:
                    all_team_dfs.append(team_df)
                else:
                    logger.warning(f"No valid data found for {team_name}")
            
            # Delay before next batch if not the last batch
            if batch_num < len(batches):
                delay = batch_delay + random.uniform(-30, 30)  # Add some randomness
                logger.info(f"Batch {batch_num} complete. Waiting {delay:.1f} seconds before next batch...")
                time.sleep(delay)
        
        if not all_team_dfs:
            logger.error("No data collected from any team")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_team_dfs, ignore_index=True)
        combined_df.columns = [c.lower() for c in combined_df.columns]
        
        return combined_df


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
            corners_for, comp, round, season, is_home, scrape_date
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
            corners_for = EXCLUDED.corners_for,
            scrape_date;
        """
        
        # Prepare data for insertion
        columns = ['match_id', 'date', 'team', 'opponent', 'venue', 'result', 
                  'gf', 'ga', 'points', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt',
                  'corners_for', 'comp', 'round', 'season', 'is_home', 'scrape_date']
        
        # Ensure all required columns exist
        for col in columns:
            if col not in df.columns and col not in ['is_home', 'scrape_date', 'corners_for']:
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
                elif col == 'corners_for':
                    # Handle corners data if available
                    if 'ck' in df.columns and not pd.isna(row.get('ck')):
                        value.append(int(row['ck']))
                    else:
                        value.append(None)
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