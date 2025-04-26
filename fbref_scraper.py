import pandas as pd
import time
from bs4 import BeautifulSoup
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import random
import psycopg2
from psycopg2.extras import execute_values
import io
import hashlib
import sys
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

# Expanded list of User Agents to better mimic real browsers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 OPR/97.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; rv:112.0) Gecko/20100101 Firefox/112.0'
]

class FBrefScraper:
    """Class to scrape football data from FBref.com with improved rate limit handling"""
    
    def __init__(self, base_url: str = "https://fbref.com/en/comps/9/Premier-League-Stats", 
                 years: List[int] = None, 
                 min_delay: int = 15, 
                 max_delay: int = 30,
                 cache_dir: str = "data/cache",
                 max_cache_age: int = 168):  # 1 week cache
        """
        Initialize the scraper with base URL and years to scrape
        
        Args:
            base_url: Starting URL for the Premier League stats
            years: List of years to scrape (e.g. [2024, 2023, 2022])
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            cache_dir: Directory to store cached HTML files
            max_cache_age: Maximum age of cached files in hours (default: 1 week)
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
        self.processed_teams = []
        
        # Create a session for persistent connections
        self.session = requests.Session()
        
        # Initialize the rate limit handler
        self.rate_handler = RateLimitHandler(min_delay=min_delay, max_delay=max_delay)
        
        # Try to load previous rate limiting state
        self.rate_handler.load_state()
        
        # Initialize the proxy manager
        self.proxy_manager = ProxyManager()
        
        # Create required directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize browser-like session
        self.init_browser_session()
    
    def init_browser_session(self):
        """Initialize a session with browser-like behavior"""
        self.session = requests.Session()
        
        # First, visit the homepage to get cookies
        headers = self.get_random_headers()
        try:
            self.session.get("https://fbref.com/", headers=headers, timeout=20)
            time.sleep(random.uniform(2, 5))
            
            # Then visit a few random pages to simulate normal browsing
            sample_pages = [
                "https://fbref.com/en/",
                "https://fbref.com/en/comps/",
                "https://fbref.com/en/comps/9/Premier-League-Stats"
            ]
            
            for page in random.sample(sample_pages, 2):
                self.session.get(page, headers=headers, timeout=20)
                time.sleep(random.uniform(3, 7))
                
            logger.info("Browser session initialized with cookies")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser session: {e}")
            return False
    
    def get_random_headers(self):
        """Get random user agent headers to avoid detection"""
        user_agent = random.choice(USER_AGENTS)
        
        # Add browser-like accept headers
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://fbref.com/',
            'DNT': '1'
        }
        
        return headers
    
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
    
    def is_rate_limited(self, response):
        """
        Check if a response indicates rate limiting
        
        Args:
            response: The requests response object
            
        Returns:
            True if rate limited, False otherwise
        """
        # Check for 429 status code
        if response.status_code == 429:
            return True
            
        # Check for rate limit messages in the content
        rate_limit_indicators = [
            "too many requests",
            "rate limit exceeded",
            "please slow down",
            "try again later"
        ]
        
        # Convert response content to lowercase string for checking
        content = response.text.lower()
        return any(indicator in content for indicator in rate_limit_indicators)

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
        current_proxy = None
        
        while retry_count <= max_retries:
            try:
                # Wait before request (handled by rate_handler)
                self.rate_handler.wait_before_request()
                
                # Get with random headers
                headers = self.get_random_headers()
                
                # Get proxy if available
                proxies = self.proxy_manager.get_proxy()
                current_proxy = proxies
                
                # Add referrer to make request look more natural
                if 'Referer' not in headers:
                    base_domain = "/".join(url.split("/")[:3])
                    headers['Referer'] = base_domain
                
                # Make the request
                response = self.session.get(
                    url, 
                    headers=headers, 
                    proxies=proxies, 
                    timeout=30,
                    allow_redirects=True
                )
                
                # Check for rate limiting
                if self.is_rate_limited(response):
                    retry_count += 1
                    
                    # Handle rate limit with rate_handler
                    should_retry, backoff_time = self.rate_handler.handle_rate_limit(retry_count, max_retries)
                    
                    # Mark proxy as failed
                    if current_proxy:
                        self.proxy_manager.mark_proxy_failed(current_proxy)
                    
                    if should_retry:
                        time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(f"Maximum retries reached for {url}")
                        return None
                
                # For other errors, fail fast
                response.raise_for_status()
                
                # Mark proxy as successful
                if current_proxy:
                    self.proxy_manager.mark_proxy_success(current_proxy)
                
                # Reset rate limit counter on success
                self.rate_handler.reset_after_success()
                
                # Save rate limit state
                self.rate_handler.save_state()
                
                # Cache successful response
                self.save_to_cache(url, response.text)
                return response.text
                
            except requests.RequestException as e:
                if current_proxy:
                    self.proxy_manager.mark_proxy_failed(current_proxy)
                
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
            # Try multiple selectors to find the table
            selectors = ['table.stats_table', 'table#results', '#standings_1 table']
            standings_table = None
            
            for selector in selectors:
                tables = soup.select(selector)
                if tables:
                    standings_table = tables[0]
                    break
            
            if not standings_table:
                # Fallback: try to find any table that might contain teams
                tables = soup.find_all('table')
                if tables:
                    # Look for tables that likely contain teams
                    for table in tables:
                        if table.find('a', href=lambda href: href and '/squads/' in href):
                            standings_table = table
                            break
            
            if not standings_table:
                logger.error("Could not find standings table in HTML")
                return []
                
            links = [l.get("href") for l in standings_table.find_all('a')]
            links = [l for l in links if l and '/squads/' in l]
            
            # Remove duplicates while preserving order
            unique_links = []
            seen = set()
            for link in links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)
            
            return [f"https://fbref.com{l}" for l in unique_links]
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
            # Look for links with 'prev' class or text containing 'Previous Season'
            prev_links = soup.select("a.prev")
            if prev_links:
                return f"https://fbref.com{prev_links[0].get('href')}"
                
            # Alternative approach
            all_links = soup.find_all('a')
            for link in all_links:
                if link.text and ('previous' in link.text.lower() or 'prev' in link.text.lower()) and 'season' in link.text.lower():
                    return f"https://fbref.com{link.get('href')}"
                
            return None
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
        
        # Try multiple approaches to find the shooting stats link
        
        # First approach: direct link containing 'shooting'
        links = [l.get("href") for l in soup.find_all('a')]
        shooting_links = [l for l in links if l and 'all_comps/shooting/' in l]
        
        if shooting_links:
            return f"https://fbref.com{shooting_links[0]}"
            
        # Second approach: look for links with shooting text
        for link in soup.find_all('a'):
            if link.text and 'shooting' in link.text.lower():
                href = link.get('href')
                if href:
                    return f"https://fbref.com{href}"
        
        logger.warning("No shooting stats link found")
        return None

    def get_standard_stats_link(self, html: str) -> Optional[str]:
        """
        Extract standard stats link from team page (for corners data)
        
        Args:
            html: HTML content of the team page
            
        Returns:
            URL for the standard stats or None if not found
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Multiple approaches to find standard stats link
        
        # First approach: direct link containing 'stats'
        links = [l.get("href") for l in soup.find_all('a')]
        standard_links = [l for l in links if l and 'all_comps/stats/' in l]
        
        if standard_links:
            return f"https://fbref.com{standard_links[0]}"
            
        # Second approach: look for links with 'standard' or 'stats' text
        for link in soup.find_all('a'):
            if link.text and ('standard' in link.text.lower() or 'stats' in link.text.lower()):
                href = link.get('href')
                if href and 'matchlogs' in href:
                    return f"https://fbref.com{href}"
        
        logger.warning("No standard stats link found")
        return None

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
            tables = pd.read_html(html_io)
            
            # Find the table with scores and fixtures
            matches_df = None
            for table in tables:
                if 'Date' in table.columns and 'Opponent' in table.columns:
                    matches_df = table
                    break
                    
            if matches_df is None:
                logger.error(f"Failed to find matches table for {team_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse matches for {team_name}: {e}")
            return None
            
        # Get shooting stats
        shooting_link = self.get_shooting_link(team_html)
        shooting = None
        
        if shooting_link:
            shooting_html = self.get_html(shooting_link)
            if shooting_html:
                try:
                    # Using StringIO to handle the warning about literal HTML
                    html_io = io.StringIO(shooting_html)
                    shooting_tables = pd.read_html(html_io)
                    
                    # Find shooting table
                    for table in shooting_tables:
                        if isinstance(table.columns, pd.MultiIndex):
                            table.columns = table.columns.droplevel(0)
                        
                        # Check if it's likely the shooting table
                        if 'Date' in table.columns and ('Sh' in table.columns or 'Shots' in table.columns):
                            shooting = table
                            break
                            
                    if shooting is None:
                        logger.warning(f"No shooting table found for {team_name}")
                except Exception as e:
                    logger.error(f"Failed to parse shooting stats for {team_name}: {e}")
            else:
                logger.error(f"Failed to get shooting stats HTML for {team_name}")
        else:
            logger.warning(f"No shooting link found for {team_name}")
        
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
                        if 'Date' in stats_table.columns and ('CK' in stats_table.columns or 'Corners' in stats_table.columns):
                            corners_col = 'CK' if 'CK' in stats_table.columns else 'Corners'
                            corners_data = stats_table[['Date', corners_col]].copy()
                            if corners_col != 'CK':
                                corners_data.rename(columns={corners_col: 'CK'}, inplace=True)
                            break
                except Exception as e:
                    logger.warning(f"Failed to parse corners data for {team_name}: {e}")
        
        # Merge data
        try:
            team_data = matches_df.copy()
            
            # Add shooting data if available
            if shooting is not None and not shooting.empty:
                # Get essential shooting columns
                shooting_cols = ["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
                available_cols = [col for col in shooting_cols if col in shooting.columns]
                
                if 'Date' in available_cols and len(available_cols) > 1:
                    # Ensure Date column is properly formatted in both DataFrames
                    # Convert to strings if not already
                    if not pd.api.types.is_string_dtype(team_data['Date']):
                        team_data['Date'] = team_data['Date'].astype(str)
                    if not pd.api.types.is_string_dtype(shooting['Date']):
                        shooting['Date'] = shooting['Date'].astype(str)
                        
                    team_data = team_data.merge(shooting[available_cols], on="Date", how="left")
            
            # Add corners data if available
            if corners_data is not None and not corners_data.empty:
                # Ensure Date column is properly formatted
                if not pd.api.types.is_string_dtype(corners_data['Date']):
                    corners_data['Date'] = corners_data['Date'].astype(str)
                if not pd.api.types.is_string_dtype(team_data['Date']):
                    team_data['Date'] = team_data['Date'].astype(str)
                    
                team_data = team_data.merge(corners_data, on="Date", how="left")
            
            # Filter for Premier League matches only if 'Comp' column exists
            if 'Comp' in team_data.columns:
                team_data = team_data[team_data["Comp"].str.contains("Premier League", case=False, na=False)]
            
            # Add team and season info
            team_data["Season"] = year
            team_data["Team"] = team_name
            
            # Calculate points based on result
            if 'Result' in team_data.columns:
                result_to_points = {'W': 3, 'D': 1, 'L': 0}
                team_data['Points'] = team_data['Result'].map(result_to_points)
            
            # Add is_home field if venue column exists
            if 'Venue' in team_data.columns:
                team_data['is_home'] = team_data['Venue'].str.lower() == 'home'
            
            # Generate match_id
            if 'Date' in team_data.columns and 'Opponent' in team_data.columns:
                team_data['match_id'] = team_data.apply(
                    lambda row: f"{row['Date']}_{team_name}_{row['Opponent']}".replace(' ', '_'),
                    axis=1
                )
            
            return team_data
            
        except Exception as e:
            logger.error(f"Failed to merge data for {team_name}: {e}")
            return None

    def save_scraping_state(self, processed_teams):
        """
        Save the current scraping state to a file
        
        Args:
            processed_teams: List of team names that have been processed
        """
        state_file = os.path.join(self.data_dir, "scrape_state.json")
        state = {
            "timestamp": datetime.now().isoformat(),
            "processed_teams": processed_teams,
            "current_year": self.years[0] if self.years else datetime.now().year
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            logger.info(f"Saved scraping state with {len(processed_teams)} processed teams")
            return True
        except Exception as e:
            logger.warning(f"Failed to save scraping state: {e}")
            return False

    def load_scraping_state(self):
        """
        Load the previous scraping state if available
        
        Returns:
            List of team names that have been processed
        """
        state_file = os.path.join(self.data_dir, "scrape_state.json")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                processed_teams = state.get("processed_teams", [])
                logger.info(f"Loaded previous state with {len(processed_teams)} processed teams")
                
                # Update the year if stored in state
                if "current_year" in state and self.years:
                    self.years[0] = state["current_year"]
                
                return processed_teams
            except Exception as e:
                logger.warning(f"Failed to load scraping state: {e}")
        
        return []

    def combine_intermediate_results(self):
        """
        Combine all intermediate batch results into a single DataFrame
        
        Returns:
            Combined DataFrame from all intermediate files
        """
        intermediate_files = [f for f in os.listdir(self.data_dir) 
                             if f.startswith("temp_batch_") and f.endswith(".csv")]
        
        if not intermediate_files:
            logger.warning("No intermediate batch files found")
            return pd.DataFrame()
        
        all_dfs = []
        for file in intermediate_files:
            try:
                filepath = os.path.join(self.data_dir, file)
                df = pd.read_csv(filepath)
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read intermediate file {file}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates
        if 'match_id' in combined_df.columns:
            combined_df.drop_duplicates(subset=['match_id'], inplace=True)
        else:
            # Create a temp ID based on available columns for deduplication
            id_cols = []
            for col in ['date', 'team', 'opponent']:
                if col in combined_df.columns:
                    id_cols.append(col)
            
            if id_cols:
                combined_df.drop_duplicates(subset=id_cols, inplace=True)
        
        return combined_df
    
    def scrape_in_batches(self, batch_size=3, batch_delay=600):
        """
        Scrape teams in batches with delays between batches to avoid rate limiting
        
        Args:
            batch_size: Number of teams to scrape in each batch (reduced to 3)
            batch_delay: Delay in seconds between batches (increased to 10 minutes)
            
        Returns:
            Combined DataFrame with all matches
        """
        logger.info(f"Starting batch scraping with {batch_size} teams per batch")
        
        # Load previous state
        self.processed_teams = self.load_scraping_state()
        
        # Get all team URLs first
        html = self.get_html(self.base_url)
        if not html:
            logger.error("Failed to get league page HTML")
            
            # Try to recover from intermediate results
            logger.info("Attempting to recover from intermediate results...")
            return self.combine_intermediate_results()
        
        team_urls = self.extract_team_urls(html)
        if not team_urls:
            logger.error("No team URLs found")
            return self.combine_intermediate_results()
        
        # Filter out already processed teams
        filtered_team_urls = []
        for url in team_urls:
            team_name = self.extract_team_name(url)
            if team_name not in self.processed_teams:
                filtered_team_urls.append(url)
            else:
                logger.info(f"Skipping already processed team: {team_name}")
        
        if not filtered_team_urls and self.processed_teams:
            logger.info("All teams already processed, combining existing results")
            return self.combine_intermediate_results()
            
        logger.info(f"Found {len(filtered_team_urls)} teams to scrape (out of {len(team_urls)} total)")
        
        # Split into batches
        batches = [filtered_team_urls[i:i+batch_size] for i in range(0, len(filtered_team_urls), batch_size)]
        all_team_dfs = []
        
        for batch_num, batch_urls in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_urls)} teams")
            batch_dfs = []
            
            # Process this batch
            for team_url in batch_urls:
                team_name = self.extract_team_name(team_url)
                logger.info(f"Scraping {team_name}")
                
                try:
                    team_df = self.parse_matches_and_stats(team_url, self.years[0])
                    if team_df is not None and not team_df.empty:
                        all_team_dfs.append(team_df)
                        batch_dfs.append(team_df)
                        # Mark as processed
                        self.processed_teams.append(team_name)
                        self.save_scraping_state(self.processed_teams)
                    else:
                        logger.warning(f"No valid data found for {team_name}")
                except Exception as e:
                    logger.error(f"Error processing {team_name}: {e}")
                    # Save current progress before continuing
                    self.save_scraping_state(self.processed_teams)
                
                # Add delay between teams within batch (30-60 seconds)
                team_delay = random.uniform(30, 60)
                logger.info(f"Waiting {team_delay:.1f} seconds before next team...")
                time.sleep(team_delay)
            
            # Save intermediate results after each batch
            if batch_dfs:
                temp_df = pd.concat(batch_dfs, ignore_index=True)
                temp_df.columns = [c.lower() for c in temp_df.columns]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filepath = os.path.join(self.data_dir, f"temp_batch_{batch_num}_{timestamp}.csv")
                temp_df.to_csv(temp_filepath, index=False)
                logger.info(f"Saved intermediate results for batch {batch_num} to {temp_filepath}")
            
            # Delay before next batch if not the last batch
            if batch_num < len(batches):
                actual_delay = batch_delay + random.uniform(-60, 60)  # Add some randomness (±1 minute)
                logger.info(f"Batch {batch_num} complete. Waiting {actual_delay:.1f} seconds before next batch...")
                time.sleep(actual_delay)
        
        # Combine with previously processed results
        if all_team_dfs:
            new_df = pd.concat(all_team_dfs, ignore_index=True)
            new_df.columns = [c.lower() for c in new_df.columns]
            
            # Combine with any intermediate results from previous runs
            prev_df = self.combine_intermediate_results()
            if not prev_df.empty:
                logger.info("Combining with previous intermediate results")
                full_df = pd.concat([new_df, prev_df], ignore_index=True)
                
                # Remove duplicates
                if 'match_id' in full_df.columns:
                    full_df.drop_duplicates(subset=['match_id'], inplace=True)
                elif all(col in full_df.columns for col in ['date', 'team', 'opponent']):
                    full_df.drop_duplicates(subset=['date', 'team', 'opponent'], inplace=True)
                
                return full_df
            return new_df
        
        # If no new data collected, return any existing intermediate results
        intermediate_df = self.combine_intermediate_results()
        if not intermediate_df.empty:
            logger.info("No new data collected, returning existing intermediate results")
            return intermediate_df
            
        logger.error("No data collected from any team")
        return pd.DataFrame()