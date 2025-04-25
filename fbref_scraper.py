import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fbref_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_scraper")

# Add user agent to mimic a browser to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class FBrefScraper:
    """Class to scrape football data from FBref.com"""
    
    def __init__(self, base_url: str = "https://fbref.com/en/comps/9/Premier-League-Stats", years: List[int] = None):
        """
        Initialize the scraper with base URL and years to scrape
        
        Args:
            base_url: Starting URL for the Premier League stats
            years: List of years to scrape (e.g. [2024, 2023, 2022])
        """
        self.base_url = base_url
        self.years = years or [datetime.now().year]
        self.all_matches = []
        self.team_urls = []
        self.data_dir = "fbref_data"
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_html(self, url: str) -> Optional[str]:
        """
        Download HTML from URL with error handling and random delay
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string or None if request failed
        """
        try:
            # Random delay between 1-3 seconds to be respectful to the server
            time.sleep(1 + random.random() * 2)
            
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
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
            return None
            
        # Parse matches
        try:
            matches = pd.read_html(team_html, match="Scores & Fixtures")[0]
        except Exception as e:
            logger.error(f"Failed to parse matches for {team_name}: {e}")
            return None
            
        # Get shooting stats
        shooting_link = self.get_shooting_link(team_html)
        if not shooting_link:
            return None
            
        shooting_html = self.get_html(shooting_link)
        if not shooting_html:
            return None
            
        # Parse shooting stats
        try:
            shooting = pd.read_html(shooting_html, match="Shooting")[0]
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

def main():
    """Main function to run the scraper and store data in PostgreSQL"""
    # Define the year(s) to scrape (e.g. current season)
    current_year = datetime.now().year
    
    # Initialize the scraper for recent matches only
    scraper = FBrefScraper(years=[current_year])
    
    # Initialize database manager with your PostgreSQL settings
    db_manager = FBrefDatabaseManager(
        db_name="YOUR_DB_NAME",       # e.g. "fbref_data"
        user="YOUR_DB_USER",          # e.g. "fbref_user" or "postgres"
        password="YOUR_DB_PASSWORD",  # the password you set in Postgres
        host="localhost",             # or your DB host
        port="5432"                   # change if you’re using a non‐default port
    )
    
    # Create tables if they don't already exist
    db_manager.initialize_db()
    
    # Scrape only recent matches
    logger.info("Starting to scrape recent matches...")
    recent_matches = scraper.get_recent_matches(limit=7)
    
    if not recent_matches.empty:
        # (Optional) save a CSV backup
        scraper.save_data(recent_matches, "recent_matches.csv")
        
        # Upsert into your Postgres table
        stored = db_manager.store_recent_matches(recent_matches)
        print(f"\nStored {stored} recent matches in the database.")
        
        # Example: pull back data for the first team scraped
        team_name = recent_matches['team'].iloc[0]
        team_matches = db_manager.get_recent_team_matches(team_name)
        print(f"\n{team_name}'s recent matches:")
        if not team_matches.empty:
            print(team_matches[['date','opponent','result','gf','ga']].to_string(index=False))
    else:
        logger.error("No recent matches found—nothing to store.")
