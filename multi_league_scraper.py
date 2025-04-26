#!/usr/bin/env python3
"""
Multi-league FBref scraper that can collect data from various competitions.
Works with the existing fbref_scraper infrastructure and shares common utilities.
"""

import os
import sys
import time
import logging
import random
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Make sure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from existing modules
from fbref_scraper import FBrefScraper, FBrefDatabaseManager
from rate_limit_handler import RateLimitHandler
from polite_request import PoliteRequester
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/multi_league_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_league_scraper")

def get_available_leagues():
    """
    Get a comprehensive list of leagues available on FBref by scraping the coverage page
    
    Returns:
        Dictionary of league IDs and names
    """
    try:
        import re
        import pandas as pd
        from urllib.parse import unquote
        
        url = "https://fbref.com/en/about/coverage"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch coverage page: {response.status_code}")
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Method 1: Try to find all tables on the page
        try:
            all_tables = pd.read_html(html)
            if all_tables:
                # Look for the largest table that likely contains all competitions
                largest_table = max(all_tables, key=len)
                if len(largest_table) > 50:  # Assume it's the competitions table if it has many rows
                    df = largest_table
                    
                    # Check if 'Tier' column exists
                    if 'Tier' in df.columns:
                        # Process this table
                        logger.info(f"Found competition table with {len(df)} rows (Method 1)")
                        
                        # Clean up - drop any duplicate header rows
                        if df['Tier'].astype(str).str.contains('Tier').any():
                            df = df[~df['Tier'].astype(str).str.contains('Tier')]
                        
                        # Get IDs from the actual page HTML since pandas doesn't preserve href attributes
                        leagues = {}
                        
                        # Find all links that look like competition links
                        comp_links = soup.select('a[href*="/comps/"]')
                        for link in comp_links:
                            href = link.get('href', '')
                            name = link.text.strip()
                            
                            if name and '/comps/' in href:
                                # Extract ID from URL pattern like /en/comps/9/Premier-League-Stats
                                match = re.search(r'/comps/(\d+)/', href)
                                if match:
                                    league_id = match.group(1)
                                    if league_id.isdigit():
                                        leagues[league_id] = name
                        
                        logger.info(f"Found {len(leagues)} leagues on FBref")
                        if leagues:
                            return leagues
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
        
        # Method 2: Direct BeautifulSoup parsing
        try:
            # Find all tables and look for one with competition data
            tables = soup.find_all('table')
            for table in tables:
                # Check if this table has competition data by looking for characteristic columns
                headers = [th.text.strip() for th in table.select('thead th')]
                if 'Tier' in headers and 'Comp' in headers and 'Country' in headers:
                    logger.info(f"Found competition table (Method 2)")
                    
                    leagues = {}
                    # Process all rows
                    for row in table.select('tbody tr'):
                        cells = row.select('td')
                        if len(cells) < 4:  # Need at least Tier, Comp, Country columns
                            continue
                        
                        try:
                            # Check if this is a league (Tier >= 1)
                            tier_text = cells[0].text.strip()
                            if not tier_text.isdigit() or int(tier_text) < 1:
                                continue  # Skip cups, etc.
                            
                            # Extract league name and ID
                            comp_cell = cells[1]
                            link = comp_cell.find('a')
                            
                            if link and 'href' in link.attrs:
                                league_name = link.text.strip()
                                href = link['href']
                                
                                # Extract ID from URL
                                match = re.search(r'/comps/(\d+)/', href)
                                if match:
                                    league_id = match.group(1)
                                    if league_id.isdigit():
                                        leagues[league_id] = league_name
                        except Exception as row_error:
                            logger.debug(f"Error parsing row: {row_error}")
                            continue
                    
                    logger.info(f"Found {len(leagues)} leagues on FBref")
                    if leagues:
                        return leagues
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Fallback - directly parse competition links from the page
        try:
            leagues = {}
            # Find all links that might be competition links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/comps/' in href:
                    match = re.search(r'/comps/(\d+)/', href)
                    if match:
                        league_id = match.group(1)
                        name = link.text.strip()
                        if league_id.isdigit() and name:
                            leagues[league_id] = name
            
            logger.info(f"Found {len(leagues)} leagues using fallback method")
            if leagues:
                return leagues
        except Exception as e:
            logger.warning(f"Method 3 failed: {e}")
        
        raise Exception("Could not extract league information using any method")
    
    except Exception as e:
        logger.error(f"Error getting available leagues: {e}")
        # Fallback to a basic list of top leagues
        logger.info("Using hardcoded list of top 5 leagues as fallback")
        return {
            "9": "Premier League",
            "12": "La Liga",
            "11": "Serie A",
            "20": "Bundesliga",
            "13": "Ligue 1",
        }

def get_league_url(league_id, league_name):
    """
    Construct the league URL for FBref
    
    Args:
        league_id: FBref competition ID
        league_name: Name of the league
    
    Returns:
        URL for the league page
    """
    # Replace spaces with hyphens and remove special characters
    clean_name = league_name.replace(" ", "-").replace(".", "")
    return f"https://fbref.com/en/comps/{league_id}/{clean_name}-Stats"

class LeagueScraper(FBrefScraper):
    """Extended FBref scraper for specific leagues"""
    
    def __init__(self, league_id, league_name):
        """
        Initialize for a specific league
        
        Args:
            league_id: FBref competition ID
            league_name: Name of the league
        """
        self.league_id = league_id
        self.league_name = league_name
        self.base_url = get_league_url(league_id, league_name)
        
        # Initialize the base scraper
        super().__init__()
        
        # Update data specific to this league
        self.league_dir = os.path.join(self.data_dir, f"league_{league_id}")
        os.makedirs(self.league_dir, exist_ok=True)
    
    def parse_team_matches(self, team_url):
        """
        Override to add league information
        
        Args:
            team_url: URL of the team page
            
        Returns:
            DataFrame with match data or None if failed
        """
        # Call the parent method
        team_df = super().parse_team_matches(team_url)
        
        if team_df is not None and not team_df.empty:
            # Add league information
            team_df["league_id"] = self.league_id
            team_df["league_name"] = self.league_name
            
            # Keep only recent matches (last 7)
            if len(team_df) > 7:
                team_df = team_df.sort_values(by="date", ascending=False).head(7).reset_index(drop=True)
            
            return team_df
        
        return None

class MultiLeagueScraper:
    """
    Class to scrape data from multiple leagues on FBref
    """
    
    def __init__(self):
        """Initialize the multi-league scraper"""
        self.leagues = get_available_leagues()
        self.data_dir = "data"
        self.league_data_dir = os.path.join(self.data_dir, "leagues")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.league_data_dir, exist_ok=True)
        
        # Track processed leagues
        self.processed_leagues = self.load_processed_leagues()
        
        # Rate limit handler for adaptive delays
        self.rate_handler = RateLimitHandler(min_delay=15, max_delay=30)
        
        # Polite requester for main page requests
        self.requester = PoliteRequester(
            cache_dir=os.path.join(self.data_dir, "cache"),
            max_cache_age=24  # Cache for 24 hours
        )
    
    def load_processed_leagues(self):
        """
        Load list of already processed leagues
        
        Returns:
            Dictionary of processed leagues
        """
        state_file = os.path.join(self.data_dir, "processed_leagues.json")
        
        if not os.path.exists(state_file):
            return {}
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            logger.info(f"Loaded previously processed leagues: {len(state)} entries")
            return state
            
        except Exception as e:
            logger.warning(f"Failed to load processed leagues: {e}")
            return {}
    
    def save_processed_leagues(self):
        """Save the list of processed leagues"""
        state_file = os.path.join(self.data_dir, "processed_leagues.json")
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self.processed_leagues, f, indent=2)
                
            logger.info(f"Saved state with {len(self.processed_leagues)} processed leagues")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save processed leagues: {e}")
            return False
    
    def check_if_recently_processed(self, league_id, hours=24):
        """
        Check if a league was processed recently
        
        Args:
            league_id: FBref competition ID
            hours: Number of hours to consider as "recent"
            
        Returns:
            True if recently processed, False otherwise
        """
        if league_id not in self.processed_leagues:
            return False
            
        last_processed = self.processed_leagues[league_id].get("last_processed")
        if not last_processed:
            return False
            
        try:
            last_time = datetime.fromisoformat(last_processed)
            time_diff = datetime.now() - last_time
            return time_diff.total_seconds() < hours * 3600
        except (ValueError, TypeError):
            return False
    
    def scrape_league(self, league_id, max_teams=None, force=False):
        """
        Scrape a specific league
        
        Args:
            league_id: FBref competition ID
            max_teams: Maximum number of teams to scrape (None for all)
            force: Force scraping even if recently processed
            
        Returns:
            DataFrame with league data
        """
        # Get league name
        league_name = self.leagues.get(league_id)
        if not league_name:
            logger.error(f"Unknown league ID: {league_id}")
            return pd.DataFrame()
            
        logger.info(f"Scraping {league_name} (ID: {league_id})")
        
        # Check if recently processed
        if not force and self.check_if_recently_processed(league_id):
            logger.info(f"League {league_name} was processed recently, skipping (use --force to override)")
            
            # Try to load existing data
            clean_name = league_name.replace(" ", "_").lower()
            league_dir = os.path.join(self.league_data_dir, clean_name)
            league_file = os.path.join(league_dir, f"{clean_name}_latest.csv")
            
            if os.path.exists(league_file):
                try:
                    return pd.read_csv(league_file)
                except Exception as e:
                    logger.warning(f"Failed to load existing data: {e}")
        
        # Create a directory for this league
        clean_name = league_name.replace(" ", "_").lower()
        league_dir = os.path.join(self.league_data_dir, clean_name)
        os.makedirs(league_dir, exist_ok=True)
        
        # Initialize league-specific scraper
        scraper = LeagueScraper(league_id, league_name)
        
        # Get team URLs for this league
        league_url = get_league_url(league_id, league_name)
        
        # Apply rate limiting
        self.rate_handler.wait_before_request()
        
        league_html = self.requester.fetch(league_url)
        if not league_html:
            logger.error(f"Failed to get league page for {league_name}")
            return pd.DataFrame()
        
        team_urls = scraper.extract_team_urls(league_html)
        
        if not team_urls:
            logger.error(f"No team URLs found for {league_name}")
            return pd.DataFrame()
            
        logger.info(f"Found {len(team_urls)} teams in {league_name}")
        
        # Apply max_teams limit if specified
        if max_teams is not None and max_teams < len(team_urls):
            logger.info(f"Limiting to {max_teams} teams (out of {len(team_urls)})")
            team_urls = team_urls[:max_teams]
        
        # Process each team
        all_team_dfs = []
        
        for i, team_url in enumerate(team_urls, 1):
            team_name = scraper.extract_team_name(team_url)
            logger.info(f"Processing {team_name} ({i}/{len(team_urls)})")
            
            try:
                # Apply rate limiting
                self.rate_handler.wait_before_request()
                
                # Parse team matches
                team_df = scraper.parse_team_matches(team_url)
                
                if team_df is not None and not team_df.empty:
                    logger.info(f"Got {len(team_df)} matches for {team_name}")
                    all_team_dfs.append(team_df)
                    
                    # Save individual team data
                    clean_team_name = team_name.replace(" ", "_").lower()
                    team_file = os.path.join(league_dir, f"{clean_team_name}.csv")
                    team_df.to_csv(team_file, index=False)
                else:
                    logger.warning(f"No valid data found for {team_name}")
            
            except Exception as e:
                logger.error(f"Error processing {team_name}: {e}")
            
            # Add delay between teams
            if i < len(team_urls):
                delay = random.uniform(20, 40)
                logger.info(f"Waiting {delay:.1f}s before next team...")
                time.sleep(delay)
        
        # Combine all team data for this league
        if all_team_dfs:
            league_df = pd.concat(all_team_dfs, ignore_index=True)
            
            # Save league data
            league_file = os.path.join(league_dir, f"{clean_name}_latest.csv")
            league_df.to_csv(league_file, index=False)
            
            # Save timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(league_dir, f"{clean_name}_{timestamp}.csv")
            league_df.to_csv(backup_file, index=False)
            
            logger.info(f"Saved {len(league_df)} matches for {league_name}")
            
            # Mark as processed
            self.processed_leagues[league_id] = {
                "name": league_name,
                "last_processed": datetime.now().isoformat(),
                "teams_count": len(team_urls),
                "matches_count": len(league_df)
            }
            self.save_processed_leagues()
            
            return league_df
        else:
            logger.warning(f"No data collected for {league_name}")
            return pd.DataFrame()
    
    def scrape_multiple_leagues(self, league_ids=None, max_teams_per_league=None, force=False):
        """
        Scrape multiple leagues
        
        Args:
            league_ids: List of league IDs to scrape (None for all major leagues)
            max_teams_per_league: Maximum teams to scrape per league
            force: Force scraping even if recently processed
            
        Returns:
            Dictionary mapping league IDs to DataFrames
        """
        # Default to major top 5 leagues if none specified
        if league_ids is None:
            league_ids = ["9", "12", "11", "20", "13"]
        
        results = {}
        
        for i, league_id in enumerate(league_ids, 1):
            # Get league name
            league_name = self.leagues.get(league_id, f"Unknown League {league_id}")
            logger.info(f"Processing league {i}/{len(league_ids)}: {league_name}")
            
            league_df = self.scrape_league(league_id, max_teams_per_league, force)
            results[league_id] = league_df
            
            # Add longer delay between leagues
            if i < len(league_ids):
                delay = random.uniform(60, 120)  # 1-2 minute delay between leagues
                logger.info(f"Waiting {delay:.1f}s before next league...")
                time.sleep(delay)
        
        # Combine all leagues into a master file
        all_league_dfs = [df for df in results.values() if not df.empty]
        
        if all_league_dfs:
            master_df = pd.concat(all_league_dfs, ignore_index=True)
            
            # Create all_leagues directory
            all_leagues_dir = os.path.join(self.data_dir, "all_leagues")
            os.makedirs(all_leagues_dir, exist_ok=True)
            
            # Save master file
            master_file = os.path.join(all_leagues_dir, "all_leagues_latest.csv")
            master_df.to_csv(master_file, index=False)
            
            # Save timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(all_leagues_dir, f"all_leagues_{timestamp}.csv")
            master_df.to_csv(backup_file, index=False)
            
            logger.info(f"Saved master file with {len(master_df)} matches from {len(all_league_dfs)} leagues")
            
            # Also save to database if URI is provided
            try:
                db_uri = os.getenv('PG_URI')
                if db_uri:
                    db_manager = FBrefDatabaseManager(connection_uri=db_uri)
                    db_manager.initialize_db()
                    count = db_manager.store_recent_matches(master_df)
                    logger.info(f"Stored/updated {count} matches in the database")
            except Exception as e:
                logger.error(f"Database error: {e}")
        
        return results

def list_available_leagues():
    """Print list of available leagues"""
    leagues = get_available_leagues()
    print("\nAvailable Leagues on FBref:")
    print("=" * 60)
    print(f"{'ID':<6} {'League Name':<40} ")
    print("-" * 60)
    
    # Sort leagues by name for easier reading
    sorted_leagues = sorted(leagues.items(), key=lambda x: x[1])
    for league_id, name in sorted_leagues:
        print(f"{league_id:<6} {name:<40}")
        
    print("=" * 60)
    print(f"Total: {len(leagues)} leagues available")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-league FBref scraper")
    
    parser.add_argument("--list", action="store_true", 
                        help="List available leagues and exit")
    
    parser.add_argument("--leagues", type=str, default="9",
                        help="Comma-separated list of league IDs to scrape (default: 9 for Premier League)")
    
    parser.add_argument("--max-teams", type=int, default=None,
                        help="Maximum number of teams to scrape per league")
    
    parser.add_argument("--top5", action="store_true",
                        help="Scrape the top 5 European leagues (England, Spain, Italy, Germany, France)")
    
    parser.add_argument("--force", action="store_true",
                        help="Force scrape even if league was recently processed")
    
    args = parser.parse_args()
    
    # Just list leagues if requested
    if args.list:
        list_available_leagues()
        return 0
    
    try:
        # Initialize scraper
        scraper = MultiLeagueScraper()
        
        # Determine which leagues to scrape
        if args.top5:
            league_ids = ["9", "12", "11", "20", "13"]  # Top 5 European leagues
        else:
            league_ids = [lid.strip() for lid in args.leagues.split(",") if lid.strip()]
        
        # Check if leagues exist
        unknown_leagues = [lid for lid in league_ids if lid not in scraper.leagues]
        if unknown_leagues:
            logger.warning(f"Unknown league IDs: {', '.join(unknown_leagues)}")
            print(f"WARNING: Unknown league IDs: {', '.join(unknown_leagues)}")
            print("Use --list to see available leagues")
        
        known_leagues = [lid for lid in league_ids if lid in scraper.leagues]
        if not known_leagues:
            logger.error("No valid leagues specified")
            print("ERROR: No valid leagues specified. Use --list to see available leagues.")
            return 1
        
        league_names = [scraper.leagues[lid] for lid in known_leagues]
        logger.info(f"Will scrape {len(known_leagues)} leagues: {', '.join(league_names)}")
        
        # Run the scraper
        results = scraper.scrape_multiple_leagues(
            league_ids=known_leagues, 
            max_teams_per_league=args.max_teams,
            force=args.force
        )
        
        # Report results
        successful = [lid for lid, df in results.items() if not df.empty]
        logger.info(f"Successfully scraped {len(successful)}/{len(known_leagues)} leagues")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())