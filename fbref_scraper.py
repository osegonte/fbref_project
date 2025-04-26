#!/usr/bin/env python3
"""
Optimized FBref scraper with robust rate limit handling and batch processing
- Implements domain-level cooldown on rate limits
- Uses adaptive delays based on FBref's robots.txt
- Processes teams in small batches with long delays between batches
- Preserves partial results to minimize data loss
"""

import pandas as pd
import time
import logging
import os
import json
import random
import sys
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import argparse
import psycopg2
from psycopg2.extras import execute_values
from rate_limit_handler import RateLimitHandler
from polite_request import PoliteRequester
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/fbref_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_scraper")

class FBrefDatabaseManager:
    """Class to manage database operations for FBref data"""
    
    def __init__(self, connection_uri=None):
        """
        Initialize database manager with connection parameters
        
        Args:
            connection_uri: PostgreSQL connection URI
        """
        self.connection_uri = connection_uri
        self.conn = None
    
    def initialize_db(self):
        """Initialize database connection and create tables if needed"""
        try:
            self.conn = psycopg2.connect(self.connection_uri)
            
            # Create tables if they don't exist
            with self.conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recent_matches (
                    match_id VARCHAR(100) PRIMARY KEY,
                    date DATE,
                    team VARCHAR(50),
                    opponent VARCHAR(50),
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
                    scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indexes if they don't exist
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_team ON recent_matches(team);
                CREATE INDEX IF NOT EXISTS idx_matches_date ON recent_matches(date);
                """)
                
                self.conn.commit()
                logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False
    
    def store_recent_matches(self, matches_df):
        """
        Store or update matches in database
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            Number of matches stored/updated
        """
        if self.conn is None:
            self.initialize_db()
            
        if self.conn is None:
            logger.error("Could not connect to database")
            return 0
            
        try:
            # Ensure DataFrame columns are lowercase to match DB
            matches_df.columns = [c.lower() for c in matches_df.columns]
            
            # Convert date strings to dates
            if 'date' in matches_df.columns and matches_df['date'].dtype == 'object':
                matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
            
            # Prepare data for insert/update
            required_columns = ['match_id', 'date', 'team', 'opponent']
            
            # Check if required columns exist
            for col in required_columns:
                if col not in matches_df.columns:
                    logger.error(f"Required column '{col}' missing from DataFrame")
                    return 0
            
            # Generate the SQL
            columns = [c for c in matches_df.columns if c != 'scrape_date']
            column_str = ', '.join(columns)
            values_placeholder = ', '.join(['%s'] * len(columns))
            
            # Create upsert query using ON CONFLICT
            query = f"""
            INSERT INTO recent_matches ({column_str})
            VALUES ({values_placeholder})
            ON CONFLICT (match_id) 
            DO UPDATE SET 
            """
            
            # Add all columns to update
            update_clauses = []
            for col in columns:
                if col != 'match_id':  # Don't update the PK
                    update_clauses.append(f"{col} = EXCLUDED.{col}")
            
            query += ', '.join(update_clauses)
            
            # Prepare values
            values = []
            for _, row in matches_df.iterrows():
                row_values = [row[col] if col in row else None for col in columns]
                values.append(row_values)
            
            # Execute the upsert
            with self.conn.cursor() as cursor:
                execute_values(cursor, query, values)
                self.conn.commit()
                
            row_count = len(values)
            logger.info(f"Stored/updated {row_count} matches in the database")
            return row_count
            
        except Exception as e:
            logger.error(f"Error storing matches in database: {e}")
            self.conn.rollback()
            return 0
    
    def get_recent_matches(self, team_name, limit=7):
        """
        Get recent matches for a team
        
        Args:
            team_name: Name of the team
            limit: Maximum number of matches to return
            
        Returns:
            DataFrame with recent matches
        """
        if self.conn is None:
            self.initialize_db()
            
        try:
            query = """
            SELECT * FROM recent_matches
            WHERE team = %s
            ORDER BY date DESC
            LIMIT %s
            """
            
            return pd.read_sql_query(query, self.conn, params=(team_name, limit))
            
        except Exception as e:
            logger.error(f"Error retrieving recent matches: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

class FBrefScraper:
    """Class to scrape football data from FBref.com with improved rate limit handling"""
    
    def __init__(self, base_url="https://fbref.com/en/comps/9/Premier-League-Stats"):
        """
        Initialize the scraper with configuration
        
        Args:
            base_url: Starting URL for Premier League stats
        """
        self.base_url = base_url
        self.data_dir = "data"
        self.temp_dir = os.path.join(self.data_dir, "temp")
        
        # Current season (based on date)
        current_month = datetime.now().month
        current_year = datetime.now().year
        self.season = current_year if current_month > 6 else current_year - 1
        
        # Initialize rate limit handler and requester
        self.rate_handler = RateLimitHandler(min_delay=10, max_delay=20)
        self.requester = PoliteRequester(
            cache_dir=os.path.join(self.data_dir, "cache"),
            max_cache_age=168  # Cache for 1 week
        )
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load previous state
        self.processed_teams = self.load_processed_teams()
    
    def extract_team_urls(self, html):
        """
        Extract team URLs from the league table
        
        Args:
            html: HTML content of the league table page
            
        Returns:
            List of team URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        team_urls = []
        
        try:
            # Try multiple selectors to find team links
            # First: standard league table
            tables = soup.select('table.stats_table')
            
            if not tables:
                # Try alternative selectors
                tables = soup.select('#results, #stats_squads_standard_for')
            
            if tables:
                # Process the first table that looks like a standings table
                for table in tables:
                    links = table.select('a[href*="/squads/"]')
                    if links:
                        for link in links:
                            href = link.get('href')
                            if href and '/squads/' in href and href not in team_urls:
                                team_urls.append(f"https://fbref.com{href}")
                        
                        if team_urls:
                            # If we found links, stop processing tables
                            break
            
            # If still no team links found, try a more generic approach
            if not team_urls:
                all_links = soup.select('a[href*="/squads/"]')
                seen_urls = set()
                
                for link in all_links:
                    href = link.get('href')
                    if href and '/squads/' in href and 'Stats' in href and href not in seen_urls:
                        team_urls.append(f"https://fbref.com{href}")
                        seen_urls.add(href)
            
            logger.info(f"Found {len(team_urls)} teams")
            return team_urls
            
        except Exception as e:
            logger.error(f"Error extracting team URLs: {e}")
            return []
    
    def extract_team_name(self, team_url):
        """
        Extract team name from team URL
        
        Args:
            team_url: Team URL
            
        Returns:
            Team name
        """
        parts = team_url.split('/')
        for part in reversed(parts):
            if part and part != "Stats":
                return part.replace('-', ' ')
        
        # Last resort: use the last part of the URL
        return parts[-1].replace('-', ' ').replace('Stats', '').strip()
    
    def parse_team_matches(self, team_url):
        """
        Parse match data for a team
        
        Args:
            team_url: URL of the team page
            
        Returns:
            DataFrame with match data or None if failed
        """
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {team_name} for {self.season} season")
        
        # Get team page HTML
        team_html = self.requester.fetch(team_url)
        if not team_html:
            logger.error(f"Failed to get team page HTML for {team_name}")
            return None
        
        # Parse matches and stats
        try:
            # Parse the main team page for matches
            soup = BeautifulSoup(team_html, 'html.parser')
            
            # Find the Scores & Fixtures table
            matches_table = None
            for table in soup.select('table'):
                if table.select('caption'):
                    caption_text = table.select_one('caption').text.strip()
                    if 'Scores & Fixtures' in caption_text:
                        matches_table = table
                        break
            
            if not matches_table:
                logger.error(f"Scores & Fixtures table not found for {team_name}")
                return None
            
            # Parse the table into a DataFrame
            matches_data = []
            
            # Get header row
            headers = []
            header_row = matches_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    # Get the column name from data-stat attribute or text
                    col_name = th.get('data-stat', th.text.strip())
                    headers.append(col_name)
            
            # Get data rows
            for row in matches_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                        
                        # Extract links where available
                        if cell.select_one('a'):
                            link = cell.select_one('a').get('href', '')
                            row_data[f"{col_name}_link"] = link
                
                if row_data:
                    matches_data.append(row_data)
            
            if not matches_data:
                logger.error(f"No match data found for {team_name}")
                return None
                
            # Convert to DataFrame
            matches_df = pd.DataFrame(matches_data)
            
            # Now find link to shooting stats
            shooting_link = None
            for link in soup.select('a'):
                href = link.get('href', '')
                if 'matchlogs/all_comps/shooting' in href:
                    shooting_link = f"https://fbref.com{href}"
                    break
            
            # Get shooting stats if available
            shooting_df = None
            if shooting_link:
                logger.info(f"Getting shooting stats for {team_name}")
                # Apply rate-limiting delay
                self.rate_handler.wait_before_request()
                
                shooting_html = self.requester.fetch(shooting_link)
                if shooting_html:
                    # Parse shooting data
                    shooting_df = self._parse_shooting_stats(shooting_html)
                else:
                    logger.warning(f"Failed to get shooting stats for {team_name}")
            else:
                logger.warning(f"No shooting stats link found for {team_name}")
            
            # Get standard stats for corners data
            standard_link = None
            for link in soup.select('a'):
                href = link.get('href', '')
                if 'matchlogs/all_comps/stats' in href:
                    standard_link = f"https://fbref.com{href}"
                    break
            
            # Get standard stats if available
            standard_df = None
            if standard_link:
                logger.info(f"Getting standard stats for {team_name}")
                # Apply rate-limiting delay
                self.rate_handler.wait_before_request()
                
                standard_html = self.requester.fetch(standard_link)
                if standard_html:
                    # Parse standard stats
                    standard_df = self._parse_standard_stats(standard_html)
                else:
                    logger.warning(f"Failed to get standard stats for {team_name}")
            else:
                logger.warning(f"No standard stats link found for {team_name}")
            
            # Combine all the data
            result_df = matches_df.copy()
            
            # Merge with shooting data if available
            if shooting_df is not None and not shooting_df.empty:
                # Ensure date format is consistent
                if 'date' in result_df.columns and 'date' in shooting_df.columns:
                    result_df = result_df.merge(shooting_df, on='date', how='left')
            
            # Merge with standard data if available
            if standard_df is not None and not standard_df.empty:
                # Ensure date format is consistent
                if 'date' in result_df.columns and 'date' in standard_df.columns:
                    result_df = result_df.merge(standard_df, on='date', how='left')
            
            # Add team information
            result_df['team'] = team_name
            result_df['season'] = self.season
            
            # Create match_id
            if 'date' in result_df.columns and 'opponent' in result_df.columns:
                result_df['match_id'] = result_df.apply(
                    lambda row: f"{row['date']}_{team_name}_{row['opponent']}".replace(' ', '_'),
                    axis=1
                )
            
            # Calculate points based on result
            if 'result' in result_df.columns:
                result_df['points'] = result_df['result'].apply(
                    lambda x: 3 if x == 'W' else (1 if x == 'D' else 0)
                )
            
            # Add home/away indicator
            if 'venue' in result_df.columns:
                result_df['is_home'] = result_df['venue'].apply(
                    lambda x: x.lower() == 'home' if pd.notna(x) else None
                )
            
            # Filter for Premier League matches only
            if 'comp' in result_df.columns:
                # Keep only Premier League matches
                premier_league_df = result_df[result_df['comp'].str.contains('Premier League', case=False, na=False)]
                
                if not premier_league_df.empty:
                    return premier_league_df
                    
                # If no PL matches found, return all matches
                return result_df
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error parsing matches for {team_name}: {e}")
            return None
    
    def _parse_shooting_stats(self, html):
        """
        Parse shooting stats from HTML
        
        Args:
            html: HTML content of shooting stats page
            
        Returns:
            DataFrame with shooting stats
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find shooting stats table
            stats_table = None
            for table in soup.select('table'):
                if table.get('id') and 'shooting' in table.get('id'):
                    stats_table = table
                    break
                
                # Backup method: look for shooting in caption
                if table.select('caption') and 'shooting' in table.select_one('caption').text.lower():
                    stats_table = table
                    break
            
            if not stats_table:
                return None
                
            # Parse the table
            data = []
            headers = []
            
            # Get header row
            header_row = stats_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    col_name = th.get('data-stat', th.text.strip())
                    # Remove any multi-level header indicators
                    if '.' in col_name:
                        col_name = col_name.split('.')[-1]
                    headers.append(col_name)
            
            # Get data rows
            for row in stats_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                
                if row_data:
                    data.append(row_data)
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            column_map = {
                'sh': 'sh',
                'sot': 'sot',
                'dist': 'dist',
                'fk': 'fk',
                'pk': 'pk',
                'pkatt': 'pkatt',
                'date': 'date'
            }
            
            # Keep only relevant columns
            keep_cols = ['date'] + [col for col in column_map.values() if col != 'date' and col in df.columns]
            if len(keep_cols) > 1:  # Must have date and at least one stat
                return df[keep_cols]
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing shooting stats: {e}")
            return None
    
    def _parse_standard_stats(self, html):
        """
        Parse standard stats from HTML
        
        Args:
            html: HTML content of standard stats page
            
        Returns:
            DataFrame with standard stats (focus on corners)
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find standard stats table
            stats_table = None
            for table in soup.select('table'):
                if table.get('id') and 'stats' in table.get('id'):
                    stats_table = table
                    break
                
                # Backup method: look for stats in caption
                if table.select('caption') and 'stats' in table.select_one('caption').text.lower():
                    stats_table = table
                    break
            
            if not stats_table:
                return None
                
            # Parse the table
            data = []
            headers = []
            
            # Get header row
            header_row = stats_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    col_name = th.get('data-stat', th.text.strip())
                    # Remove any multi-level header indicators
                    if '.' in col_name:
                        col_name = col_name.split('.')[-1]
                    headers.append(col_name)
            
            # Get data rows
            for row in stats_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                
                if row_data:
                    data.append(row_data)
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Look for corners column - it could be named 'ck', 'corners', or similar
            corner_cols = []
            for col in df.columns:
                if col.lower() in ['ck', 'corners', 'corner_kicks', 'corner']:
                    corner_cols.append(col)
            
            # If corners found, keep only date and corners
            if corner_cols and 'date' in df.columns:
                result_df = df[['date'] + corner_cols].copy()
                
                # Rename corner column to standard name
                for col in corner_cols:
                    result_df.rename(columns={col: 'corners_for'}, inplace=True)
                
                return result_df
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing standard stats: {e}")
            return None
    
    def scrape_in_batches(self, batch_size=3, batch_delay_minutes=10, max_teams=None):
        """
        Scrape teams in small batches with delays between batches
        
        Args:
            batch_size: Number of teams to process in each batch
            batch_delay_minutes: Delay between batches in minutes
            max_teams: Maximum number of teams to process (None for all)
            
        Returns:
            DataFrame with all scraped matches
        """
        # Create temporary directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Get league page with team links
        logger.info(f"Fetching Premier League teams from {self.base_url}")
        league_html = self.requester.fetch(self.base_url)
        
        if not league_html:
            logger.error("Failed to get league page HTML")
            return self._combine_existing_results()
        
        # Extract team URLs
        all_team_urls = self.extract_team_urls(league_html)
        
        if not all_team_urls:
            logger.error("No team URLs found")
            return self._combine_existing_results()
        
        # Filter out already processed teams
        team_urls = []
        for url in all_team_urls:
            team_name = self.extract_team_name(url)
            if team_name not in self.processed_teams:
                team_urls.append(url)
                logger.info(f"Added {team_name} to processing queue")
            else:
                logger.info(f"Skipping already processed team: {team_name}")
        
        # Apply max_teams limit if specified
        if max_teams is not None and len(team_urls) > max_teams:
            logger.info(f"Limiting to {max_teams} teams (from {len(team_urls)} available)")
            team_urls = team_urls[:max_teams]
        
        if not team_urls:
            logger.info("No new teams to process")
            return self._combine_existing_results()
        
        # Split into batches
        batches = [team_urls[i:i+batch_size] for i in range(0, len(team_urls), batch_size)]
        logger.info(f"Split {len(team_urls)} teams into {len(batches)} batches")
        
        # Process each batch
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} teams")
            batch_dfs = []
            
            for team_url in batch:
                team_name = self.extract_team_name(team_url)
                logger.info(f"Processing {team_name} (team {len(self.processed_teams) + 1})")
                
                try:
                    # Apply rate-limiting delay
                    self.rate_handler.wait_before_request()
                    
                    # Parse team matches
                    team_df = self.parse_team_matches(team_url)
                    
                    if team_df is not None and not team_df.empty:
                        logger.info(f"Successfully scraped {len(team_df)} matches for {team_name}")
                        batch_dfs.append(team_df)
                        
                        # Mark as processed and save state
                        self.processed_teams.append(team_name)
                        self.save_processed_teams()
                        
                        # Save individual team results
                        self._save_team_results(team_name, team_df)
                    else:
                        logger.warning(f"No valid data found for {team_name}")
                
                except Exception as e:
                    logger.error(f"Error processing {team_name}: {e}")
                
                # Add extra delay between teams (30-60 seconds)
                team_delay = random.uniform(30, 60)
                logger.info(f"Waiting {team_delay:.1f}s before next team...")
                time.sleep(team_delay)
            
            # Save batch results
            if batch_dfs:
                try:
                    batch_df = pd.concat(batch_dfs)
                    self._save_batch_results(batch_num, batch_df)
                except Exception as e:
                    logger.error(f"Error saving batch {batch_num} results: {e}")
            
            # Add delay between batches (if not the last batch)
            if batch_num < len(batches):
                delay_minutes = batch_delay_minutes + random.uniform(-1, 1)  # Add jitter
                delay_seconds = delay_minutes * 60
                logger.info(f"Completed batch {batch_num}. Waiting {delay_minutes:.1f} minutes before next batch...")
                time.sleep(delay_seconds)
        
        # Combine all results
        return self._combine_all_results()
    
    def _save_team_results(self, team_name, team_df):
        """
        Save results for a single team
        
        Args:
            team_name: Name of the team
            team_df: DataFrame with team's matches
        """
        try:
            # Clean team name for filename
            clean_name = team_name.replace(' ', '_').lower()
            filepath = os.path.join(self.temp_dir, f"team_{clean_name}.csv")
            
            # Save to CSV
            team_df.to_csv(filepath, index=False)
            logger.debug(f"Saved {len(team_df)} matches for {team_name} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving team results for {team_name}: {e}")
    
    def _save_batch_results(self, batch_num, batch_df):
        """
        Save results for a batch of teams
        
        Args:
            batch_num: Batch number
            batch_df: DataFrame with batch results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.temp_dir, f"batch_{batch_num}_{timestamp}.csv")
            
            # Save to CSV
            batch_df.to_csv(filepath, index=False)
            logger.info(f"Saved batch {batch_num} with {len(batch_df)} matches to {filepath}")
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")
    
    def _combine_existing_results(self):
        """
        Combine all existing team and batch files
        
        Returns:
            Combined DataFrame
        """
        all_files = []
        
        # Get all CSV files from temp directory
        for filename in os.listdir(self.temp_dir):
            if filename.endswith('.csv'):
                all_files.append(os.path.join(self.temp_dir, filename))
        
        # Also check for main results file
        main_file = os.path.join(self.data_dir, "recent_matches.csv")
        if os.path.exists(main_file):
            all_files.append(main_file)
        
        if not all_files:
            logger.warning("No existing result files found")
            return pd.DataFrame()
        
        # Combine all files
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine and deduplicate
        combined_df = pd.concat(dfs)
        
        # Deduplicate based on match_id if available
        if 'match_id' in combined_df.columns:
            before_count = len(combined_df)
            combined_df.drop_duplicates(subset=['match_id'], inplace=True)
            after_count = len(combined_df)
            logger.info(f"Removed {before_count - after_count} duplicate matches based on match_id")
        elif all(col in combined_df.columns for col in ['date', 'team', 'opponent']):
            before_count = len(combined_df)
            combined_df.drop_duplicates(subset=['date', 'team', 'opponent'], inplace=True)
            after_count = len(combined_df)
            logger.info(f"Removed {before_count - after_count} duplicate matches based on date+team+opponent")
        
        return combined_df
    
    def _combine_all_results(self):
        """
        Combine all results and save to main file
        
        Returns:
            Combined DataFrame
        """
        # Get combined results
        combined_df = self._combine_existing_results()
        
        if combined_df.empty:
            logger.warning("No results to combine")
            return combined_df
        
        # Standardize column names
        combined_df.columns = [c.lower() for c in combined_df.columns]
        
        # Add timestamp
        combined_df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save to main file
        main_file = os.path.join(self.data_dir, "recent_matches.csv")
        combined_df.to_csv(main_file, index=False)
        logger.info(f"Saved combined results with {len(combined_df)} matches to {main_file}")
        
        # Create a backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.data_dir, f"recent_matches_backup_{timestamp}.csv")
        combined_df.to_csv(backup_file, index=False)
        logger.debug(f"Created backup at {backup_file}")
        
        return combined_df
    
    def load_processed_teams(self):
        """
        Load previously processed teams
        
        Returns:
            List of processed team names
        """
        state_file = os.path.join(self.data_dir, "processed_teams.json")
        
        if not os.path.exists(state_file):
            return []
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            processed = state.get("processed_teams", [])
            logger.info(f"Loaded {len(processed)} previously processed teams")
            return processed
            
        except Exception as e:
            logger.warning(f"Failed to load processed teams: {e}")
            return []
    
    def save_processed_teams(self):
        """Save the current list of processed teams"""
        state_file = os.path.join(self.data_dir, "processed_teams.json")
        
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "processed_teams": self.processed_teams,
                "season": self.season
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f)
                
            logger.debug(f"Saved state with {len(self.processed_teams)} processed teams")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save processed teams: {e}")
            return False
    
    def reset_processed_teams(self):
        """Reset the list of processed teams"""
        self.processed_teams = []
        self.save_processed_teams()
        logger.info("Reset processed teams list")

def main():
    """Main function to run the optimized scraper"""
    parser = argparse.ArgumentParser(description="Optimized FBref scraper with rate limit handling")
    
    parser.add_argument("--batch-size", type=int, default=3, 
                        help="Number of teams to process in each batch (default: 3)")
    parser.add_argument("--batch-delay", type=int, default=10, 
                        help="Delay between batches in minutes (default: 10)")
    parser.add_argument("--max-teams", type=int, default=None,
                        help="Maximum number of teams to process (default: all)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset the list of processed teams")
    parser.add_argument("--db-uri", type=str, default=None,
                        help="PostgreSQL connection URI")
    
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = FBrefScraper()
        
        # Reset processed teams if requested
        if args.reset:
            scraper.reset_processed_teams()
        
        # Scrape in batches
        matches_df = scraper.scrape_in_batches(
            batch_size=args.batch_size,
            batch_delay_minutes=args.batch_delay,
            max_teams=args.max_teams
        )
        
        if matches_df.empty:
            logger.warning("No matches scraped")
            return 1
            
        logger.info(f"Successfully scraped {len(matches_df)} matches")
        
        # Store in database if URI provided
        if args.db_uri:
            db_manager = FBrefDatabaseManager(connection_uri=args.db_uri)
            db_manager.initialize_db()
            count = db_manager.store_recent_matches(matches_df)
            logger.info(f"Stored/updated {count} matches in the database")
            
            # Get recent matches for Liverpool as a test
            liverpool_matches = db_manager.get_recent_matches("Liverpool", 7)
            logger.info(f"Retrieved {len(liverpool_matches)} recent matches for Liverpool")
            
            # Close database connection
            db_manager.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())