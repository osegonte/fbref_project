#!/usr/bin/env python3
"""
FBref Match Stats Collector

This module takes fixture data from SofaScore daily CSV files and enriches it with 
detailed match statistics from FBref for each team's recent matches.

Usage:
  # Process all teams from daily files
  python fbref_stats_collector.py
  
  # Process just a few teams to verify the script works
  python fbref_stats_collector.py --max-teams 5
  
  # Use a specific input file instead of daily files
  python fbref_stats_collector.py --input-file all_matches_20250427_to_20250504.csv
"""

import os
import sys
import time
import random
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import argparse
from pathlib import Path
from io import StringIO
import re
from urllib.parse import urljoin
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/fbref_stats_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_stats_collector")

# User-agent rotation for avoiding detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
]

class RateLimitHandler:
    """Handles rate limiting with exponential backoff and cooldown periods."""
    
    def __init__(self, min_delay=8, max_delay=15, cooldown_threshold=3):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cooldown_threshold = cooldown_threshold
        self.rate_limited_count = 0
        self.last_request = 0
        self.domain_cooldown_until = 0

    def wait(self):
        """Wait an appropriate amount of time between requests."""
        now = time.time()
        
        # Check if we're in a cooldown period
        if now < self.domain_cooldown_until:
            to_wait = self.domain_cooldown_until - now
            logger.info(f"Domain cooldown: sleeping {to_wait:.1f}s")
            time.sleep(to_wait)
            now = time.time()
        
        # Calculate base delay with some randomness
        base = random.uniform(self.min_delay, self.max_delay)
        if self.rate_limited_count:
            base += min(self.rate_limited_count * 5, 30)
        
        # Enforce minimum delay between requests
        elapsed = now - self.last_request
        if elapsed < base:
            time.sleep(base - elapsed)
        
        self.last_request = time.time()

    def backoff(self):
        """Implement exponential backoff after rate limiting."""
        self.rate_limited_count += 1
        backoff = min(2 ** self.rate_limited_count, 90) * random.uniform(0.8, 1.2)
        
        # If we've been rate limited several times, enter a longer cooldown
        if self.rate_limited_count >= self.cooldown_threshold:
            cd = random.uniform(180, 300)  # 3-5 minute cooldown
            self.domain_cooldown_until = time.time() + cd
            logger.warning(f"Entering domain cooldown for {cd:.1f}s")
        
        logger.warning(f"Rate limited; backing off {backoff:.1f}s")
        time.sleep(backoff)

    def success(self):
        """Reset the rate limited count after a successful request."""
        if self.rate_limited_count > 0:
            self.rate_limited_count = max(0, self.rate_limited_count - 1)


class WebRequester:
    """Handles web requests with caching, rate limiting, and error handling."""
    
    def __init__(self, cache_dir="data/cache", cache_max_age=24, min_delay=8, max_delay=15):
        """Initialize the requester with caching and rate limiting."""
        self.cache_dir = cache_dir
        self.cache_max_age = cache_max_age
        self.rate_handler = RateLimitHandler(min_delay, max_delay)
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
    
    def _get_cache_path(self, url):
        """Generate a cache file path for a URL."""
        from hashlib import md5
        url_hash = md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.html")
    
    def fetch(self, url, use_cache=True, retries=3):
        """Fetch a URL with caching and rate limiting."""
        # Check cache first
        cache_path = self._get_cache_path(url)
        if use_cache and os.path.exists(cache_path):
            age = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age < self.cache_max_age:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        # If not in cache or too old, fetch from web
        for attempt in range(retries):
            # Wait appropriate time before request
            self.rate_handler.wait()
            
            # Use a random user agent
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            try:
                response = self.session.get(url, headers=headers, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(f"Rate limited for URL: {url}")
                    self.rate_handler.backoff()
                    continue
                
                # Handle other errors
                response.raise_for_status()
                
                # Get content
                content = response.text
                
                # Save to cache
                if use_cache:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # Mark as success for rate handler
                self.rate_handler.success()
                
                return content
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt + random.uniform(1, 5))  # Exponential backoff with jitter
        
        # If we've exhausted retries, raise an exception
        raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


class FBrefStatsCollector:
    """Collects detailed match statistics from FBref based on fixture data."""
    
    # Static mapping for major leagues
    LEAGUE_MAP = {
        "Premier League": {"id": "9", "name": "Premier League", "country": "England"},
        "La Liga": {"id": "12", "name": "La Liga", "country": "Spain"},
        "Bundesliga": {"id": "20", "name": "Bundesliga", "country": "Germany"},
        "Serie A": {"id": "11", "name": "Serie A", "country": "Italy"},
        "Ligue 1": {"id": "13", "name": "Ligue 1", "country": "France"},
        "Champions League": {"id": "8", "name": "Champions League", "country": "Europe"},
        "UEFA Champions League": {"id": "8", "name": "Champions League", "country": "Europe"},
        "Europa League": {"id": "19", "name": "Europa League", "country": "Europe"},
        "UEFA Europa League": {"id": "19", "name": "Europa League", "country": "Europe"},
        "Conference League": {"id": "882", "name": "Conference League", "country": "Europe"},
        "UEFA Europa Conference League": {"id": "882", "name": "Conference League", "country": "Europe"},
        "FA Cup": {"id": "45", "name": "FA Cup", "country": "England"},
        "EFL Cup": {"id": "47", "name": "EFL Cup", "country": "England"},
        "Copa del Rey": {"id": "569", "name": "Copa del Rey", "country": "Spain"},
        "DFB-Pokal": {"id": "81", "name": "DFB-Pokal", "country": "Germany"},
        "Coppa Italia": {"id": "79", "name": "Coppa Italia", "country": "Italy"},
        "Coupe de France": {"id": "61", "name": "Coupe de France", "country": "France"},
        "Eredivisie": {"id": "23", "name": "Eredivisie", "country": "Netherlands"},
        "Primeira Liga": {"id": "32", "name": "Primeira Liga", "country": "Portugal"},
        "Championship": {"id": "10", "name": "Championship", "country": "England"}
    }
    
    def __init__(self, 
                 output_dir="data/team_stats",
                 cache_dir="data/cache",
                 lookback=7, 
                 batch_size=3, 
                 delay_between_batches=60):
        """Initialize the collector with configuration parameters."""
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.lookback = lookback
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize requester
        self.requester = WebRequester(
            cache_dir=cache_dir,
            cache_max_age=24,  # Cache for 24 hours
            min_delay=8,       # Minimum delay between requests
            max_delay=15       # Maximum delay between requests
        )
        
        # Track league mappings
        self.league_map = {}
        
        # Track team IDs
        self.team_ids = {}
    
    def load_fixture_data(self, input_file=None, date_range=7, source_dir="sofascore_data/daily", max_teams=0):
        """
        Load fixture data from a file or daily CSV files.
        
        Args:
            input_file: Input CSV file path, if None, use daily files
            date_range: Number of days to load when using daily files
            source_dir: Directory with daily CSV files
            max_teams: Maximum number of teams to include (for testing)
            
        Returns:
            DataFrame with upcoming fixtures, optionally limited to max_teams
        """
        if input_file:
            logger.info(f"Loading fixture data from {input_file}")
            try:
                fixtures = pd.read_csv(input_file)
                logger.info(f"Loaded {len(fixtures)} fixtures from {input_file}")
            except Exception as e:
                logger.error(f"Error loading fixture data: {e}")
                return pd.DataFrame()
        else:
            # Load from daily files
            logger.info(f"Loading fixture data for the next {date_range} days" + 
                      (f" (limited to {max_teams} teams)" if max_teams > 0 else ""))
            
            # Determine date range
            today = datetime.now().date()
            end_date = today + timedelta(days=date_range)
            
            # Load from individual daily files
            logger.info(f"Loading from daily files in {source_dir}...")
            daily_files = []
            current_date = today
            
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                file_path = os.path.join(source_dir, f"matches_{date_str}.csv")
                
                if os.path.exists(file_path):
                    daily_files.append(file_path)
                
                current_date += timedelta(days=1)
            
            if not daily_files:
                logger.error("No fixture data found")
                return pd.DataFrame()
            
            # Load and combine data
            dfs = []
            for file in daily_files:
                try:
                    logger.info(f"Loading {file}")
                    df = pd.read_csv(file)
                    if 'date' not in df.columns:
                        df['date'] = os.path.basename(file).replace('matches_', '').replace('.csv', '')
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
            
            if not dfs:
                logger.error("No valid fixture data loaded")
                return pd.DataFrame()
                
            fixtures = pd.concat(dfs, ignore_index=True)
            
            # Filter for the date range
            fixtures['date'] = pd.to_datetime(fixtures['date']).dt.date
            fixtures = fixtures[(fixtures['date'] >= today) & (fixtures['date'] <= end_date)]
        
        # Limit to max_teams (randomly sampled from different leagues)
        if max_teams > 0:
            # Get unique leagues
            leagues = fixtures['league'].unique()
            
            # For each league, get some teams
            teams_per_league = max(1, max_teams // len(leagues))
            selected_teams = set()
            
            for league in leagues:
                league_fixtures = fixtures[fixtures['league'] == league]
                league_teams = set(league_fixtures['home_team'].tolist() + league_fixtures['away_team'].tolist())
                
                # Select random teams from this league
                if len(league_teams) > 0:
                    league_selection = random.sample(list(league_teams), min(teams_per_league, len(league_teams)))
                    selected_teams.update(league_selection)
                
                # Stop if we've reached max_teams
                if len(selected_teams) >= max_teams:
                    break
            
            # Make sure we don't exceed max_teams
            if len(selected_teams) > max_teams:
                selected_teams = set(list(selected_teams)[:max_teams])
            
            # Filter fixtures to only include the selected teams
            fixtures = fixtures[
                fixtures['home_team'].isin(selected_teams) | 
                fixtures['away_team'].isin(selected_teams)
            ]
            
            logger.info(f"Limited to {len(selected_teams)} teams: {', '.join(selected_teams)}")
        
        logger.info(f"Loaded {len(fixtures)} fixtures")
        return fixtures
    
    def extract_teams_and_leagues(self, fixtures):
        """Extract unique teams and their leagues from fixtures."""
        team_leagues = {}
        
        # Process teams
        for _, row in fixtures.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            league = row['league']
            country = row.get('country', 'Unknown')
            
            # Add to team_leagues dictionary
            for team in [home_team, away_team]:
                if team not in team_leagues:
                    team_leagues[team] = {
                        'league': league,
                        'country': country
                    }
        
        logger.info(f"Extracted {len(team_leagues)} unique teams from {len(fixtures)} fixtures")
        return team_leagues
    
    def map_league_to_fbref(self, league_name, country=None):
        """Map league names to FBref league IDs."""
        # Check if we already have this league mapped
        cache_key = f"{country}_{league_name}"
        if cache_key in self.league_map:
            return self.league_map[cache_key]
        
        # Try exact match in static map
        if league_name in self.LEAGUE_MAP:
            logger.info(f"League '{league_name}' found in static map")
            self.league_map[cache_key] = self.LEAGUE_MAP[league_name]
            return self.LEAGUE_MAP[league_name]
            
        # Try fuzzy matching
        for key, value in self.LEAGUE_MAP.items():
            if key.lower() in league_name.lower() or league_name.lower() in key.lower():
                logger.info(f"League '{league_name}' fuzzy matched to '{key}'")
                self.league_map[cache_key] = value
                return value
        
        # If no match found, use the first league that matches the country
        if country and country != "Unknown":
            for key, value in self.LEAGUE_MAP.items():
                if value['country'].lower() == country.lower():
                    logger.info(f"League '{league_name}' matched via country '{country}' to '{key}'")
                    self.league_map[cache_key] = value
                    return value
        
        # Fallback to Premier League
        logger.warning(f"Could not map league '{league_name}', using Premier League as fallback")
        fallback = self.LEAGUE_MAP["Premier League"]
        self.league_map[cache_key] = fallback
        return fallback
    
    def find_team_fbref_id(self, team_name, league_id):
        """Find FBref team ID by searching the league page."""
        # Check cache first
        cache_key = f"{team_name}_{league_id}"
        if cache_key in self.team_ids:
            return self.team_ids[cache_key]
        
        # Generate the league standings URL
        url = f"https://fbref.com/en/comps/{league_id}/stats/squads/for/"
        
        try:
            # Fetch the league page
            html = self.requester.fetch(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find team links
            for a in soup.select('a[href*="/squads/"]'):
                if team_name.lower() in a.text.lower():
                    # Extract team ID from href
                    parts = a['href'].split('/')
                    for i, part in enumerate(parts):
                        if part == 'squads' and i+1 < len(parts):
                            team_id = parts[i+1]
                            self.team_ids[cache_key] = team_id
                            return team_id
            
            # If no exact match, try partial match
            best_match = None
            best_similarity = 0.5  # Threshold for partial match
            
            for a in soup.select('a[href*="/squads/"]'):
                # Calculate similarity between team names
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, team_name.lower(), a.text.lower()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    parts = a['href'].split('/')
                    for i, part in enumerate(parts):
                        if part == 'squads' and i+1 < len(parts):
                            best_match = parts[i+1]
            
            if best_match:
                logger.info(f"Partial match for '{team_name}' with similarity {best_similarity:.2f}")
                self.team_ids[cache_key] = best_match
                return best_match
            
            logger.warning(f"No team ID found for '{team_name}' in league {league_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding team ID for '{team_name}': {e}")
            return None
    
    def get_team_match_data(self, team_name, league_info, lookback=7):
        """Get detailed match data for a team."""
        league_id = league_info['id']
        league_name = league_info['name']
        
        try:
            # Try different approaches in sequence
            logger.info(f"Getting match data for {team_name} in {league_name}")
            
            # Approach 1: Try to get data from team page (most complete)
            df = self.get_team_data_from_team_page(team_name, league_id, league_name, lookback)
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} matches for {team_name} from team page")
                return df
            
            # Approach 2: Try to get data from league schedule page
            df = self.get_team_data_from_schedule(team_name, league_id, league_name, lookback)
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} matches for {team_name} from league schedule")
                return df
            
            logger.warning(f"Could not find any match data for {team_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting match data for {team_name}: {e}")
            return None
    
    def get_team_data_from_team_page(self, team_name, league_id, league_name, lookback=7):
        """Get team data from the team's FBref page."""
        # Get team ID
        team_id = self.find_team_fbref_id(team_name, league_id)
        if not team_id:
            logger.warning(f"Could not find team ID for {team_name}")
            return None
        
        # Generate team URL
        team_url = f"https://fbref.com/en/squads/{team_id}/{team_name.replace(' ', '-')}-Stats"
        
        try:
            # Fetch team page
            html = self.requester.fetch(team_url)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find and fetch matches page
            matches_link = None
            for a in soup.select('a'):
                if 'Scores & Fixtures' in a.text:
                    matches_link = a['href']
                    break
            
            if not matches_link:
                logger.warning(f"Could not find matches link for {team_name}")
                return None
            
            # Fetch matches page
            matches_url = urljoin('https://fbref.com', matches_link)
            matches_html = self.requester.fetch(matches_url)
            
            # Parse tables
            tables = pd.read_html(StringIO(matches_html))
            if not tables:
                logger.warning(f"No tables found in matches page for {team_name}")
                return None
            
            # The first table should be the matches
            df = tables[0]
            
            # Normalize column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Filter to completed matches (with a score)
            if 'Score' in df.columns:
                df = df[df['Score'].notna() & (df['Score'] != '')]
            
            if df.empty:
                logger.warning(f"No valid matches found for {team_name}")
                return None
            
            # Process data
            df['team'] = team_name
            
            # Handle opponent
            if 'Opponent' in df.columns:
                df['opponent'] = df['Opponent']
            
            # Handle venue
            if 'Venue' in df.columns:
                df['is_home'] = df['Venue'] == 'Home'
                df['venue'] = df['Venue']
            
            # Extract goals
            if 'GF' in df.columns and 'GA' in df.columns:
                df['gf'] = pd.to_numeric(df['GF'], errors='coerce')
                df['ga'] = pd.to_numeric(df['GA'], errors='coerce')
            elif 'Score' in df.columns:
                # Parse from score
                df['Score'] = df['Score'].astype(str)
                goals = df['Score'].str.split('–', expand=True)
                
                # Determine home and away
                is_home = df['venue'] == 'Home'
                df['gf'] = pd.to_numeric(goals[0], errors='coerce').where(is_home, pd.to_numeric(goals[1], errors='coerce'))
                df['ga'] = pd.to_numeric(goals[1], errors='coerce').where(is_home, pd.to_numeric(goals[0], errors='coerce'))
            
            # Calculate result and points
            df['result'] = np.where(df['gf'] > df['ga'], 'W',
                               np.where(df['gf'] < df['ga'], 'L', 'D'))
            df['points'] = df['result'].map({'W': 3, 'D': 1, 'L': 0})
            
            # Extract expected goals (if available)
            if 'xG' in df.columns and 'xGA' in df.columns:
                df['xg'] = pd.to_numeric(df['xG'], errors='coerce')
                df['xga'] = pd.to_numeric(df['xGA'], errors='coerce')
            
            # Extract shooting stats (if available)
            if 'Sh' in df.columns:
                df['sh'] = pd.to_numeric(df['Sh'], errors='coerce')
            if 'SoT' in df.columns:
                df['sot'] = pd.to_numeric(df['SoT'], errors='coerce')
            if 'Dist' in df.columns:
                df['dist'] = pd.to_numeric(df['Dist'], errors='coerce')
            if 'FK' in df.columns:
                df['fk'] = pd.to_numeric(df['FK'], errors='coerce')
            if 'PK' in df.columns:
                df['pk'] = pd.to_numeric(df['PK'], errors='coerce')
            if 'PKatt' in df.columns:
                df['pkatt'] = pd.to_numeric(df['PKatt'], errors='coerce')
            
            # Extract possession (if available)
            if 'Poss' in df.columns:
                df['possession'] = pd.to_numeric(df['Poss'], errors='coerce')
            
            # Add metadata
            df['comp'] = league_name if 'Comp' not in df.columns else df['Comp']
            df['league_id'] = league_id
            df['league_name'] = league_name
            if 'Round' in df.columns:
                df['round'] = df['Round']
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Handle missing dates
            if df['date'].isna().any():
                logger.warning(f"Some dates could not be parsed for {team_name}")
                df = df.dropna(subset=['date'])
            
            # Generate match_id
            df['match_id'] = (
                df['date'].dt.strftime('%Y%m%d') + '_' +
                df['team'].str.replace(r'\W+', '', regex=True) + '_' +
                df['opponent'].str.replace(r'\W+', '', regex=True)
            )
            
            # Add scrape timestamp
            df['scrape_date'] = datetime.now()
            
            # Extract season
            # Estimate season from date (assuming season starts in August)
            df['season'] = np.where(
                df['date'].dt.month >= 8,
                df['date'].dt.year,
                df['date'].dt.year - 1
            )
            
            # Take only the most recent matches
            df = df.sort_values('date', ascending=False).head(lookback)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting team data from team page for {team_name}: {e}")
            return None
    
    def get_team_data_from_schedule(self, team_name, league_id, league_name, lookback=7):
        """Get team data from league schedule page."""
        # Generate the league schedule URL
        pretty = league_name.replace(" ", "-")
        url = f"https://fbref.com/en/comps/{league_id}/schedule/{pretty}-Scores-and-Fixtures"
        
        try:
            # Fetch the schedule page
            html = self.requester.fetch(url)
            
            # Parse tables from the HTML
            tables = pd.read_html(StringIO(html))
            if not tables:
                logger.warning(f"No tables found in league schedule for {league_name}")
                return None
            
            # The first table should be the schedule
            df = tables[0]
            
            # Normalize column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Filter to matches involving the team
            is_home = df['Home'] == team_name
            is_away = df['Away'] == team_name
            team_matches = df[is_home | is_away].copy()
            
            if team_matches.empty:
                logger.warning(f"No matches found for {team_name} in {league_name} schedule")
                return None
            
            # Process data
            team_matches['team'] = team_name
            team_matches['opponent'] = team_matches['Away'].where(is_home, team_matches['Home'])
            team_matches['is_home'] = is_home
            team_matches['venue'] = team_matches['is_home'].map({True: 'Home', False: 'Away'})
            
            # Extract goals (if available)
            if 'Score' in team_matches.columns:
                team_matches['Score'] = team_matches['Score'].astype(str)
                # Remove non-numeric scores (e.g., "Match Postponed")
                team_matches = team_matches[team_matches['Score'].str.contains('–')]
                
                # Split score column
                goals = team_matches['Score'].str.split('–', expand=True)
                goals[0] = pd.to_numeric(goals[0], errors='coerce')
                goals[1] = pd.to_numeric(goals[1], errors='coerce')
                
                # Assign goals for and against
                team_matches['gf'] = goals[0].where(is_home, goals[1])
                team_matches['ga'] = goals[1].where(is_home, goals[0])
                
                # Calculate result and points
                team_matches['result'] = np.where(team_matches['gf'] > team_matches['ga'], 'W',
                                              np.where(team_matches['gf'] < team_matches['ga'], 'L', 'D'))
                team_matches['points'] = team_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
            
            # Extract expected goals (if available)
            if 'xG' in team_matches.columns and 'xGA' in team_matches.columns:
                team_matches['xg'] = pd.to_numeric(team_matches['xG'], errors='coerce')
                team_matches['xga'] = pd.to_numeric(team_matches['xGA'], errors='coerce')
            
            # Extract shooting stats (if available)
            for stat in ['Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']:
                if stat in team_matches.columns:
                    team_matches[stat.lower()] = pd.to_numeric(team_matches[stat], errors='coerce')
            
            # Extract possession (if available)
            if 'Poss' in team_matches.columns:
                team_matches['possession'] = pd.to_numeric(team_matches['Poss'], errors='coerce')
            
            # Add metadata
            team_matches['comp'] = league_name
            team_matches['league_id'] = league_id
            team_matches['league_name'] = league_name
            if 'Round' in team_matches.columns:
                team_matches['round'] = team_matches['Round']
            
            # Convert date to datetime
            team_matches['date'] = pd.to_datetime(team_matches['Date'], errors='coerce')
            
            # Handle missing dates
            if team_matches['date'].isna().any():
                logger.warning(f"Some dates could not be parsed for {team_name}")
                team_matches = team_matches.dropna(subset=['date'])
            
            # Generate match_id
            team_matches['match_id'] = (
                team_matches['date'].dt.strftime('%Y%m%d') + '_' +
                team_matches['team'].str.replace(r'\W+', '', regex=True) + '_' +
                team_matches['opponent'].str.replace(r'\W+', '', regex=True)
            )
            
            # Add scrape timestamp
            team_matches['scrape_date'] = datetime.now()
            
            # Extract season
            # Estimate season from date (assuming season starts in August)
            team_matches['season'] = np.where(
                team_matches['date'].dt.month >= 8,
                team_matches['date'].dt.year,
                team_matches['date'].dt.year - 1
            )
            
            # Take only the most recent matches
            team_matches = team_matches.sort_values('date', ascending=False).head(lookback)
            
            return team_matches
            
        except Exception as e:
            logger.error(f"Error getting team data from schedule for {team_name}: {e}")
            return None
            
    def process_teams_in_batches(self, team_leagues):
        """Process teams in batches to avoid rate limiting."""
        all_data = []
        teams = list(team_leagues.keys())
        
        # Process in batches
        for i in range(0, len(teams), self.batch_size):
            batch = teams[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(teams)-1)//self.batch_size + 1} with {len(batch)} teams")
            
            batch_data = []
            for team in batch:
                league_info = self.map_league_to_fbref(
                    team_leagues[team]['league'],
                    team_leagues[team]['country']
                )
                
                df = self.get_team_match_data(team, league_info, self.lookback)
                if df is not None and not df.empty:
                    batch_data.append(df)
            
            # Add batch data to all data
            if batch_data:
                all_data.extend(batch_data)
                
            # Sleep between batches to avoid rate limiting
            if i + self.batch_size < len(teams):
                sleep_time = random.uniform(self.delay_between_batches * 0.8, self.delay_between_batches * 1.2)
                logger.info(f"Sleeping for {sleep_time:.1f} seconds between batches")
                time.sleep(sleep_time)
        
        # Combine all data
        if not all_data:
            logger.error("No match data collected for any team")
            return pd.DataFrame()
            
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected {len(combined)} total matches for {len(all_data)} teams")
        return combined
    
    def save_output(self, match_data):
        """Save the collected match data to CSV files."""
        if match_data.empty:
            logger.error("No data to save")
            return None
            
        # Create output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"team_match_stats_{timestamp}.csv")
        
        # Save to CSV
        match_data.to_csv(output_file, index=False)
        logger.info(f"Saved {len(match_data)} matches to {output_file}")
        
        # Also save by league
        leagues = match_data['league_name'].unique()
        for league in leagues:
            league_data = match_data[match_data['league_name'] == league]
            league_file = os.path.join(self.output_dir, f"{league.replace(' ', '_')}_{timestamp}.csv")
            league_data.to_csv(league_file, index=False)
            logger.info(f"Saved {len(league_data)} {league} matches to {league_file}")
        
        return output_file

    def run(self, input_file=None, date_range=7, source_dir="sofascore_data/daily", max_teams=0):
        """Run the full collection process."""
        # Step 1: Load fixture data
        fixtures = self.load_fixture_data(input_file, date_range, source_dir, max_teams)
        if fixtures.empty:
            logger.error("No fixtures found, aborting")
            return None
        
        # Step 2: Extract teams and leagues
        team_leagues = self.extract_teams_and_leagues(fixtures)
        if not team_leagues:
            logger.error("No teams extracted, aborting")
            return None
        
        # Step 3: Process teams in batches
        match_data = self.process_teams_in_batches(team_leagues)
        if match_data.empty:
            logger.error("No match data collected, aborting")
            return None
        
        # Step 4: Save output
        output_file = self.save_output(match_data)
        return output_file

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="FBref Match Stats Collector")
    parser.add_argument("--input-file", help="Input CSV file with fixture data")
    parser.add_argument("--daily-dir", default="sofascore_data/daily", help="Directory with daily CSV files")
    parser.add_argument("--output-dir", default="data/team_stats", help="Directory for output data")
    parser.add_argument("--cache-dir", default="data/cache", help="Directory for HTTP cache")
    parser.add_argument("--lookback", type=int, default=7, help="Number of past matches to collect per team")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of teams to process in a batch")
    parser.add_argument("--batch-delay", type=int, default=60, help="Delay between batches in seconds")
    parser.add_argument("--max-teams", type=int, default=0, help="Max teams to process (0 for all)")
    parser.add_argument("--date-range", type=int, default=7, help="Number of days to consider")
    
    args = parser.parse_args()
    
    try:
        # Initialize collector
        collector = FBrefStatsCollector(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            lookback=args.lookback,
            batch_size=args.batch_size,
            delay_between_batches=args.batch_delay
        )
        
        # Run collector
        output_file = collector.run(
            input_file=args.input_file,
            date_range=args.date_range,
            source_dir=args.daily_dir,
            max_teams=args.max_teams
        )
        
        if output_file:
            print(f"\nSuccess! Match data collected and saved to: {output_file}")
            return 0
        else:
            print("\nFailed to collect match data. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())