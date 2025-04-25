"""FBref scraping module with improved error handling and async support."""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, Tag
from unidecode import unidecode

from http_utils import fetch, soupify
from database_utils import upsert, Match, Player
from config import BASE_URL, LEAGUES, CACHE_TTL

# Configure logging
logger = logging.getLogger("fbref_toolkit.scraper")

class ScraperError(Exception):
    """Base class for scraper exceptions."""
    pass

class ParseError(ScraperError):
    """Exception raised when parsing fails."""
    pass

class DataError(ScraperError):
    """Exception raised when data validation fails."""
    pass

###############################################################################
# URL Generation                                                              #
###############################################################################

def season_schedule_url(league_id: str, season: int) -> str:
    """Generate URL for a league's season schedule."""
    return f"{BASE_URL}/en/comps/{league_id}/{season}-{season+1}/schedule/{season}-{season+1}-Scores-and-Fixtures"

def league_table_url(league_id: str, season: int) -> str:
    """Generate URL for a league's standings table."""
    return f"{BASE_URL}/en/comps/{league_id}/{season}-{season+1}/standings/"

def tidy_fbref_table(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Flatten multi-index columns, drop header separators, dedupe names, cast numerics.
    
    Args:
        df: DataFrame to clean
        numeric_cols: Optional list of columns to convert to numeric
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
        
    # 1) flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # Get the most informative level for each column
        flat_cols = []
        for col in df.columns:
            # Skip empty strings and use the most specific non-empty name
            names = [name for name in col if name and not pd.isna(name) and str(name).strip()]
            flat_cols.append(names[-1] if names else col[-1])
        df.columns = flat_cols

    # 2) drop separator rows (where ranking column is NaN or has specific markers)
    rank_col = next((c for c in df.columns if isinstance(c, str) and c.lower().startswith(("rk", "rank"))), None)
    if rank_col:
        df = df.dropna(subset=[rank_col])
        # Also drop rows where rank contains separators like '-' or 'Matches'
        if df[rank_col].dtype == object:
            df = df[~df[rank_col].astype(str).str.match(r'^[-–—]+$|^Matcheseam_url(team_id: str, season: int) -> str:
    """Generate URL for a team's page."""
    return f"{BASE_URL}/en/squads/{team_id}/{season}-{season+1}/matchlogs/all_comps/schedule/"

def player_url(player_id: str) -> str:
    """Generate URL for a player's page."""
    return f"{BASE_URL}/en/players/{player_id}/"

###############################################################################
# Table cleaning utilities                                                   #
###############################################################################

def t)]

    # 3) deduplicate column names
    counts: Dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        c_str = str(c)
        if c_str in counts:
            counts[c_str] += 1
            new_cols.append(f"{c_str}_{counts[c_str]}")
        else:
            counts[c_str] = 0
            new_cols.append(c_str)
    df.columns = [unidecode(str(col).lower().strip()) for col in new_cols]

    # 4) cast numerics and handle percentage columns
    if numeric_cols is None:
        # Try to auto-detect numeric columns (skip date-like columns)
        numeric_cols = []
        for c in df.columns:
            if c.lower() in ('date', 'match_date', 'datetime'):
                continue
            # Check if column contains mostly numeric data
            sample = df[c].dropna().head(5)
            if len(sample) > 0 and all(isinstance(x, (int, float)) or 
                                     (isinstance(x, str) and x.replace('.', '', 1).replace('-', '', 1).isdigit()) 
                                     for x in sample):
                numeric_cols.append(c)
    
    for col in numeric_cols:
        if col in df.columns:
            # Clean percentage values
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 5) Convert date columns to datetime
    date_cols = [c for c in df.columns if c.lower() in ('date', 'match_date', 'datetime')]
    for col in date_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df.reset_index(drop=True)


def standardize_team_names(df: pd.DataFrame, name_col: str = 'team') -> pd.DataFrame:
    """Standardize team names for consistency across different sources.
    
    Args:
        df: DataFrame with team names
        name_col: Column containing team names
        
    Returns:
        DataFrame with standardized team names
    """
    if name_col not in df.columns:
        return df
        
    # Common name variations mapping
    replacements = {
        "manchester utd": "manchester united",
        "manchester city": "man city",
        "tottenham": "tottenham hotspur",
        "wolves": "wolverhampton",
        "brighton": "brighton & hove albion",
        "newcastle": "newcastle united",
        "leeds": "leeds united",
        "norwich": "norwich city",
        "leicester": "leicester city",
        "west ham": "west ham united",
        "aston villa": "aston villa",
        "everton fc": "everton",
        "southampton fc": "southampton",
        "chelsea fc": "chelsea",
        "arsenal fc": "arsenal",
        "liverpool fc": "liverpool",
        "atletico madrid": "atlético madrid",
        "atletico": "atlético madrid",
        "real madrid cf": "real madrid",
        "fc barcelona": "barcelona",
        "barca": "barcelona",
        "bayern munich": "bayern münchen",
        "bayern": "bayern münchen",
        "borussia dortmund": "dortmund",
        "paris saint-germain": "psg",
        "paris saint germain": "psg",
        "paris sg": "psg",
        "juventus fc": "juventus",
        "ac milan": "milan",
        "internazionale": "inter",
        "inter milan": "inter",
    }
    
    # Apply standardization
    df[name_col] = df[name_col].str.lower().map(
        lambda x: replacements.get(x, x) if x in replacements else x
    )
    
    return df

###############################################################################
# Scraping functions                                                         #
###############################################################################

def read_table(html: str, caption_kw: str) -> pd.DataFrame:
    """Extract and clean a table from HTML based on caption keyword.
    
    Args:
        html: HTML content
        caption_kw: Keyword to find in table caption
        
    Returns:
        Parsed and cleaned DataFrame
        
    Raises:
        ParseError: If table cannot be found or parsed
    """
    try:
        df = pd.read_html(html, match=caption_kw, flavor="lxml")[0]
        return tidy_fbref_table(df)
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not read table with caption {caption_kw}: {e}")
        raise ParseError(f"Failed to extract table with caption '{caption_kw}': {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error reading table with caption {caption_kw}: {e}")
        raise ParseError(f"Unexpected error extracting table: {str(e)}")


def get_squad_urls(league_id: str, season: int) -> List[str]:
    """Get URLs for all team squad pages in a league season.
    
    Args:
        league_id: League ID from FBref
        season: Season year (starting year)
        
    Returns:
        List of team URLs
        
    Raises:
        ScraperError: If team URLs cannot be extracted
    """
    url = season_schedule_url(league_id, season)
    try:
        html = fetch(url)
        soup = soupify(html)
        table = soup.select_one("table.stats_table")
        
        if not table:
            raise ScraperError(f"Could not find teams table for league {league_id}, season {season}")
            
        hrefs = {a["href"] for a in table.select("a") if "/squads/" in a.get("href", "")}
        return sorted(f"{BASE_URL}{h}" for h in hrefs)
    except Exception as e:
        logger.error(f"Failed to get squad URLs for league {league_id}, season {season}: {e}")
        raise ScraperError(f"Failed to extract team URLs: {str(e)}")


def scrape_team(team_url: str, league_name: str, season: int, recent_only: bool = False) -> pd.DataFrame:
    """Scrape team match data from FBref.
    
    Args:
        team_url: URL to team's season page
        league_name: Name of the league (to filter matches)
        season: Season year (starting year)
        recent_only: If True, only get last 7 matches
        
    Returns:
        DataFrame with match data
        
    Raises:
        ScraperError: If scraping fails
    """
    logger.info(f"Scraping team data from {team_url}")
    
    try:
        html = fetch(team_url)
        soup = soupify(html)

        # Get team name from URL or page title
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        page_title = soup.find("title")
        if page_title:
            # Extract team name from title like "Manchester City Stats..."
            title_parts = page_title.text.split(" Stats")
            if title_parts:
                team_name = title_parts[0]

        # Extract team ID from URL
        team_id = None
        parts = team_url.split("/")
        for i, part in enumerate(parts):
            if part == "squads" and i+1 < len(parts):
                team_id = parts[i+1]
                break

        # scores & fixtures
        matches = read_table(html, "Scores & Fixtures")
        if matches.empty:
            logger.warning(f"No matches found for {team_name}")
            return pd.DataFrame()
            
        # Filter for specified league
        matches = matches[matches["comp"].str.contains(league_name, case=False)]
        if matches.empty:
            logger.warning(f"No {league_name} matches found for {team_name}")
            return pd.DataFrame()

        # optional recent slice
        if recent_only:
            matches = matches.tail(7)

        # Extract additional stats tables
        extra_stats = _extract_extra_stats(soup, team_url)
            
        # Merge all data
        df = matches.copy()
        for stat_name, extra_df in extra_stats.items():
            if not extra_df.empty:
                try:
                    df = df.merge(extra_df, on="date", how="left")
                    logger.debug(f"Merged {stat_name} stats for {team_name}")
                except Exception as e:
                    logger.warning(f"Error merging {stat_name} data: {e}")

        # Add team and season info
        df = df.assign(
            team=unidecode(team_name), 
            team_id=team_id,
            season=season,
            scrape_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Create unique match ID
        df["match_id"] = (
            df["date"].astype(str) + "_" + 
            df["team"].str.replace(" ", "_", regex=False) + "_" + 
            df["opponent"].str.replace(" ", "_", regex=False)
        )
        
        # Add calculated columns for analysis
        df["is_home"] = df["venue"] == "Home"
        df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})
        
        # Clean up any empty strings in numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].replace('', pd.NA)
        
        return df
    
    except ParseError as e:
        # Re-raise with more context
        raise ScraperError(f"Failed to parse team data from {team_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error scraping team {team_url}: {e}", exc_info=True)
        raise ScraperError(f"Unexpected error scraping team: {str(e)}")


def _extract_extra_stats(soup: BeautifulSoup, base_url: str) -> Dict[str, pd.DataFrame]:
    """Extract additional statistics tables from team page.
    
    Args:
        soup: BeautifulSoup object of the team page
        base_url: Base URL for the team
        
    Returns:
        Dictionary of DataFrames with extra stats
    """
    extra_stats = {}
    
    # Define the stats we want to extract
    stats_types = {
        "shooting": ["date", "sh", "sot", "dist", "fk", "pk", "pkatt"],
        "passing": ["date", "cmp", "att", "cmp_pct", "progressive_passes"],
        "possession": ["date", "poss", "touches", "def_pen", "def_3rd", "mid_3rd", "att_3rd"],
        "misc": ["date", "cards_yellow", "cards_red", "fouls", "offsides", "crosses", "interceptions"],
        "defense": ["date", "tackles", "tackles_won", "blocks", "clearances"],
        "keeper": ["date", "saves", "clean_sheets", "psxg", "psxg_net"],
    }
    
    # Find links to these stat pages
    for stat_key, columns in stats_types.items():
        try:
            # Look for links with this stat type in URL
            links = [f"{BASE_URL}{a['href']}" for a in soup.find_all("a", href=True) 
                    if f"all_comps/{stat_key}" in a["href"]]
            
            if not links:
                continue
                
            # Fetch and parse the first matching link
            html = fetch(links[0])
            extra_df = read_table(html, stat_key.split("/")[-1])
            
            # Keep only the columns we're interested in
            keep_cols = [c for c in columns if c in extra_df.columns]
            if keep_cols:
                extra_stats[stat_key] = extra_df[keep_cols]
        except Exception as e:
            logger.warning(f"Error fetching {stat_key} data: {e}")
    
    return extra_stats


def scrape_player_stats(team_url: str, season: int) -> pd.DataFrame:
    """Scrape player statistics for a team.
    
    Args:
        team_url: URL to team's season page
        season: Season year
        
    Returns:
        DataFrame with player statistics
        
    Raises:
        ScraperError: If scraping fails
    """
    logger.info(f"Scraping player data from {team_url}")
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    
    try:
        html = fetch(team_url)
        soup = soupify(html)
        
        # Find player stats tables - look for the standard stats table first
        stats_types = [
            ("stats_standard", "Standard Stats"),
            ("stats_shooting", "Shooting"),
            ("stats_passing", "Passing"),
            ("stats_defense", "Defensive Actions"),
            ("stats_possession", "Possession"),
            ("stats_keeper", "Goalkeeping")
        ]
        
        all_player_data = []
        
        for stats_id, caption in stats_types:
            try:
                # Try to find the table by ID first, then by caption
                table = soup.find("table", id=stats_id)
                if not table:
                    tables = soup.find_all("caption")
                    for cap in tables:
                        if caption in cap.text:
                            table = cap.find_parent("table")
                            break
                    
                if table:
                    df = pd.read_html(str(table))[0]
                    df = tidy_fbref_table(df)
                    
                    # Add type of statistics
                    df["stats_type"] = caption
                    
                    # Add team name
                    df["team"] = unidecode(team_name)
                    
                    # Add season
                    df["season"] = season
                    
                    # Add scrape date
                    df["scrape_date"] = datetime.now().strftime("%Y-%m-%d")
                    
                    # Create player ID based on name and team
                    if "player" in df.columns:
                        df["player_id"] = (
                            df["player"].str.lower().str.replace(" ", "_") + 
                            "_" + team_name.lower().replace(" ", "_") + 
                            f"_{season}"
                        )
                    
                    all_player_data.append(df)
            except Exception as e:
                logger.warning(f"Error scraping {caption} player data for {team_name}: {e}")
        
        # Combine all stats types into one DataFrame
        if all_player_data:
            all_players = pd.concat(all_player_data, ignore_index=True)
            return all_players
        else:
            logger.warning(f"No player data found for {team_name}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error scraping player data for {team_name}: {e}", exc_info=True)
        raise ScraperError(f"Failed to scrape player stats: {str(e)}")


def scrape_league_table(league_id: str, season: int) -> pd.DataFrame:
    """Scrape current league standings table.
    
    Args:
        league_id: League ID from FBref
        season: Season year
        
    Returns:
        DataFrame with league table data
        
    Raises:
        ScraperError: If scraping fails
    """
    url = league_table_url(league_id, season)
    try:
        html = fetch(url, cache_ttl=3600)  # Shorter cache for standings (1 hour)
        
        # Try to get main standings table
        df = pd.read_html(html, match="Regular Season")[0]
        df = tidy_fbref_table(df)
        
        # Add league and season info
        for league_name, data in LEAGUES.items():
            if data["id"] == league_id:
                df["league"] = league_name
                df["country"] = data["country"]
                break
        
        df["season"] = season
        
        # Add scrape date
        df["scrape_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df
    except Exception as e:
        logger.error(f"Error scraping league table for league ID {league_id}, season {season}: {e}")
        raise ScraperError(f"Failed to scrape league table: {str(e)}")


def scrape_head_to_head(team1: str, team2: str, num_matches: int = 10) -> pd.DataFrame:
    """Scrape head-to-head match history between two teams.
    
    Args:
        team1: First team name
        team2: Second team name
        num_matches: Number of recent matches to fetch
        
    Returns:
        DataFrame with head-to-head match data
        
    Raises:
        ScraperError: If scraping fails
    """
    # For this we need to search
    search_url = f"{BASE_URL}/en/search/matches/?from=matchlogs&q={team1}+vs+{team2}"
    
    try:
        html = fetch(search_url, cache_ttl=86400)  # 1 day cache
        soup = soupify(html)
        
        # Find the match results table
        matches_section = soup.find("div", {"id": "matches"})
        if not matches_section:
            logger.warning(f"No head-to-head data found for {team1} vs {team2}")
            return pd.DataFrame()
        
        # Extract match data
        df = pd.read_html(str(matches_section))[0]
        df = tidy_fbref_table(df)
        
        # Take only the most recent matches
        if len(df) > num_matches:
            df = df.head(num_matches)
            
        # Add metadata
        df["team1"] = team1
        df["team2"] = team2
        df["scrape_date"] = datetime.now().strftime("%Y-%m-%d")
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping head-to-head data for {team1} vs {team2}: {e}")
        raise ScraperError(f"Failed to scrape head-to-head data: {str(e)}")

###############################################################################
# Async scraping functions                                                   #
###############################################################################

async def fetch_async(url: str, session: aiohttp.ClientSession, timeout: int = 30) -> str:
    """Fetch URL content asynchronously.
    
    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        timeout: Request timeout in seconds
        
    Returns:
        HTML content as string
        
    Raises:
        ScraperError: If request fails
    """
    # Check if we have a cached version first
    from http_utils import _cache_path, _is_cache_valid
    from config import CACHE_TTL
    
    cache_path = _cache_path(url)
    if _is_cache_valid(cache_path, CACHE_TTL):
        return cache_path.read_text(encoding="utf-8")
    
    # No valid cache, make the request
    try:
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            html = await response.text()
            
            # Save to cache
            cache_path.write_text(html, encoding="utf-8")
            return html
    except Exception as e:
        logger.error(f"Async fetch failed for {url}: {e}")
        raise ScraperError(f"Failed to fetch {url}: {str(e)}")


async def scrape_teams_async(league_id: str, season: int, recent_only: bool = False) -> List[pd.DataFrame]:
    """Scrape all teams in a league asynchronously.
    
    Args:
        league_id: League ID from FBref
        season: Season year
        recent_only: If True, only get last 7 matches
        
    Returns:
        List of DataFrames with team match data
        
    Raises:
        ScraperError: If scraping fails
    """
    # First get all team URLs
    team_urls = get_squad_urls(league_id, season)
    
    if not team_urls:
        logger.warning(f"No team URLs found for league {league_id}, season {season}")
        return []
    
    # Get league name for filtering
    league_name = next((name for name, data in LEAGUES.items() 
                      if data["id"] == league_id), None)
    
    if not league_name:
        raise ScraperError(f"Unknown league ID: {league_id}")
    
    # Set up async requests
    headers = {"User-Agent": "Mozilla/5.0 (FBrefToolkit; +https://github.com/your/repo)"}
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Create tasks for all teams
        tasks = []
        for url in team_urls:
            tasks.append(scrape_team_async(url, league_name, season, session, recent_only))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        dataframes = []
        for i, result in enumerate(results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                dataframes.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error scraping team {team_urls[i]}: {result}")
            
        return dataframes


async def scrape_team_async(
    team_url: str, 
    league_name: str, 
    season: int, 
    session: aiohttp.ClientSession,
    recent_only: bool = False
) -> pd.DataFrame:
    """Scrape team data asynchronously.
    
    Args:
        team_url: Team URL
        league_name: League name for filtering
        season: Season year
        session: aiohttp ClientSession
        recent_only: If True, only get last 7 matches
        
    Returns:
        DataFrame with team data
        
    Raises:
        ScraperError: If scraping fails
    """
    try:
        html = await fetch_async(team_url, session)
        
        # Process the same way as the synchronous version
        # but return as DataFrame instead of saving to database
        soup = soupify(html)
        
        # Get team name from URL or page title
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        page_title = soup.find("title")
        if page_title:
            title_parts = page_title.text.split(" Stats")
            if title_parts:
                team_name = title_parts[0]
        
        # Parse matches
        matches = None
        try:
            matches = pd.read_html(html, match="Scores & Fixtures")[0]
            matches = tidy_fbref_table(matches)
        except Exception as e:
            logger.warning(f"Could not parse matches table for {team_name}: {e}")
            return pd.DataFrame()
        
        if matches.empty:
            return pd.DataFrame()
            
        # Filter for specified league
        matches = matches[matches["comp"].str.contains(league_name, case=False)]
        if matches.empty:
            return pd.DataFrame()
            
        # Optional recent slice
        if recent_only:
            matches = matches.tail(7)
            
        # Add team and season info
        matches = matches.assign(
            team=unidecode(team_name),
            season=season,
            scrape_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Create unique match ID
        matches["match_id"] = (
            matches["date"].astype(str) + "_" + 
            matches["team"].str.replace(" ", "_", regex=False) + "_" + 
            matches["opponent"].str.replace(" ", "_", regex=False)
        )
        
        # Add calculated columns for analysis
        matches["is_home"] = matches["venue"] == "Home"
        matches["points"] = matches["result"].map({"W": 3, "D": 1, "L": 0})
        
        return matches
        
    except Exception as e:
        logger.error(f"Error in async scrape for {team_url}: {e}")
        raise ScraperError(f"Failed to scrape team asynchronously: {str(e)}")
eam_url(team_id: str, season: int) -> str:
    """Generate URL for a team's page."""
    return f"{BASE_URL}/en/squads/{team_id}/{season}-{season+1}/matchlogs/all_comps/schedule/"

def player_url(player_id: str) -> str:
    """Generate URL for a player's page."""
    return f"{BASE_URL}/en/players/{player_id}/"

###############################################################################
# Table cleaning utilities                                                   #
###############################################################################

def t