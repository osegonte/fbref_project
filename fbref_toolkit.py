#!/usr/bin/env python
"""
fbref_toolkit.py
================
An advanced Python module for football analytics that bundles data collection, storage,
and visualization. Features include:

• FBref match + player scraping (daily refresh or bulk)
• Multiple league support with configurable parameters
• League standings tables
• Advanced team form analysis and head-to-head statistics
• Transfermarkt integration for player data enrichment
• Comprehensive visualization suite (treemaps, bar plots, line charts)
• Automated data collection with scheduling
• Exportable data in multiple formats (CSV, JSON)
• API endpoints for data access (optional Fast API integration)
• Streamlit dashboard for interactive analytics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import squarify
from bs4 import BeautifulSoup, Tag
from matplotlib import cm, colors
from sqlalchemy import create_engine, text
from unidecode import unidecode

# Import configuration
from config import POSTGRES_URI, REQUEST_DELAY, CACHE_DIR, LOG_DIR, LEAGUES

# Optional imports for scheduling 
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

# Optional imports for dashboard/API functionality
try:
    import streamlit as st
    import fastapi
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

###############################################################################
# Configuration
###############################################################################
BASE_URL = "https://fbref.com"
TRANSFERMARKT_URL = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (FBrefToolkit; +https://github.com/your/repo)"
}

# Set up cache directories
cache_dir = Path(os.path.expanduser(CACHE_DIR))
cache_dir.mkdir(exist_ok=True)

log_dir = Path(os.path.expanduser(LOG_DIR))
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "fbref_toolkit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_toolkit")

###############################################################################
# HTTP helpers (file cache to avoid hammering FBref)                         #
###############################################################################

def _cache_path(url: str) -> Path:
    """Generate a cache file path from a URL."""
    # Create a more unique cache filename based on URL components
    url_parts = url.replace("https://", "").replace("http://", "").split("/")
    cache_name = "_".join(url_parts[-3:] if len(url_parts) >= 3 else url_parts)
    return cache_dir / f"{cache_name}.html"


def fetch(url: str, use_cache: bool = True, cache_ttl: int = 86400) -> str:
    """Fetch URL content with caching and rate limiting.
    
    Args:
        url: The URL to fetch
        use_cache: Whether to use cached responses
        cache_ttl: Cache time-to-live in seconds (default: 1 day)
    
    Returns:
        HTML content as string
    """
    path = _cache_path(url)
    
    # Check for fresh cache
    if use_cache and path.exists():
        # Check cache age
        cache_age = time.time() - path.stat().st_mtime
        if cache_age < cache_ttl:
            logger.debug(f"Using cached response for {url}")
            return path.read_text(encoding="utf-8")
        else:
            logger.debug(f"Cache expired for {url}")
    
    # Fetch with retry logic
    for attempt in range(3):
        try:
            time.sleep(REQUEST_DELAY * (attempt + 1))  # Exponential backoff
            logger.debug(f"Fetching {url} (attempt {attempt + 1}/3)")
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.ok:
                html = r.text
                path.write_text(html, encoding="utf-8")
                return html
            else:
                logger.warning(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}")
    
    # If we get here, all attempts failed
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")


def soupify(html: str) -> BeautifulSoup:
    """Convert HTML string to BeautifulSoup object."""
    return BeautifulSoup(html, "html.parser")

###############################################################################
# Generic table-cleaning utilities                                           #
###############################################################################

def tidy_fbref_table(df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Flatten multi-index columns, drop header separators, dedupe names, cast numerics."""
    if df.empty:
        return df
        
    # 1) flatten
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
            df = df[~df[rank_col].astype(str).str.match(r'^[-–—]+$|^Matches$')]

    # 3) deduplicate column names
    counts: dict[str, int] = {}
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
            df[col] = pd.to_numeric(df[col], errors="ignore")
    
    return df.reset_index(drop=True)


def standardize_team_names(df: pd.DataFrame, name_col: str = 'team') -> pd.DataFrame:
    """Standardize team names for consistency across different sources."""
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
        # Add more as needed
    }
    
    # Apply standardization
    df[name_col] = df[name_col].str.lower().map(
        lambda x: replacements.get(x, x) if x in replacements else x
    )
    
    return df

###############################################################################
# FBref scrape logic                                                         #
###############################################################################

def season_schedule_url(league_id: str, season: int) -> str:
    """Generate URL for a league's season schedule."""
    return f"{BASE_URL}/en/comps/{league_id}/{season}-{season+1}/schedule/{season}-{season+1}-Scores-and-Fixtures"


def league_table_url(league_id: str, season: int) -> str:
    """Generate URL for a league's standings table."""
    return f"{BASE_URL}/en/comps/{league_id}/{season}-{season+1}/standings/"


def squad_urls(league_id: str, season: int) -> List[str]:
    """Get URLs for all team squad pages in a league season."""
    html = fetch(season_schedule_url(league_id, season))
    soup = soupify(html)
    table = soup.select_one("table.stats_table")
    if not table:
        logger.error(f"Could not find teams table for league {league_id}, season {season}")
        return []
        
    hrefs = {a["href"] for a in table.select("a") if "/squads/" in a.get("href", "")}
    return sorted(f"{BASE_URL}{h}" for h in hrefs)


def read_table(html: str, caption_kw: str) -> pd.DataFrame:
    """Extract and clean a table from HTML based on caption keyword."""
    try:
        df = pd.read_html(html, match=caption_kw, flavor="lxml")[0]
        return tidy_fbref_table(df)
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not read table with caption {caption_kw}: {e}")
        return pd.DataFrame()


def scrape_team(team_url: str, league_name: str, season: int, recent_only: bool = False) -> pd.DataFrame:
    """Scrape team match data from FBref.
    
    Args:
        team_url: URL to team's season page
        league_name: Name of the league (to filter matches)
        season: Season year (starting year)
        recent_only: If True, only get last 7 matches
        
    Returns:
        DataFrame with match data
    """
    logger.info(f"Scraping team data from {team_url}")
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

    # Function to merge additional stats
    def _merge_extra(keyword: str, keep_cols: List[str]) -> pd.DataFrame:
        try:
            links = [f"{BASE_URL}{a['href']}" for a in soup.find_all("a", href=True) 
                      if keyword in a["href"]]
            
            if not links:
                return pd.DataFrame()
                
            extra = read_table(fetch(links[0]), keyword.split("/")[-1])
            return extra[[c for c in keep_cols if c in extra.columns]]
        except (IndexError, KeyError) as e:
            logger.warning(f"Error fetching {keyword} data: {e}")
            return pd.DataFrame()

    # Get additional statistics
    shooting = _merge_extra("all_comps/shooting", ["date", "sh", "sot", "dist", "fk", "pk", "pkatt"])
    corners = _merge_extra("corner", ["date", "corner_for", "corner_against", "cf", "ca"])
    possession = _merge_extra("possession", ["date", "poss", "touches", "def_pen", "def_3rd", "mid_3rd", "att_3rd"])
    passes = _merge_extra("passing", ["date", "cmp", "att", "cmp_pct", "progressive_passes"])
    
    # Merge all data
    df = matches.copy()
    for extra in (shooting, corners, possession, passes):
        if not extra.empty:
            try:
                df = df.merge(extra, on="date", how="left")
            except Exception as e:
                logger.warning(f"Error merging data: {e}")

    # Add team and season info
    df = df.assign(
        team=unidecode(team_name), 
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


def scrape_player_stats(team_url: str, season: int) -> pd.DataFrame:
    """Scrape player statistics for a team.
    
    Args:
        team_url: URL to team's season page
        season: Season year
        
    Returns:
        DataFrame with player statistics
    """
    logger.info(f"Scraping player data from {team_url}")
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    
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
                
                # Create player ID based on name and team
                if "player" in df.columns:
                    df["player_id"] = df["player"].str.lower().str.replace(" ", "_") + "_" + team_name.lower().replace(" ", "_") + f"_{season}"
                
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


def scrape_league_table(league_id: str, season: int) -> pd.DataFrame:
    """Scrape current league standings table.
    
    Args:
        league_id: League ID from FBref
        season: Season year
        
    Returns:
        DataFrame with league table data
    """
    url = league_table_url(league_id, season)
    html = fetch(url, cache_ttl=3600)  # Shorter cache for standings (1 hour)
    
    try:
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
        return df
    except Exception as e:
        logger.warning(f"Error scraping league table: {e}")
        return pd.DataFrame()


def scrape_head_to_head(team1: str, team2: str, num_matches: int = 10) -> pd.DataFrame:
    """Scrape head-to-head match history between two teams.
    
    Args:
        team1: First team name
        team2: Second team name
        num_matches: Number of recent matches to fetch
        
    Returns:
        DataFrame with head-to-head match data
    """
    # For this we need to search
    search_url = f"{BASE_URL}/en/search/matches/?from=matchlogs&q={team1}+vs+{team2}"
    html = fetch(search_url, cache_ttl=86400)  # 1 day cache
    soup = soupify(html)
    
    # Find the match results table
    matches_section = soup.find("div", {"id": "matches"})
    if not matches_section:
        logger.warning(f"No head-to-head data found for {team1} vs {team2}")
        return pd.DataFrame()
    
    try:
        # Extract match data
        df = pd.read_html(str(matches_section))[0]
        df = tidy_fbref_table(df)
        
        # Take only the most recent matches
        if len(df) > num_matches:
            df = df.head(num_matches)
            
        # Add metadata
        df["team1"] = team1
        df["team2"] = team2
        
        return df
    except Exception as e:
        logger.warning(f"Error scraping head-to-head data: {e}")
        return pd.DataFrame()

###############################################################################
# Transfermarkt enrichment (contract expiry + market value)                 #
###############################################################################

def tmkt_contract_expiry(player_id: int) -> Optional[str]:
    """Get player contract expiry date from Transfermarkt.
    
    Args:
        player_id: Transfermarkt player ID
        
    Returns:
        Contract expiry date (YYYY-MM-DD) or None if not found
    """
    url = f"{TRANSFERMARKT_URL}/transfers/contractEndAjax?ajax=1&id={player_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.ok:
            data = response.json()
            # Take first entry
            if data and len(data) > 0:
                return data[0]["dateTo"]  # YYYY-MM-DD
    except Exception as e:
        logger.warning(f"Error fetching contract data for player {player_id}: {e}")
    return None


def tmkt_market_value_history(player_id: int) -> pd.DataFrame:
    """Get player market value history from Transfermarkt.
    
    Args:
        player_id: Transfermarkt player ID
        
    Returns:
        DataFrame with market value history
    """
    url = f"{TRANSFERMARKT_URL}/player/getMarketValueGraphData/{player_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.ok:
            data = response.json()
            
            if "data" in data and "marker" in data["data"]:
                values = []
                for point in data["data"]["marker"]:
                    values.append({
                        "player_id": player_id,
                        "date": point["datum"],
                        "value_eur": point["mw"],
                        "club": point.get("verein", ""),
                        "age": point.get("age", "")
                    })
                return pd.DataFrame(values)
    except Exception as e:
        logger.warning(f"Error fetching market value for player {player_id}: {e}")
    
    return pd.DataFrame()


def find_tmkt_player_id(player_name: str, club_name: Optional[str] = None) -> Optional[int]:
    """Search for a player's Transfermarkt ID.
    
    Args:
        player_name: Player name to search
        club_name: Optional club name to refine search
        
    Returns:
        Transfermarkt player ID or None if not found
    """
    search_term = player_name
    if club_name:
        search_term += f" {club_name}"
        
    url = f"{TRANSFERMARKT_URL}/search/players/?query={search_term.replace(' ', '+')}"
    
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        
        player_table = soup.select_one("table.items")
        if not player_table:
            return None
            
        # Find first player link
        player_link = player_table.select_one("td.hauptlink a")
        if player_link and "href" in player_link.attrs:
            href = player_link["href"]
            # Extract ID from URL pattern like /player/playerid/123456
            parts = href.split("/")
            if len(parts) >= 3:
                try:
                    return int(parts[-1])
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"Error searching for player {player_name}: {e}")
    
    return None

###############################################################################
# Database helpers                                                           #
###############################################################################

def pg_engine():
    """Create SQLAlchemy engine with connection pooling."""
    return create_engine(
        POSTGRES_URI,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )


def ensure_tables_exist():
    """Create database tables if they don't exist."""
    eng = pg_engine()
    with eng.begin() as conn:
        # Check if tables exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                date DATE,
                team TEXT,
                opponent TEXT,
                result TEXT,
                gf INTEGER,
                ga INTEGER,
                venue TEXT,
                is_home BOOLEAN,
                points INTEGER,
                sh INTEGER,
                sot INTEGER,
                corner_for INTEGER,
                corner_against INTEGER,
                poss NUMERIC,
                xg NUMERIC,
                xga NUMERIC,
                comp TEXT,
                season INTEGER,
                scrape_date DATE
            );
            
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                player TEXT,
                team TEXT,
                nation TEXT,
                pos TEXT,
                age NUMERIC,
                minutes INTEGER,
                goals INTEGER,
                assists INTEGER,
                season INTEGER,
                stats_type TEXT,
                scrape_date DATE
            );
            
            CREATE TABLE IF NOT EXISTS league_tables (
                id SERIAL PRIMARY KEY,
                rank INTEGER,
                squad TEXT,
                matches_played INTEGER,
                wins INTEGER,
                draws INTEGER,
                losses INTEGER,
                goals_for INTEGER,
                goals_against INTEGER,
                goal_diff INTEGER,
                points INTEGER,
                points_per_match NUMERIC,
                xg NUMERIC,
                xga NUMERIC,
                xg_diff NUMERIC,
                league TEXT,
                country TEXT,
                season INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS player_valuations (
                id SERIAL PRIMARY KEY,
                player_id INTEGER,
                date DATE,
                value_eur INTEGER,
                club TEXT,
                age TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_matches_team ON matches (team);
            CREATE INDEX IF NOT EXISTS idx_matches_season ON matches (season);
            CREATE INDEX IF NOT EXISTS idx_players_team ON players (team);
            CREATE INDEX IF NOT EXISTS idx_league_tables_season ON league_tables (season);
        """))


def upsert(df: pd.DataFrame, table: str, key_cols: Sequence[str]) -> None:
    """Insert or update records in database table.
    
    Args:
        df: DataFrame with data to upsert
        table: Target table name
        key_cols: List of column names that form the primary key
    """
    if df.empty:
        logger.warning(f"Empty DataFrame provided for upsert to {table}")
        return
        
    # Clean DataFrame before upsert
    # Replace NaN with None for database compatibility
    df = df.replace({pd.NA: None})
    
    eng = pg_engine()
    with eng.begin() as conn:
        tmp = f"tmp_{table}"
        df.to_sql(tmp, conn, index=False, if_exists="replace")
        
        cols = ", ".join(df.columns)
        keys = ", ".join(key_cols)
        # Only update non-key columns
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in df.columns if c not in key_cols)
        
        # Execute upsert
        if sets:  # Only if there are columns to update
            query = f"""
                INSERT INTO {table} ({cols}) 
                SELECT {cols} FROM {tmp}
                ON CONFLICT ({keys}) 
                DO UPDATE SET {sets};
                DROP TABLE {tmp};
            """
        else:
            # If only key columns are present, just do nothing on conflict
            query = f"""
                INSERT INTO {table} ({cols}) 
                SELECT {cols} FROM {tmp}
                ON CONFLICT ({keys}) DO NOTHING;
                DROP TABLE {tmp};
            """
            
        conn.execute(text(query))
        logger.info(f"Upserted {len(df)} rows to {table}")


def export_data(table: str, query: str = None, format: str = "csv", output_path: str = None) -> str:
    """Export data from database to file.
    
    Args:
        table: Table name to export
        query: Optional custom SQL query
        format: Output format (csv, json, excel)
        output_path: Optional output file path
        
    Returns:
        Path to output file
    """
    eng = pg_engine()
    
    # Use either provided query or simple SELECT
    if query:
        df = pd.read_sql(query, eng)
    else:
        df = pd.read_sql(f"SELECT * FROM {table}", eng)
    
    if df.empty:
        logger.warning(f"No data found for export from {table}")
        return None
        
    # Generate default filename if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/{table}_{timestamp}.{format}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to specified format
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", date_format="iso")
    elif format == "excel":
        df.to_excel(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {format}")
        
    logger.info(f"Exported {len(df)} rows to {output_path}")
    return output_path

###############################################################################
# Analytics functions                                                        #
###############################################################################

def team_form_analysis(team: str, matches: int = 5) -> pd.DataFrame:
    """Calculate recent form statistics for a team.
    
    Args:
        team: Team name
        matches: Number of recent matches to analyze
        
    Returns:
        DataFrame with form statistics
    """
    eng = pg_engine()
    query = """
    SELECT * FROM matches 
    WHERE team = %s 
    ORDER BY date DESC 
    LIMIT %s
    """
    
    df = pd.read_sql_query(query, eng, params=(team, matches))
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate form metrics
    form_metrics = {
        "team": team,
        "period": f"Last {matches} matches",
        "wins": len(df[df["result"] == "W"]),
        "draws": len(df[df["result"] == "D"]),
        "losses": len(df[df["result"] == "L"]),
        "points": df["points"].sum(),
        "goals_for": df["gf"].sum(),
        "goals_against": df["ga"].sum(),
        "goal_diff": df["gf"].sum() - df["ga"].sum(),
        "clean_sheets": len(df[df["ga"] == 0]),
        "failed_to_score": len(df[df["gf"] == 0]),
        "avg_corners_for": df["corner_for"].mean() if "corner_for" in df.columns else None,
        "avg_shots": df["sh"].mean() if "sh" in df.columns else None,
        "avg_shots_on_target": df["sot"].mean() if "sot" in df.columns else None,
        "avg_possession": df["poss"].mean() if "poss" in df.columns else None,
    }
    
    # Add win rate
    total_matches = form_metrics["wins"] + form_metrics["draws"] + form_metrics["losses"]
    form_metrics["win_rate"] = form_metrics["wins"] / total_matches if total_matches > 0 else 0
    
    # Add form string (W, D, L for last 5 matches from oldest to newest)
    form_string = "".join(df["result"].iloc[::-1])
    form_metrics["form_string"] = form_string
    
    return pd.DataFrame([form_metrics])


def calculate_head_to_head_stats(team1: str, team2: str, limit: int = 10) -> Dict[str, Any]:
    """Calculate head-to-head statistics between two teams.
    
    Args:
        team1: First team name
        team2: Second team name
        limit: Maximum number of matches to consider
    
    Returns:
        Dictionary with head-to-head statistics
    """
    eng = pg_engine()
    
    # Get matches where these teams played each other
    query = """
    SELECT * FROM matches 
    WHERE (team = %s AND opponent = %s) OR (team = %s AND opponent = %s)
    ORDER BY date DESC
    LIMIT %s
    """
    
    df = pd.read_sql_query(query, eng, params=(team1, team2, team2, team1, limit))
    
    if df.empty:
        return {"error": f"No head-to-head data found for {team1} vs {team2}"}
    
    # Initialize stats
    stats = {
        "team1": team1,
        "team2": team2,
        "total_matches": len(df),
        "team1_wins": 0,
        "team2_wins": 0,
        "draws": 0,
        "team1_goals": 0,
        "team2_goals": 0,
        "matches": []
    }
    
    # Calculate stats
    for _, match in df.iterrows():
        # Record match details
        match_info = {
            "date": match["date"],
            "score": f"{match['gf']}-{match['ga']}",
            "venue": match["venue"]
        }
        
        # Determine who was home/away and who won
        if match["team"] == team1:
            match_info["home_team"] = team1
            match_info["away_team"] = team2
            
            if match["result"] == "W":
                stats["team1_wins"] += 1
            elif match["result"] == "L":
                stats["team2_wins"] += 1
            else:
                stats["draws"] += 1
                
            stats["team1_goals"] += match["gf"]
            stats["team2_goals"] += match["ga"]
            
        else:  # match["team"] == team2
            match_info["home_team"] = team2
            match_info["away_team"] = team1
            
            if match["result"] == "W":
                stats["team2_wins"] += 1
            elif match["result"] == "L":
                stats["team1_wins"] += 1
            else:
                stats["draws"] += 1
                
            stats["team2_goals"] += match["gf"]
            stats["team1_goals"] += match["ga"]
        
        stats["matches"].append(match_info)
    
    # Calculate additional stats
    stats["avg_goals_per_match"] = (stats["team1_goals"] + stats["team2_goals"]) / stats["total_matches"]
    stats["team1_win_percentage"] = (stats["team1_wins"] / stats["total_matches"]) * 100
    stats["team2_win_percentage"] = (stats["team2_wins"] / stats["total_matches"]) * 100
    stats["draw_percentage"] = (stats["draws"] / stats["total_matches"]) * 100
    
    return stats

###############################################################################
# Visualization helpers                                                     #
###############################################################################

def treemap(df: pd.DataFrame, value_col: str, label_col: str, 
           title: str, out: Optional[Path] = None,
           cmap: str = "viridis", figsize: Tuple[int, int] = (12, 8)) -> None:
    """Create a treemap visualization of data.
    
    Args:
        df: DataFrame with data
        value_col: Column name for values (box sizes)
        label_col: Column name for labels
        title: Chart title
        out: Optional output file path
        cmap: Colormap name
        figsize: Figure size as (width, height) tuple
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for treemap")
        return
        
    sizes = df[value_col].astype(float).tolist()
    labels = [f"{l}\n{v:.1f}" for l, v in zip(df[label_col], df[value_col])]

    norm = colors.Normalize(vmin=min(sizes), vmax=max(sizes))
    color_map = cm.get_cmap(cmap)
    clrs = [color_map(norm(s)) for s in sizes]

    fig, ax = plt.subplots(figsize=figsize)
    squarify.plot(sizes=sizes, label=labels, color=clrs, ax=ax,
                 text_kwargs=dict(fontsize=12, weight="bold", color="white"))
    
    plt.suptitle(title, fontsize=16, y=0.95)
    ax.axis("off")
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved treemap to {out}")
    else:
        plt.show()

###############################################################################
# API and Dashboard Integration                                             #
###############################################################################

def create_api_app():
    """Create FastAPI app for data access."""
    if not HAS_DASHBOARD:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None
        
    app = FastAPI(title="FBref API", description="Football data API powered by FBref")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def read_root():
        return {"message": "Welcome to FBref API", "version": "1.0.0"}
    
    @app.get("/teams")
    def get_teams(league: str = None, season: int = None):
        eng = pg_engine()
        query = "SELECT DISTINCT team FROM matches"
        params = []
        
        if league or season:
            query += " WHERE"
            
        if league:
            query += " comp = %s"
            params.append(league)
            
        if season:
            query += " AND" if league else ""
            query += " season = %s"
            params.append(season)
            
        query += " ORDER BY team"
        
        df = pd.read_sql_query(query, eng, params=params)
        return {"teams": df["team"].tolist()}
    
    @app.get("/matches")
    def get_matches(team: str = None, season: int = None, limit: int = 10):
        eng = pg_engine()
        query = "SELECT * FROM matches"
        params = []
        
        if team or season:
            query += " WHERE"
            
        if team:
            query += " team = %s"
            params.append(team)
            
        if season:
            query += " AND" if team else ""
            query += " season = %s"
            params.append(season)
            
        query += " ORDER BY date DESC LIMIT %s"
        params.append(limit)
        
        df = pd.read_sql_query(query, eng, params=params)
        return df.to_dict(orient="records")
    
    @app.get("/league-table")
    def get_league_table(league: str, season: int = None):
        if not season:
            season = datetime.now().year
            
        eng = pg_engine()
        query = "SELECT * FROM league_tables WHERE league = %s AND season = %s ORDER BY rank"
        
        df = pd.read_sql_query(query, eng, params=(league, season))
        return df.to_dict(orient="records")
    
    @app.get("/team-form")
    def get_team_form(team: str, matches: int = 5):
        form_data = team_form_analysis(team, matches)
        return form_data.to_dict(orient="records")[0] if not form_data.empty else {}
    
    @app.get("/head-to-head")
    def get_head_to_head(team1: str, team2: str, limit: int = 10):
        return calculate_head_to_head_stats(team1, team2, limit)
    
    return app


def run_api(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI app."""
    if not HAS_DASHBOARD:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return
        
    import uvicorn
    app = create_api_app()
    if app:
        logger.info(f"Starting API server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)


def run_dashboard():
    """Run Streamlit dashboard for interactive analysis."""
    if not HAS_DASHBOARD:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return
        
    logger.info("Please run 'streamlit run fbref_dashboard.py' to start the dashboard")

###############################################################################
# Scheduling and automation                                                 #
###############################################################################

def schedule_daily_update(hour: int = 6, minute: int = 0, 
                      leagues: List[str] = None) -> None:
    """Schedule daily data updates using APScheduler.
    
    Args:
        hour: Hour of day to run (24-hour format)
        minute: Minute of hour to run
        leagues: List of league names to update (default: Premier League only)
    """
    if not HAS_SCHEDULER:
        logger.error("APScheduler not installed. Install with: pip install apscheduler")
        return
        
    if leagues is None:
        leagues = ["Premier League"]
        
    logger.info(f"Scheduling daily updates at {hour:02d}:{minute:02d}")
    
    def daily_job():
        """Function to run for the daily update."""
        logger.info("Running daily update")
        
        # Ensure database tables exist
        ensure_tables_exist()
        
        current_year = datetime.now().year
        
        # Update each league
        for league_name in leagues:
            league_id = next((data["id"] for name, data in LEAGUES.items() 
                          if name.lower() == league_name.lower()), None)
            
            if not league_id:
                logger.warning(f"Unknown league: {league_name}")
                continue
                
            logger.info(f"Updating {league_name} data")
            
            try:
                # Update league table
                league_table = scrape_league_table(league_id, current_year)
                if not league_table.empty:
                    upsert(league_table, "league_tables", ["rank", "squad", "season", "league"])
                
                # Update team matches (last 7 matches only)
                for team_url in squad_urls(league_id, current_year):
                    try:
                        matches_df = scrape_team(team_url, league_name, current_year, recent_only=True)
                        if not matches_df.empty:
                            upsert(matches_df, "matches", ["match_id"])
                            
                            # Optionally update player data
                            players_df = scrape_player_stats(team_url, current_year)
                            if not players_df.empty:
                                upsert(players_df, "players", ["player_id"])
                    except Exception as e:
                        logger.error(f"Error updating team {team_url}: {e}")
            except Exception as e:
                logger.error(f"Error updating league {league_name}: {e}")
                
        logger.info("Daily update completed")
    
    # Create scheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(daily_job, 'cron', hour=hour, minute=minute, id='daily_update')
    
    try:
        logger.info("Starting scheduler")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def run_one_time_update(league_name: str, season: int, recent_only: bool = False) -> None:
    """Run a one-time update of data.
    
    Args:
        league_name: League name to update
        season: Season year to update
        recent_only: If True, only get last 7 matches
    """
    # Get league ID
    league_id = next((data["id"] for name, data in LEAGUES.items() 
                   if name.lower() == league_name.lower()), None)
    
    if not league_id:
        logger.error(f"Unknown league: {league_name}")
        return
        
    logger.info(f"Running one-time update for {league_name} {season}")
    
    # Ensure database tables exist
    ensure_tables_exist()
    
    # Update league table
    league_table = scrape_league_table(league_id, season)
    if not league_table.empty:
        upsert(league_table, "league_tables", ["rank", "squad", "season", "league"])
    
    # Update team matches
    for team_url in squad_urls(league_id, season):
        try:
            matches_df = scrape_team(team_url, league_name, season, recent_only=recent_only)
            if not matches_df.empty:
                upsert(matches_df, "matches", ["match_id"])
                logger.info(f"Updated {len(matches_df)} matches for {matches_df['team'].iloc[0]}")
                
                # Update player data
                players_df = scrape_player_stats(team_url, season)
                if not players_df.empty:
                    upsert(players_df, "players", ["player_id"])
                    logger.info(f"Updated {len(players_df)} player records for {matches_df['team'].iloc[0]}")
        except Exception as e:
            logger.error(f"Error updating team {team_url}: {e}")
    
    logger.info(f"One-time update completed for {league_name} {season}")

###############################################################################
# CLI                                                                       #
###############################################################################

def cmd_scrape(args):
    """Command to scrape data."""
    run_one_time_update(args.league_name, args.season, args.recent)


def cmd_plot(args):
    """Command to create visualization."""
    eng = pg_engine()
    
    if args.type == "treemap":
        # Treemap of goals per club
        sql = """
        SELECT team, SUM(gf) AS goals
        FROM   matches
        WHERE  season = %(season)s AND comp LIKE %(league)s
        GROUP  BY team ORDER BY goals DESC;
        """
        df = pd.read_sql(sql, eng, params=dict(season=args.season, league=f"%{args.league_name}%"))
        treemap(df, "goals", "team", 
               f"Goals – {args.league_name} {args.season}/{args.season+1}", 
               Path(args.out) if args.out else None)
               
    elif args.type == "form":
        # Form heatmap for top teams
        sql = """
        SELECT squad FROM league_tables
        WHERE season = %(season)s AND league = %(league)s
        ORDER BY rank
        LIMIT 6;
        """
        df = pd.read_sql(sql, eng, params=dict(season=args.season, league=args.league_name))
        
        if not df.empty:
            teams = df["squad"].tolist()
            form_heatmap(teams, args.season, args.limit or 5, 
                        Path(args.out) if args.out else None)
        else:
            # Fallback to getting teams from matches
            sql = """
            SELECT DISTINCT team FROM matches 
            WHERE season = %(season)s AND comp LIKE %(league)s
            LIMIT 6;
            """
            df = pd.read_sql(sql, eng, params=dict(season=args.season, league=f"%{args.league_name}%"))
            teams = df["team"].tolist()
            form_heatmap(teams, args.season, args.limit or 5, 
                        Path(args.out) if args.out else None)
    
    elif args.type == "bar":
        # Bar chart of goals, points, or other metrics
        metric = args.metric or "gf"
        metric_name = {
            "gf": "Goals For",
            "ga": "Goals Against",
            "points": "Points",
            "sh": "Shots",
            "sot": "Shots on Target",
            "corner_for": "Corners"
        }.get(metric, metric)
        
        sql = f"""
        SELECT team, SUM({metric}) AS value
        FROM   matches
        WHERE  season = %(season)s AND comp LIKE %(league)s
        GROUP  BY team ORDER BY value DESC;
        """
        df = pd.read_sql(sql, eng, params=dict(season=args.season, league=f"%{args.league_name}%"))
        
        bar_plot(df, "team", "value", 
               f"{metric_name} – {args.league_name} {args.season}/{args.season+1}", 
               Path(args.out) if args.out else None,
               limit=args.limit)


def cmd_export(args):
    """Command to export data to file."""
    if args.query:
        # Custom SQL query
        export_data(args.table, args.query, args.format, args.out)
    else:
        # Direct table export
        export_data(args.table, None, args.format, args.out)


def cmd_api(args):
    """Command to start API server."""
    run_api(args.host, args.port)


def cmd_dashboard(args):
    """Command to start dashboard."""
    run_dashboard()


def cmd_schedule(args):
    """Command to schedule daily updates."""
    leagues = args.leagues.split(",") if args.leagues else None
    schedule_daily_update(args.hour, args.minute, leagues)


def cmd_analyze(args):
    """Command to run analysis on teams or matches."""
    if args.type == "form":
        # Team form analysis
        form_df = team_form_analysis(args.team, args.matches)
        if not form_df.empty:
            if args.json:
                print(form_df.to_json(orient="records"))
            else:
                print(form_df.to_string())
        else:
            print(f"No form data available for {args.team}")
            
    elif args.type == "h2h":
        # Head-to-head analysis
        h2h_stats = calculate_head_to_head_stats(args.team1, args.team2, args.matches)
        if "error" not in h2h_stats:
            if args.json:
                print(json.dumps(h2h_stats, default=str))
            else:
                print(f"\nHead-to-head: {args.team1} vs {args.team2}")
                print(f"Total matches: {h2h_stats['total_matches']}")
                print(f"{args.team1} wins: {h2h_stats['team1_wins']} ({h2h_stats['team1_win_percentage']:.1f}%)")
                print(f"{args.team2} wins: {h2h_stats['team2_wins']} ({h2h_stats['team2_win_percentage']:.1f}%)")
                print(f"Draws: {h2h_stats['draws']} ({h2h_stats['draw_percentage']:.1f}%)")
                print(f"Goals: {args.team1} {h2h_stats['team1_goals']} - {h2h_stats['team2_goals']} {args.team2}")
                print(f"Avg. goals per match: {h2h_stats['avg_goals_per_match']:.2f}")
                
                print("\nRecent matches:")
                for i, match in enumerate(h2h_stats['matches'][:5]):
                    print(f"{match['date']}: {match['home_team']} {match['score']} {match['away_team']}")
        else:
            print(h2h_stats["error"])


def build_cli():
    """Build command-line interface."""
    ap = argparse.ArgumentParser(description="FBref Toolkit – Football Data Scraper and Analyzer")
    subparsers = ap.add_subparsers(dest="cmd", required=True)

    # scrape command
    sc = subparsers.add_parser("scrape", help="Pull match data from FBref and store to database")
    sc.add_argument("--league-name", default="Premier League", help="League name (e.g. 'Premier League')")
    sc.add_argument("--season", type=int, default=datetime.now().year, help="Season starting year")
    sc.add_argument("--recent", action="store_true", help="Only last 7 matches (ignores --season)")
    sc.set_defaults(func=cmd_scrape)

    # plot command
    pl = subparsers.add_parser("plot", help="Create visualizations of football data")
    pl.add_argument("--type", choices=["treemap", "bar", "form"], default="treemap", help="Plot type")
    pl.add_argument("--league-name", default="Premier League", help="League name")
    pl.add_argument("--season", type=int, default=datetime.now().year, help="Season starting year")
    pl.add_argument("--metric", help="Metric to plot (for bar charts): gf, ga, points, sh, sot, etc.")
    pl.add_argument("--limit", type=int, help="Limit number of items to display")
    pl.add_argument("--out", help="Output file path (PNG)")
    pl.set_defaults(func=cmd_plot)
    
    # export command
    ex = subparsers.add_parser("export", help="Export data to file formats")
    ex.add_argument("table", help="Table name to export")
    ex.add_argument("--query", help="Custom SQL query (optional)")
    ex.add_argument("--format", choices=["csv", "json", "excel", "parquet"], default="csv", help="Output format")
    ex.add_argument("--out", help="Output file path")
    ex.set_defaults(func=cmd_export)
    
    # analyze command
    an = subparsers.add_parser("analyze", help="Run analysis on teams or matches")
    an.add_argument("--type", choices=["form", "h2h"], required=True, help="Analysis type")
    an.add_argument("--team", help="Team name for form analysis")
    an.add_argument("--team1", help="First team name for head-to-head analysis")
    an.add_argument("--team2", help="Second team name for head-to-head analysis")
    an.add_argument("--matches", type=int, default=5, help="Number of matches to analyze")
    an.add_argument("--json", action="store_true", help="Output results as JSON")
    an.set_defaults(func=cmd_analyze)
    
    # api command
    api = subparsers.add_parser("api", help="Start API server")
    api.add_argument("--host", default="127.0.0.1", help="API server host")
    api.add_argument("--port", type=int, default=8000, help="API server port")
    api.set_defaults(func=cmd_api)
    
    # dashboard command
    dash = subparsers.add_parser("dashboard", help="Run interactive dashboard")
    dash.set_defaults(func=cmd_dashboard)
    
    # schedule command
    sch = subparsers.add_parser("schedule", help="Schedule automated data updates")
    sch.add_argument("--hour", type=int, default=6, help="Hour to run update (24-hour format)")
    sch.add_argument("--minute", type=int, default=0, help="Minute to run update")
    sch.add_argument("--leagues", help="Comma-separated list of leagues to update")
    sch.set_defaults(func=cmd_schedule)

    return ap

###############################################################################
# Entry-point                                                               #
###############################################################################

if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()
    
    try:
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def bar_plot(df: pd.DataFrame, x_col: str, y_col: str, 
           title: str, out: Optional[Path] = None,
           color: str = "skyblue", figsize: Tuple[int, int] = (12, 8),
           sort: bool = True, limit: int = None) -> None:
    """Create a bar plot visualization.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        out: Optional output file path
        color: Bar color
        figsize: Figure size as (width, height) tuple
        sort: Whether to sort by y values
        limit: Optional limit on number of bars to show
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for bar plot")
        return
    
    # Sort data if requested
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(y_col, ascending=False)
    
    # Limit number of bars if requested
    if limit and len(plot_df) > limit:
        plot_df = plot_df.head(limit)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar plot
    bars = ax.bar(plot_df[x_col], plot_df[y_col], color=color)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.1f}', ha='center', va='bottom')
    
    # Customize appearance
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved bar plot to {out}")
    else:
        plt.show()


def line_plot(df: pd.DataFrame, x_col: str, y_cols: List[str], 
            title: str, out: Optional[Path] = None,
            figsize: Tuple[int, int] = (12, 8)) -> None:
    """Create a multi-line plot visualization.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis lines
        title: Chart title
        out: Optional output file path
        figsize: Figure size as (width, height) tuple
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for line plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each line
    for col in y_cols:
        ax.plot(df[x_col], df[col], marker='o', linewidth=2, label=col)
    
    # Customize appearance
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved line plot to {out}")
    else:
        plt.show()


def form_heatmap(teams: List[str], season: int, last_n: int = 5, 
               out: Optional[Path] = None) -> None:
    """Create a heatmap showing form of multiple teams.
    
    Args:
        teams: List of team names
        season: Season year
        last_n: Number of recent matches to include
        out: Optional output file path
    """
    eng = pg_engine()
    
    results_data = []
    
    for team in teams:
        query = """
        SELECT date, team, opponent, result, gf, ga, venue 
        FROM matches 
        WHERE team = %s AND season = %s
        ORDER BY date DESC 
        LIMIT %s
        """
        
        df = pd.read_sql_query(query, eng, params=(team, season, last_n))
        
        if not df.empty:
            # Reverse to show oldest to newest
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Create result array with color codes
            result_array = []
            for _, row in df.iterrows():
                if row["result"] == "W":
                    result_array.append(3)  # Win
                elif row["result"] == "D":
                    result_array.append(1)  # Draw
                else:
                    result_array.append(0)  # Loss
            
            results_data.append({
                "team": team,
                "results": result_array,
                "opponents": df["opponent"].tolist(),
                "scores": [f"{r['gf']}-{r['ga']}" for _, r in df.iterrows()]
            })
    
    if not results_data:
        logger.warning("No data found for form heatmap")
        return
        
    # Create plot
    fig, ax = plt.subplots(figsize=(12, len(teams) * 0.8 + 2))
    
    # Custom color map: red for loss, yellow for draw, green for win
    cmap = colors.ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
    bounds = [-0.5, 0.5, 1.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Prepare data for heatmap
    data = np.zeros((len(teams), last_n))
    yticks = []
    
    for i, team_data in enumerate(results_data):
        # Fill in available results, leaving zeros for missing matches
        for j, result in enumerate(team_data["results"]):
            if j < last_n:
                data[i, j] = result
        
        yticks.append(team_data["team"])
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    
    # Add match labels
    for i, team_data in enumerate(results_data):
        for j, (result, opponent, score) in enumerate(zip(team_data.get("results", []), 
                                                      team_data.get("opponents", []), 
                                                      team_data.get("scores", []))):
            if j < last_n:
                text = f"{opponent}\n{score}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    
    match_numbers = [f"Match {i+1}" for i in range(last_n)]
    ax.set_xticks(np.arange(last_n))
    ax.set_xticklabels(match_numbers)
    
    plt.title(f"Team Form - Last {last_n} Matches", fontsize=16)
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved form heatmap to {out}")
    else:
        plt.show()