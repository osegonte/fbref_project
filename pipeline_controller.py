#!/usr/bin/env python3
"""
Football Data Pipeline Controller (Fixed Version)

This script orchestrates the entire data pipeline:
1. Fetch fixtures from SofaScore
2. Collect team statistics from FBref
3. Process and transform the data
4. Store in PostgreSQL database
5. Export to CSV files for visualization

Usage:
  python pipeline_controller_fixed.py --full-run      # Run the complete pipeline
  python pipeline_controller_fixed.py --fixtures-only # Only fetch fixture data
  python pipeline_controller_fixed.py --stats-only    # Only collect team statistics
  python pipeline_controller_fixed.py --specific-date 2025-05-01  # Process specific date
  python pipeline_controller_fixed.py --verify-team Liverpool  # Verify data for a specific team
"""

import os
import sys
import time
import argparse
import logging
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline")

def get_db_engine():
    """Get SQLAlchemy engine from environment variables"""
    pg_uri = os.getenv('PG_URI')
    if not pg_uri:
        logger.error("PG_URI environment variable not found")
        raise ValueError("Database connection string not configured")
    
    return create_engine(pg_uri)

def fetch_fixtures(date_range=7, specific_date=None):
    """Fetch fixture data from SofaScore"""
    try:
        from daily_match_scraper import AdvancedSofaScoreScraper
        
        logger.info("Initializing SofaScore scraper...")
        scraper = AdvancedSofaScoreScraper()
        
        if specific_date:
            # Parse specific date
            if isinstance(specific_date, str):
                target_date = datetime.strptime(specific_date, "%Y-%m-%d").date()
            else:
                target_date = specific_date
                
            end_date = target_date
            logger.info(f"Fetching fixtures for specific date: {target_date}")
        else:
            # Use date range
            target_date = date.today()
            end_date = target_date + timedelta(days=date_range)
            logger.info(f"Fetching fixtures from {target_date} to {end_date}")
        
        # Fetch matches
        matches_by_date, total_matches = scraper.fetch_matches_for_date_range(target_date, end_date)
        
        if total_matches > 0:
            logger.info(f"Successfully fetched {total_matches} matches across {len(matches_by_date)} days")
            
            # Get path to combined CSV
            all_file = os.path.join(
                scraper.data_dir, 
                f"all_matches_{target_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
            )
            
            return {
                'success': True,
                'total_matches': total_matches,
                'days': len(matches_by_date),
                'matches_by_date': matches_by_date,
                'csv_path': all_file
            }
        else:
            logger.error("No fixtures fetched")
            return {'success': False, 'error': 'No fixtures found'}
            
    except Exception as e:
        logger.exception(f"Error fetching fixtures: {e}")
        return {'success': False, 'error': str(e)}

def collect_team_stats(fixtures_csv, max_teams=0):
    """Collect team statistics from FBref"""
    try:
        from fbref_stats_collector import FBrefStatsCollector
        
        logger.info(f"Initializing FBref stats collector for {fixtures_csv}")
        
        # Configure directories
        output_dir = "data/team_stats"
        cache_dir = "data/cache"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize collector
        collector = FBrefStatsCollector(
            output_dir=output_dir,
            cache_dir=cache_dir,
            lookback=7,            # Get 7 past matches per team
            batch_size=3,          # Process 3 teams at a time
            delay_between_batches=60  # Wait 60 seconds between batches
        )
        
        # Run collector
        output_file = collector.run(
            input_file=fixtures_csv,
            max_teams=max_teams
        )
        
        if output_file:
            logger.info(f"Successfully collected team stats, saved to: {output_file}")
            return {'success': True, 'output_file': output_file}
        else:
            logger.error("Failed to collect team stats")
            return {'success': False, 'error': 'Failed to collect team stats'}
            
    except Exception as e:
        logger.exception(f"Error collecting team stats: {e}")
        return {'success': False, 'error': str(e)}

def store_fixtures_in_db(fixtures_csv, engine):
    """Store fixture data in the database with improved error handling"""
    try:
        logger.info(f"Loading fixtures data from {fixtures_csv}")
        df = pd.read_csv(fixtures_csv)
        
        if df.empty:
            logger.warning("No fixture data to store")
            return {'success': False, 'error': 'No fixture data'}
        
        logger.info(f"Processing {len(df)} fixtures for database storage")
        
        # First, ensure all teams exist in the teams table
        teams = set()
        for _, row in df.iterrows():
            teams.add(row['home_team'])
            teams.add(row['away_team'])
        
        logger.info(f"Found {len(teams)} unique teams")
        
        # Create an export folder for real-time CSV checks
        export_dir = "data/db_exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Store teams
        with engine.begin() as conn:
            for team in teams:
                # Check if team exists
                result = conn.execute(
                    text("SELECT team_id FROM teams WHERE team_name = :team"),
                    {"team": team}
                )
                if not result.first():
                    # Insert team
                    conn.execute(
                        text("INSERT INTO teams (team_name, country, league) VALUES (:team, :country, :league)"),
                        {"team": team, "country": df[df['home_team'] == team]['country'].iloc[0] if len(df[df['home_team'] == team]) > 0 else None, 
                         "league": df[df['home_team'] == team]['league'].iloc[0] if len(df[df['home_team'] == team]) > 0 else None}
                    )
            
            logger.info("Teams stored in database")
        
        # Create fixtures table if not exists
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fixtures (
                    fixture_id SERIAL PRIMARY KEY,
                    match_id VARCHAR(100) UNIQUE,
                    date DATE NOT NULL,
                    home_team VARCHAR(100) NOT NULL,
                    away_team VARCHAR(100) NOT NULL,
                    league VARCHAR(100),
                    country VARCHAR(50),
                    start_time VARCHAR(20),
                    start_timestamp BIGINT,
                    venue VARCHAR(100),
                    status VARCHAR(50),
                    source VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes if not exists
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(home_team, away_team)"))
        
        # Prepare data for insertion
        fixtures_data = []
        for _, row in df.iterrows():
            # Create a unique match_id if not present
            if 'id' not in row:
                match_id = (f"{row['date']}_{row['home_team'].replace(' ', '_')}"
                           f"_{row['away_team'].replace(' ', '_')}")
            else:
                match_id = row['id']
            
            fixture = {
                'match_id': match_id,
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'league': row['league'],
                'country': row.get('country', 'Unknown'),
                'start_time': row.get('start_time', None),
                'start_timestamp': row.get('start_timestamp', None),
                'venue': row.get('venue', None),
                'status': row.get('status', 'Scheduled'),
                'source': row.get('source', 'SofaScore')
            }
            fixtures_data.append(fixture)
        
        # Store fixtures in database with error handling
        successful_inserts = 0
        failed_inserts = 0
        
        for fixture in fixtures_data:
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO fixtures 
                        (match_id, date, home_team, away_team, league, country, 
                         start_time, start_timestamp, venue, status, source)
                        VALUES 
                        (:match_id, :date, :home_team, :away_team, :league, :country,
                         :start_time, :start_timestamp, :venue, :status, :source)
                        ON CONFLICT (match_id) 
                        DO UPDATE SET
                        status = EXCLUDED.status,
                        start_time = EXCLUDED.start_time,
                        start_timestamp = EXCLUDED.start_timestamp,
                        venue = EXCLUDED.venue
                    """), fixture)
                    successful_inserts += 1
            except SQLAlchemyError as e:
                logger.error(f"Error inserting fixture {fixture['match_id']}: {e}")
                failed_inserts += 1
        
        # Export all fixtures to CSV for verification
        with engine.connect() as conn:
            query = text("SELECT * FROM fixtures ORDER BY date DESC")
            result = conn.execute(query)
            fixtures_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            fixtures_export = os.path.join(export_dir, "fixtures_export.csv")
            fixtures_df.to_csv(fixtures_export, index=False)
            logger.info(f"Exported all fixtures to {fixtures_export}")
        
        logger.info(f"Successfully stored {successful_inserts} fixtures in database ({failed_inserts} failed)")
        return {'success': True, 'fixtures_count': successful_inserts, 'failed_count': failed_inserts}
            
    except Exception as e:
        logger.exception(f"Error storing fixtures in database: {e}")
        return {'success': False, 'error': str(e)}

def store_team_stats_in_db(stats_csv, engine):
    """Store team statistics in the database with improved error handling"""
    try:
        logger.info(f"Loading team stats from {stats_csv}")
        df = pd.read_csv(stats_csv)
        
        if df.empty:
            logger.warning("No team stats to store")
            return {'success': False, 'error': 'No team stats data'}
        
        logger.info(f"Processing {len(df)} match records for database storage")
        
        # Create an export folder for real-time CSV checks
        export_dir = "data/db_exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Process and store matches
        stats_data = []
        for _, row in df.iterrows():
            # Handle missing or NaN values
            stat = {
                'match_id': row['match_id'],
                'date': row['date'],
                'team': row['team'],
                'opponent': row['opponent'],
                'venue': row['venue'],
                'result': row['result'],
                'gf': None if pd.isna(row.get('gf')) else row.get('gf'),
                'ga': None if pd.isna(row.get('ga')) else row.get('ga'),
                'points': None if pd.isna(row.get('points')) else row.get('points'),
                'sh': None if pd.isna(row.get('sh')) else row.get('sh'),
                'sot': None if pd.isna(row.get('sot')) else row.get('sot'),
                'dist': None if pd.isna(row.get('dist')) else row.get('dist'),
                'fk': None if pd.isna(row.get('fk')) else row.get('fk'),
                'pk': None if pd.isna(row.get('pk')) else row.get('pk'),
                'pkatt': None if pd.isna(row.get('pkatt')) else row.get('pkatt'),
                'possession': None if pd.isna(row.get('possession')) else row.get('possession'),
                'xg': None if pd.isna(row.get('xg')) else row.get('xg'),
                'xga': None if pd.isna(row.get('xga')) else row.get('xga'),
                'comp': row.get('comp', None),
                'round': row.get('round', None),
                'season': row.get('season', None),
                'is_home': row.get('is_home', None),
                'scrape_date': datetime.now()
            }
            stats_data.append(stat)
        
        # Create recent_matches table if it doesn't exist
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS recent_matches (
                    match_id VARCHAR(100) PRIMARY KEY,
                    date DATE NOT NULL,
                    team VARCHAR(100) NOT NULL,
                    opponent VARCHAR(100) NOT NULL,
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
                    comp VARCHAR(100),
                    round VARCHAR(100),
                    season INTEGER,
                    is_home BOOLEAN,
                    scrape_date TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_recent_matches_team ON recent_matches(team)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_recent_matches_date ON recent_matches(date)"))
        
        # Store matches in database with error handling
        successful_inserts = 0
        failed_inserts = 0
        
        for stat in stats_data:
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO recent_matches 
                        (match_id, date, team, opponent, venue, result, gf, ga, points,
                         sh, sot, dist, fk, pk, pkatt, possession, xg, xga,
                         comp, round, season, is_home, scrape_date)
                        VALUES 
                        (:match_id, :date, :team, :opponent, :venue, :result, :gf, :ga, :points,
                         :sh, :sot, :dist, :fk, :pk, :pkatt, :possession, :xg, :xga,
                         :comp, :round, :season, :is_home, :scrape_date)
                        ON CONFLICT (match_id) 
                        DO UPDATE SET
                        gf = COALESCE(EXCLUDED.gf, recent_matches.gf),
                        ga = COALESCE(EXCLUDED.ga, recent_matches.ga),
                        points = COALESCE(EXCLUDED.points, recent_matches.points),
                        sh = COALESCE(EXCLUDED.sh, recent_matches.sh),
                        sot = COALESCE(EXCLUDED.sot, recent_matches.sot),
                        dist = COALESCE(EXCLUDED.dist, recent_matches.dist),
                        fk = COALESCE(EXCLUDED.fk, recent_matches.fk),
                        pk = COALESCE(EXCLUDED.pk, recent_matches.pk),
                        pkatt = COALESCE(EXCLUDED.pkatt, recent_matches.pkatt),
                        possession = COALESCE(EXCLUDED.possession, recent_matches.possession),
                        xg = COALESCE(EXCLUDED.xg, recent_matches.xg),
                        xga = COALESCE(EXCLUDED.xga, recent_matches.xga),
                        scrape_date = EXCLUDED.scrape_date
                    """), stat)
                    successful_inserts += 1
            except SQLAlchemyError as e:
                logger.error(f"Error inserting match {stat['match_id']}: {e}")
                failed_inserts += 1
        
        # Export recent matches to CSV for verification
        with engine.connect() as conn:
            query = text("SELECT * FROM recent_matches ORDER BY date DESC")
            result = conn.execute(query)
            matches_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            matches_export = os.path.join(export_dir, "recent_matches_export.csv")
            matches_df.to_csv(matches_export, index=False)
            logger.info(f"Exported all recent matches to {matches_export}")
        
        logger.info(f"Successfully stored {successful_inserts} match records in database ({failed_inserts} failed)")
        return {'success': True, 'matches_count': successful_inserts, 'failed_count': failed_inserts}
            
    except Exception as e:
        logger.exception(f"Error storing team stats in database: {e}")
        return {'success': False, 'error': str(e)}

def export_team_reports(engine, output_dir="data/reports"):
    """Export team reports from database to CSV files"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating team reports...")
        
        # Get all teams
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT team FROM recent_matches"))
            teams = [row[0] for row in result]
        
        logger.info(f"Found {len(teams)} teams with match data")
        
        # For each team, export recent match data
        for team in teams:
            team_file = os.path.join(output_dir, f"{team.replace(' ', '_')}_recent.csv")
            
            # Query recent matches
            with engine.connect() as conn:
                query = text("""
                    SELECT 
                        date, opponent, venue, result, gf, ga, points, 
                        sh, sot, possession, xg, xga, comp
                    FROM recent_matches
                    WHERE team = :team
                    ORDER BY date DESC
                    LIMIT 7
                """)
                
                result = conn.execute(query, {"team": team})
                
                # Convert to DataFrame
                columns = result.keys()
                matches = pd.DataFrame([dict(zip(columns, row)) for row in result])
                
                if not matches.empty:
                    # Save to CSV
                    matches.to_csv(team_file, index=False)
                    logger.info(f"Exported report for {team} to {team_file}")
        
        # Export head-to-head reports for upcoming fixtures
        logger.info("Generating head-to-head reports for upcoming fixtures...")
        
        # Get upcoming fixtures
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    date, home_team, away_team, league
                FROM fixtures
                WHERE date >= CURRENT_DATE
                ORDER BY date ASC
                LIMIT 50
            """)
            
            result = conn.execute(query)
            
            # Convert to DataFrame
            columns = result.keys()
            fixtures = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        # For each fixture, export head-to-head report
        for _, fixture in fixtures.iterrows():
            team1 = fixture['home_team']
            team2 = fixture['away_team']
            fixture_date = fixture['date']
            
            # Format for filename
            h2h_file = os.path.join(
                output_dir, 
                f"h2h_{team1.replace(' ', '_')}_{team2.replace(' ', '_')}_{fixture_date}.csv"
            )
            
            # Query head-to-head matches
            with engine.connect() as conn:
                query = text("""
                    SELECT 
                        date, team as home_team, opponent as away_team,
                        gf as home_goals, ga as away_goals, result,
                        CASE 
                            WHEN result = 'W' THEN team
                            WHEN result = 'L' THEN opponent
                            ELSE 'Draw'
                        END as winner
                    FROM recent_matches
                    WHERE (team = :team1 AND opponent = :team2)
                       OR (team = :team2 AND opponent = :team1)
                    ORDER BY date DESC
                """)
                
                result = conn.execute(query, {"team1": team1, "team2": team2})
                
                # Convert to DataFrame
                columns = result.keys()
                h2h_matches = pd.DataFrame([dict(zip(columns, row)) for row in result])
                
                if not h2h_matches.empty:
                    # Save to CSV
                    h2h_matches.to_csv(h2h_file, index=False)
                    logger.info(f"Exported H2H report for {team1} vs {team2} to {h2h_file}")
        
        # Export form table
        form_file = os.path.join(output_dir, "form_table.csv")
        
        with engine.connect() as conn:
            # Use form table query from SQL file
            sql_path = os.path.join(os.path.dirname(__file__), 'sql/form_table.sql')
            
            if os.path.exists(sql_path):
                with open(sql_path, 'r') as f:
                    form_query = f.read()
                
                result = conn.execute(text(form_query))
                
                # Convert to DataFrame
                columns = result.keys()
                form_table = pd.DataFrame([dict(zip(columns, row)) for row in result])
                
                if not form_table.empty:
                    # Save to CSV
                    form_table.to_csv(form_file, index=False)
                    logger.info(f"Exported form table to {form_file}")
        
        return {'success': True, 'teams_exported': len(teams), 'fixtures_exported': len(fixtures)}
            
    except Exception as e:
        logger.exception(f"Error exporting team reports: {e}")
        return {'success': False, 'error': str(e)}

def find_latest_fixtures_csv():
    """Find the most recent fixtures CSV file"""
    try:
        # Look for all fixtures CSV files
        fixture_files = glob.glob("sofascore_data/all_matches_*.csv")
        
        if not fixture_files:
            # Try daily files
            fixture_files = glob.glob("sofascore_data/daily/matches_*.csv")
            
        if fixture_files:
            # Get the most recent file
            latest_file = max(fixture_files, key=os.path.getctime)
            logger.info(f"Found most recent fixtures CSV: {latest_file}")
            return latest_file
        else:
            logger.error("No fixtures CSV files found")
            return None
    except Exception as e:
        logger.exception(f"Error finding fixtures CSV: {e}")
        return None

def verify_team_data(team_name, engine, output_dir="data/verification"):
    """Verify scraped data for a specific team"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Verifying data for team: {team_name}")
        
        results = {}
        
        # 1. Check if team exists in the database
        with engine.connect() as conn:
            team_result = conn.execute(text("SELECT * FROM teams WHERE team_name = :team"), {"team": team_name})
            team_exists = team_result.rowcount > 0
            results['team_exists'] = team_exists
            
            if team_exists:
                # Export team data
                team_data = pd.DataFrame([dict(zip(team_result.keys(), row)) for row in team_result])
                team_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_team_info.csv")
                team_data.to_csv(team_file, index=False)
                logger.info(f"Team exists in database: {team_exists}")
            else:
                logger.warning(f"Team '{team_name}' not found in database")
        
        # 2. Check fixtures with this team
        with engine.connect() as conn:
            fixture_query = text("""
                SELECT * FROM fixtures 
                WHERE home_team = :team OR away_team = :team
                ORDER BY date DESC
            """)
            
            fixture_result = conn.execute(fixture_query, {"team": team_name})
            fixtures = [dict(zip(fixture_result.keys(), row)) for row in fixture_result]
            
            if fixtures:
                fixtures_df = pd.DataFrame(fixtures)
                fixtures_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_fixtures.csv")
                fixtures_df.to_csv(fixtures_file, index=False)
                logger.info(f"Found {len(fixtures)} fixtures for {team_name}")
                results['fixtures_count'] = len(fixtures)
                results['fixtures_file'] = fixtures_file
            else:
                logger.warning(f"No fixtures found for team '{team_name}'")
                results['fixtures_count'] = 0
        
        # 3. Check recent match data
        with engine.connect() as conn:
            match_query = text("""
                SELECT * FROM recent_matches
                WHERE team = :team
                ORDER BY date DESC
            """)
            
            match_result = conn.execute(match_query, {"team": team_name})
            matches = [dict(zip(match_result.keys(), row)) for row in match_result]
            
            if matches:
                matches_df = pd.DataFrame(matches)
                matches_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_matches.csv")
                matches_df.to_csv(matches_file, index=False)
                logger.info(f"Found {len(matches)} match records for {team_name}")
                results['matches_count'] = len(matches)
                results['matches_file'] = matches_file
                
                # Check for missing data
                null_counts = matches_df.isnull().sum().to_dict()
                results['missing_data'] = {k: v for k, v in null_counts.items() if v > 0}
                
                if results['missing_data']:
                    logger.warning(f"Missing data detected in match records: {results['missing_data']}")
                else:
                    logger.info("All match data fields are complete")
            else:
                logger.warning(f"No match records found for team '{team_name}'")
                results['matches_count'] = 0
        
        # 4. Fetch test data from FBref to compare
        try:
            from fbref_stats_collector import FBrefStatsCollector
            
            collector = FBrefStatsCollector(
                output_dir=output_dir,
                cache_dir="data/cache",
                lookback=2,  # Just get a couple of matches for verification
                batch_size=1,
                delay_between_batches=5
            )
            
            # Create a small test file with just this team
            test_fixtures = [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'home_team': team_name,
                'away_team': 'Test Opponent',
                'league': 'Premier League',
                'country': 'England'
            }]
            
            test_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_test_fixture.csv")
            pd.DataFrame(test_fixtures).to_csv(test_file, index=False)
            
            # Run a test collection
            test_result = collector.run(test_file, max_teams=1)
            
            if test_result:
                logger.info(f"Successfully fetched test data for {team_name}")
                results['test_success'] = True
                results['test_file'] = test_result
            else:
                logger.warning(f"Failed to fetch test data for {team_name}")
                results['test_success'] = False
        except Exception as e:
            logger.error(f"Error fetching test data: {e}")
            results['test_success'] = False
            results['test_error'] = str(e)
        
        # Generate summary report
        summary = {
            'team': team_name,
            'verification_date': datetime.now().isoformat(),
            'in_database': results.get('team_exists', False),
            'fixtures_found': results.get('fixtures_count', 0),
            'matches_found': results.get('matches_count', 0),
            'test_successful': results.get('test_success', False),
            'missing_data_fields': list(results.get('missing_data', {}).keys()),
            'files_created': [
                results.get('fixtures_file', ''),
                results.get('matches_file', ''),
                results.get('test_file', '')
            ]
        }
        
        summary_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_verification_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Verification summary saved to {summary_file}")
        return {'success': True, 'summary': summary, 'summary_file': summary_file}
        
    except Exception as e:
        logger.exception(f"Error verifying team data: {e}")
        return {'success': False, 'error': str(e)}

def prompt_for_date():
    """Prompt the user to enter a specific date for scraping"""
    while True:
        try:
            date_input = input("Enter date to scrape (YYYY-MM-DD) or 'today': ")
            if date_input.lower() == 'today':
                return date.today()
            else:
                return datetime.strptime(date_input, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")

def run_full_pipeline(args):
    """Run the complete data pipeline"""
    logger.info("Starting full pipeline run")
    
    # Initialize pipeline results
    results = {
        'start_time': datetime.now().isoformat(),
        'fixtures': None,
        'team_stats': None,
        'db_fixtures': None,
        'db_team_stats': None,
        'reports': None
    }
    
    try:
        # If user wants to input a specific date
        if args.prompt_date:
            args.specific_date = prompt_for_date()
            logger.info(f"Using user-specified date: {args.specific_date}")
        
        # Step 1: Fetch fixture data
        fixtures_csv = None
        
        if args.fixtures_only or not args.stats_only:
            logger.info("Step 1: Fetching fixture data")
            fixtures_result = fetch_fixtures(
                date_range=args.date_range, 
                specific_date=args.specific_date
            )
            results['fixtures'] = fixtures_result
            
            if not fixtures_result['success']:
                logger.error("Failed to fetch fixtures, stopping pipeline")
                return results
            
            fixtures_csv = fixtures_result['csv_path']
            
            # Store fixtures in database
            engine = get_db_engine()
            db_fixtures_result = store_fixtures_in_db(fixtures_csv, engine)
            results['db_fixtures'] = db_fixtures_result
            
            if not db_fixtures_result['success']:
                logger.warning("Failed to store fixtures in database, continuing pipeline")
        
        # Step 2: Collect team statistics
        if args.stats_only or not args.fixtures_only:
            logger.info("Step 2: Collecting team statistics")
            
            # If fixtures were fetched in this run, use that CSV
            if fixtures_csv:
                logger.info(f"Using fixtures CSV from current run: {fixtures_csv}")
            # If stats-only, find the most recent fixtures file
            elif args.stats_only:
                fixtures_csv = find_latest_fixtures_csv()
                if not fixtures_csv:
                    logger.error("No fixtures CSV file found. Please run with --fixtures-only first")
                    return results
                logger.info(f"Using most recent fixtures file: {fixtures_csv}")
            # For specific date, try to find a matching file
            elif args.specific_date:
                # Parse the date
                if isinstance(args.specific_date, str):
                    target_date = datetime.strptime(args.specific_date, "%Y-%m-%d").date()
                else:
                    target_date = args.specific_date
                
                # Look for a file with this date
                date_str = target_date.strftime("%Y-%m-%d")
                daily_file = f"sofascore_data/daily/matches_{date_str}.csv"
                
                if os.path.exists(daily_file):
                    fixtures_csv = daily_file
                    logger.info(f"Using daily fixtures file for {date_str}: {fixtures_csv}")
                else:
                    # Try to find a multi-day file that includes this date
                    fixture_files = glob.glob("sofascore_data/all_matches_*.csv")
                    for file in fixture_files:
                        if date_str in file:
                            fixtures_csv = file
                            logger.info(f"Using fixtures file that includes {date_str}: {fixtures_csv}")
                            break
                    
                    if not fixtures_csv:
                        logger.error(f"No fixtures file found for {date_str}")
                        return results
            
            # Check if we have a fixtures CSV to use
            if not fixtures_csv:
                logger.error("No fixtures CSV file available. Please run with --fixtures-only first")
                return results
            
            # Collect team statistics
            team_stats_result = collect_team_stats(fixtures_csv, max_teams=args.max_teams)
            results['team_stats'] = team_stats_result
            
            if not team_stats_result['success']:
                logger.error("Failed to collect team statistics, stopping pipeline")
                return results
            
            team_stats_csv = team_stats_result['output_file']
            
            # Store team stats in database
            engine = get_db_engine()
            db_team_stats_result = store_team_stats_in_db(team_stats_csv, engine)
            results['db_team_stats'] = db_team_stats_result
            
            if not db_team_stats_result['success']:
                logger.warning("Failed to store team stats in database, continuing pipeline")
        
        # Step 3: Export team reports
        logger.info("Step 3: Exporting team reports")
        engine = get_db_engine()
        reports_result = export_team_reports(engine)
        results['reports'] = reports_result
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        results['error'] = str(e)
    
    # Record end time
    results['end_time'] = datetime.now().isoformat()
    results['duration'] = (datetime.fromisoformat(results['end_time']) - 
                          datetime.fromisoformat(results['start_time'])).total_seconds()
    
    # Save pipeline results
    output_dir = "data/pipeline_runs"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(
        output_dir,
        f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Pipeline results saved to {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Football Data Pipeline Controller")
    parser.add_argument('--full-run', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--fixtures-only', action='store_true', help='Only fetch fixture data')
    parser.add_argument('--stats-only', action='store_true', help='Only collect team statistics')
    parser.add_argument('--specific-date', help='Process specific date (YYYY-MM-DD)')
    parser.add_argument('--prompt-date', action='store_true', help='Prompt for a date to scrape')
    parser.add_argument('--date-range', type=int, default=7, help='Number of days to fetch (default: 7)')
    parser.add_argument('--max-teams', type=int, default=0, help='Maximum teams to process (0 for all)')
    parser.add_argument('--verify-team', help='Verify data for a specific team')
    
    args = parser.parse_args()
    
    # Default to full run if no options specified
    if not (args.full_run or args.fixtures_only or args.stats_only or args.verify_team):
        args.prompt_date = True
        args.full_run = True
    
    # Verification mode
    if args.verify_team:
        engine = get_db_engine()
        verify_result = verify_team_data(args.verify_team, engine)
        if verify_result['success']:
            print(f"Verification for {args.verify_team} completed successfully")
            print(f"Summary: {verify_result['summary_file']}")
            return 0
        else:
            print(f"Verification for {args.verify_team} failed: {verify_result.get('error')}")
            return 1
    
    # Run pipeline
    results = run_full_pipeline(args)
    
    # Check pipeline success
    if args.fixtures_only:
        if results['fixtures'] and results['fixtures']['success']:
            logger.info("Fixtures pipeline completed successfully")
            return 0
        else:
            logger.error("Fixtures pipeline failed")
            return 1
    elif args.stats_only:
        if results['team_stats'] and results['team_stats']['success']:
            logger.info("Team stats pipeline completed successfully")
            return 0
        else:
            logger.error("Team stats pipeline failed")
            return 1
    else:  # Full run
        if (results['fixtures'] and results['fixtures']['success'] and
            results['team_stats'] and results['team_stats']['success']):
            logger.info("Full pipeline completed successfully")
            return 0
        else:
            logger.error("Pipeline failed at one or more stages")
            return 1

if __name__ == "__main__":
    sys.exit(main())