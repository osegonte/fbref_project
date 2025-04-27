#!/usr/bin/env python3
"""
Football Data Pipeline Controller

This script orchestrates the entire data pipeline:
1. Fetch fixtures from SofaScore
2. Collect team statistics from FBref
3. Process and transform the data
4. Store in PostgreSQL database
5. Export to CSV files for visualization

Usage:
  python pipeline_controller.py --full-run      # Run the complete pipeline
  python pipeline_controller.py --fixtures-only # Only fetch fixture data
  python pipeline_controller.py --stats-only    # Only collect team statistics
  python pipeline_controller.py --specific-date 2025-05-01  # Process specific date
"""

import os
import sys
import time
import argparse
import logging
import json
import glob
from datetime import datetime, timedelta, date
import pandas as pd
from sqlalchemy import create_engine, text
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
    """Store fixture data in the database"""
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
                    home_team VARCHAR(50) NOT NULL,
                    away_team VARCHAR(50) NOT NULL,
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
        
        # Store fixtures in database
        with engine.begin() as conn:
            for fixture in fixtures_data:
                # Use upsert (insert or update)
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
        
        logger.info(f"Successfully stored {len(fixtures_data)} fixtures in database")
        return {'success': True, 'fixtures_count': len(fixtures_data)}
            
    except Exception as e:
        logger.exception(f"Error storing fixtures in database: {e}")
        return {'success': False, 'error': str(e)}

def store_team_stats_in_db(stats_csv, engine):
    """Store team statistics in the database"""
    try:
        logger.info(f"Loading team stats from {stats_csv}")
        df = pd.read_csv(stats_csv)
        
        if df.empty:
            logger.warning("No team stats to store")
            return {'success': False, 'error': 'No team stats data'}
        
        logger.info(f"Processing {len(df)} match records for database storage")
        
        # Process and store matches
        stats_data = []
        for _, row in df.iterrows():
            stat = {
                'match_id': row['match_id'],
                'date': row['date'],
                'team': row['team'],
                'opponent': row['opponent'],
                'venue': row['venue'],
                'result': row['result'],
                'gf': row.get('gf', None),
                'ga': row.get('ga', None),
                'points': row.get('points', None),
                'sh': row.get('sh', None),
                'sot': row.get('sot', None),
                'dist': row.get('dist', None),
                'fk': row.get('fk', None),
                'pk': row.get('pk', None),
                'pkatt': row.get('pkatt', None),
                'possession': row.get('possession', None),
                'xg': row.get('xg', None),
                'xga': row.get('xga', None),
                'comp': row.get('comp', None),
                'round': row.get('round', None),
                'season': row.get('season', None),
                'is_home': row.get('is_home', None),
                'scrape_date': datetime.now()
            }
            stats_data.append(stat)
        
        # Store matches in database
        with engine.begin() as conn:
            for stat in stats_data:
                # Use upsert (insert or update)
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
                    gf = EXCLUDED.gf,
                    ga = EXCLUDED.ga,
                    points = EXCLUDED.points,
                    sh = EXCLUDED.sh,
                    sot = EXCLUDED.sot,
                    dist = EXCLUDED.dist,
                    fk = EXCLUDED.fk,
                    pk = EXCLUDED.pk,
                    pkatt = EXCLUDED.pkatt,
                    possession = EXCLUDED.possession,
                    xg = EXCLUDED.xg,
                    xga = EXCLUDED.xga,
                    scrape_date = EXCLUDED.scrape_date
                """), stat)
        
        logger.info(f"Successfully stored {len(stats_data)} match records in database")
        return {'success': True, 'matches_count': len(stats_data)}
            
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
    parser.add_argument('--date-range', type=int, default=7, help='Number of days to fetch (default: 7)')
    parser.add_argument('--max-teams', type=int, default=0, help='Maximum teams to process (0 for all)')
    
    args = parser.parse_args()
    
    # Default to full run if no options specified
    if not (args.full_run or args.fixtures_only or args.stats_only):
        args.full_run = True
    
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