#!/usr/bin/env python3
"""
Database to CSV Export Utility

This script exports data from the PostgreSQL database to CSV files for easier verification.
It can export specific tables, team data, or run custom queries.

Usage:
  python export_db_to_csv.py --all  # Export all tables to CSV
  python export_db_to_csv.py --team "Liverpool"  # Export data for a specific team
  python export_db_to_csv.py --table "recent_matches"  # Export a specific table
  python export_db_to_csv.py --date "2025-05-01"  # Export fixtures for a specific date
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/export_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("export_db")

def get_db_engine():
    """Get SQLAlchemy engine from environment variables"""
    pg_uri = os.getenv('PG_URI')
    if not pg_uri:
        logger.error("PG_URI environment variable not found")
        raise ValueError("Database connection string not configured")
    
    return create_engine(pg_uri)

def export_table(table_name, output_dir="data/exports"):
    """Export a specific table to CSV"""
    try:
        engine = get_db_engine()
        logger.info(f"Exporting table: {table_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Query the database
        with engine.connect() as conn:
            # Check if table exists
            check_query = text(f"SELECT to_regclass('public.{table_name}')")
            result = conn.execute(check_query)
            if result.scalar() is None:
                logger.error(f"Table '{table_name}' does not exist")
                print(f"Error: Table '{table_name}' does not exist in the database")
                return {'success': False, 'error': f"Table '{table_name}' does not exist"}
            
            # Get the table data
            query = text(f"SELECT * FROM {table_name}")
            result = conn.execute(query)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(zip(result.keys(), row)) for row in result])
            
            if df.empty:
                logger.warning(f"Table '{table_name}' is empty")
                print(f"Warning: Table '{table_name}' is empty")
                return {'success': True, 'rows': 0, 'message': 'Table is empty'}
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{table_name}_{timestamp}.csv")
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Exported {len(df)} rows from '{table_name}' to {output_file}")
            print(f"Successfully exported {len(df)} rows from '{table_name}' to {output_file}")
            
            return {'success': True, 'rows': len(df), 'file': output_file}
            
    except Exception as e:
        logger.exception(f"Error exporting table '{table_name}': {e}")
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def export_team_data(team_name, output_dir="data/exports/teams"):
    """Export all data for a specific team"""
    try:
        engine = get_db_engine()
        logger.info(f"Exporting data for team: {team_name}")
        
        # Create output directory
        team_dir = os.path.join(output_dir, team_name.replace(' ', '_'))
        os.makedirs(team_dir, exist_ok=True)
        
        # Dictionary to store results for each query
        results = {}
        
        # Check if team exists in teams table
        with engine.connect() as conn:
            team_query = text("SELECT * FROM teams WHERE team_name = :team")
            result = conn.execute(team_query, {"team": team_name})
            team_data = [dict(zip(result.keys(), row)) for row in result]
            
            if not team_data:
                logger.warning(f"Team '{team_name}' not found in database")
                print(f"Warning: Team '{team_name}' not found in database")
                return {'success': False, 'error': f"Team '{team_name}' not found"}
            
            # Save team data
            team_df = pd.DataFrame(team_data)
            team_file = os.path.join(team_dir, "team_info.csv")
            team_df.to_csv(team_file, index=False)
            results['team_info'] = {'rows': len(team_df), 'file': team_file}
        
        # Export fixtures involving this team
        with engine.connect() as conn:
            fixtures_query = text("""
                SELECT * FROM fixtures 
                WHERE home_team = :team OR away_team = :team
                ORDER BY date DESC
            """)
            result = conn.execute(fixtures_query, {"team": team_name})
            fixtures_data = [dict(zip(result.keys(), row)) for row in result]
            
            if fixtures_data:
                fixtures_df = pd.DataFrame(fixtures_data)
                fixtures_file = os.path.join(team_dir, "fixtures.csv")
                fixtures_df.to_csv(fixtures_file, index=False)
                results['fixtures'] = {'rows': len(fixtures_df), 'file': fixtures_file}
                logger.info(f"Exported {len(fixtures_df)} fixtures for {team_name}")
                print(f"Exported {len(fixtures_df)} fixtures for {team_name}")
        
        # Export recent matches data
        with engine.connect() as conn:
            matches_query = text("""
                SELECT * FROM recent_matches
                WHERE team = :team
                ORDER BY date DESC
            """)
            result = conn.execute(matches_query, {"team": team_name})
            matches_data = [dict(zip(result.keys(), row)) for row in result]
            
            if matches_data:
                matches_df = pd.DataFrame(matches_data)
                matches_file = os.path.join(team_dir, "recent_matches.csv")
                matches_df.to_csv(matches_file, index=False)
                results['recent_matches'] = {'rows': len(matches_df), 'file': matches_file}
                logger.info(f"Exported {len(matches_df)} match records for {team_name}")
                print(f"Exported {len(matches_df)} match records for {team_name}")
        
        # Export head-to-head data against other teams
        with engine.connect() as conn:
            h2h_query = text("""
                SELECT 
                    opponent,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                    SUM(gf) as goals_for,
                    SUM(ga) as goals_against,
                    SUM(points) as points
                FROM recent_matches
                WHERE team = :team
                GROUP BY opponent
                ORDER BY points DESC
            """)
            result = conn.execute(h2h_query, {"team": team_name})
            h2h_data = [dict(zip(result.keys(), row)) for row in result]
            
            if h2h_data:
                h2h_df = pd.DataFrame(h2h_data)
                h2h_file = os.path.join(team_dir, "head_to_head.csv")
                h2h_df.to_csv(h2h_file, index=False)
                results['head_to_head'] = {'rows': len(h2h_df), 'file': h2h_file}
                logger.info(f"Exported head-to-head data against {len(h2h_df)} opponents")
                print(f"Exported head-to-head data against {len(h2h_df)} opponents")
        
        # Export home/away performance
        with engine.connect() as conn:
            venue_query = text("""
                SELECT 
                    venue,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                    SUM(gf) as goals_for,
                    SUM(ga) as goals_against,
                    ROUND(AVG(points)::numeric, 2) as avg_points,
                    ROUND(AVG(possession)::numeric, 2) as avg_possession
                FROM recent_matches
                WHERE team = :team
                GROUP BY venue
            """)
            result = conn.execute(venue_query, {"team": team_name})
            venue_data = [dict(zip(result.keys(), row)) for row in result]
            
            if venue_data:
                venue_df = pd.DataFrame(venue_data)
                venue_file = os.path.join(team_dir, "home_away_split.csv")
                venue_df.to_csv(venue_file, index=False)
                results['home_away'] = {'rows': len(venue_df), 'file': venue_file}
                logger.info(f"Exported home/away performance data")
                print(f"Exported home/away performance data")
        
        logger.info(f"Successfully exported data for {team_name}")
        return {'success': True, 'results': results}
            
    except Exception as e:
        logger.exception(f"Error exporting team data: {e}")
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def export_date_fixtures(target_date, output_dir="data/exports/dates"):
    """Export fixtures for a specific date"""
    try:
        engine = get_db_engine()
        
        # Convert string date to date object if needed
        if isinstance(target_date, str):
            try:
                date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
                date_str = target_date
            except ValueError:
                logger.error(f"Invalid date format: {target_date}")
                print(f"Error: Invalid date format '{target_date}'. Use YYYY-MM-DD format.")
                return {'success': False, 'error': 'Invalid date format'}
        else:
            date_obj = target_date
            date_str = date_obj.strftime("%Y-%m-%d")
        
        logger.info(f"Exporting fixtures for date: {date_str}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Query fixtures for the target date
        with engine.connect() as conn:
            fixtures_query = text("""
                SELECT * FROM fixtures
                WHERE date = :date
                ORDER BY start_time
            """)
            result = conn.execute(fixtures_query, {"date": date_str})
            fixtures_data = [dict(zip(result.keys(), row)) for row in result]
            
            if not fixtures_data:
                logger.warning(f"No fixtures found for date {date_str}")
                print(f"Warning: No fixtures found for date {date_str}")
                return {'success': True, 'rows': 0, 'message': 'No fixtures found'}
            
            # Convert to DataFrame
            fixtures_df = pd.DataFrame(fixtures_data)
            
            # Generate output filename
            output_file = os.path.join(output_dir, f"fixtures_{date_str}.csv")
            
            # Save to CSV
            fixtures_df.to_csv(output_file, index=False)
            
            logger.info(f"Exported {len(fixtures_df)} fixtures for {date_str} to {output_file}")
            print(f"Successfully exported {len(fixtures_df)} fixtures for {date_str} to {output_file}")
            
            return {'success': True, 'rows': len(fixtures_df), 'file': output_file}
            
    except Exception as e:
        logger.exception(f"Error exporting fixtures for date {target_date}: {e}")
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def export_all_tables(output_dir="data/exports/all_tables"):
    """Export all tables from the database"""
    try:
        engine = get_db_engine()
        logger.info("Exporting all tables")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of all tables in the database
        with engine.connect() as conn:
            tables_query = text("""
                SELECT tablename FROM pg_catalog.pg_tables 
                WHERE schemaname = 'public'
            """)
            result = conn.execute(tables_query)
            tables = [row[0] for row in result]
            
            if not tables:
                logger.warning("No tables found in database")
                print("Warning: No tables found in database")
                return {'success': True, 'tables': 0, 'message': 'No tables found'}
        
        # Export each table
        results = {}
        for table in tables:
            table_result = export_table(table, output_dir)
            results[table] = table_result
        
        # Generate summary
        successful = sum(1 for table, result in results.items() if result['success'])
        total_rows = sum(result.get('rows', 0) for result in results.values() if 'rows' in result)
        
        logger.info(f"Successfully exported {successful}/{len(tables)} tables with {total_rows} total rows")
        print(f"Successfully exported {successful}/{len(tables)} tables with {total_rows} total rows")
        
        return {'success': True, 'tables': len(tables), 'successful': successful, 'total_rows': total_rows, 'results': results}
            
    except Exception as e:
        logger.exception(f"Error exporting all tables: {e}")
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Database to CSV Export Utility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Export all tables')
    group.add_argument('--table', help='Export specific table')
    group.add_argument('--team', help='Export data for a specific team')
    group.add_argument('--date', help='Export fixtures for a specific date (YYYY-MM-DD)')
    
    parser.add_argument('--output-dir', help='Custom output directory')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else "data/exports"
    
    try:
        if args.all:
            all_dir = os.path.join(output_dir, "all_tables")
            result = export_all_tables(all_dir)
            return 0 if result['success'] else 1
            
        elif args.table:
            result = export_table(args.table, output_dir)
            return 0 if result['success'] else 1
            
        elif args.team:
            team_dir = os.path.join(output_dir, "teams")
            result = export_team_data(args.team, team_dir)
            return 0 if result['success'] else 1
            
        elif args.date:
            date_dir = os.path.join(output_dir, "dates")
            result = export_date_fixtures(args.date, date_dir)
            return 0 if result['success'] else 1
            
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())