#!/usr/bin/env python3
"""
Database Export Utility

Exports data from PostgreSQL to CSV files for visualization and verification.
Allows exporting specific tables or running custom SQL queries for specialized reports.

Usage:
  python db_export.py --all                  # Export all tables
  python db_export.py --table recent_matches # Export specific table
  python db_export.py --query-file query.sql # Run custom SQL query from file
  python db_export.py --team-reports         # Generate team-specific reports
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_export")

def get_db_engine():
    """Get database connection from environment variables"""
    pg_uri = os.getenv('PG_URI')
    if not pg_uri:
        logger.error("PG_URI environment variable not found")
        raise ValueError("Database connection string not configured")
    
    return create_engine(pg_uri)

def export_table(table_name, output_dir="data/exports"):
    """Export a specific table to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info(f"Exporting table: {table_name}")
        
        # Check if table exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT to_regclass('public.{table_name}')"))
            if result.scalar() is None:
                logger.error(f"Table '{table_name}' does not exist")
                return {'success': False, 'error': f"Table '{table_name}' does not exist"}
        
        # Export to DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning(f"Table '{table_name}' is empty")
            return {'success': True, 'rows': 0, 'message': 'Table is empty'}
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{table_name}_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(df)} rows from '{table_name}' to {output_file}")
        return {'success': True, 'rows': len(df), 'file': output_file}
    
    except Exception as e:
        logger.exception(f"Error exporting table '{table_name}': {e}")
        return {'success': False, 'error': str(e)}

def export_query(query, name=None, output_dir="data/exports"):
    """Export results of a custom SQL query to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        if name is None:
            name = "custom_query"
            
        logger.info(f"Executing query: {name}")
        
        # Execute query
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning(f"Query '{name}' returned no results")
            return {'success': True, 'rows': 0, 'message': 'Query returned no results'}
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{name}_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(df)} rows from query '{name}' to {output_file}")
        return {'success': True, 'rows': len(df), 'file': output_file}
    
    except Exception as e:
        logger.exception(f"Error executing query '{name}': {e}")
        return {'success': False, 'error': str(e)}

def export_query_from_file(query_file, output_dir="data/exports"):
    """Export results of a SQL query from a file to CSV"""
    try:
        # Check if file exists
        if not os.path.exists(query_file):
            logger.error(f"Query file '{query_file}' does not exist")
            return {'success': False, 'error': f"Query file '{query_file}' does not exist"}
        
        # Read query from file
        with open(query_file, 'r') as f:
            query = f.read()
        
        # Extract name from filename (without extension)
        name = os.path.splitext(os.path.basename(query_file))[0]
        
        return export_query(query, name, output_dir)
    
    except Exception as e:
        logger.exception(f"Error executing query from file '{query_file}': {e}")
        return {'success': False, 'error': str(e)}

def export_all_tables(output_dir="data/exports"):
    """Export all tables to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info("Exporting all tables")
        
        # Get list of tables
        with engine.connect() as conn:
            query = text("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            result = conn.execute(query)
            tables = [row[0] for row in result]
        
        if not tables:
            logger.warning("No tables found in database")
            return {'success': True, 'tables': 0, 'message': 'No tables found'}
        
        # Export each table
        results = {}
        for table in tables:
            result = export_table(table, output_dir)
            results[table] = result
        
        logger.info(f"Exported {len(tables)} tables to {output_dir}")
        return {'success': True, 'tables': len(tables), 'results': results}
    
    except Exception as e:
        logger.exception(f"Error exporting all tables: {e}")
        return {'success': False, 'error': str(e)}

def generate_team_reports(output_dir="data/exports/teams"):
    """Generate and export team-specific reports"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info("Generating team reports")
        
        # Get list of teams
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT team 
                FROM recent_matches 
                ORDER BY team
            """)
            result = conn.execute(query)
            teams = [row[0] for row in result]
        
        if not teams:
            logger.warning("No teams found in database")
            return {'success': True, 'teams': 0, 'message': 'No teams found'}
        
        # For each team, generate reports
        results = {}
        for team in teams:
            # Create team directory
            team_dir = os.path.join(output_dir, team.replace(' ', '_'))
            os.makedirs(team_dir, exist_ok=True)
            
            logger.info(f"Generating reports for: {team}")
            
            # 1. Recent matches
            with engine.connect() as conn:
                query = text("""
                    SELECT * 
                    FROM recent_matches 
                    WHERE team = :team 
                    ORDER BY date DESC
                """)
                df = pd.read_sql(query, engine, params={"team": team})
                
                if not df.empty:
                    recent_file = os.path.join(team_dir, "recent_matches.csv")
                    df.to_csv(recent_file, index=False)
            
            # 2. Home/Away split
            with engine.connect() as conn:
                query = text("""
                    SELECT 
                        venue,
                        COUNT(*) as matches_played,
                        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                        SUM(gf) as goals_for,
                        SUM(ga) as goals_against,
                        SUM(gf) - SUM(ga) as goal_diff,
                        SUM(points) as total_points,
                        ROUND(AVG(points)::numeric, 2) as avg_points_per_game
                    FROM recent_matches
                    WHERE team = :team
                    GROUP BY venue
                    ORDER BY venue
                """)
                df = pd.read_sql(query, engine, params={"team": team})
                
                if not df.empty:
                    venue_file = os.path.join(team_dir, "home_away_split.csv")
                    df.to_csv(venue_file, index=False)
            
            # 3. Opponent analysis
            with engine.connect() as conn:
                query = text("""
                    SELECT 
                        opponent,
                        COUNT(*) as matches_played,
                        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                        SUM(gf) as goals_for,
                        SUM(ga) as goals_against,
                        SUM(gf) - SUM(ga) as goal_diff,
                        SUM(points) as total_points
                    FROM recent_matches
                    WHERE team = :team
                    GROUP BY opponent
                    ORDER BY total_points DESC
                """)
                df = pd.read_sql(query, engine, params={"team": team})
                
                if not df.empty:
                    opponents_file = os.path.join(team_dir, "opponent_analysis.csv")
                    df.to_csv(opponents_file, index=False)
            
            # 4. Get upcoming fixtures
            with engine.connect() as conn:
                query = text("""
                    SELECT 
                        date, 
                        CASE 
                            WHEN home_team = :team THEN away_team
                            ELSE home_team
                        END as opponent,
                        CASE 
                            WHEN home_team = :team THEN 'Home'
                            ELSE 'Away'
                        END as venue,
                        league, start_time, status
                    FROM fixtures
                    WHERE (home_team = :team OR away_team = :team)
                    AND date >= CURRENT_DATE
                    ORDER BY date ASC, start_time ASC
                    LIMIT 10
                """)
                df = pd.read_sql(query, engine, params={"team": team})
                
                if not df.empty:
                    fixtures_file = os.path.join(team_dir, "upcoming_fixtures.csv")
                    df.to_csv(fixtures_file, index=False)
            
            # Track files created
            results[team] = {
                'recent_matches': os.path.exists(os.path.join(team_dir, "recent_matches.csv")),
                'home_away_split': os.path.exists(os.path.join(team_dir, "home_away_split.csv")),
                'opponent_analysis': os.path.exists(os.path.join(team_dir, "opponent_analysis.csv")),
                'upcoming_fixtures': os.path.exists(os.path.join(team_dir, "upcoming_fixtures.csv"))
            }
        
        logger.info(f"Generated reports for {len(teams)} teams")
        return {'success': True, 'teams': len(teams), 'output_dir': output_dir, 'results': results}
    
    except Exception as e:
        logger.exception(f"Error generating team reports: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Database Export Utility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Export all tables')
    group.add_argument('--table', help='Export specific table')
    group.add_argument('--query', help='Execute custom SQL query')
    group.add_argument('--query-file', help='Execute SQL query from file')
    group.add_argument('--team-reports', action='store_true', help='Generate team-specific reports')
    
    parser.add_argument('--output-dir', default='data/exports', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.all:
            result = export_all_tables(args.output_dir)
            if result['success']:
                print(f"Successfully exported {result['tables']} tables to {args.output_dir}")
                return 0
            else:
                print(f"Failed to export all tables: {result.get('error')}")
                return 1
        
        elif args.table:
            result = export_table(args.table, args.output_dir)
            if result['success']:
                print(f"Successfully exported table '{args.table}' ({result.get('rows', 0)} rows) to {result.get('file')}")
                return 0
            else:
                print(f"Failed to export table '{args.table}': {result.get('error')}")
                return 1
        
        elif args.query:
            result = export_query(args.query, 'custom_query', args.output_dir)
            if result['success']:
                print(f"Successfully executed query ({result.get('rows', 0)} rows) to {result.get('file')}")
                return 0
            else:
                print(f"Failed to execute query: {result.get('error')}")
                return 1
        
        elif args.query_file:
            result = export_query_from_file(args.query_file, args.output_dir)
            if result['success']:
                print(f"Successfully executed query from '{args.query_file}' ({result.get('rows', 0)} rows) to {result.get('file')}")
                return 0
            else:
                print(f"Failed to execute query from '{args.query_file}': {result.get('error')}")
                return 1
        
        elif args.team_reports:
            result = generate_team_reports(args.output_dir)
            if result['success']:
                print(f"Successfully generated reports for {result.get('teams', 0)} teams in {result.get('output_dir')}")
                return 0
            else:
                print(f"Failed to generate team reports: {result.get('error')}")
                return 1
        
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())