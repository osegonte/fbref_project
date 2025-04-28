#!/usr/bin/env python3
"""
Team Data Verification Utility

This script allows you to test the data collection process for a specific team
before running the full pipeline. It helps identify issues with team name matching,
data availability, and extraction accuracy.

Usage:
  python verify_team_data.py --team "Liverpool"  # Verify data for a specific team
  python verify_team_data.py --league "Premier League"  # List all teams in a league
  python verify_team_data.py --list-leagues  # List all available leagues
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
import time
import random
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/verify_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("verify_team")

def get_db_engine():
    """Get SQLAlchemy engine from environment variables"""
    pg_uri = os.getenv('PG_URI')
    if not pg_uri:
        logger.error("PG_URI environment variable not found")
        raise ValueError("Database connection string not configured")
    
    return create_engine(pg_uri)

def create_test_fixture(team_name, league="Premier League", country="England"):
    """Create a test fixture file with just the target team"""
    output_dir = "data/verification"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a small fixture file
    fixtures = [{
        'date': datetime.now().strftime('%Y-%m-%d'),
        'home_team': team_name,
        'away_team': 'Test Opponent',
        'league': league,
        'country': country
    }]
    
    test_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_test_fixture.csv")
    pd.DataFrame(fixtures).to_csv(test_file, index=False)
    logger.info(f"Created test fixture file: {test_file}")
    
    return test_file

def test_fbref_collection(team_name, league="Premier League", country="England"):
    """Test collecting data from FBref for a specific team"""
    try:
        from fbref_stats_collector import FBrefStatsCollector
        
        logger.info(f"Testing FBref data collection for {team_name}")
        
        # Create test fixture
        test_file = create_test_fixture(team_name, league, country)
        
        # Configure directories
        output_dir = "data/verification"
        cache_dir = "data/cache"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize collector with shorter delays for testing
        collector = FBrefStatsCollector(
            output_dir=output_dir,
            cache_dir=cache_dir,
            lookback=3,            # Get only 3 matches for testing
            batch_size=1,          # Process 1 team at a time
            delay_between_batches=5  # Shorter delay for testing
        )
        
        # Run collector
        output_file = collector.run(
            input_file=test_file,
            max_teams=1
        )
        
        if output_file:
            logger.info(f"Successfully collected test data for {team_name}, saved to: {output_file}")
            
            # Load the output and show some stats
            df = pd.read_csv(output_file)
            
            # Check for null values
            null_counts = df.isnull().sum().to_dict()
            missing_fields = {k: v for k, v in null_counts.items() if v > 0}
            
            # Show basic statistics
            stats = {
                'team': team_name,
                'test_file': output_file,
                'matches_found': len(df),
                'missing_fields': missing_fields,
                'fields_collected': list(df.columns),
                'dates': df['date'].tolist() if not df.empty else []
            }
            
            print("\n=== Test Results ===")
            print(f"Team: {team_name}")
            print(f"Matches Found: {stats['matches_found']}")
            if stats['matches_found'] > 0:
                print(f"Most Recent Match: {stats['dates'][0]}")
                print(f"Fields Collected: {len(stats['fields_collected'])}")
                if missing_fields:
                    print(f"Missing Fields: {list(missing_fields.keys())}")
                else:
                    print("All fields collected successfully!")
                    
                # Show one row as an example
                print("\n=== Sample Match Data ===")
                sample = df.iloc[0].to_dict()
                for key, value in sample.items():
                    print(f"{key}: {value}")
            
            # Save test results
            result_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_test_results.json")
            with open(result_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return {'success': True, 'stats': stats, 'result_file': result_file}
        else:
            logger.error(f"Failed to collect test data for {team_name}")
            return {'success': False, 'error': 'Failed to collect test data'}
            
    except Exception as e:
        logger.exception(f"Error testing FBref collection: {e}")
        return {'success': False, 'error': str(e)}

def test_team_name_matching(team_name):
    """Test if the team name can be matched correctly in FBref"""
    try:
        from fbref_stats_collector import FBrefStatsCollector
        
        logger.info(f"Testing team name matching for {team_name}")
        
        # Create a collector instance
        collector = FBrefStatsCollector(
            output_dir="data/verification",
            cache_dir="data/cache",
            lookback=1,
            batch_size=1,
            delay_between_batches=1
        )
        
        # Top leagues to check
        leagues = [
            {"id": "9", "name": "Premier League"},
            {"id": "12", "name": "La Liga"},
            {"id": "20", "name": "Bundesliga"},
            {"id": "11", "name": "Serie A"},
            {"id": "13", "name": "Ligue 1"},
            {"id": "23", "name": "Eredivisie"},
            {"id": "32", "name": "Primeira Liga"},
            {"id": "10", "name": "Championship"}
        ]
        
        # Try to find the team in each league
        for league in leagues:
            team_id = collector.find_team_fbref_id(team_name, league["id"])
            if team_id:
                logger.info(f"Found team ID {team_id} for {team_name} in {league['name']}")
                return {
                    'success': True, 
                    'team_id': team_id, 
                    'league': league['name'], 
                    'league_id': league['id']
                }
            
            # Add a small delay between requests
            time.sleep(random.uniform(1, 2))
        
        logger.warning(f"Could not find team ID for {team_name} in any major league")
        return {'success': False, 'error': 'Team not found in any major league'}
            
    except Exception as e:
        logger.exception(f"Error testing team name matching: {e}")
        return {'success': False, 'error': str(e)}

def list_leagues():
    """List all available leagues from FBref"""
    try:
        # Define the major leagues we support
        static_leagues = [
            {"id": "9", "name": "Premier League", "country": "England"},
            {"id": "12", "name": "La Liga", "country": "Spain"},
            {"id": "20", "name": "Bundesliga", "country": "Germany"},
            {"id": "11", "name": "Serie A", "country": "Italy"},
            {"id": "13", "name": "Ligue 1", "country": "France"},
            {"id": "8", "name": "Champions League", "country": "Europe"},
            {"id": "19", "name": "Europa League", "country": "Europe"},
            {"id": "882", "name": "Conference League", "country": "Europe"},
            {"id": "23", "name": "Eredivisie", "country": "Netherlands"},
            {"id": "32", "name": "Primeira Liga", "country": "Portugal"},
            {"id": "10", "name": "Championship", "country": "England"}
        ]
        
        print("\n=== Supported Leagues ===")
        print(f"{'ID':<5} {'Country':<15} {'League':<30}")
        print("-" * 50)
        for league in static_leagues:
            print(f"{league['id']:<5} {league['country']:<15} {league['name']:<30}")
        
        return {'success': True, 'leagues': static_leagues}
            
    except Exception as e:
        logger.exception(f"Error listing leagues: {e}")
        return {'success': False, 'error': str(e)}

def list_teams_in_league(league_id, league_name=None):
    """List all teams in a specific league"""
    try:
        from fbref_stats_collector import FBrefStatsCollector
        import requests
        from bs4 import BeautifulSoup
        
        # If league name not provided, use some defaults
        if not league_name:
            league_map = {
                "9": "Premier League",
                "12": "La Liga",
                "20": "Bundesliga",
                "11": "Serie A",
                "13": "Ligue 1"
            }
            league_name = league_map.get(league_id, f"League {league_id}")
        
        logger.info(f"Listing teams in {league_name} (ID: {league_id})")
        
        # Fetch the league standings page
        url = f"https://fbref.com/en/comps/{league_id}/stats/squads/for/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        teams = []
        
        # Extract teams from the page
        for a in soup.select('a[href*="/squads/"]'):
            team_name = a.text.strip()
            if team_name and len(team_name) > 1:  # Filter out empty or single-character results
                # Extract team ID from href
                parts = a['href'].split('/')
                team_id = None
                for i, part in enumerate(parts):
                    if part == 'squads' and i+1 < len(parts):
                        team_id = parts[i+1]
                        break
                
                teams.append({
                    'name': team_name,
                    'id': team_id
                })
        
        # Sort by team name
        teams = sorted(teams, key=lambda x: x['name'])
        
        # Remove duplicates (keep first occurrence)
        unique_teams = []
        seen = set()
        for team in teams:
            if team['name'] not in seen:
                unique_teams.append(team)
                seen.add(team['name'])
        
        print(f"\n=== Teams in {league_name} ===")
        print(f"Found {len(unique_teams)} teams:")
        for i, team in enumerate(unique_teams, 1):
            print(f"{i:2d}. {team['name']} (ID: {team['id']})")
        
        # Save to file
        output_dir = "data/verification"
        os.makedirs(output_dir, exist_ok=True)
        teams_file = os.path.join(output_dir, f"{league_name.replace(' ', '_')}_teams.json")
        with open(teams_file, 'w') as f:
            json.dump(unique_teams, f, indent=2)
        
        print(f"\nTeam list saved to {teams_file}")
        
        return {'success': True, 'teams': unique_teams, 'teams_file': teams_file}
            
    except Exception as e:
        logger.exception(f"Error listing teams: {e}")
        return {'success': False, 'error': str(e)}

def check_database_for_team(team_name):
    """Check if the team exists in the database and show available data"""
    try:
        engine = get_db_engine()
        logger.info(f"Checking database for team: {team_name}")
        
        with engine.connect() as conn:
            # Check teams table
            team_query = text("SELECT * FROM teams WHERE team_name = :team")
            team_result = conn.execute(team_query, {"team": team_name})
            team_rows = [dict(zip(team_result.keys(), row)) for row in team_result]
            
            # Check fixtures
            fixture_query = text("""
                SELECT * FROM fixtures 
                WHERE home_team = :team OR away_team = :team
                ORDER BY date DESC
                LIMIT 5
            """)
            fixture_result = conn.execute(fixture_query, {"team": team_name})
            fixture_rows = [dict(zip(fixture_result.keys(), row)) for row in fixture_result]
            
            # Check match data
            match_query = text("""
                SELECT * FROM recent_matches
                WHERE team = :team
                ORDER BY date DESC
                LIMIT 5
            """)
            match_result = conn.execute(match_query, {"team": team_name})
            match_rows = [dict(zip(match_result.keys(), row)) for row in match_result]
        
        print("\n=== Database Check Results ===")
        print(f"Team: {team_name}")
        print(f"Team in database: {len(team_rows) > 0}")
        print(f"Fixtures in database: {len(fixture_rows)}")
        print(f"Match records in database: {len(match_rows)}")
        
        if match_rows:
            print("\nMost recent matches:")
            for i, match in enumerate(match_rows, 1):
                print(f"{i}. {match['date']} vs {match['opponent']} - Result: {match['result']}, Goals: {match.get('gf', 'N/A')}-{match.get('ga', 'N/A')}")
        
        # Save results to file
        output_dir = "data/verification"
        os.makedirs(output_dir, exist_ok=True)
        db_check_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_db_check.json")
        
        results = {
            'team': team_name,
            'check_time': datetime.now().isoformat(),
            'team_in_db': len(team_rows) > 0,
            'fixtures_count': len(fixture_rows),
            'matches_count': len(match_rows),
            'team_data': team_rows,
            'recent_fixtures': fixture_rows,
            'recent_matches': match_rows
        }
        
        with open(db_check_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Database check results saved to {db_check_file}")
        return {'success': True, 'results': results, 'file': db_check_file}
            
    except Exception as e:
        logger.exception(f"Error checking database: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Team Data Verification Utility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--team', help='Verify data for a specific team')
    group.add_argument('--match-test', help='Test team name matching for a team')
    group.add_argument('--league', help='List all teams in a league (provide league ID)')
    group.add_argument('--list-leagues', action='store_true', help='List all available leagues')
    group.add_argument('--db-check', help='Check database for a specific team')
    
    parser.add_argument('--league-name', help='League name (for use with --league)')
    parser.add_argument('--country', default="England", help='Country name (for use with --team)')
    
    args = parser.parse_args()
    
    if args.list_leagues:
        result = list_leagues()
        return 0 if result['success'] else 1
    
    elif args.league:
        result = list_teams_in_league(args.league, args.league_name)
        return 0 if result['success'] else 1
    
    elif args.match_test:
        result = test_team_name_matching(args.match_test)
        if result['success']:
            print(f"\nTeam Name Matching Test: SUCCESS")
            print(f"Team: {args.match_test}")
            print(f"FBref ID: {result['team_id']}")
            print(f"League: {result['league']}")
            return 0
        else:
            print(f"\nTeam Name Matching Test: FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.db_check:
        result = check_database_for_team(args.db_check)
        return 0 if result['success'] else 1
    
    elif args.team:
        # First, test team name matching
        match_result = test_team_name_matching(args.team)
        
        if match_result['success']:
            print(f"\nTeam Name Matching: SUCCESS")
            print(f"Team: {args.team}")
            print(f"FBref ID: {match_result['team_id']}")
            print(f"League: {match_result['league']}")
            
            # Now test data collection
            league = match_result['league']
            test_result = test_fbref_collection(args.team, league, args.country)
            
            if test_result['success']:
                return 0
            else:
                print(f"\nData Collection Test: FAILED")
                print(f"Error: {test_result.get('error', 'Unknown error')}")
                return 1
        else:
            print(f"\nTeam Name Matching: FAILED")
            print(f"Error: {match_result.get('error', 'Unknown error')}")
            return 1

if __name__ == "__main__":
    sys.exit(main())