#!/usr/bin/env python3
"""
Test Team Collection Script

This script allows you to test data collection for a specific team without running the full pipeline.
It creates a temporary fixture with just the target team and collects match data for it.

Usage:
  python test_team_collection.py --team "Liverpool"  # Test collection for a specific team
  python test_team_collection.py --team "Liverpool" --league "Premier League"  # Specify league
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/test_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_collection")

def create_test_fixture(team_name, league="Premier League", country="England"):
    """Create a test fixture CSV with just the target team"""
    # Create output directory
    output_dir = "data/test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a fixture with the target team
    fixture = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'home_team': team_name,
        'away_team': 'Test Opponent',
        'league': league,
        'country': country
    }
    
    # Save to CSV
    fixture_df = pd.DataFrame([fixture])
    fixture_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_test_fixture.csv")
    fixture_df.to_csv(fixture_file, index=False)
    
    logger.info(f"Created test fixture file: {fixture_file}")
    print(f"Created test fixture file: {fixture_file}")
    
    return fixture_file

def test_team_collection(team_name, league="Premier League", country="England", lookback=7):
    """Test collecting data for a specific team"""
    try:
        from fbref_stats_collector import FBrefStatsCollector
        
        logger.info(f"Testing data collection for {team_name} in {league}")
        print(f"\nTesting data collection for {team_name} in {league}")
        print("=" * 60)
        
        # Create test fixture
        fixture_file = create_test_fixture(team_name, league, country)
        
        # Configure directories
        output_dir = "data/test"
        cache_dir = "data/cache"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Looking for the last {lookback} matches...")
        
        # Initialize collector
        collector = FBrefStatsCollector(
            output_dir=output_dir,
            cache_dir=cache_dir,
            lookback=lookback,
            batch_size=1,
            delay_between_batches=5  # Shorter delay for testing
        )
        
        # Track time for performance measurement
        start_time = time.time()
        
        # Run collector
        output_file = collector.run(
            input_file=fixture_file,
            max_teams=1
        )
        
        elapsed_time = time.time() - start_time
        
        if output_file:
            logger.info(f"Successfully collected data for {team_name}, saved to: {output_file}")
            print(f"\nSuccess! Data collected in {elapsed_time:.2f} seconds")
            print(f"Output saved to: {output_file}")
            
            # Load the collected data
            df = pd.read_csv(output_file)
            
            # Basic statistics
            stats = {
                'team': team_name,
                'league': league,
                'matches_found': len(df),
                'collection_time': elapsed_time,
                'output_file': output_file,
                'null_counts': df.isnull().sum().to_dict(),
                'date_range': {
                    'earliest': df['date'].min() if not df.empty else None,
                    'latest': df['date'].max() if not df.empty else None
                }
            }
            
            # Print summary
            print(f"\nFound {stats['matches_found']} matches for {team_name}")
            
            if not df.empty:
                print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
                
                # Check for missing values
                missing_fields = {k: v for k, v in stats['null_counts'].items() if v > 0}
                if missing_fields:
                    print("\nWarning: Some fields have missing values:")
                    for field, count in missing_fields.items():
                        print(f"  - {field}: {count} missing values")
                else:
                    print("\nAll fields have complete data!")
                
                # Print match results
                print("\nMatch Results:")
                print("-" * 60)
                print(f"{'Date':<12} {'Opponent':<25} {'Venue':<8} {'Result':<8} {'Score':<8} {'xG':<6}")
                print("-" * 60)
                
                for _, row in df.iterrows():
                    date = str(row['date'])[:10]
                    opponent = row['opponent'][:25]
                    venue = row['venue']
                    result = row['result']
                    score = f"{row.get('gf', '-')}-{row.get('ga', '-')}"
                    xg = f"{row.get('xg', '-'):.2f}" if pd.notna(row.get('xg')) else '-'
                    
                    print(f"{date:<12} {opponent:<25} {venue:<8} {result:<8} {score:<8} {xg:<6}")
                
                # Save all collected data to a detailed JSON file
                detailed_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_details.json")
                all_matches = []
                
                for _, row in df.iterrows():
                    match_dict = row.to_dict()
                    # Handle non-serializable values
                    for key, value in match_dict.items():
                        if pd.isna(value):
                            match_dict[key] = None
                    all_matches.append(match_dict)
                
                details = {
                    'team': team_name,
                    'league': league,
                    'test_date': datetime.now().isoformat(),
                    'collection_time_seconds': elapsed_time,
                    'matches_count': len(df),
                    'missing_fields': missing_fields,
                    'matches': all_matches
                }
                
                with open(detailed_file, 'w') as f:
                    json.dump(details, f, indent=2)
                
                print(f"\nDetailed match data saved to: {detailed_file}")
            
            # Save summary stats
            summary_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Summary stats saved to: {summary_file}")
            
            return {'success': True, 'stats': stats}
        else:
            logger.error(f"Failed to collect data for {team_name}")
            print(f"\nFailed to collect data for {team_name}. Check the log file for details: {log_file}")
            return {'success': False, 'error': 'Collection failed'}
            
    except Exception as e:
        logger.exception(f"Error testing team collection: {e}")
        print(f"\nError: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test Team Collection Script")
    parser.add_argument('--team', required=True, help='Team name to test')
    parser.add_argument('--league', default="Premier League", help='League name (default: Premier League)')
    parser.add_argument('--country', default="England", help='Country name (default: England)')
    parser.add_argument('--lookback', type=int, default=7, help='Number of matches to collect (default: 7)')
    
    args = parser.parse_args()
    
    # Run the test
    result = test_team_collection(
        team_name=args.team,
        league=args.league,
        country=args.country,
        lookback=args.lookback
    )
    
    return 0 if result['success'] else 1

if __name__ == "__main__":
    sys.exit(main())