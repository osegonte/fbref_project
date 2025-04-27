#!/usr/bin/env python3
"""
Test script to scrape a specific Premier League team
"""

import os
import pandas as pd
from fbref_stats_collector import FBrefStatsCollector

# Create a simple team fixture CSV
def create_test_fixture():
    fixtures = [
        {
            'date': '2025-05-01',
            'home_team': 'Liverpool',
            'away_team': 'Arsenal',
            'league': 'Premier League',
            'country': 'England'
        },
        {
            'date': '2025-05-03',
            'home_team': 'Manchester City',
            'away_team': 'Chelsea',
            'league': 'Premier League',
            'country': 'England'
        }
    ]
    
    df = pd.DataFrame(fixtures)
    test_file = "test_fixtures.csv"
    df.to_csv(test_file, index=False)
    print(f"Created test fixture file: {test_file}")
    return test_file

# Main function
def main():
    # Create test fixture
    fixture_file = create_test_fixture()
    
    # Configure directories
    output_dir = "data/team_stats"
    cache_dir = "data/cache"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize collector
    collector = FBrefStatsCollector(
        output_dir=output_dir,
        cache_dir=cache_dir,
        lookback=7,
        batch_size=1,
        delay_between_batches=5
    )
    
    # Run collector with the test fixture
    output_file = collector.run(
        input_file=fixture_file,
        max_teams=0  # Process all teams in our fixture
    )
    
    if output_file:
        print(f"Successfully collected team stats, saved to: {output_file}")
    else:
        print("Failed to collect team stats")

if __name__ == "__main__":
    main()