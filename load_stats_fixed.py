#!/usr/bin/env python3
"""Script to load team statistics into the database with more robust column handling."""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection
pg_uri = os.getenv('PG_URI')
engine = create_engine(pg_uri)

def load_stats_to_db(stats_file):
    """Load team statistics to database with better handling of missing columns"""
    print(f"Loading team stats from {stats_file}")
    df = pd.read_csv(stats_file)
    
    if df.empty:
        print("No team stats to load")
        return False
    
    print(f"Processing {len(df)} match records for database storage")
    
    # Print column names for debugging
    print("CSV columns:", df.columns.tolist())
    
    # Process data and ensure all required columns exist
    processed_data = []
    
    # Map CSV column names to DB column names if needed
    column_mapping = {
        'GF': 'gf',
        'GA': 'ga',
        'Sh': 'sh',
        'SoT': 'sot',
        'Dist': 'dist',
        'FK': 'fk',
        'PK': 'pk',
        'PKatt': 'pkatt',
        'Poss': 'possession',
        'xG': 'xg',
        'xGA': 'xga',
        'Round': 'round',
        'Comp': 'comp',
        'Venue': 'venue',
        'Result': 'result',
        'Date': 'date',
        'Opponent': 'opponent',
        'Team': 'team'
    }
    
    # Rename columns based on mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure all required columns exist
    required_columns = ['match_id', 'date', 'team', 'opponent', 'venue', 'result', 
                        'gf', 'ga', 'points', 'sh', 'sot', 'dist', 'fk', 'pk', 
                        'pkatt', 'possession', 'xg', 'xga', 'comp', 'round', 'season', 'is_home']
    
    # Fill missing columns with NULL/None
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Convert date strings to datetime if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Process is_home if it doesn't exist
    if 'is_home' not in df.columns and 'venue' in df.columns:
        df['is_home'] = df['venue'].apply(lambda x: x == 'Home' if pd.notna(x) else None)
    
    # Create or update table with all columns
    with engine.begin() as conn:
        # Create table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recent_matches (
                match_id VARCHAR(100) PRIMARY KEY,
                date DATE NOT NULL,
                team VARCHAR(50) NOT NULL,
                opponent VARCHAR(50) NOT NULL,
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
                xg FLOAT,
                xga FLOAT,
                comp VARCHAR(50),
                round VARCHAR(50),
                season INTEGER,
                is_home BOOLEAN,
                scrape_date TIMESTAMP
            )
        """))
        
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_recent_matches_date ON recent_matches(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_recent_matches_team ON recent_matches(team)"))
    
    # Insert data row by row
    success_count = 0
    for _, row in df.iterrows():
        try:
            # Prepare data for insertion
            data = {}
            for col in required_columns:
                # Handle special cases
                if col in row and pd.notna(row[col]):
                    data[col] = row[col]
                else:
                    data[col] = None
            
            # Insert data
            with engine.begin() as conn:
                conn.execute(text(f"""
                    INSERT INTO recent_matches 
                    (match_id, date, team, opponent, venue, result, gf, ga, points,
                     sh, sot, dist, fk, pk, pkatt, possession, xg, xga,
                     comp, round, season, is_home)
                    VALUES 
                    (:match_id, :date, :team, :opponent, :venue, :result, :gf, :ga, :points,
                     :sh, :sot, :dist, :fk, :pk, :pkatt, :possession, :xg, :xga,
                     :comp, :round, :season, :is_home)
                    ON CONFLICT (match_id) 
                    DO UPDATE SET
                    result = COALESCE(EXCLUDED.result, recent_matches.result),
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
                    xga = COALESCE(EXCLUDED.xga, recent_matches.xga)
                """), data)
            
            print(f"Inserted/updated match: {row['team']} vs {row['opponent']}")
            success_count += 1
            
        except Exception as e:
            print(f"Error inserting {row.get('match_id', 'unknown')}: {e}")
    
    print(f"Successfully loaded {success_count} of {len(df)} matches into database")
    return success_count > 0

# Main function
def main():
    # Find the most recent stats file
    stats_dir = "data/team_stats"
    files = [f for f in os.listdir(stats_dir) if f.startswith("team_match_stats_")]
    
    if not files:
        print("No team stats files found")
        return
    
    # Get the most recent file
    latest_file = max(files)
    stats_file = os.path.join(stats_dir, latest_file)
    
    # Load stats to database
    success = load_stats_to_db(stats_file)
    
    if success:
        print("Now run match analyzer: python match_analyzer.py --team Liverpool")
        print("And then export reports: python db_export.py --team-reports")

if __name__ == "__main__":
    main()