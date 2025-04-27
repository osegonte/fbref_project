#!/usr/bin/env python3
"""Script to load team statistics into the database."""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection
pg_uri = os.getenv('PG_URI')
engine = create_engine(pg_uri)

def load_stats_to_db(stats_file):
    """Load team statistics to database"""
    print(f"Loading team stats from {stats_file}")
    df = pd.read_csv(stats_file)
    
    if df.empty:
        print("No team stats to load")
        return False
    
    print(f"Processing {len(df)} match records for database storage")
    
    # Process and store matches
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
                corners_for INTEGER,
                corners_against INTEGER,
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
    for _, row in df.iterrows():
        try:
            # Convert date string to proper format if needed
            if 'date' in row and isinstance(row['date'], str):
                row['date'] = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            
            with engine.begin() as conn:
                # Use upsert (insert or update)
                conn.execute(text("""
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
                    xga = EXCLUDED.xga
                """), row.to_dict())
                
            print(f"Inserted/updated match: {row['team']} vs {row['opponent']}")
            
        except Exception as e:
            print(f"Error inserting {row['match_id']}: {e}")
    
    print(f"Successfully loaded {len(df)} matches into database")
    return True

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