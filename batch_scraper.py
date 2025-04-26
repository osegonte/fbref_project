#!/usr/bin/env python3
"""
Batch scheduler script for FBref scraper to run on different days of the week.
This approach distributes scraping across multiple days to avoid rate limiting.
"""

import os
import sys
import time
import random
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Make sure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our scraper modules
from fbref_scraper import FBrefScraper, FBrefDatabaseManager
from improved_rate_limit_handler import RateLimitHandler

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/batch_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("batch_scraper")

def get_team_subset_for_day(all_team_urls, day_of_week):
    """
    Get a subset of teams based on the day of the week
    
    Args:
        all_team_urls: List of all team URLs
        day_of_week: Day of the week (0-6, where 0 is Monday)
        
    Returns:
        List of team URLs for the specified day
    """
    # Split teams into 5 groups (Monday-Friday)
    group_count = 5
    team_count = len(all_team_urls)
    teams_per_group = team_count // group_count
    
    # Adjust for weekends - if it's weekend, choose a random subset
    if day_of_week >= 5:  # Saturday or Sunday
        # On weekends, only process a small subset (20% of teams)
        sample_size = max(2, team_count // 5)
        logger.info(f"Weekend day ({day_of_week}), selecting random subset of {sample_size} teams")
        return random.sample(all_team_urls, sample_size)
    
    # For weekdays, divide evenly
    start_idx = day_of_week * teams_per_group
    end_idx = (day_of_week + 1) * teams_per_group if day_of_week < group_count - 1 else team_count
    
    day_teams = all_team_urls[start_idx:end_idx]
    logger.info(f"Selected {len(day_teams)} teams for day {day_of_week} (out of {team_count} total)")
    return day_teams

def get_processed_teams():
    """
    Get the list of already processed teams from previous runs
    
    Returns:
        Set of team names that have been processed
    """
    state_file = os.path.join("data", "batch_processed_teams.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            processed = set(state.get("processed_teams", []))
            logger.info(f"Loaded {len(processed)} previously processed teams")
            return processed
        except Exception as e:
            logger.warning(f"Failed to load processed teams: {e}")
    
    return set()

def save_processed_teams(processed_teams):
    """
    Save the list of processed teams
    
    Args:
        processed_teams: Set or list of team names that have been processed
    """
    state_file = os.path.join("data", "batch_processed_teams.json")
    state = {
        "timestamp": datetime.now().isoformat(),
        "processed_teams": list(processed_teams)
    }
    
    try:
        os.makedirs("data", exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f)
        logger.info(f"Saved {len(processed_teams)} processed teams to state file")
        return True
    except Exception as e:
        logger.error(f"Failed to save processed teams: {e}")
        return False

def merge_results_with_existing(new_df, filename="recent_matches.csv"):
    """
    Merge new results with existing data file
    
    Args:
        new_df: DataFrame with new data
        filename: Filename to merge with
        
    Returns:
        Combined DataFrame
    """
    filepath = os.path.join("data", filename)
    
    # If file doesn't exist or is empty, return new data
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        logger.info(f"No existing data file found at {filepath}, using only new data")
        return new_df
    
    try:
        existing_df = pd.read_csv(filepath)
        logger.info(f"Loaded existing data with {len(existing_df)} rows")
        
        # Combine data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates
        if 'match_id' in combined_df.columns:
            before_count = len(combined_df)
            combined_df.drop_duplicates(subset=['match_id'], inplace=True)
            after_count = len(combined_df)
            logger.info(f"Removed {before_count - after_count} duplicate matches based on match_id")
        elif all(col in combined_df.columns for col in ['date', 'team', 'opponent']):
            before_count = len(combined_df)
            combined_df.drop_duplicates(subset=['date', 'team', 'opponent'], inplace=True)
            after_count = len(combined_df)
            logger.info(f"Removed {before_count - after_count} duplicate matches based on date+team+opponent")
        
        return combined_df
    except Exception as e:
        logger.error(f"Error merging with existing data: {e}")
        return new_df

def main():
    """Main function to run the batch scraper"""
    try:
        logger.info("Starting batch scraper")
        
        # Get current day of week (0 is Monday, 6 is Sunday)
        day_of_week = datetime.now().weekday()
        logger.info(f"Current day of the week: {day_of_week}")
        
        # Initialize the scraper
        scraper = FBrefScraper(
            min_delay=20,         # Increased minimum delay
            max_delay=45,         # Increased maximum delay
            max_cache_age=168     # Cache for 1 week (hours)
        )
        
        # Get the base league page
        base_html = scraper.get_html(scraper.base_url)
        if not base_html:
            logger.error("Failed to get base league page, aborting")
            return False
        
        # Extract all team URLs
        all_team_urls = scraper.extract_team_urls(base_html)
        
        if not all_team_urls:
            logger.error("No team URLs found, aborting")
            return False
            
        logger.info(f"Found {len(all_team_urls)} total teams")
        
        # Get the subset of teams for today
        teams_for_today = get_team_subset_for_day(all_team_urls, day_of_week)
        logger.info(f"Selected {len(teams_for_today)} teams to process today")
        
        # Get already processed teams
        processed_teams = get_processed_teams()
        
        # Filter out already processed teams
        teams_to_process = []
        for team_url in teams_for_today:
            team_name = scraper.extract_team_name(team_url)
            if team_name not in processed_teams:
                teams_to_process.append(team_url)
            else:
                logger.info(f"Skipping already processed team: {team_name}")
        
        logger.info(f"Found {len(teams_to_process)} unprocessed teams for today")
        
        # If all teams already processed, we might reset once per week
        if not teams_to_process and day_of_week == 0:  # Monday
            logger.info("No new teams to process on Monday, resetting processed teams list")
            processed_teams.clear()
            save_processed_teams(processed_teams)
            teams_to_process = teams_for_today
            logger.info(f"Reset processed teams, now processing {len(teams_to_process)} teams")
        
        # If still nothing to process, we're done
        if not teams_to_process:
            logger.info("No teams to process today, exiting")
            return True
        
        # Process each team one by one with delays between them
        all_team_dfs = []
        newly_processed = set()
        
        for team_url in teams_to_process:
            team_name = scraper.extract_team_name(team_url)
            logger.info(f"Processing team: {team_name}")
            
            try:
                # Parse match data for this team
                team_df = scraper.parse_matches_and_stats(team_url, scraper.years[0])
                
                if team_df is not None and not team_df.empty:
                    logger.info(f"Successfully scraped {len(team_df)} matches for {team_name}")
                    all_team_dfs.append(team_df)
                    
                    # Mark as processed
                    processed_teams.add(team_name)
                    newly_processed.add(team_name)
                    save_processed_teams(processed_teams)
                else:
                    logger.warning(f"No valid data found for {team_name}")
            except Exception as e:
                logger.error(f"Error processing team {team_name}: {e}")
            
            # Add a significant delay between teams (30-90 seconds)
            delay = random.uniform(30, 90)
            logger.info(f"Waiting {delay:.1f} seconds before next team...")
            time.sleep(delay)
        
        # Combine all the data
        if all_team_dfs:
            new_df = pd.concat(all_team_dfs, ignore_index=True)
            logger.info(f"Collected {len(new_df)} new matches from {len(all_team_dfs)} teams")
            
            # Standardize column names
            new_df.columns = [c.lower() for c in new_df.columns]
            
            # Create match_id if not present
            if 'match_id' not in new_df.columns and all(col in new_df.columns for col in ['date', 'team', 'opponent']):
                new_df['match_id'] = new_df.apply(
                    lambda row: f"{row['date']}_{row['team']}_{row['opponent']}".replace(' ', '_'),
                    axis=1
                )
            
            # Add scrape_date
            new_df['scrape_date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Merge with existing data
            combined_df = merge_results_with_existing(new_df)
            
            # Save the combined data
            os.makedirs("data", exist_ok=True)
            output_path = os.path.join("data", "recent_matches.csv")
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(combined_df)} total matches to {output_path}")
            
            # Make a backup with timestamp
            backup_path = os.path.join("data", f"matches_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            combined_df.to_csv(backup_path, index=False)
            logger.info(f"Created backup at {backup_path}")
            
            # Store in database if configured
            db_uri = os.getenv('PG_URI')
            if db_uri:
                try:
                    db_manager = FBrefDatabaseManager(connection_uri=db_uri)
                    db_manager.initialize_db()
                    count = db_manager.store_recent_matches(combined_df)
                    logger.info(f"Stored/updated {count} matches in the database")
                except Exception as e:
                    logger.error(f"Database error: {e}")
            
            logger.info(f"Successfully processed {len(newly_processed)} new teams: {', '.join(newly_processed)}")
            return True
        else:
            logger.warning("No new data collected")
            return False
            
    except Exception as e:
        logger.error(f"Batch scraper failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Add a random startup delay to avoid predictable patterns
    startup_delay = random.uniform(1, 60)  # 1-60 seconds
    logger.info(f"Waiting {startup_delay:.1f} seconds before starting...")
    time.sleep(startup_delay)
    
    success = main()
    
    if success:
        logger.info("Batch scraper completed successfully")
        sys.exit(0)
    else:
        logger.error("Batch scraper encountered errors")
        sys.exit(1)
        