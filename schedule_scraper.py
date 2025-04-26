#!/usr/bin/env python3
"""
Smart scheduler for FBref scraper
Creates a distributed scraping schedule that respects rate limits
"""

import os
import sys
import argparse
import random
import logging
import json
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scheduler")

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def get_teams_list(scraper_path="optimized_fbref_scraper.py"):
    """
    Get list of Premier League teams by running scraper in list-only mode
    
    Args:
        scraper_path: Path to the scraper script
        
    Returns:
        List of team names
    """
    try:
        # Import the scraper module directly
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from optimized_fbref_scraper import FBrefScraper
        
        scraper = FBrefScraper()
        
        # Get the league page
        league_html = scraper.requester.fetch(scraper.base_url)
        if not league_html:
            logger.error("Failed to get league page")
            return []
            
        # Extract team URLs
        team_urls = scraper.extract_team_urls(league_html)
        
        # Extract team names
        teams = [scraper.extract_team_name(url) for url in team_urls]
        logger.info(f"Found {len(teams)} teams")
        
        return teams
        
    except Exception as e:
        logger.error(f"Error getting teams list: {e}")
        return []

def create_daily_teams_schedule(teams, days=7):
    """
    Distribute teams across days of the week
    
    Args:
        teams: List of team names
        days: Number of days to distribute across
        
    Returns:
        Dictionary mapping day number (0-6) to list of teams
    """
    # Randomize team order
    random.shuffle(teams)
    
    # Calculate teams per day
    teams_per_day = len(teams) // days
    remainder = len(teams) % days
    
    # Distribute teams
    schedule = {}
    
    team_index = 0
    for day in range(days):
        # Allocate teams_per_day plus 1 extra if we have remainder
        day_count = teams_per_day + (1 if day < remainder else 0)
        day_teams = teams[team_index:team_index + day_count]
        schedule[day] = day_teams
        team_index += day_count
    
    return schedule

def generate_cron_entries(schedule, script_path="optimized_fbref_scraper.py"):
    """
    Generate cron entries for the schedule
    
    Args:
        schedule: Dictionary mapping day number to list of teams
        script_path: Path to the scraper script
        
    Returns:
        List of cron command strings
    """
    cron_entries = []
    
    # Get absolute path to script
    abs_script_path = os.path.abspath(script_path)
    
    # Get project directory
    project_dir = os.path.dirname(abs_script_path)
    
    # Generate entry for each day
    for day, teams in schedule.items():
        if not teams:
            continue
            
        # Generate random time between 1-5 AM (low traffic period)
        hour = random.randint(1, 5)
        minute = random.randint(0, 59)
        
        # Format team list for command line
        team_list = ",".join(teams)
        
        # Create log file path
        log_file = f"logs/scraper_day{day}.log"
        abs_log_path = os.path.join(project_dir, log_file)
        
        # Create cron command
        cmd = f"{minute} {hour} * * {day} cd {project_dir} && python3 {abs_script_path} --teams '{team_list}' --batch-size 1 --batch-delay 30 >> {abs_log_path} 2>&1"
        
        cron_entries.append(cmd)
    
    return cron_entries

def save_schedule(schedule, filename="data/teams_schedule.json"):
    """
    Save the schedule to a file
    
    Args:
        schedule: Dictionary mapping day number to list of teams
        filename: File to save schedule to
    """
    try:
        # Add metadata
        schedule_data = {
            "created_at": datetime.now().isoformat(),
            "schedule": schedule,
            "day_names": {
                "0": "Monday",
                "1": "Tuesday",
                "2": "Wednesday",
                "3": "Thursday",
                "4": "Friday",
                "5": "Saturday",
                "6": "Sunday"
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(schedule_data, f, indent=2)
            
        logger.info(f"Saved schedule to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving schedule: {e}")
        return False

def install_cron_jobs(cron_entries):
    """
    Install cron jobs
    
    Args:
        cron_entries: List of cron command strings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get existing crontab
        current_crontab = subprocess.check_output(['crontab', '-l']).decode('utf-8')
    except subprocess.CalledProcessError:
        # No existing crontab
        current_crontab = ""
    
    # Check if our jobs are already installed
    if 'fbref_scraper' in current_crontab or 'optimized_fbref_scraper' in current_crontab:
        logger.info("FBref scraper cron jobs already exist. Updating...")
        # Remove existing jobs
        new_crontab_lines = []
        for line in current_crontab.splitlines():
            if 'fbref_scraper' not in line and 'optimized_fbref_scraper' not in line:
                new_crontab_lines.append(line)
        current_crontab = '\n'.join(new_crontab_lines)
    
    # Add new jobs
    new_crontab = current_crontab.strip() + '\n'
    new_crontab += '# FBref scraper jobs - Generated ' + datetime.now().isoformat() + '\n'
    for entry in cron_entries:
        new_crontab += entry + '\n'
    
    # Write to temp file
    temp_file = f"/tmp/fbref_crontab_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with open(temp_file, 'w') as f:
        f.write(new_crontab)
    
    # Install crontab
    try:
        subprocess.run(['crontab', temp_file], check=True)
        logger.info(f"Successfully installed {len(cron_entries)} cron jobs")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing crontab: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
def print_schedule_summary(schedule):
    """
    Print a human-readable summary of the schedule
    
    Args:
        schedule: Dictionary mapping day number to list of teams
    """
    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    
    print("\nFBref Scraper Schedule Summary:\n" + "="*30)
    
    for day_num, teams in sorted(schedule.items()):
        print(f"\n{day_names[day_num]}:")
        print("-" * 20)
        if teams:
            for i, team in enumerate(teams, 1):
                print(f"{i}. {team}")
        else:
            print("No teams scheduled")
    
    print("\n" + "="*30)
    print(f"Total: {sum(len(teams) for teams in schedule.values())} teams distributed across {len(schedule)} days")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Smart scheduler for FBref scraper")
    
    parser.add_argument("--install-cron", action="store_true",
                        help="Install cron jobs for scheduled scraping")
    parser.add_argument("--days", type=int, default=5,
                        help="Number of days to distribute teams across (default: 5)")
    parser.add_argument("--script", type=str, default="optimized_fbref_scraper.py",
                        help="Path to the scraper script")
    parser.add_argument("--print-only", action="store_true",
                        help="Just print the schedule without saving or installing")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Get teams list
    teams = get_teams_list(args.script)
    
    if not teams:
        logger.error("Failed to get teams list")
        return 1
    
    # Create schedule
    schedule = create_daily_teams_schedule(teams, args.days)
    
    # Print schedule summary
    print_schedule_summary(schedule)
    
    if args.print_only:
        return 0
    
    # Save schedule
    save_schedule(schedule)
    
    # Generate cron entries
    cron_entries = generate_cron_entries(schedule, args.script)
    
    # Install cron jobs if requested
    if args.install_cron:
        if install_cron_jobs(cron_entries):
            logger.info("Cron jobs installed successfully")
        else:
            logger.error("Failed to install cron jobs")
            return 1
    else:
        print("\nCron entries (not installed):")
        for entry in cron_entries:
            print(entry)
        print("\nUse --install-cron to install these jobs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())