#!/usr/bin/env python3
"""
Football Data Pipeline Setup Script

This script sets up and configures the entire football data pipeline system:
1. Creates required directories
2. Installs dependencies
3. Sets up the database
4. Configures environment variables
5. Performs initial data collection

Usage:
  python setup.py --install          # Install dependencies and set up directories
  python setup.py --configure-db     # Configure and initialize the database
  python setup.py --initial-scrape   # Perform initial data scraping
  python setup.py --all              # Perform all setup steps
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime
import json
import getpass
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup")

def create_directory_structure():
    """Create all required directories for the pipeline"""
    logger.info("Creating directory structure...")
    
    # Define directories
    directories = [
        # Data directories
        "data",
        "data/cache",
        "data/matches",
        "data/matches/daily",
        "data/matches/historical",
        "data/stats",
        "data/stats/team",
        "data/stats/player",
        "data/fixtures",
        "data/fixtures/premier_league",
        "data/fixtures/champions_league",
        "data/fixtures/other_leagues",
        "data/league_tables",
        "data/exports",
        "data/reports",
        "data/analysis",
        "data/backups",
        
        # SofaScore data
        "sofascore_data",
        "sofascore_data/daily",
        "sofascore_data/raw",
        
        # Logs directory
        "logs",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return {'success': True, 'directories_created': len(directories)}

def install_dependencies():
    """Install required Python dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix
        
        if not in_venv:
            logger.warning("Not running in a virtual environment. It's recommended to use a virtual environment.")
            response = input("Continue with system-wide installation? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Installation aborted by user")
                return {'success': False, 'message': 'Installation aborted by user'}
        
        # Install dependencies from requirements.txt
        logger.info("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install specific packages needed for scraping
        additional_packages = [
            "cloudscraper",      # For SofaScore scraping
            "selenium",          # For browser automation
            "webdriver-manager", # For managing Chrome driver
        ]
        
        logger.info(f"Installing additional packages: {', '.join(additional_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + additional_packages)
        
        logger.info("Dependencies installed successfully")
        return {'success': True}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.exception(f"Unexpected error installing dependencies: {e}")
        return {'success': False, 'error': str(e)}

def configure_environment():
    """Configure environment variables"""
    logger.info("Configuring environment variables...")
    
    # Check if .env already exists
    if os.path.exists(".env"):
        logger.info(".env file already exists")
        response = input("Override existing .env file? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Environment configuration skipped")
            return {'success': True, 'message': 'Configuration skipped by user'}
    
    try:
        # Prompt for database configuration
        print("\nDatabase Configuration:")
        db_name = input("Database name [fbref]: ").strip() or "fbref"
        db_user = input("Database user [postgres]: ").strip() or "postgres"
        db_pass = getpass.getpass("Database password [password]: ") or "password"
        db_host = input("Database host [localhost]: ").strip() or "localhost"
        db_port = input("Database port [5432]: ").strip() or "5432"
        
        # Generate connection string
        pg_uri = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        
        # Scraper configuration
        print("\nScraper Configuration:")
        base_url = input("Base FBref URL [https://fbref.com/en/comps/9/Premier-League-Stats]: ").strip() \
                  or "https://fbref.com/en/comps/9/Premier-League-Stats"
        lookback = input("Match lookback count [7]: ").strip() or "7"
        batch_mode = input("Use batch mode? (true/false) [true]: ").strip().lower() or "true"
        
        # Write to .env file
        with open(".env", "w") as f:
            f.write(f"# Database Connection\n")
            f.write(f"PG_URI={pg_uri}\n")
            f.write(f"PG_DB_NAME={db_name}\n")
            f.write(f"PG_USER={db_user}\n")
            f.write(f"PG_PASSWORD={db_pass}\n")
            f.write(f"PG_HOST={db_host}\n")
            f.write(f"PG_PORT={db_port}\n\n")
            
            f.write(f"# Scraper Configuration\n")
            f.write(f"SCRAPER_BASE_URL={base_url}\n")
            f.write(f"SCRAPER_LOOKBACK={lookback}\n")
            f.write(f"USE_BATCH_MODE={batch_mode}\n\n")
            
            f.write(f"# Logging Configuration\n")
            f.write(f"LOG_LEVEL=INFO\n")
            f.write(f"LOG_FILE=logs/football_pipeline.log\n")
        
        logger.info("Environment configuration saved to .env")
        return {'success': True}
        
    except Exception as e:
        logger.exception(f"Error configuring environment: {e}")
        return {'success': False, 'error': str(e)}

def initialize_database():
    """Initialize and set up the database"""
    logger.info("Initializing database...")
    
    try:
        # Check if init_database.py exists
        if not os.path.exists("init_database.py"):
            logger.error("init_database.py not found")
            return {'success': False, 'error': 'init_database.py not found'}
        
        # Run the database initialization script
        logger.info("Creating database...")
        result = subprocess.run(
            [sys.executable, "init_database.py", "--create-db"], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Database creation failed: {result.stderr}")
            return {'success': False, 'error': result.stderr}
        
        logger.info("Setting up database schema...")
        result = subprocess.run(
            [sys.executable, "init_database.py", "--reset-db"], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Schema setup failed: {result.stderr}")
            return {'success': False, 'error': result.stderr}
        
        logger.info("Database initialized successfully")
        return {'success': True}
        
    except Exception as e:
        logger.exception(f"Error initializing database: {e}")
        return {'success': False, 'error': str(e)}

def perform_initial_scrape():
    """Perform initial data scraping"""
    logger.info("Performing initial data scraping...")
    
    try:
        # Check if pipeline_controller.py exists
        if not os.path.exists("pipeline_controller.py"):
            logger.error("pipeline_controller.py not found")
            return {'success': False, 'error': 'pipeline_controller.py not found'}
        
        # Run the pipeline to fetch fixtures
        logger.info("Fetching initial fixtures...")
        fixture_result = subprocess.run(
            [sys.executable, "pipeline_controller.py", "--fixtures-only"], 
            capture_output=True, text=True
        )
        
        if fixture_result.returncode != 0:
            logger.error(f"Fixture scraping failed: {fixture_result.stderr}")
            return {'success': False, 'error': fixture_result.stderr}
        
        # Run the pipeline to collect team stats
        logger.info("Collecting team statistics...")
        stats_result = subprocess.run(
            [sys.executable, "pipeline_controller.py", "--stats-only", "--max-teams", "5"], 
            capture_output=True, text=True
        )
        
        if stats_result.returncode != 0:
            logger.warning(f"Team stats collection failed: {stats_result.stderr}")
            logger.warning("Continuing with setup despite team stats collection failure")
        
        logger.info("Initial data scraping completed")
        return {'success': True}
        
    except Exception as e:
        logger.exception(f"Error during initial scrape: {e}")
        return {'success': False, 'error': str(e)}

def create_scheduled_tasks():
    """Create scheduled tasks for regular data collection"""
    logger.info("Setting up scheduled tasks...")
    
    try:
        # Create a simple cron-like schedule file
        schedule_file = "schedule.json"
        
        schedule = {
            "daily_fixture_update": {
                "script": "pipeline_controller.py",
                "args": ["--fixtures-only"],
                "schedule": "daily",
                "time": "06:00"
            },
            "daily_team_stats": {
                "script": "pipeline_controller.py",
                "args": ["--stats-only"],
                "schedule": "daily",
                "time": "08:00"
            },
            "match_analyzer": {
                "script": "match_analyzer.py",
                "args": ["--upcoming"],
                "schedule": "daily",
                "time": "10:00"
            },
            "weekly_export": {
                "script": "db_export.py",
                "args": ["--team-reports"],
                "schedule": "weekly",
                "day": "Monday",
                "time": "00:00"
            }
        }
        
        with open(schedule_file, "w") as f:
            json.dump(schedule, f, indent=2)
        
        # Create a basic run script
        run_script = """#!/bin/bash
# Script to run the scheduled tasks

TASK=$1
NOW=$(date +"%Y-%m-%d %H:%M:%S")

echo "[$NOW] Running task: $TASK"

case "$TASK" in
  daily_fixture_update)
    python pipeline_controller.py --fixtures-only
    ;;
  daily_team_stats)
    python pipeline_controller.py --stats-only
    ;;
  match_analyzer)
    python match_analyzer.py --upcoming
    ;;
  weekly_export)
    python db_export.py --team-reports
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

EXIT_CODE=$?
NOW=$(date +"%Y-%m-%d %H:%M:%S")

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$NOW] Task completed successfully"
else
  echo "[$NOW] Task failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
"""
        
        with open("run_task.sh", "w") as f:
            f.write(run_script)
        
        # Make the script executable
        os.chmod("run_task.sh", 0o755)
        
        logger.info("Created schedule.json and run_task.sh")
        
        # Provide instructions
        print("\nScheduled Tasks Setup")
        print("=====================")
        print("To set up regular data collection, add the following to your crontab:")
        print("  crontab -e")
        print("\nThen add these lines:")
        print("  0 6 * * * cd /path/to/football-pipeline && ./run_task.sh daily_fixture_update >> logs/cron.log 2>&1")
        print("  0 8 * * * cd /path/to/football-pipeline && ./run_task.sh daily_team_stats >> logs/cron.log 2>&1")
        print("  0 10 * * * cd /path/to/football-pipeline && ./run_task.sh match_analyzer >> logs/cron.log 2>&1")
        print("  0 0 * * 1 cd /path/to/football-pipeline && ./run_task.sh weekly_export >> logs/cron.log 2>&1")
        
        return {'success': True}
        
    except Exception as e:
        logger.exception(f"Error setting up scheduled tasks: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Football Data Pipeline Setup")
    parser.add_argument('--install', action='store_true', help='Install dependencies and set up directories')
    parser.add_argument('--configure-db', action='store_true', help='Configure and initialize the database')
    parser.add_argument('--initial-scrape', action='store_true', help='Perform initial data scraping')
    parser.add_argument('--schedule', action='store_true', help='Configure scheduled tasks')
    parser.add_argument('--all', action='store_true', help='Perform all setup steps')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 1
    
    results = {}
    
    # Setup directory structure
    if args.install or args.all:
        print("\n=== Creating Directory Structure ===")
        results['directories'] = create_directory_structure()
        
        print("\n=== Installing Dependencies ===")
        results['dependencies'] = install_dependencies()
        
        print("\n=== Configuring Environment Variables ===")
        results['environment'] = configure_environment()
    
    # Initialize database
    if args.configure_db or args.all:
        print("\n=== Initializing Database ===")
        results['database'] = initialize_database()
    
    # Perform initial scrape
    if args.initial_scrape or args.all:
        print("\n=== Performing Initial Data Scraping ===")
        results['initial_scrape'] = perform_initial_scrape()
    
    # Configure scheduled tasks
    if args.schedule or args.all:
        print("\n=== Setting Up Scheduled Tasks ===")
        results['schedule'] = create_scheduled_tasks()
    
    # Check for failures
    failures = [k for k, v in results.items() if not v.get('success', False)]
    
    if failures:
        print("\n=== Setup Summary ===")
        print("The following steps failed:")
        for failure in failures:
            error = results[failure].get('error', 'Unknown error')
            print(f"  - {failure}: {error}")
        
        print("\nSetup incomplete. Please fix the issues and retry.")
        return 1
    
    print("\n=== Setup Complete ===")
    print("The football data pipeline has been successfully set up!")
    print(f"Log file: {log_file}")
    
    # Final instructions
    print("\nNext Steps:")
    print("1. Run the pipeline controller to collect data:")
    print("   python pipeline_controller.py --full-run")
    print("2. Analyze matches:")
    print("   python match_analyzer.py --upcoming")
    print("3. Export data from the database:")
    print("   python db_export.py --team-reports")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())