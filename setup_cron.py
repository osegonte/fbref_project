#!/usr/bin/env python3
"""
Enhanced cron setup script for FBref scraper that uses randomized timing
and distributes scraping across the week to avoid detection.
"""

import os
import sys
import subprocess
import random
from pathlib import Path
import argparse
from datetime import datetime

def get_project_dir():
    """Get the absolute path to the project directory"""
    return str(Path(__file__).resolve().parent)

def create_batch_cron_commands(project_dir):
    """
    Create multiple cron commands for distributed scraping throughout the week
    using the batch_scraper.py script, which processes different teams each day.
    
    Args:
        project_dir: Absolute path to the project directory
        
    Returns:
        List of formatted cron commands
    """
    script_path = os.path.join(project_dir, 'batch_scraper.py')
    
    # Make sure the script is executable
    os.chmod(script_path, 0o755)
    
    log_dir = os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate commands for each weekday with randomized timing
    commands = []
    
    # Monday - Friday: Run once per day at a random time
    for day in range(1, 6):  # 1-5 for Monday-Friday
        # Random hour between 1 AM and 5 AM (low traffic time)
        hour = random.randint(1, 5)
        minute = random.randint(0, 59)
        
        # Create cron string
        # Format: minute hour * * day-of-week
        cron_time = f"{minute} {hour} * * {day}"
        log_path = os.path.join(log_dir, f"batch_scraper_weekday_{day}.log")
        
        cmd = f"{cron_time} cd {project_dir} && python3 {script_path} >> {log_path} 2>&1"
        commands.append(cmd)
    
    # Weekend: Only run Saturday at a random time (skip Sunday to reduce load)
    sat_hour = random.randint(2, 6)  # Slightly later on weekend
    sat_minute = random.randint(0, 59)
    
    sat_cron_time = f"{sat_minute} {sat_hour} * * 6"  # 6 = Saturday
    sat_log_path = os.path.join(log_dir, "batch_scraper_weekend.log")
    
    sat_cmd = f"{sat_cron_time} cd {project_dir} && python3 {script_path} >> {sat_log_path} 2>&1"
    commands.append(sat_cmd)
    
    return commands

def setup_cron():
    """Set up the cron jobs for distributed scraping"""
    project_dir = get_project_dir()
    cron_commands = create_batch_cron_commands(project_dir)
    
    # Get existing crontab
    try:
        existing_crontab = subprocess.check_output(['crontab', '-l']).decode('utf-8')
    except subprocess.CalledProcessError:
        # No existing crontab
        existing_crontab = ""
    
    # Check if our jobs are already in crontab
    if 'batch_scraper.py' in existing_crontab:
        print("FBref scraper cron jobs already exist. Updating...")
        # Remove old jobs
        new_crontab_lines = []
        for line in existing_crontab.splitlines():
            if 'batch_scraper.py' not in line and 'fbref_scraper.py' not in line:
                new_crontab_lines.append(line)
        existing_crontab = '\n'.join(new_crontab_lines)
    
    # Add our new jobs
    new_crontab = existing_crontab.strip() + '\n'
    for cmd in cron_commands:
        new_crontab += cmd + '\n'
    
    # Write to temp file
    temp_file = os.path.join('/tmp', f'crontab_{datetime.now().strftime("%Y%m%d%H%M%S")}')
    with open(temp_file, 'w') as f:
        f.write(new_crontab)
    
    # Install new crontab
    try:
        subprocess.run(['crontab', temp_file], check=True)
        print(f"Successfully installed {len(cron_commands)} cron jobs for distributed scraping")
        for i, cmd in enumerate(cron_commands):
            print(f"Job {i+1}: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing crontab: {e}")
        sys.exit(1)
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description='Set up improved cron jobs for FBref scraper')
    parser.add_argument('--preview', action='store_true', help='Preview cron jobs without installing')
    
    args = parser.parse_args()
    
    project_dir = get_project_dir()
    cron_commands = create_batch_cron_commands(project_dir)
    
    if args.preview:
        print("Preview of cron jobs (will not be installed):")
        for i, cmd in enumerate(cron_commands):
            print(f"Job {i+1}: {cmd}")
    else:
        print("Setting up distributed cron jobs for FBref scraper...")
        setup_cron()

if __name__ == "__main__":
    main()