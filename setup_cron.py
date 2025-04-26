#!/usr/bin/env python3
"""
Script to set up cron job for running the FBref scraper.
This is a helper script to automatically configure scheduled runs of the scraper.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime

def get_project_dir():
    """Get the absolute path to the project directory"""
    return str(Path(__file__).resolve().parent)

def create_cron_command(project_dir, schedule='0 4 * * *'):
    """
    Create the cron command for running the scraper
    
    Args:
        project_dir: Absolute path to the project directory
        schedule: Cron schedule expression (default: run at 4 AM daily)
        
    Returns:
        Formatted cron command
    """
    script_path = os.path.join(project_dir, 'run_scraper.sh')
    
    # Make sure the script is executable
    os.chmod(script_path, 0o755)
    
    log_path = os.path.join(project_dir, 'logs', 'cron.log')
    
    # Create cron command
    cron_cmd = f"{schedule} cd {project_dir} && ./run_scraper.sh >> {log_path} 2>&1"
    
    return cron_cmd

def setup_cron(schedule):
    """
    Set up a cron job to run the scraper
    
    Args:
        schedule: Cron schedule expression
    """
    project_dir = get_project_dir()
    cron_cmd = create_cron_command(project_dir, schedule)
    
    # Get existing crontab
    try:
        existing_crontab = subprocess.check_output(['crontab', '-l']).decode('utf-8')
    except subprocess.CalledProcessError:
        # No existing crontab
        existing_crontab = ""
    
    # Check if our job is already in crontab
    if 'run_scraper.sh' in existing_crontab:
        print("Cron job already exists. Updating...")
        # Remove old job
        new_crontab_lines = []
        for line in existing_crontab.splitlines():
            if 'run_scraper.sh' not in line:
                new_crontab_lines.append(line)
        existing_crontab = '\n'.join(new_crontab_lines)
    
    # Add our job
    new_crontab = existing_crontab.strip() + '\n' + cron_cmd + '\n'
    
    # Write to temp file
    temp_file = os.path.join('/tmp', f'crontab_{datetime.now().strftime("%Y%m%d%H%M%S")}')
    with open(temp_file, 'w') as f:
        f.write(new_crontab)
    
    # Install new crontab
    try:
        subprocess.run(['crontab', temp_file], check=True)
        print(f"Cron job installed successfully to run on schedule: {schedule}")
        print(f"Command: {cron_cmd}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing crontab: {e}")
        sys.exit(1)
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description='Set up cron job for FBref scraper')
    parser.add_argument(
        '--schedule', 
        default='0 4 * * *',
        help='Cron schedule expression (default: "0 4 * * *" = run at 4 AM daily)'
    )
    
    args = parser.parse_args()
    
    print("Setting up cron job for FBref scraper...")
    setup_cron(args.schedule)

if __name__ == "__main__":
    main()