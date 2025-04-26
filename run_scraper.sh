#!/bin/bash
# Script to run the FBref scraper with appropriate settings

# Set environment variables
export USE_BATCH_MODE=true  # Use batch mode to avoid rate limiting

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Install missing packages
echo "Checking for required packages..."
pip install sqlalchemy --quiet
pip install python-dotenv --quiet

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current date for log file naming
DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/scraper_run_$DATE.log"

echo "Starting FBref scraper at $(date)"
echo "Logs will be saved to $LOG_FILE"

# Run the scraper with output to both console and file
python fbref_scraper.py 2>&1 | tee -a "$LOG_FILE"

# Check for successful run
if [ $? -eq 0 ]; then
    echo "Scraper completed successfully at $(date)"
else
    echo "Scraper encountered errors. Check the log file for details."
fi

echo "Run completed" >> "$LOG_FILE"

# Deactivate virtual environment
deactivate