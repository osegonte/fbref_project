#!/bin/bash
# Improved script to run the FBref scraper with better rate limit handling

# Set environment variables
export USE_BATCH_MODE=true  # Use batch mode to avoid rate limiting

# Verify script is not already running to avoid parallel scraping
SCRIPT_NAME=$(basename "$0")
if pgrep -f "$SCRIPT_NAME" > /dev/null; then
    echo "Another instance of $SCRIPT_NAME is already running. Exiting."
    exit 1
fi

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

# Install required packages
echo "Checking for required packages..."
pip install pandas requests beautifulsoup4 psycopg2-binary sqlalchemy python-dotenv --quiet

# Create directories if they don't exist
mkdir -p logs
mkdir -p data
mkdir -p data/cache

# Get current date for log file naming
DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/scraper_run_$DATE.log"

# Add random delay to avoid predictable patterns (1-5 minutes)
RANDOM_DELAY=$((RANDOM % 300 + 60))
echo "Adding random delay of $RANDOM_DELAY seconds before starting..."
sleep $RANDOM_DELAY

echo "Starting FBref scraper at $(date)"
echo "Logs will be saved to $LOG_FILE"

# Run the scraper with output to both console and file
python fbref_scraper.py 2>&1 | tee -a "$LOG_FILE"

# Check for successful run
if [ $? -eq 0 ]; then
    echo "Scraper completed successfully at $(date)"
    
    # Copy data to backup location
    BACKUP_DIR="data/backups/$(date +%Y%m%d)"
    mkdir -p "$BACKUP_DIR"
    
    # Copy most recent CSVs to backup
    find data -name "*.csv" -type f -not -path "*/backups/*" -exec cp {} "$BACKUP_DIR" \;
    echo "Data backed up to $BACKUP_DIR"
else
    echo "Scraper encountered errors. Check the log file for details."
fi

echo "Run completed" >> "$LOG_FILE"

# Perform cleanup of old files
echo "Cleaning up old log files..."
find logs -name "scraper_run_*.log" -type f -mtime +14 -delete

# Deactivate virtual environment
deactivate