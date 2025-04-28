#!/bin/bash
# Quick test script for verifying the fixed pipeline components

echo "========================================================="
echo "Football Data Pipeline Quick Test Tool"
echo "========================================================="
echo ""

# Function to test team data collection
test_team() {
  echo "Testing collection for team: $1"
  python test_team_collection.py --team "$1" --lookback 3
  if [ $? -eq 0 ]; then
    echo "Team test successful!"
  else
    echo "Team test failed!"
  fi
}

# Function to export team data from database
export_team() {
  echo "Exporting data for team: $1"
  python export_db_to_csv.py --team "$1"
  if [ $? -eq 0 ]; then
    echo "Export successful!"
  else
    echo "Export failed!"
  fi
}

# Function to verify team data
verify_team() {
  echo "Verifying team: $1"
  python verify_team_data.py --team "$1"
  if [ $? -eq 0 ]; then
    echo "Verification successful!"
  else
    echo "Verification failed!"
  fi
}

# Main menu
show_menu() {
  echo ""
  echo "Select an option:"
  echo "1. Test data collection for a specific team"
  echo "2. Verify team data"
  echo "3. Export team data from database"
  echo "4. Run pipeline for a specific date"
  echo "5. Export all database tables to CSV"
  echo "6. List supported leagues"
  echo "7. Exit"
  echo ""
  echo -n "Enter your choice (1-7): "
  read choice

  case $choice in
    1)
      echo -n "Enter team name: "
      read team_name
      test_team "$team_name"
      show_menu
      ;;
    2)
      echo -n "Enter team name: "
      read team_name
      verify_team "$team_name"
      show_menu
      ;;
    3)
      echo -n "Enter team name: "
      read team_name
      export_team "$team_name"
      show_menu
      ;;
    4)
      echo -n "Enter date (YYYY-MM-DD) or 'today': "
      read date_input
      if [ "$date_input" = "today" ]; then
        python pipeline_controller_fixed.py --full-run --prompt-date
      else
        python pipeline_controller_fixed.py --full-run --specific-date "$date_input"
      fi
      show_menu
      ;;
    5)
      echo "Exporting all database tables..."
      python export_db_to_csv.py --all
      show_menu
      ;;
    6)
      echo "Listing supported leagues..."
      python verify_team_data.py --list-leagues
      show_menu
      ;;
    7)
      echo "Exiting..."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please try again."
      show_menu
      ;;
  esac
}

# Start the menu
show_menu