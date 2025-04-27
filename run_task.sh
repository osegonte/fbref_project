#!/bin/bash
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
