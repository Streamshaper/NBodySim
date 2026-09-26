#!/bin/bash

# Find all .out files recursively in the logs directory, sort by modification time, and grab the newest
LATEST_LOG=$(find logs -type f -name "*.out" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    echo "No .out log files found in the logs/ directory tree."
    exit 1
fi

echo "Following log: $LATEST_LOG"
echo "----------------------------------------"

# Run tail in the background and discard stderr to hide the "inaccessible" error
tail -f "$LATEST_LOG" 2>/dev/null &
TAIL_PID=$!

# Monitor the file. Once SLURM moves it to the date folder, the file path disappears.
while [ -f "$LATEST_LOG" ]; do
    sleep 1
done

# Clean up the tail process and exit gracefully
kill $TAIL_PID 2>/dev/null
echo "----------------------------------------"
echo "Job completed and log archived. Stopped following."