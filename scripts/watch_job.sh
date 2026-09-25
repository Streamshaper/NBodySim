#!/bin/bash

# Find all .out files recursively in the logs directory, sort by modification time, and grab the newest
LATEST_LOG=$(find logs -type f -name "*.out" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    echo "No .out log files found in the logs/ directory tree."
    exit 1
fi

echo "Following log: $LATEST_LOG"
echo "----------------------------------------"

# Tail the file (using -F tracks by file descriptor, so it won't break if the log gets moved to the date folder while watching it)
tail -F "$LATEST_LOG"