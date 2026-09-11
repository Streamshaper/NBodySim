#!/bin/bash

# Find all julia_*.out files, sort them by version/number (-V), and grab the last one
LATEST_LOG=$(find logs -maxdepth 1 -name "julia_*.out" | sort -V | tail -n 1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found matching logs/julia_*.out"
    exit 1
fi

echo "Following log: $LATEST_LOG"
echo "----------------------------------------"

# Tail the file
tail -f "$LATEST_LOG"
