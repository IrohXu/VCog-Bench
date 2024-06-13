#!/bin/bash

# Set the target directory
TARGET_DIR=""

# Find and remove all directories named "results" recursively within the target directory
find "$TARGET_DIR" -type d -name "result" -exec rm -rf {} +

echo "All 'results' subfolders have been removed from $TARGET_DIR and its subdirectories."