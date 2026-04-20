#!/bin/bash

set -e  # exit on any error

echo "Offline Pipeline started"

CURR_DIR="$1"

# -----------------------------
# Validate input
# -----------------------------
if [ -z "$CURR_DIR" ]; then
    echo "Usage: ./offline.sh <workdir>"
    exit 1
fi

if [ ! -d "$CURR_DIR/images" ]; then
    echo "Error: $CURR_DIR/images does not exist"
    exit 1
fi

echo "Feature extraction..."

colmap feature_extractor \
    --database_path "$CURR_DIR/database.db" \
    --image_path "$CURR_DIR/images"

echo "Matching..."

colmap exhaustive_matcher \
    --database_path "$CURR_DIR/database.db"

echo "Sparse reconstruction..."

mkdir -p "$CURR_DIR/sparse"

colmap mapper \
    --database_path "$CURR_DIR/database.db" \
    --image_path "$CURR_DIR/images" \
    --output_path "$CURR_DIR/sparse"

echo "Completed offline pipeline"