#!/bin/bash

set -e  # stop on error

echo "Tracker Running"

WORKDIR="$1"
QUERY_IMAGE="$2"

# -----------------------------
# Validate inputs
# -----------------------------
if [ -z "$WORKDIR" ] || [ -z "$QUERY_IMAGE" ]; then
    echo "Usage:"
    echo "./tracker.sh <workdir> <query_image>"
    exit 1
fi

if [ ! -d "$WORKDIR/images" ]; then
    echo "Error: $WORKDIR/images not found"
    exit 1
fi

if [ ! -f "$QUERY_IMAGE" ]; then
    echo "Error: query image not found"
    exit 1
fi

# -----------------------------
# Run COLMAP only if needed
# -----------------------------
if [ ! -f "$WORKDIR/sparse/0/points3D.bin" ]; then
    echo "Running offline pipeline..."
    sh ./offline.sh "$WORKDIR"
else
    echo "Skipping COLMAP (already exists)"
fi

# -----------------------------
# Stage 2: Build map
# -----------------------------
if [ ! -f "$WORKDIR/map.npz" ]; then
    echo "Building map..."

    python3 build_map.py \
        --sparse_path "$WORKDIR/sparse/0" \
        --database_path "$WORKDIR/database.db" \
        --output_path "$WORKDIR/map.npz" \
        --min_views 3
else
    echo "Skipping map build (already exists)"
fi

# -----------------------------
# Stage 3: Localization
# -----------------------------
echo "Running localization..."

python3 localise.py \
    --map_path "$WORKDIR/map.npz" \
    --image_path "$QUERY_IMAGE" \
    --cameras_path "$WORKDIR/sparse/0/cameras.bin" \
    --output_path "$WORKDIR/result.jpg"

echo "Completed"
