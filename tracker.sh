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
    echo "  ./tracker.sh <workdir> <query_image>"
    echo ""
    echo "Example:"
    echo "  ./tracker.sh ./my_scene ./query.jpg"
    exit 1
fi

if [ ! -d "$WORKDIR/images" ]; then
    echo "Error: $WORKDIR/images not found"
    exit 1
fi

if [ ! -f "$QUERY_IMAGE" ]; then
    echo "Error: query image not found: $QUERY_IMAGE"
    exit 1
fi

# -----------------------------
# Run COLMAP only if needed
# -----------------------------
if [ ! -f "$WORKDIR/sparse/0/points3D.bin" ]; then
    echo "Running offline COLMAP pipeline..."
    bash ./offline.sh "$WORKDIR"
else
    echo "Skipping COLMAP (sparse/0/points3D.bin already exists)"
fi

# -----------------------------
# Stage 2: Build map
# -----------------------------
if [ ! -f "$WORKDIR/map.npz" ]; then
    echo "Building 3D descriptor map..."

    python3 build_map.py \
        --sparse_path "$WORKDIR/sparse/0" \
        --database_path "$WORKDIR/database.db" \
        --output_path "$WORKDIR/map.npz" \
        --min_views 3 \
        --max_descs_per_point 4
else
    echo "Skipping map build (map.npz already exists)"
fi

# -----------------------------
# Stage 3: Localization
# FIX (Bug 5): was "localise.py" (British spelling) — now consistent with
# the actual filename "localize.py". Mismatched spelling caused a hard crash
# on case-sensitive filesystems (Linux).
# -----------------------------
echo "Running localization..."

python3 localise.py \
    --map_path "$WORKDIR/map.npz" \
    --image_path "$QUERY_IMAGE" \
    --cameras_path "$WORKDIR/sparse/0/cameras.bin" \
    --output_path "$WORKDIR/result.jpg" \
    --gimbal_scale 0.3

echo ""
echo "Done. Output saved to $WORKDIR/result.jpg"
echo ""
echo "Tip: if the gimbal scale looks wrong (too small or giant), adjust"
echo "     --gimbal_scale to match your scene's world unit size."