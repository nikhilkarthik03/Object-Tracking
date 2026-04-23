#!/bin/bash

set -e  # exit on any error

echo "Offline Pipeline (masked + semi-dense) started"

CURR_DIR="$1"

# -----------------------------
# Validate input
# -----------------------------
if [ -z "$CURR_DIR" ]; then
    echo "Usage: ./offline_masked_semidense.sh <workdir>"
    exit 1
fi

if [ ! -d "$CURR_DIR/frames" ]; then
    echo "Error: $CURR_DIR/frames does not exist"
    exit 1
fi

# if [ ! -d "$CURR_DIR/masks" ]; then
#     echo "Warning: $CURR_DIR/masks not found, proceeding without masks"
#     USE_MASKS=false
# else
#     USE_MASKS=true
# fi

USE_MASKS=false

# -----------------------------
# Feature extraction (with mask)
# -----------------------------
echo "Feature extraction..."

if [ "$USE_MASKS" = true ]; then
    colmap feature_extractor \
        --database_path "$CURR_DIR/database.db" \
        --image_path "$CURR_DIR/frames" \
        --ImageReader.mask_path "$CURR_DIR/masks" \
        --SiftExtraction.max_num_features 20000 \
        --SiftExtraction.peak_threshold 0.004
else
    colmap feature_extractor \
        --database_path "$CURR_DIR/database.db" \
        --image_path "$CURR_DIR/frames" \
        --SiftExtraction.max_num_features 20000 \
        --SiftExtraction.peak_threshold 0.004
fi

# -----------------------------
# Matching
# -----------------------------
echo "Matching..."

colmap sequential_matcher \
    --database_path "$CURR_DIR/database.db"

# -----------------------------
# Sparse reconstruction (denser tuning)
# -----------------------------
echo "Sparse reconstruction..."

mkdir -p "$CURR_DIR/sparse"

colmap global_mapper \
    --database_path "$CURR_DIR/database.db" \
    --image_path "$CURR_DIR/frames" \
    --output_path "$CURR_DIR/sparse"

# -----------------------------
# Export sparse as PLY (fast semi-dense option)
# -----------------------------
echo "Exporting sparse (enhanced) to PLY..."

colmap model_converter \
    --input_path "$CURR_DIR/sparse/0" \
    --output_path "$CURR_DIR/sparse/output_sparse.ply" \
    --output_type PLY

# -----------------------------
# OPTIONAL: lightweight semi-dense (uncomment if needed)
# -----------------------------
# This is NOT full dense, but a lighter PatchMatch setup

RUN_SEMI_DENSE=false

if [ "$RUN_SEMI_DENSE" = true ]; then
    echo "Running lightweight semi-dense reconstruction..."

    mkdir -p "$CURR_DIR/dense"

    colmap image_undistorter \
        --image_path "$CURR_DIR/frames" \
        --input_path "$CURR_DIR/sparse/0" \
        --output_path "$CURR_DIR/dense" \
        --output_type COLMAP

    colmap patch_match_stereo \
        --workspace_path "$CURR_DIR/dense" \
        --workspace_format COLMAP \
        --PatchMatchStereo.max_image_size 1000 \
        --PatchMatchStereo.window_radius 3 \
        --PatchMatchStereo.num_samples 5 \
        --PatchMatchStereo.num_iterations 3

    colmap stereo_fusion \
        --workspace_path "$CURR_DIR/dense" \
        --workspace_format COLMAP \
        --output_path "$CURR_DIR/dense/semi_dense.ply" \
        --StereoFusion.min_num_pixels 5

    echo "Semi-dense PLY saved to dense/semi_dense.ply"
fi

echo "Pipeline completed successfully"
