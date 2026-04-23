#!/bin/bash

set -e

echo "Tracker Running"

WORKDIR="$1"
QUERY_DIR="$2"

# PYTHON="/home/datascience-mini/miniconda3/envs/object-tracking/bin/python3"
PYTHON="python"
# -----------------------------
# Validate inputs
# -----------------------------
if [ -z "$WORKDIR" ] || [ -z "$QUERY_DIR" ]; then
    echo "Usage:"
    echo "  ./tracker.sh <workdir> <query_images_dir>"
    echo ""
    echo "Example:"
    echo "  ./tracker.sh ./bottle ./bottle/images"
    exit 1
fi

if [ ! -d "$WORKDIR/frames" ]; then
    echo "Error: $WORKDIR/frames not found"
    exit 1
fi

if [ ! -d "$QUERY_DIR" ]; then
    echo "Error: query image directory not found: $QUERY_DIR"
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

    $PYTHON build_map.py \
        --sparse_path "$WORKDIR/sparse/0" \
        --database_path "$WORKDIR/database.db" \
        --output_path "$WORKDIR/map.npz" \
        --pruned_ply "$WORKDIR/pruned.ply" \
        --radius 0.01
else
    echo "Skipping map build (map.npz already exists)"
fi

# -----------------------------
# Stage 3: Localize all images
# -----------------------------
FRAMES_DIR="$WORKDIR/images"
mkdir -p "$FRAMES_DIR"

IMAGE_LIST=$(find "$QUERY_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)
TOTAL=$(echo "$IMAGE_LIST" | wc -l)

echo ""
echo "Found $TOTAL images in $QUERY_DIR"
echo "Running localization on each..."
echo ""

FRAME_IDX=0
SUCCESS=0
FAILED=0

for IMG in $IMAGE_LIST; do
    BASENAME=$(basename "$IMG")
    FRAME_OUT=$(printf "%s/frame_%06d.jpg" "$FRAMES_DIR" "$FRAME_IDX")

    echo -n "[$((FRAME_IDX+1))/$TOTAL] $BASENAME ... "

    if $PYTHON localise.py \
        --map_path "$WORKDIR/map.npz" \
        --image_path "$IMG" \
        --cameras_path "$WORKDIR/sparse/0/cameras.bin" \
        --output_path "$FRAME_OUT" \
        --ratio 0.85 \
        --ransac_thresh 12.0 \
        --ransac_iters 3000 \
        --gimbal_scale 0.3 2>/dev/null; then
        echo "OK"
        SUCCESS=$((SUCCESS+1))
    else
        # Localization failed — write the clean original image with just a
        # small unobtrusive label in the corner so you can judge pose accuracy
        # on surrounding frames. No overlays, no banners hiding the image.
        $PYTHON - "$IMG" "$FRAME_OUT" <<'PYEOF'
import sys
import cv2

src, dst = sys.argv[1], sys.argv[2]
img = cv2.imread(src)

# Small grey label in bottom-left — doesn't obscure the object at all
label = "no pose"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.6
thickness = 2
(tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
x, y = 12, img.shape[0] - 12
# Subtle dark background just behind the text so it's readable on any image
cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline), (30, 30, 30), -1)
cv2.putText(img, label, (x, y), font, scale, (200, 200, 200), thickness)

cv2.imwrite(dst, img)
PYEOF
        echo "no pose"
        FAILED=$((FAILED+1))
    fi

    FRAME_IDX=$((FRAME_IDX+1))
done

echo ""
echo "Localization: $SUCCESS succeeded, $FAILED failed out of $TOTAL frames"
echo ""

# -----------------------------
# Stage 4: Render video
# -----------------------------
VIDEO_OUT="$WORKDIR/tracking_result.mp4"

echo "Rendering video -> $VIDEO_OUT"

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Install with: conda install -c conda-forge ffmpeg"
    echo "Frames are saved in: $FRAMES_DIR"
    exit 1
fi

ffmpeg -y \
    -framerate 30 \
    -i "$FRAMES_DIR/frame_%06d.jpg" \
    -c:v libx264 \
    -crf 18 \
    -preset slow \
    -pix_fmt yuv420p \
    "$VIDEO_OUT"

echo ""
echo "======================================="
echo "Done."
echo "Video : $VIDEO_OUT"
echo "Frames: $FRAMES_DIR"
echo "Localized: $SUCCESS / $TOTAL frames"
echo "======================================="
echo ""
echo "Tips:"
echo "  - If success rate is low, delete map.npz and rebuild:"
echo "    rm $WORKDIR/map.npz && ./tracker.sh $WORKDIR $QUERY_DIR"
echo "  - Tune --gimbal_scale (currently 0.3) if axes look too big/small"
echo "  - Tune --ransac_thresh (currently 12.0) and --ratio (currently 0.85)"
