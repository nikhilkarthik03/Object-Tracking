#!/bin/bash

WORKDIR="$1"

sh ffmpeg.sh "$WORKDIR"

# python mask.py \
#     --input_dir "$WORKDIR/frames" \
#     --output_dir "$WORKDIR/masks" \
#     --checkpoint sam_vit_b_01ec64.pth \
#     --model_type vit_b \
#     --device cuda

bash ./offline.sh "$WORKDIR"