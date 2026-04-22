#!/bin/bash

mkdir -p "$1/frames"

ffmpeg -i "$1/$1.MOV" -vf fps=30 "$1/frames/frame_%04d.jpg"

echo "Completed Frames extraction"