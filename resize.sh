#!/bin/bash
# Resize all images in WHITE/ to max 1280px wide, preserving aspect ratio.
# Resized copies go into WHITE/resized/

SRC_DIR="WHITE"
OUT_DIR="WHITE/resized"
MAX_WIDTH=1280

mkdir -p "$OUT_DIR"

for img in "$SRC_DIR"/*.jpg "$SRC_DIR"/*.JPG; do
  [ -f "$img" ] || continue
  filename=$(basename "$img")
  echo "Resizing: $filename"
  sips --resampleWidth "$MAX_WIDTH" "$img" --out "$OUT_DIR/$filename"
done

echo "Done. Resized images saved to $OUT_DIR/"
