#!/usr/bin/env bash
# AI Revenue Intelligence Inference CLI Runner
# Usage: ./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# Exit immediately if a command exits with a non-zero status
set -e

DATA_DIR=${1:-"./data"}
MODEL_PATH=${2:-"./pickle/model.pkl"}
OUTPUT_PATH=${3:-"./output/predictions.csv"}

echo "=== AI Revenue Intelligence: Inference CLI ==="
echo "Data Directory: $DATA_DIR"
echo "Model Path:     $MODEL_PATH"
echo "Output Path:    $OUTPUT_PATH"
echo "=============================================="

# Run inference script
python src/predict.py "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH"
