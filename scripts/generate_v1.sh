#!/bin/bash
# Generate iPlotBench test dataset
#
# Usage:
#   ./scripts/generate_v1.sh           # Generate 500 figures (default)
#   ./scripts/generate_v1.sh 1000      # Generate 1,000 figures
#   ./scripts/generate_v1.sh --no-png  # Skip PNG generation (faster)
#
# Note: Total is rounded down to nearest multiple of 5 (equal figures per type).
#       e.g., 999 -> 995 (199 per type x 5 types)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default size
DEFAULT_SIZE=500

# Parse arguments
SIZE="$DEFAULT_SIZE"
NO_PNG=""

for arg in "$@"; do
    case $arg in
        --no-png)
            NO_PNG="--no-png"
            ;;
        [0-9]*)
            SIZE=$arg
            ;;
    esac
done

# Calculate actual size (rounded to multiple of 5)
PER_TYPE=$((SIZE / 5))
ACTUAL_SIZE=$((PER_TYPE * 5))

OUTPUT_DIR="v1/test"

echo ""
echo "============================================================"
echo "iPlotBench Dataset Generation"
echo "============================================================"
if [ "$SIZE" != "$ACTUAL_SIZE" ]; then
    echo "Requested: $SIZE figures"
    echo "Actual:    $ACTUAL_SIZE figures ($PER_TYPE per type x 5 types)"
else
    echo "Size: $ACTUAL_SIZE figures ($PER_TYPE per type x 5 types)"
fi
echo ""

# Clean up old data
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning up old $OUTPUT_DIR/..."
    rm -rf "$OUTPUT_DIR"
fi

# Generate source data
echo "Step 1/3: Generating source data..."
python scripts/generate_source_data.py --total "$SIZE" --seed 42
echo ""

# Convert to Plotly
echo "Step 2/3: Converting to Plotly figures..."
python scripts/convert_figureqa.py --output "$OUTPUT_DIR" $NO_PNG

# Clean up intermediate data/
echo ""
echo "Step 3/3: Cleaning up intermediate data/..."
rm -rf data/
echo ""
