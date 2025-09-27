#!/usr/bin/env bash
set -e

# Defaults
FORCE_DATA_UPDATE=False

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--start_year) START_YEAR="$2"; shift ;;
        -e|--end_year) END_YEAR="$2"; shift ;;
        -f|--force) FORCE_DATA_UPDATE=True ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure required arguments
if [[ -z "$START_YEAR" || -z "$END_YEAR" ]]; then
    echo "Usage: $0 -s <start_year> -e <end_year> [-f]"
    exit 1
fi

# Setup Python path
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT/py/fourth_down_pipeline:$PYTHONPATH"

# Setup conda environment
ENV_NAME="4thdown"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Activating conda environment: $ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "Conda environment '$ENV_NAME' not found. Creating..."
    conda env create -f "$PROJECT_ROOT/env.yml"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# Run job with force flag if needed
FORCE_FLAG=""
if [ "$FORCE_DATA_UPDATE" = "True" ]; then
    FORCE_FLAG="--force"
fi

# Run as module
python -m jobs \
    --jobname season_range \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    $FORCE_FLAG