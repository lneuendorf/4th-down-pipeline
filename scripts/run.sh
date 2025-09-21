#!/usr/bin/env bash
set -e

# Defaults
FORCE_DATA_UPDATE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -y|--year) YEAR="$2"; shift ;;
        -w|--week) WEEK="$2"; shift ;;
        -s|--season_type) SEASON_TYPE="$2"; shift ;;
        -f|--force) FORCE_DATA_UPDATE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure required arguments
if [[ -z "$YEAR" || -z "$WEEK" || -z "$SEASON_TYPE" ]]; then
    echo "Usage: $0 -y <year> -w <week> -s <season_type> [-f]"
    exit 1
fi

# Setup Python path
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$PROJECT_ROOT/py:$PYTHONPATH"

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

# Run recommender
python "$PROJECT_ROOT/py/4th_down_pipeline/recommender.py" \
    --year "$YEAR" \
    --week "$WEEK" \
    --season_type "$SEASON_TYPE" \
    $( [[ "$FORCE_DATA_UPDATE" == true ]] && echo "--force" )