#!/usr/bin/env bash
set -e

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

# Run as module
python -m jobs --jobname postprocess