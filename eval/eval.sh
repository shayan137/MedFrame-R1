#!/bin/bash
# eval.sh - Run the full evaluation of one model
#
# Generates responses, scores them with fixed rules and each judge listed in the
# eval section of the config file, merges the judge votes and prints the scores.
#
# Usage:
#   bash eval.sh [config]

CONFIG=${1:-../configs/qwen3.5_4b.yaml}

source .env # load physionet credentials from .env file
export PHYSIONET_UNAME
export PHYSIONET_PWD

# Set environment variables for evaluation
export HF_HUB_READ_TIMEOUT=600
export HF_HUB_CONNECTION_TIMEOUT=1200

export OMP_NUM_THREADS=5
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Run evaluation
python eval.py \
    --config "$CONFIG" \
    --generate true \
    --evaluate true \
    --score true

if [ $? -ne 0 ]; then
    echo ""
    echo "Evaluation failed. Check the error above."
    exit 1
fi
