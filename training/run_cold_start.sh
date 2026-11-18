#!/bin/bash
# run_cold_start.sh - Launch the supervised cold start before RL
#
# Fine-tunes the base model on chain-of-thought targets with the loss set in the
# cold_start section of the config file. The final model is saved to
# <output_dir>/final and is the starting point of RL. Configs with
# cold_start.enable set to false skip this step.
#
# Usage:
#   bash run_cold_start.sh [config]
#
# Before running, configure the variables below:
#   NUM_GPUS     - Number of GPUs, one training process per GPU

CONFIG=${1:-../configs/qwen3vl_8b.yaml}
NUM_GPUS=4

set -x

export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=$NUM_GPUS cold_start/train_sft.py config=$CONFIG
