#!/bin/bash
# run_train.sh - Launch RL training
#
# This is the main entry point for RL training. All settings are read from one
# config file, pass a different file to train another model.
#
# Available configs:
#   ../configs/qwen3.5_4b.yaml
#   ../configs/qwen3.5_9b.yaml
#   ../configs/qwen3vl_4b.yaml
#   ../configs/qwen3vl_8b.yaml
#
# Usage:
#   bash run_train.sh [config]

CONFIG=${1:-../configs/qwen3.5_4b.yaml}

set -x

export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main config=$CONFIG
