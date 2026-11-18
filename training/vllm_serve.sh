#!/bin/bash
# vllm_serve.sh - Start the vLLM server of the reward judge
#
# Serves the judge model using vLLM with tensor parallelism across multiple GPUs,
# exposing an OpenAI-compatible API. The model name and the port are read from
# worker.reward.reward_function_kwargs in the config file, so the server always
# matches what the reward function calls.
#
# Usage:
#   bash vllm_serve.sh [config]
#
# Before running, configure the variables below:
#   GPUS         - Comma-separated GPU IDs (e.g., "0,1")
#   TP_SIZE      - Tensor parallel size, should match number of GPUs
#   MAX_LEN      - Maximum sequence length
#   GPU_MEM_UTIL - Fraction of GPU memory to use (0.0-1.0)
#
# Test the server with curl:
#   curl -X POST "http://localhost:${PORT}/v1/chat/completions" \
#     -H "Content-Type: application/json" \
#     --data '{
#       "model": "'"${MODEL}"'",
#       "messages": [{"role": "user", "content": "Hello!"}]
#     }'

CONFIG=${1:-../configs/qwen3.5_4b.yaml}
GPUS="0,1"
TP_SIZE=2
MAX_LEN=8192
GPU_MEM_UTIL=0.97

# Read the judge model and port from the config file
MODEL=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').worker.reward.reward_function_kwargs.judge_model)")
PORT=$(python3 -c "from omegaconf import OmegaConf; from urllib.parse import urlparse; print(urlparse(OmegaConf.load('$CONFIG').worker.reward.reward_function_kwargs.judge_url).port)")

export CUDA_VISIBLE_DEVICES=$GPUS

vllm serve "$MODEL" \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN
