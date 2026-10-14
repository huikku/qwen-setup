#!/usr/bin/env bash
# multi-gpu-setup.sh - Multi-GPU Tensor/Pipeline Parallel Launcher
# Supports 2x, 4x, 8x RTX 3090 / 4090 / A6000 / H100 clusters via llama.cpp or vLLM.

set -euo pipefail

MODEL_PATH="${1:-/models/qwen3.8-27b/Qwen3.8-27B-f16.gguf}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-131072}"
GPU_COUNT=$(command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 1)

echo "================================================================================"
echo "                MULTI-GPU TENSOR PARALLEL INFERENCE LAUNCHER                   "
echo "================================================================================"
echo "Detected GPUs: $GPU_COUNT"
echo "Model:         $MODEL_PATH"
echo "Context:       $CTX_SIZE"
echo "Port:          $PORT"
echo "================================================================================"

# Launch with split-mode across all detected GPUs
llama-server \
    -m "$MODEL_PATH" \
    -ngl 999 \
    -sm row \
    -c "$CTX_SIZE" \
    -b 2048 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    --host 0.0.0.0 --port "$PORT"
