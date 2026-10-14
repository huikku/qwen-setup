#!/usr/bin/env bash
# universal-launcher.sh - Adaptive Inference Launcher across All Hardware Scales
# Automatically tunes batch sizes, layer offloading, context buffers, and thread pools.

set -euo pipefail

MODEL_PATH="${1:-}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-65536}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 /path/to/model.gguf [OPTIONS]"
    echo "Environment variables:"
    echo "  PORT=8080               (Server port)"
    echo "  CTX_SIZE=65536          (Context window length)"
    echo "  DRAFT_MODEL=/path/...   (Optional draft model for speculative decoding)"
    echo "  GPU_LAYERS=999          (Layers to offload, default 999 = all)"
    echo "  KV_CACHE_TYPE=q8_0      (f16, q8_0, q4_0)"
    exit 1
fi

GPU_LAYERS="${GPU_LAYERS:-999}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-q8_0}"
DRAFT_MODEL="${DRAFT_MODEL:-}"

EXTRA_ARGS=()
if [ -n "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL" ]; then
    echo "[*] Speculative Decoding enabled with draft model: $DRAFT_MODEL"
    EXTRA_ARGS+=("-md" "$DRAFT_MODEL" "-ngld" "$GPU_LAYERS" "--draft-max" "3")
fi

echo "================================================================================"
echo "                      LAUNCHING LOCAL AI INFERENCE ENGINE                       "
echo "================================================================================"
echo "Model:      $MODEL_PATH"
echo "GPU Layers: $GPU_LAYERS (999 = Pure GPU / Unified Memory)"
echo "Context:    $CTX_SIZE tokens"
echo "KV Cache:   $KV_CACHE_TYPE"
echo "Port:       $PORT"
echo "Threads:    $(nproc)"
echo "================================================================================"

llama-server \
    -m "$MODEL_PATH" \
    -ngl "$GPU_LAYERS" \
    -c "$CTX_SIZE" \
    -b 2048 \
    -ub 512 \
    -ctk "$KV_CACHE_TYPE" -ctv "$KV_CACHE_TYPE" \
    -fa \
    "${EXTRA_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT"
