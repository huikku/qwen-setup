#!/usr/bin/env bash
# rtx-5090-setup.sh - Extreme Bandwidth Launcher for NVIDIA RTX 5090 (32GB VRAM)
# Memory Bandwidth: 1,792 GB/s (GDDR7)
# Configuration: Qwen 3.8 27B Q6_K / Q8_0 High-Precision OR Q4_K_M with 131k-262k context
# Throughput: 65-85 tok/s (base) -> 130-160+ tok/s (speculative)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
# Options: Q6_K (22 GB), Q8_0 (28.5 GB), or Q4_K_M (17.5 GB with 131k context)
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-Q6_K.gguf}"
DRAFT_PATH="${DRAFT_PATH:-${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-MTP-draft.gguf}"
CTX_SIZE="${CTX_SIZE:-131072}"
PORT="${PORT:-8080}"

echo "================================================================================"
echo "          NVIDIA RTX 5090 (32GB GDDR7) HIGH-PRECISION LAUNCHER                 "
echo "================================================================================"
echo "GPU:         NVIDIA GeForce RTX 5090 (32GB GDDR7 @ 1,792 GB/s)"
echo "Model:       $MODEL_PATH"
echo "Context:     $CTX_SIZE tokens"
echo "Port:        $PORT"
echo "================================================================================"

EXTRA_DRAFT_ARGS=()
if [ -f "$DRAFT_PATH" ]; then
    echo "[✓] Speculative Decoding enabled: Bandwidth multiplier active"
    EXTRA_DRAFT_ARGS=("-md" "$DRAFT_PATH" "-ngld" "65" "--draft-max" "4")
fi

llama-server \
    -m "$MODEL_PATH" \
    -ngl 65 \
    -c "$CTX_SIZE" \
    -b 4096 \
    -ub 1024 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    "${EXTRA_DRAFT_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT"
