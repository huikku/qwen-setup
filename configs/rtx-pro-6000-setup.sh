#!/usr/bin/env bash
# rtx-pro-6000-setup.sh - Professional Workstation Launcher for RTX Pro 6000 Ada (48GB) & Blackwell (96GB)
# Workloads:
# - 48GB: Qwen 3.8 27B UNQUANTIZED FP8 (27.8 GB) with full 262k context, OR Qwen 2.5 72B Q4_K_M (41.5 GB)
# - 96GB: Qwen 3.8 Flash-Next 125B MoE at 4-Bit Native (94 GB) 100% inside single GPU VRAM!

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
PORT="${PORT:-8080}"
MODE="${1:-27b-max}" # Options: 27b-max, 72b-dense, 125b-moe

echo "================================================================================"
echo "          NVIDIA RTX PRO 6000 WORKSTATION INFERENCE LAUNCHER                   "
echo "================================================================================"
echo "Target:      NVIDIA RTX Pro 6000 Ada (48GB) / Blackwell (96GB)"
echo "Mode:        $MODE"
echo "Port:        $PORT"
echo "================================================================================"

case "$MODE" in
    27b-max)
        MODEL_PATH="${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf"
        echo "[*] Launching Qwen 3.8 27B Q8_0 High-Precision (28.5 GB) with 262k context..."
        llama-server \
            -m "$MODEL_PATH" \
            -ngl 999 \
            -c 262144 \
            -b 2048 \
            -ctk q8_0 -ctv q8_0 \
            -fa \
            --host 0.0.0.0 --port "$PORT"
        ;;
        
    72b-dense)
        MODEL_PATH="${MODEL_DIR}/qwen2.5-72b/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
        echo "[*] Launching Qwen 2.5 72B Dense Q4_K_M (41.5 GB into 48GB VRAM)..."
        llama-server \
            -m "$MODEL_PATH" \
            -ngl 999 \
            -c 65536 \
            -b 2048 \
            -ctk q8_0 -ctv q8_0 \
            -fa \
            --host 0.0.0.0 --port "$PORT"
        ;;

    125b-moe)
        MODEL_PATH="${MODEL_DIR}/qwen3.8-flash-next/Qwen3.8-Flash-Next-Q4_K_M.gguf"
        echo "[*] Launching Qwen 3.8 Flash-Next (125B MoE, 94 GB) into 96GB VRAM..."
        llama-server \
            -m "$MODEL_PATH" \
            -ngl 999 \
            -c 131072 \
            -b 2048 \
            -ctk q8_0 -ctv q8_0 \
            -fa \
            --host 0.0.0.0 --port "$PORT"
        ;;

    *)
        echo "Unknown mode: $MODE. Choose: 27b-max, 72b-dense, 125b-moe"
        exit 1
        ;;
esac
