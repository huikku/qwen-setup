#!/usr/bin/env bash
# thor-128gb-setup.sh
# Production inference launcher for NVIDIA DRIVE / Jetson THOR Developer Kit (128GB Unified Memory)
# Leverages Blackwell Tensor Cores, ARM64 architecture, and unified high-bandwidth LPDDR5x.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
PORT="${PORT:-8080}"
MODE="${1:-flash-next}" # Options: flash-next, 27b-fp16, 27b-fp8, 27b-q4

echo "=========================================================="
echo "    NVIDIA THOR 128GB UNIFIED INFERENCE LAUNCHER          "
echo "=========================================================="
echo "Platform:  ARM64 / Blackwell GPU (128GB Unified Memory)"
echo "Mode:      $MODE"
echo "Port:      $PORT"
echo "=========================================================="

case "$MODE" in
    flash-next)
        MODEL_PATH="${MODEL_DIR}/qwen3.8-flash-next/Qwen3.8-Flash-Next-Q4_K_M.gguf"
        echo "[*] Launching Qwen 3.8 Flash-Next (125B MoE, 4-bit Q4_K_M)..."
        echo "[*] Allocating ~94 GB into unified memory with 131,072 context..."
        
        llama-server \
            -m "$MODEL_PATH" \
            -ngl 999 \
            -c 131072 \
            -b 2048 \
            -ub 512 \
            -ctk q8_0 -ctv q8_0 \
            -fa \
            --host 0.0.0.0 --port "$PORT"
        ;;
        
    27b-fp16)
        MODEL_PATH="${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-f16.gguf"
        echo "[*] Launching Qwen 3.8 27B UNQUANTIZED FP16 (55.6 GB)..."
        echo "[*] Full 262k native context in unified memory..."
        
        llama-server \
            -m "$MODEL_PATH" \
            -ngl 999 \
            -c 262144 \
            -b 2048 \
            -ctk f16 -ctv f16 \
            -fa \
            --host 0.0.0.0 --port "$PORT"
        ;;

    27b-fp8)
        echo "[*] Launching Qwen 3.8 27B via vLLM with Blackwell FP8 / NVFP4..."
        python3 -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3.8-27B-FP8 \
            --quantization fp8 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 131072 \
            --port "$PORT" \
            --host 0.0.0.0
        ;;

    27b-q4)
        MODEL_PATH="${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf"
        DRAFT_PATH="${MODEL_DIR}/qwen3.8-27b/Qwen3.8-27B-MTP-draft.gguf"
        echo "[*] Launching Qwen 3.8 27B Q4_K_M with Speculative Draft Head..."
        
        llama-server \
            -m "$MODEL_PATH" \
            -md "$DRAFT_PATH" \
            -ngl 999 \
            -ngld 999 \
            -c 131072 \
            -ctk q8_0 -ctv q8_0 \
            -fa \
            --draft-max 3 \
            --host 0.0.0.0 --port "$PORT"
        ;;

    *)
        echo "Unknown mode: $MODE. Choose: flash-next, 27b-fp16, 27b-fp8, 27b-q4"
        exit 1
        ;;
esac
