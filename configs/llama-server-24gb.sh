#!/usr/bin/env bash
# Launch llama-server on 24GB GPU (RTX 3090 / 4090) with Draft Head Speculative Decoding
# Achieves 66-72 tokens/sec decode speed

MODEL_PATH="${MODEL_PATH:-/models/qwen3.8-27b/qwen3.8-27b-q4_k_m.gguf}"
DRAFT_PATH="${DRAFT_PATH:-/models/qwen3.8-27b/qwen3.8-27b-draft.gguf}"
CTX_SIZE="${CTX_SIZE:-65536}"
PORT="${PORT:-8080}"

echo "Starting llama-server with speculative decoding on port ${PORT}..."

llama-server \
    -m "${MODEL_PATH}" \
    -md "${DRAFT_PATH}" \
    -ngl 65 \
    -ngld 65 \
    -c "${CTX_SIZE}" \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    --draft-max 2 \
    --host 0.0.0.0 --port "${PORT}"
