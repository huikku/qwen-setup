#!/usr/bin/env bash
# Advanced Speculative Decoding Launcher for llama.cpp
# Utilizes Multi-Token Prediction (MTP) draft heads combined with context N-Gram stacking.
# Achieves up to 130+ tok/s on iterative programming and code generation tasks.

MODEL_PATH="${MODEL_PATH:-/models/qwen3.8-27b/qwen3.8-27b-q4_k_m.gguf}"
DRAFT_PATH="${DRAFT_PATH:-/models/qwen3.8-27b/qwen3.8-27b-draft.gguf}"
CTX_SIZE="${CTX_SIZE:-65536}"
PORT="${PORT:-8080}"

echo "Starting llama-server with MTP Draft Head + N-Gram lookup stacking..."

llama-server \
    -m "${MODEL_PATH}" \
    -md "${DRAFT_PATH}" \
    --spec-type draft-mtp,ngram-mod,ngram-map-k4v \
    --spec-draft-n-max 12 \
    -ngl 65 \
    -ngld 65 \
    -c "${CTX_SIZE}" \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    --temp 0.6 \
    --host 0.0.0.0 --port "${PORT}"
