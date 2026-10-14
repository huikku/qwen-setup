#!/usr/bin/env bash
# Launch Strata with Expert Offloading for Qwen 3.8 Flash-Next (125B MoE)
# Runs active weights (6B) on 12GB GPU and offloads 51B lookup table to 64GB host RAM.
# Achieves 50-62 tok/s decode and 1,750 tok/s prefill.

MODEL_PATH="${MODEL_PATH:-/models/qwen3.8-flash-next-3bit/qwen3.8-flash-next-q3.gguf}"
CTX_SIZE="${CTX_SIZE:-32768}"
PORT="${PORT:-8080}"

./strata-server \
    --model "${MODEL_PATH}" \
    --gpu-layers 32 \
    --offload-lookup-table \
    --ram-budget-gb 48 \
    --ctx-size "${CTX_SIZE}" \
    --threads $(nproc) \
    --host 0.0.0.0 --port "${PORT}"
