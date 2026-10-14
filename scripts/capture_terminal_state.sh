#!/usr/bin/env bash
# capture_terminal_state.sh
# Diagnostic capture tool for LLMs and coding agents to visually or textually verify local AI states.

set -euo pipefail
OUT_DIR="/tmp/agent_captures"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Capturing diagnostic state at $TIMESTAMP..."

# 1. GPU VRAM & Compute Status
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv > "$OUT_DIR/gpu_summary_${TIMESTAMP}.csv"
    nvidia-smi > "$OUT_DIR/nvidia_smi_${TIMESTAMP}.txt"
    echo "Saved nvidia-smi metrics to $OUT_DIR/nvidia_smi_${TIMESTAMP}.txt"
fi

# 2. Running Inference Processes
ps aux | grep -E "(llama-server|ollama|strata|gufo|vllm)" | grep -v grep > "$OUT_DIR/ai_processes_${TIMESTAMP}.txt" || true

# 3. Graphical Screen Capture (if X11/Wayland display available)
if [ -n "${DISPLAY:-}" ]; then
    if command -v maim &> /dev/null; then
        maim "$OUT_DIR/screen_${TIMESTAMP}.png"
        echo "Captured desktop screenshot: $OUT_DIR/screen_${TIMESTAMP}.png"
    elif command -v import &> /dev/null; then
        import -window root "$OUT_DIR/screen_${TIMESTAMP}.png"
        echo "Captured desktop screenshot: $OUT_DIR/screen_${TIMESTAMP}.png"
    elif command -v scrot &> /dev/null; then
        scrot "$OUT_DIR/screen_${TIMESTAMP}.png"
        echo "Captured desktop screenshot: $OUT_DIR/screen_${TIMESTAMP}.png"
    fi
fi

# 4. Tmux / Terminal Pane Scrape
if [ -n "${TMUX:-}" ]; then
    tmux capture-pane -p > "$OUT_DIR/tmux_pane_${TIMESTAMP}.txt"
    echo "Saved tmux terminal buffer: $OUT_DIR/tmux_pane_${TIMESTAMP}.txt"
fi

echo "State capture complete. Output directory: $OUT_DIR"
