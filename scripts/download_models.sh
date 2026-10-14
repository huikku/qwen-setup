#!/usr/bin/env bash
# download_models.sh - Universal Model Fetcher for Local AI
# Allows one-click downloading across all hardware tiers or arbitrary custom Hugging Face models.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "${MODEL_DIR}"

echo "================================================================================"
echo "          UNIVERSAL MODEL DOWNLOADER (ALL HARDWARE TIERS & MODELS)              "
echo "================================================================================"
echo "Target Base Directory: ${MODEL_DIR}"
echo ""
echo "Select Model & Hardware Tier:"
echo ""
echo "--- ENTERPRISE & LARGE UNIFIED HARDWARE (128GB+ Thor / DGX / Mac Ultra / Multi-GPU) ---"
echo "1) Qwen 3.8 Flash-Next (125B MoE) 4-bit Q4_K_M (94 GB - Native Unified VRAM)"
echo "2) Qwen 3.8 27B UNQUANTIZED FP16 (55.6 GB - Maximum Accuracy, Full 262k context)"
echo "3) Qwen 3.8 27B Blackwell FP8 (27.8 GB - High-throughput vLLM)"
echo ""
echo "--- HIGH-END WORKSTATIONS (24GB - 48GB VRAM: RTX 3090 / 4090 / 6000 Ada) ---"
echo "4) Qwen 3.8 27B Q4_K_M + Speculative MTP Draft Head (19.0 GB Total - 66-72 tok/s)"
echo "5) Qwen 3.8 27B Q8_0 High-Precision Quant (28.5 GB)"
echo ""
echo "--- MID-RANGE & BALANCED (16GB GPUs / 32GB Mac) ---"
echo "6) Qwen 3.8 27B IQ4_XS (14.25 GB - 20k context)"
echo "7) Qwen 3.8 27B IQ3_XXS (11.0 GB - 128k context with 4-bit KV)"
echo ""
echo "--- HYBRID EXPERT OFFLOADING (12GB GPU + 48GB+ Host RAM) ---"
echo "8) Qwen 3.8 Flash-Next 3-bit for Strata (48 GB - 50-62 tok/s decode, 1750 tok/s prefill)"
echo ""
echo "--- UTILITIES & CUSTOM ---"
echo "9) Vision Projector mmproj (Vision-language input adapter - 620 MB)"
echo "10) Ollama Pull (qwen3.8:27b or qwen3.8:flash-next)"
echo "11) Custom Hugging Face Repo (Download ANY model or file directly)"
echo "12) Exit"
echo "================================================================================"

read -rp "Enter choice [1-12]: " CHOICE

check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        echo "[*] Installing huggingface_hub via uv/pip..."
        pip install --upgrade huggingface_hub[cli] || pip3 install --upgrade huggingface_hub[cli]
    fi
}

case "$CHOICE" in
    1)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 Flash-Next 4-bit (94 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-flash-next"
        huggingface-cli download unsloth/Qwen3.8-Flash-Next-GGUF Qwen3.8-Flash-Next-Q4_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-flash-next" --local-dir-use-symlinks False
        ;;
    2)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B FP16 Unquantized (55.6 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-f16.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    3)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B FP8 (~27.8 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b-fp8"
        huggingface-cli download Qwen/Qwen3.8-27B-FP8 \
            --local-dir "${MODEL_DIR}/qwen3.8-27b-fp8" --local-dir-use-symlinks False
        ;;
    4)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B Q4_K_M + MTP Draft Head (19.0 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-MTP-draft.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    5)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B Q8_0 (28.5 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    6)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B IQ4_XS (14.25 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ4_XS.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    7)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B IQ3_XXS (11.0 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ3_XXS.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    8)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 Flash-Next 3-bit GGUF (~48 GB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-flash-next"
        huggingface-cli download unsloth/Qwen3.8-Flash-Next-GGUF Qwen3.8-Flash-Next-Q3_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-flash-next" --local-dir-use-symlinks False
        ;;
    9)
        check_hf_cli
        echo "[*] Downloading Vision mmproj adapter (620 MB)..."
        mkdir -p "${MODEL_DIR}/qwen3.8-27b"
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF mmproj-model-q8_0.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        ;;
    10)
        read -rp "Enter Ollama model tag [e.g. qwen3.8:27b]: " OLLAMA_TAG
        ollama pull "$OLLAMA_TAG"
        ;;
    11)
        check_hf_cli
        read -rp "Enter Hugging Face Repo ID (e.g. Qwen/Qwen2.5-Coder-32B-Instruct): " HF_REPO
        read -rp "Enter filename or wildcard [press enter for full repo]: " HF_FILE
        TARGET_DIR="${MODEL_DIR}/$(basename "$HF_REPO")"
        mkdir -p "$TARGET_DIR"
        if [ -n "$HF_FILE" ]; then
            huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$TARGET_DIR" --local-dir-use-symlinks False
        else
            huggingface-cli download "$HF_REPO" --local-dir "$TARGET_DIR" --local-dir-use-symlinks False
        fi
        echo "[✓] Download complete to $TARGET_DIR"
        ;;
    12)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid selection."
        exit 1
        ;;
esac

echo "[✓] Operation completed successfully."
