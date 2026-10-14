#!/usr/bin/env bash
# download_models.sh - Automated model & engine fetch script for Qwen 3.8 and Flash-Next
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "${MODEL_DIR}/qwen3.8-27b" "${MODEL_DIR}/qwen3.8-flash-next"

echo "=========================================================="
echo "    Qwen 3.8 Local AI Model Downloader                    "
echo "=========================================================="
echo "Destination directory: ${MODEL_DIR}"
echo ""
echo "Select which model package to download:"
echo "1) [24GB GPU] Qwen 3.8 27B Q4_K_M + Speculative Draft Head (~19 GB total)"
echo "2) [12GB/16GB GPU] Qwen 3.8 27B IQ3_XXS (~11 GB)"
echo "3) [12GB GPU + 64GB RAM] Qwen 3.8 Flash-Next 3-bit for Strata (~48 GB)"
echo "4) [THOR / 128GB] Qwen 3.8 Flash-Next 4-bit Q4_K_M (94 GB - Full MoE in Unified VRAM)"
echo "5) [THOR / 128GB] Qwen 3.8 27B High-Precision FP16 / FP8 (28-55 GB)"
echo "6) Vision Projector mmproj (~0.6 GB)"
echo "7) Pull via Ollama (qwen3.8:27b & qwen3.8:flash-next)"
echo "8) Exit"
echo ""

read -rp "Enter choice [1-8]: " CHOICE

check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        echo "[*] Installing huggingface_hub via uv/pip..."
        pip install --upgrade huggingface_hub[cli] || pip3 install --upgrade huggingface_hub[cli]
    fi
}

case "$CHOICE" in
    1)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B Q4_K_M (17.5 GB)..."
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
            
        echo "[*] Downloading Qwen 3.8 27B Speculative Draft Head (1.5 GB)..."
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-MTP-draft.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        echo "[✓] Download complete. Files saved to ${MODEL_DIR}/qwen3.8-27b"
        ;;
    2)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B IQ3_XXS (~11.0 GB)..."
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ3_XXS.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        echo "[✓] Download complete. File saved to ${MODEL_DIR}/qwen3.8-27b"
        ;;
    3)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 Flash-Next 3-bit GGUF (~48 GB)..."
        huggingface-cli download unsloth/Qwen3.8-Flash-Next-GGUF Qwen3.8-Flash-Next-Q3_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-flash-next" --local-dir-use-symlinks False
        echo "[✓] Download complete. File saved to ${MODEL_DIR}/qwen3.8-flash-next"
        ;;
    4)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 Flash-Next 4-bit Q4_K_M (94 GB) for Thor 128GB Unified Memory..."
        huggingface-cli download unsloth/Qwen3.8-Flash-Next-GGUF Qwen3.8-Flash-Next-Q4_K_M.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-flash-next" --local-dir-use-symlinks False
        echo "[✓] 94GB MoE downloaded to ${MODEL_DIR}/qwen3.8-flash-next"
        ;;
    5)
        check_hf_cli
        echo "[*] Downloading Qwen 3.8 27B FP8 (~28 GB)..."
        huggingface-cli download Qwen/Qwen3.8-27B-FP8 \
            --local-dir "${MODEL_DIR}/qwen3.8-27b-fp8" --local-dir-use-symlinks False
        echo "[✓] FP8 weights downloaded to ${MODEL_DIR}/qwen3.8-27b-fp8"
        ;;
    6)
        check_hf_cli
        echo "[*] Downloading Vision mmproj adapter (q8_0 - 620 MB)..."
        huggingface-cli download unsloth/Qwen3.8-27B-GGUF mmproj-model-q8_0.gguf \
            --local-dir "${MODEL_DIR}/qwen3.8-27b" --local-dir-use-symlinks False
        echo "[✓] Vision adapter downloaded to ${MODEL_DIR}/qwen3.8-27b/mmproj-model-q8_0.gguf"
        ;;
    7)
        if ! command -v ollama &> /dev/null; then
            echo "[!] Ollama is not installed. Install via: curl -fsSL https://ollama.com/install.sh | sh"
            exit 1
        fi
        echo "[*] Pulling Qwen 3.8 27B into Ollama..."
        ollama pull qwen3.8:27b
        echo "[*] Pulling Qwen 3.8 Flash-Next into Ollama..."
        ollama pull qwen3.8:flash-next || true
        echo "[✓] Ollama models pulled."
        ;;
    8)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid selection."
        exit 1
        ;;
esac
