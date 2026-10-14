#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "    Local AI Hardware Diagnostic & Quant Picker           "
echo "=========================================================="

ARCH=$(uname -m)
OS_TYPE=$(uname -s)

# 1. System Host Memory
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_KB / 1024 / 1024}")
FREE_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
FREE_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $FREE_RAM_KB / 1024 / 1024}")

# 2. CPU Specs
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name:" | sed 's/Model name:[ \t]*//' || echo "$ARCH Processor")
CPU_CORES=$(nproc)

# 3. GPU Detection
HAS_NVIDIA=false
GPU_NAME="None"
VRAM_TOTAL_MB=0
VRAM_FREE_MB=0
IS_TEGRA_THOR=false

if [ -f "/proc/device-tree/model" ]; then
    DEVICE_MODEL=$(tr -d '\0' < /proc/device-tree/model || true)
    if [[ "$DEVICE_MODEL" =~ (Thor|Jetson|DRIVE) ]]; then
        IS_TEGRA_THOR=true
        GPU_NAME="NVIDIA Thor (Blackwell Architecture, Unified Memory)"
    fi
fi

if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    if [ "$IS_TEGRA_THOR" = false ]; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    fi
    VRAM_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)
fi

# On Tegra / Unified architectures, host RAM IS GPU VRAM
if [ "$IS_TEGRA_THOR" = true ] || (( $(echo "$VRAM_TOTAL_MB < 1000" | bc -l) )); then
    if (( $(echo "$TOTAL_RAM_GB >= 96.0" | bc -l) )); then
        VRAM_TOTAL_GB="$TOTAL_RAM_GB"
        VRAM_FREE_GB="$FREE_RAM_GB"
        UNIFIED_MEMORY=true
    else
        VRAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_TOTAL_MB / 1024}")
        VRAM_FREE_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_FREE_MB / 1024}")
        UNIFIED_MEMORY=false
    fi
else
    VRAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_TOTAL_MB / 1024}")
    VRAM_FREE_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_FREE_MB / 1024}")
    UNIFIED_MEMORY=false
fi

echo "Architecture: $ARCH"
echo "CPU:          $CPU_MODEL ($CPU_CORES execution threads)"
echo "Host RAM:     ${TOTAL_RAM_GB} GB Total (${FREE_RAM_GB} GB Available)"
echo "GPU / Device: $GPU_NAME"
echo "Addressable:  ${VRAM_TOTAL_GB} GB (Unified Memory: ${UNIFIED_MEMORY})"
echo "=========================================================="

echo ""
echo "--- AGENT / LLM DEPLOYMENT SPECIFICATION (JSON) ---"

python3 -c "
vram = float('${VRAM_TOTAL_GB}')
ram = float('${TOTAL_RAM_GB}')
arch = '${ARCH}'
is_thor = '${IS_TEGRA_THOR}' == 'true'

import json

if is_thor or (vram >= 110.0 and arch == 'aarch64') or (vram >= 110.0 and '${UNIFIED_MEMORY}' == 'true'):
    res = {
        'hardware_tier': 'NVIDIA_THOR_128GB_UNIFIED',
        'device': 'NVIDIA Thor Dev Kit (Blackwell / 128GB Unified)',
        'target_models': [
            {
                'model': 'Qwen 3.8 Flash-Next (125B MoE)',
                'quant': 'Q4_K_M (or NVFP4 / FP8)',
                'footprint_gb': '94-105 GB',
                'context_window': 131072,
                'status': 'FITS FULLY IN 128GB UNIFIED VRAM',
                'engine': 'vLLM / llama-server (CUDA aarch64)',
                'expected_tok_s': '45-60 tok/s decode, 1500+ tok/s prefill'
            },
            {
                'model': 'Qwen 3.8 27B',
                'quant': 'FP16 (Unquantized) or FP8 / Q8_0',
                'footprint_gb': '28-56 GB',
                'context_window': 262144,
                'status': 'MAXIMUM PRECISION & MASSIVE CONTEXT',
                'engine': 'vLLM / llama-server',
                'expected_tok_s': '50-80 tok/s'
            }
        ],
        'launcher_script': 'configs/thor-128gb-setup.sh',
        'notes': 'Unified memory architecture gives the GPU direct zero-copy access to the full 128GB pool. No PCIe transfer penalty.'
    }
elif vram >= 23.0:
    res = {
        'hardware_tier': '24GB_DISCRETE_GPU',
        'target_model': 'Qwen 3.8 27B',
        'quant': 'Q4_K_M',
        'model_size_gb': 17.5,
        'draft_head_enabled': True,
        'draft_model': 'qwen3.8-27b-draft.gguf (1.5 GB)',
        'kv_cache_type': 'q8_0',
        'max_ctx': 65536,
        'expected_decode_tok_s': '66-72 (with draft head) / 41-46 (base)',
        'launcher_script': 'configs/llama-server-24gb.sh'
    }
elif vram >= 15.0:
    res = {
        'hardware_tier': '16GB_DISCRETE_GPU',
        'target_model': 'Qwen 3.8 27B',
        'quant': 'IQ3_XXS (for 128k context) OR IQ4_XS (for 20k context)',
        'model_size_gb': 11.0,
        'draft_head_enabled': False,
        'kv_cache_type': 'q4_0',
        'max_ctx': 131072,
        'expected_decode_tok_s': '28-35'
    }
elif vram >= 11.0:
    if ram >= 48.0:
        res = {
            'hardware_tier': '12GB_GPU_HIGH_RAM',
            'target_model': 'Qwen 3.8 Flash-Next (125B MoE)',
            'quant': '3-bit',
            'engine': 'Strata (Expert Offloading)',
            'vram_allocation_gb': 10.5,
            'ram_allocation_gb': 48.0,
            'max_ctx': 32768,
            'expected_decode_tok_s': '50-62',
            'expected_prefill_tok_s': 1750,
            'launcher_script': 'configs/strata-12gb-setup.sh'
        }
    else:
        res = {
            'hardware_tier': '12GB_GPU_LOW_RAM',
            'target_model': 'Qwen 3.8 27B',
            'quant': 'IQ3_XXS',
            'model_size_gb': 11.0,
            'kv_cache_type': 'q8_0',
            'max_ctx': 16384,
            'expected_decode_tok_s': '28-33'
        }
else:
    res = {
        'hardware_tier': 'LOW_VRAM',
        'target_model': 'Qwen 3.8 14B or Qwen 2.5 Coder 7B'
    }

print(json.dumps(res, indent=2))
"
