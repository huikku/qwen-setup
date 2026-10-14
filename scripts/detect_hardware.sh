#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "    Local AI Hardware Diagnostic & Quant Picker           "
echo "=========================================================="

# 1. System Host Memory
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_KB / 1024 / 1024}")
FREE_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
FREE_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $FREE_RAM_KB / 1024 / 1024}")

# 2. CPU Specs
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//')
CPU_CORES=$(nproc)

# 3. GPU Detection
HAS_NVIDIA=false
GPU_NAME="None"
VRAM_TOTAL_MB=0
VRAM_FREE_MB=0

if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
fi

VRAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_TOTAL_MB / 1024}")
VRAM_FREE_GB=$(awk "BEGIN {printf \"%.1f\", $VRAM_FREE_MB / 1024}")

echo "CPU:  $CPU_MODEL ($CPU_CORES execution threads)"
echo "RAM:  ${TOTAL_RAM_GB} GB Total (${FREE_RAM_GB} GB Available)"
echo "GPU:  $GPU_NAME"
echo "VRAM: ${VRAM_TOTAL_GB} GB Total (${VRAM_FREE_GB} GB Free)"
echo "=========================================================="

# 4. Recommendation Output for LLMs / Agents
echo ""
echo "--- AGENT / LLM DEPLOYMENT SPECIFICATION (JSON) ---"

python3 -c "
vram = float('${VRAM_TOTAL_GB}')
vram_free = float('${VRAM_FREE_GB}')
ram = float('${TOTAL_RAM_GB}')
has_nv = '${HAS_NVIDIA}' == 'true'

import json

if vram >= 23.0:
    res = {
        'hardware_tier': '24GB_GPU',
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
        'hardware_tier': '16GB_GPU',
        'target_model': 'Qwen 3.8 27B',
        'quant': 'IQ3_XXS (for 128k context) OR IQ4_XS (for 20k context)',
        'model_size_gb': 11.0,
        'draft_head_enabled': False,
        'kv_cache_type': 'q4_0',
        'max_ctx': 131072,
        'expected_decode_tok_s': '28-35',
        'note': 'Draft head omitted to preserve VRAM for KV cache.'
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
            'draft_head_enabled': False,
            'kv_cache_type': 'q8_0',
            'max_ctx': 16384,
            'expected_decode_tok_s': '28-33',
            'warning': 'Close browsers/desktop monitors to free VRAM slice 5.'
        }
elif ram >= 96.0:
    res = {
        'hardware_tier': '128GB_UNIFIED_MEMORY',
        'target_model': 'Qwen 3.8 Flash-Next (4-bit)',
        'quant': 'Q4_K_M',
        'engine': 'Gufo / Strata / MLX',
        'max_ctx': 131072,
        'expected_decode_tok_s': '45-75',
        'requirement': 'Check OS memory limits (Linux amdgpu.gttsize or macOS sysctl wired_mem_limit_mb).'
    }
else:
    res = {
        'hardware_tier': 'LOW_VRAM',
        'target_model': 'Qwen 3.8 14B or Qwen 2.5 Coder 7B',
        'warning': '27B requires at least 11GB VRAM to avoid the 5 tok/s CPU DDR4 cliff.'
    }

print(json.dumps(res, indent=2))
"
