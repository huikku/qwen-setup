#!/usr/bin/env bash
# detect_hardware.sh - Universal Hardware Profiler & LLM Deployment Recommender
# Evaluates single-GPU, multi-GPU, unified memory (Thor/Jetson/Mac), and CPU topologies.
set -euo pipefail

echo "================================================================================"
echo "          UNIVERSAL LOCAL AI HARDWARE PROFILER & CONFIGURATION ENGINE           "
echo "================================================================================"

ARCH=$(uname -m)
OS_NAME=$(uname -s)

# 1. CPU & Instruction Sets
CPU_CORES=$(nproc)
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name:" | sed 's/Model name:[ \t]*//' || echo "$ARCH Processor")
HAS_AVX512=$(grep -q "avx512" /proc/cpuinfo 2>/dev/null && echo true || echo false)
HAS_AMX=$(grep -q "amx" /proc/cpuinfo 2>/dev/null && echo true || echo false)
HAS_ARM_NEON=$([ "$ARCH" = "aarch64" ] && echo true || echo false)

# 2. Host System RAM
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_KB / 1024 / 1024}")
FREE_RAM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
FREE_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $FREE_RAM_KB / 1024 / 1024}")

# 3. GPU Discovery (NVIDIA CUDA / Tegra / AMD ROCm / Apple Metal)
GPU_COUNT=0
GPU_NAMES=()
TOTAL_VRAM_MB=0
FREE_VRAM_MB=0
IS_UNIFIED=false
HARDWARE_PLATFORM="Generic_Linux"

# Check for NVIDIA Thor / Jetson / DRIVE
if [ -f "/proc/device-tree/model" ]; then
    DEVICE_TREE_MODEL=$(tr -d '\0' < /proc/device-tree/model || true)
    if [[ "$DEVICE_TREE_MODEL" =~ (Thor|Jetson|DRIVE) ]]; then
        IS_UNIFIED=true
        HARDWARE_PLATFORM="NVIDIA_THOR_OR_JETSON"
        GPU_NAMES+=("$DEVICE_TREE_MODEL (Blackwell Unified Memory)")
    fi
fi

# NVIDIA SMI Discovery
if command -v nvidia-smi &> /dev/null; then
    NV_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$NV_COUNT" -gt 0 ]; then
        GPU_COUNT=$NV_COUNT
        while IFS= read -r line; do
            GPU_NAMES+=("$line")
        done < <(nvidia-smi --query-gpu=name --format=csv,noheader)
        
        TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
        FREE_VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    fi
fi

# Check for Apple Silicon
if [ "$OS_NAME" = "Darwin" ]; then
    IS_UNIFIED=true
    HARDWARE_PLATFORM="Apple_Silicon"
    TOTAL_RAM_BYTES=$(sysctl -n hw.memsize)
    TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_RAM_BYTES / 1024 / 1024 / 1024}")
    GPU_NAMES+=("Apple M-Series GPU (Unified Memory)")
    TOTAL_VRAM_MB=$(awk "BEGIN {printf \"%.0f\", $TOTAL_RAM_GB * 1024 * 0.75}")
fi

# If unified memory platform, addressable memory is host memory
if [ "$IS_UNIFIED" = true ] || [ "$HARDWARE_PLATFORM" = "NVIDIA_THOR_OR_JETSON" ]; then
    TOTAL_ADDRESSABLE_GB="$TOTAL_RAM_GB"
    FREE_ADDRESSABLE_GB="$FREE_RAM_GB"
    IS_UNIFIED=true
else
    TOTAL_ADDRESSABLE_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_VRAM_MB / 1024}")
    FREE_ADDRESSABLE_GB=$(awk "BEGIN {printf \"%.1f\", $FREE_VRAM_MB / 1024}")
fi

echo "Architecture:        $ARCH ($OS_NAME)"
echo "CPU Specs:           $CPU_MODEL ($CPU_CORES execution threads)"
echo "Accelerators:        ${#GPU_NAMES[@]} detected"
for g in "${GPU_NAMES[@]}"; do
    echo "  - $g"
done
echo "Host RAM:            ${TOTAL_RAM_GB} GB Total (${FREE_RAM_GB} GB Free)"
echo "Total Addressable:   ${TOTAL_ADDRESSABLE_GB} GB (Unified Pool: ${IS_UNIFIED})"
echo "================================================================================"

echo ""
echo "--- AUTONOMOUS AGENT DEPLOYMENT DIRECTIVE (JSON) ---"

python3 -c "
vram = float('${TOTAL_ADDRESSABLE_GB}')
ram = float('${TOTAL_RAM_GB}')
gpu_count = int('${GPU_COUNT}')
is_unified = '${IS_UNIFIED}' == 'true'
platform = '${HARDWARE_PLATFORM}'
arch = '${ARCH}'
gpu_names = '''${GPU_NAMES[*]}'''

import json

tier = 'UNKNOWN'
recommendations = []
launcher = 'configs/universal-launcher.sh'

if 'Thor' in gpu_names or (is_unified and vram >= 120.0):
    tier = 'NVIDIA_THOR_128GB_UNIFIED'
    launcher = 'configs/thor-128gb-setup.sh'
    recommendations = [
        {
            'profile': 'Qwen 3.8 Flash-Next (125B MoE)',
            'quant': 'Q4_K_M (94 GB) or NVFP4',
            'context_window': 131072,
            'expected_speed': '45-60 tok/s decode, 1500+ tok/s prefill',
            'command': './configs/thor-128gb-setup.sh flash-next'
        },
        {
            'profile': 'Qwen 3.8 27B Unquantized FP16 / FP8',
            'quant': 'FP16 (55.6 GB) or FP8 (27.8 GB)',
            'context_window': 262144,
            'expected_speed': '50-85 tok/s',
            'command': './configs/thor-128gb-setup.sh 27b-fp16'
        }
    ]
elif '6000' in gpu_names or vram >= 44.0:
    tier = 'RTX_PRO_6000_WORKSTATION_48GB_96GB'
    launcher = 'configs/rtx-pro-6000-setup.sh'
    recommendations = [
        {
            'profile': 'Qwen 3.8 27B High-Precision Q8_0 (28.5 GB)',
            'context_window': 262144,
            'expected_speed': '55-75 tok/s',
            'command': './configs/rtx-pro-6000-setup.sh 27b-max'
        },
        {
            'profile': 'Qwen 2.5 72B Dense Q4_K_M (41.5 GB)',
            'context_window': 65536,
            'expected_speed': '35-48 tok/s',
            'command': './configs/rtx-pro-6000-setup.sh 72b-dense'
        },
        {
            'profile': 'Qwen 3.8 Flash-Next 125B MoE (For 96GB Card)',
            'context_window': 131072,
            'expected_speed': '50-65 tok/s',
            'command': './configs/rtx-pro-6000-setup.sh 125b-moe'
        }
    ]
elif '5090' in gpu_names or vram >= 30.0:
    tier = 'RTX_5090_32GB_GDDR7'
    launcher = 'configs/rtx-5090-setup.sh'
    recommendations = [
        {
            'profile': 'Qwen 3.8 27B Q6_K / Q8_0 High-Precision',
            'context_window': 131072,
            'expected_speed': '65-85 tok/s (base) / 130-160+ tok/s (speculative)',
            'command': './configs/rtx-5090-setup.sh'
        }
    ]
elif '4090' in gpu_names or '3090' in gpu_names or vram >= 23.0:
    tier = 'RTX_4090_3090_24GB_GDDR6X'
    launcher = 'configs/rtx-4090-setup.sh'
    recommendations = [
        {
            'profile': 'Qwen 3.8 27B Q4_K_M (17.5 GB) + Speculative MTP Draft Head (1.5 GB)',
            'context_window': 65536,
            'expected_speed': '46 tok/s (base) -> 72 tok/s (draft) -> 135 tok/s (code n-gram)',
            'command': './configs/rtx-4090-setup.sh'
        }
    ]
elif vram >= 15.0:
    tier = 'MID_RANGE_16GB_GPU'
    launcher = 'configs/universal-launcher.sh'
    recommendations = [
        {
            'profile': 'Qwen 3.8 27B IQ3_XXS (11.0 GB) with 4-bit KV Cache',
            'context_window': 131072,
            'expected_speed': '28-35 tok/s',
            'command': './configs/universal-launcher.sh /models/qwen3.8-27b/Qwen3.8-27B-IQ3_XXS.gguf'
        }
    ]
elif vram >= 10.0:
    if ram >= 48.0:
        tier = 'HYBRID_OFFLOAD_12GB_GPU_HIGH_RAM'
        launcher = 'configs/strata-12gb-setup.sh'
        recommendations = [
            {
                'profile': 'Qwen 3.8 Flash-Next (125B MoE) 3-bit via Strata',
                'context_window': 32768,
                'expected_speed': '50-62 tok/s decode, 1750 tok/s prefill',
                'command': './configs/strata-12gb-setup.sh'
            }
        ]
    else:
        tier = 'CONSTRAINED_12GB'
        launcher = 'configs/universal-launcher.sh'
        recommendations = [
            {
                'profile': 'Qwen 3.8 27B IQ3_XXS',
                'context_window': 16384,
                'expected_speed': '28-32 tok/s'
            }
        ]
else:
    tier = 'ENTRY_LEVEL_OR_CPU'
    recommendations = [
        {
            'profile': 'Qwen 2.5 Coder 7B or Qwen 3.8 14B Q4_K_M',
            'note': '27B will spill to host DDR4 RAM and drop to 3-5 tok/s.'
        }
    ]

out = {
    'hardware_tier': tier,
    'total_addressable_gb': vram,
    'host_ram_gb': ram,
    'is_unified_memory': is_unified,
    'gpu_count': gpu_count,
    'launcher_script': launcher,
    'recommendations': recommendations
}
print(json.dumps(out, indent=2))
"
