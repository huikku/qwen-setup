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

import json

tier = 'UNKNOWN'
recommendations = []

if vram >= 120.0 or (is_unified and ram >= 120.0):
    tier = 'TIER_1_ULTRA_ENTERPRISE_128GB_PLUS'
    recommendations = [
        {
            'workload': 'Maximum Quality Dense (Coding / Long Context)',
            'model': 'Qwen 3.8 27B',
            'precision': 'FP16 Unquantized (55.6 GB) or Blackwell FP8 (27.8 GB)',
            'context_window': 262144,
            'expected_throughput': '50-90 tok/s decode, 1800+ tok/s prefill',
            'vram_usage_gb': '56-65 GB',
            'execution_mode': 'Native GPU Unified / TensorRT-LLM / vLLM'
        },
        {
            'workload': 'Maximum Frontier MoE Scale',
            'model': 'Qwen 3.8 Flash-Next (125B MoE)',
            'precision': 'Q4_K_M (94 GB) or NVFP4',
            'context_window': 131072,
            'expected_throughput': '45-65 tok/s decode, 1500+ tok/s prefill',
            'vram_usage_gb': '94-110 GB',
            'execution_mode': 'Fits 100% in Unified Memory (Zero PCIe transfer penalties)'
        }
    ]
elif vram >= 44.0 or gpu_count >= 2:
    tier = 'TIER_2_MULTI_GPU_OR_WORKSTATION_48GB_96GB'
    recommendations = [
        {
            'workload': 'Unquantized or Q8 Dense Workloads',
            'model': 'Qwen 3.8 27B',
            'precision': 'FP16 or Q8_0',
            'context_window': 131072,
            'expected_throughput': '60-80 tok/s (Multi-GPU Tensor Parallelism / Pipeline)',
            'vram_usage_gb': '32-58 GB'
        },
        {
            'workload': 'MoE Frontier Model',
            'model': 'Qwen 3.8 Flash-Next (125B MoE)',
            'precision': 'Q3_K_M or Q4_K_M split across cards',
            'context_window': 65536,
            'expected_throughput': '50-70 tok/s'
        }
    ]
elif vram >= 23.0:
    tier = 'TIER_3_HIGH_END_CONSUMER_24GB_32GB'
    recommendations = [
        {
            'workload': 'High-Speed Coding Agent',
            'model': 'Qwen 3.8 27B',
            'precision': 'Q4_K_M (16.5-17.5 GB) + 1.5GB Speculative Draft Head',
            'context_window': 65536,
            'expected_throughput': '66-72 tok/s decode',
            'vram_usage_gb': '21.5 GB'
        }
    ]
elif vram >= 15.0:
    tier = 'TIER_4_MID_RANGE_16GB'
    recommendations = [
        {
            'workload': 'Long-Context Reasoning',
            'model': 'Qwen 3.8 27B',
            'precision': 'IQ3_XXS (11.0 GB) with 4-bit KV Cache',
            'context_window': 131072,
            'expected_throughput': '28-35 tok/s'
        }
    ]
elif vram >= 10.0:
    if ram >= 48.0:
        tier = 'TIER_5_HYBRID_OFFLOADING_12GB_GPU_HIGH_RAM'
        recommendations = [
            {
                'workload': 'Sparse MoE Architecture (Bypasses dense VRAM wall)',
                'model': 'Qwen 3.8 Flash-Next (125B MoE)',
                'precision': '3-bit via Strata Engine (Active 6B on GPU + 51B table in RAM)',
                'context_window': 32768,
                'expected_throughput': '50-62 tok/s decode, 1750 tok/s prefill'
            }
        ]
    else:
        tier = 'TIER_6_CONSTRAINED_12GB'
        recommendations = [
            {
                'model': 'Qwen 3.8 27B',
                'precision': 'IQ3_XXS (11.0 GB) with 8-bit KV Cache',
                'context_window': 16384,
                'expected_throughput': '28-32 tok/s'
            }
        ]
else:
    tier = 'TIER_7_LOW_VRAM_OR_CPU'
    recommendations = [
        {
            'model': 'Qwen 2.5 Coder 7B or Qwen 3.8 14B',
            'precision': 'Q4_K_M',
            'note': 'Dense 27B will spill layers into system memory, reducing throughput to 3-6 tok/s.'
        }
    ]

out = {
    'hardware_tier': tier,
    'total_addressable_gb': vram,
    'host_ram_gb': ram,
    'is_unified_memory': is_unified,
    'gpu_count': gpu_count,
    'recommendations': recommendations
}
print(json.dumps(out, indent=2))
"
