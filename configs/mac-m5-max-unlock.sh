#!/usr/bin/env bash
# Adjust macOS wired memory limit on 128GB Mac Studio M5 Max
# macOS defaults to capping GPU wired memory at ~75% (96GB).
# This command raises the ceiling to 112GB so 4-bit Flash-Next fits without kernel panics.

echo "Current wired memory limit:"
sysctl iogpu.wired_mem_limit_mb || true

echo "Applying 112GB wired allocation ceiling..."
sudo sysctl iogpu.wired_mem_limit_mb=114688

echo "New wired memory limit:"
sysctl iogpu.wired_mem_limit_mb
