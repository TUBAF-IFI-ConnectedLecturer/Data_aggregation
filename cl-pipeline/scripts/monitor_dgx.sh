#!/bin/bash
# Live GPU monitoring for DGX
# Usage: ./monitor_dgx.sh [refresh_interval_seconds]
# Default: 2 seconds

INTERVAL=${1:-2}

watch -n $INTERVAL '
echo "=== DGX GPU Monitor ==="
echo ""
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
awk -F", " "{printf \"  GPU %2d | %-20s | GPU: %3d%% | MEM: %3d%% | %6d / %6d MiB | %3d°C\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7}"
echo ""
echo "=== Ollama Processes ==="
echo ""
docker exec ${OLLAMA_CONTAINER:-ollama} ollama ps 2>/dev/null || echo "  Ollama container not running"
'
