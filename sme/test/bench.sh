#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ../env.sh 2>/dev/null || true
THREADS="${1:-38}"
numactl --cpunodebind=0 --membind=0 ./direct.out bench "$THREADS"
