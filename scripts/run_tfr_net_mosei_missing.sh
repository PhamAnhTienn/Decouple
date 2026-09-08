#!/bin/bash
# Sequential Benchmark Runner for TFRNet on CMU-MOSEI across 5 seeds and missing rates 0.0 to 0.9.
# Usage: ./scripts/run_tfr_net_mosei_missing.sh [gpu_id] [data_dir]

set -e

GPU_ID=${1:-0}
DATA_DIR=${2:-./data}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_tfr_net_missing.sh" mosei "${GPU_ID}" "${DATA_DIR}"
