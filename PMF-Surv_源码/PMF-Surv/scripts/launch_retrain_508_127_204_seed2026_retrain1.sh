#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_CONFIG="$PROJECT_ROOT/../restot_surv_hard_top1/config/hard_top1.yaml"
CONFIG_PATH="$PROJECT_ROOT/config/hard_top1.yaml"
if [[ -f "$SERVER_CONFIG" ]]; then
  CONFIG_PATH="$SERVER_CONFIG"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
fi
cd "$PROJECT_ROOT"
exec "${PMF_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node "${PMF_PROCESSES:-3}" \
  scripts/train.py --config "${PMF_CONFIG:-$CONFIG_PATH}" "$@"
