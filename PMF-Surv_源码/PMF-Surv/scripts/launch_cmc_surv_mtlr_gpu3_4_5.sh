#!/usr/bin/env bash
set -euo pipefail

project=/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig
output="$project/comparison_baselines_20260901/cmc_surv_mtlr"
mkdir -p "$output"
cd "$project/code"

export CUDA_VISIBLE_DEVICES=3,4,5
export PYTHONPATH="$project/code/src"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

exec /home/image020/anaconda3/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=3 \
    scripts/run_cmc_surv_mtlr_baseline.py \
    --output-dir "$output" \
    "$@"
