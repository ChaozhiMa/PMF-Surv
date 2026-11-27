#!/usr/bin/env bash
set -euo pipefail

project=/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig
output="$project/comparison_baselines_20260901/lite_prosenet"
mkdir -p "$output"

cd "$project/code"
export PYTHONPATH="$project/code/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=3,4,5
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

nohup python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=3 \
  scripts/run_lite_prosenet_baseline.py \
  --output-dir "$output" \
  --epochs 150 \
  --patience 30 \
  --batch-size-per-gpu 2 \
  --eval-batch-size 2 \
  > "$output/train.log" 2>&1 &

echo $! > "$output/launcher.pid"
echo "Lite-ProSENet launched with CUDA_VISIBLE_DEVICES=3,4,5; pid=$(cat "$output/launcher.pid")"
