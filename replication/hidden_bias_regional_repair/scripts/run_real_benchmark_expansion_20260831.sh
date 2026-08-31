#!/bin/sh
set -eu

python_bin="${PYTHON_BIN:-/tmp/dml_tmle_floor_grid_venv/bin/python}"
run_dir="${RUN_DIR:-/tmp/dml_real_benchmark_expand_20260831/dml_real_benchmark_v1}"
source_commit="${SOURCE_COMMIT:-$(git rev-parse --short HEAD)}"
source_file="${FROZEN_SOURCE:-scripts/validated_reference_transfer.py}"
support_data="${DML_SUPPORT_DATA:-/tmp/dml_real_benchmark_support_data}"

cd "$(dirname "$0")/.."
export USHMOO_SUPPORT_DATA="$support_data"

"$python_bin" scripts/launch_section4_breadth_shards.py \
  --run-dir "$run_dir" \
  --frozen-source "$source_file" \
  --wrapper scripts/section4_breadth_experiments.py \
  --python "$python_bin" \
  --owner dml \
  --groups real_benchmark \
  --methods aipw ctmle cui_selective_ml ma_dr_bc tmle \
  --chunks 24 \
  --reps-per-chunk 4 \
  --bootstraps 0 \
  --max-workers 32 \
  --seed-base 1900000000 \
  --shrink-c 2 \
  --repair-mode if_residual \
  --region-selector-ablation legacy \
  --region-detector-c 4 \
  --region-damp-grid 0 0.25 0.5 1 \
  --validation-risk balanced_mse \
  --validation-loss-se 1 \
  --source-commit "$source_commit"
