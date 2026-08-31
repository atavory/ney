#!/bin/sh
set -eu

python_bin="${PYTHON_BIN:-/tmp/dml_tmle_floor_grid_venv/bin/python}"
run_dir="${RUN_DIR:-/tmp/dml_public_covariate_expand_20260831/dml2_real_diabetes_wine_v1}"
source_commit="${SOURCE_COMMIT:-3bec073}"

cd "$(dirname "$0")/.."

"$python_bin" scripts/launch_section4_breadth_shards.py \
  --run-dir "$run_dir" \
  --frozen-source scripts/validated_reference_transfer.py \
  --wrapper scripts/section4_breadth_experiments.py \
  --python "$python_bin" \
  --owner dml2 \
  --groups real \
  --design-filter \
    real_diabetes_misaligned \
    real_wine_misaligned \
    real_diabetes_aligned \
    real_wine_aligned \
  --methods aipw ctmle cui_selective_ml ma_dr_bc tmle \
  --chunks 24 \
  --reps-per-chunk 4 \
  --bootstraps 0 \
  --max-workers 32 \
  --seed-base 1800000000 \
  --shrink-c 2 \
  --repair-mode if_residual \
  --region-selector-ablation legacy \
  --region-detector-c 4 \
  --region-damp-grid 0 0.25 0.5 1 \
  --validation-risk balanced_mse \
  --validation-loss-se 1 \
  --source-commit "$source_commit"
