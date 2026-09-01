# High-Response Placebo Ablation

This bundle summarizes the 24-setting regional-placement placebo ablation used
as a Section 4 diagnostic. The diagnostic uses the same benchmark settings as
the primary table: eight Kang--Schafer settings and four settings each for
IHDP, ACIC 2016, ACIC 2017, and Twins. It reports the four primary expert
families: AIPW, selective ML, Ma DR-BC, and C-TMLE.

The diagnostic is support-restricted. It is not the primary global
`if_residual` repair used in the main results table. Each run uses
`repair_mode=regional_if_residual` and compares two placements of the same
residual correction:

- `low_response`: restrict the correction to the selected low-response support.
- `high_response_placebo`: restrict the correction to a matched high-response
  support.

Both placements use the same damping grid `0,0.25,0.5,1`,
one-standard-error `balanced_mse` selection rule, final `c=2` plug-in
contrast shrinkage, three-fold cross-fitting, and XGBoost nuisance learners.

## Headline

Equal-setting percent MSE gains over the 24 benchmark settings are:

| expert family | low-response repair | high-response placebo | low minus high |
|---|---:|---:|---:|
| AIPW | 1.945 [1.262, 2.602] | -0.036 [-0.285, 0.121] | 1.981 [1.263, 2.664] |
| selective ML | 0.372 [-0.076, 0.790] | 0.195 [0.013, 0.408] | 0.178 [-0.294, 0.596] |
| Ma DR-BC | 1.052 [0.373, 1.687] | 0.959 [0.596, 1.306] | 0.093 [-0.653, 0.825] |
| C-TMLE | 0.124 [-0.046, 0.357] | 0.009 [-0.029, 0.053] | 0.115 [-0.064, 0.351] |

The placement separation is clear for AIPW. For selective ML and Ma DR-BC, the
high-response placebo remains active enough that the low-minus-high contrast is
not separated from zero. C-TMLE mostly stands down in both arms.

## Raw Archive

The raw shard outputs, logs, provenance, and status files are archived at:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/high_response_placebo_ablation_20260831/dml_high_response_placebo_ablation_20260831_raw.tar.zst
```

Archive SHA-256:

```text
6b3b12b38797375846b99b7042de80e99402f798902e241f9a2c9f76f7a96e7b
```

The raw archive contains four shard runs:

| run | settings | shards | rows | failures |
|---|---:|---:|---:|---:|
| Kang--Schafer low response | 8 | 768 | 3,072 | 0 |
| Kang--Schafer high response | 8 | 768 | 3,072 | 0 |
| IHDP/ACIC/Twins low response | 16 | 1,536 | 6,144 | 0 |
| IHDP/ACIC/Twins high response | 16 | 1,536 | 6,144 | 0 |

Total raw coverage is 4,608 shard files and 18,432 replication rows.

## Reproduce

The shard launcher wraps the public Section 4 breadth launcher and changes only
the `--analysis-region` argument. The four raw runs were launched with:

```bash
python scripts/dml_launch_section4_placebo_shards.py \
  --launcher-source /tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/scripts/launch_section4_breadth_shards.py \
  --analysis-region estimated_residual_lowp_supported \
  --run-dir /tmp/dml_high_response_placebo_20260831/ks_low_regional \
  ...

python scripts/dml_launch_section4_placebo_shards.py \
  --launcher-source /tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/scripts/launch_section4_breadth_shards.py \
  --analysis-region estimated_residual_highp_matched_lowp_supported \
  --run-dir /tmp/dml_high_response_placebo_20260831/ks_high_regional \
  ...

python scripts/dml_launch_section4_placebo_shards.py \
  --launcher-source /tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/scripts/launch_section4_breadth_shards.py \
  --analysis-region estimated_residual_lowp_supported \
  --run-dir /tmp/dml_high_response_placebo_20260831/bench_low_regional \
  ...

python scripts/dml_launch_section4_placebo_shards.py \
  --launcher-source /tmp/ney_global_recreate_20260830/replication/hidden_bias_regional_repair/scripts/launch_section4_breadth_shards.py \
  --analysis-region estimated_residual_highp_matched_lowp_supported \
  --run-dir /tmp/dml_high_response_placebo_20260831/bench_high_regional \
  ...
```

Regenerate the compact summary and manuscript table from the raw directories:

```bash
python scripts/summarize_high_response_placebo_ablation.py \
  --low-run-dir /tmp/dml_high_response_placebo_20260831/ks_low_regional \
  --low-run-dir /tmp/dml_high_response_placebo_20260831/bench_low_regional \
  --high-run-dir /tmp/dml_high_response_placebo_20260831/ks_high_regional \
  --high-run-dir /tmp/dml_high_response_placebo_20260831/bench_high_regional \
  --out-dir support_csv/dml_high_response_placebo_ablation_20260831 \
  --draws 20000 \
  --seed 20260831
```

## Files

- `high_response_placebo_cell_summary.csv`: setting-level low/high summaries.
- `high_response_placebo_summary.csv`: 24-setting expert-family summaries.
- `section4_high_response_placebo_ablation_table.tex`: generated manuscript
  table.
- `verification.json`: row counts, source directories, output hashes, and
  verification status.
- `provenance_*.json` and `status_*.json`: copied status and provenance from
  the four raw shard runs.
