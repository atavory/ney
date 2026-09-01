# Real-Benchmark Public-Covariate Expansion

This directory records the 2026-08-31 DML real-benchmark expansion for the
unified global-residual repair.  It adds diabetes, IHDP, and ACIC 2016
covariate structures to the Section 4 evidence ledger.  The run is companion
breadth evidence; it does not replace the Aug. 14 primary Section 4 source
unless the manuscript table policy is explicitly updated.

For paper-facing benchmark organization, IHDP and ACIC 2016 are the benchmark
datasets in this directory.  Diabetes is a supplementary public-covariate
check, not part of the customary five-dataset benchmark battery.  The
`*_semisynth` and `*_misaligned` source-design names record hidden
known-truth construction settings and should not be presented as observable
dataset categories.

## Scope

The run evaluates the same Section 4 repair settings on three covariate
sources, each with two internal response-surface settings and two strengths.

```text
datasets: diabetes, IHDP, ACIC 2016
source settings: *_semisynth, *_misaligned
strengths: 0, 3
experts: AIPW, fixed-floor TMLE, C-TMLE, selective ML, Ma DR-BC
jobs: 1,440
replication rows: 5,760
expert-setting combinations: 60
```

In the `*_semisynth` source settings, the added response-surface defect is
placed in the low-response region.  In the `*_misaligned` source settings, the
defect is placed in a disjoint signal region.  These labels are used for
known-truth audit only.

Every accepted row uses `repair_mode=if_residual`, `validation_risk=balanced_mse`,
`validation_loss_se=1.0`, `shrink_c=2.0`, `bootstraps=0`,
`region_selector_ablation=legacy`, and the damping grid
`0.0|0.25|0.5|1.0`.  The archived algorithm source SHA-256 is
`98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce`; the
public generator commit is `f6b7d74`.

## Generated Outputs

The run was launched from the public repository with:

```bash
RUN_DIR=/tmp/dml_real_benchmark_expand_20260831/dml_real_benchmark_v1 \
SOURCE_COMMIT=f6b7d74 \
FROZEN_SOURCE=/tmp/dml_unified_source_check.ycpy6z/replication/hidden_bias_regional_repair/scripts/validated_reference_transfer.py \
DML_SUPPORT_DATA=/tmp/dml_real_benchmark_support_data \
PYTHON_BIN=/tmp/dml_tmle_floor_grid_venv/bin/python \
./scripts/run_real_benchmark_expansion_20260831.sh
```

The compact summaries were generated with:

```bash
python replication/hidden_bias_regional_repair/scripts/materialize_section4_release_bundle.py \
  --source-dir /tmp/dml_real_benchmark_expand_20260831/dml_real_benchmark_v1 \
  --out-dir /tmp/dml_real_benchmark_expand_20260831/materialized_v1 \
  --expected-jobs 1440 \
  --reps-per-job 4 \
  --expected-c 2 \
  --nboot 20000 \
  --seed 20260831
```

The generated files are:

- `cell_summary.csv`: setting-level gains and harm rates.
- `family_summary.csv`: equal-setting summaries by source setting.
- `raw_rows.csv`: compact per-replication rows behind the summaries.
- `release_manifest.csv`: portable manifest with one row per shard.
- `verification.json`: materializer completeness report.
- `manifest.tsv`, `provenance.json`, and `status.json`: original launcher
  provenance from the accepted run.
- `support_inputs.json`: hashes and remote recovery details for IHDP and ACIC
  input data.

## Readout

Equal-setting MSE gains by source setting are:

| design | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| diabetes semisynth | +3.97% [+0.95, +7.61] | +0.91% [-0.20, +2.10] | +8.27% [+4.43, +12.11] | +0.92% [-2.15, +4.23] | +0.00% [+0.00, +0.00] |
| diabetes misaligned | +3.90% [+1.21, +7.14] | +2.82% [+0.03, +6.23] | +17.14% [+9.41, +25.49] | +1.69% [+0.03, +4.17] | +0.00% [+0.00, +0.00] |
| IHDP semisynth | +1.27% [-1.05, +3.47] | +0.15% [+0.02, +0.33] | +1.75% [-0.13, +4.24] | +1.09% [-1.23, +3.11] | +0.00% [+0.00, +0.00] |
| IHDP misaligned | +1.84% [+0.15, +4.49] | +0.04% [-0.24, +0.31] | +1.63% [-1.48, +4.95] | +0.84% [+0.00, +2.50] | +0.00% [+0.00, +0.00] |
| ACIC 2016 semisynth | -0.23% [-1.08, +0.43] | -0.20% [-0.73, +0.17] | +0.26% [-0.07, +0.74] | +0.40% [-0.43, +1.46] | +0.00% [+0.00, +0.00] |
| ACIC 2016 misaligned | -0.05% [-0.13, +0.00] | -0.14% [-0.51, +0.16] | +0.43% [-0.20, +1.20] | -0.26% [-0.93, +0.00] | +0.00% [+0.00, +0.00] |

Auxiliary dataset-level averages over the four settings in each dataset are:

| dataset | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| diabetes | +3.93% | +1.86% | +12.70% | +1.30% | +0.00% |
| IHDP | +1.55% | +0.09% | +1.69% | +0.96% | +0.00% |
| ACIC 2016 | -0.14% | -0.17% | +0.35% | +0.07% | +0.00% |

For the manuscript's primary benchmark table, report the four IHDP settings
and four ACIC 2016 settings separately.  Diabetes is supplementary.  C-TMLE
stands down exactly in this run.

## Input Data And Remote Recovery

The full shard/log archive is stored in Manifold:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_v1.tar.zst
```

Archive SHA-256:

```text
f82919fd706df611ed7ba846fb403c75fb8ad7d09897e70094e0ebb50fcb388e
```

The support-data archive is:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v1.tar.zst
```

Support-data archive SHA-256:

```text
e040a1721531757f8c1582be69555874fc81519eb1bf23f150c477a3a6810757
```

Input hashes:

```text
IHDP train NPZ: 750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240
IHDP test NPZ:  a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c
ACIC input_2016.RData: 068453aac851f1f4620bc6c6dff9877cd7cdc9c785d22668a762a9b6ce7e365b
ACIC parameters_2016.RData: 0ab47990af7b3c4872f36323812f9e1be2eae06842d4e5c9ff28f011df9831f0
ACIC converted CSV: 6a12ed5e139b88e40001e603d8e57be83f4e61dd1c8d1f76e28bc995aa1998d0
```

The ACIC CSV was converted from the official `input_2016.RData` using
`scripts/convert_acic2016_rdata_to_csv.py` with the pure-Python `rdata`
checkout at commit `0edddf4`.
