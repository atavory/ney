# ACIC 2017 Benchmark Expansion

This directory records the 2026-08-31 DML ACIC 2017 expansion for the unified
global-residual repair.  It is companion breadth evidence and does not replace
the Aug. 14 primary Section 4 source unless the manuscript table policy changes.
For paper-facing benchmark organization, ACIC 2017 is one dataset source.  The
`acic2017_semisynth` and `acic2017_misaligned` source-design names record
hidden known-truth construction settings and should not be presented as
observable dataset categories.

## Scope

```text
dataset: ACIC 2017
source settings: acic2017_semisynth, acic2017_misaligned
strengths: 0, 3
experts: AIPW, fixed-floor TMLE, C-TMLE, selective ML, Ma DR-BC
jobs: 480
replication rows: 1,920
expert-setting combinations: 20
```

In the `acic2017_semisynth` design, the added response-surface defect is placed
in the low-response region.  In the `acic2017_misaligned` design, the defect is
placed in a disjoint signal region.  These labels are used for known-truth
audit only.  Every accepted row uses
`repair_mode=if_residual`, `validation_risk=balanced_mse`,
`validation_loss_se=1.0`, `shrink_c=2.0`, `bootstraps=0`,
`region_selector_ablation=legacy`, and damping grid `0.0|0.25|0.5|1.0`.

## Generated Outputs

The public generator commit is `88fee2e`; the archived algorithm source SHA-256
is `98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce`.
The launch command and wrapper hashes are recorded in `provenance.json`.

The generated files are:

- `cell_summary.csv`: setting-level gains and harm rates.
- `family_summary.csv`: equal-setting summaries by source setting.
- `raw_rows.csv`: compact per-replication rows behind the summaries.
- `release_manifest.csv`: portable manifest with one row per shard.
- `verification.json`: materializer completeness report.
- `manifest.tsv`, `provenance.json`, and `status.json`: launcher provenance.
- `support_inputs.json`: hashes and remote recovery details for ACIC 2017 input data.

## Readout

Auxiliary dataset-level average MSE gains over the four ACIC 2017 settings are:

| dataset | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| ACIC 2017 | +0.96% [+0.01, +2.22] | -0.24% [-1.15, +0.41] | +0.51% [-0.01, +1.17] | +0.73% [-0.18, +2.44] | +0.00% [+0.00, +0.00] |

Source-setting readout for audit:

| design | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| ACIC 2017 semisynth | +1.858% [+0.067, +4.338] | +0.341% [-0.280, +0.971] | +0.471% [-0.317, +1.590] | +1.636% [+0.000, +5.050] | +0.000% [+0.000, +0.000] |
| ACIC 2017 misaligned | +0.056% [-0.531, +0.658] | -0.828% [-2.503, +0.286] | +0.541% [-0.066, +1.341] | -0.182% [-0.626, +0.000] | +0.000% [+0.000, +0.000] |

For the manuscript's primary benchmark table, report the four ACIC 2017
settings separately.  ACIC 2017 is small-effect breadth evidence.  The
source-setting split is a known-truth diagnostic, not an observable dataset
classification.

## Remote Recovery

The full shard/log archive is stored in Manifold:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_acic2017_v1.tar.zst
```

Archive SHA-256:

```text
5f874dfa02e808bb160a56c53431dfa7329f5a14cc7d9fcb1b9c994a6d26bdcb
```

The support-data archive is:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v2.tar.zst
```

Support-data archive SHA-256:

```text
de5a55054d6178840c6aca3b4440fa2e835502b6c5fec32e490095f355dfc494
```
